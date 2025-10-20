"""Parallel trainer for Mario Kart DS agents using DeSmuME, multiprocessing,
shared memory frame streaming, and optional live GTK visualization.

This module orchestrates end-to-end evaluation and evolution of a population of
neural network controllers (NEAT-style) for Mario Kart DS. It supports three
execution modes per evaluated individual:

  1) **Headless** (`run_process`) — fast evaluation with no display.
  2) **Display worker** (`run_window_process`) — renders frames and writes them
     into a per-process shared-memory buffer; no GTK loop.
  3) **Display host** (`run_window_host_process`) — renders frames, writes them
     into shared memory, and owns the GTK window that tiles and presents all
     *display-enabled* workers in real time.

Key concepts
------------
- **Shared memory frames:** Each display-enabled process writes an RGBX
  framebuffer of shape ``(SCREEN_HEIGHT, SCREEN_WIDTH, 4)`` (dtype
  ``np.uint8``) to a POSIX shared-memory segment named ``f"emu_frame_{id}"``.
  The host window process opens these buffers read-only for display tiling.
- **Overlays:** Optional per-frame overlays are computed off the main emulation
  loop by a single background thread fed via a queue. Overlays are composited in
  the worker before writing to shared memory using
  :func:`src.visualization.window.on_draw_memoryview`.
- **Statistics / fitness:** Each process records split times and distances at
  track checkpoints as a ``dict[int, list[tuple[float, float]]]`` mapping
  ``checkpoint_id -> [(delta_time, distance_at_split), ...]``. A simple fitness
  function sums the recorded distances.
- **Batching & evolution:** :func:`run_training_session` evaluates a subset
  (batch) of the population in parallel (bounded by ``num_proc``), aggregates
  stats, then :func:`train` evolves the population.

Threading & processes
---------------------
- The DeSmuME emulator is created and used **inside each process** that runs it.
- The GTK main loop **must** run in a single process. This module designates one
  display-enabled process per batch as the *host* that creates the window and
  drives GTK via `GLib.timeout_add`.
- Overlays are computed by a **single background thread** (daemon) within each
  display-enabled process to keep the emulation loop responsive.

Shared-memory lifetime
----------------------
- Creation: Call :func:`safe_shared_memory` to create (or replace) a named
  shared-memory segment.
- Ownership: Workers **open** their frame buffer (``emu_frame_{id}``) by name
  and keep a persistent ``SharedMemory`` handle as long as they render frames.
- Teardown: After processes finish, the parent should **close and unlink**
  per-process frame segments to avoid resource-tracker warnings.

Examples
--------
Run 10 generations with a population of 32 where only one sample is displayed
each batch and overlays with IDs 0, 3, and 4 are enabled:

    >>> if __name__ == "__main__":
    ...     train(
    ...         num_iters=10,
    ...         pop_size=32,
    ...         show_samples=[False],   # broadcast later per batch
    ...         overlay_ids=[0, 3, 4],
    ...     )

Notes
-----
- This module expects the `mariokart_ds.nds` ROM to be available in the working
  directory and a valid savestate at index 3.
- ``on_draw_memoryview`` expects an emulator-provided RGBX memory buffer
  (4 bytes/pixel), and returns a premultiplied ARGB32 array suitable for Cairo.
- ``MODEL_KEY_MAP`` defines a simple thresholded policy:
  values >= 0.5 are pressed, and the accelerator button is always pressed when
  any action is taken.

"""

# Builtin dependencies
from __future__ import annotations
import random, math, os, sys, copy
from multiprocessing.managers import DictProxy
from multiprocessing import Process, Manager, Lock
from multiprocessing.shared_memory import SharedMemory
from queue import Queue
from threading import Thread

# External dependencies
from desmume.emulator import DeSmuME, SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_PIXEL_SIZE
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from desmume.controls import Keys, keymask
from torch._prims_common import DeviceLikeType
import numpy as np
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

# Local dependencies
from src.core.memory import *
from src.core.memory import read_clock
from src.core.model import Genome, EvolvedNet
from src.utils.vector import get_mps_device
from src.visualization.window import SharedEmulatorWindow, on_draw_memoryview
from src.visualization.overlay import AVAILABLE_OVERLAYS

MODEL_KEY_MAP = {
    3: Keys.KEY_UP,
    2: Keys.KEY_DOWN,
    1: Keys.KEY_LEFT,
    0: Keys.KEY_RIGHT,
    4: Keys.KEY_B,
    5: Keys.KEY_A,
    6: Keys.KEY_X,
    7: Keys.KEY_Y,
    8: Keys.KEY_L,
    9: Keys.KEY_R,
    10: Keys.KEY_START,
    11: Keys.KEY_LEFT,
    12: Keys.KEY_RIGHT,
    13: Keys.KEY_UP,
    14: Keys.KEY_DOWN,
}


def safe_shared_memory(name: str, size: int):
    """Create or replace a named POSIX shared-memory segment.

    This helper guarantees that a shared-memory block with the given `name`
    exists with the requested `size`. If a stale block exists (e.g., from an
    earlier crashed run), it is closed and unlinked before creating a fresh one.

    Args:
      name: Symbolic name of the shared memory region
        (e.g., ``"emu_frame_0"``).
      size: Size in **bytes** to allocate for the region.

    Returns:
      multiprocessing.shared_memory.SharedMemory: An opened handle to the new
      shared-memory block. The caller owns the handle and is responsible for
      closing it (and unlinking at teardown time).

    Raises:
      ValueError: If `size <= 0`.
      OSError: If the OS cannot allocate or map the segment.

    Side Effects:
      - May unlink an existing segment of the same name.
      - Creates a new segment in the system shared-memory namespace.

    """
    from multiprocessing import shared_memory

    if size <= 0:
        raise ValueError("safe_shared_memory: size must be > 0")

    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        old = shared_memory.SharedMemory(name=name)
        old.close()
        old.unlink()
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    return shm


def initialize_emulator() -> DeSmuME:
    """Initialize and prime a DeSmuME emulator instance.

    Loads the MKDS ROM, restores a savestate (slot 3), mutes audio, and cycles
    once to ensure memory is initialized. Then spins until the emulator reports
    it is running.

    Returns:
      DeSmuME: A ready-to-use emulator instance positioned at the savestate.

    Notes:
      - This function blocks until ``emu.is_running()`` returns True.
      - The ROM path ``"mariokart_ds.nds"`` and savestate index are hard-coded.

    """
    emu = DeSmuME()
    emu.open("mariokart_ds.nds")
    emu.savestate.load(3)
    emu.volume_set(0)
    emu.cycle()

    while not emu.is_running():
        print("Waiting for emulator...")

    return emu


def initialize_window(emu, display_count, shm_names) -> SharedEmulatorWindow:
    """Create and initialize the tiled GTK window for live visualization.

    Computes a near-square grid (``n_rows`` × ``n_cols``) based on the number
    of display-enabled processes, instantiates a renderer bound to `emu`, and
    returns a :class:`SharedEmulatorWindow` configured to read from the provided
    shared-memory frame names.

    Args:
      emu: Active :class:`DeSmuME` instance (used to build the renderer).
      display_count: Number of display-enabled workers to tile.
      shm_names: List of shared-memory segment names (``"emu_frame_{id}"``).

    Returns:
      SharedEmulatorWindow: GTK window object ready to be shown.

    Side Effects:
      - Initializes a GTK/Cairo renderer via :class:`AbstractRenderer`.

    """
    width = 1000
    height = math.floor(width * (SCREEN_HEIGHT / SCREEN_WIDTH))
    n_cols = math.ceil(math.sqrt(display_count))
    n_rows = math.ceil(display_count / n_cols)
    renderer = AbstractRenderer.impl(emu)
    renderer.init()
    window = SharedEmulatorWindow(
        width=width,
        height=height,
        n_cols=n_cols,
        n_rows=n_rows,
        renderer=renderer,
        shm_names=shm_names,
    )
    return window


def initialize_overlays(
    overlay_ids: list[int], device: DeviceLikeType | None
) -> Queue | None:
    """Start a background overlay thread and return its work queue.

    Given a list of overlay IDs, looks them up in :data:`AVAILABLE_OVERLAYS`,
    starts a single daemon thread that consumes :class:`DeSmuME` instances from
    a queue and applies the overlays. The queue is returned to the caller to
    submit per-frame overlay requests.

    Args:
      overlay_ids: List of overlay identifiers to enable (indexes into
        :data:`AVAILABLE_OVERLAYS`).
      device: Torch device on which overlay computations (if any) should run.

    Returns:
      Queue | None: If `overlay_ids` is non-empty, a ``Queue`` into which the
      caller should ``put(emu)`` once per frame, and ``put(None)`` on shutdown.
      Returns ``None`` when `overlay_ids` is empty.

    Notes:
      - The overlay worker catches exceptions per overlay and propagates a
        summarized error message on failure via :func:`safe_thread`.
      - Overlays are executed off the emulation thread to avoid jitter.

    """
    overlays = []
    for id in overlay_ids:
        overlays.append(AVAILABLE_OVERLAYS[id])

    emu_queue = None
    if len(overlays) != 0:
        emu_queue = Queue()

        def worker():
            nonlocal overlays, emu_queue, id
            assert emu_queue is not None
            while True:
                emu_instance = emu_queue.get()
                if emu_instance is None:
                    break
                for overlay in overlays:
                    safe_overlay = safe_thread(overlay, proc_id=id)
                    safe_overlay(emu_instance, device=device)

                emu_queue.task_done()

        thread = Thread(target=worker, daemon=True)
        thread.start()

    return emu_queue


def handle_controls(emu: DeSmuME, logits: torch.Tensor):
    """Apply model outputs to emulator controls with a simple threshold policy.

    All values ``>= 0.5`` are considered pressed for the corresponding
    ``MODEL_KEY_MAP`` entry. Additionally, when any action is pressed, the
    accelerator (mapped to ``MODEL_KEY_MAP[5]``) is also pressed to keep the
    kart moving.

    Args:
      emu: Active emulator instance whose keypad state will be updated.
      logits: 1D tensor of action activations aligned with ``MODEL_KEY_MAP``.

    Side Effects:
      - Calls ``emu.input.keypad_update(0)`` and
        ``emu.input.keypad_add_key(...)`` multiple times.

    """
    logits_list = logits.tolist()
    emu.input.keypad_update(0)
    for i, v in enumerate(logits_list):
        if v < 0.5:
            continue

        emu.input.keypad_add_key(keymask(MODEL_KEY_MAP[i]))
        emu.input.keypad_add_key(keymask(MODEL_KEY_MAP[5]))


def run_window_host_process(
    id: int,
    sample: Genome,
    shm_names: list[str],
    overlay_ids: list[int],
    batch_size: int,
    lock,
    pop_stats: dict[int, dict[int, list[tuple[float, float]]]],
):
    """Display-host evaluation process: emulates, renders, and owns the GTK loop.

    This process:
      1) Initializes the emulator and the evolved model for `sample`.
      2) Attaches to its per-process shared-memory frame ``emu_frame_{id}``.
      3) Creates a tiled :class:`SharedEmulatorWindow` reading all `shm_names`.
      4) Starts a background overlay thread (if overlays are enabled).
      5) Schedules two GLib timers:
         - ``_forward`` (~30 FPS): cycle, composite overlay, write shared frame,
           run model step, and enqueue overlay work.
         - ``check_end`` (5 Hz): polls whether all display frames are zeroed and
           exits the GTK loop when they are (end-of-batch signal).
      6) On exit, shuts down the overlay thread and writes per-individual stats
         back to the manager dict.

    Args:
      id: Process index for this individual (used in shared-memory naming).
      sample: Genome to evaluate in this process.
      shm_names: Names of shared-memory frame buffers to tile in the host window.
      overlay_ids: Enabled overlay identifiers.
      batch_size: Number of individuals in the current batch.
      lock: IPC lock used when writing results to `pop_stats`.
      pop_stats: Manager-backed dict to receive results for this process.

    Side Effects:
      - Opens GTK window and runs a blocking GTK main loop.
      - Writes ARGB32 frames into ``emu_frame_{id}`` shared memory.
      - Updates `pop_stats[id]` with checkpoint timing/distances upon completion.

    """
    # Set torch device
    device = get_mps_device()

    # Initialize emulator
    emu = initialize_emulator()

    # Initialize model
    model = EvolvedNet(sample).to(device)
    forward = get_forward_func(emu, model, device)

    # Attach shared window frame
    shm_frame = SharedMemory(name=f"emu_frame_{id}")
    frame = np.ndarray(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=shm_frame.buf
    )

    # Initialize window
    display_count = len(shm_names)
    window = initialize_window(emu, display_count, shm_names)

    # Initialize overlay thread
    emu_queue = initialize_overlays(overlay_ids, device)

    # Will incrementally check if the population has died
    def check_end():
        """Quit GTK when all visible frames are cleared to zeros."""
        for name in shm_names:
            shm = SharedMemory(name=name)
            arr = np.ndarray(
                (SCREEN_WIDTH, SCREEN_HEIGHT, 4), dtype=np.uint8, buffer=shm.buf
            )
            if arr.sum() != 0:
                return True

        Gtk.main_quit()
        return False

    # Cycle like usual
    stats = {}

    def _forward():
        """Per-frame emulation, overlay composition, and model step (≈30 FPS)."""
        nonlocal stats
        emu.cycle()

        # Copy display data to shared memory buffer
        buf = emu.display_buffer_as_rgbx()[: SCREEN_PIXEL_SIZE * 4]
        new_frame = on_draw_memoryview(buf, SCREEN_WIDTH, SCREEN_HEIGHT, 1.0, 1.0)
        np.copyto(frame, new_frame)

        # Inference / Game Update
        logits = forward()
        if isinstance(logits, dict):
            send_window_end_signal(id)
            stats = logits
            return False

        # Queue Overlay Request
        if emu_queue is not None:
            emu_queue.put(emu)

        tmp_logits = logits
        return True

    GLib.timeout_add(200, check_end)  # non-blocking
    GLib.timeout_add(33, _forward)  # non-blocking
    window.show_all()
    Gtk.main()  # blocking

    # Safe thread shutdown for overlay
    if emu_queue is not None:
        emu_queue.put(None)

    # Log results
    with lock:
        pop_stats[id] = stats


def safe_thread(func, proc_id, thread_id=0):
    """Wrap a function for background execution with nicer error reporting.

    The returned wrapper calls `func(*args, **kwargs)` and converts any
    exception into a concise message identifying the logical process and thread
    of origin.

    Args:
      func: Callable to wrap.
      proc_id: Integer process identifier for error messages.
      thread_id: Integer thread identifier for error messages.

    Returns:
      Callable: A new callable with identical signature that raises a concise
      :class:`Exception` on failure.

    """
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            raise Exception(f"Error on thread {thread_id} of process {proc_id}")

    return wrapper


def run_window_process(
    id: int,
    sample: Genome,
    shm_names: list[str],
    overlay_ids: list[int],
    batch_size: int,
    lock,
    pop_stats: dict[int, dict[int, list[tuple[float, float]]]],
):
    """Display worker process: emulates, renders, **no** GTK ownership.

    This process is identical to the host process except it does **not** create
    nor run a GTK window. It writes its per-frame ARGB32 overlay-composited
    output to ``emu_frame_{id}`` and exits when the evaluation finishes.

    Args:
      id: Process index for this individual (used in shared-memory naming).
      sample: Genome to evaluate in this process.
      shm_names: Names of all display frame segments (unused directly here).
      overlay_ids: Enabled overlay identifiers.
      batch_size: Number of individuals in the current batch.
      lock: IPC lock used when writing results to `pop_stats`.
      pop_stats: Manager-backed dict to receive results for this process.

    Side Effects:
      - Writes ARGB32 frames into ``emu_frame_{id}`` shared memory.
      - Updates `pop_stats[id]` with checkpoint timing/distances upon completion.

    """
    # Set torch device
    device = get_mps_device()

    # Initialize emulator
    emu = initialize_emulator()

    # Initialize model
    model = EvolvedNet(sample).to(device)
    forward = get_forward_func(emu, model, device)

    # Attach shared window frame
    shm_frame = SharedMemory(name=f"emu_frame_{id}")
    frame = np.ndarray(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=shm_frame.buf
    )

    # Initialize overlay thread
    emu_queue = initialize_overlays(overlay_ids, device)

    # Cycle like usual
    stats = {}
    while True:
        emu.cycle()

        # Copy display data to shared memory buffer
        buf = emu.display_buffer_as_rgbx()[: SCREEN_PIXEL_SIZE * 4]
        new_frame = on_draw_memoryview(buf, SCREEN_WIDTH, SCREEN_HEIGHT, 1.0, 1.0)
        np.copyto(frame, new_frame)

        # Inference / Game Update
        logits = forward()
        if isinstance(logits, dict):
            send_window_end_signal(id)
            stats = logits
            break

        # Queue Overlay Request
        if emu_queue is not None:
            emu_queue.put(emu)

        tmp_logits = logits

    # Safe thread shutdown for overlay
    if emu_queue is not None:
        emu_queue.put(None)

    # Log results
    with lock:
        pop_stats[id] = stats


def send_window_end_signal(id):
    """Zero a per-process frame buffer to signal the host window to exit.

    Args:
      id: Process index whose frame buffer should be cleared.

    Side Effects:
      - Writes zeros into the shared frame ``emu_frame_{id}``, which is used by
        the host window's polling logic to detect end-of-batch.

    """
    shm = SharedMemory(name=f"emu_frame_{id}")
    arr = np.ndarray((SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=shm.buf)
    arr[:] = 0


def run_process(
    id: int,
    sample: Genome,
    shm_names: list[str],
    overlay_ids: list[int],
    batch_size: int,
    lock,
    pop_stats: dict[int, dict[int, list[tuple[float, float]]]],
):
    """Headless evaluation process: emulates and updates controls, no display.

    Runs the emulator to completion for the given `sample` genome. No shared
    memory frame is produced in this mode; it focuses on throughput.

    Args:
      id: Process index for this individual.
      sample: Genome to evaluate.
      shm_names: Names of all display frame segments (unused in headless mode).
      overlay_ids: Enabled overlay identifiers (unused in headless mode).
      batch_size: Number of individuals in the current batch.
      lock: IPC lock used when writing results to `pop_stats`.
      pop_stats: Manager-backed dict to receive results for this process.

    Side Effects:
      - Calls :func:`handle_controls` each step to apply model outputs.
      - Updates `pop_stats[id]` with final checkpoint stats.

    """
    # Set torch device
    device = get_mps_device()

    # Initialize emulator
    emu = initialize_emulator()

    # Initialize model
    model = EvolvedNet(sample).to(device)
    forward = get_forward_func(emu, model, device)

    # Cycle like usual
    stats = {}
    while True:
        emu.cycle()

        # Inference / Game Update
        logits = forward()
        if isinstance(logits, dict):
            stats = logits
            break

        # Controls
        handle_controls(emu, logits)

    # Log results
    with lock:
        pop_stats[id] = stats


def get_forward_func(emu: DeSmuME, model: EvolvedNet, device):
    """Build a closure that performs one model step and checkpoint bookkeeping.

    The returned callable reads emulator memory for sensor inputs, constructs the
    model input vector, computes the control logits, and records per-checkpoint
    split times and distances. When a terminal condition is reached (e.g.
    clock > 10000), it returns the accumulated stats dict instead of logits.

    Args:
      emu: Active emulator instance to read game state from.
      model: Evolved network to evaluate (expects 6 inputs → action logits).
      device: Torch device on which tensors are constructed and the model runs.

    Returns:
      Callable[[], torch.Tensor | dict[int, list[tuple[float, float]]]]:
        A no-argument function that returns either a 1D tensor of action logits
        or a stats dictionary signaling the end of this individual's run.

    Sensor model:
      - Distances: forward/left/right obstacle distances are read and mapped
        through ``tanh(1 - d / s1)`` with ``s1 = 60.0`` to compress range.
      - Angles: direction to the next checkpoint as (cos θ, sin θ, -sin θ).

    Notes:
      - Checkpoint bookkeeping appends tuples of ``(delta_time, distance_at_split)``.
      - This function reads directly from emulator memory via utility helpers.

    """
    current_time = read_clock(emu)
    prev_time = current_time
    current_id = read_current_checkpoint(emu)
    prev_id = current_id
    prev_dist = read_checkpoint_distance_altitude(emu, device=device).item()
    times: dict[int, list[tuple[float, float]]] = {}

    def forward() -> torch.Tensor | dict[int, list[tuple[float, float]]]:
        nonlocal current_id, prev_id, current_time, prev_time, prev_dist, model, emu

        clock = read_clock(emu)
        if clock > 10000:
            return times

        emu = emu
        prev_id = read_current_checkpoint(emu)
        clock = read_clock(emu)

        if current_id != prev_id:
            assert isinstance(current_id, int)
            if current_id not in times:
                times[current_id] = []

            current_time = clock
            times[current_id].append((current_time - prev_time, prev_dist))
            prev_time = current_time
            prev_dist = read_checkpoint_distance_altitude(emu, device=device).item()

        s1 = 60.0

        # Sensor inputs (obstacle distances)
        forward_d = read_forward_distance_obstacle(emu, device=device)
        left_d = read_left_distance_obstacle(emu, device=device)
        right_d = read_right_distance_obstacle(emu, device=device)
        inputs_dist1 = torch.tensor([forward_d, left_d, right_d], device=device)
        inputs_dist1 = torch.tanh(1 - inputs_dist1 / s1)

        # Angular relationship to next checkpoint
        angle = read_direction_to_checkpoint(emu, device=device)
        forward_a = torch.cos(angle)
        left_a = torch.sin(angle)
        right_a = -torch.sin(angle)
        inputs_dist2 = torch.tensor([forward_a, left_a, right_a], device=device)

        # Model inference
        inputs = torch.cat([inputs_dist1, inputs_dist2])
        logits = model(inputs)

        return logits

    return forward


def run_training_batch(
    batch_pop: list[Genome],
    show_samples: list[bool],
    overlay_ids: list[int],
    lock,
    pop_stats: DictProxy[int, dict[int, list[tuple[float, float]]]],
):
    """Evaluate a batch of genomes concurrently, optionally with live display.

    One process in the batch is promoted to the *display host* (first True in
    `show_samples`) and creates a tiled GTK window that reads from the
    per-process shared-memory frames listed in `shm_names`. Additional True
    entries run as display workers; False entries run headless.

    Args:
      batch_pop: Slice of the population to evaluate in this batch.
      show_samples: Per-individual flags controlling display mode; exactly one
        True is chosen as the host (the first True), additional Trues are
        workers; all False means fully headless batch.
      overlay_ids: Enabled overlay identifiers.
      lock: IPC lock for synchronized writes to `pop_stats`.
      pop_stats: Manager dict where each process writes a stats dict under its
        local batch index.

    Side Effects:
      - Creates per-process shared-memory frame segments for display-enabled
        individuals.
      - Spawns processes with appropriate targets and joins them.

    """
    global shm_names
    batch_size = len(batch_pop)

    processes = []
    shm_names = []

    for id in range(batch_size):
        if show_samples[id]:
            shm_names.append(f"emu_frame_{id}")

    host_proc_found = False
    for id, sample, show in zip(range(batch_size), batch_pop, show_samples):
        target = run_process  # defaults to headless emulator

        if show and not host_proc_found:
            target = run_window_host_process
            host_proc_found = True
        elif show and host_proc_found:
            target = run_window_process

        if show:
            # Attach/create shared memory buffer for emulator frame
            size = SCREEN_HEIGHT * SCREEN_WIDTH * 4
            shm_frame = safe_shared_memory(name=f"emu_frame_{id}", size=size)
            frame = np.ndarray(
                shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4),
                dtype=np.uint8,
                buffer=shm_frame.buf,
            )
            frame[:] = 1.0  # non-zero sentinel until first frame arrives

        process = Process(
            target=target,
            args=(id, sample, shm_names, overlay_ids, batch_size, lock, pop_stats),
            daemon=True,
        )
        processes.append(process)

    # Start processes
    for p in processes:
        p.start()

    # Join processes
    for p in processes:
        p.join()


def run_training_session(
    pop: list[Genome],
    num_proc: int | None = None,
    show_samples: list[bool] = [True],
    overlay_ids: list[int] = [],
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    """Evaluate the full population in parallel batches and collect statistics.

    The population is partitioned into batches of size ``min(num_proc, remaining)``.
    Each batch is launched via :func:`run_training_batch`, returning when all
    processes in the batch complete and their stats have been merged.

    Args:
      pop: Full population of genomes to evaluate.
      num_proc: Maximum number of concurrent processes. If ``None``, uses
        ``os.cpu_count() - 1``.
      show_samples: List of booleans determining which individuals in each batch
        should display; a one-element list (e.g., ``[False]``) is **broadcast**
        to the batch size on each iteration.
      overlay_ids: Overlay identifiers to enable in display-enabled processes.

    Returns:
      dict[int, dict[int, list[tuple[float, float]]]]: Mapping of *global*
      population index to that individual's checkpoint stats dict.

    Notes:
      - This function uses a :class:`multiprocessing.Manager` ``dict`` so that
        per-process stats can be retrieved without explicit pipes or queues.
      - Shared-memory frame buffers are currently **not** unlinked here; consider
        cleaning them in a higher-level teardown if needed.

    """
    global shm_names

    # If no number of processes is specified, then we'll use the max minus one
    if num_proc is None:
        num_proc = os.cpu_count()
        assert num_proc is not None
        num_proc -= 1

    pop_stats: dict[int, dict[int, list[tuple[float, float]]]] = {}
    pop_size = len(pop)
    count = 0
    while count < pop_size:
        batch_size = min(pop_size - count, num_proc)

        # Broadcast display argument to all running processes
        if len(show_samples) == 1:
            show_samples *= batch_size
        elif len(show_samples) > batch_size:
            show_samples = show_samples[:batch_size]

        with Manager() as manager:
            # Create shared list for stats (locking)
            shared_pop_stats = manager.dict()
            lock = Lock()

            run_training_batch(
                pop[count : count + batch_size],
                show_samples=show_samples,
                overlay_ids=overlay_ids,
                pop_stats=shared_pop_stats,
                lock=lock,
            )

            for k, s in shared_pop_stats.items():
                pop_stats[count + k] = s[k]

        count += batch_size

    # TODO: Cleanup all shared memory buffers here
    return pop_stats


def fitness(pop_stats: dict[int, dict[int, list[tuple[float, float]]]]) -> list[float]:
    """Compute scalar fitness from per-checkpoint stats.

    The current fitness heuristic sums the recorded distances across all
    checkpoint splits for each individual.

    Args:
      pop_stats: Mapping of population index to that individual's stats dict
        (``checkpoint_id -> [(delta_time, distance_at_split), ...]``).

    Returns:
      list[float]: Fitness values *ordered by population index*.

    """
    def total_dist(v: dict[int, list[tuple[float, float]]]):
        result = 0.0
        for x in v.values():
            for y in x:
                result += y[1]

        return result

    pop_stats_list = [(k, total_dist(s)) for k, s in pop_stats.items()]
    pop_stats_list.sort(key=lambda x: x[0])
    return [x[0] for x in pop_stats_list]


def train(
    num_iters: int,
    pop_size: int,
    log_interval: int = 1,
    top_k: int | float = 0.1,
    **simulation_kwargs,
):
    """Main evolutionary training loop (selection + mutation).

    Repeats:
      1) Evaluate the current population via :func:`run_training_session`.
      2) Rank by fitness.
      3) Keep the best, and refill the population by mutating uniformly sampled
         parents from the top-k set.

    Args:
      num_iters: Number of generations to run.
      pop_size: Number of individuals per generation.
      log_interval: Print progress every N generations.
      top_k: Either the number of top individuals to sample parents from, or a
        fraction in ``(0, 1]`` interpreted as a proportion of the population.
      **simulation_kwargs: Passed through to :func:`run_training_session`.

    Side Effects:
      - Prints best fitness per `log_interval`.
      - Mutates and replaces the population in place each generation.

    """
    pop = [Genome(6, 4) for _ in range(pop_size)]

    if top_k <= 1:
        top_k = int(round(len(pop) * top_k))
    assert isinstance(top_k, int), "top_k must be an integer or a float less than 1"

    for n in range(num_iters):
        stats = run_training_session(pop, **simulation_kwargs)
        scores = fitness(stats)
        scores = [(p, s) for p, s in zip(pop, scores)]
        scores.sort(reverse=True, key=lambda x: x[1])

        if n % log_interval == 0:
            os.system("clear")
            print(f"Best Fitness: {scores[0][0]}")

        newpop = [copy.deepcopy(scores[0][0])]
        for _ in range(len(pop) - 1):
            g = copy.deepcopy(random.choice(scores[:top_k])[0])
            random.choice([g.mutate_weight, g.mutate_add_conn, g.mutate_add_node])()
            newpop.append(g)
        pop = newpop


if __name__ == "__main__":
    train(
        num_iters=10,
        pop_size=32,
        show_samples=[False],
        overlay_ids=[0, 3, 4],
    )
