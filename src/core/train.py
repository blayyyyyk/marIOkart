from __future__ import annotations
from typing import TypeVar, ParamSpec, Callable, Concatenate
from multiprocessing import Process, shared_memory
from threading import Thread, current_thread
import os, copy, random, math
from queue import Queue

# Dependencies
import torch
import numpy as np
from desmume.emulator import DeSmuME
from desmume.controls import keymask

# Local Dependencies
from src.core.model import Genome, EvolvedNet
from src.core.memory import *
from src.visualization.overlay import AVAILABLE_OVERLAYS, DeviceLikeType
from src.visualization.window import (
    EmulatorWindow,
    input_state,
    KEY_MAP,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    on_draw_memoryview,
)
from src.utils.vector import get_mps_device


class EmulatorProcess:
    """Stores model-specific runtime information across emulator frames.

    Tracks checkpoint progress, timing data, and previous distances during
    evaluation of a Mario Kart DS track to enable adaptive fitness evaluation.

    Attributes:
        current_id (int): Current checkpoint identifier.
        times (dict[int, list[tuple[float, float]]]): Per-checkpoint timing and distance data.
        prev_time (float): Time at previous checkpoint.
        prev_dist (float): Distance from previous checkpoint.
        prev_id (int): Identifier for previous checkpoint.
        device (torch.device): Target computation device.
    """

    def __init__(
        self,
        emu: DeSmuME,
        device: DeviceLikeType | None,
        model,
        pop_size: int,
        shm_names: dict[str, str],
        proc_id: int,
    ):
        self.emu = emu
        self.current_id: int = read_current_checkpoint(emu)
        self.times: dict[int, list[tuple[float, float]]] = {}
        self.prev_time: float = 0
        self.prev_dist: float = read_checkpoint_distance_altitude(
            emu, device=device
        ).item()
        self.prev_id: int = read_current_checkpoint(emu)
        self.device = device
        self.model = model
        self.pop_size = pop_size
        self.shm_names = shm_names
        self.proc_id = proc_id

    def get_scores_shared_data(self) -> np.ndarray:
        shm = shared_memory.SharedMemory(name=self.shm_names["scores"])
        scores = np.ndarray(shape=(self.pop_size,), dtype=np.float32, buffer=shm.buf)
        return scores

    def get_display_status_shared_data(self) -> np.ndarray:
        shm_display_statuses = shared_memory.SharedMemory(
            name=self.shm_names["display_status"]
        )
        status = np.ndarray(
            shape=(self.pop_size,), dtype=bool, buffer=shm_display_statuses.buf
        )
        return status

    def get_display_shared_data(self) -> np.ndarray:
        shm_display = shared_memory.SharedMemory(name=self.shm_names["display"])
        display = np.ndarray(
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4),
            dtype=np.uint8,
            buffer=shm_display.buf,
        )
        return display

    def get_display_count(self):
        return np.count_nonzero(self.get_display_status_shared_data())

    @property
    def is_display_enabled(self) -> bool:
        return self.get_display_status_shared_data()[self.proc_id] == True

    def enable_display(self):
        display_status = self.get_display_status_shared_data()
        display_status[self.proc_id] = True

    def get_frame_global_coords(self):
        if not self.is_display_enabled:
            return None, None, None, None

        display_count = self.get_display_count()
        col_count = math.ceil(math.sqrt(display_count))
        row_count = math.ceil(display_count / col_count)
        screen_width = SCREEN_WIDTH // col_count
        screen_height = SCREEN_HEIGHT // row_count
        col_start = (self.proc_id % col_count) * screen_width
        col_end = col_start + screen_width
        row_start = (self.proc_id // col_count) * screen_height
        row_end = row_start + screen_height
        return col_start, col_end, row_start, row_end

    def get_shared_display_clip(self):
        if not self.is_display_enabled:
            return

        x1, x2, y1, y2 = self.get_frame_global_coords()
        display = self.get_display_shared_data()
        return display[y1:y2, x1:x2]

    def set_shared_display_clip(self, arr: np.ndarray):
        x1, x2, y1, y2 = self.get_frame_global_coords()
        display = self.get_display_shared_data()
        display[y1:y2, x1:x2] = arr


def update(pop: list[Genome], scores: list[float]):
    """Update the genetic population based on fitness scores.

    This function is a placeholder for the genetic update rule, which would
    typically include operations such as crossover, mutation, and selection.

    Args:
        pop (list[Genome]): The current population of genomes.
        scores (list[float]): The fitness scores associated with each genome.
    """
    pass


emu_queue = Queue()


def overlay_worker(overlay_func: Callable[[DeSmuME], None]):
    while True:
        emu_instance = emu_queue.get()
        if emu_instance is None:
            break
        try:
            overlay_func(emu_instance)
        except Exception as e:
            raise RuntimeError(f"error on thread {current_thread().name}: {e}")

        emu_queue.task_done()


def wrap_window_func(
    emu_process: EmulatorProcess,
    window: EmulatorWindow | None,
    window_func: Callable[[EmulatorProcess], None],
):
    def on_frame():
        nonlocal window, emu_process

        emu = emu_process.emu
        emu.cycle()
        emu.input.keypad_update(0)
        for key in input_state:
            emu.input.keypad_add_key(keymask(KEY_MAP[key]))

        result = window_func(emu_process)
        buffer = emu.display_buffer_as_rgbx()
        display = emu_process.get_display_shared_data()
        screen_height, screen_width = display.shape[:2]
        x1, x2, y1, y2 = emu_process.get_frame_global_coords()
        assert x1 is not None and x2 is not None and y1 is not None and y2 is not None
        scale_x = (x2 - x1) / screen_width
        scale_y = (y2 - y1) / screen_height
        overlay_arr = on_draw_memoryview(buffer, screen_width, screen_height, scale_x, scale_y)
        emu_process.set_shared_display_clip(overlay_arr)
            

        if result is None and window is not None:
            window.kill()
            return True

        if result is None and window is None:
            emu_queue.put(None)
            return True

        emu_queue.put(emu)

        return True

    return on_frame


P = ParamSpec("P")
R = TypeVar("R")


def try_worker(func: Callable[P, R], proc_id, shm: shared_memory.SharedMemory):
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in process {proc_id}: {e}")
            shm.close()
            shm.unlink()

    return wrapper


def emu_worker_entry(
    watch_func,
    shm_names: dict[str, str],
    proc_id: int,
    pop_size: int,
    model,
    show_display: bool,
    overlays: list[Callable[[DeSmuME, DeviceLikeType], None]],
):
    """Entry point for an emulator worker process.

    Each worker loads a DeSmuME instance, initializes the game state, and
    continuously calls `watch_func` to simulate gameplay and evaluate
    the performance of a given NEAT model.

    Args:
        watch_func (Callable): A function that runs per emulator frame and returns
            results or `None` to stop the worker loop.
        shm_name (str): Name of the shared memory region used to communicate scores.
        proc_id (int): The worker’s unique process ID within the population batch.
        pop_size (int): Total size of the population.
        model (Genome | torch.nn.Module): The model to evaluate.

    Returns:
        Any: The result from the last call to `watch_func`, typically a fitness metric.
    """

    # Prevents DeSmuMe from creating an SDL window
    if not show_display:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    emu = DeSmuME()
    emu.open("mariokart_ds.nds")
    emu.savestate.load(3)
    emu.volume_set(0)
    emu.cycle()

    device = get_mps_device()
    emu_process = EmulatorProcess(emu, device, model, pop_size, shm_names, proc_id)

    # Standby phase — keep cycling until a valid result is produced
    result = None
    while result is not None:
        emu.cycle()
        result = watch_func(emu_process)

    # optional display
    window = None
    if show_display:
        emu_process.enable_display()

    if show_display and proc_id == 0:
        window = EmulatorWindow(emu)
        window.show_all()

    # optional overlays
    if len(overlays) != 0 and show_display:
        # compose overlay function to send to worker
        def overlay_bundle(emu: DeSmuME):
            nonlocal device
            for overlay in overlays:

                overlay(emu, device)

        thread = Thread(target=overlay_worker, args=(overlay_bundle,), daemon=True)
        thread.start()  # non-blocking

    # Active phase — simulate and collect results (no display)
    while not show_display:
        emu.cycle()
        tmp = watch_func(emu_process)
        if tmp is None:
            break
        result = tmp

    # Active phase — simulate and collect results (show display)
    if show_display and proc_id == 0:
        assert window is not None
        window_func = wrap_window_func(emu_process, window, watch_func)

        global emu_queue
        window.start(window_func, emu_queue)  # blocking

    # Report fitness via shared memory
    scores = emu_process.get_scores_shared_data()
    scores[proc_id] = 20

    return result


def worker(emu_process: EmulatorProcess):
    """Runs one frame of model evaluation in the emulator.

    Reads race state from emulator memory (clock, checkpoint, obstacle distances, etc.),
    converts data to normalized tensors, feeds them through the NEAT model,
    and determines when the racer should terminate based on elapsed time.

    Args:
        emu (DeSmuME): Active emulator instance.
        shm_name (str): Name of the shared memory block.
        proc_id (int): Index of the current process in the population.
        pop_size (int): Total population size.
        model (Genome | torch.nn.Module): Model being evaluated.
        model_state (ModelState): Persistent state for checkpoint tracking.

    Returns:
        float | None: Current in-game clock if still running, or `None` to end.
    """
    emu = emu_process.emu
    prev_id = read_current_checkpoint(emu)
    clock = read_clock(emu)

    if emu_process.current_id != emu_process.prev_id:
        assert isinstance(emu_process.current_id, int)
        if emu_process.current_id not in emu_process.times:
            emu_process.times[emu_process.current_id] = []

        current_time = clock
        emu_process.times[emu_process.current_id].append(
            (current_time - emu_process.prev_time, emu_process.prev_dist)
        )
        emu_process.prev_time = current_time
        emu_process.prev_dist = read_checkpoint_distance_altitude(
            emu, device=emu_process.device
        ).item()

    s1 = 60.0

    # Sensor inputs (obstacle distances)
    forward_d = read_forward_distance_obstacle(emu, device=emu_process.device)
    left_d = read_left_distance_obstacle(emu, device=emu_process.device)
    right_d = read_right_distance_obstacle(emu, device=emu_process.device)
    inputs_dist1 = torch.tensor([forward_d, left_d, right_d], device=emu_process.device)
    inputs_dist1 = torch.tanh(1 - inputs_dist1 / s1)

    # Angular relationship to next checkpoint
    angle = read_direction_to_checkpoint(emu, device=emu_process.device)
    forward_a = torch.cos(angle)
    left_a = torch.sin(angle)
    right_a = -torch.sin(angle)
    inputs_dist2 = torch.tensor([forward_a, left_a, right_a], device=emu_process.device)

    # Model inference
    inputs = torch.cat([inputs_dist1, inputs_dist2])
    logits = emu_process.model(inputs)

    clock = read_clock(emu)
    if clock > 10000:
        return None

    return clock


def run_training_session(
    pop: list[Genome],
    shm_names: dict[str, str],
    num_proc: int | None = None,
    show_sample: list[bool] = [],
    display_overlay_ids: list[int] = [],
):
    """Run a full training evaluation cycle across multiple processes.

    Launches emulator instances in parallel to evaluate a population of NEAT genomes.
    Shared memory is used for inter-process communication of fitness scores.

    Args:
        pop (list[Genome]): The genome population to evaluate.
        num_proc (int | None): Number of processes to spawn in parallel.
            If None, automatically uses all available cores minus one.
        show_sample (bool): Whether to display one of the population
            simulations to a GTK window. Defaults to False.

    Returns:
        np.ndarray: Array of fitness scores corresponding to each genome.
    """
    global shm_score

    print(f"Running {num_proc} processes")
    pop_size = len(pop)
    idx = 0

    max_cores = os.cpu_count() or 1
    if num_proc is None:
        num_proc = max_cores - 1 if max_cores > 1 else 1

    if len(show_sample) == 0:
        show_sample = [False] * num_proc
    elif len(show_sample) == 1:
        show_sample = show_sample * num_proc

    assert len(show_sample) == num_proc

    overlays = []
    for id in display_overlay_ids:
        overlays.append(AVAILABLE_OVERLAYS[id])

    while idx < pop_size:
        group_size = min(num_proc, pop_size - idx)
        procs = []

        for i in range(group_size):
            proc_id = idx + i
            model = EvolvedNet(pop[proc_id])
            proc = Process(
                target=emu_worker_entry,
                args=(
                    worker,
                    shm_names,
                    proc_id,
                    pop_size,
                    model,
                    proc_id == 0 and show_sample[proc_id],
                    overlays,
                ),
            )
            procs.append(proc)

        for p in procs:
            p.start()
        for p in procs:
            p.join()

        idx += group_size
        print(f"Finished batch ending at proc_id={idx-1}")

    scores = np.ndarray(shape=(pop_size,), dtype=np.float32, buffer=shm_score.buf)
    return scores


def train(
    num_iters: int,
    pop_size: int,
    log_interval: int = 1,
    top_k: int | float = 0.1,
    **simulation_kwargs,
):
    """Main NEAT training loop with multiprocessing and shared-memory fitness evaluation.

    Performs iterative evolution of neural networks that control the racer in
    Mario Kart DS via the DeSmuME emulator. Each iteration evaluates the population,
    sorts them by fitness, and generates the next generation through mutation.

    Args:
        num_iters (int): Number of generations to train.
        pop_size (int): Population size.
        num_proc (int | None): Number of emulator workers to run in parallel.
        log_interval (int): Print interval for training progress.
        top_k (int | float): Number (or proportion) of top genomes to select for mutation.
    """
    pop = [Genome(6, 4) for _ in range(pop_size)]

    if top_k <= 1:
        top_k = int(round(len(pop) * top_k))
    assert isinstance(top_k, int), "top_k must be an integer or a float less than 1"

    for n in range(num_iters):
        scores = run_training_session(pop, **simulation_kwargs)
        scores = scores.tolist()
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
    pop_size = 64

    shm_score = shared_memory.SharedMemory(
        create=True, size=pop_size * np.float32().nbytes
    )
    shm_display_statuses = shared_memory.SharedMemory(
        create=True, size=pop_size * np.bool().nbytes
    )
    shm_display = shared_memory.SharedMemory(
        create=True, size=SCREEN_WIDTH * SCREEN_HEIGHT * 4 * np.uint8().nbytes
    )
    shm_names = {
        "display_status": shm_display_statuses.name,
        "score": shm_score.name,
        "display": shm_display.name,
    }

    train(
        num_iters=2000,
        pop_size=1000,
        num_proc=8,
        show_sample=[True],
        display_overlay_ids=[0, 3, 4],
        shm_names=shm_names,
    )
    shm_score.close()
    shm_score.unlink()
