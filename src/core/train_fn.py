from __future__ import annotations
from typing import TypeVar, ParamSpec, Callable, Generator, Concatenate, Union, Generic
from typing_extensions import NoReturn
import torch
from desmume.emulator import DeSmuME
from src.core.model import Genome
from src.core.memory import read_clock, read_position
from typing import Any
from multiprocessing import Process, shared_memory
import numpy as np
from threading import Thread
import os, copy, random
from src.core.memory import *
from src.utils.vector import get_mps_device


def update(pop: list[Genome], scores: list[float]):
    """Update the genetic population based on fitness scores.

    This function is a placeholder for the genetic update rule, which would
    typically include operations such as crossover, mutation, and selection.

    Args:
        pop (list[Genome]): The current population of genomes.
        scores (list[float]): The fitness scores associated with each genome.
    """
    pass


def emu_worker_entry(watch_func, shm_name: str, proc_id: int, pop_size: int, model):
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
    emu = DeSmuME()
    emu.open("mariokart_ds.nds")
    emu.savestate.load(3)
    emu.volume_set(0)
    emu.cycle()

    result = None

    # Standby phase — keep cycling until a valid result is produced
    while result is not None:
        emu.cycle()
        result = watch_func(emu, shm_name, proc_id, pop_size, model)

    device = get_mps_device()
    model_state = ModelState(
        read_current_checkpoint(emu),
        {},
        0,
        read_checkpoint_distance_altitude(emu, device=device).item(),
        read_current_checkpoint(emu),
        device,
    )

    # Active phase — simulate and collect results
    while True:
        emu.cycle()
        tmp = watch_func(emu, shm_name, proc_id, pop_size, model, model_state)
        if tmp is None:
            break
        result = tmp

    # Report fitness via shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    scores = np.ndarray(shape=(pop_size,), dtype=np.float32, buffer=shm.buf)
    scores[proc_id] = 20

    return result


class ModelState:
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

    def __init__(self, current_id, times, prev_time, prev_dist, prev_id, device):
        self.current_id: int = current_id
        self.times: dict[int, list[tuple[float, float]]] = times
        self.prev_time: float = prev_time
        self.prev_dist: float = prev_dist
        self.prev_id: int = prev_id
        self.device = device


def worker(
    emu: DeSmuME,
    shm_name: str,
    proc_id: int,
    pop_size: int,
    model,
    model_state: ModelState,
):
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
    prev_id = read_current_checkpoint(emu)
    clock = read_clock(emu)

    if model_state.current_id != model_state.prev_id:
        assert isinstance(model_state.current_id, int)
        if model_state.current_id not in model_state.times:
            model_state.times[model_state.current_id] = []

        current_time = clock
        model_state.times[model_state.current_id].append(
            (current_time - model_state.prev_time, model_state.prev_dist)
        )
        model_state.prev_time = current_time
        model_state.prev_dist = read_checkpoint_distance_altitude(
            emu, device=model_state.device
        ).item()

    s1 = 60.0

    # Sensor inputs (obstacle distances)
    forward_d = read_forward_distance_obstacle(emu, device=model_state.device)
    left_d = read_left_distance_obstacle(emu, device=model_state.device)
    right_d = read_right_distance_obstacle(emu, device=model_state.device)
    inputs_dist1 = torch.tensor([forward_d, left_d, right_d], device=model_state.device)
    inputs_dist1 = torch.tanh(1 - inputs_dist1 / s1)

    # Angular relationship to next checkpoint
    angle = read_direction_to_checkpoint(emu, device=model_state.device)
    forward_a = torch.cos(angle)
    left_a = torch.sin(angle)
    right_a = -torch.sin(angle)
    inputs_dist2 = torch.tensor([forward_a, left_a, right_a], device=model_state.device)

    # Model inference
    inputs = torch.cat([inputs_dist1, inputs_dist2])
    logits = model(inputs)

    clock = read_clock(emu)
    if clock > 5000:
        return None

    return clock


def run_training_session(pop: list[Genome], num_proc: int | None = None):
    """Run a full training evaluation cycle across multiple processes.

    Launches emulator instances in parallel to evaluate a population of NEAT genomes.
    Shared memory is used for inter-process communication of fitness scores.

    Args:
        pop (list[Genome]): The genome population to evaluate.
        num_proc (int | None): Number of processes to spawn in parallel.
            If None, automatically uses all available cores minus one.

    Returns:
        np.ndarray: Array of fitness scores corresponding to each genome.
    """
    global shm
    print(f"Running {num_proc} processes")
    pop_size = len(pop)
    idx = 0

    max_cores = os.cpu_count() or 1
    if num_proc is None:
        num_proc = max_cores - 1 if max_cores > 1 else 1

    while idx < pop_size:
        group_size = min(num_proc, pop_size - idx)
        procs = []

        for i in range(group_size):
            proc_id = idx + i
            model = pop[proc_id]
            proc = Process(
                target=emu_worker_entry,
                args=(worker, shm.name, proc_id, pop_size, model),
            )
            procs.append(proc)

        for p in procs:
            p.start()
        for p in procs:
            p.join()

        idx += group_size
        print(f"Finished batch ending at proc_id={idx-1}")

    scores = np.ndarray(shape=(pop_size,), dtype=np.float32, buffer=shm.buf)
    return scores


def train(
    num_iters: int,
    pop_size: int,
    num_proc: int | None = None,
    log_interval: int = 1,
    top_k: int | float = 0.1,
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
    pop = [Genome(10, 10) for _ in range(pop_size)]

    if top_k <= 1:
        top_k = int(round(len(pop) * top_k))
    assert isinstance(top_k, int), "top_k must be an integer or a float less than 1"

    for n in range(num_iters):
        scores = run_training_session(pop, num_proc=num_proc)
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
    shm = shared_memory.SharedMemory(create=True, size=pop_size * np.float32().nbytes)
    train(num_iters=4, pop_size=64)
    shm.close()
    shm.unlink()
