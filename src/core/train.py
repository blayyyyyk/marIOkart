from types import GeneratorType
from src.core.memory import *
from src.core.model import EvolvedNet, Genome, NodeGene, ConnGene
from src.utils.vector import get_mps_device
import random, copy
from desmume.emulator import DeSmuME
import torch
from desmume.controls import Keys, keymask
import os


from typing import Generator


course_parent_directory = "courses"

in_nodes = 6
out_nodes = 4



max_dist = 0
racer = None
device = None

controls = [Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_A, Keys.KEY_B]


def calculate_max_course_distance(pts1: torch.Tensor, pts2: torch.Tensor):
    pts3 = torch.cat([pts1[1:], pts1[None, 0]])
    pts4 = torch.cat([pts2[1:], pts2[None, 0]])
    sum = 0
    for p1, p2 in zip(pts1.unbind(), pts2.unbind()):
        for p3, p4 in zip(pts3.unbind(), pts4.unbind()):
            d1 = torch.norm(p1 - p3)
            d2 = torch.norm(p1 - p4)
            d3 = torch.norm(p2 - p3)
            d4 = torch.norm(p2 - p4)
            max_dist = max(d1, d2, d3, d4)
            sum += max_dist

    return sum


def fitness(emu: DeSmuME, genome: Genome, max_time: int = 20000, device=None):
    emu.volume_set(0)
    emu.savestate.load(SAVE_STATE_ID)
    pts1, pts2 = read_checkpoint_positions(emu, device=device).chunk(2, dim=1)
    max_dist = calculate_max_course_distance(pts1, pts2)
    
    score = 0
    model = EvolvedNet(genome).to(device)
    times = {}
    prev_time = 0
    prev_dist = read_checkpoint_distance_altitude(emu, device=device).item()
    current_id = read_current_checkpoint(emu)

    while True:
        prev_id = read_current_checkpoint(emu)
        clock = read_clock(emu)

        if current_id != prev_id:
            if current_id not in times:
                times[prev_id] = []

            current_time = clock
            times[prev_id].append((current_time - prev_time, prev_dist))
            prev_time = current_time
            prev_dist = read_checkpoint_distance_altitude(emu, device=device).item()

        s1 = 60.0
        
        forward_d = read_forward_distance_obstacle(emu, device=device)
        left_d = read_left_distance_obstacle(emu, device=device)
        right_d = read_right_distance_obstacle(emu, device=device)
        inputs_dist1 = torch.tensor([forward_d, left_d, right_d], device=device)
        inputs_dist1 = torch.tanh(1 - inputs_dist1 / s1)

        angle = read_direction_to_checkpoint(emu, device=device)
        forward_a = torch.cos(angle)
        left_a = torch.sin(angle)
        right_a = -torch.sin(angle)
        inputs_dist2 = torch.tensor([forward_a, left_a, right_a], device=device)

        inputs = torch.cat([inputs_dist1, inputs_dist2])
        
        logits = model(inputs)
        

        if clock > max_time:
            if current_id not in times:
                times[prev_id] = []
            
            current_time = clock
            current_dist = read_checkpoint_distance_altitude(emu, device=device).item()
            times[prev_id].append((current_time - prev_time, prev_dist - current_dist))
            break

        
        yield logits

    dist = 0
    for id, l in times.items():
        dist += sum([t for t, _ in l]) / len(l)

    score = dist / max_dist

    yield score


def train(
    emu: DeSmuME,
    save_id: int,
    fitness_fn,
    target_fitness=0,
    genome_pop_size=20,
    max_epoch=50,
    top_k=10,
    log_interval=5,
    device=None
) -> Generator[torch.Tensor | None]:
    emu.savestate.load(save_id)
    
    yield None

    min_fitness = 10000000
    pop = [Genome(in_nodes, out_nodes) for i in range(genome_pop_size)]
    for g in pop:
        g.mutate_add_conn()

    count = 0
    while min_fitness > target_fitness or count < max_epoch:
        # evaluate
        #
        scores = []
        logits = None
        for g in pop:
            for x in fitness_fn(emu, g, device=device):
                if x.ndim != 0:
                    logits = x
                    yield logits
                    continue

                
                scores.append((x, g))

        scores.sort(reverse=True, key=lambda x: x[0])
        if scores[0][0] < min_fitness:
            min_fitness = scores[0][0]

        if count % log_interval == 0:
            os.system("clear")
            print(f"Best Fitness Overall: {min_fitness}\nBest Fitness: {scores[0][0]}")

        # apply update rule
        newpop = [copy.deepcopy(scores[0][1])]
        for _ in range(len(pop) - 1):
            g = copy.deepcopy(random.choice(scores[:top_k])[1])
            random.choice([g.mutate_weight, g.mutate_add_conn, g.mutate_add_node])()
            newpop.append(g)
        pop = newpop

        count += 1
