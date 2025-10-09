from utils.racer import Racer
from utils.model import EvolvedNet, Genome, NodeGene, ConnGene
from utils.vector import get_mps_device
import random, copy
from desmume.emulator import DeSmuME
from utils.emulator import run_emulator
import json
import torch
from desmume.controls import Keys, keymask
import os

in_nodes = 6
out_nodes = 4

SAVE_STATE_ID = 0

max_dist = 0
racer = None
device = None

controls = [Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_A, Keys.KEY_B]


def calculate_max_course_distance(pts1: torch.Tensor, pts2: torch.Tensor):
    pts3 = torch.cat([pts1[1:], pts1[0]])
    pts4 = torch.cat([pts2[1:], pts2[0]])
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


def fitness(genome: Genome, emu: DeSmuME, max_time: int = 60000):
    global racer, max_dist
    assert racer is not None, "racer has not been initialized"
    emu.volume_set(0)
    emu.savestate.load(SAVE_STATE_ID)
    pts1 = racer.nkm._CPOI.position1
    pts2 = racer.nkm._CPOI.position2
    max_dist = calculate_max_course_distance(pts1, pts2)

    score = 0
    model = EvolvedNet(genome)
    device = racer.device
    times = {}
    prev_time = 0
    racer.memory = emu.memory.unsigned
    prev_dist = racer.get_checkpoint_distance_altitude().item()
    
    while True:
        prev_id = racer.get_checkpoint_details()["checkpoint"]

        racer.memory = emu.memory.unsigned
        current_id = prev_id = racer.get_checkpoint_details()["checkpoint"]

        if current_id != prev_id:
            if current_id not in times:
                times[prev_id] = []

            current_time = racer.clock
            times[prev_id].append((current_time - prev_time, prev_dist))
            prev_time = current_time
            prev_dist = racer.get_checkpoint_distance_altitude().item()

        s1 = 60.0

        inputs_dist1 = torch.cat(
            [
                racer.get_forward_distance_obstacle(),
                racer.get_left_distance_obstacle(),
                racer.get_right_distance_obstacle(),
            ]
        )
        inputs_dist1 = torch.tanh(1 - inputs_dist1 / s1)

        angle = racer.get_direction_checkpoint()
        inputs_dist2 = torch.cat(
            [torch.cos(angle), torch.sin(angle), -torch.sin(angle)]
        )

        inputs = torch.cat([inputs_dist1, inputs_dist2])

        logits = model(inputs)
        for i, v in enumerate(logits.tolist()):
            if v > 0:
                emu.input.keypad_add_key(keymask(controls[i]))
            else:
                emu.input.keypad_rm_key(keymask(controls[i]))

        if racer.clock > max_time:
            current_time = racer.clock
            current_dist = racer.get_checkpoint_distance_altitude().item()
            times[prev_id] = (current_time - prev_time, prev_dist - current_dist)
            break

        yield None

    dist = 0
    for id, l in times.items():
        dist += sum([t for t, _ in l]) / len(l)

    score = dist / max_dist

    yield score


def train(
    emu,
    fitness_fn,
    target_fitness=0,
    genome_pop_size=20,
    max_epoch=50,
    top_k=10,
    log_interval=5,
):
    global racer, device

    min_fitness = 10000000
    pop = [Genome(in_nodes, out_nodes) for i in range(genome_pop_size)]
    for g in pop:
        g.mutate_add_conn()

    count = 0
    while min_fitness > target_fitness or count < max_epoch:
        # evaluate
        #
        scores = []
        for g in pop:
            for x in fitness_fn(g, emu):
                if x is None:
                    yield True
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


def main(emu: DeSmuME):
    train_gen = train(
        emu, fitness, genome_pop_size=20, max_epoch=50, top_k=10, log_interval=5
    )
    for i in train_gen:
        yield i
