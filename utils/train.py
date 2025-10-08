from utils.model import EvolvedNet, Genome, NodeGene, ConnGene

in_nodes = 10
out_nodes = 4


def fitness(genome: Genome):
    pass

def train(genome_pop_size, max_epoch, target_fitness, fitness_fn):
    min_fitness = 10000000
    pop = [Genome(in_nodes, out_nodes) for i in range(genome_pop_size)]
    for g in pop: g.mutate_add_conn()
    
    while min_fitness > target_fitness:
        scores = [(fitness_fn(g), g) for g in pop]
        scores.sort(reverse=True, key=lambda x:x[0])
        
        