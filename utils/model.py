import torchlens as tl
import torch, torch.nn as nn, torch.nn.functional as F
import random, copy
from utils.vector import get_mps_device

device = get_mps_device()

class NodeGene:
    def __init__(self, nid, ntype):
        self.id = nid
        self.type = ntype # input | hidden | output

class ConnGene:
    def __init__(self, in_id, out_id, w, enabled=True):
        self.in_id = in_id
        self.out_id = out_id
        self.w = w
        self.enabled = enabled

class Genome:
    def __init__(self, n_inputs, n_outputs):
        self.nodes = [NodeGene(i, "input") for i in range(n_inputs)] + \
                     [NodeGene(n_inputs+i, "output") for i in range(n_outputs)]
        self.conns = []
        self.next_node_id = n_inputs + n_outputs

    def mutate_weight(self):
        if self.conns:
            c = random.choice(self.conns)
            c.w += torch.randn(1).item() * 0.1

    def mutate_add_conn(self):
        a, b = random.sample(self.nodes, 2)
        if a.type == "output" or b.type == "input": return
        self.conns.append(ConnGene(a.id, b.id, random.uniform(-1, 1)))

    def mutate_add_node(self):
        if not self.conns: return
        conn = random.choice(self.conns)
        if not conn.enabled: return
        conn.enabled = False
        new_id = self.next_node_id; self.next_node_id += 1
        new_node = NodeGene(new_id, "hidden")
        self.nodes.append(new_node)
        self.conns.append(ConnGene(conn.in_id, new_id, 1.0))
        self.conns.append(ConnGene(new_id, conn.out_id, conn.w))


class EvolvedNet(nn.Module):
    def __init__(self, genome):
        super().__init__()
        self.g = genome
        self.params = nn.ParameterDict()
        for c in genome.conns:
            if c.enabled:
                self.params[f"w_{c.in_id}_{c.out_id}"] = nn.Parameter(torch.tensor(c.w))
        self.inputs = [n.id for n in genome.nodes if n.type=="input"]
        self.outputs = [n.id for n in genome.nodes if n.type=="output"]

    def forward(self, x):
        # map node IDs to torch tensors
        vals = {nid: torch.tensor(0.0) for nid in [n.id for n in self.g.nodes]}
    
        # assign input activations
        for i, nid in enumerate(self.inputs):
            vals[nid] = x[i]
    
        # propagate through connections iteratively
        for _ in range(len(self.g.nodes)):
            for (k, v) in self.params.items():
                i, o = map(int, k[2:].split("_"))
                vals[o] = vals[o] + torch.tanh(vals[i] * v)
                
    
        # collect output activations as tensor
        return torch.stack([torch.tanh(vals[o]).to(device) for o in self.outputs])

if __name__ == "__main__":
    def fitness(genome) -> float:
        net = EvolvedNet(genome)
        
        data = [([0,0],0),([0,1],1),([1,0],1),([1,1],0)]
        err = 0
        for x,y in data:
            
            y_hat = net(torch.tensor(x, dtype=torch.float32))
            err += (y_hat - y)**2
        return -err.item()  # higher is better
    
    # -------------------------
    #  Evolution loop
    # -------------------------
    POP: list[Genome] = [Genome(2,1) for _ in range(30)]
    for g in POP: g.mutate_add_conn()  # start with random links
    
    for gen in range(50):
        scores = [(fitness(g), g) for g in POP]
        scores.sort(reverse=True, key=lambda x:x[0])
        #print(f"Gen {gen}: best fitness {scores[0][0]:.3f}")
        tl.show_model_graph(EvolvedNet(scores[0][1]), torch.tensor([0, 0], dtype=torch.float32), vis_opt="rolled")
        print(len(scores[0][1].conns))
        if scores[0][0] > -0.05:
            print("✅ Solved!")
            break
        # selection and mutation
        newpop = [copy.deepcopy(scores[0][1])]
        for _ in range(len(POP)-1):
            g = copy.deepcopy(random.choice(scores[:10])[1])
            random.choice([g.mutate_weight, g.mutate_add_conn, g.mutate_add_node])()
            newpop.append(g)
        POP = newpop