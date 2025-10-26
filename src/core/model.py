from __future__ import annotations
from torch._prims_common import DeviceLikeType
import torchlens as tl
import torch, torch.nn as nn, torch.nn.functional as F
import random, copy
from src.utils.vector import get_mps_device
import dataclasses, json
from pathlib import Path

device = None


class NodeGene:
    def __init__(self, nid, ntype):
        self.id = nid
        self.type = ntype # input | hidden | output
        
    def __eq__(self, other):
        if not isinstance(other, NodeGene):
            return False
        
        return self.id == other.id and self.type == other.type
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __hash__(self):
        return hash((self.id, self.type))


class ConnGene:
    def __init__(self, in_id, out_id, w, enabled=True):
        self.in_id = in_id
        self.out_id = out_id
        self.w = w
        self.enabled = enabled
        
    def __eq__(self, other):
        if not isinstance(other, ConnGene):
            return False
        
        return self.in_id == other.in_id and self.out_id == other.out_id and self.w == other.w and self.enabled == other.enabled
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __hash__(self):
        return hash((self.in_id, self.out_id, self.w, self.enabled))

class Genome:
    def __init__(self, n_inputs, n_outputs, device: DeviceLikeType | None = None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes = [NodeGene(i, "input") for i in range(n_inputs)] + \
                     [NodeGene(n_inputs+i, "output") for i in range(n_outputs)]
        self.conns = []
        self.next_node_id = n_inputs + n_outputs
        self.device = device
        

    def mutate_weight(self):
        if self.conns:
            c = random.choice(self.conns)
            c.w += torch.randn(1, device=self.device).item() * 0.1

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
        
    def __eq__(self, other):
        return set(self.nodes) == set(other.nodes) and set(self.conns) == set(other.conns) and self.next_node_id == other.next_node_id and self.n_inputs == other.n_inputs and self.n_outputs == other.n_outputs

    def __hash__(self):
        if len(self.conns) == 0:
            return hash((len(self.nodes), len(self.conns), self.next_node_id, self.n_inputs, self.n_outputs))
            
        return hash((self.conns[0], len(self.nodes), len(self.conns), self.next_node_id, self.n_inputs, self.n_outputs))

class EvolvedNet(nn.Module):
    def __init__(self, genome, device: DeviceLikeType | None = None):
        super().__init__()
        self.g = genome
        self.device = device
        self.params = nn.ParameterDict()
        for c in genome.conns:
            if c.enabled:
                self.params[f"w_{c.in_id}_{c.out_id}"] = nn.Parameter(torch.tensor(c.w, device=device))
        self.inputs = [n.id for n in genome.nodes if n.type=="input"]
        self.outputs = [n.id for n in genome.nodes if n.type=="output"]


    def forward(self, x):
        # map node IDs to torch tensors
        vals = {nid: torch.tensor(0.0, device=self.device) for nid in [n.id for n in self.g.nodes]}

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

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Genome):
            return {
                "n_inputs": o.n_inputs,
                "n_outputs": o.n_outputs,
                "nodes": o.nodes,
                "conns": o.conns
            }
        if isinstance(o, NodeGene):
            return {
                "id": o.id,
                "type": o.type
            }
        if isinstance(o, ConnGene):
            return {
                "enabled": o.enabled,
                "in_id": o.in_id,
                "out_id": o.out_id,
                "w": o.w
            }

        return super().default(o)

def load_genome(file_path: str | Path):
    tmp_conns: list[ConnGene] = []
    tmp_nodes: list[NodeGene] = []
    tmp_object: None | Genome = None
    
    def as_node(dct):
        nonlocal tmp_object, tmp_nodes
        
        if 'id' in dct and 'type' in dct:
            node = NodeGene(dct['id'], dct['type'])
            tmp_nodes.append(node)
            return
            
    def as_conn(dct):
        nonlocal tmp_object, tmp_conns
        
        if 'enabled' in dct and 'in_id' in dct and 'out_id' in dct and 'w' in dct:
            conn = ConnGene(dct['in_id'], dct['out_id'], dct['w'], dct['enabled'])
            tmp_conns.append(conn)
            return
    
    def as_genome(dct):
        nonlocal tmp_object
        if 'n_outputs' in dct and 'n_inputs' in dct and 'conns' in dct and 'nodes' in dct:
            tmp_object = Genome(dct['n_inputs'], dct['n_outputs'])
            tmp_object.nodes = tmp_nodes
            tmp_object.conns = tmp_conns
            
        if 'id' in dct and 'type' in dct:
            as_node(dct)
            if tmp_object is not None:
                tmp_object.nodes = tmp_nodes
            
        if 'enabled' in dct and 'in_id' in dct and 'out_id' in dct and 'w' in dct:
            as_conn(dct)
            if tmp_object is not None:
                tmp_object.conns = tmp_conns
            
        return tmp_object
        
    with open("test_genome.json", 'r') as f:
        data = f.read()
        genome = json.loads(data, object_hook=as_genome)
        return genome

def save_genome(genome: Genome, file_path: str | Path):
    with open(file_path, 'w') as f:
        json.dump(genome, f, indent=4, cls=JSONEncoder)

if __name__ == "__main__":
    device = get_mps_device()

    POP: list[Genome] = [Genome(6,3) for _ in range(30)]
    for g in POP: g.mutate_add_conn()  # start with random links

    save_genome(POP[0], "test_genome.json")
    loaded_genome = load_genome("test_genome.json")
    loaded_genome.mutate_add_conn()
    print(hash(loaded_genome))
