import os, sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.vector import cross_product, get_mps_device

device = get_mps_device()

def main():
    a = torch.tensor([0.,1.,0.], device=device)
    b = torch.tensor([ 0.0000,  0.2650, -0.9642], device=device)
    c = cross_product(a, b)
    print(c)

if __name__ == "__main__":
    main()