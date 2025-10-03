import os, sys
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.kcl_torch import KCLTensor as KCL
from mkds.utils import (
    read_u16
)

if __name__ == "__main__":
    kcl = KCL.from_file("desert_course/src/course_collision.kcl")
    point = (-1500, 300, 1000)
    tri = kcl.triangles
    print(tri.shape)