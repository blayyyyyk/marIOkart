from __future__ import annotations
from mkds.nkm import NKM, CPOI
from src.utils.vector import extrapolate
import torch


class CPOITensor(CPOI):
    def __init__(self, data, device=None):
        super().__init__(data)
        self.position1 = torch.tensor(self.position1, device=device)
        self.position2 = torch.tensor(self.position2, device=device)


class NKMTensor(NKM):
    def __init__(self, data, device=None):
        super().__init__(data)

        self._CPOI = CPOITensor(
            self._data[self._CPOI_offset : self._CPAT_offset], device
        )
