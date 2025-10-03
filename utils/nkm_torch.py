from mkds.nkm import NKM as _NKM, CPOI as _CPOI
import torch

class CPOI(_CPOI):
    def __init__(self, data, device=None):
        super().__init__(data)
        self.position1 = torch.tensor(self.position1, device=device)
        self.position2 = torch.tensor(self.position2, device=device)

class NKM(_NKM):
    def __init__(self, data, device=None):
        super().__init__(data)
        
        self._CPOI = CPOI(self._data[self._CPOI_offset:self._CPAT_offset], device)
        