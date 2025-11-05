from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjConfig import *
from private.mkds_python_bindings.testing.nkm import *
from private.mkds_python_bindings.testing.sfx import *
from private.mkds_python_bindings.testing.stateMachine import *
from private.mkds_python_bindings.testing.types import *


class mobj_inst_t(Structure):
    _fields_ = [
        ('objectId', u16),
        ('flags', u16),
        ('position', VecFx32),
        ('velocity', VecFx32),
        ('scale', VecFx32),
        ('mtx', MtxFx43),
        ('size', VecFx32),
        ('colEntryId', s16),
        ('alpha', u16),
        ('nearClip', fx32),
        ('farClip', fx32),
        ('sfxMaxDistanceSquare', c_int),
        ('clipAreaMask', u32),
        ('visibilityFlags', u32),
        ('has3DModel', u16),
        ('rotY', u16),
        ('stateMachine', state_machine_t),
        ('soundEmitter', u32), #POINTER(sfx_emitter_t)),
        ('config', u32), #POINTER(mobj_config_t)),
        ('objiEntry', u32), #POINTER(nkm_obji_entry_t)),
    ]
