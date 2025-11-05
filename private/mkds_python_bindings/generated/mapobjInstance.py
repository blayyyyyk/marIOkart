from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjConfig import *
from private.mkds_python_bindings.generated.nkm import *
from private.mkds_python_bindings.generated.sfx import *
from private.mkds_python_bindings.generated.stateMachine import *
from private.mkds_python_bindings.generated.types import *


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
        ('soundEmitter', POINTER32(sfx_emitter_t)),
        ('config', POINTER32(mobj_config_t)),
        ('objiEntry', POINTER32(nkm_obji_entry_t)),
    ]
