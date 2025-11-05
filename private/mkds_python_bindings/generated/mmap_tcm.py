from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_hw_ARM9_mmap_tcm_h_87_18(Structure):
    _fields_ = [
        ('sys_and_irq_stack', (u8 * 16256)),
        ('svc_stack', (u8 * 64)),
        ('reserved', (u8 * 56)),
        ('intr_check', u32),
        ('intr_vector', c_void_p32),
    ]
