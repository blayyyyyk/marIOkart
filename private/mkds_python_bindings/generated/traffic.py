from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.animationManager import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.light import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.mapobjRenderPart import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.model2 import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.quaternion import *
from private.mkds_python_bindings.generated.sfx import *
from private.mkds_python_bindings.generated.types import *


class traffic_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fieldA0', c_int),
        ('fieldA4', quaternion_t),
        ('fieldB4', quaternion_t),
        ('fieldC4', quaternion_t),
        ('fieldD4', c_int),
        ('pathWalker', pw_pathwalker_t),
        ('initialPoint', s16),
        ('fieldFE', s16),
        ('field100', fx32),
        ('field104', fx32),
        ('field108', fx32),
        ('field10C', u16),
        ('field110', fx32),
        ('field114', fx32),
        ('field118', fx32),
        ('field11C', fx32),
        ('field120', fx32),
        ('model', POINTER32(model_t)),
        ('shadowModel', POINTER32(shadowmodel_t)),
        ('nsbtpAnim', POINTER32(anim_manager_t)),
        ('nsbtpFrame', u16),
        ('light', light_t),
        ('field144', VecFx32),
        ('field150', u16),
        ('field152', u16),
        ('params', POINTER32(traffic_params_t)),
        ('sfxEmitterExParams', POINTER32(sfx_emitter_ex_params_t)),
        ('field15C', c_int),
        ('field160', c_int),
    ]

class traffic_renderpart_t(Structure):
    _fields_ = [
        ('renderPart', mobj_render_part_t),
        ('playerIpatDir', VecFx32),
        ('updateIpatCulling', BOOL),
        ('performIpatCulling', BOOL),
    ]

class traffic_params_t(Structure):
    _fields_ = [
        ('field0', fx32),
        ('field4', fx32),
        ('field8', fx32),
        ('fieldC', fx32),
        ('field10', fx32),
        ('field14', fx32),
        ('field18', fx32),
        ('field1C', fx32),
        ('model', POINTER32(POINTER32(model_t))),
        ('shadowModel', POINTER32(POINTER32(shadowmodel_t))),
        ('nsbtpAnim', POINTER32(POINTER32(anim_manager_t))),
        ('field2C', u32),
        ('field30', u32),
        ('field34', c_void_p32),
    ]
