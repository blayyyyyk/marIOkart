from ctypes import *
from private.mkds_python_bindings.testing.animationManager import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.light import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.mapobjRenderPart import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.model2 import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.quaternion import *
from private.mkds_python_bindings.testing.sfx import *
from private.mkds_python_bindings.testing.types import *


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
        ('model', u32), #POINTER(POINTER(model_t))),
        ('shadowModel', u32), #POINTER(POINTER(shadowmodel_t))),
        ('nsbtpAnim', u32), #POINTER(POINTER(anim_manager_t))),
        ('field2C', u32),
        ('field30', u32),
        ('field34', u32),
    ]

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
        ('model', u32), #POINTER(model_t)),
        ('shadowModel', u32), #POINTER(shadowmodel_t)),
        ('nsbtpAnim', u32), #POINTER(anim_manager_t)),
        ('nsbtpFrame', u16),
        ('light', light_t),
        ('field144', VecFx32),
        ('field150', u16),
        ('field152', u16),
        ('params', u32), #POINTER(traffic_params_t)),
        ('sfxEmitterExParams', u32), #POINTER(sfx_emitter_ex_params_t)),
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
