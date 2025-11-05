from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.animationManager import *
from private.mkds_python_bindings.generated.animator import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.gxcommon import *
from private.mkds_python_bindings.generated.kartOffsetData import *
from private.mkds_python_bindings.generated.light import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.raceConfig import *
from private.mkds_python_bindings.generated.types import *


class charkart_field24_t(Structure):
    _fields_ = [
        ('field0', c_int),
        ('field4', c_int),
        ('field8', c_int),
        ('fieldC', c_int),
        ('field10', u16),
        ('field14', c_int),
        ('field18', u16),
        ('field1A', u16),
    ]

class charkart_colors_t(Structure):
    _fields_ = [
        ('diffuse', GXRgb),
        ('emission', GXRgb),
        ('ambient', GXRgb),
        ('diffR', u16),
        ('diffG', u16),
        ('diffB', u16),
        ('diffRDelta', s16),
        ('diffGDelta', s16),
        ('diffBDelta', s16),
        ('emiR', u16),
        ('emiG', u16),
        ('emiB', u16),
        ('emiRDelta', s16),
        ('emiGDelta', s16),
        ('emiBDelta', s16),
        ('ambiR', u16),
        ('ambiG', u16),
        ('amibB', u16),
        ('amibRDelta', s16),
        ('ambiGDelta', s16),
        ('ambiBDelta', s16),
        ('progress', fx16),
    ]

class charkart_t(Structure):
    _fields_ = [
        ('characterId', CharacterId),
        ('kartId', s32),
        ('characterNsbcaAnim', POINTER32(anim_manager_t)),
        ('characterNsbtpAnim', POINTER32(anim_manager_t)),
        ('characterModel', POINTER32(model_t)),
        ('kartModel', POINTER32(model_t)),
        ('kartTireModel', POINTER32(model_t)),
        ('kartShadowModel', POINTER32(model_t)),
        ('kartOffsetData', POINTER32(kofs_entry_t)),
        ('field24', charkart_field24_t),
        ('light', light_t),
        ('field54', u32),
        ('isKartInvisible', BOOL),
        ('isCharacterInvisible', BOOL),
        ('useSeparateTires', BOOL),
        ('inStarToonMode', BOOL),
        ('kartABC', u16),
        ('colors', charkart_colors_t),
        ('field98', anim_animator_t),
        ('nsbtpAnimDisabled', BOOL),
        ('fieldB0', u32),
    ]
