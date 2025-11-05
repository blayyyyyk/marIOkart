from ctypes import *
from private.mkds_python_bindings.testing.animationManager import *
from private.mkds_python_bindings.testing.animator import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.gxcommon import *
from private.mkds_python_bindings.testing.kartOffsetData import *
from private.mkds_python_bindings.testing.light import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.raceConfig import *
from private.mkds_python_bindings.testing.types import *


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
        ('characterNsbcaAnim', u32), #POINTER(anim_manager_t)),
        ('characterNsbtpAnim', u32), #POINTER(anim_manager_t)),
        ('characterModel', u32), #POINTER(model_t)),
        ('kartModel', u32), #POINTER(model_t)),
        ('kartTireModel', u32), #POINTER(model_t)),
        ('kartShadowModel', u32), #POINTER(model_t)),
        ('kartOffsetData', u32), #POINTER(kofs_entry_t)),
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
