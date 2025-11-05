from ctypes import *
from private.mkds_python_bindings.testing.tick import *
from private.mkds_python_bindings.testing.types import *

DC_BG23_MODE_TEXT = 1
DC_BG23_MODE_AFFINE = 2
DC_BG23_MODE_AFFINE_EXT = 3
DC_BG23_MODE_256_BMP = 4
DC_BG23_MODE_DC_BMP = 5

DC_BG23_MODE_TEXT = 1
DC_BG23_MODE_AFFINE = 2
DC_BG23_MODE_AFFINE_EXT = 3
DC_BG23_MODE_256_BMP = 4
DC_BG23_MODE_DC_BMP = 5

DcBg23Mode = c_int
display_config_dword_21755A4_func_t = u32
display_config_vblank_func_t = u32


class dc_masterbright_t(Structure):
    _fields_ = [
        ('state', u16),
        ('fadeType', u16),
        ('curFrame', s16),
        ('frameCount', s16),
        ('brightness', s16),
        ('fieldA', u16),
    ]

class display_config_base_t(Structure):
    _fields_ = [
        ('vblankWaitCount', u16),
        ('mainVisiblePlane', u16),
        ('subVisiblePlane', u16),
        ('mainDisplayMode', u16),
        ('mainBgMode', u16),
        ('mainBg03d', u16),
        ('subBgMode', u16),
        ('mainBgBank', u16),
        ('mainObjBank', u16),
        ('mainBgExtPlttBank', u16),
        ('mainObjExtPlttBank', u16),
        ('texBank', u16),
        ('texPlttBank', u16),
        ('clearImgBank', u16),
        ('subBgBank', u16),
        ('subObjBank', u16),
        ('subBgExtPlttBank', u16),
        ('subObjExtPlttBank', u16),
        ('arm7Bank', u16),
        ('lcdcBank', u16),
    ]

class display_config_bgcommon_t(Structure):
    _fields_ = [
        ('priority', u16),
        ('mosaic', u16),
        ('screenSize', u16),
        ('colorMode', u16),
        ('screenBase', u16),
        ('characterBase', u16),
    ]

class display_config_bg01_t(Structure):
    _fields_ = [
        ('common', display_config_bgcommon_t),
        ('extPlttSlot', u16),
        ('unk', u16),
    ]

class display_config_bg23_t(Structure):
    _fields_ = [
        ('mode', DcBg23Mode),
        ('common', display_config_bgcommon_t),
    ]

class display_config_engine_t(Structure):
    _fields_ = [
        ('bg0Config', display_config_bg01_t),
        ('bg1Config', display_config_bg01_t),
        ('bg2Config', display_config_bg23_t),
        ('bg3Config', display_config_bg23_t),
        ('objVRamModeChar', u16),
        ('objVRamModeBmp', u16),
    ]

class display_config_3d_t(Structure):
    _fields_ = [
        ('clearColor', u16),
        ('sortMode', u8),
        ('bufferMode', u8),
    ]

class display_config_t(Structure):
    _fields_ = [
        ('baseConfig', display_config_base_t),
        ('mainConfig', display_config_engine_t),
        ('config3d', u32), #POINTER(display_config_3d_t)),
        ('subConfig', display_config_engine_t),
        ('fieldB4', u32),
        ('vblankFunc', display_config_vblank_func_t),
        ('frameStartTime', OSTick),
        ('vblankTime', OSTick),
        ('renderDuration', u32),
        ('lastTotalDuration', u32),
        ('lastRenderDuration', u32),
        ('flags', u8),
    ]
