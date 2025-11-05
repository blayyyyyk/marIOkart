from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
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


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_11_9(Structure):
    _fields_ = [
        ('state', c_int),
        ('fadeType', c_int),
        ('curFrame', c_int),
        ('frameCount', c_int),
        ('brightness', c_int),
        ('fieldA', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_21_9(Structure):
    _fields_ = [
        ('vblankWaitCount', c_int),
        ('mainVisiblePlane', c_int),
        ('subVisiblePlane', c_int),
        ('mainDisplayMode', c_int),
        ('mainBgMode', c_int),
        ('mainBg03d', c_int),
        ('subBgMode', c_int),
        ('mainBgBank', c_int),
        ('mainObjBank', c_int),
        ('mainBgExtPlttBank', c_int),
        ('mainObjExtPlttBank', c_int),
        ('texBank', c_int),
        ('texPlttBank', c_int),
        ('clearImgBank', c_int),
        ('subBgBank', c_int),
        ('subObjBank', c_int),
        ('subBgExtPlttBank', c_int),
        ('subObjExtPlttBank', c_int),
        ('arm7Bank', c_int),
        ('lcdcBank', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_45_9(Structure):
    _fields_ = [
        ('priority', c_int),
        ('mosaic', c_int),
        ('screenSize', c_int),
        ('colorMode', c_int),
        ('screenBase', c_int),
        ('characterBase', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_55_9(Structure):
    _fields_ = [
        ('common', display_config_bgcommon_t),
        ('extPlttSlot', c_int),
        ('unk', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_71_9(Structure):
    _fields_ = [
        ('mode', DcBg23Mode),
        ('common', display_config_bgcommon_t),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_89_9(Structure):
    _fields_ = [
        ('bg0Config', display_config_bg01_t),
        ('bg1Config', display_config_bg01_t),
        ('bg2Config', display_config_bg23_t),
        ('bg3Config', display_config_bg23_t),
        ('objVRamModeChar', c_int),
        ('objVRamModeBmp', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_101_9(Structure):
    _fields_ = [
        ('clearColor', c_int),
        ('sortMode', c_int),
        ('bufferMode', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_displayConfig_h_113_9(Structure):
    _fields_ = [
        ('baseConfig', display_config_base_t),
        ('mainConfig', display_config_engine_t),
        ('config3d', POINTER32(display_config_3d_t)),
        ('subConfig', display_config_engine_t),
        ('fieldB4', c_int),
        ('vblankFunc', display_config_vblank_func_t),
        ('frameStartTime', c_int),
        ('vblankTime', c_int),
        ('renderDuration', c_int),
        ('lastTotalDuration', c_int),
        ('lastRenderDuration', c_int),
        ('flags', c_int),
    ]
display_config_dword_21755A4_func_t = c_void_p32
DcBg23Mode = c_int
display_config_vblank_func_t = c_void_p32
