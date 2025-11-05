from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.nkm import *
from private.mkds_python_bindings.generated.types import *


class mdat_mapdata_t(Structure):
    _fields_ = [
        ('obji', POINTER32(nkm_obji_entry_t)),
        ('objiCount', u16),
        ('path', POINTER32(nkm_path_entry_t)),
        ('pathCount', u16),
        ('poit', POINTER32(nkm_poit_entry_t)),
        ('poitCount', u16),
        ('stag', POINTER32(nkm_stag_data_t)),
        ('ktps', POINTER32(nkm_ktps_entry_t)),
        ('ktpsCount', u16),
        ('ktpj', POINTER32(nkm_ktpj_entry_t)),
        ('ktpjCount', u16),
        ('ktp2', POINTER32(nkm_ktp2_entry_t)),
        ('ktp2Count', u16),
        ('ktpc', POINTER32(nkm_ktpc_entry_t)),
        ('ktpcCount', u16),
        ('ktpm', POINTER32(nkm_ktpm_entry_t)),
        ('ktpmCount', u16),
        ('cpoi', POINTER32(nkm_cpoi_entry_t)),
        ('cpoiCount', u16),
        ('cpat', POINTER32(nkm_cpat_entry_t)),
        ('cpatCount', u16),
        ('ipoi', nkm_ipoi_entry_pointer_t),
        ('ipoiCount', u16),
        ('ipat', POINTER32(nkm_ipat_entry_t)),
        ('ipatCount', u16),
        ('epoi', POINTER32(nkm_epoi_entry_t)),
        ('epoiCount', u16),
        ('epat', POINTER32(nkm_epat_entry_t)),
        ('epatCount', u16),
        ('area', POINTER32(nkm_area_entry_t)),
        ('areaCount', u16),
        ('came', POINTER32(nkm_came_entry_t)),
        ('cameCount', u16),
        ('mepo', POINTER32(nkm_mepo_entry_t)),
        ('mepoCount', u16),
        ('mepa', POINTER32(nkm_mepa_entry_t)),
        ('mepaCount', u16),
        ('paths', POINTER32(mdat_path_t)),
        ('cpoiKeyCount', u16),
        ('cpatLastCpoiIndex', u16),
        ('cpatMaxSectionOrder', u16),
        ('unknown49', u8),
        ('unknown50', u8),
        ('enemyPathData', mdat_enemypath_data_t),
        ('itemPathData', mdat_itempath_data_t),
        ('mgEnemyPathData', mdat_mgenemypath_data_t),
        ('cameIntroFirstTopCam', POINTER32(nkm_came_entry_t)),
        ('cameIntroFirstBottomCam', POINTER32(nkm_came_entry_t)),
        ('cameType6', POINTER32(nkm_came_entry_t)),
        ('cameBattleIntroCam', POINTER32(nkm_came_entry_t)),
        ('cameMissionFinishCam', POINTER32(nkm_came_entry_t)),
        ('clipAreaLists', (POINTER32(mdat_clip_area_list_entry_t) * 8)),
        ('ktpjIndexTable', POINTER32(u16)),
        ('ktpcIndexTable', POINTER32(u16)),
        ('curMgRespawnId', u16),
        ('enemyRespawnRouteAreaCount', u16),
        ('trackLength', fx32),
        ('trackLengthDiv15000', u32),
        ('nkmVersion', u16),
        ('unknown141', u8),
        ('missionEndAreaCount', u8),
    ]

class mdat_itempath_data_t(Structure):
    _fields_ = [
        ('points', POINTER32(mdat_itempoint_t)),
        ('firstPoint', POINTER32(mdat_itempoint_t)),
        ('lastPoint', POINTER32(mdat_itempoint_t)),
    ]

class mdat_mgenemypath_data_t(Structure):
    _fields_ = [
        ('points', POINTER32(mdat_mgenemypoint_t)),
        ('firstPoint', POINTER32(mdat_mgenemypoint_t)),
        ('lastPoint', POINTER32(mdat_mgenemypoint_t)),
    ]

class mdat_enemypath_data_t(Structure):
    _fields_ = [
        ('points', POINTER32(mdat_enemypoint_t)),
        ('firstPoint', POINTER32(mdat_enemypoint_t)),
        ('lastPoint', POINTER32(mdat_enemypoint_t)),
    ]

class mdat_clip_area_list_entry_t(Structure):
    _fields_ = [
        ('entry', POINTER32(nkm_area_entry_t)),
        ('next', POINTER32(mdat_clip_area_list_entry_t)),
    ]

class mdat_path_t(Structure):
    _fields_ = [
        ('path', POINTER32(nkm_path_entry_t)),
        ('poit', POINTER32(nkm_poit_entry_t)),
    ]

class mdat_enemypoint_t(Structure):
    _fields_ = [
        ('next', (POINTER32(mdat_enemypoint_t) * 3)),
        ('previous', (POINTER32(mdat_enemypoint_t) * 3)),
        ('position', POINTER32(VecFx32)),
        ('radius', fx32),
        ('settings', POINTER32(nkm_epoi_entry_settings_t)),
        ('nextCount', u16),
        ('previousCount', u16),
    ]

class mdat_mgenemypoint_t(Structure):
    _fields_ = [
        ('next', (POINTER32(mdat_mgenemypoint_t) * 8)),
        ('position', POINTER32(VecFx32)),
        ('radius', fx32),
        ('settings', POINTER32(nkm_mepo_entry_settings_t)),
        ('nextCount', u16),
        ('nextIsNewPathMask', u8),
    ]

class mdat_itempoint_t(Structure):
    _fields_ = [
        ('next', (POINTER32(mdat_itempoint_t) * 3)),
        ('previous', (POINTER32(mdat_itempoint_t) * 3)),
        ('position', POINTER32(VecFx32)),
        ('radius', fx32),
        ('recalcIdx', u8),
        ('dirX', s8),
        ('dirY', s8),
        ('dirZ', s8),
        ('nextCount', u16),
        ('previousCount', u16),
    ]
