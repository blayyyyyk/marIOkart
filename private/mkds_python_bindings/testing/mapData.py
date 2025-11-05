from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.nkm import *
from private.mkds_python_bindings.testing.types import *

mdat_clip_area_list_entry_t_p = u32
mdat_enemypoint_t_p = u32
mdat_mgenemypoint_t_p = u32
mdat_itempoint_t_p = u32

class mdat_clip_area_list_entry_t(Structure):
    _fields_ = [
        ('entry', u32), #POINTER(nkm_area_entry_t)),
        ('next', mdat_clip_area_list_entry_t_p),
    ]

class mdat_path_t(Structure):
    _fields_ = [
        ('path', u32), #POINTER(nkm_path_entry_t)),
        ('poit', u32), #POINTER(nkm_poit_entry_t)),
    ]

class mdat_enemypoint_t(Structure):
    _fields_ = [
        ('next', (mdat_enemypoint_t_p * 3)),
        ('previous', (mdat_enemypoint_t_p * 3)),
        ('position', u32), #POINTER(VecFx32)),
        ('radius', fx32),
        ('settings', u32), #POINTER(nkm_epoi_entry_settings_t)),
        ('nextCount', u16),
        ('previousCount', u16),
    ]

class mdat_enemypath_data_t(Structure):
    _fields_ = [
        ('points', u32), #POINTER(mdat_enemypoint_t)),
        ('firstPoint', u32), #POINTER(mdat_enemypoint_t)),
        ('lastPoint', u32), #POINTER(mdat_enemypoint_t)),
    ]

class mdat_mgenemypoint_t(Structure):
    _fields_ = [
        ('next', (mdat_mgenemypoint_t_p * 8)),
        ('position', u32), #POINTER(VecFx32)),
        ('radius', fx32),
        ('settings', u32), #POINTER(nkm_mepo_entry_settings_t)),
        ('nextCount', u16),
        ('nextIsNewPathMask', u8),
    ]

class mdat_mgenemypath_data_t(Structure):
    _fields_ = [
        ('points', u32), #POINTER(mdat_mgenemypoint_t)),
        ('firstPoint', u32), #POINTER(mdat_mgenemypoint_t)),
        ('lastPoint', u32), #POINTER(mdat_mgenemypoint_t)),
    ]

class mdat_itempoint_t(Structure):
    _fields_ = [
        ('next', (mdat_itempoint_t_p * 3)),
        ('previous', (mdat_itempoint_t_p * 3)),
        ('position', u32), #POINTER(VecFx32)),
        ('radius', fx32),
        ('recalcIdx', u8),
        ('dirX', s8),
        ('dirY', s8),
        ('dirZ', s8),
        ('nextCount', u16),
        ('previousCount', u16),
    ]

class mdat_itempath_data_t(Structure):
    _fields_ = [
        ('points', u32), #POINTER(mdat_itempoint_t)),
        ('firstPoint', u32), #POINTER(mdat_itempoint_t)),
        ('lastPoint', u32), #POINTER(mdat_itempoint_t)),
    ]

class mdat_mapdata_t(Structure):
    _fields_ = [
        ('obji', u32), #POINTER(nkm_obji_entry_t)),
        ('objiCount', u16),
        ('path', u32), #POINTER(nkm_path_entry_t)),
        ('pathCount', u16),
        ('poit', u32), #POINTER(nkm_poit_entry_t)),
        ('poitCount', u16),
        ('stag', u32), #POINTER(nkm_stag_data_t)),
        ('ktps', u32), #POINTER(nkm_ktps_entry_t)),
        ('ktpsCount', u16),
        ('ktpj', u32), #POINTER(nkm_ktpj_entry_t)),
        ('ktpjCount', u16),
        ('ktp2', u32), #POINTER(nkm_ktp2_entry_t)),
        ('ktp2Count', u16),
        ('ktpc', u32), #POINTER(nkm_ktpc_entry_t)),
        ('ktpcCount', u16),
        ('ktpm', u32), #POINTER(nkm_ktpm_entry_t)),
        ('ktpmCount', u16),
        ('cpoi', u32), #POINTER(nkm_cpoi_entry_t)),
        ('cpoiCount', u16),
        ('cpat', u32), #POINTER(nkm_cpat_entry_t)),
        ('cpatCount', u16),
        ('ipoi', nkm_ipoi_entry_pointer_t),
        ('ipoiCount', u16),
        ('ipat', u32), #POINTER(nkm_ipat_entry_t)),
        ('ipatCount', u16),
        ('epoi', u32), #POINTER(nkm_epoi_entry_t)),
        ('epoiCount', u16),
        ('epat', u32), #POINTER(nkm_epat_entry_t)),
        ('epatCount', u16),
        ('area', u32), #POINTER(nkm_area_entry_t)),
        ('areaCount', u16),
        ('came', u32), #POINTER(nkm_came_entry_t)),
        ('cameCount', u16),
        ('mepo', u32), #POINTER(nkm_mepo_entry_t)),
        ('mepoCount', u16),
        ('mepa', u32), #POINTER(nkm_mepa_entry_t)),
        ('mepaCount', u16),
        ('paths', u32), #POINTER(mdat_path_t)),
        ('cpoiKeyCount', u16),
        ('cpatLastCpoiIndex', u16),
        ('cpatMaxSectionOrder', u16),
        ('unknown49', u8),
        ('unknown50', u8),
        ('enemyPathData', mdat_enemypath_data_t),
        ('itemPathData', mdat_itempath_data_t),
        ('mgEnemyPathData', mdat_mgenemypath_data_t),
        ('cameIntroFirstTopCam', u32), #POINTER(nkm_came_entry_t)),
        ('cameIntroFirstBottomCam', u32), #POINTER(nkm_came_entry_t)),
        ('cameType6', u32), #POINTER(nkm_came_entry_t)),
        ('cameBattleIntroCam', u32), #POINTER(nkm_came_entry_t)),
        ('cameMissionFinishCam', u32), #POINTER(nkm_came_entry_t)),
        ('clipAreaLists', (POINTER(mdat_clip_area_list_entry_t) * 8)),
        ('ktpjIndexTable', u32), #POINTER(u16)),
        ('ktpcIndexTable', u32), #POINTER(u16)),
        ('curMgRespawnId', u16),
        ('enemyRespawnRouteAreaCount', u16),
        ('trackLength', fx32),
        ('trackLengthDiv15000', u32),
        ('nkmVersion', u16),
        ('unknown141', u8),
        ('missionEndAreaCount', u8),
    ]
