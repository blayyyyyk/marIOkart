from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.nkm import *
from private.mkds_python_bindings.testing.types import *

CAMERA_MODE_NORMAL = 0
CAMERA_MODE_DOUBLE_TOP = 1
CAMERA_MODE_DOUBLE_BOTTOM = 2

CAMERA_MODE_NORMAL = 0
CAMERA_MODE_DOUBLE_TOP = 1
CAMERA_MODE_DOUBLE_BOTTOM = 2

CAMERA_MODE_NORMAL = 0
CAMERA_MODE_DOUBLE_TOP = 1
CAMERA_MODE_DOUBLE_BOTTOM = 2

CAMERA_MODE_NORMAL = 0
CAMERA_MODE_DOUBLE_TOP = 1
CAMERA_MODE_DOUBLE_BOTTOM = 2

CAMERA_MODE_NORMAL = 0
CAMERA_MODE_DOUBLE_TOP = 1
CAMERA_MODE_DOUBLE_BOTTOM = 2

CAMERA_MODE_NORMAL = 0
CAMERA_MODE_DOUBLE_TOP = 1
CAMERA_MODE_DOUBLE_BOTTOM = 2

MKDSCameraMode = c_int
MKDSCameraMode = c_int
MKDSCameraMode = c_int


class cam_params_t(Structure):
    _fields_ = [
        ('distance', fx32),
        ('elevation', fx32),
        ('maxTargetElevation', fx32),
    ]

class came_routestat_t(Structure):
    _fields_ = [
        ('pointCache', (VecFx32 * 4)),
        ('progress', c_int),
        ('index', c_int),
        ('field38', c_int),
    ]

class came_unknown_t(Structure):
    _fields_ = [
        ('cameEntry', u32), #POINTER(nkm_came_entry_t)),
        ('routestat1', came_routestat_t),
        ('routestat2', came_routestat_t),
        ('routeSpeed', s16),
        ('field7E', u16),
        ('field80', u32),
        ('field84', u32),
    ]

class camera_t(Structure):
    _fields_ = [
        ('up', VecFx32),
        ('right', VecFx32),
        ('target', VecFx32),
        ('position', VecFx32),
        ('mtx', MtxFx43),
        ('fov', s32),
        ('targetFov', s32),
        ('fovSin', fx16),
        ('fovCos', fx16),
        ('aspectRatio', fx32),
        ('frustumNear', fx32),
        ('frustumFar', fx32),
        ('frustumTop', fx32),
        ('frustumBottom', fx32),
        ('frustumLeft', fx32),
        ('frustumRight', fx32),
        ('field88', fx32),
        ('skyFrustumFar', fx32),
        ('lookAtTarget', VecFx32),
        ('lookAtPosition', VecFx32),
        ('fieldA8', VecFx32),
        ('fieldB4', VecFx32),
        ('upConst', VecFx32),
        ('fieldCC', fx32),
        ('fieldD0', BOOL),
        ('targetElevation', fx32),
        ('fieldD8', u32),
        ('fieldDC', u32),
        ('fieldE0', u32),
        ('fieldE4', VecFx32),
        ('playerOffsetDirection', fx32),
        ('fieldF4', VecFx32),
        ('field100', VecFx32),
        ('field10C', VecFx32),
        ('field118', VecFx32),
        ('field124', VecFx32),
        ('field130', u8),
        ('prevPosition', VecFx32),
        ('isShaking', BOOL),
        ('field144', fx32),
        ('shakeAmount', fx32),
        ('field14C', u32),
        ('field150', s16),
        ('shakeDecay', fx32),
        ('field158', u32),
        ('targetDirection', VecFx32),
        ('field168', fx32),
        ('field16C', u32),
        ('field170', u32),
        ('field174', u32),
        ('elevation', fx32),
        ('field17C', VecFx32),
        ('field188', VecFx32),
        ('routeStat', came_routestat_t),
        ('routeStat2', came_routestat_t),
        ('field20C', u16),
        ('field20E', u16),
        ('targetDriverId', u16),
        ('currentCamId', u32),
        ('cameEntry', u32), #POINTER(nkm_came_entry_t)),
        ('unknownMgCams', u32), #POINTER(came_unknown_t)),
        ('unknownMgCamsCopy', u32), #POINTER(came_unknown_t)),
        ('field224', u16),
        ('field228', u32),
        ('field22C', u16),
        ('field230', u32),
        ('field234', u32),
        ('field238', BOOL),
        ('frameCounter', u16),
        ('fovProgress', fx32),
        ('targetProgress', fx32),
        ('field248', u32),
        ('mode', MKDSCameraMode),
        ('field250', u32),
        ('field254', u32),
        ('field258', BOOL),
        ('field25C', u32),
        ('field260', VecFx32),
        ('field26C', s16),
        ('field26E', u16),
    ]
