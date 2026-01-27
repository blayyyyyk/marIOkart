from ctypes import *
from ctypes import _Pointer, _SimpleCData, _CFunctionType
from typing import ClassVar, Any, List, Callable, TypeVar, Union as UnionT
from typing import TYPE_CHECKING
import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
T = TypeVar('T', bound=UnionT[Structure, Union, _SimpleCData, _Pointer])
POINTER_T = _Pointer[T]

# Forward declarations

class MATHRandContext32(Structure):
    """
    ```python
    x: int (POINTER(L))
    mul: int (POINTER(L))
    add: int (POINTER(L))
    ```
    """
    _pack_: ClassVar[int] = 1
    x: int
    mul: int
    add: int

class MtxFx33(Union):
    """
    ```python
    _0: struct_MtxFx33_0 (struct_MtxFx33_0)
    m: Float[Tensor, "3 3"] (struct_fx32[3][3])
    a: Float[Tensor, "9"] (struct_fx32[9])
    ```
    """
    _pack_: ClassVar[int] = 1
    _0: struct_MtxFx33_0
    m: Float[Tensor, "3 3"]
    a: Float[Tensor, "9"]

class MtxFx43(Union):
    """
    ```python
    _0: struct_MtxFx43_0 (struct_MtxFx43_0)
    m: Float[Tensor, "4 3"] (struct_fx32[3][4])
    a: Float[Tensor, "12"] (struct_fx32[12])
    ```
    """
    _pack_: ClassVar[int] = 1
    _0: struct_MtxFx43_0
    m: Float[Tensor, "4 3"]
    a: Float[Tensor, "12"]

class NNSFndLink(Structure):
    """
    ```python
    prevObject: c_void_p (c_void_p)
    nextObject: c_void_p (c_void_p)
    ```
    """
    _pack_: ClassVar[int] = 1
    prevObject: c_void_p
    nextObject: c_void_p

class NNSFndList(Structure):
    """
    ```python
    headObject: c_void_p (c_void_p)
    tailObject: c_void_p (c_void_p)
    numObjects: int (POINTER(H))
    offset: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    headObject: c_void_p
    tailObject: c_void_p
    numObjects: int
    offset: int

class NNSG3dJntAnmResult(Structure):
    """
    ```python
    flag: int (POINTER(I))
    scale: Float[Tensor, "3"] (struct_VecFx32)
    scaleEx0: Float[Tensor, "3"] (struct_VecFx32)
    scaleEx1: Float[Tensor, "3"] (struct_VecFx32)
    rot: Float[Tensor, "3 3"] (union_MtxFx33)
    trans: Float[Tensor, "3"] (struct_VecFx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    flag: int
    scale: Float[Tensor, "3"]
    scaleEx0: Float[Tensor, "3"]
    scaleEx1: Float[Tensor, "3"]
    rot: Float[Tensor, "3 3"]
    trans: Float[Tensor, "3"]

class NNSG3dMatAnmResult(Structure):
    """
    ```python
    flag: int (POINTER(I))
    prmMatColor0: int (POINTER(I))
    prmMatColor1: int (POINTER(I))
    prmPolygonAttr: int (POINTER(I))
    prmTexImage: int (POINTER(I))
    prmTexPltt: int (POINTER(I))
    scaleS: float (struct_fx32)
    scaleT: float (struct_fx32)
    sinR: float (struct_fx16)
    cosR: float (struct_fx16)
    transS: float (struct_fx32)
    transT: float (struct_fx32)
    origWidth: int (POINTER(H))
    origHeight: int (POINTER(H))
    magW: float (struct_fx32)
    magH: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    flag: int
    prmMatColor0: int
    prmMatColor1: int
    prmPolygonAttr: int
    prmTexImage: int
    prmTexPltt: int
    scaleS: float
    scaleT: float
    sinR: float
    cosR: float
    transS: float
    transT: float
    origWidth: int
    origHeight: int
    magW: float
    magH: float

class NNSG3dRenderObj(Structure):
    """
    ```python
    flag: int (POINTER(I))
    resMdl: POINTER_T[struct_NNSG3dResMdl_] (POINTER(struct_NNSG3dResMdl_))
    anmMat: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    funcBlendMat: Callable[[POINTER_T[struct_NNSG3dMatAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int] (CFunctionType)
    anmJnt: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    funcBlendJnt: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int] (CFunctionType)
    anmVis: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    funcBlendVis: Callable[[POINTER_T[struct_NNSG3dVisAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int] (CFunctionType)
    cbFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None] (CFunctionType)
    cbCmd: int (POINTER(B))
    cbTiming: int (POINTER(B))
    dummy_: int (POINTER(H))
    cbInitFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None] (CFunctionType)
    ptrUser: c_void_p (c_void_p)
    ptrUserSbc: POINTER_T[c_ubyte] (POINTER(POINTER(B)))
    recJntAnm: POINTER_T[struct_NNSG3dJntAnmResult_] (POINTER(struct_NNSG3dJntAnmResult_))
    recMatAnm: POINTER_T[struct_NNSG3dMatAnmResult_] (POINTER(struct_NNSG3dMatAnmResult_))
    hintMatAnmExist: list[int] (POINTER(I)[2])
    hintJntAnmExist: list[int] (POINTER(I)[2])
    hintVisAnmExist: list[int] (POINTER(I)[2])
    ```
    """
    _pack_: ClassVar[int] = 1
    flag: int
    resMdl: POINTER_T[struct_NNSG3dResMdl_]
    anmMat: POINTER_T[struct_NNSG3dAnmObj_]
    funcBlendMat: Callable[[POINTER_T[struct_NNSG3dMatAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int]
    anmJnt: POINTER_T[struct_NNSG3dAnmObj_]
    funcBlendJnt: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int]
    anmVis: POINTER_T[struct_NNSG3dAnmObj_]
    funcBlendVis: Callable[[POINTER_T[struct_NNSG3dVisAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int]
    cbFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None]
    cbCmd: int
    cbTiming: int
    dummy_: int
    cbInitFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None]
    ptrUser: c_void_p
    ptrUserSbc: POINTER_T[c_ubyte]
    recJntAnm: POINTER_T[struct_NNSG3dJntAnmResult_]
    recMatAnm: POINTER_T[struct_NNSG3dMatAnmResult_]
    hintMatAnmExist: list[int]
    hintJntAnmExist: list[int]
    hintVisAnmExist: list[int]

class NNSG3dResDataBlockHeader(Structure):
    """
    ```python
    _0: union_NNSG3dResDataBlockHeader__0 (union_NNSG3dResDataBlockHeader__0)
    size: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    _0: union_NNSG3dResDataBlockHeader__0
    size: int

class NNSG3dResDict(Structure):
    """
    ```python
    revision: int (POINTER(B))
    numEntry: int (POINTER(B))
    sizeDictBlk: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsEntry: int (POINTER(H))
    node: list[struct_NNSG3dResDictTreeNode_] (struct_NNSG3dResDictTreeNode_[1])
    ```
    """
    _pack_: ClassVar[int] = 1
    revision: int
    numEntry: int
    sizeDictBlk: int
    dummy_: int
    ofsEntry: int
    node: list[struct_NNSG3dResDictTreeNode_]

class NNSG3dResMdlInfo(Structure):
    """
    ```python
    sbcType: int (POINTER(B))
    scalingRule: int (POINTER(B))
    texMtxMode: int (POINTER(B))
    numNode: int (POINTER(B))
    numMat: int (POINTER(B))
    numShp: int (POINTER(B))
    firstUnusedMtxStackID: int (POINTER(B))
    dummy_: int (POINTER(B))
    posScale: float (struct_fx32)
    invPosScale: float (struct_fx32)
    numVertex: int (POINTER(H))
    numPolygon: int (POINTER(H))
    numTriangle: int (POINTER(H))
    numQuad: int (POINTER(H))
    boxX: float (struct_fx16)
    boxY: float (struct_fx16)
    boxZ: float (struct_fx16)
    boxW: float (struct_fx16)
    boxH: float (struct_fx16)
    boxD: float (struct_fx16)
    boxPosScale: float (struct_fx32)
    boxInvPosScale: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    sbcType: int
    scalingRule: int
    texMtxMode: int
    numNode: int
    numMat: int
    numShp: int
    firstUnusedMtxStackID: int
    dummy_: int
    posScale: float
    invPosScale: float
    numVertex: int
    numPolygon: int
    numTriangle: int
    numQuad: int
    boxX: float
    boxY: float
    boxZ: float
    boxW: float
    boxH: float
    boxD: float
    boxPosScale: float
    boxInvPosScale: float

class NNSG3dResNodeInfo(Structure):
    """
    ```python
    dict: struct_NNSG3dResDict_ (struct_NNSG3dResDict_)
    ```
    """
    _pack_: ClassVar[int] = 1
    dict: struct_NNSG3dResDict_

class NNSG3dResPlttInfo(Structure):
    """
    ```python
    vramKey: int (POINTER(I))
    sizePltt: int (POINTER(H))
    flag: int (POINTER(H))
    ofsDict: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsPlttData: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    vramKey: int
    sizePltt: int
    flag: int
    ofsDict: int
    dummy_: int
    ofsPlttData: int

class NNSG3dResTex4x4Info(Structure):
    """
    ```python
    vramKey: int (POINTER(I))
    sizeTex: int (POINTER(H))
    ofsDict: int (POINTER(H))
    flag: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsTex: int (POINTER(I))
    ofsTexPlttIdx: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    vramKey: int
    sizeTex: int
    ofsDict: int
    flag: int
    dummy_: int
    ofsTex: int
    ofsTexPlttIdx: int

class NNSG3dResTexInfo(Structure):
    """
    ```python
    vramKey: int (POINTER(I))
    sizeTex: int (POINTER(H))
    ofsDict: int (POINTER(H))
    flag: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsTex: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    vramKey: int
    sizeTex: int
    ofsDict: int
    flag: int
    dummy_: int
    ofsTex: int

class NNSG3dVisAnmResult(Structure):
    """
    ```python
    isVisible: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    isVisible: int

class VecFx16(Structure):
    """
    ```python
    x: float (struct_fx16)
    y: float (struct_fx16)
    z: float (struct_fx16)
    ```
    """
    _pack_: ClassVar[int] = 1
    x: float
    y: float
    z: float

class VecFx32(Structure):
    """
    ```python
    x: float (struct_fx32)
    y: float (struct_fx32)
    z: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    x: float
    y: float
    z: float

class anim_animator_t(Structure):
    """
    ```python
    loopMode: int (POINTER(H))
    hasEnded: int (POINTER(H))
    animLength: float (struct_fx32)
    speed: float (struct_fx32)
    progress: float (struct_fx32)
    loopIteration: int (POINTER(H))
    loopCount: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    loopMode: int
    hasEnded: int
    animLength: float
    speed: float
    progress: float
    loopIteration: int
    loopCount: int

class came_routestat_t(Structure):
    """
    ```python
    pointCache: list[Float[Tensor, "3"]] (struct_VecFx32[4])
    progress: int (POINTER(i))
    index: int (POINTER(i))
    field38: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    pointCache: list[Float[Tensor, "3"]]
    progress: int
    index: int
    field38: int

class camera_t(Structure):
    """
    ```python
    up: Float[Tensor, "3"] (struct_VecFx32)
    right: Float[Tensor, "3"] (struct_VecFx32)
    target: Float[Tensor, "3"] (struct_VecFx32)
    position: Float[Tensor, "3"] (struct_VecFx32)
    mtx: Float[Tensor, "4 3"] (union_MtxFx43)
    fov: int (POINTER(i))
    targetFov: int (POINTER(i))
    fovSin: float (struct_fx16)
    fovCos: float (struct_fx16)
    aspectRatio: float (struct_fx32)
    frustumNear: float (struct_fx32)
    frustumFar: float (struct_fx32)
    frustumTop: float (struct_fx32)
    frustumBottom: float (struct_fx32)
    frustumLeft: float (struct_fx32)
    frustumRight: float (struct_fx32)
    field88: float (struct_fx32)
    skyFrustumFar: float (struct_fx32)
    lookAtTarget: Float[Tensor, "3"] (struct_VecFx32)
    lookAtPosition: Float[Tensor, "3"] (struct_VecFx32)
    fieldA8: Float[Tensor, "3"] (struct_VecFx32)
    fieldB4: Float[Tensor, "3"] (struct_VecFx32)
    upConst: Float[Tensor, "3"] (struct_VecFx32)
    fieldCC: float (struct_fx32)
    fieldD0: int (POINTER(i))
    targetElevation: float (struct_fx32)
    fieldD8: int (POINTER(I))
    fieldDC: int (POINTER(I))
    fieldE0: int (POINTER(I))
    fieldE4: Float[Tensor, "3"] (struct_VecFx32)
    playerOffsetDirection: float (struct_fx32)
    fieldF4: Float[Tensor, "3"] (struct_VecFx32)
    field100: Float[Tensor, "3"] (struct_VecFx32)
    field10C: Float[Tensor, "3"] (struct_VecFx32)
    field118: Float[Tensor, "3"] (struct_VecFx32)
    field124: Float[Tensor, "3"] (struct_VecFx32)
    field130: int (POINTER(B))
    PADDING_0: list[int] (POINTER(B)[3])
    prevPosition: Float[Tensor, "3"] (struct_VecFx32)
    isShaking: int (POINTER(i))
    field144: float (struct_fx32)
    shakeAmount: float (struct_fx32)
    field14C: int (POINTER(I))
    field150: int (POINTER(h))
    PADDING_1: list[int] (POINTER(B)[2])
    shakeDecay: float (struct_fx32)
    field158: int (POINTER(I))
    targetDirection: Float[Tensor, "3"] (struct_VecFx32)
    field168: float (struct_fx32)
    field16C: int (POINTER(I))
    field170: int (POINTER(I))
    field174: int (POINTER(I))
    elevation: float (struct_fx32)
    field17C: Float[Tensor, "3"] (struct_VecFx32)
    field188: Float[Tensor, "3"] (struct_VecFx32)
    routeStat: struct_came_routestat_t (struct_came_routestat_t)
    routeStat2: struct_came_routestat_t (struct_came_routestat_t)
    field20C: int (POINTER(H))
    field20E: int (POINTER(H))
    targetDriverId: int (POINTER(H))
    PADDING_2: list[int] (POINTER(B)[2])
    currentCamId: int (POINTER(I))
    cameEntry: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    unknownMgCams: POINTER_T[struct_came_unknown_t] (POINTER(struct_came_unknown_t))
    unknownMgCamsCopy: POINTER_T[struct_came_unknown_t] (POINTER(struct_came_unknown_t))
    field224: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    field228: int (POINTER(I))
    field22C: int (POINTER(H))
    PADDING_4: list[int] (POINTER(B)[2])
    field230: int (POINTER(I))
    field234: int (POINTER(I))
    field238: int (POINTER(i))
    frameCounter: int (POINTER(H))
    PADDING_5: list[int] (POINTER(B)[2])
    fovProgress: float (struct_fx32)
    targetProgress: float (struct_fx32)
    field248: int (POINTER(I))
    mode: int (POINTER(I))
    field250: int (POINTER(I))
    field254: int (POINTER(I))
    field258: int (POINTER(i))
    field25C: int (POINTER(I))
    field260: Float[Tensor, "3"] (struct_VecFx32)
    field26C: int (POINTER(h))
    field26E: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    up: Float[Tensor, "3"]
    right: Float[Tensor, "3"]
    target: Float[Tensor, "3"]
    position: Float[Tensor, "3"]
    mtx: Float[Tensor, "4 3"]
    fov: int
    targetFov: int
    fovSin: float
    fovCos: float
    aspectRatio: float
    frustumNear: float
    frustumFar: float
    frustumTop: float
    frustumBottom: float
    frustumLeft: float
    frustumRight: float
    field88: float
    skyFrustumFar: float
    lookAtTarget: Float[Tensor, "3"]
    lookAtPosition: Float[Tensor, "3"]
    fieldA8: Float[Tensor, "3"]
    fieldB4: Float[Tensor, "3"]
    upConst: Float[Tensor, "3"]
    fieldCC: float
    fieldD0: int
    targetElevation: float
    fieldD8: int
    fieldDC: int
    fieldE0: int
    fieldE4: Float[Tensor, "3"]
    playerOffsetDirection: float
    fieldF4: Float[Tensor, "3"]
    field100: Float[Tensor, "3"]
    field10C: Float[Tensor, "3"]
    field118: Float[Tensor, "3"]
    field124: Float[Tensor, "3"]
    field130: int
    PADDING_0: list[int]
    prevPosition: Float[Tensor, "3"]
    isShaking: int
    field144: float
    shakeAmount: float
    field14C: int
    field150: int
    PADDING_1: list[int]
    shakeDecay: float
    field158: int
    targetDirection: Float[Tensor, "3"]
    field168: float
    field16C: int
    field170: int
    field174: int
    elevation: float
    field17C: Float[Tensor, "3"]
    field188: Float[Tensor, "3"]
    routeStat: struct_came_routestat_t
    routeStat2: struct_came_routestat_t
    field20C: int
    field20E: int
    targetDriverId: int
    PADDING_2: list[int]
    currentCamId: int
    cameEntry: POINTER_T[struct_nkm_came_entry_t]
    unknownMgCams: POINTER_T[struct_came_unknown_t]
    unknownMgCamsCopy: POINTER_T[struct_came_unknown_t]
    field224: int
    PADDING_3: list[int]
    field228: int
    field22C: int
    PADDING_4: list[int]
    field230: int
    field234: int
    field238: int
    frameCounter: int
    PADDING_5: list[int]
    fovProgress: float
    targetProgress: float
    field248: int
    mode: int
    field250: int
    field254: int
    field258: int
    field25C: int
    field260: Float[Tensor, "3"]
    field26C: int
    field26E: int

class charkart_colors_t(Structure):
    """
    ```python
    diffuse: int (POINTER(H))
    emission: int (POINTER(H))
    ambient: int (POINTER(H))
    diffR: int (POINTER(H))
    diffG: int (POINTER(H))
    diffB: int (POINTER(H))
    diffRDelta: int (POINTER(h))
    diffGDelta: int (POINTER(h))
    diffBDelta: int (POINTER(h))
    emiR: int (POINTER(H))
    emiG: int (POINTER(H))
    emiB: int (POINTER(H))
    emiRDelta: int (POINTER(h))
    emiGDelta: int (POINTER(h))
    emiBDelta: int (POINTER(h))
    ambiR: int (POINTER(H))
    ambiG: int (POINTER(H))
    amibB: int (POINTER(H))
    amibRDelta: int (POINTER(h))
    ambiGDelta: int (POINTER(h))
    ambiBDelta: int (POINTER(h))
    progress: float (struct_fx16)
    ```
    """
    _pack_: ClassVar[int] = 1
    diffuse: int
    emission: int
    ambient: int
    diffR: int
    diffG: int
    diffB: int
    diffRDelta: int
    diffGDelta: int
    diffBDelta: int
    emiR: int
    emiG: int
    emiB: int
    emiRDelta: int
    emiGDelta: int
    emiBDelta: int
    ambiR: int
    ambiG: int
    amibB: int
    amibRDelta: int
    ambiGDelta: int
    ambiBDelta: int
    progress: float

class charkart_field24_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(i))
    field8: int (POINTER(i))
    fieldC: int (POINTER(i))
    field10: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    field14: int (POINTER(i))
    field18: int (POINTER(H))
    field1A: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    field8: int
    fieldC: int
    field10: int
    PADDING_0: list[int]
    field14: int
    field18: int
    field1A: int

class driver_field450_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(B))
    PADDING_0: list[int] (POINTER(B)[3])
    field8: float (struct_fx32)
    fieldC: int (POINTER(i))
    field10: int (POINTER(i))
    field14: int (POINTER(i))
    field18: float (struct_fx32)
    field1C: float (struct_fx32)
    field20: float (struct_fx32)
    field24: float (struct_fx32)
    prevLapProgress: int (POINTER(i))
    kaidanSfxAlternateCounter: int (POINTER(B))
    PADDING_1: list[int] (POINTER(B)[3])
    field30: int (POINTER(i))
    field34: int (POINTER(i))
    sfxId: int (POINTER(i))
    computePitchOffsetFunc: Callable[[POINTER_T[struct_sfx_emitter_ex_params_t]], c_int] (CFunctionType)
    field40: struct_struc_334 (struct_struc_334)
    field68: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    PADDING_0: list[int]
    field8: float
    fieldC: int
    field10: int
    field14: int
    field18: float
    field1C: float
    field20: float
    field24: float
    prevLapProgress: int
    kaidanSfxAlternateCounter: int
    PADDING_1: list[int]
    field30: int
    field34: int
    sfxId: int
    computePitchOffsetFunc: Callable[[POINTER_T[struct_sfx_emitter_ex_params_t]], c_int]
    field40: struct_struc_334
    field68: int

class driver_t(Structure):
    """
    ```python
    soundEmitter: struct_sfx_emitter_t (struct_sfx_emitter_t)
    field44: int (POINTER(I))
    flags: int (POINTER(I))
    flags2: int (POINTER(I))
    direction: Float[Tensor, "3"] (struct_VecFx32)
    drivingDirection: Float[Tensor, "3"] (struct_VecFx32)
    velocity: Float[Tensor, "3"] (struct_VecFx32)
    id: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    inputId: int (POINTER(I))
    field7C: int (POINTER(I))
    position: Float[Tensor, "3"] (struct_VecFx32)
    lastPosition: Float[Tensor, "3"] (struct_VecFx32)
    kartTiresPosition: Float[Tensor, "3"] (struct_VecFx32)
    deltaPos: Float[Tensor, "3"] (struct_VecFx32)
    deltaPosNormalized: Float[Tensor, "3"] (struct_VecFx32)
    scale: Float[Tensor, "3"] (struct_VecFx32)
    fieldC8: float (struct_fx32)
    targetMaxSpeed: float (struct_fx32)
    maxSpeed: float (struct_fx32)
    fieldD4: int (POINTER(I))
    slipstreamSpeedMultiplier: float (struct_fx32)
    speedMultiplier: float (struct_fx32)
    rotation: Float[Tensor, "4"] (struct_quaternion_t)
    fieldF0: Float[Tensor, "4"] (struct_quaternion_t)
    field100: Float[Tensor, "4"] (struct_quaternion_t)
    field110: Float[Tensor, "4"] (struct_quaternion_t)
    mainMtx: Float[Tensor, "4 3"] (union_MtxFx43)
    field150: Float[Tensor, "4 3"] (union_MtxFx43)
    colReaction: int (POINTER(I))
    field184: Float[Tensor, "4 3"] (union_MtxFx43)
    charKartMtx: int (POINTER(I))
    colPos: Float[Tensor, "3"] (struct_VecFx32)
    prevColPos: Float[Tensor, "3"] (struct_VecFx32)
    colSphereSize: float (struct_fx32)
    colSphereZOffset: float (struct_fx32)
    netColPos: Float[Tensor, "3"] (struct_VecFx32)
    lastNetColPos: Float[Tensor, "3"] (struct_VecFx32)
    colPos2: Float[Tensor, "3"] (struct_VecFx32)
    field1FC: Float[Tensor, "3"] (struct_VecFx32)
    field208: POINTER_T[c_uint] (POINTER(POINTER(I)))
    field20C: list[c_void_p] (c_void_p[9])
    field230: Callable[[POINTER_T[struct_driver_t]], None] (CFunctionType)
    xRot: int (POINTER(h))
    yRot: int (POINTER(H))
    boostTimer: int (POINTER(h))
    field23A: int (POINTER(h))
    driftBoostCounter: int (POINTER(h))
    PADDING_1: list[int] (POINTER(B)[2])
    velocityMinusDirMultiplier: float (struct_fx32)
    upDir: Float[Tensor, "3"] (struct_VecFx32)
    field250: Float[Tensor, "3"] (struct_VecFx32)
    velocityY: Float[Tensor, "3"] (struct_VecFx32)
    fallsWaterForward: Float[Tensor, "3"] (struct_VecFx32)
    fallsWaterStrength: float (struct_fx32)
    forwardDir: Float[Tensor, "3"] (struct_VecFx32)
    jumpDriftUp: Float[Tensor, "3"] (struct_VecFx32)
    jumpDriftForward: Float[Tensor, "3"] (struct_VecFx32)
    collisionMode: int (POINTER(I))
    maxSpeedFraction: float (struct_fx32)
    deltaPosMag: float (struct_fx32)
    speed: float (struct_fx32)
    field2AC: float (struct_fx32)
    driverHitCheckMask: int (POINTER(H))
    driverHitMask: int (POINTER(H))
    lastDriverHitMask: int (POINTER(H))
    gap2B6: list[int] (POINTER(B)[2])
    field2B8: int (POINTER(i))
    field2BC: int (POINTER(i))
    field2C0: int (POINTER(H))
    PADDING_2: list[int] (POINTER(B)[2])
    leftRightDir: float (struct_fx32)
    colEntryId1: int (POINTER(h))
    colEntryId2: int (POINTER(h))
    kartPhysicalParams: POINTER_T[struct_physp_kart_params_t] (POINTER(struct_physp_kart_params_t))
    charPhysicalParams: POINTER_T[struct_physp_char_params_t] (POINTER(struct_physp_char_params_t))
    turningAmount: float (struct_fx32)
    field2D8: Float[Tensor, "3"] (struct_VecFx32)
    field2E4: Float[Tensor, "3"] (struct_VecFx32)
    field2F0: Float[Tensor, "3"] (struct_VecFx32)
    driftLeftRightCount: int (POINTER(i))
    driftLeftCount: int (POINTER(H))
    driftRightCount: int (POINTER(H))
    driftDir1CountPtr: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    driftDir2CountPtr: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    driftLeftRightTimeout: int (POINTER(i))
    enemyState: POINTER_T[struct_enemy_t] (POINTER(struct_enemy_t))
    field314: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    field318: float (struct_fx32)
    field31C: Float[Tensor, "3"] (struct_VecFx32)
    field328: Float[Tensor, "3"] (struct_VecFx32)
    field334: int (POINTER(I))
    field338: int (POINTER(H))
    PADDING_4: list[int] (POINTER(B)[2])
    field33C: int (POINTER(I))
    field340: float (struct_fx32)
    field344: int (POINTER(I))
    field348: int (POINTER(I))
    field34C: Float[Tensor, "4"] (struct_quaternion_t)
    colReactionCounter: int (POINTER(h))
    PADDING_5: list[int] (POINTER(B)[2])
    field360: float (struct_fx32)
    spinOutAngle: int (POINTER(H))
    spinOutSpinCount: int (POINTER(H))
    spinOutProgress: float (struct_fx32)
    spinOutVelocity: int (POINTER(I))
    field370: int (POINTER(H))
    PADDING_6: list[int] (POINTER(B)[2])
    field374: Float[Tensor, "3"] (struct_VecFx32)
    field380: int (POINTER(I))
    ghostFlickerPhase: int (POINTER(H))
    wallRotYSpeed: int (POINTER(h))
    driftRotY: int (POINTER(h))
    extraDrift: float (struct_fx16)
    field38C: float (struct_fx32)
    gap390: list[int] (POINTER(B)[4])
    field394: int (POINTER(I))
    field398: float (struct_fx32)
    field39C: float (struct_fx32)
    field3A0: float (struct_fx32)
    tireRotX: int (POINTER(H))
    PADDING_7: list[int] (POINTER(B)[2])
    field3A8: int (POINTER(i))
    respawnCounter: int (POINTER(H))
    PADDING_8: list[int] (POINTER(B)[2])
    field3B0: Float[Tensor, "3"] (struct_VecFx32)
    field3BC: int (POINTER(H))
    field3BE: int (POINTER(h))
    preRespawnCounter: int (POINTER(h))
    PADDING_9: list[int] (POINTER(B)[2])
    respawnId: int (POINTER(I))
    killTimer: int (POINTER(h))
    PADDING_10: list[int] (POINTER(B)[2])
    driverVoiceIdx: int (POINTER(I))
    kartABC: int (POINTER(h))
    field3D2: int (POINTER(h))
    field3D4: int (POINTER(h))
    PADDING_11: list[int] (POINTER(B)[2])
    place: int (POINTER(i))
    floorDriverColType: int (POINTER(I))
    floorColType: int (POINTER(I))
    floorColVariant: int (POINTER(i))
    field3E8: int (POINTER(h))
    PADDING_12: list[int] (POINTER(B)[2])
    yRotSpeedTarget: int (POINTER(I))
    yRotSpeed: int (POINTER(I))
    field3F4: float (struct_fx32)
    field3F8: float (struct_fx32)
    field3FC: int (POINTER(H))
    field3FE: int (POINTER(H))
    field400: int (POINTER(H))
    PADDING_13: list[int] (POINTER(B)[2])
    field404: float (struct_fx32)
    field408: int (POINTER(I))
    respawnStartFrame: int (POINTER(I))
    respawnAPressFrame: int (POINTER(I))
    field414: float (struct_fx32)
    field418: float (struct_fx32)
    growBackScale: Float[Tensor, "3"] (struct_VecFx32)
    thunderScale: Float[Tensor, "3"] (struct_VecFx32)
    dossunYScale: float (struct_fx32)
    mobjHitList: list[POINTER_T[struct_mobj_inst_t]] (POINTER(struct_mobj_inst_t)[2])
    mobjHitSfxTimeout: list[int] (POINTER(H)[2])
    mobjHitEmittedSfx: list[int] (POINTER(i)[2])
    smashDossun: POINTER_T[struct_mobj_inst_t] (POINTER(struct_mobj_inst_t))
    field450: struct_driver_field450_t (struct_driver_field450_t)
    field4BC: float (struct_fx32)
    colFlagsMap2DShadow: int (POINTER(I))
    jumpPadSpeed: int (POINTER(I))
    field4C8: float (struct_fx32)
    field4CC: int (POINTER(I))
    field4D0: int (POINTER(I))
    preStartEnginePower: float (struct_fx32)
    fallsWaterDstId: int (POINTER(h))
    wallTouchTimeout: int (POINTER(h))
    floorTouchTimeout: int (POINTER(h))
    field4DE: int (POINTER(h))
    field4E0: int (POINTER(h))
    field4E2: int (POINTER(h))
    field4E4: int (POINTER(H))
    field4E6: int (POINTER(H))
    field4E8: float (struct_fx32)
    field4EC: float (struct_fx32)
    idkScale: Float[Tensor, "3"] (struct_VecFx32)
    field4FC: int (POINTER(H))
    PADDING_14: list[int] (POINTER(B)[2])
    waterDepth: float (struct_fx32)
    field504: int (POINTER(H))
    field506: int (POINTER(H))
    field508: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    field50C: POINTER_T[struct_quaternion_t] (POINTER(struct_quaternion_t))
    field510: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    netState: POINTER_T[struct_driver_net_state_t] (POINTER(struct_driver_net_state_t))
    field518: struct_sfx_emitter_ex_params_t (struct_sfx_emitter_ex_params_t)
    field534: c_void_p (c_void_p)
    timers: struct_driver_timers_t (struct_driver_timers_t)
    charKart: POINTER_T[struct_charkart_t] (POINTER(struct_charkart_t))
    field594: float (struct_fx32)
    field598: int (POINTER(h))
    PADDING_15: list[int] (POINTER(B)[2])
    field59C: int (POINTER(I))
    field5A0: int (POINTER(H))
    gap5A2: list[int] (POINTER(B)[2])
    field5A4: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    soundEmitter: struct_sfx_emitter_t
    field44: int
    flags: int
    flags2: int
    direction: Float[Tensor, "3"]
    drivingDirection: Float[Tensor, "3"]
    velocity: Float[Tensor, "3"]
    id: int
    PADDING_0: list[int]
    inputId: int
    field7C: int
    position: Float[Tensor, "3"]
    lastPosition: Float[Tensor, "3"]
    kartTiresPosition: Float[Tensor, "3"]
    deltaPos: Float[Tensor, "3"]
    deltaPosNormalized: Float[Tensor, "3"]
    scale: Float[Tensor, "3"]
    fieldC8: float
    targetMaxSpeed: float
    maxSpeed: float
    fieldD4: int
    slipstreamSpeedMultiplier: float
    speedMultiplier: float
    rotation: Float[Tensor, "4"]
    fieldF0: Float[Tensor, "4"]
    field100: Float[Tensor, "4"]
    field110: Float[Tensor, "4"]
    mainMtx: Float[Tensor, "4 3"]
    field150: Float[Tensor, "4 3"]
    colReaction: int
    field184: Float[Tensor, "4 3"]
    charKartMtx: int
    colPos: Float[Tensor, "3"]
    prevColPos: Float[Tensor, "3"]
    colSphereSize: float
    colSphereZOffset: float
    netColPos: Float[Tensor, "3"]
    lastNetColPos: Float[Tensor, "3"]
    colPos2: Float[Tensor, "3"]
    field1FC: Float[Tensor, "3"]
    field208: POINTER_T[c_uint]
    field20C: list[c_void_p]
    field230: Callable[[POINTER_T[struct_driver_t]], None]
    xRot: int
    yRot: int
    boostTimer: int
    field23A: int
    driftBoostCounter: int
    PADDING_1: list[int]
    velocityMinusDirMultiplier: float
    upDir: Float[Tensor, "3"]
    field250: Float[Tensor, "3"]
    velocityY: Float[Tensor, "3"]
    fallsWaterForward: Float[Tensor, "3"]
    fallsWaterStrength: float
    forwardDir: Float[Tensor, "3"]
    jumpDriftUp: Float[Tensor, "3"]
    jumpDriftForward: Float[Tensor, "3"]
    collisionMode: int
    maxSpeedFraction: float
    deltaPosMag: float
    speed: float
    field2AC: float
    driverHitCheckMask: int
    driverHitMask: int
    lastDriverHitMask: int
    gap2B6: list[int]
    field2B8: int
    field2BC: int
    field2C0: int
    PADDING_2: list[int]
    leftRightDir: float
    colEntryId1: int
    colEntryId2: int
    kartPhysicalParams: POINTER_T[struct_physp_kart_params_t]
    charPhysicalParams: POINTER_T[struct_physp_char_params_t]
    turningAmount: float
    field2D8: Float[Tensor, "3"]
    field2E4: Float[Tensor, "3"]
    field2F0: Float[Tensor, "3"]
    driftLeftRightCount: int
    driftLeftCount: int
    driftRightCount: int
    driftDir1CountPtr: POINTER_T[c_ushort]
    driftDir2CountPtr: POINTER_T[c_ushort]
    driftLeftRightTimeout: int
    enemyState: POINTER_T[struct_enemy_t]
    field314: int
    PADDING_3: list[int]
    field318: float
    field31C: Float[Tensor, "3"]
    field328: Float[Tensor, "3"]
    field334: int
    field338: int
    PADDING_4: list[int]
    field33C: int
    field340: float
    field344: int
    field348: int
    field34C: Float[Tensor, "4"]
    colReactionCounter: int
    PADDING_5: list[int]
    field360: float
    spinOutAngle: int
    spinOutSpinCount: int
    spinOutProgress: float
    spinOutVelocity: int
    field370: int
    PADDING_6: list[int]
    field374: Float[Tensor, "3"]
    field380: int
    ghostFlickerPhase: int
    wallRotYSpeed: int
    driftRotY: int
    extraDrift: float
    field38C: float
    gap390: list[int]
    field394: int
    field398: float
    field39C: float
    field3A0: float
    tireRotX: int
    PADDING_7: list[int]
    field3A8: int
    respawnCounter: int
    PADDING_8: list[int]
    field3B0: Float[Tensor, "3"]
    field3BC: int
    field3BE: int
    preRespawnCounter: int
    PADDING_9: list[int]
    respawnId: int
    killTimer: int
    PADDING_10: list[int]
    driverVoiceIdx: int
    kartABC: int
    field3D2: int
    field3D4: int
    PADDING_11: list[int]
    place: int
    floorDriverColType: int
    floorColType: int
    floorColVariant: int
    field3E8: int
    PADDING_12: list[int]
    yRotSpeedTarget: int
    yRotSpeed: int
    field3F4: float
    field3F8: float
    field3FC: int
    field3FE: int
    field400: int
    PADDING_13: list[int]
    field404: float
    field408: int
    respawnStartFrame: int
    respawnAPressFrame: int
    field414: float
    field418: float
    growBackScale: Float[Tensor, "3"]
    thunderScale: Float[Tensor, "3"]
    dossunYScale: float
    mobjHitList: list[POINTER_T[struct_mobj_inst_t]]
    mobjHitSfxTimeout: list[int]
    mobjHitEmittedSfx: list[int]
    smashDossun: POINTER_T[struct_mobj_inst_t]
    field450: struct_driver_field450_t
    field4BC: float
    colFlagsMap2DShadow: int
    jumpPadSpeed: int
    field4C8: float
    field4CC: int
    field4D0: int
    preStartEnginePower: float
    fallsWaterDstId: int
    wallTouchTimeout: int
    floorTouchTimeout: int
    field4DE: int
    field4E0: int
    field4E2: int
    field4E4: int
    field4E6: int
    field4E8: float
    field4EC: float
    idkScale: Float[Tensor, "3"]
    field4FC: int
    PADDING_14: list[int]
    waterDepth: float
    field504: int
    field506: int
    field508: POINTER_T[struct_VecFx32]
    field50C: POINTER_T[struct_quaternion_t]
    field510: POINTER_T[struct_VecFx32]
    netState: POINTER_T[struct_driver_net_state_t]
    field518: struct_sfx_emitter_ex_params_t
    field534: c_void_p
    timers: struct_driver_timers_t
    charKart: POINTER_T[struct_charkart_t]
    field594: float
    field598: int
    PADDING_15: list[int]
    field59C: int
    field5A0: int
    gap5A2: list[int]
    field5A4: float

class driver_timers_t(Structure):
    """
    ```python
    shroomBoostTimer: int (POINTER(h))
    thunderShrinkTimer: int (POINTER(h))
    thunderGrowTimer: int (POINTER(h))
    starTimer: int (POINTER(h))
    slipstreamStartTimer: int (POINTER(h))
    slipstreamTimer: int (POINTER(h))
    dossunGrowTimer: int (POINTER(h))
    dossunFlatTimer: int (POINTER(h))
    teresaTimer: int (POINTER(h))
    teresaFlickerInterval: int (POINTER(h))
    teresaFlickerIntervalUpdateTimer: int (POINTER(h))
    teresaFlickerTimer: int (POINTER(h))
    teresaFlickerIntervalUpdateWaitTime: int (POINTER(h))
    teresaFlickerIntervalStep: int (POINTER(h))
    gessoInkTimer: int (POINTER(h))
    killerFrameCounter: int (POINTER(h))
    field20: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    isKillerFinishing: int (POINTER(i))
    killerTargetPlace: int (POINTER(h))
    gap2A: list[int] (POINTER(B)[2])
    killerState: list[int] (POINTER(B)[44])
    ```
    """
    _pack_: ClassVar[int] = 1
    shroomBoostTimer: int
    thunderShrinkTimer: int
    thunderGrowTimer: int
    starTimer: int
    slipstreamStartTimer: int
    slipstreamTimer: int
    dossunGrowTimer: int
    dossunFlatTimer: int
    teresaTimer: int
    teresaFlickerInterval: int
    teresaFlickerIntervalUpdateTimer: int
    teresaFlickerTimer: int
    teresaFlickerIntervalUpdateWaitTime: int
    teresaFlickerIntervalStep: int
    gessoInkTimer: int
    killerFrameCounter: int
    field20: int
    PADDING_0: list[int]
    isKillerFinishing: int
    killerTargetPlace: int
    gap2A: list[int]
    killerState: list[int]

class enemy_field140_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    driverFieldCC: int (POINTER(i))
    field8: int (POINTER(i))
    fieldC: int (POINTER(i))
    field10: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    driverFieldCC: int
    field8: int
    fieldC: int
    field10: int

class enemy_item_state_t(Structure):
    """
    ```python
    slotItemTimeout: int (POINTER(H))
    dragItemTimeout: int (POINTER(H))
    slotItemParams: POINTER_T[struct_enemy_item_params_t] (POINTER(struct_enemy_item_params_t))
    dragItemParams: POINTER_T[struct_enemy_item_params_t] (POINTER(struct_enemy_item_params_t))
    fieldC: int (POINTER(i))
    lxPressed: int (POINTER(i))
    dpadUpPressed: int (POINTER(i))
    dpadDownPressed: int (POINTER(i))
    dpadUpCounter: int (POINTER(H))
    dpadDownCounter: int (POINTER(H))
    field20: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    slotItemTimeout: int
    dragItemTimeout: int
    slotItemParams: POINTER_T[struct_enemy_item_params_t]
    dragItemParams: POINTER_T[struct_enemy_item_params_t]
    fieldC: int
    lxPressed: int
    dpadUpPressed: int
    dpadDownPressed: int
    dpadUpCounter: int
    dpadDownCounter: int
    field20: int

class enemy_rubberbanding_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    driverFieldCC: int (POINTER(i))
    rivalAggressiveness: int (POINTER(i))
    maxDriverFieldCC: int (POINTER(i))
    field10: int (POINTER(i))
    place: int (POINTER(i))
    field18: int (POINTER(i))
    field1C: int (POINTER(i))
    field20: int (POINTER(i))
    field24: int (POINTER(i))
    hasFailedStart: int (POINTER(i))
    hasStartBoost: int (POINTER(i))
    startBoostAmount: int (POINTER(H))
    field32: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    driverFieldCC: int
    rivalAggressiveness: int
    maxDriverFieldCC: int
    field10: int
    place: int
    field18: int
    field1C: int
    field20: int
    field24: int
    hasFailedStart: int
    hasStartBoost: int
    startBoostAmount: int
    field32: int

class fx16(Structure):
    """
    ```python
    val: int (POINTER(h))
    ```
    """
    _pack_: ClassVar[int] = 1
    val: int

class fx32(Structure):
    """
    ```python
    val: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    val: int

class kcol_header_t(Structure):
    """
    ```python
    posDataOffset: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    nrmDataOffset: POINTER_T[struct_VecFx16] (POINTER(struct_VecFx16))
    prismDataOffset: POINTER_T[struct_kcol_prism_data_t] (POINTER(struct_kcol_prism_data_t))
    blockDataOffset: POINTER_T[c_uint] (POINTER(POINTER(I)))
    prismThickness: float (struct_fx32)
    areaMinPos: Float[Tensor, "3"] (struct_VecFx32)
    areaXWidthMask: int (POINTER(I))
    areaYWidthMask: int (POINTER(I))
    areaZWidthMask: int (POINTER(I))
    blockWidthShift: int (POINTER(I))
    areaXBlocksShift: int (POINTER(I))
    areaXYBlocksShift: int (POINTER(I))
    sphereRadius: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    posDataOffset: POINTER_T[struct_VecFx32]
    nrmDataOffset: POINTER_T[struct_VecFx16]
    prismDataOffset: POINTER_T[struct_kcol_prism_data_t]
    blockDataOffset: POINTER_T[c_uint]
    prismThickness: float
    areaMinPos: Float[Tensor, "3"]
    areaXWidthMask: int
    areaYWidthMask: int
    areaZWidthMask: int
    blockWidthShift: int
    areaXBlocksShift: int
    areaXYBlocksShift: int
    sphereRadius: float

class light_t(Structure):
    """
    ```python
    color: int (POINTER(H))
    r: int (POINTER(h))
    g: int (POINTER(h))
    b: int (POINTER(h))
    rDelta: int (POINTER(h))
    gDelta: int (POINTER(h))
    bDelta: int (POINTER(h))
    lightMask: int (POINTER(H))
    progress: float (struct_fx16)
    ```
    """
    _pack_: ClassVar[int] = 1
    color: int
    r: int
    g: int
    b: int
    rDelta: int
    gDelta: int
    bDelta: int
    lightMask: int
    progress: float

class list_link_t(Structure):
    """
    ```python
    link: struct_NNSFndLink (struct_NNSFndLink)
    list: POINTER_T[struct_NNSFndList] (POINTER(struct_NNSFndList))
    ```
    """
    _pack_: ClassVar[int] = 1
    link: struct_NNSFndLink
    list: POINTER_T[struct_NNSFndList]

class mdat_enemypath_data_t(Structure):
    """
    ```python
    points: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    firstPoint: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    lastPoint: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    points: POINTER_T[struct_mdat_enemypoint_t]
    firstPoint: POINTER_T[struct_mdat_enemypoint_t]
    lastPoint: POINTER_T[struct_mdat_enemypoint_t]

class mdat_itempath_data_t(Structure):
    """
    ```python
    points: POINTER_T[struct_mdat_itempoint_t] (POINTER(struct_mdat_itempoint_t))
    firstPoint: POINTER_T[struct_mdat_itempoint_t] (POINTER(struct_mdat_itempoint_t))
    lastPoint: POINTER_T[struct_mdat_itempoint_t] (POINTER(struct_mdat_itempoint_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    points: POINTER_T[struct_mdat_itempoint_t]
    firstPoint: POINTER_T[struct_mdat_itempoint_t]
    lastPoint: POINTER_T[struct_mdat_itempoint_t]

class mdat_mapdata_t(Structure):
    """
    ```python
    obji: POINTER_T[struct_nkm_obji_entry_t] (POINTER(struct_nkm_obji_entry_t))
    objiCount: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    path: POINTER_T[struct_nkm_path_entry_t] (POINTER(struct_nkm_path_entry_t))
    pathCount: int (POINTER(H))
    PADDING_1: list[int] (POINTER(B)[2])
    poit: POINTER_T[struct_nkm_poit_entry_t] (POINTER(struct_nkm_poit_entry_t))
    poitCount: int (POINTER(H))
    PADDING_2: list[int] (POINTER(B)[2])
    stag: POINTER_T[struct_nkm_stag_data_t] (POINTER(struct_nkm_stag_data_t))
    ktps: POINTER_T[struct_nkm_ktps_entry_t] (POINTER(struct_nkm_ktps_entry_t))
    ktpsCount: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    ktpj: POINTER_T[struct_nkm_ktpj_entry_t] (POINTER(struct_nkm_ktpj_entry_t))
    ktpjCount: int (POINTER(H))
    PADDING_4: list[int] (POINTER(B)[2])
    ktp2: POINTER_T[struct_nkm_ktp2_entry_t] (POINTER(struct_nkm_ktp2_entry_t))
    ktp2Count: int (POINTER(H))
    PADDING_5: list[int] (POINTER(B)[2])
    ktpc: POINTER_T[struct_nkm_ktpc_entry_t] (POINTER(struct_nkm_ktpc_entry_t))
    ktpcCount: int (POINTER(H))
    PADDING_6: list[int] (POINTER(B)[2])
    ktpm: POINTER_T[struct_nkm_ktpm_entry_t] (POINTER(struct_nkm_ktpm_entry_t))
    ktpmCount: int (POINTER(H))
    PADDING_7: list[int] (POINTER(B)[2])
    cpoi: POINTER_T[struct_nkm_cpoi_entry_t] (POINTER(struct_nkm_cpoi_entry_t))
    cpoiCount: int (POINTER(H))
    PADDING_8: list[int] (POINTER(B)[2])
    cpat: POINTER_T[struct_nkm_cpat_entry_t] (POINTER(struct_nkm_cpat_entry_t))
    cpatCount: int (POINTER(H))
    PADDING_9: list[int] (POINTER(B)[2])
    ipoi: union_nkm_ipoi_entry_pointer_t (union_nkm_ipoi_entry_pointer_t)
    ipoiCount: int (POINTER(H))
    PADDING_10: list[int] (POINTER(B)[2])
    ipat: POINTER_T[struct_nkm_ipat_entry_t] (POINTER(struct_nkm_ipat_entry_t))
    ipatCount: int (POINTER(H))
    PADDING_11: list[int] (POINTER(B)[2])
    epoi: POINTER_T[struct_nkm_epoi_entry_t] (POINTER(struct_nkm_epoi_entry_t))
    epoiCount: int (POINTER(H))
    PADDING_12: list[int] (POINTER(B)[2])
    epat: POINTER_T[struct_nkm_epat_entry_t] (POINTER(struct_nkm_epat_entry_t))
    epatCount: int (POINTER(H))
    PADDING_13: list[int] (POINTER(B)[2])
    area: POINTER_T[struct_nkm_area_entry_t] (POINTER(struct_nkm_area_entry_t))
    areaCount: int (POINTER(H))
    PADDING_14: list[int] (POINTER(B)[2])
    came: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameCount: int (POINTER(H))
    PADDING_15: list[int] (POINTER(B)[2])
    mepo: POINTER_T[struct_nkm_mepo_entry_t] (POINTER(struct_nkm_mepo_entry_t))
    mepoCount: int (POINTER(H))
    PADDING_16: list[int] (POINTER(B)[2])
    mepa: POINTER_T[struct_nkm_mepa_entry_t] (POINTER(struct_nkm_mepa_entry_t))
    mepaCount: int (POINTER(H))
    PADDING_17: list[int] (POINTER(B)[2])
    paths: POINTER_T[struct_mdat_path_t] (POINTER(struct_mdat_path_t))
    cpoiKeyCount: int (POINTER(H))
    cpatLastCpoiIndex: int (POINTER(H))
    cpatMaxSectionOrder: int (POINTER(H))
    unknown49: int (POINTER(B))
    unknown50: int (POINTER(B))
    enemyPathData: struct_mdat_enemypath_data_t (struct_mdat_enemypath_data_t)
    itemPathData: struct_mdat_itempath_data_t (struct_mdat_itempath_data_t)
    mgEnemyPathData: struct_mdat_mgenemypath_data_t (struct_mdat_mgenemypath_data_t)
    cameIntroFirstTopCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameIntroFirstBottomCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameType6: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameBattleIntroCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameMissionFinishCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    clipAreaLists: list[POINTER_T[struct_mdat_clip_area_list_entry_t]] (POINTER(struct_mdat_clip_area_list_entry_t)[8])
    ktpjIndexTable: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    ktpcIndexTable: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    curMgRespawnId: int (POINTER(H))
    enemyRespawnRouteAreaCount: int (POINTER(H))
    trackLength: float (struct_fx32)
    trackLengthDiv15000: int (POINTER(I))
    nkmVersion: int (POINTER(H))
    unknown141: int (POINTER(B))
    missionEndAreaCount: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    obji: POINTER_T[struct_nkm_obji_entry_t]
    objiCount: int
    PADDING_0: list[int]
    path: POINTER_T[struct_nkm_path_entry_t]
    pathCount: int
    PADDING_1: list[int]
    poit: POINTER_T[struct_nkm_poit_entry_t]
    poitCount: int
    PADDING_2: list[int]
    stag: POINTER_T[struct_nkm_stag_data_t]
    ktps: POINTER_T[struct_nkm_ktps_entry_t]
    ktpsCount: int
    PADDING_3: list[int]
    ktpj: POINTER_T[struct_nkm_ktpj_entry_t]
    ktpjCount: int
    PADDING_4: list[int]
    ktp2: POINTER_T[struct_nkm_ktp2_entry_t]
    ktp2Count: int
    PADDING_5: list[int]
    ktpc: POINTER_T[struct_nkm_ktpc_entry_t]
    ktpcCount: int
    PADDING_6: list[int]
    ktpm: POINTER_T[struct_nkm_ktpm_entry_t]
    ktpmCount: int
    PADDING_7: list[int]
    cpoi: POINTER_T[struct_nkm_cpoi_entry_t]
    cpoiCount: int
    PADDING_8: list[int]
    cpat: POINTER_T[struct_nkm_cpat_entry_t]
    cpatCount: int
    PADDING_9: list[int]
    ipoi: union_nkm_ipoi_entry_pointer_t
    ipoiCount: int
    PADDING_10: list[int]
    ipat: POINTER_T[struct_nkm_ipat_entry_t]
    ipatCount: int
    PADDING_11: list[int]
    epoi: POINTER_T[struct_nkm_epoi_entry_t]
    epoiCount: int
    PADDING_12: list[int]
    epat: POINTER_T[struct_nkm_epat_entry_t]
    epatCount: int
    PADDING_13: list[int]
    area: POINTER_T[struct_nkm_area_entry_t]
    areaCount: int
    PADDING_14: list[int]
    came: POINTER_T[struct_nkm_came_entry_t]
    cameCount: int
    PADDING_15: list[int]
    mepo: POINTER_T[struct_nkm_mepo_entry_t]
    mepoCount: int
    PADDING_16: list[int]
    mepa: POINTER_T[struct_nkm_mepa_entry_t]
    mepaCount: int
    PADDING_17: list[int]
    paths: POINTER_T[struct_mdat_path_t]
    cpoiKeyCount: int
    cpatLastCpoiIndex: int
    cpatMaxSectionOrder: int
    unknown49: int
    unknown50: int
    enemyPathData: struct_mdat_enemypath_data_t
    itemPathData: struct_mdat_itempath_data_t
    mgEnemyPathData: struct_mdat_mgenemypath_data_t
    cameIntroFirstTopCam: POINTER_T[struct_nkm_came_entry_t]
    cameIntroFirstBottomCam: POINTER_T[struct_nkm_came_entry_t]
    cameType6: POINTER_T[struct_nkm_came_entry_t]
    cameBattleIntroCam: POINTER_T[struct_nkm_came_entry_t]
    cameMissionFinishCam: POINTER_T[struct_nkm_came_entry_t]
    clipAreaLists: list[POINTER_T[struct_mdat_clip_area_list_entry_t]]
    ktpjIndexTable: POINTER_T[c_ushort]
    ktpcIndexTable: POINTER_T[c_ushort]
    curMgRespawnId: int
    enemyRespawnRouteAreaCount: int
    trackLength: float
    trackLengthDiv15000: int
    nkmVersion: int
    unknown141: int
    missionEndAreaCount: int

class mdat_mgenemypath_data_t(Structure):
    """
    ```python
    points: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    firstPoint: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    lastPoint: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    points: POINTER_T[struct_mdat_mgenemypoint_t]
    firstPoint: POINTER_T[struct_mdat_mgenemypoint_t]
    lastPoint: POINTER_T[struct_mdat_mgenemypoint_t]

class model_res_t(Structure):
    """
    ```python
    link: struct_NNSFndLink (struct_NNSFndLink)
    nsbmd: c_void_p (c_void_p)
    texRes: POINTER_T[struct_NNSG3dResTex_] (POINTER(struct_NNSG3dResTex_))
    ```
    """
    _pack_: ClassVar[int] = 1
    link: struct_NNSFndLink
    nsbmd: c_void_p
    texRes: POINTER_T[struct_NNSG3dResTex_]

class nkm_epoi_entry_settings_t(Structure):
    """
    ```python
    drifting: int (POINTER(H))
    unknown1: int (POINTER(H))
    unknown2: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    drifting: int
    unknown1: int
    unknown2: int

class nkm_ipoi_entry_pointer_t(Union):
    """
    ```python
    final: POINTER_T[struct_nkm_ipoi_entry_t] (POINTER(struct_nkm_ipoi_entry_t))
    beta: POINTER_T[struct_nkm_ipoi_entry_beta_t] (POINTER(struct_nkm_ipoi_entry_beta_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    final: POINTER_T[struct_nkm_ipoi_entry_t]
    beta: POINTER_T[struct_nkm_ipoi_entry_beta_t]

class nkm_mepo_entry_settings_t(Structure):
    """
    ```python
    drifting: int (POINTER(I))
    unknown1: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    drifting: int
    unknown1: int

class quaternion_t(Structure):
    """
    ```python
    x: float (struct_fx32)
    y: float (struct_fx32)
    z: float (struct_fx32)
    w: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    x: float
    y: float
    z: float
    w: float

class race_skill_rankpoints_t(Structure):
    """
    ```python
    rankTimeDeltaPoints: int (POINTER(i))
    firstPlacePercentagePoints: int (POINTER(i))
    startBoostPoints: int (POINTER(i))
    powerSlidePoints: int (POINTER(i))
    itemHitPoints: int (POINTER(i))
    offRoadTimePoints: int (POINTER(i))
    wallHitPoints: int (POINTER(i))
    damagePoints: int (POINTER(i))
    respawnPoints: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    rankTimeDeltaPoints: int
    firstPlacePercentagePoints: int
    startBoostPoints: int
    powerSlidePoints: int
    itemHitPoints: int
    offRoadTimePoints: int
    wallHitPoints: int
    damagePoints: int
    respawnPoints: int

class race_state_t(Structure):
    """
    ```python
    state: int (POINTER(h))
    PADDING_0: list[int] (POINTER(B)[2])
    frameCounter: int (POINTER(I))
    frameCounter2: int (POINTER(i))
    frameCounterModulo8: int (POINTER(i))
    isOddFrame: int (POINTER(i))
    frameCounterModuloDriverCount: int (POINTER(i))
    toonTableOffset: int (POINTER(I))
    toonTableUpdateCounter: int (POINTER(I))
    darkeningFogState: int (POINTER(I))
    prevDarkeningFogState: int (POINTER(I))
    isCamAnimMode: int (POINTER(i))
    isCamAnimSingleScreen: int (POINTER(i))
    field30: int (POINTER(I))
    field34: int (POINTER(I))
    isAwardStaffRoll: int (POINTER(i))
    field3C: int (POINTER(I))
    light0Dir: Float[Tensor, "3"] (struct_VecFx16)
    PADDING_1: list[int] (POINTER(B)[2])
    ```
    """
    _pack_: ClassVar[int] = 1
    state: int
    PADDING_0: list[int]
    frameCounter: int
    frameCounter2: int
    frameCounterModulo8: int
    isOddFrame: int
    frameCounterModuloDriverCount: int
    toonTableOffset: int
    toonTableUpdateCounter: int
    darkeningFogState: int
    prevDarkeningFogState: int
    isCamAnimMode: int
    isCamAnimSingleScreen: int
    field30: int
    field34: int
    isAwardStaffRoll: int
    field3C: int
    light0Dir: Float[Tensor, "3"]
    PADDING_1: list[int]

class race_status_t(Structure):
    """
    ```python
    time: struct_race_status_time_t (struct_race_status_time_t)
    rankTimeGp: int (POINTER(H))
    finishedDriverCount: int (POINTER(H))
    field10: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    driverStatus: list[struct_race_driver_status_t] (struct_race_driver_status_t[8])
    placeDriverIds: list[int] (POINTER(b)[8])
    PADDING_1: list[int] (POINTER(B)[4])
    safeRng: struct_MATHRandContext32 (struct_MATHRandContext32)
    rngSeed: int (POINTER(I))
    PADDING_2: list[int] (POINTER(B)[4])
    randomRng: struct_MATHRandContext32 (struct_MATHRandContext32)
    stableRng: struct_MATHRandContext32 (struct_MATHRandContext32)
    rankTimeGpRtt: POINTER_T[struct_ranktime_gp_t] (POINTER(struct_ranktime_gp_t))
    resultsStored: int (POINTER(i))
    camAnimComplete: int (POINTER(i))
    cpoiKeyPointProgress: POINTER_T[struct_fx32] (POINTER(struct_fx32))
    gap4D8: list[int] (POINTER(B)[4])
    rankPointRpt: POINTER_T[struct_rankpoint_t] (POINTER(struct_rankpoint_t))
    missionResult: int (POINTER(I))
    oneDivCpatSegmentCount: float (struct_fx32)
    oneDivNrLaps: float (struct_fx32)
    useTimeLimit: int (POINTER(i))
    uncontrollable: int (POINTER(i))
    timeLimit: int (POINTER(I))
    field4F8: int (POINTER(B))
    field4F9: int (POINTER(B))
    mrWinDelayCounter: int (POINTER(H))
    mrLoseDelayCounter: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    skillRankPoints: struct_race_skill_rankpoints_t (struct_race_skill_rankpoints_t)
    PADDING_4: list[int] (POINTER(B)[4])
    ```
    """
    _pack_: ClassVar[int] = 1
    time: struct_race_status_time_t
    rankTimeGp: int
    finishedDriverCount: int
    field10: int
    PADDING_0: list[int]
    driverStatus: list[struct_race_driver_status_t]
    placeDriverIds: list[int]
    PADDING_1: list[int]
    safeRng: struct_MATHRandContext32
    rngSeed: int
    PADDING_2: list[int]
    randomRng: struct_MATHRandContext32
    stableRng: struct_MATHRandContext32
    rankTimeGpRtt: POINTER_T[struct_ranktime_gp_t]
    resultsStored: int
    camAnimComplete: int
    cpoiKeyPointProgress: POINTER_T[struct_fx32]
    gap4D8: list[int]
    rankPointRpt: POINTER_T[struct_rankpoint_t]
    missionResult: int
    oneDivCpatSegmentCount: float
    oneDivNrLaps: float
    useTimeLimit: int
    uncontrollable: int
    timeLimit: int
    field4F8: int
    field4F9: int
    mrWinDelayCounter: int
    mrLoseDelayCounter: int
    PADDING_3: list[int]
    skillRankPoints: struct_race_skill_rankpoints_t
    PADDING_4: list[int]

class race_status_time_t(Structure):
    """
    ```python
    frameCounter: int (POINTER(I))
    timeRunning: int (POINTER(i))
    lapTime: struct_race_time_t (struct_race_time_t)
    ```
    """
    _pack_: ClassVar[int] = 1
    frameCounter: int
    timeRunning: int
    lapTime: struct_race_time_t

class race_time_t(Structure):
    """
    ```python
    milliseconds: int (POINTER(H))
    minutes: int (POINTER(B))
    seconds: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    milliseconds: int
    minutes: int
    seconds: int

class sfx_emitter_ex_params_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(i))
    pitchOffset: int (POINTER(I))
    fieldC: int (POINTER(I))
    unk10: list[int] (POINTER(B)[12])
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    pitchOffset: int
    fieldC: int
    unk10: list[int]

class sfx_emitter_t(Structure):
    """
    ```python
    listLink: struct_list_link_t (struct_list_link_t)
    soundList: struct_NNSFndList (struct_NNSFndList)
    field18: int (POINTER(I))
    field1C: struct_list_link_t (struct_list_link_t)
    position: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    startFunc: c_void_p (c_void_p)
    updateFunc: c_void_p (c_void_p)
    field34: int (POINTER(H))
    field36: int (POINTER(H))
    field38: int (POINTER(I))
    sfxParamIdx: int (POINTER(B))
    field3D: list[int] (POINTER(B)[3])
    squareDistance: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    listLink: struct_list_link_t
    soundList: struct_NNSFndList
    field18: int
    field1C: struct_list_link_t
    position: POINTER_T[struct_VecFx32]
    startFunc: c_void_p
    updateFunc: c_void_p
    field34: int
    field36: int
    field38: int
    sfxParamIdx: int
    field3D: list[int]
    squareDistance: int

class state_machine_t(Structure):
    """
    ```python
    states: POINTER_T[struct_state_machine_state_t] (POINTER(struct_state_machine_state_t))
    counter: int (POINTER(I))
    userData: c_void_p (c_void_p)
    nrStates: int (POINTER(H))
    curState: int (POINTER(H))
    nextState: int (POINTER(H))
    gotoNextState: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    states: POINTER_T[struct_state_machine_state_t]
    counter: int
    userData: c_void_p
    nrStates: int
    curState: int
    nextState: int
    gotoNextState: int

class struc_313_mepo(Structure):
    """
    ```python
    nextMepo: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    curMepo: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    direction: Float[Tensor, "3"] (struct_VecFx32)
    areaMepo: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    areaMepoValid: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    nextMepo: POINTER_T[struct_mdat_mgenemypoint_t]
    curMepo: POINTER_T[struct_mdat_mgenemypoint_t]
    direction: Float[Tensor, "3"]
    areaMepo: POINTER_T[struct_mdat_mgenemypoint_t]
    areaMepoValid: int

class struc_316_epoi(Structure):
    """
    ```python
    nextEpoi: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    curEpoi: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    direction: Float[Tensor, "3"] (struct_VecFx32)
    areaEpoi: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    areaEpoiValid: int (POINTER(i))
    driverId: int (POINTER(H))
    field1E: int (POINTER(B))
    field1F: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    nextEpoi: POINTER_T[struct_mdat_enemypoint_t]
    curEpoi: POINTER_T[struct_mdat_enemypoint_t]
    direction: Float[Tensor, "3"]
    areaEpoi: POINTER_T[struct_mdat_enemypoint_t]
    areaEpoiValid: int
    driverId: int
    field1E: int
    field1F: int

class struc_334(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(i))
    field8: int (POINTER(i))
    fieldC: int (POINTER(i))
    field10: int (POINTER(i))
    field14: int (POINTER(i))
    field18: int (POINTER(i))
    field1C: int (POINTER(i))
    field20: int (POINTER(i))
    field24: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    field8: int
    fieldC: int
    field10: int
    field14: int
    field18: int
    field1C: int
    field20: int
    field24: int

class struct_MATHRandContext32(Structure):
    """
    ```python
    x: int (POINTER(L))
    mul: int (POINTER(L))
    add: int (POINTER(L))
    ```
    """
    _pack_: ClassVar[int] = 1
    x: int
    mul: int
    add: int

class struct_MtxFx33_0(Structure):
    """
    ```python
    _00: float (struct_fx32)
    _01: float (struct_fx32)
    _02: float (struct_fx32)
    _10: float (struct_fx32)
    _11: float (struct_fx32)
    _12: float (struct_fx32)
    _20: float (struct_fx32)
    _21: float (struct_fx32)
    _22: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    _00: float
    _01: float
    _02: float
    _10: float
    _11: float
    _12: float
    _20: float
    _21: float
    _22: float

class struct_MtxFx43_0(Structure):
    """
    ```python
    _00: float (struct_fx32)
    _01: float (struct_fx32)
    _02: float (struct_fx32)
    _10: float (struct_fx32)
    _11: float (struct_fx32)
    _12: float (struct_fx32)
    _20: float (struct_fx32)
    _21: float (struct_fx32)
    _22: float (struct_fx32)
    _30: float (struct_fx32)
    _31: float (struct_fx32)
    _32: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    _00: float
    _01: float
    _02: float
    _10: float
    _11: float
    _12: float
    _20: float
    _21: float
    _22: float
    _30: float
    _31: float
    _32: float

class struct_NNSFndLink(Structure):
    """
    ```python
    prevObject: c_void_p (c_void_p)
    nextObject: c_void_p (c_void_p)
    ```
    """
    _pack_: ClassVar[int] = 1
    prevObject: c_void_p
    nextObject: c_void_p

class struct_NNSFndList(Structure):
    """
    ```python
    headObject: c_void_p (c_void_p)
    tailObject: c_void_p (c_void_p)
    numObjects: int (POINTER(H))
    offset: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    headObject: c_void_p
    tailObject: c_void_p
    numObjects: int
    offset: int

class struct_NNSG3dAnmObj_(Structure):
    """
    ```python
    frame: float (struct_fx32)
    ratio: float (struct_fx32)
    resAnm: c_void_p (c_void_p)
    funcAnm: c_void_p (c_void_p)
    next: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    resTex: POINTER_T[struct_NNSG3dResTex_] (POINTER(struct_NNSG3dResTex_))
    priority: int (POINTER(B))
    numMapData: int (POINTER(B))
    mapData: list[int] (POINTER(H)[1])
    ```
    """
    _pack_: ClassVar[int] = 1
    frame: float
    ratio: float
    resAnm: c_void_p
    funcAnm: c_void_p
    next: POINTER_T[struct_NNSG3dAnmObj_]
    resTex: POINTER_T[struct_NNSG3dResTex_]
    priority: int
    numMapData: int
    mapData: list[int]

class struct_NNSG3dJntAnmResult_(Structure):
    """
    ```python
    flag: int (POINTER(I))
    scale: Float[Tensor, "3"] (struct_VecFx32)
    scaleEx0: Float[Tensor, "3"] (struct_VecFx32)
    scaleEx1: Float[Tensor, "3"] (struct_VecFx32)
    rot: Float[Tensor, "3 3"] (union_MtxFx33)
    trans: Float[Tensor, "3"] (struct_VecFx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    flag: int
    scale: Float[Tensor, "3"]
    scaleEx0: Float[Tensor, "3"]
    scaleEx1: Float[Tensor, "3"]
    rot: Float[Tensor, "3 3"]
    trans: Float[Tensor, "3"]

class struct_NNSG3dMatAnmResult_(Structure):
    """
    ```python
    flag: int (POINTER(I))
    prmMatColor0: int (POINTER(I))
    prmMatColor1: int (POINTER(I))
    prmPolygonAttr: int (POINTER(I))
    prmTexImage: int (POINTER(I))
    prmTexPltt: int (POINTER(I))
    scaleS: float (struct_fx32)
    scaleT: float (struct_fx32)
    sinR: float (struct_fx16)
    cosR: float (struct_fx16)
    transS: float (struct_fx32)
    transT: float (struct_fx32)
    origWidth: int (POINTER(H))
    origHeight: int (POINTER(H))
    magW: float (struct_fx32)
    magH: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    flag: int
    prmMatColor0: int
    prmMatColor1: int
    prmPolygonAttr: int
    prmTexImage: int
    prmTexPltt: int
    scaleS: float
    scaleT: float
    sinR: float
    cosR: float
    transS: float
    transT: float
    origWidth: int
    origHeight: int
    magW: float
    magH: float

class struct_NNSG3dRS_(Structure):
    """
    ```python
    c: POINTER_T[c_ubyte] (POINTER(POINTER(B)))
    pRenderObj: POINTER_T[struct_NNSG3dRenderObj_] (POINTER(struct_NNSG3dRenderObj_))
    flag: int (POINTER(I))
    cbVecFunc: list[Callable[[POINTER_T[struct_NNSG3dRS_]], None]] (CFunctionType[32])
    cbVecTiming: list[int] (POINTER(B)[32])
    currentNode: int (POINTER(B))
    currentMat: int (POINTER(B))
    currentNodeDesc: int (POINTER(B))
    dummy_: int (POINTER(B))
    pMatAnmResult: POINTER_T[struct_NNSG3dMatAnmResult_] (POINTER(struct_NNSG3dMatAnmResult_))
    pJntAnmResult: POINTER_T[struct_NNSG3dJntAnmResult_] (POINTER(struct_NNSG3dJntAnmResult_))
    pVisAnmResult: POINTER_T[struct_NNSG3dVisAnmResult_] (POINTER(struct_NNSG3dVisAnmResult_))
    isMatCached: list[int] (POINTER(I)[2])
    isScaleCacheOne: list[int] (POINTER(I)[2])
    isEvpCached: list[int] (POINTER(I)[2])
    pResNodeInfo: POINTER_T[struct_NNSG3dResNodeInfo_] (POINTER(struct_NNSG3dResNodeInfo_))
    pResMat: POINTER_T[struct_NNSG3dResMat_] (POINTER(struct_NNSG3dResMat_))
    pResShp: POINTER_T[struct_NNSG3dResShp_] (POINTER(struct_NNSG3dResShp_))
    posScale: float (struct_fx32)
    invPosScale: float (struct_fx32)
    funcJntScale: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_], POINTER_T[struct_fx32], POINTER_T[c_ubyte], int], None] (CFunctionType)
    funcJntMtx: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_]], None] (CFunctionType)
    funcTexMtx: Callable[[POINTER_T[struct_NNSG3dMatAnmResult_]], None] (CFunctionType)
    tmpMatAnmResult: struct_NNSG3dMatAnmResult_ (struct_NNSG3dMatAnmResult_)
    tmpJntAnmResult: struct_NNSG3dJntAnmResult_ (struct_NNSG3dJntAnmResult_)
    tmpVisAnmResult: struct_NNSG3dVisAnmResult_ (struct_NNSG3dVisAnmResult_)
    ```
    """
    _pack_: ClassVar[int] = 1
    c: POINTER_T[c_ubyte]
    pRenderObj: POINTER_T[struct_NNSG3dRenderObj_]
    flag: int
    cbVecFunc: list[Callable[[POINTER_T[struct_NNSG3dRS_]], None]]
    cbVecTiming: list[int]
    currentNode: int
    currentMat: int
    currentNodeDesc: int
    dummy_: int
    pMatAnmResult: POINTER_T[struct_NNSG3dMatAnmResult_]
    pJntAnmResult: POINTER_T[struct_NNSG3dJntAnmResult_]
    pVisAnmResult: POINTER_T[struct_NNSG3dVisAnmResult_]
    isMatCached: list[int]
    isScaleCacheOne: list[int]
    isEvpCached: list[int]
    pResNodeInfo: POINTER_T[struct_NNSG3dResNodeInfo_]
    pResMat: POINTER_T[struct_NNSG3dResMat_]
    pResShp: POINTER_T[struct_NNSG3dResShp_]
    posScale: float
    invPosScale: float
    funcJntScale: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_], POINTER_T[struct_fx32], POINTER_T[c_ubyte], int], None]
    funcJntMtx: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_]], None]
    funcTexMtx: Callable[[POINTER_T[struct_NNSG3dMatAnmResult_]], None]
    tmpMatAnmResult: struct_NNSG3dMatAnmResult_
    tmpJntAnmResult: struct_NNSG3dJntAnmResult_
    tmpVisAnmResult: struct_NNSG3dVisAnmResult_

class struct_NNSG3dRenderObj_(Structure):
    """
    ```python
    flag: int (POINTER(I))
    resMdl: POINTER_T[struct_NNSG3dResMdl_] (POINTER(struct_NNSG3dResMdl_))
    anmMat: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    funcBlendMat: Callable[[POINTER_T[struct_NNSG3dMatAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int] (CFunctionType)
    anmJnt: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    funcBlendJnt: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int] (CFunctionType)
    anmVis: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    funcBlendVis: Callable[[POINTER_T[struct_NNSG3dVisAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int] (CFunctionType)
    cbFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None] (CFunctionType)
    cbCmd: int (POINTER(B))
    cbTiming: int (POINTER(B))
    dummy_: int (POINTER(H))
    cbInitFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None] (CFunctionType)
    ptrUser: c_void_p (c_void_p)
    ptrUserSbc: POINTER_T[c_ubyte] (POINTER(POINTER(B)))
    recJntAnm: POINTER_T[struct_NNSG3dJntAnmResult_] (POINTER(struct_NNSG3dJntAnmResult_))
    recMatAnm: POINTER_T[struct_NNSG3dMatAnmResult_] (POINTER(struct_NNSG3dMatAnmResult_))
    hintMatAnmExist: list[int] (POINTER(I)[2])
    hintJntAnmExist: list[int] (POINTER(I)[2])
    hintVisAnmExist: list[int] (POINTER(I)[2])
    ```
    """
    _pack_: ClassVar[int] = 1
    flag: int
    resMdl: POINTER_T[struct_NNSG3dResMdl_]
    anmMat: POINTER_T[struct_NNSG3dAnmObj_]
    funcBlendMat: Callable[[POINTER_T[struct_NNSG3dMatAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int]
    anmJnt: POINTER_T[struct_NNSG3dAnmObj_]
    funcBlendJnt: Callable[[POINTER_T[struct_NNSG3dJntAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int]
    anmVis: POINTER_T[struct_NNSG3dAnmObj_]
    funcBlendVis: Callable[[POINTER_T[struct_NNSG3dVisAnmResult_], POINTER_T[struct_NNSG3dAnmObj_], int], c_int]
    cbFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None]
    cbCmd: int
    cbTiming: int
    dummy_: int
    cbInitFunc: Callable[[POINTER_T[struct_NNSG3dRS_]], None]
    ptrUser: c_void_p
    ptrUserSbc: POINTER_T[c_ubyte]
    recJntAnm: POINTER_T[struct_NNSG3dJntAnmResult_]
    recMatAnm: POINTER_T[struct_NNSG3dMatAnmResult_]
    hintMatAnmExist: list[int]
    hintJntAnmExist: list[int]
    hintVisAnmExist: list[int]

class struct_NNSG3dResDataBlockHeader_(Structure):
    """
    ```python
    _0: union_NNSG3dResDataBlockHeader__0 (union_NNSG3dResDataBlockHeader__0)
    size: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    _0: union_NNSG3dResDataBlockHeader__0
    size: int

class struct_NNSG3dResDictTreeNode_(Structure):
    """
    ```python
    refBit: int (POINTER(B))
    idxLeft: int (POINTER(B))
    idxRight: int (POINTER(B))
    idxEntry: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    refBit: int
    idxLeft: int
    idxRight: int
    idxEntry: int

class struct_NNSG3dResDict_(Structure):
    """
    ```python
    revision: int (POINTER(B))
    numEntry: int (POINTER(B))
    sizeDictBlk: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsEntry: int (POINTER(H))
    node: list[struct_NNSG3dResDictTreeNode_] (struct_NNSG3dResDictTreeNode_[1])
    ```
    """
    _pack_: ClassVar[int] = 1
    revision: int
    numEntry: int
    sizeDictBlk: int
    dummy_: int
    ofsEntry: int
    node: list[struct_NNSG3dResDictTreeNode_]

class struct_NNSG3dResMat_(Structure):
    """
    ```python
    ofsDictTexToMatList: int (POINTER(H))
    ofsDictPlttToMatList: int (POINTER(H))
    dict: struct_NNSG3dResDict_ (struct_NNSG3dResDict_)
    ```
    """
    _pack_: ClassVar[int] = 1
    ofsDictTexToMatList: int
    ofsDictPlttToMatList: int
    dict: struct_NNSG3dResDict_

class struct_NNSG3dResMdlInfo_(Structure):
    """
    ```python
    sbcType: int (POINTER(B))
    scalingRule: int (POINTER(B))
    texMtxMode: int (POINTER(B))
    numNode: int (POINTER(B))
    numMat: int (POINTER(B))
    numShp: int (POINTER(B))
    firstUnusedMtxStackID: int (POINTER(B))
    dummy_: int (POINTER(B))
    posScale: float (struct_fx32)
    invPosScale: float (struct_fx32)
    numVertex: int (POINTER(H))
    numPolygon: int (POINTER(H))
    numTriangle: int (POINTER(H))
    numQuad: int (POINTER(H))
    boxX: float (struct_fx16)
    boxY: float (struct_fx16)
    boxZ: float (struct_fx16)
    boxW: float (struct_fx16)
    boxH: float (struct_fx16)
    boxD: float (struct_fx16)
    boxPosScale: float (struct_fx32)
    boxInvPosScale: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    sbcType: int
    scalingRule: int
    texMtxMode: int
    numNode: int
    numMat: int
    numShp: int
    firstUnusedMtxStackID: int
    dummy_: int
    posScale: float
    invPosScale: float
    numVertex: int
    numPolygon: int
    numTriangle: int
    numQuad: int
    boxX: float
    boxY: float
    boxZ: float
    boxW: float
    boxH: float
    boxD: float
    boxPosScale: float
    boxInvPosScale: float

class struct_NNSG3dResMdl_(Structure):
    """
    ```python
    size: int (POINTER(I))
    ofsSbc: int (POINTER(I))
    ofsMat: int (POINTER(I))
    ofsShp: int (POINTER(I))
    ofsEvpMtx: int (POINTER(I))
    info: struct_NNSG3dResMdlInfo_ (struct_NNSG3dResMdlInfo_)
    nodeInfo: struct_NNSG3dResNodeInfo_ (struct_NNSG3dResNodeInfo_)
    ```
    """
    _pack_: ClassVar[int] = 1
    size: int
    ofsSbc: int
    ofsMat: int
    ofsShp: int
    ofsEvpMtx: int
    info: struct_NNSG3dResMdlInfo_
    nodeInfo: struct_NNSG3dResNodeInfo_

class struct_NNSG3dResNodeInfo_(Structure):
    """
    ```python
    dict: struct_NNSG3dResDict_ (struct_NNSG3dResDict_)
    ```
    """
    _pack_: ClassVar[int] = 1
    dict: struct_NNSG3dResDict_

class struct_NNSG3dResPlttInfo_(Structure):
    """
    ```python
    vramKey: int (POINTER(I))
    sizePltt: int (POINTER(H))
    flag: int (POINTER(H))
    ofsDict: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsPlttData: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    vramKey: int
    sizePltt: int
    flag: int
    ofsDict: int
    dummy_: int
    ofsPlttData: int

class struct_NNSG3dResShp_(Structure):
    """
    ```python
    dict: struct_NNSG3dResDict_ (struct_NNSG3dResDict_)
    ```
    """
    _pack_: ClassVar[int] = 1
    dict: struct_NNSG3dResDict_

class struct_NNSG3dResTex4x4Info_(Structure):
    """
    ```python
    vramKey: int (POINTER(I))
    sizeTex: int (POINTER(H))
    ofsDict: int (POINTER(H))
    flag: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsTex: int (POINTER(I))
    ofsTexPlttIdx: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    vramKey: int
    sizeTex: int
    ofsDict: int
    flag: int
    dummy_: int
    ofsTex: int
    ofsTexPlttIdx: int

class struct_NNSG3dResTexInfo_(Structure):
    """
    ```python
    vramKey: int (POINTER(I))
    sizeTex: int (POINTER(H))
    ofsDict: int (POINTER(H))
    flag: int (POINTER(H))
    dummy_: int (POINTER(H))
    ofsTex: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    vramKey: int
    sizeTex: int
    ofsDict: int
    flag: int
    dummy_: int
    ofsTex: int

class struct_NNSG3dResTex_(Structure):
    """
    ```python
    header: struct_NNSG3dResDataBlockHeader_ (struct_NNSG3dResDataBlockHeader_)
    texInfo: struct_NNSG3dResTexInfo_ (struct_NNSG3dResTexInfo_)
    tex4x4Info: struct_NNSG3dResTex4x4Info_ (struct_NNSG3dResTex4x4Info_)
    plttInfo: struct_NNSG3dResPlttInfo_ (struct_NNSG3dResPlttInfo_)
    dict: struct_NNSG3dResDict_ (struct_NNSG3dResDict_)
    ```
    """
    _pack_: ClassVar[int] = 1
    header: struct_NNSG3dResDataBlockHeader_
    texInfo: struct_NNSG3dResTexInfo_
    tex4x4Info: struct_NNSG3dResTex4x4Info_
    plttInfo: struct_NNSG3dResPlttInfo_
    dict: struct_NNSG3dResDict_

class struct_NNSG3dVisAnmResult_(Structure):
    """
    ```python
    isVisible: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    isVisible: int

class struct_VecFx16(Structure):
    """
    ```python
    x: float (struct_fx16)
    y: float (struct_fx16)
    z: float (struct_fx16)
    ```
    """
    _pack_: ClassVar[int] = 1
    x: float
    y: float
    z: float

class struct_VecFx32(Structure):
    """
    ```python
    x: float (struct_fx32)
    y: float (struct_fx32)
    z: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    x: float
    y: float
    z: float

class struct_anim_animator_t(Structure):
    """
    ```python
    loopMode: int (POINTER(H))
    hasEnded: int (POINTER(H))
    animLength: float (struct_fx32)
    speed: float (struct_fx32)
    progress: float (struct_fx32)
    loopIteration: int (POINTER(H))
    loopCount: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    loopMode: int
    hasEnded: int
    animLength: float
    speed: float
    progress: float
    loopIteration: int
    loopCount: int

class struct_anim_manager_t(Structure):
    """
    ```python
    animator: struct_anim_animator_t (struct_anim_animator_t)
    model: POINTER_T[struct_model_t] (POINTER(struct_model_t))
    animCount: int (POINTER(H))
    curAnimIdx: int (POINTER(H))
    anmObjs: POINTER_T[POINTER_T[struct_NNSG3dAnmObj_]] (POINTER(POINTER(struct_NNSG3dAnmObj_)))
    loopFlags: POINTER_T[c_int] (POINTER(POINTER(i)))
    field24: int (POINTER(I))
    animKind: int (POINTER(I))
    isBlending: int (POINTER(i))
    blendSpeed: float (struct_fx32)
    blendAnmObj: POINTER_T[struct_NNSG3dAnmObj_] (POINTER(struct_NNSG3dAnmObj_))
    ```
    """
    _pack_: ClassVar[int] = 1
    animator: struct_anim_animator_t
    model: POINTER_T[struct_model_t]
    animCount: int
    curAnimIdx: int
    anmObjs: POINTER_T[POINTER_T[struct_NNSG3dAnmObj_]]
    loopFlags: POINTER_T[c_int]
    field24: int
    animKind: int
    isBlending: int
    blendSpeed: float
    blendAnmObj: POINTER_T[struct_NNSG3dAnmObj_]

class struct_came_routestat_t(Structure):
    """
    ```python
    pointCache: list[Float[Tensor, "3"]] (struct_VecFx32[4])
    progress: int (POINTER(i))
    index: int (POINTER(i))
    field38: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    pointCache: list[Float[Tensor, "3"]]
    progress: int
    index: int
    field38: int

class struct_came_unknown_t(Structure):
    """
    ```python
    cameEntry: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    routestat1: struct_came_routestat_t (struct_came_routestat_t)
    routestat2: struct_came_routestat_t (struct_came_routestat_t)
    routeSpeed: int (POINTER(h))
    field7E: int (POINTER(H))
    field80: int (POINTER(I))
    field84: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    cameEntry: POINTER_T[struct_nkm_came_entry_t]
    routestat1: struct_came_routestat_t
    routestat2: struct_came_routestat_t
    routeSpeed: int
    field7E: int
    field80: int
    field84: int

class struct_camera_t(Structure):
    """
    ```python
    up: Float[Tensor, "3"] (struct_VecFx32)
    right: Float[Tensor, "3"] (struct_VecFx32)
    target: Float[Tensor, "3"] (struct_VecFx32)
    position: Float[Tensor, "3"] (struct_VecFx32)
    mtx: Float[Tensor, "4 3"] (union_MtxFx43)
    fov: int (POINTER(i))
    targetFov: int (POINTER(i))
    fovSin: float (struct_fx16)
    fovCos: float (struct_fx16)
    aspectRatio: float (struct_fx32)
    frustumNear: float (struct_fx32)
    frustumFar: float (struct_fx32)
    frustumTop: float (struct_fx32)
    frustumBottom: float (struct_fx32)
    frustumLeft: float (struct_fx32)
    frustumRight: float (struct_fx32)
    field88: float (struct_fx32)
    skyFrustumFar: float (struct_fx32)
    lookAtTarget: Float[Tensor, "3"] (struct_VecFx32)
    lookAtPosition: Float[Tensor, "3"] (struct_VecFx32)
    fieldA8: Float[Tensor, "3"] (struct_VecFx32)
    fieldB4: Float[Tensor, "3"] (struct_VecFx32)
    upConst: Float[Tensor, "3"] (struct_VecFx32)
    fieldCC: float (struct_fx32)
    fieldD0: int (POINTER(i))
    targetElevation: float (struct_fx32)
    fieldD8: int (POINTER(I))
    fieldDC: int (POINTER(I))
    fieldE0: int (POINTER(I))
    fieldE4: Float[Tensor, "3"] (struct_VecFx32)
    playerOffsetDirection: float (struct_fx32)
    fieldF4: Float[Tensor, "3"] (struct_VecFx32)
    field100: Float[Tensor, "3"] (struct_VecFx32)
    field10C: Float[Tensor, "3"] (struct_VecFx32)
    field118: Float[Tensor, "3"] (struct_VecFx32)
    field124: Float[Tensor, "3"] (struct_VecFx32)
    field130: int (POINTER(B))
    PADDING_0: list[int] (POINTER(B)[3])
    prevPosition: Float[Tensor, "3"] (struct_VecFx32)
    isShaking: int (POINTER(i))
    field144: float (struct_fx32)
    shakeAmount: float (struct_fx32)
    field14C: int (POINTER(I))
    field150: int (POINTER(h))
    PADDING_1: list[int] (POINTER(B)[2])
    shakeDecay: float (struct_fx32)
    field158: int (POINTER(I))
    targetDirection: Float[Tensor, "3"] (struct_VecFx32)
    field168: float (struct_fx32)
    field16C: int (POINTER(I))
    field170: int (POINTER(I))
    field174: int (POINTER(I))
    elevation: float (struct_fx32)
    field17C: Float[Tensor, "3"] (struct_VecFx32)
    field188: Float[Tensor, "3"] (struct_VecFx32)
    routeStat: struct_came_routestat_t (struct_came_routestat_t)
    routeStat2: struct_came_routestat_t (struct_came_routestat_t)
    field20C: int (POINTER(H))
    field20E: int (POINTER(H))
    targetDriverId: int (POINTER(H))
    PADDING_2: list[int] (POINTER(B)[2])
    currentCamId: int (POINTER(I))
    cameEntry: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    unknownMgCams: POINTER_T[struct_came_unknown_t] (POINTER(struct_came_unknown_t))
    unknownMgCamsCopy: POINTER_T[struct_came_unknown_t] (POINTER(struct_came_unknown_t))
    field224: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    field228: int (POINTER(I))
    field22C: int (POINTER(H))
    PADDING_4: list[int] (POINTER(B)[2])
    field230: int (POINTER(I))
    field234: int (POINTER(I))
    field238: int (POINTER(i))
    frameCounter: int (POINTER(H))
    PADDING_5: list[int] (POINTER(B)[2])
    fovProgress: float (struct_fx32)
    targetProgress: float (struct_fx32)
    field248: int (POINTER(I))
    mode: int (POINTER(I))
    field250: int (POINTER(I))
    field254: int (POINTER(I))
    field258: int (POINTER(i))
    field25C: int (POINTER(I))
    field260: Float[Tensor, "3"] (struct_VecFx32)
    field26C: int (POINTER(h))
    field26E: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    up: Float[Tensor, "3"]
    right: Float[Tensor, "3"]
    target: Float[Tensor, "3"]
    position: Float[Tensor, "3"]
    mtx: Float[Tensor, "4 3"]
    fov: int
    targetFov: int
    fovSin: float
    fovCos: float
    aspectRatio: float
    frustumNear: float
    frustumFar: float
    frustumTop: float
    frustumBottom: float
    frustumLeft: float
    frustumRight: float
    field88: float
    skyFrustumFar: float
    lookAtTarget: Float[Tensor, "3"]
    lookAtPosition: Float[Tensor, "3"]
    fieldA8: Float[Tensor, "3"]
    fieldB4: Float[Tensor, "3"]
    upConst: Float[Tensor, "3"]
    fieldCC: float
    fieldD0: int
    targetElevation: float
    fieldD8: int
    fieldDC: int
    fieldE0: int
    fieldE4: Float[Tensor, "3"]
    playerOffsetDirection: float
    fieldF4: Float[Tensor, "3"]
    field100: Float[Tensor, "3"]
    field10C: Float[Tensor, "3"]
    field118: Float[Tensor, "3"]
    field124: Float[Tensor, "3"]
    field130: int
    PADDING_0: list[int]
    prevPosition: Float[Tensor, "3"]
    isShaking: int
    field144: float
    shakeAmount: float
    field14C: int
    field150: int
    PADDING_1: list[int]
    shakeDecay: float
    field158: int
    targetDirection: Float[Tensor, "3"]
    field168: float
    field16C: int
    field170: int
    field174: int
    elevation: float
    field17C: Float[Tensor, "3"]
    field188: Float[Tensor, "3"]
    routeStat: struct_came_routestat_t
    routeStat2: struct_came_routestat_t
    field20C: int
    field20E: int
    targetDriverId: int
    PADDING_2: list[int]
    currentCamId: int
    cameEntry: POINTER_T[struct_nkm_came_entry_t]
    unknownMgCams: POINTER_T[struct_came_unknown_t]
    unknownMgCamsCopy: POINTER_T[struct_came_unknown_t]
    field224: int
    PADDING_3: list[int]
    field228: int
    field22C: int
    PADDING_4: list[int]
    field230: int
    field234: int
    field238: int
    frameCounter: int
    PADDING_5: list[int]
    fovProgress: float
    targetProgress: float
    field248: int
    mode: int
    field250: int
    field254: int
    field258: int
    field25C: int
    field260: Float[Tensor, "3"]
    field26C: int
    field26E: int

class struct_charkart_colors_t(Structure):
    """
    ```python
    diffuse: int (POINTER(H))
    emission: int (POINTER(H))
    ambient: int (POINTER(H))
    diffR: int (POINTER(H))
    diffG: int (POINTER(H))
    diffB: int (POINTER(H))
    diffRDelta: int (POINTER(h))
    diffGDelta: int (POINTER(h))
    diffBDelta: int (POINTER(h))
    emiR: int (POINTER(H))
    emiG: int (POINTER(H))
    emiB: int (POINTER(H))
    emiRDelta: int (POINTER(h))
    emiGDelta: int (POINTER(h))
    emiBDelta: int (POINTER(h))
    ambiR: int (POINTER(H))
    ambiG: int (POINTER(H))
    amibB: int (POINTER(H))
    amibRDelta: int (POINTER(h))
    ambiGDelta: int (POINTER(h))
    ambiBDelta: int (POINTER(h))
    progress: float (struct_fx16)
    ```
    """
    _pack_: ClassVar[int] = 1
    diffuse: int
    emission: int
    ambient: int
    diffR: int
    diffG: int
    diffB: int
    diffRDelta: int
    diffGDelta: int
    diffBDelta: int
    emiR: int
    emiG: int
    emiB: int
    emiRDelta: int
    emiGDelta: int
    emiBDelta: int
    ambiR: int
    ambiG: int
    amibB: int
    amibRDelta: int
    ambiGDelta: int
    ambiBDelta: int
    progress: float

class struct_charkart_field24_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(i))
    field8: int (POINTER(i))
    fieldC: int (POINTER(i))
    field10: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    field14: int (POINTER(i))
    field18: int (POINTER(H))
    field1A: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    field8: int
    fieldC: int
    field10: int
    PADDING_0: list[int]
    field14: int
    field18: int
    field1A: int

class struct_charkart_t(Structure):
    """
    ```python
    characterId: int (POINTER(I))
    kartId: int (POINTER(i))
    characterNsbcaAnim: POINTER_T[struct_anim_manager_t] (POINTER(struct_anim_manager_t))
    characterNsbtpAnim: POINTER_T[struct_anim_manager_t] (POINTER(struct_anim_manager_t))
    characterModel: POINTER_T[struct_model_t] (POINTER(struct_model_t))
    kartModel: POINTER_T[struct_model_t] (POINTER(struct_model_t))
    kartTireModel: POINTER_T[struct_model_t] (POINTER(struct_model_t))
    kartShadowModel: POINTER_T[struct_model_t] (POINTER(struct_model_t))
    kartOffsetData: POINTER_T[struct_kofs_entry_t] (POINTER(struct_kofs_entry_t))
    field24: struct_charkart_field24_t (struct_charkart_field24_t)
    light: struct_light_t (struct_light_t)
    PADDING_0: list[int] (POINTER(B)[2])
    field54: int (POINTER(I))
    isKartInvisible: int (POINTER(i))
    isCharacterInvisible: int (POINTER(i))
    useSeparateTires: int (POINTER(i))
    inStarToonMode: int (POINTER(i))
    kartABC: int (POINTER(H))
    colors: struct_charkart_colors_t (struct_charkart_colors_t)
    PADDING_1: list[int] (POINTER(B)[2])
    field98: struct_anim_animator_t (struct_anim_animator_t)
    nsbtpAnimDisabled: int (POINTER(i))
    fieldB0: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    characterId: int
    kartId: int
    characterNsbcaAnim: POINTER_T[struct_anim_manager_t]
    characterNsbtpAnim: POINTER_T[struct_anim_manager_t]
    characterModel: POINTER_T[struct_model_t]
    kartModel: POINTER_T[struct_model_t]
    kartTireModel: POINTER_T[struct_model_t]
    kartShadowModel: POINTER_T[struct_model_t]
    kartOffsetData: POINTER_T[struct_kofs_entry_t]
    field24: struct_charkart_field24_t
    light: struct_light_t
    PADDING_0: list[int]
    field54: int
    isKartInvisible: int
    isCharacterInvisible: int
    useSeparateTires: int
    inStarToonMode: int
    kartABC: int
    colors: struct_charkart_colors_t
    PADDING_1: list[int]
    field98: struct_anim_animator_t
    nsbtpAnimDisabled: int
    fieldB0: int

class struct_driver_field450_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(B))
    PADDING_0: list[int] (POINTER(B)[3])
    field8: float (struct_fx32)
    fieldC: int (POINTER(i))
    field10: int (POINTER(i))
    field14: int (POINTER(i))
    field18: float (struct_fx32)
    field1C: float (struct_fx32)
    field20: float (struct_fx32)
    field24: float (struct_fx32)
    prevLapProgress: int (POINTER(i))
    kaidanSfxAlternateCounter: int (POINTER(B))
    PADDING_1: list[int] (POINTER(B)[3])
    field30: int (POINTER(i))
    field34: int (POINTER(i))
    sfxId: int (POINTER(i))
    computePitchOffsetFunc: Callable[[POINTER_T[struct_sfx_emitter_ex_params_t]], c_int] (CFunctionType)
    field40: struct_struc_334 (struct_struc_334)
    field68: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    PADDING_0: list[int]
    field8: float
    fieldC: int
    field10: int
    field14: int
    field18: float
    field1C: float
    field20: float
    field24: float
    prevLapProgress: int
    kaidanSfxAlternateCounter: int
    PADDING_1: list[int]
    field30: int
    field34: int
    sfxId: int
    computePitchOffsetFunc: Callable[[POINTER_T[struct_sfx_emitter_ex_params_t]], c_int]
    field40: struct_struc_334
    field68: int

class struct_driver_net_state_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    fieldC: int (POINTER(H))
    fieldE: int (POINTER(H))
    field10: Float[Tensor, "3"] (struct_VecFx32)
    field1C: int (POINTER(i))
    field20: int (POINTER(i))
    field24: float (struct_fx32)
    field28: int (POINTER(i))
    flags: int (POINTER(i))
    lastFlags: int (POINTER(i))
    field34: Float[Tensor, "3"] (struct_VecFx32)
    driftRotY: int (POINTER(h))
    PADDING_0: list[int] (POINTER(B)[2])
    field44: int (POINTER(i))
    field48: int (POINTER(I))
    field4C: Float[Tensor, "4"] (struct_quaternion_t)
    field5C: int (POINTER(h))
    PADDING_1: list[int] (POINTER(B)[2])
    field60: int (POINTER(i))
    field64: int (POINTER(i))
    gap68: list[int] (POINTER(B)[24])
    field80: struct_NNSFndList (struct_NNSFndList)
    field8C: struct_NNSFndList (struct_NNSFndList)
    field98: Float[Tensor, "3"] (struct_VecFx32)
    fieldA4: int (POINTER(I))
    fieldA8: Float[Tensor, "3"] (struct_VecFx32)
    fieldB4: int (POINTER(H))
    gapB6: list[int] (POINTER(B)[1])
    fieldB7: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    fieldC: int
    fieldE: int
    field10: Float[Tensor, "3"]
    field1C: int
    field20: int
    field24: float
    field28: int
    flags: int
    lastFlags: int
    field34: Float[Tensor, "3"]
    driftRotY: int
    PADDING_0: list[int]
    field44: int
    field48: int
    field4C: Float[Tensor, "4"]
    field5C: int
    PADDING_1: list[int]
    field60: int
    field64: int
    gap68: list[int]
    field80: struct_NNSFndList
    field8C: struct_NNSFndList
    field98: Float[Tensor, "3"]
    fieldA4: int
    fieldA8: Float[Tensor, "3"]
    fieldB4: int
    gapB6: list[int]
    fieldB7: int

class struct_driver_t(Structure):
    """
    ```python
    soundEmitter: struct_sfx_emitter_t (struct_sfx_emitter_t)
    field44: int (POINTER(I))
    flags: int (POINTER(I))
    flags2: int (POINTER(I))
    direction: Float[Tensor, "3"] (struct_VecFx32)
    drivingDirection: Float[Tensor, "3"] (struct_VecFx32)
    velocity: Float[Tensor, "3"] (struct_VecFx32)
    id: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    inputId: int (POINTER(I))
    field7C: int (POINTER(I))
    position: Float[Tensor, "3"] (struct_VecFx32)
    lastPosition: Float[Tensor, "3"] (struct_VecFx32)
    kartTiresPosition: Float[Tensor, "3"] (struct_VecFx32)
    deltaPos: Float[Tensor, "3"] (struct_VecFx32)
    deltaPosNormalized: Float[Tensor, "3"] (struct_VecFx32)
    scale: Float[Tensor, "3"] (struct_VecFx32)
    fieldC8: float (struct_fx32)
    targetMaxSpeed: float (struct_fx32)
    maxSpeed: float (struct_fx32)
    fieldD4: int (POINTER(I))
    slipstreamSpeedMultiplier: float (struct_fx32)
    speedMultiplier: float (struct_fx32)
    rotation: Float[Tensor, "4"] (struct_quaternion_t)
    fieldF0: Float[Tensor, "4"] (struct_quaternion_t)
    field100: Float[Tensor, "4"] (struct_quaternion_t)
    field110: Float[Tensor, "4"] (struct_quaternion_t)
    mainMtx: Float[Tensor, "4 3"] (union_MtxFx43)
    field150: Float[Tensor, "4 3"] (union_MtxFx43)
    colReaction: int (POINTER(I))
    field184: Float[Tensor, "4 3"] (union_MtxFx43)
    charKartMtx: int (POINTER(I))
    colPos: Float[Tensor, "3"] (struct_VecFx32)
    prevColPos: Float[Tensor, "3"] (struct_VecFx32)
    colSphereSize: float (struct_fx32)
    colSphereZOffset: float (struct_fx32)
    netColPos: Float[Tensor, "3"] (struct_VecFx32)
    lastNetColPos: Float[Tensor, "3"] (struct_VecFx32)
    colPos2: Float[Tensor, "3"] (struct_VecFx32)
    field1FC: Float[Tensor, "3"] (struct_VecFx32)
    field208: POINTER_T[c_uint] (POINTER(POINTER(I)))
    field20C: list[c_void_p] (c_void_p[9])
    field230: Callable[[POINTER_T[struct_driver_t]], None] (CFunctionType)
    xRot: int (POINTER(h))
    yRot: int (POINTER(H))
    boostTimer: int (POINTER(h))
    field23A: int (POINTER(h))
    driftBoostCounter: int (POINTER(h))
    PADDING_1: list[int] (POINTER(B)[2])
    velocityMinusDirMultiplier: float (struct_fx32)
    upDir: Float[Tensor, "3"] (struct_VecFx32)
    field250: Float[Tensor, "3"] (struct_VecFx32)
    velocityY: Float[Tensor, "3"] (struct_VecFx32)
    fallsWaterForward: Float[Tensor, "3"] (struct_VecFx32)
    fallsWaterStrength: float (struct_fx32)
    forwardDir: Float[Tensor, "3"] (struct_VecFx32)
    jumpDriftUp: Float[Tensor, "3"] (struct_VecFx32)
    jumpDriftForward: Float[Tensor, "3"] (struct_VecFx32)
    collisionMode: int (POINTER(I))
    maxSpeedFraction: float (struct_fx32)
    deltaPosMag: float (struct_fx32)
    speed: float (struct_fx32)
    field2AC: float (struct_fx32)
    driverHitCheckMask: int (POINTER(H))
    driverHitMask: int (POINTER(H))
    lastDriverHitMask: int (POINTER(H))
    gap2B6: list[int] (POINTER(B)[2])
    field2B8: int (POINTER(i))
    field2BC: int (POINTER(i))
    field2C0: int (POINTER(H))
    PADDING_2: list[int] (POINTER(B)[2])
    leftRightDir: float (struct_fx32)
    colEntryId1: int (POINTER(h))
    colEntryId2: int (POINTER(h))
    kartPhysicalParams: POINTER_T[struct_physp_kart_params_t] (POINTER(struct_physp_kart_params_t))
    charPhysicalParams: POINTER_T[struct_physp_char_params_t] (POINTER(struct_physp_char_params_t))
    turningAmount: float (struct_fx32)
    field2D8: Float[Tensor, "3"] (struct_VecFx32)
    field2E4: Float[Tensor, "3"] (struct_VecFx32)
    field2F0: Float[Tensor, "3"] (struct_VecFx32)
    driftLeftRightCount: int (POINTER(i))
    driftLeftCount: int (POINTER(H))
    driftRightCount: int (POINTER(H))
    driftDir1CountPtr: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    driftDir2CountPtr: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    driftLeftRightTimeout: int (POINTER(i))
    enemyState: POINTER_T[struct_enemy_t] (POINTER(struct_enemy_t))
    field314: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    field318: float (struct_fx32)
    field31C: Float[Tensor, "3"] (struct_VecFx32)
    field328: Float[Tensor, "3"] (struct_VecFx32)
    field334: int (POINTER(I))
    field338: int (POINTER(H))
    PADDING_4: list[int] (POINTER(B)[2])
    field33C: int (POINTER(I))
    field340: float (struct_fx32)
    field344: int (POINTER(I))
    field348: int (POINTER(I))
    field34C: Float[Tensor, "4"] (struct_quaternion_t)
    colReactionCounter: int (POINTER(h))
    PADDING_5: list[int] (POINTER(B)[2])
    field360: float (struct_fx32)
    spinOutAngle: int (POINTER(H))
    spinOutSpinCount: int (POINTER(H))
    spinOutProgress: float (struct_fx32)
    spinOutVelocity: int (POINTER(I))
    field370: int (POINTER(H))
    PADDING_6: list[int] (POINTER(B)[2])
    field374: Float[Tensor, "3"] (struct_VecFx32)
    field380: int (POINTER(I))
    ghostFlickerPhase: int (POINTER(H))
    wallRotYSpeed: int (POINTER(h))
    driftRotY: int (POINTER(h))
    extraDrift: float (struct_fx16)
    field38C: float (struct_fx32)
    gap390: list[int] (POINTER(B)[4])
    field394: int (POINTER(I))
    field398: float (struct_fx32)
    field39C: float (struct_fx32)
    field3A0: float (struct_fx32)
    tireRotX: int (POINTER(H))
    PADDING_7: list[int] (POINTER(B)[2])
    field3A8: int (POINTER(i))
    respawnCounter: int (POINTER(H))
    PADDING_8: list[int] (POINTER(B)[2])
    field3B0: Float[Tensor, "3"] (struct_VecFx32)
    field3BC: int (POINTER(H))
    field3BE: int (POINTER(h))
    preRespawnCounter: int (POINTER(h))
    PADDING_9: list[int] (POINTER(B)[2])
    respawnId: int (POINTER(I))
    killTimer: int (POINTER(h))
    PADDING_10: list[int] (POINTER(B)[2])
    driverVoiceIdx: int (POINTER(I))
    kartABC: int (POINTER(h))
    field3D2: int (POINTER(h))
    field3D4: int (POINTER(h))
    PADDING_11: list[int] (POINTER(B)[2])
    place: int (POINTER(i))
    floorDriverColType: int (POINTER(I))
    floorColType: int (POINTER(I))
    floorColVariant: int (POINTER(i))
    field3E8: int (POINTER(h))
    PADDING_12: list[int] (POINTER(B)[2])
    yRotSpeedTarget: int (POINTER(I))
    yRotSpeed: int (POINTER(I))
    field3F4: float (struct_fx32)
    field3F8: float (struct_fx32)
    field3FC: int (POINTER(H))
    field3FE: int (POINTER(H))
    field400: int (POINTER(H))
    PADDING_13: list[int] (POINTER(B)[2])
    field404: float (struct_fx32)
    field408: int (POINTER(I))
    respawnStartFrame: int (POINTER(I))
    respawnAPressFrame: int (POINTER(I))
    field414: float (struct_fx32)
    field418: float (struct_fx32)
    growBackScale: Float[Tensor, "3"] (struct_VecFx32)
    thunderScale: Float[Tensor, "3"] (struct_VecFx32)
    dossunYScale: float (struct_fx32)
    mobjHitList: list[POINTER_T[struct_mobj_inst_t]] (POINTER(struct_mobj_inst_t)[2])
    mobjHitSfxTimeout: list[int] (POINTER(H)[2])
    mobjHitEmittedSfx: list[int] (POINTER(i)[2])
    smashDossun: POINTER_T[struct_mobj_inst_t] (POINTER(struct_mobj_inst_t))
    field450: struct_driver_field450_t (struct_driver_field450_t)
    field4BC: float (struct_fx32)
    colFlagsMap2DShadow: int (POINTER(I))
    jumpPadSpeed: int (POINTER(I))
    field4C8: float (struct_fx32)
    field4CC: int (POINTER(I))
    field4D0: int (POINTER(I))
    preStartEnginePower: float (struct_fx32)
    fallsWaterDstId: int (POINTER(h))
    wallTouchTimeout: int (POINTER(h))
    floorTouchTimeout: int (POINTER(h))
    field4DE: int (POINTER(h))
    field4E0: int (POINTER(h))
    field4E2: int (POINTER(h))
    field4E4: int (POINTER(H))
    field4E6: int (POINTER(H))
    field4E8: float (struct_fx32)
    field4EC: float (struct_fx32)
    idkScale: Float[Tensor, "3"] (struct_VecFx32)
    field4FC: int (POINTER(H))
    PADDING_14: list[int] (POINTER(B)[2])
    waterDepth: float (struct_fx32)
    field504: int (POINTER(H))
    field506: int (POINTER(H))
    field508: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    field50C: POINTER_T[struct_quaternion_t] (POINTER(struct_quaternion_t))
    field510: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    netState: POINTER_T[struct_driver_net_state_t] (POINTER(struct_driver_net_state_t))
    field518: struct_sfx_emitter_ex_params_t (struct_sfx_emitter_ex_params_t)
    field534: c_void_p (c_void_p)
    timers: struct_driver_timers_t (struct_driver_timers_t)
    charKart: POINTER_T[struct_charkart_t] (POINTER(struct_charkart_t))
    field594: float (struct_fx32)
    field598: int (POINTER(h))
    PADDING_15: list[int] (POINTER(B)[2])
    field59C: int (POINTER(I))
    field5A0: int (POINTER(H))
    gap5A2: list[int] (POINTER(B)[2])
    field5A4: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    soundEmitter: struct_sfx_emitter_t
    field44: int
    flags: int
    flags2: int
    direction: Float[Tensor, "3"]
    drivingDirection: Float[Tensor, "3"]
    velocity: Float[Tensor, "3"]
    id: int
    PADDING_0: list[int]
    inputId: int
    field7C: int
    position: Float[Tensor, "3"]
    lastPosition: Float[Tensor, "3"]
    kartTiresPosition: Float[Tensor, "3"]
    deltaPos: Float[Tensor, "3"]
    deltaPosNormalized: Float[Tensor, "3"]
    scale: Float[Tensor, "3"]
    fieldC8: float
    targetMaxSpeed: float
    maxSpeed: float
    fieldD4: int
    slipstreamSpeedMultiplier: float
    speedMultiplier: float
    rotation: Float[Tensor, "4"]
    fieldF0: Float[Tensor, "4"]
    field100: Float[Tensor, "4"]
    field110: Float[Tensor, "4"]
    mainMtx: Float[Tensor, "4 3"]
    field150: Float[Tensor, "4 3"]
    colReaction: int
    field184: Float[Tensor, "4 3"]
    charKartMtx: int
    colPos: Float[Tensor, "3"]
    prevColPos: Float[Tensor, "3"]
    colSphereSize: float
    colSphereZOffset: float
    netColPos: Float[Tensor, "3"]
    lastNetColPos: Float[Tensor, "3"]
    colPos2: Float[Tensor, "3"]
    field1FC: Float[Tensor, "3"]
    field208: POINTER_T[c_uint]
    field20C: list[c_void_p]
    field230: Callable[[POINTER_T[struct_driver_t]], None]
    xRot: int
    yRot: int
    boostTimer: int
    field23A: int
    driftBoostCounter: int
    PADDING_1: list[int]
    velocityMinusDirMultiplier: float
    upDir: Float[Tensor, "3"]
    field250: Float[Tensor, "3"]
    velocityY: Float[Tensor, "3"]
    fallsWaterForward: Float[Tensor, "3"]
    fallsWaterStrength: float
    forwardDir: Float[Tensor, "3"]
    jumpDriftUp: Float[Tensor, "3"]
    jumpDriftForward: Float[Tensor, "3"]
    collisionMode: int
    maxSpeedFraction: float
    deltaPosMag: float
    speed: float
    field2AC: float
    driverHitCheckMask: int
    driverHitMask: int
    lastDriverHitMask: int
    gap2B6: list[int]
    field2B8: int
    field2BC: int
    field2C0: int
    PADDING_2: list[int]
    leftRightDir: float
    colEntryId1: int
    colEntryId2: int
    kartPhysicalParams: POINTER_T[struct_physp_kart_params_t]
    charPhysicalParams: POINTER_T[struct_physp_char_params_t]
    turningAmount: float
    field2D8: Float[Tensor, "3"]
    field2E4: Float[Tensor, "3"]
    field2F0: Float[Tensor, "3"]
    driftLeftRightCount: int
    driftLeftCount: int
    driftRightCount: int
    driftDir1CountPtr: POINTER_T[c_ushort]
    driftDir2CountPtr: POINTER_T[c_ushort]
    driftLeftRightTimeout: int
    enemyState: POINTER_T[struct_enemy_t]
    field314: int
    PADDING_3: list[int]
    field318: float
    field31C: Float[Tensor, "3"]
    field328: Float[Tensor, "3"]
    field334: int
    field338: int
    PADDING_4: list[int]
    field33C: int
    field340: float
    field344: int
    field348: int
    field34C: Float[Tensor, "4"]
    colReactionCounter: int
    PADDING_5: list[int]
    field360: float
    spinOutAngle: int
    spinOutSpinCount: int
    spinOutProgress: float
    spinOutVelocity: int
    field370: int
    PADDING_6: list[int]
    field374: Float[Tensor, "3"]
    field380: int
    ghostFlickerPhase: int
    wallRotYSpeed: int
    driftRotY: int
    extraDrift: float
    field38C: float
    gap390: list[int]
    field394: int
    field398: float
    field39C: float
    field3A0: float
    tireRotX: int
    PADDING_7: list[int]
    field3A8: int
    respawnCounter: int
    PADDING_8: list[int]
    field3B0: Float[Tensor, "3"]
    field3BC: int
    field3BE: int
    preRespawnCounter: int
    PADDING_9: list[int]
    respawnId: int
    killTimer: int
    PADDING_10: list[int]
    driverVoiceIdx: int
    kartABC: int
    field3D2: int
    field3D4: int
    PADDING_11: list[int]
    place: int
    floorDriverColType: int
    floorColType: int
    floorColVariant: int
    field3E8: int
    PADDING_12: list[int]
    yRotSpeedTarget: int
    yRotSpeed: int
    field3F4: float
    field3F8: float
    field3FC: int
    field3FE: int
    field400: int
    PADDING_13: list[int]
    field404: float
    field408: int
    respawnStartFrame: int
    respawnAPressFrame: int
    field414: float
    field418: float
    growBackScale: Float[Tensor, "3"]
    thunderScale: Float[Tensor, "3"]
    dossunYScale: float
    mobjHitList: list[POINTER_T[struct_mobj_inst_t]]
    mobjHitSfxTimeout: list[int]
    mobjHitEmittedSfx: list[int]
    smashDossun: POINTER_T[struct_mobj_inst_t]
    field450: struct_driver_field450_t
    field4BC: float
    colFlagsMap2DShadow: int
    jumpPadSpeed: int
    field4C8: float
    field4CC: int
    field4D0: int
    preStartEnginePower: float
    fallsWaterDstId: int
    wallTouchTimeout: int
    floorTouchTimeout: int
    field4DE: int
    field4E0: int
    field4E2: int
    field4E4: int
    field4E6: int
    field4E8: float
    field4EC: float
    idkScale: Float[Tensor, "3"]
    field4FC: int
    PADDING_14: list[int]
    waterDepth: float
    field504: int
    field506: int
    field508: POINTER_T[struct_VecFx32]
    field50C: POINTER_T[struct_quaternion_t]
    field510: POINTER_T[struct_VecFx32]
    netState: POINTER_T[struct_driver_net_state_t]
    field518: struct_sfx_emitter_ex_params_t
    field534: c_void_p
    timers: struct_driver_timers_t
    charKart: POINTER_T[struct_charkart_t]
    field594: float
    field598: int
    PADDING_15: list[int]
    field59C: int
    field5A0: int
    gap5A2: list[int]
    field5A4: float

class struct_driver_timers_t(Structure):
    """
    ```python
    shroomBoostTimer: int (POINTER(h))
    thunderShrinkTimer: int (POINTER(h))
    thunderGrowTimer: int (POINTER(h))
    starTimer: int (POINTER(h))
    slipstreamStartTimer: int (POINTER(h))
    slipstreamTimer: int (POINTER(h))
    dossunGrowTimer: int (POINTER(h))
    dossunFlatTimer: int (POINTER(h))
    teresaTimer: int (POINTER(h))
    teresaFlickerInterval: int (POINTER(h))
    teresaFlickerIntervalUpdateTimer: int (POINTER(h))
    teresaFlickerTimer: int (POINTER(h))
    teresaFlickerIntervalUpdateWaitTime: int (POINTER(h))
    teresaFlickerIntervalStep: int (POINTER(h))
    gessoInkTimer: int (POINTER(h))
    killerFrameCounter: int (POINTER(h))
    field20: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    isKillerFinishing: int (POINTER(i))
    killerTargetPlace: int (POINTER(h))
    gap2A: list[int] (POINTER(B)[2])
    killerState: list[int] (POINTER(B)[44])
    ```
    """
    _pack_: ClassVar[int] = 1
    shroomBoostTimer: int
    thunderShrinkTimer: int
    thunderGrowTimer: int
    starTimer: int
    slipstreamStartTimer: int
    slipstreamTimer: int
    dossunGrowTimer: int
    dossunFlatTimer: int
    teresaTimer: int
    teresaFlickerInterval: int
    teresaFlickerIntervalUpdateTimer: int
    teresaFlickerTimer: int
    teresaFlickerIntervalUpdateWaitTime: int
    teresaFlickerIntervalStep: int
    gessoInkTimer: int
    killerFrameCounter: int
    field20: int
    PADDING_0: list[int]
    isKillerFinishing: int
    killerTargetPlace: int
    gap2A: list[int]
    killerState: list[int]

class struct_enemy_field140_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    driverFieldCC: int (POINTER(i))
    field8: int (POINTER(i))
    fieldC: int (POINTER(i))
    field10: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    driverFieldCC: int
    field8: int
    fieldC: int
    field10: int

class struct_enemy_item_params_t(Structure):
    """
    ```python
    updateFunc: c_void_p (c_void_p)
    field4: int (POINTER(i))
    minTimeout: int (POINTER(H))
    timeoutRandomMax: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    updateFunc: c_void_p
    field4: int
    minTimeout: int
    timeoutRandomMax: int

class struct_enemy_item_state_t(Structure):
    """
    ```python
    slotItemTimeout: int (POINTER(H))
    dragItemTimeout: int (POINTER(H))
    slotItemParams: POINTER_T[struct_enemy_item_params_t] (POINTER(struct_enemy_item_params_t))
    dragItemParams: POINTER_T[struct_enemy_item_params_t] (POINTER(struct_enemy_item_params_t))
    fieldC: int (POINTER(i))
    lxPressed: int (POINTER(i))
    dpadUpPressed: int (POINTER(i))
    dpadDownPressed: int (POINTER(i))
    dpadUpCounter: int (POINTER(H))
    dpadDownCounter: int (POINTER(H))
    field20: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    slotItemTimeout: int
    dragItemTimeout: int
    slotItemParams: POINTER_T[struct_enemy_item_params_t]
    dragItemParams: POINTER_T[struct_enemy_item_params_t]
    fieldC: int
    lxPressed: int
    dpadUpPressed: int
    dpadDownPressed: int
    dpadUpCounter: int
    dpadDownCounter: int
    field20: int

class struct_enemy_rubberbanding_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    driverFieldCC: int (POINTER(i))
    rivalAggressiveness: int (POINTER(i))
    maxDriverFieldCC: int (POINTER(i))
    field10: int (POINTER(i))
    place: int (POINTER(i))
    field18: int (POINTER(i))
    field1C: int (POINTER(i))
    field20: int (POINTER(i))
    field24: int (POINTER(i))
    hasFailedStart: int (POINTER(i))
    hasStartBoost: int (POINTER(i))
    startBoostAmount: int (POINTER(H))
    field32: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    driverFieldCC: int
    rivalAggressiveness: int
    maxDriverFieldCC: int
    field10: int
    place: int
    field18: int
    field1C: int
    field20: int
    field24: int
    hasFailedStart: int
    hasStartBoost: int
    startBoostAmount: int
    field32: int

class struct_enemy_t(Structure):
    """
    ```python
    driver: POINTER_T[struct_driver_t] (POINTER(struct_driver_t))
    driverId: int (POINTER(H))
    field6: int (POINTER(H))
    epoi: struct_struc_316_epoi (struct_struc_316_epoi)
    mepo: struct_struc_313_mepo (struct_struc_313_mepo)
    targetPos: Float[Tensor, "3"] (struct_VecFx32)
    field50: Float[Tensor, "3"] (struct_VecFx32)
    field5C: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    driftOffset: Float[Tensor, "3"] (struct_VecFx32)
    driftEpoiRadiusScaleUpdateCounter: int (POINTER(H))
    driftEpoiRadiusScaleUpdateFrames: int (POINTER(H))
    field70: int (POINTER(i))
    driftEpoiRadiusScale: float (struct_fx32)
    field78: int (POINTER(H))
    field7A: int (POINTER(H))
    field7C: int (POINTER(i))
    field80: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    field84: int (POINTER(i))
    field88: int (POINTER(i))
    field8C: int (POINTER(i))
    driftState: int (POINTER(i))
    driftDirection: int (POINTER(i))
    field98: int (POINTER(i))
    field9C: int (POINTER(i))
    fieldA0: int (POINTER(i))
    fieldA4: int (POINTER(i))
    fieldA8: int (POINTER(i))
    fieldAC: int (POINTER(H))
    PADDING_1: list[int] (POINTER(B)[2])
    targetDriver: POINTER_T[struct_driver_t] (POINTER(struct_driver_t))
    fieldB4: int (POINTER(i))
    fieldB8: int (POINTER(i))
    fieldBC: int (POINTER(i))
    targetBalloonCount: int (POINTER(H))
    balloonInflateMicTimeout: int (POINTER(H))
    isInflatingBalloon: int (POINTER(i))
    fieldC8: int (POINTER(H))
    fieldCA: int (POINTER(H))
    fieldCC: int (POINTER(i))
    targetShine: c_void_p (c_void_p)
    shineIdx: int (POINTER(H))
    PADDING_2: list[int] (POINTER(B)[2])
    fieldD8: int (POINTER(i))
    fieldDC: Float[Tensor, "3"] (struct_VecFx32)
    rubberbanding: struct_enemy_rubberbanding_t (struct_enemy_rubberbanding_t)
    itemState: struct_enemy_item_state_t (struct_enemy_item_state_t)
    field140: struct_enemy_field140_t (struct_enemy_field140_t)
    field154: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    driver: POINTER_T[struct_driver_t]
    driverId: int
    field6: int
    epoi: struct_struc_316_epoi
    mepo: struct_struc_313_mepo
    targetPos: Float[Tensor, "3"]
    field50: Float[Tensor, "3"]
    field5C: POINTER_T[struct_VecFx32]
    driftOffset: Float[Tensor, "3"]
    driftEpoiRadiusScaleUpdateCounter: int
    driftEpoiRadiusScaleUpdateFrames: int
    field70: int
    driftEpoiRadiusScale: float
    field78: int
    field7A: int
    field7C: int
    field80: int
    PADDING_0: list[int]
    field84: int
    field88: int
    field8C: int
    driftState: int
    driftDirection: int
    field98: int
    field9C: int
    fieldA0: int
    fieldA4: int
    fieldA8: int
    fieldAC: int
    PADDING_1: list[int]
    targetDriver: POINTER_T[struct_driver_t]
    fieldB4: int
    fieldB8: int
    fieldBC: int
    targetBalloonCount: int
    balloonInflateMicTimeout: int
    isInflatingBalloon: int
    fieldC8: int
    fieldCA: int
    fieldCC: int
    targetShine: c_void_p
    shineIdx: int
    PADDING_2: list[int]
    fieldD8: int
    fieldDC: Float[Tensor, "3"]
    rubberbanding: struct_enemy_rubberbanding_t
    itemState: struct_enemy_item_state_t
    field140: struct_enemy_field140_t
    field154: int

class struct_fx16(Structure):
    """
    ```python
    val: int (POINTER(h))
    ```
    """
    _pack_: ClassVar[int] = 1
    val: int

class struct_fx32(Structure):
    """
    ```python
    val: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    val: int

class struct_it_item_inst_t(Structure):
    """
    ```python
    sfxEmitter: struct_sfx_emitter_t (struct_sfx_emitter_t)
    type: int (POINTER(I))
    field48: int (POINTER(I))
    field4C: int (POINTER(H))
    field4E: int (POINTER(H))
    position: Float[Tensor, "3"] (struct_VecFx32)
    velocity: Float[Tensor, "3"] (struct_VecFx32)
    scale: Float[Tensor, "3"] (struct_VecFx32)
    flags: int (POINTER(I))
    field78: int (POINTER(H))
    field7A: int (POINTER(H))
    light: struct_light_t (struct_light_t)
    PADDING_0: list[int] (POINTER(B)[2])
    lightPtr: POINTER_T[struct_light_t] (POINTER(struct_light_t))
    mtx: Float[Tensor, "4 3"] (union_MtxFx43)
    gapC4: list[int] (POINTER(B)[12])
    fieldD0: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    visibilityFlags: int (POINTER(I))
    alpha: int (POINTER(h))
    colEntryId: int (POINTER(H))
    fieldDC: int (POINTER(I))
    sphereSize: float (struct_fx32)
    fieldE4: Float[Tensor, "3"] (struct_VecFx32)
    fieldF0: Float[Tensor, "3"] (struct_VecFx32)
    fieldFC: int (POINTER(I))
    field100: int (POINTER(I))
    field104: int (POINTER(I))
    field108: Float[Tensor, "3"] (struct_VecFx32)
    field114: int (POINTER(H))
    field116: int (POINTER(H))
    field118: int (POINTER(H))
    field11A: int (POINTER(H))
    field11C: int (POINTER(I))
    field120: int (POINTER(I))
    field124: int (POINTER(I))
    field128DriverMask: int (POINTER(B))
    PADDING_1: list[int] (POINTER(B)[3])
    ```
    """
    _pack_: ClassVar[int] = 1
    sfxEmitter: struct_sfx_emitter_t
    type: int
    field48: int
    field4C: int
    field4E: int
    position: Float[Tensor, "3"]
    velocity: Float[Tensor, "3"]
    scale: Float[Tensor, "3"]
    flags: int
    field78: int
    field7A: int
    light: struct_light_t
    PADDING_0: list[int]
    lightPtr: POINTER_T[struct_light_t]
    mtx: Float[Tensor, "4 3"]
    gapC4: list[int]
    fieldD0: POINTER_T[struct_VecFx32]
    visibilityFlags: int
    alpha: int
    colEntryId: int
    fieldDC: int
    sphereSize: float
    fieldE4: Float[Tensor, "3"]
    fieldF0: Float[Tensor, "3"]
    fieldFC: int
    field100: int
    field104: int
    field108: Float[Tensor, "3"]
    field114: int
    field116: int
    field118: int
    field11A: int
    field11C: int
    field120: int
    field124: int
    field128DriverMask: int
    PADDING_1: list[int]

class struct_kcol_header_t(Structure):
    """
    ```python
    posDataOffset: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    nrmDataOffset: POINTER_T[struct_VecFx16] (POINTER(struct_VecFx16))
    prismDataOffset: POINTER_T[struct_kcol_prism_data_t] (POINTER(struct_kcol_prism_data_t))
    blockDataOffset: POINTER_T[c_uint] (POINTER(POINTER(I)))
    prismThickness: float (struct_fx32)
    areaMinPos: Float[Tensor, "3"] (struct_VecFx32)
    areaXWidthMask: int (POINTER(I))
    areaYWidthMask: int (POINTER(I))
    areaZWidthMask: int (POINTER(I))
    blockWidthShift: int (POINTER(I))
    areaXBlocksShift: int (POINTER(I))
    areaXYBlocksShift: int (POINTER(I))
    sphereRadius: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    posDataOffset: POINTER_T[struct_VecFx32]
    nrmDataOffset: POINTER_T[struct_VecFx16]
    prismDataOffset: POINTER_T[struct_kcol_prism_data_t]
    blockDataOffset: POINTER_T[c_uint]
    prismThickness: float
    areaMinPos: Float[Tensor, "3"]
    areaXWidthMask: int
    areaYWidthMask: int
    areaZWidthMask: int
    blockWidthShift: int
    areaXBlocksShift: int
    areaXYBlocksShift: int
    sphereRadius: float

class struct_kcol_prism_data_t(Structure):
    """
    ```python
    height: float (struct_fx32)
    posIdx: int (POINTER(H))
    fNrmIdx: int (POINTER(H))
    eNrm1Idx: int (POINTER(H))
    eNrm2Idx: int (POINTER(H))
    eNrm3Idx: int (POINTER(H))
    attribute: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    height: float
    posIdx: int
    fNrmIdx: int
    eNrm1Idx: int
    eNrm2Idx: int
    eNrm3Idx: int
    attribute: int

class struct_kofs_entry_t(Structure):
    """
    ```python
    tireName: list[int] (POINTER(B)[16])
    frontTireScale: float (struct_fx32)
    tirePositions: list[Float[Tensor, "3"]] (struct_VecFx32[4])
    characterPositions: list[Float[Tensor, "3"]] (struct_VecFx32[13])
    ```
    """
    _pack_: ClassVar[int] = 1
    tireName: list[int]
    frontTireScale: float
    tirePositions: list[Float[Tensor, "3"]]
    characterPositions: list[Float[Tensor, "3"]]

class struct_light_t(Structure):
    """
    ```python
    color: int (POINTER(H))
    r: int (POINTER(h))
    g: int (POINTER(h))
    b: int (POINTER(h))
    rDelta: int (POINTER(h))
    gDelta: int (POINTER(h))
    bDelta: int (POINTER(h))
    lightMask: int (POINTER(H))
    progress: float (struct_fx16)
    ```
    """
    _pack_: ClassVar[int] = 1
    color: int
    r: int
    g: int
    b: int
    rDelta: int
    gDelta: int
    bDelta: int
    lightMask: int
    progress: float

class struct_list_link_t(Structure):
    """
    ```python
    link: struct_NNSFndLink (struct_NNSFndLink)
    list: POINTER_T[struct_NNSFndList] (POINTER(struct_NNSFndList))
    ```
    """
    _pack_: ClassVar[int] = 1
    link: struct_NNSFndLink
    list: POINTER_T[struct_NNSFndList]

class struct_mdat_clip_area_list_entry_t(Structure):
    """
    ```python
    entry: POINTER_T[struct_nkm_area_entry_t] (POINTER(struct_nkm_area_entry_t))
    next: POINTER_T[struct_mdat_clip_area_list_entry_t] (POINTER(struct_mdat_clip_area_list_entry_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    entry: POINTER_T[struct_nkm_area_entry_t]
    next: POINTER_T[struct_mdat_clip_area_list_entry_t]

class struct_mdat_enemypath_data_t(Structure):
    """
    ```python
    points: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    firstPoint: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    lastPoint: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    points: POINTER_T[struct_mdat_enemypoint_t]
    firstPoint: POINTER_T[struct_mdat_enemypoint_t]
    lastPoint: POINTER_T[struct_mdat_enemypoint_t]

class struct_mdat_enemypoint_t(Structure):
    """
    ```python
    next: list[POINTER_T[struct_mdat_enemypoint_t]] (POINTER(struct_mdat_enemypoint_t)[3])
    previous: list[POINTER_T[struct_mdat_enemypoint_t]] (POINTER(struct_mdat_enemypoint_t)[3])
    position: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    radius: float (struct_fx32)
    settings: POINTER_T[struct_nkm_epoi_entry_settings_t] (POINTER(struct_nkm_epoi_entry_settings_t))
    nextCount: int (POINTER(H))
    previousCount: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    next: list[POINTER_T[struct_mdat_enemypoint_t]]
    previous: list[POINTER_T[struct_mdat_enemypoint_t]]
    position: POINTER_T[struct_VecFx32]
    radius: float
    settings: POINTER_T[struct_nkm_epoi_entry_settings_t]
    nextCount: int
    previousCount: int

class struct_mdat_itempath_data_t(Structure):
    """
    ```python
    points: POINTER_T[struct_mdat_itempoint_t] (POINTER(struct_mdat_itempoint_t))
    firstPoint: POINTER_T[struct_mdat_itempoint_t] (POINTER(struct_mdat_itempoint_t))
    lastPoint: POINTER_T[struct_mdat_itempoint_t] (POINTER(struct_mdat_itempoint_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    points: POINTER_T[struct_mdat_itempoint_t]
    firstPoint: POINTER_T[struct_mdat_itempoint_t]
    lastPoint: POINTER_T[struct_mdat_itempoint_t]

class struct_mdat_itempoint_t(Structure):
    """
    ```python
    next: list[POINTER_T[struct_mdat_itempoint_t]] (POINTER(struct_mdat_itempoint_t)[3])
    previous: list[POINTER_T[struct_mdat_itempoint_t]] (POINTER(struct_mdat_itempoint_t)[3])
    position: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    radius: float (struct_fx32)
    recalcIdx: int (POINTER(B))
    dirX: int (POINTER(b))
    dirY: int (POINTER(b))
    dirZ: int (POINTER(b))
    nextCount: int (POINTER(H))
    previousCount: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    next: list[POINTER_T[struct_mdat_itempoint_t]]
    previous: list[POINTER_T[struct_mdat_itempoint_t]]
    position: POINTER_T[struct_VecFx32]
    radius: float
    recalcIdx: int
    dirX: int
    dirY: int
    dirZ: int
    nextCount: int
    previousCount: int

class struct_mdat_mapdata_t(Structure):
    """
    ```python
    obji: POINTER_T[struct_nkm_obji_entry_t] (POINTER(struct_nkm_obji_entry_t))
    objiCount: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    path: POINTER_T[struct_nkm_path_entry_t] (POINTER(struct_nkm_path_entry_t))
    pathCount: int (POINTER(H))
    PADDING_1: list[int] (POINTER(B)[2])
    poit: POINTER_T[struct_nkm_poit_entry_t] (POINTER(struct_nkm_poit_entry_t))
    poitCount: int (POINTER(H))
    PADDING_2: list[int] (POINTER(B)[2])
    stag: POINTER_T[struct_nkm_stag_data_t] (POINTER(struct_nkm_stag_data_t))
    ktps: POINTER_T[struct_nkm_ktps_entry_t] (POINTER(struct_nkm_ktps_entry_t))
    ktpsCount: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    ktpj: POINTER_T[struct_nkm_ktpj_entry_t] (POINTER(struct_nkm_ktpj_entry_t))
    ktpjCount: int (POINTER(H))
    PADDING_4: list[int] (POINTER(B)[2])
    ktp2: POINTER_T[struct_nkm_ktp2_entry_t] (POINTER(struct_nkm_ktp2_entry_t))
    ktp2Count: int (POINTER(H))
    PADDING_5: list[int] (POINTER(B)[2])
    ktpc: POINTER_T[struct_nkm_ktpc_entry_t] (POINTER(struct_nkm_ktpc_entry_t))
    ktpcCount: int (POINTER(H))
    PADDING_6: list[int] (POINTER(B)[2])
    ktpm: POINTER_T[struct_nkm_ktpm_entry_t] (POINTER(struct_nkm_ktpm_entry_t))
    ktpmCount: int (POINTER(H))
    PADDING_7: list[int] (POINTER(B)[2])
    cpoi: POINTER_T[struct_nkm_cpoi_entry_t] (POINTER(struct_nkm_cpoi_entry_t))
    cpoiCount: int (POINTER(H))
    PADDING_8: list[int] (POINTER(B)[2])
    cpat: POINTER_T[struct_nkm_cpat_entry_t] (POINTER(struct_nkm_cpat_entry_t))
    cpatCount: int (POINTER(H))
    PADDING_9: list[int] (POINTER(B)[2])
    ipoi: union_nkm_ipoi_entry_pointer_t (union_nkm_ipoi_entry_pointer_t)
    ipoiCount: int (POINTER(H))
    PADDING_10: list[int] (POINTER(B)[2])
    ipat: POINTER_T[struct_nkm_ipat_entry_t] (POINTER(struct_nkm_ipat_entry_t))
    ipatCount: int (POINTER(H))
    PADDING_11: list[int] (POINTER(B)[2])
    epoi: POINTER_T[struct_nkm_epoi_entry_t] (POINTER(struct_nkm_epoi_entry_t))
    epoiCount: int (POINTER(H))
    PADDING_12: list[int] (POINTER(B)[2])
    epat: POINTER_T[struct_nkm_epat_entry_t] (POINTER(struct_nkm_epat_entry_t))
    epatCount: int (POINTER(H))
    PADDING_13: list[int] (POINTER(B)[2])
    area: POINTER_T[struct_nkm_area_entry_t] (POINTER(struct_nkm_area_entry_t))
    areaCount: int (POINTER(H))
    PADDING_14: list[int] (POINTER(B)[2])
    came: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameCount: int (POINTER(H))
    PADDING_15: list[int] (POINTER(B)[2])
    mepo: POINTER_T[struct_nkm_mepo_entry_t] (POINTER(struct_nkm_mepo_entry_t))
    mepoCount: int (POINTER(H))
    PADDING_16: list[int] (POINTER(B)[2])
    mepa: POINTER_T[struct_nkm_mepa_entry_t] (POINTER(struct_nkm_mepa_entry_t))
    mepaCount: int (POINTER(H))
    PADDING_17: list[int] (POINTER(B)[2])
    paths: POINTER_T[struct_mdat_path_t] (POINTER(struct_mdat_path_t))
    cpoiKeyCount: int (POINTER(H))
    cpatLastCpoiIndex: int (POINTER(H))
    cpatMaxSectionOrder: int (POINTER(H))
    unknown49: int (POINTER(B))
    unknown50: int (POINTER(B))
    enemyPathData: struct_mdat_enemypath_data_t (struct_mdat_enemypath_data_t)
    itemPathData: struct_mdat_itempath_data_t (struct_mdat_itempath_data_t)
    mgEnemyPathData: struct_mdat_mgenemypath_data_t (struct_mdat_mgenemypath_data_t)
    cameIntroFirstTopCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameIntroFirstBottomCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameType6: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameBattleIntroCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    cameMissionFinishCam: POINTER_T[struct_nkm_came_entry_t] (POINTER(struct_nkm_came_entry_t))
    clipAreaLists: list[POINTER_T[struct_mdat_clip_area_list_entry_t]] (POINTER(struct_mdat_clip_area_list_entry_t)[8])
    ktpjIndexTable: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    ktpcIndexTable: POINTER_T[c_ushort] (POINTER(POINTER(H)))
    curMgRespawnId: int (POINTER(H))
    enemyRespawnRouteAreaCount: int (POINTER(H))
    trackLength: float (struct_fx32)
    trackLengthDiv15000: int (POINTER(I))
    nkmVersion: int (POINTER(H))
    unknown141: int (POINTER(B))
    missionEndAreaCount: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    obji: POINTER_T[struct_nkm_obji_entry_t]
    objiCount: int
    PADDING_0: list[int]
    path: POINTER_T[struct_nkm_path_entry_t]
    pathCount: int
    PADDING_1: list[int]
    poit: POINTER_T[struct_nkm_poit_entry_t]
    poitCount: int
    PADDING_2: list[int]
    stag: POINTER_T[struct_nkm_stag_data_t]
    ktps: POINTER_T[struct_nkm_ktps_entry_t]
    ktpsCount: int
    PADDING_3: list[int]
    ktpj: POINTER_T[struct_nkm_ktpj_entry_t]
    ktpjCount: int
    PADDING_4: list[int]
    ktp2: POINTER_T[struct_nkm_ktp2_entry_t]
    ktp2Count: int
    PADDING_5: list[int]
    ktpc: POINTER_T[struct_nkm_ktpc_entry_t]
    ktpcCount: int
    PADDING_6: list[int]
    ktpm: POINTER_T[struct_nkm_ktpm_entry_t]
    ktpmCount: int
    PADDING_7: list[int]
    cpoi: POINTER_T[struct_nkm_cpoi_entry_t]
    cpoiCount: int
    PADDING_8: list[int]
    cpat: POINTER_T[struct_nkm_cpat_entry_t]
    cpatCount: int
    PADDING_9: list[int]
    ipoi: union_nkm_ipoi_entry_pointer_t
    ipoiCount: int
    PADDING_10: list[int]
    ipat: POINTER_T[struct_nkm_ipat_entry_t]
    ipatCount: int
    PADDING_11: list[int]
    epoi: POINTER_T[struct_nkm_epoi_entry_t]
    epoiCount: int
    PADDING_12: list[int]
    epat: POINTER_T[struct_nkm_epat_entry_t]
    epatCount: int
    PADDING_13: list[int]
    area: POINTER_T[struct_nkm_area_entry_t]
    areaCount: int
    PADDING_14: list[int]
    came: POINTER_T[struct_nkm_came_entry_t]
    cameCount: int
    PADDING_15: list[int]
    mepo: POINTER_T[struct_nkm_mepo_entry_t]
    mepoCount: int
    PADDING_16: list[int]
    mepa: POINTER_T[struct_nkm_mepa_entry_t]
    mepaCount: int
    PADDING_17: list[int]
    paths: POINTER_T[struct_mdat_path_t]
    cpoiKeyCount: int
    cpatLastCpoiIndex: int
    cpatMaxSectionOrder: int
    unknown49: int
    unknown50: int
    enemyPathData: struct_mdat_enemypath_data_t
    itemPathData: struct_mdat_itempath_data_t
    mgEnemyPathData: struct_mdat_mgenemypath_data_t
    cameIntroFirstTopCam: POINTER_T[struct_nkm_came_entry_t]
    cameIntroFirstBottomCam: POINTER_T[struct_nkm_came_entry_t]
    cameType6: POINTER_T[struct_nkm_came_entry_t]
    cameBattleIntroCam: POINTER_T[struct_nkm_came_entry_t]
    cameMissionFinishCam: POINTER_T[struct_nkm_came_entry_t]
    clipAreaLists: list[POINTER_T[struct_mdat_clip_area_list_entry_t]]
    ktpjIndexTable: POINTER_T[c_ushort]
    ktpcIndexTable: POINTER_T[c_ushort]
    curMgRespawnId: int
    enemyRespawnRouteAreaCount: int
    trackLength: float
    trackLengthDiv15000: int
    nkmVersion: int
    unknown141: int
    missionEndAreaCount: int

class struct_mdat_mgenemypath_data_t(Structure):
    """
    ```python
    points: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    firstPoint: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    lastPoint: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    points: POINTER_T[struct_mdat_mgenemypoint_t]
    firstPoint: POINTER_T[struct_mdat_mgenemypoint_t]
    lastPoint: POINTER_T[struct_mdat_mgenemypoint_t]

class struct_mdat_mgenemypoint_t(Structure):
    """
    ```python
    next: list[POINTER_T[struct_mdat_mgenemypoint_t]] (POINTER(struct_mdat_mgenemypoint_t)[8])
    position: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    radius: float (struct_fx32)
    settings: POINTER_T[struct_nkm_mepo_entry_settings_t] (POINTER(struct_nkm_mepo_entry_settings_t))
    nextCount: int (POINTER(H))
    nextIsNewPathMask: int (POINTER(B))
    PADDING_0: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    next: list[POINTER_T[struct_mdat_mgenemypoint_t]]
    position: POINTER_T[struct_VecFx32]
    radius: float
    settings: POINTER_T[struct_nkm_mepo_entry_settings_t]
    nextCount: int
    nextIsNewPathMask: int
    PADDING_0: int

class struct_mdat_path_t(Structure):
    """
    ```python
    path: POINTER_T[struct_nkm_path_entry_t] (POINTER(struct_nkm_path_entry_t))
    poit: POINTER_T[struct_nkm_poit_entry_t] (POINTER(struct_nkm_poit_entry_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    path: POINTER_T[struct_nkm_path_entry_t]
    poit: POINTER_T[struct_nkm_poit_entry_t]

class struct_mobj_config_t(Structure):
    """
    ```python
    driverCollideFunc: Callable[[POINTER_T[struct_mobj_inst_t], POINTER_T[struct_driver_t], POINTER_T[c_ubyte], POINTER_T[c_ubyte]], c_uint] (CFunctionType)
    itemCollideFunc: Callable[[POINTER_T[struct_mobj_inst_t], POINTER_T[struct_it_item_inst_t], POINTER_T[c_ubyte], POINTER_T[c_ubyte]], c_uint] (CFunctionType)
    colType: int (POINTER(I))
    size: Float[Tensor, "3"] (struct_VecFx32)
    sphereCollideFunc: Callable[[POINTER_T[struct_mobj_inst_t], POINTER_T[struct_VecFx32], float, POINTER_T[struct_VecFx32]], c_int] (CFunctionType)
    providesPushback: int (POINTER(i))
    logicType: int (POINTER(I))
    driverHitSfxId: int (POINTER(i))
    itemHitSfxId: int (POINTER(I))
    nearClip: float (struct_fx32)
    farClip: float (struct_fx32)
    has3DModel: int (POINTER(i))
    sfxAudibleMaxCamYDiff: float (struct_fx32)
    sfxAudibleMinCamYDiff: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    driverCollideFunc: Callable[[POINTER_T[struct_mobj_inst_t], POINTER_T[struct_driver_t], POINTER_T[c_ubyte], POINTER_T[c_ubyte]], c_uint]
    itemCollideFunc: Callable[[POINTER_T[struct_mobj_inst_t], POINTER_T[struct_it_item_inst_t], POINTER_T[c_ubyte], POINTER_T[c_ubyte]], c_uint]
    colType: int
    size: Float[Tensor, "3"]
    sphereCollideFunc: Callable[[POINTER_T[struct_mobj_inst_t], POINTER_T[struct_VecFx32], float, POINTER_T[struct_VecFx32]], c_int]
    providesPushback: int
    logicType: int
    driverHitSfxId: int
    itemHitSfxId: int
    nearClip: float
    farClip: float
    has3DModel: int
    sfxAudibleMaxCamYDiff: float
    sfxAudibleMinCamYDiff: float

class struct_mobj_inst_t(Structure):
    """
    ```python
    objectId: int (POINTER(H))
    flags: int (POINTER(H))
    position: Float[Tensor, "3"] (struct_VecFx32)
    velocity: Float[Tensor, "3"] (struct_VecFx32)
    scale: Float[Tensor, "3"] (struct_VecFx32)
    mtx: Float[Tensor, "4 3"] (union_MtxFx43)
    size: Float[Tensor, "3"] (struct_VecFx32)
    colEntryId: int (POINTER(h))
    alpha: int (POINTER(H))
    nearClip: float (struct_fx32)
    farClip: float (struct_fx32)
    sfxMaxDistanceSquare: int (POINTER(i))
    clipAreaMask: int (POINTER(I))
    visibilityFlags: int (POINTER(I))
    has3DModel: int (POINTER(H))
    rotY: int (POINTER(H))
    stateMachine: struct_state_machine_t (struct_state_machine_t)
    soundEmitter: POINTER_T[struct_sfx_emitter_t] (POINTER(struct_sfx_emitter_t))
    config: POINTER_T[struct_mobj_config_t] (POINTER(struct_mobj_config_t))
    objiEntry: POINTER_T[struct_nkm_obji_entry_t] (POINTER(struct_nkm_obji_entry_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    objectId: int
    flags: int
    position: Float[Tensor, "3"]
    velocity: Float[Tensor, "3"]
    scale: Float[Tensor, "3"]
    mtx: Float[Tensor, "4 3"]
    size: Float[Tensor, "3"]
    colEntryId: int
    alpha: int
    nearClip: float
    farClip: float
    sfxMaxDistanceSquare: int
    clipAreaMask: int
    visibilityFlags: int
    has3DModel: int
    rotY: int
    stateMachine: struct_state_machine_t
    soundEmitter: POINTER_T[struct_sfx_emitter_t]
    config: POINTER_T[struct_mobj_config_t]
    objiEntry: POINTER_T[struct_nkm_obji_entry_t]

class struct_model_res_t(Structure):
    """
    ```python
    link: struct_NNSFndLink (struct_NNSFndLink)
    nsbmd: c_void_p (c_void_p)
    texRes: POINTER_T[struct_NNSG3dResTex_] (POINTER(struct_NNSG3dResTex_))
    ```
    """
    _pack_: ClassVar[int] = 1
    link: struct_NNSFndLink
    nsbmd: c_void_p
    texRes: POINTER_T[struct_NNSG3dResTex_]

class struct_model_t(Structure):
    """
    ```python
    renderObj: struct_NNSG3dRenderObj_ (struct_NNSG3dRenderObj_)
    cullReversed: int (POINTER(I))
    render1Mat1Shp: int (POINTER(i))
    res: struct_model_res_t (struct_model_res_t)
    ```
    """
    _pack_: ClassVar[int] = 1
    renderObj: struct_NNSG3dRenderObj_
    cullReversed: int
    render1Mat1Shp: int
    res: struct_model_res_t

class struct_nkm_area_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    length: Float[Tensor, "3"] (struct_VecFx32)
    xVector: Float[Tensor, "3"] (struct_VecFx32)
    yVector: Float[Tensor, "3"] (struct_VecFx32)
    zVector: Float[Tensor, "3"] (struct_VecFx32)
    param0: int (POINTER(h))
    param1: int (POINTER(h))
    enemyPointId: int (POINTER(h))
    shape: int (POINTER(B))
    cameraId: int (POINTER(B))
    type: int (POINTER(B))
    field45: int (POINTER(B))
    field46: int (POINTER(B))
    field47: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    length: Float[Tensor, "3"]
    xVector: Float[Tensor, "3"]
    yVector: Float[Tensor, "3"]
    zVector: Float[Tensor, "3"]
    param0: int
    param1: int
    enemyPointId: int
    shape: int
    cameraId: int
    type: int
    field45: int
    field46: int
    field47: int

class struct_nkm_came_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    rotation: Float[Tensor, "3"] (struct_VecFx32)
    target1: Float[Tensor, "3"] (struct_VecFx32)
    target2: Float[Tensor, "3"] (struct_VecFx32)
    fovBegin: int (POINTER(H))
    fovBeginSin: float (struct_fx16)
    fovBeginCos: float (struct_fx16)
    fovEnd: int (POINTER(H))
    fovEndSin: float (struct_fx16)
    fovEndCos: float (struct_fx16)
    fovSpeed: int (POINTER(h))
    type: int (POINTER(H))
    pathId: int (POINTER(h))
    routeSpeed: int (POINTER(h))
    targetSpeed: int (POINTER(H))
    duration: int (POINTER(H))
    next: int (POINTER(h))
    introFirst: int (POINTER(B))
    unknown: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    rotation: Float[Tensor, "3"]
    target1: Float[Tensor, "3"]
    target2: Float[Tensor, "3"]
    fovBegin: int
    fovBeginSin: float
    fovBeginCos: float
    fovEnd: int
    fovEndSin: float
    fovEndCos: float
    fovSpeed: int
    type: int
    pathId: int
    routeSpeed: int
    targetSpeed: int
    duration: int
    next: int
    introFirst: int
    unknown: int

class struct_nkm_cpat_entry_t(Structure):
    """
    ```python
    start: int (POINTER(H))
    length: int (POINTER(H))
    next: list[int] (POINTER(B)[3])
    previous: list[int] (POINTER(B)[3])
    sectionOrder: int (POINTER(B))
    PADDING_0: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    start: int
    length: int
    next: list[int]
    previous: list[int]
    sectionOrder: int
    PADDING_0: int

class struct_nkm_cpoi_entry_t(Structure):
    """
    ```python
    x1: float (struct_fx32)
    z1: float (struct_fx32)
    x2: float (struct_fx32)
    z2: float (struct_fx32)
    sin: float (struct_fx32)
    cos: float (struct_fx32)
    distance: float (struct_fx32)
    gotoSection: int (POINTER(h))
    startSection: int (POINTER(h))
    keyId: int (POINTER(h))
    respawnId: int (POINTER(B))
    flags: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    x1: float
    z1: float
    x2: float
    z2: float
    sin: float
    cos: float
    distance: float
    gotoSection: int
    startSection: int
    keyId: int
    respawnId: int
    flags: int

class struct_nkm_epat_entry_t(Structure):
    """
    ```python
    start: int (POINTER(H))
    length: int (POINTER(H))
    next: list[int] (POINTER(B)[3])
    previous: list[int] (POINTER(B)[3])
    sectionOrder: int (POINTER(h))
    ```
    """
    _pack_: ClassVar[int] = 1
    start: int
    length: int
    next: list[int]
    previous: list[int]
    sectionOrder: int

class struct_nkm_epoi_entry_settings_t(Structure):
    """
    ```python
    drifting: int (POINTER(H))
    unknown1: int (POINTER(H))
    unknown2: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    drifting: int
    unknown1: int
    unknown2: int

class struct_nkm_epoi_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    radius: float (struct_fx32)
    settings: struct_nkm_epoi_entry_settings_t (struct_nkm_epoi_entry_settings_t)
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    radius: float
    settings: struct_nkm_epoi_entry_settings_t

class struct_nkm_ipat_entry_t(Structure):
    """
    ```python
    start: int (POINTER(H))
    length: int (POINTER(H))
    next: list[int] (POINTER(B)[3])
    previous: list[int] (POINTER(B)[3])
    sectionOrder: int (POINTER(h))
    ```
    """
    _pack_: ClassVar[int] = 1
    start: int
    length: int
    next: list[int]
    previous: list[int]
    sectionOrder: int

class struct_nkm_ipoi_entry_beta_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    radius: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    radius: float

class struct_nkm_ipoi_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    radius: float (struct_fx32)
    recalcIdx: int (POINTER(B))
    PADDING_0: list[int] (POINTER(B)[3])
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    radius: float
    recalcIdx: int
    PADDING_0: list[int]

class struct_nkm_ktp2_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    rotation: Float[Tensor, "3"] (struct_VecFx32)
    padding: int (POINTER(H))
    id: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    rotation: Float[Tensor, "3"]
    padding: int
    id: int

class struct_nkm_ktpc_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    rotation: Float[Tensor, "3"] (struct_VecFx32)
    nextMepo: int (POINTER(h))
    id: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    rotation: Float[Tensor, "3"]
    nextMepo: int
    id: int

class struct_nkm_ktpj_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    rotation: Float[Tensor, "3"] (struct_VecFx32)
    enemyPointId: int (POINTER(H))
    itemPointId: int (POINTER(H))
    id: int (POINTER(h))
    PADDING_0: list[int] (POINTER(B)[2])
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    rotation: Float[Tensor, "3"]
    enemyPointId: int
    itemPointId: int
    id: int
    PADDING_0: list[int]

class struct_nkm_ktpm_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    rotation: Float[Tensor, "3"] (struct_VecFx32)
    padding: int (POINTER(H))
    id: int (POINTER(h))
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    rotation: Float[Tensor, "3"]
    padding: int
    id: int

class struct_nkm_ktps_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    rotation: Float[Tensor, "3"] (struct_VecFx32)
    padding: int (POINTER(H))
    index: int (POINTER(h))
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    rotation: Float[Tensor, "3"]
    padding: int
    index: int

class struct_nkm_mepa_entry_t(Structure):
    """
    ```python
    start: int (POINTER(H))
    length: int (POINTER(H))
    next: list[int] (POINTER(B)[8])
    previous: list[int] (POINTER(B)[8])
    ```
    """
    _pack_: ClassVar[int] = 1
    start: int
    length: int
    next: list[int]
    previous: list[int]

class struct_nkm_mepo_entry_settings_t(Structure):
    """
    ```python
    drifting: int (POINTER(I))
    unknown1: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    drifting: int
    unknown1: int

class struct_nkm_mepo_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    radius: float (struct_fx32)
    settings: struct_nkm_mepo_entry_settings_t (struct_nkm_mepo_entry_settings_t)
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    radius: float
    settings: struct_nkm_mepo_entry_settings_t

class struct_nkm_obji_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    rotation: Float[Tensor, "3"] (struct_VecFx32)
    scale: Float[Tensor, "3"] (struct_VecFx32)
    objectId: int (POINTER(H))
    pathId: int (POINTER(h))
    settings: list[int] (POINTER(h)[7])
    flags: int (POINTER(h))
    showsInTT: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    rotation: Float[Tensor, "3"]
    scale: Float[Tensor, "3"]
    objectId: int
    pathId: int
    settings: list[int]
    flags: int
    showsInTT: int
    PADDING_0: list[int]

class struct_nkm_path_entry_t(Structure):
    """
    ```python
    id: int (POINTER(B))
    loop: int (POINTER(B))
    pointCount: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    id: int
    loop: int
    pointCount: int

class struct_nkm_poit_entry_t(Structure):
    """
    ```python
    position: Float[Tensor, "3"] (struct_VecFx32)
    pointIndex: int (POINTER(B))
    unknown1: int (POINTER(B))
    duration: int (POINTER(h))
    _0: union_nkm_poit_entry_t_0 (union_nkm_poit_entry_t_0)
    ```
    """
    _pack_: ClassVar[int] = 1
    position: Float[Tensor, "3"]
    pointIndex: int
    unknown1: int
    duration: int
    _0: union_nkm_poit_entry_t_0

class struct_nkm_poit_entry_t_0_0(Structure):
    """
    ```python
    unknown2_s16: int (POINTER(h))
    unknown2_s16_hi: int (POINTER(h))
    ```
    """
    _pack_: ClassVar[int] = 1
    unknown2_s16: int
    unknown2_s16_hi: int

class struct_nkm_stag_data_t(Structure):
    """
    ```python
    courseId: int (POINTER(H))
    nrLaps: int (POINTER(H))
    polePosition: int (POINTER(B))
    fogEnabled: int (POINTER(B))
    fogTableGenMode: int (POINTER(B))
    fogSlope: int (POINTER(B))
    unknown2: list[int] (POINTER(B)[8])
    fogDensity: float (struct_fx32)
    fogColor: int (POINTER(I))
    kclColors: list[int] (POINTER(H)[4])
    mobjFarClip: float (struct_fx32)
    frustumFar: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    courseId: int
    nrLaps: int
    polePosition: int
    fogEnabled: int
    fogTableGenMode: int
    fogSlope: int
    unknown2: list[int]
    fogDensity: float
    fogColor: int
    kclColors: list[int]
    mobjFarClip: float
    frustumFar: float

class struct_physp_char_params_t(Structure):
    """
    ```python
    field0: float (struct_fx32)
    weight: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: float
    weight: float

class struct_physp_kart_params_t(Structure):
    """
    ```python
    colSphereSize: float (struct_fx32)
    colSphereZOffset: float (struct_fx32)
    gap8: list[int] (POINTER(B)[4])
    weight: float (struct_fx16)
    driftBoostTime: int (POINTER(h))
    maxSpeed: float (struct_fx32)
    baseAcceleration: float (struct_fx32)
    field18: float (struct_fx32)
    field1C: float (struct_fx32)
    driftBaseAcceleration: float (struct_fx32)
    field24: float (struct_fx32)
    field28: float (struct_fx32)
    deceleration: float (struct_fx32)
    handling: float (struct_fx16)
    drift: float (struct_fx16)
    driftTurningCompensation: float (struct_fx32)
    collisionVelocityMinusDirMultipliers: Float[Tensor, "12"] (struct_fx32[12])
    collisionSpeedMultipliers: Float[Tensor, "12"] (struct_fx32[12])
    ```
    """
    _pack_: ClassVar[int] = 1
    colSphereSize: float
    colSphereZOffset: float
    gap8: list[int]
    weight: float
    driftBoostTime: int
    maxSpeed: float
    baseAcceleration: float
    field18: float
    field1C: float
    driftBaseAcceleration: float
    field24: float
    field28: float
    deceleration: float
    handling: float
    drift: float
    driftTurningCompensation: float
    collisionVelocityMinusDirMultipliers: Float[Tensor, "12"]
    collisionSpeedMultipliers: Float[Tensor, "12"]

class struct_quaternion_t(Structure):
    """
    ```python
    x: float (struct_fx32)
    y: float (struct_fx32)
    z: float (struct_fx32)
    w: float (struct_fx32)
    ```
    """
    _pack_: ClassVar[int] = 1
    x: float
    y: float
    z: float
    w: float

class struct_race_driver_status_t(Structure):
    """
    ```python
    state: int (POINTER(I))
    lapFrameCounter: int (POINTER(I))
    field8: int (POINTER(I))
    lapTimes: list[struct_race_time_t] (struct_race_time_t[5])
    totalTime: struct_race_time_t (struct_race_time_t)
    curLap: int (POINTER(i))
    firstPlaceTime: int (POINTER(I))
    totalMilliseconds: int (POINTER(I))
    flags: int (POINTER(H))
    curCpoi: int (POINTER(H))
    lastCorrectKeyPoint: int (POINTER(h))
    curKeyPoint: int (POINTER(h))
    curCpat: int (POINTER(H))
    highestReachedLap: int (POINTER(H))
    place: int (POINTER(H))
    driverId: int (POINTER(H))
    field3CBit89: int (POINTER(H))
    PADDING_0: int (POINTER(B))
    skillRankPoints: int (POINTER(h))
    cpoiProgress: float (struct_fx32)
    raceProgress: float (struct_fx32)
    lapProgress: float (struct_fx32)
    cpoiMask: list[int] (POINTER(I)[16])
    ```
    """
    _pack_: ClassVar[int] = 1
    state: int
    lapFrameCounter: int
    field8: int
    lapTimes: list[struct_race_time_t]
    totalTime: struct_race_time_t
    curLap: int
    firstPlaceTime: int
    totalMilliseconds: int
    flags: int
    curCpoi: int
    lastCorrectKeyPoint: int
    curKeyPoint: int
    curCpat: int
    highestReachedLap: int
    place: int
    driverId: int
    field3CBit89: int
    PADDING_0: int
    skillRankPoints: int
    cpoiProgress: float
    raceProgress: float
    lapProgress: float
    cpoiMask: list[int]

class struct_race_skill_rankpoints_t(Structure):
    """
    ```python
    rankTimeDeltaPoints: int (POINTER(i))
    firstPlacePercentagePoints: int (POINTER(i))
    startBoostPoints: int (POINTER(i))
    powerSlidePoints: int (POINTER(i))
    itemHitPoints: int (POINTER(i))
    offRoadTimePoints: int (POINTER(i))
    wallHitPoints: int (POINTER(i))
    damagePoints: int (POINTER(i))
    respawnPoints: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    rankTimeDeltaPoints: int
    firstPlacePercentagePoints: int
    startBoostPoints: int
    powerSlidePoints: int
    itemHitPoints: int
    offRoadTimePoints: int
    wallHitPoints: int
    damagePoints: int
    respawnPoints: int

class struct_race_state_t(Structure):
    """
    ```python
    state: int (POINTER(h))
    PADDING_0: list[int] (POINTER(B)[2])
    frameCounter: int (POINTER(I))
    frameCounter2: int (POINTER(i))
    frameCounterModulo8: int (POINTER(i))
    isOddFrame: int (POINTER(i))
    frameCounterModuloDriverCount: int (POINTER(i))
    toonTableOffset: int (POINTER(I))
    toonTableUpdateCounter: int (POINTER(I))
    darkeningFogState: int (POINTER(I))
    prevDarkeningFogState: int (POINTER(I))
    isCamAnimMode: int (POINTER(i))
    isCamAnimSingleScreen: int (POINTER(i))
    field30: int (POINTER(I))
    field34: int (POINTER(I))
    isAwardStaffRoll: int (POINTER(i))
    field3C: int (POINTER(I))
    light0Dir: Float[Tensor, "3"] (struct_VecFx16)
    PADDING_1: list[int] (POINTER(B)[2])
    ```
    """
    _pack_: ClassVar[int] = 1
    state: int
    PADDING_0: list[int]
    frameCounter: int
    frameCounter2: int
    frameCounterModulo8: int
    isOddFrame: int
    frameCounterModuloDriverCount: int
    toonTableOffset: int
    toonTableUpdateCounter: int
    darkeningFogState: int
    prevDarkeningFogState: int
    isCamAnimMode: int
    isCamAnimSingleScreen: int
    field30: int
    field34: int
    isAwardStaffRoll: int
    field3C: int
    light0Dir: Float[Tensor, "3"]
    PADDING_1: list[int]

class struct_race_status_t(Structure):
    """
    ```python
    time: struct_race_status_time_t (struct_race_status_time_t)
    rankTimeGp: int (POINTER(H))
    finishedDriverCount: int (POINTER(H))
    field10: int (POINTER(H))
    PADDING_0: list[int] (POINTER(B)[2])
    driverStatus: list[struct_race_driver_status_t] (struct_race_driver_status_t[8])
    placeDriverIds: list[int] (POINTER(b)[8])
    PADDING_1: list[int] (POINTER(B)[4])
    safeRng: struct_MATHRandContext32 (struct_MATHRandContext32)
    rngSeed: int (POINTER(I))
    PADDING_2: list[int] (POINTER(B)[4])
    randomRng: struct_MATHRandContext32 (struct_MATHRandContext32)
    stableRng: struct_MATHRandContext32 (struct_MATHRandContext32)
    rankTimeGpRtt: POINTER_T[struct_ranktime_gp_t] (POINTER(struct_ranktime_gp_t))
    resultsStored: int (POINTER(i))
    camAnimComplete: int (POINTER(i))
    cpoiKeyPointProgress: POINTER_T[struct_fx32] (POINTER(struct_fx32))
    gap4D8: list[int] (POINTER(B)[4])
    rankPointRpt: POINTER_T[struct_rankpoint_t] (POINTER(struct_rankpoint_t))
    missionResult: int (POINTER(I))
    oneDivCpatSegmentCount: float (struct_fx32)
    oneDivNrLaps: float (struct_fx32)
    useTimeLimit: int (POINTER(i))
    uncontrollable: int (POINTER(i))
    timeLimit: int (POINTER(I))
    field4F8: int (POINTER(B))
    field4F9: int (POINTER(B))
    mrWinDelayCounter: int (POINTER(H))
    mrLoseDelayCounter: int (POINTER(H))
    PADDING_3: list[int] (POINTER(B)[2])
    skillRankPoints: struct_race_skill_rankpoints_t (struct_race_skill_rankpoints_t)
    PADDING_4: list[int] (POINTER(B)[4])
    ```
    """
    _pack_: ClassVar[int] = 1
    time: struct_race_status_time_t
    rankTimeGp: int
    finishedDriverCount: int
    field10: int
    PADDING_0: list[int]
    driverStatus: list[struct_race_driver_status_t]
    placeDriverIds: list[int]
    PADDING_1: list[int]
    safeRng: struct_MATHRandContext32
    rngSeed: int
    PADDING_2: list[int]
    randomRng: struct_MATHRandContext32
    stableRng: struct_MATHRandContext32
    rankTimeGpRtt: POINTER_T[struct_ranktime_gp_t]
    resultsStored: int
    camAnimComplete: int
    cpoiKeyPointProgress: POINTER_T[struct_fx32]
    gap4D8: list[int]
    rankPointRpt: POINTER_T[struct_rankpoint_t]
    missionResult: int
    oneDivCpatSegmentCount: float
    oneDivNrLaps: float
    useTimeLimit: int
    uncontrollable: int
    timeLimit: int
    field4F8: int
    field4F9: int
    mrWinDelayCounter: int
    mrLoseDelayCounter: int
    PADDING_3: list[int]
    skillRankPoints: struct_race_skill_rankpoints_t
    PADDING_4: list[int]

class struct_race_status_time_t(Structure):
    """
    ```python
    frameCounter: int (POINTER(I))
    timeRunning: int (POINTER(i))
    lapTime: struct_race_time_t (struct_race_time_t)
    ```
    """
    _pack_: ClassVar[int] = 1
    frameCounter: int
    timeRunning: int
    lapTime: struct_race_time_t

class struct_race_time_t(Structure):
    """
    ```python
    milliseconds: int (POINTER(H))
    minutes: int (POINTER(B))
    seconds: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    milliseconds: int
    minutes: int
    seconds: int

class struct_rankpoint_entry_t(Structure):
    """
    ```python
    points: list[int] (POINTER(B)[8])
    ```
    """
    _pack_: ClassVar[int] = 1
    points: list[int]

class struct_rankpoint_t(Structure):
    """
    ```python
    signature: int (POINTER(I))
    fileSize: int (POINTER(I))
    entries: list[struct_rankpoint_entry_t] (struct_rankpoint_entry_t[1])
    ```
    """
    _pack_: ClassVar[int] = 1
    signature: int
    fileSize: int
    entries: list[struct_rankpoint_entry_t]

class struct_ranktime_gp_entry_t(Structure):
    """
    ```python
    courseId: int (POINTER(B))
    PADDING_0: int (POINTER(B))
    rankTime50cc: int (POINTER(H))
    rankTime100cc: int (POINTER(H))
    rankTime150cc: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    courseId: int
    PADDING_0: int
    rankTime50cc: int
    rankTime100cc: int
    rankTime150cc: int

class struct_ranktime_gp_t(Structure):
    """
    ```python
    signature: int (POINTER(I))
    fileSize: int (POINTER(I))
    courseCount: int (POINTER(H))
    rankTimeDeltaFactor: int (POINTER(H))
    firstPlacePercentageFactor: int (POINTER(H))
    startBoostFactor: int (POINTER(H))
    powerSlideFactor: int (POINTER(H))
    itemHitFactor: int (POINTER(H))
    offRoadTimeFactor: int (POINTER(H))
    wallHitFactor: int (POINTER(H))
    damageFactor: int (POINTER(H))
    respawnFactor: int (POINTER(H))
    courseRankTimes: list[struct_ranktime_gp_entry_t] (struct_ranktime_gp_entry_t[1])
    ```
    """
    _pack_: ClassVar[int] = 1
    signature: int
    fileSize: int
    courseCount: int
    rankTimeDeltaFactor: int
    firstPlacePercentageFactor: int
    startBoostFactor: int
    powerSlideFactor: int
    itemHitFactor: int
    offRoadTimeFactor: int
    wallHitFactor: int
    damageFactor: int
    respawnFactor: int
    courseRankTimes: list[struct_ranktime_gp_entry_t]

class struct_sfx_emitter_ex_params_t(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(i))
    pitchOffset: int (POINTER(I))
    fieldC: int (POINTER(I))
    unk10: list[int] (POINTER(B)[12])
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    pitchOffset: int
    fieldC: int
    unk10: list[int]

class struct_sfx_emitter_t(Structure):
    """
    ```python
    listLink: struct_list_link_t (struct_list_link_t)
    soundList: struct_NNSFndList (struct_NNSFndList)
    field18: int (POINTER(I))
    field1C: struct_list_link_t (struct_list_link_t)
    position: POINTER_T[struct_VecFx32] (POINTER(struct_VecFx32))
    startFunc: c_void_p (c_void_p)
    updateFunc: c_void_p (c_void_p)
    field34: int (POINTER(H))
    field36: int (POINTER(H))
    field38: int (POINTER(I))
    sfxParamIdx: int (POINTER(B))
    field3D: list[int] (POINTER(B)[3])
    squareDistance: int (POINTER(I))
    ```
    """
    _pack_: ClassVar[int] = 1
    listLink: struct_list_link_t
    soundList: struct_NNSFndList
    field18: int
    field1C: struct_list_link_t
    position: POINTER_T[struct_VecFx32]
    startFunc: c_void_p
    updateFunc: c_void_p
    field34: int
    field36: int
    field38: int
    sfxParamIdx: int
    field3D: list[int]
    squareDistance: int

class struct_state_machine_state_t(Structure):
    """
    ```python
    initFunc: Callable[[c_void_p], None] (CFunctionType)
    stateFunc: Callable[[c_void_p], None] (CFunctionType)
    ```
    """
    _pack_: ClassVar[int] = 1
    initFunc: Callable[[c_void_p], None]
    stateFunc: Callable[[c_void_p], None]

class struct_state_machine_t(Structure):
    """
    ```python
    states: POINTER_T[struct_state_machine_state_t] (POINTER(struct_state_machine_state_t))
    counter: int (POINTER(I))
    userData: c_void_p (c_void_p)
    nrStates: int (POINTER(H))
    curState: int (POINTER(H))
    nextState: int (POINTER(H))
    gotoNextState: int (POINTER(H))
    ```
    """
    _pack_: ClassVar[int] = 1
    states: POINTER_T[struct_state_machine_state_t]
    counter: int
    userData: c_void_p
    nrStates: int
    curState: int
    nextState: int
    gotoNextState: int

class struct_struc_313_mepo(Structure):
    """
    ```python
    nextMepo: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    curMepo: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    direction: Float[Tensor, "3"] (struct_VecFx32)
    areaMepo: POINTER_T[struct_mdat_mgenemypoint_t] (POINTER(struct_mdat_mgenemypoint_t))
    areaMepoValid: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    nextMepo: POINTER_T[struct_mdat_mgenemypoint_t]
    curMepo: POINTER_T[struct_mdat_mgenemypoint_t]
    direction: Float[Tensor, "3"]
    areaMepo: POINTER_T[struct_mdat_mgenemypoint_t]
    areaMepoValid: int

class struct_struc_316_epoi(Structure):
    """
    ```python
    nextEpoi: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    curEpoi: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    direction: Float[Tensor, "3"] (struct_VecFx32)
    areaEpoi: POINTER_T[struct_mdat_enemypoint_t] (POINTER(struct_mdat_enemypoint_t))
    areaEpoiValid: int (POINTER(i))
    driverId: int (POINTER(H))
    field1E: int (POINTER(B))
    field1F: int (POINTER(B))
    ```
    """
    _pack_: ClassVar[int] = 1
    nextEpoi: POINTER_T[struct_mdat_enemypoint_t]
    curEpoi: POINTER_T[struct_mdat_enemypoint_t]
    direction: Float[Tensor, "3"]
    areaEpoi: POINTER_T[struct_mdat_enemypoint_t]
    areaEpoiValid: int
    driverId: int
    field1E: int
    field1F: int

class struct_struc_334(Structure):
    """
    ```python
    field0: int (POINTER(i))
    field4: int (POINTER(i))
    field8: int (POINTER(i))
    fieldC: int (POINTER(i))
    field10: int (POINTER(i))
    field14: int (POINTER(i))
    field18: int (POINTER(i))
    field1C: int (POINTER(i))
    field20: int (POINTER(i))
    field24: int (POINTER(i))
    ```
    """
    _pack_: ClassVar[int] = 1
    field0: int
    field4: int
    field8: int
    fieldC: int
    field10: int
    field14: int
    field18: int
    field1C: int
    field20: int
    field24: int

class union_MtxFx33(Union):
    """
    ```python
    _0: struct_MtxFx33_0 (struct_MtxFx33_0)
    m: Float[Tensor, "3 3"] (struct_fx32[3][3])
    a: Float[Tensor, "9"] (struct_fx32[9])
    ```
    """
    _pack_: ClassVar[int] = 1
    _0: struct_MtxFx33_0
    m: Float[Tensor, "3 3"]
    a: Float[Tensor, "9"]

class union_MtxFx43(Union):
    """
    ```python
    _0: struct_MtxFx43_0 (struct_MtxFx43_0)
    m: Float[Tensor, "4 3"] (struct_fx32[3][4])
    a: Float[Tensor, "12"] (struct_fx32[12])
    ```
    """
    _pack_: ClassVar[int] = 1
    _0: struct_MtxFx43_0
    m: Float[Tensor, "4 3"]
    a: Float[Tensor, "12"]

class union_NNSG3dResDataBlockHeader__0(Union):
    """
    ```python
    kind: int (POINTER(I))
    chr: list[int] (POINTER(B)[4])
    ```
    """
    _pack_: ClassVar[int] = 1
    kind: int
    chr: list[int]

class union_nkm_ipoi_entry_pointer_t(Union):
    """
    ```python
    final: POINTER_T[struct_nkm_ipoi_entry_t] (POINTER(struct_nkm_ipoi_entry_t))
    beta: POINTER_T[struct_nkm_ipoi_entry_beta_t] (POINTER(struct_nkm_ipoi_entry_beta_t))
    ```
    """
    _pack_: ClassVar[int] = 1
    final: POINTER_T[struct_nkm_ipoi_entry_t]
    beta: POINTER_T[struct_nkm_ipoi_entry_beta_t]

class union_nkm_poit_entry_t_0(Union):
    """
    ```python
    unknown2: int (POINTER(I))
    _0: struct_nkm_poit_entry_t_0_0 (struct_nkm_poit_entry_t_0_0)
    ```
    """
    _pack_: ClassVar[int] = 1
    unknown2: int
    _0: struct_nkm_poit_entry_t_0_0