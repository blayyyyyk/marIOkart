from src.mkdslib.codegen_out import *
import torch
import numpy as np
import ctypes

RACER_PTR_ADDR = 0x0217ACF8
COURSE_ID_ADDR = 0x23CDCD8
OBJECTS_PTR_ADDR = 0x0217B588
CHECKPOINT_PTR_ADDR = 0x021755FC
CLOCK_DATA_PTR = 0x0217AA34
CAMERA_PTR_ADDR = 0x0217AA4C
RACE_STATUS_PTR_ADDR = 0x021755FC
MAP_DATA_PTR_ADDR = 0x02175600
COLLISION_DATA_ADDR = 0x0217B5F4
RACE_STATE_PTR_ADDR = 0x0217AA34
FX32_SCALE_FACTOR = 1 / (1 << 12)

Array3x32 = ctypes.c_int32 * 3
Array3x16 = ctypes.c_int16 * 3

def as_tensor(cls):
    orig_getattribute = cls.__getattribute__
    def __getattribute__(self, name):
        val = orig_getattribute(self, name)
        target_array = None
        dtype = None
        shape = (-1,)
        
        # fx32/fx16 fallback
        if isinstance(val, (struct_fx32, struct_fx16, fx32, fx16)):
            return float(val.val) * FX32_SCALE_FACTOR
        
        if isinstance(val, ctypes.Array) and val._type_ is fx32:
            target_array = val
            dtype = torch.int32
            #elif isinstance(val, ctypes.Array) and issubclass(val._type_, ctypes.Structure):
            
            
        elif isinstance(val, VecFx32):
            # Create an array view of the struct memory
            target_array = Array3x32.from_buffer(val)
            dtype = torch.int32
        elif isinstance(val, VecFx16):
            target_array = Array3x16.from_buffer(val)
            dtype = torch.int16
        elif isinstance(val, (union_MtxFx43, union_MtxFx33)):
            target_array = val.a  # This is already a ctypes array
            dtype = torch.int32
            if isinstance(val, union_MtxFx33):
                shape = (3, 3)
            if isinstance(val, union_MtxFx43):
                shape = (4, 3)
        

        if target_array is not None and dtype is not None and shape is not None:
            raw = torch.frombuffer(target_array, dtype=dtype)
            return raw.float().view(*shape) * FX32_SCALE_FACTOR
        
        return val

    cls.__getattribute__ = __getattribute__
    return cls
    

struct_nkm_cpoi_entry_t = as_tensor(struct_nkm_cpoi_entry_t)
kcol_header_t = as_tensor(kcol_header_t)
driver_t = as_tensor(driver_t)
camera_t = as_tensor(camera_t)
struct_race_status_t = as_tensor(struct_race_status_t)
struct_race_driver_status_t = as_tensor(struct_race_driver_status_t)
race_status_t = as_tensor(race_status_t)