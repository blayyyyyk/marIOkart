from mkds.utils import (
    read_u32,
    read_s32,
    read_u8,
    read_s8,
    read_s16,
    read_u16,
    read_vector_3d_fx32
)

OBJECT_DATA_PTR = 0x0217B588

# Object flags
FLAG_DYNAMIC = 0x1000
FLAG_MAPOBJ  = 0x2000
FLAG_ITEM    = 0x4000
FLAG_RACER   = 0x8000

class GameObject:
    size = 0x1C
    
    def __init__(self, 
        id: int,
        ptr: int,
        flags: int,
        is_deleted: bool,
        is_deactivated: bool,
        is_removed: bool,
        pos_ptr: int,
        position: tuple[float, float, float],
        type_id: int | None,
        skip: bool,
        is_item: bool
    ):
        self.id = id
        self.ptr = ptr
        self.flags = flags
        self.is_deleted = is_deleted
        self.is_deactivated = is_deactivated
        self.is_removed = is_removed
        self.pos_ptr = pos_ptr
        self.position = position
        self.type_id = type_id
        self.skip = skip
        self.is_item = is_item
        
        
        
    @classmethod
    def from_bytes(cls, id: int, data: bytes):
        offset = read_s32(data, OBJECT_DATA_PTR + 0x10) + id * cls.size
        obj_ptr = read_u32(data, offset + 0x18)
        flags = read_u16(data, offset + 0x14)
        is_deleted = False
        is_deactivated = False
        is_removed = False
        
        if obj_ptr == 0:
            is_deleted = True
            
        if flags & 0x200 != 0:
            is_deactivated = True
            
        pos_ptr = read_u32(data, offset + 0x0C)
        if pos_ptr == 0:
            is_removed = True
        
        position = read_vector_3d_fx32(data, pos_ptr)
        skip = False
        type_id = None
        is_item = False
        
        if flags & FLAG_MAPOBJ != 0:
            type_id = read_s16(data, obj_ptr)
            is_coin_collected = read_u16(data, obj_ptr + 0x02) & 0x01 != 0
            if type_id == 0x68 and is_coin_collected:
                skip = True
                
        elif flags & FLAG_RACER != 0:
            ghost_flag = read_u8(data, obj_ptr + 0x7C)
            is_ghost = ghost_flag & 0x04 !=0
            if is_ghost:
                skip = True
                
        elif flags & FLAG_ITEM != 0:
            is_item = True
            
        elif flags & FLAG_DYNAMIC == 0:
            skip = True
            
        return cls(
            id,
            obj_ptr,
            flags,
            is_deleted,
            is_deactivated,
            is_removed,
            pos_ptr,
            position,
            type_id,
            skip,
            is_item
        )
        
class MapObject(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
      
class GameItemObject(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class DynamicObject(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
    
class RacerObject(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_item_pos(self, data: bytes):
        return read_vector_3d_fx32(data, self.ptr + 0x1D8)
        
    def get_object_radius(self, data: bytes):
        return read_s32(data, self.ptr + 0x1D0)