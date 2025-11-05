from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.driver import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.item import *
from private.mkds_python_bindings.generated.mapData import *
from private.mkds_python_bindings.generated.types import *


class it_driver_itemslot_t(Structure):
    _fields_ = [
        ('itemConfigId', c_int),
        ('driverItemStatus', POINTER32(it_driver_item_status_t)),
        ('itemCount', c_int),
        ('fieldC', c_int),
        ('timeout', u16),
        ('field12', u16),
        ('items', (POINTER32(it_item_inst_t) * 3)),
        ('field20', c_int),
        ('field24', c_int),
    ]

class it_driver_dragitem_t(Structure):
    _fields_ = [
        ('itemType', c_int),
        ('itemConfigId', c_int),
        ('field8', c_int),
        ('driverItemStatus', POINTER32(it_driver_item_status_t)),
        ('driver', POINTER32(driver_t)),
        ('items', (POINTER32(it_item_inst_t) * 3)),
        ('itemCount', c_int),
        ('field24', c_int),
        ('driverId', u16),
        ('field2C', MtxFx43),
        ('field5C', VecFx32),
        ('field68', c_int),
        ('field6C', VecFx32),
        ('field78', c_int),
        ('field7C', c_int),
        ('field80', c_int),
        ('field84', VecFx32),
        ('field90', VecFx32),
        ('field9C', VecFx32),
        ('fieldA8', VecFx32),
        ('fieldB4', VecFx32),
        ('gapC0', (u8 * 12)),
        ('fieldCC', (c_int * 3)),
        ('fieldD8', VecFx32),
        ('fieldE4', c_int),
        ('fieldE8', c_int),
        ('fieldEC', (c_int * 16)),
        ('field12C', (c_int * 16)),
        ('field16C', c_int),
        ('field170', c_int),
        ('field174', u16),
        ('field176', u16),
        ('field178', c_int),
        ('field17C', c_int),
        ('field180', VecFx32),
        ('field18C', (u16 * 3)),
        ('field192', u16),
    ]

class it_driver_item_status_t(Structure):
    _fields_ = [
        ('field0', c_int),
        ('field4', c_int),
        ('field8', c_int),
        ('fieldC', c_int),
        ('slotItemConfigId', c_int),
        ('dragItemConfigId', c_int),
        ('field18', POINTER32(it_driver_item_status_t)),
        ('field1C', c_int),
        ('field20', c_int),
        ('field24', c_int),
        ('field28', c_int),
        ('field2C', c_int),
        ('itemSlot', it_driver_itemslot_t),
        ('dragItem', it_driver_dragitem_t),
        ('field1EC', u16),
        ('ipoi', POINTER32(mdat_itempoint_t)),
        ('field1F4', c_int),
        ('driverId', u16),
        ('driver', POINTER32(driver_t)),
        ('driverIndex', c_int),
        ('isUsingShroom', c_int),
        ('field208', c_int),
        ('field20C', c_int),
    ]
