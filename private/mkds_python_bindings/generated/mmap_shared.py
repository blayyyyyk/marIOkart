from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.spinLock import *
from private.mkds_python_bindings.generated.thread import *
from private.mkds_python_bindings.generated.types import *


class OSDmaClearSrc(Union):
    _fields_ = [
        ('b32', u32),
        ('b16', u16),
    ]

class OSSystemWork(Structure):
    _fields_ = [
        ('bootCheckInfo', (u8 * 32)),
        ('resetParameter', u32),
        ('padding5', (u8 * 8)),
        ('romBaseOffset', u32),
        ('cartridgeModuleInfo', (u8 * 12)),
        ('vblankCount', u32),
        ('wmBootBuf', (u8 * 64)),
        ('nvramUserInfo', (u8 * 256)),
        ('isd_reserved1', (u8 * 32)),
        ('arenaInfo', (u8 * 72)),
        ('real_time_clock', (u8 * 8)),
        ('dmaClearBuf', (u32 * 4)),
        ('rom_header', (u8 * 352)),
        ('isd_reserved2', (u8 * 32)),
        ('pxiSignalParam', (u32 * 2)),
        ('pxiHandleChecker', (u32 * 2)),
        ('mic_last_address', u32),
        ('mic_sampling_data', u16),
        ('wm_callback_control', u16),
        ('wm_rssi_pool', u16),
        ('padding3', (u8 * 2)),
        ('component_param', u32),
        ('threadinfo_mainp', POINTER32(OSThreadInfo)),
        ('threadinfo_subp', POINTER32(OSThreadInfo)),
        ('button_XY', u16),
        ('touch_panel', (u8 * 4)),
        ('autoloadSync', u16),
        ('lockIDFlag_mainp', (u32 * 2)),
        ('lockIDFlag_subp', (u32 * 2)),
        ('lock_VRAM_C', OSLockWord),
        ('lock_VRAM_D', OSLockWord),
        ('lock_WRAM_BLOCK0', OSLockWord),
        ('lock_WRAM_BLOCK1', OSLockWord),
        ('lock_CARD', OSLockWord),
        ('lock_CARTRIDGE', OSLockWord),
        ('lock_INIT', OSLockWord),
        ('mmem_checker_mainp', u16),
        ('mmem_checker_subp', u16),
        ('padding4', (u8 * 2)),
        ('command_area', u16),
    ]
