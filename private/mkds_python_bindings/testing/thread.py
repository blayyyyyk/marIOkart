from ctypes import *
from private.mkds_python_bindings.testing.context import *
from private.mkds_python_bindings.testing.types import *

OS_THREAD_STATE_WAITING = 0
OS_THREAD_STATE_READY = 1
OS_THREAD_STATE_TERMINATED = 2

OS_THREAD_STATE_WAITING = 0
OS_THREAD_STATE_READY = 1
OS_THREAD_STATE_TERMINATED = 2

OS_STACK_NO_ERROR = 0
OS_STACK_OVERFLOW = 1
OS_STACK_ABOUT_TO_OVERFLOW = 2
OS_STACK_UNDERFLOW = 3

OS_STACK_NO_ERROR = 0
OS_STACK_OVERFLOW = 1
OS_STACK_ABOUT_TO_OVERFLOW = 2
OS_STACK_UNDERFLOW = 3

OSStackStatus = c_int
OSSwitchThreadCallback = u32
OSThreadDestructor = u32
OSThreadState = c_int

os_thread_p = u32
os_mutex_p = u32

class _OSThreadQueue(Structure):
    _fields_ = [
        ('head', os_thread_p),
        ('tail', os_thread_p),
    ]

class _OSThreadLink(Structure):
    _fields_ = [
        ('prev', os_thread_p),
        ('next', os_thread_p),
    ]

class _OSMutexQueue(Structure):
    _fields_ = [
        ('head', os_mutex_p),
        ('tail', os_mutex_p),
    ]

class _OSMutexLink(Structure):
    _fields_ = [
        ('next', os_mutex_p),
        ('prev', os_mutex_p),
    ]


class _OSThread(Structure):
    _fields_ = [
        ('context', OSContext),
        ('state', OSThreadState),
        ('next', u32),
        ('id', u32),
        ('priority', u32),
        ('profiler', u32),
        ('queue', u32), #POINTER(_OSThreadQueue)),
        ('link', _OSThreadLink),
        ('mutex', os_mutex_p),
        ('mutexQueue', _OSMutexQueue),
        ('stackTop', u32),
        ('stackBottom', u32),
        ('stackWarningOffset', u32),
        ('joinQueue', _OSThreadQueue),
        ('specific', (u32 * 3)),
        ('alarmForSleep', u32),
        ('destructor', OSThreadDestructor),
        ('userParameter', u32),
        ('systemErrno', c_int),
    ]
    
class OSThreadInfo(Structure):
    _fields_ = [
        ('isNeedRescheduling', u16),
        ('irqDepth', u16),
        ('current', u32), #POINTER(_OSThread)),
        ('list', u32), #POINTER(_OSThread)),
        ('switchCallback', u32),
    ]