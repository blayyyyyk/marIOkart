from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.gsPlatformThread import *

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSCore_IN_USE = 0
GSCore_SHUTDOWN_PENDING = 1
GSCore_SHUTDOWN_COMPLETE = 2

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4

GSTaskResult_None = 0
GSTaskResult_InProgress = 1
GSTaskResult_Canceled = 2
GSTaskResult_TimedOut = 3
GSTaskResult_Finished = 4


class GSCoreMgr(Structure):
    _fields_ = [
        ('mRefCount', gsi_u32),
        ('mIsStaticInitComplete', gsi_bool),
        ('mIsInitialized', gsi_bool),
        ('mIsShuttingDown', gsi_bool),
        ('mQueueCrit', GSICriticalSection),
        ('mTaskArray', (POINTER32(GSTask) * 40)),
    ]
GSTaskResult = c_int
GSCoreValue = c_int

class GSTask(Structure):
    _fields_ = [
        ('mId', c_int),
        ('mTimeout', gsi_time),
        ('mStartTime', gsi_time),
        ('mAutoThink', gsi_bool),
        ('mIsStarted', gsi_i32),
        ('mIsRunning', gsi_i32),
        ('mIsCanceled', gsi_i32),
        ('mIsCallbackPending', gsi_i32),
        ('mTaskData', c_void_p32),
        ('mExecuteFunc', GSTaskExecuteFunc),
        ('mCallbackFunc', GSTaskCallbackFunc),
        ('mCancelFunc', GSTaskCancelFunc),
        ('mCleanupFunc', GSTaskCleanupFunc),
        ('mThinkFunc', GSTaskThinkFunc),
    ]
GSTaskThinkFunc = c_void_p32
GSTaskCallbackFunc = c_void_p32
GSTaskCleanupFunc = c_void_p32
GSTaskExecuteFunc = c_void_p32
GSTaskCancelFunc = c_void_p32
