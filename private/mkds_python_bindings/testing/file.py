from ctypes import *
from private.mkds_python_bindings.testing.archive import *
from private.mkds_python_bindings.testing.thread import *
from private.mkds_python_bindings.testing.types import *

FS_SEEK_SET = 0
FS_SEEK_CUR = 1
FS_SEEK_END = 2

FS_SEEK_SET = 0
FS_SEEK_CUR = 1
FS_SEEK_END = 2

FSSeekFileMode = c_int


class FSDirPos(Structure):
    _fields_ = [
        ('arc', u32), #POINTER(FSArchive)),
        ('own_id', u16),
        ('index', u16),
        ('pos', u32),
    ]

class FSFileID(Structure):
    _fields_ = [
        ('arc', u32), #POINTER(FSArchive)),
        ('file_id', u32),
    ]

class FSDirEntry(Structure):
    _fields_ = [
        ('is_directory', u32),
        ('name_len', u32),
        ('name', (c_char * 128)),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_99_5_(Union):
    _fields_ = [
        ('file_id', FSFileID),
        ('dir_id', FSDirPos),
    ]

class FSSeekDirInfo(Structure):
    _fields_ = [
        ('pos', FSDirPos),
    ]

class FSReadDirInfo(Structure):
    _fields_ = [
        ('p_entry', u32), #POINTER(FSDirEntry)),
        ('skip_string', BOOL),
    ]

class FSFindPathInfo(Structure):
    _fields_ = [
        ('pos', FSDirPos),
        ('path', u32), #POINTER(c_char)),
        ('find_directory', BOOL),
        ('result', union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_135_5_),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_135_5_(Union):
    _fields_ = [
        ('file', u32), #POINTER(FSFileID)),
        ('dir', u32), #POINTER(FSDirPos)),
    ]

class FSGetPathInfo(Structure):
    _fields_ = [
        ('buf', u32), #POINTER(u8)),
        ('buf_len', u32),
        ('total_len', u16),
        ('dir_id', u16),
    ]

class FSOpenFileFastInfo(Structure):
    _fields_ = [
        ('id', FSFileID),
    ]

class FSOpenFileDirectInfo(Structure):
    _fields_ = [
        ('top', u32),
        ('bottom', u32),
        ('index', u32),
    ]

class FSCloseFileInfo(Structure):
    _fields_ = [
        ('reserved', u32),
    ]

class FSReadFileInfo(Structure):
    _fields_ = [
        ('dst', u32),
        ('len_org', u32),
        ('len', u32),
    ]

class FSWriteFileInfo(Structure):
    _fields_ = [
        ('src', u32),
        ('len_org', u32),
        ('len', u32),
    ]

class FSFile(Structure):
    _fields_ = [
        ('link', FSFileLink),
        ('arc', u32), #POINTER(FSArchive)),
        ('stat', u32),
        ('command', FSCommandType),
        ('error', FSResult),
        ('queue', (_OSThreadQueue * 1)),
        ('prop', union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_217_5_),
        ('arg', union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_239_5_),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_217_5_(Union):
    _fields_ = [
        ('file', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_220_9_),
        ('dir', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_229_9_),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_220_9_(Structure):
    _fields_ = [
        ('own_id', u32),
        ('top', u32),
        ('bottom', u32),
        ('pos', u32),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_229_9_(Structure):
    _fields_ = [
        ('pos', FSDirPos),
        ('parent', u32),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_fs_file_h_239_5_(Union):
    _fields_ = [
        ('readfile', FSReadFileInfo),
        ('writefile', FSWriteFileInfo),
        ('seekdir', FSSeekDirInfo),
        ('readdir', FSReadDirInfo),
        ('findpath', FSFindPathInfo),
        ('getpath', FSGetPathInfo),
        ('openfilefast', FSOpenFileFastInfo),
        ('openfiledirect', FSOpenFileDirectInfo),
        ('closefile', FSCloseFileInfo),
    ]
