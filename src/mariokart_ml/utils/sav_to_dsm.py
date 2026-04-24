from pathlib import Path

from desmume.emulator import DeSmuME

from ..config import ROM_PATH
from ..utils.recording import convert_to_dsm
from ..utils.recording.extract_ghost import (
    COURSE_ABBREVIATIONS,
    data_to_dict,
    get_normal_ghost_data,
    has_ghost,
)


def sav_to_dsm_file(emu: DeSmuME, sav_path: Path, out_path: Path, verbose=False):
    # read byte contents from save file
    contents: bytearray
    with open(sav_path, "rb") as fs:
        contents = bytearray(fs.read())

    assert contents, "contents could not be read"
    course_data_entries = {}
    for course_i in range(32):
        # ensure ghost exists
        if not has_ghost(contents, course_i):
            continue

        # extract ghost inputs course at current index
        ghost_data = get_normal_ghost_data(contents, course_i)

        course_data = data_to_dict(ghost_data)
        course_id = COURSE_ABBREVIATIONS[course_i]
        course_data_entries[course_id] = course_data

    if verbose:
        print(f"{len(course_data_entries)} entries found at {str(sav_path)}")

    for course_id, course_data in course_data_entries.items():
        out_file_path = out_path / f"{course_id.lower()}_{sav_path.stem}.dsm"
        convert_to_dsm(emu, str(out_file_path), **course_data)
        if not verbose:
            continue
        print(f"saving inputs to {out_file_path}")


def sav_to_dsm(in_path: Path | list[Path], out_path: Path | None = None, verbose=False):
    emu = DeSmuME()
    emu.open(str(ROM_PATH))

    def _sav_to_dsm(ip: Path, op: Path | None):
        if ip.is_dir():
            sav_found = False
            for sav_path in ip.rglob("*.sav"):
                sav_found = True
                rel_path = sav_path.relative_to(ip)
                dest_path = (op or ip) / rel_path.parent / rel_path.stem
                dest_path.mkdir(parents=True, exist_ok=True)
                sav_to_dsm_file(emu, sav_path, dest_path, verbose)

            if verbose and not sav_found:
                print("no sav files found in directory")
        elif ip.is_file() and ip.suffix == ".sav":
            dest_path = (op or ip) / ip.stem
            dest_path.mkdir(parents=True, exist_ok=True)
            sav_to_dsm_file(emu, ip, dest_path, verbose)

    if isinstance(in_path, list):
        for ip in in_path:
            _sav_to_dsm(ip, out_path)
    else:
        _sav_to_dsm(in_path, out_path)

    emu.close()
