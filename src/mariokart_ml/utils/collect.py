from pathlib import Path


def collect_dsm(in_path: Path, course_name: str | None = None) -> list[Path]:
    movie_paths = []
    if in_path.is_dir():
        for mp in in_path.rglob("*.dsm"):
            if course_name is not None and course_name not in mp.name:
                continue
            movie_paths.append(mp)
    elif in_path.is_file() and in_path.suffix == ".dsm":
        movie_paths.append(in_path)

    return movie_paths


def collect_dat(in_path: Path, course_name: str | None = None):
    dat_paths = set([])
    for dat_path in in_path.rglob("*.dat"):
        if course_name is not None:
            if course_name not in str(dat_path.parent).lower():
                continue

        dat_paths.add(dat_path.parent)

    return dat_paths
