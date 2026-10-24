"""Turn dropped files and folders into test scenes: ToF (PCD or RAW) + optional SL image + ground truth."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT = CODE_DIR.parent
PCD_EXT = {".pcd"}
IMG_EXT = {".exr", ".png", ".tif", ".tiff"}
RAW_EXT = {".npy"}


@dataclass
class Scene:
    name: str
    tof: Optional[Path] = None     # .pcd
    raw: Optional[Path] = None     # .npy frame or a folder of frames (Pos*_set0)
    sl: Optional[Path] = None      # .exr / .png; None → gray values of the cloud
    gt: Optional[float] = None     # [m]
    enabled: bool = True

    def to_json(self):
        """Session form: paths relative to the repository root where possible."""
        d = asdict(self)
        for k in ("tof", "raw", "sl"):
            d[k] = to_session_path(d[k])
        return d

    @classmethod
    def from_json(cls, d):
        """Inverse of to_json."""
        d = dict(d)
        for k in ("tof", "raw", "sl"):
            d[k] = from_session_path(d.get(k))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def to_session_path(p):
    """Path for a session file: POSIX, relative to ROOT when inside the repository."""
    if p is None:
        return None
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def from_session_path(s):
    """Inverse of to_session_path, valid on Windows and Linux."""
    if not s:
        return None
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def _key(path):
    """Pairing key: stem without SL_/ToF_ prefix."""
    return re.sub(r"^(sl|tof)_", "", Path(path).stem, flags=re.IGNORECASE)


def _is_raw_dir(d):
    return d.is_dir() and any(p.suffix.lower() in RAW_EXT for p in d.iterdir())


def collect(paths, parse_gt=None):
    """Scenes from a list of files/folders; folders are searched recursively, a folder of .npy frames is one RAW scene."""
    tofs, sls, raws = [], [], []
    for p in map(Path, paths):
        if p.is_dir():
            if _is_raw_dir(p):
                raws.append(p)
                continue
            for sub in sorted(p.rglob("*")):
                if sub.is_dir() and _is_raw_dir(sub):
                    raws.append(sub)
                elif sub.is_file() and sub.parent not in raws:
                    ext = sub.suffix.lower()
                    (tofs if ext in PCD_EXT else sls if ext in IMG_EXT else []).append(sub)
        elif p.is_file():
            ext = p.suffix.lower()
            (tofs if ext in PCD_EXT else sls if ext in IMG_EXT else raws if ext in RAW_EXT else []).append(p)

    used = set()

    def match_sl(src):
        same_dir = [s for s in sls if s not in used and s.parent == src.parent and _key(s) == _key(src)]
        anywhere = [s for s in sls if s not in used and _key(s) == _key(src)]
        if same_dir or len(anywhere) == 1:
            sl = (same_dir or anywhere)[0]
            used.add(sl)
            return sl
        return None

    scenes = [Scene(_key(t), tof=t, sl=match_sl(t)) for t in tofs]
    scenes += [Scene(r.stem if r.is_file() else r.name, raw=r, sl=match_sl(r)) for r in raws]
    # one ToF + one SL left in a folder with different names: they still belong together
    for s in scenes:
        src = s.tof or s.raw
        if s.sl is None:
            free = [x for x in sls if x not in used and x.parent == src.parent]
            alone = sum((x.tof or x.raw).parent == src.parent for x in scenes) == 1
            if len(free) == 1 and alone:
                s.sl = free[0]
                used.add(free[0])

    names = [s.name for s in scenes]
    for s in scenes:
        if names.count(s.name) > 1:
            s.name = f"{(s.tof or s.raw).parent.name}/{s.name}"
        if parse_gt is not None:
            s.gt = parse_gt(str(s.sl or s.tof or s.raw))
    return scenes, [s for s in sls if s not in used]
