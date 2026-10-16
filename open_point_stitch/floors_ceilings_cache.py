"""One scan's floor and ceiling pieces, worked out once from every point and kept in the project's preview cache.
Shared by the bench (to draw them), the height settle (to match on them) and the 3D export."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .exclusions import filter_scan
from .find_floors_ceilings import find_floors_and_ceilings, levelled_points
from .placement import Exclusion
from .pts_file import find_scans, read_pts

log = logging.getLogger(__name__)

VERSION = 2           # bump when find_floors_ceilings' output changes, so old saved pieces are not reused
SUBDIR = "floors_ceilings"


def _key(path: str, level_rotation, floor_z: float, exclusions: Optional[list[Exclusion]]) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime, "version": VERSION,
            "level": [[round(float(v), 8) for v in row] for row in level_rotation],
            "floor_z": round(float(floor_z), 5),
            "exclusions": sorted(json.dumps(vars(e), sort_keys=True, default=list) for e in exclusions or [])}


def cache_path(cache_dir: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "scan"
    return os.path.join(cache_dir, SUBDIR, slug + ".json")


def cached_pieces(path: str, cache_dir: str, level_rotation, floor_z: float,
                  exclusions: Optional[list[Exclusion]] = None) -> list[dict]:
    """The scan's pieces as JSON rows, in its levelled frame (heights above its detected floor). Exclusions and
    review removals are cut first. Reads the saved copy when nothing it depends on has changed."""
    name = os.path.splitext(os.path.basename(path))[0]
    key = _key(path, level_rotation, floor_z, exclusions)
    cpath = cache_path(cache_dir, name)
    try:
        with open(cpath) as fh:
            saved = json.load(fh)
        if saved.get("_key") == key:
            return saved["pieces"]
    except (OSError, ValueError):
        pass
    pieces = [p.to_json() for p in find_floors_and_ceilings(levelled_scan(path, level_rotation, floor_z, exclusions))]
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    tmp = cpath + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"_key": key, "pieces": pieces}, fh)
    os.replace(tmp, cpath)
    return pieces


def levelled_scan(path: str, level_rotation, floor_z: float, exclusions: Optional[list[Exclusion]] = None):
    """Every point of the scan, exclusions and review removals cut, levelled, its detected floor at height 0."""
    scan = read_pts(path)
    if exclusions:
        scan = filter_scan(scan, exclusions)
    return levelled_points(scan.points, level_rotation, floor_z)


def pieces_for_scans(scans_dir: str, cache_dir: str, payloads: dict[str, dict], saved: dict,
                     names, progress=None) -> dict[str, list[dict]]:
    """{name: pieces} for each named scan that has a preview; a scan that fails is left out and logged. Works in
    a job's own process as well as in the server. `progress(i, n, name)` is told before each scan."""
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in find_scans(scans_dir)}
    todo = [n for n in names if n in payloads and n in paths]
    out = {}
    for i, n in enumerate(todo):
        if progress:
            progress(i, len(todo), n)
        try:
            pieces = cached_pieces(paths[n], cache_dir, payloads[n]["level_rotation"], payloads[n]["floor_z"],
                                   scan_exclusions(saved, n))
        except Exception:  # noqa: BLE001 - one scan's pieces must not stop a run or an export
            log.exception("%s: finding floors and ceilings failed", n)
            continue
        if pieces:
            out[n] = pieces
    return out


def scan_exclusions(saved: dict, name: str) -> list[Exclusion]:
    """Every exclusion and review removal on this scan in the project file: both are kept out of floor finding."""
    return [Exclusion.from_dict(e) for e in saved.get("exclusions", []) if e.get("scan") == name]
