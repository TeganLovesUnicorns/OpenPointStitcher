"""The walls the bench shows and the drawings start from: the plane search's walls cut to where the scan really
has wall, plus short pieces only the from-above finder sees, with doubles (a second line just in front of a wall)
removed."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from .find_wall_runs import drop_crossing, wall_runs, without_past
from .find_walls import extract_walls
from .find_walls_above import walls_from_above

SAME_M = 0.10          # a line lies on another when within this across it ...
SAME_DEG = 10.0        # ... and turned less than this from it
STEP_M = 0.05          # lines are compared at points this far apart
DOUBLE_GAP_M = 0.35    # two parallel lines closer than this ...
DOUBLE_DEG = 5.0
DOUBLE_SHARE = 0.5     # ... overlapping along half the shorter are one wall: the better supported stays
SUPPORT_M = 0.06       # points this close to a line support it


def segment_row(w) -> dict:
    """A plane search wall as a plan line (centroid, normal and length to two ends)."""
    along = np.array([-w.normal[1], w.normal[0]])
    c = w.centroid[:2]
    p1, p2 = c - along * (w.length / 2.0), c + along * (w.length / 2.0)
    return {"i": int(w.index),
            "x1": round(float(p1[0]), 3), "y1": round(float(p1[1]), 3),
            "x2": round(float(p2[0]), 3), "y2": round(float(p2[1]), 3),
            "len": round(float(w.length), 2), "h": round(float(w.height), 2), "pts": int(w.support),
            "behind_m": round(float(w.behind_curtain_m), 3)}   # >0: found behind a curtain


def _ends(w: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.array([w["x1"], w["y1"]], float), np.array([w["x2"], w["y2"]], float)


def share_on(w: dict, others: list[dict]) -> float:
    """How much of w (by length) lies on some line in `others`, close and turned alike."""
    a, b = _ends(w)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return 0.0
    xy = a + np.linspace(0, 1, max(2, int(length / STEP_M) + 1))[:, None] * d
    hit = np.zeros(len(xy), bool)
    for o in others:
        p, q = _ends(o)
        e = q - p
        ee = float(e @ e)
        if ee < 1e-12 or abs(float(d @ e)) / (length * np.sqrt(ee)) < np.cos(np.radians(SAME_DEG)):
            continue
        t = np.clip(((xy - p) @ e) / ee, 0.0, 1.0)
        hit |= np.linalg.norm(xy - (p + t[:, None] * e), axis=1) < SAME_M
    return float(hit.mean())


def _support(w: dict, xy: np.ndarray) -> int:
    a, b = _ends(w)
    d = b - a
    length = float(np.linalg.norm(d)) or 1e-9
    u = d / length
    rel = xy - a
    t, off = rel @ u, rel @ np.array([-u[1], u[0]])
    return int(((t > 0) & (t < length) & (np.abs(off) < SUPPORT_M)).sum())


def drop_doubles(walls: list[dict], xy: np.ndarray) -> list[dict]:
    """Of two near-parallel lines closer than DOUBLE_GAP_M that overlap along most of the shorter, keep the one with
    more points on it. Walls behind curtains are never dropped (their own points are hidden)."""
    support = [_support(w, xy) for w in walls]
    order = sorted(range(len(walls)), key=lambda i: (float(walls[i].get("behind_m", 0)) <= 0, -support[i]))
    kept: list[int] = []
    for i in order:
        a, b = _ends(walls[i])
        la = float(np.linalg.norm(b - a)) or 1e-9
        u = (b - a) / la
        double = False
        for j in kept:
            p, q = _ends(walls[j])
            lb = float(np.linalg.norm(q - p)) or 1e-9
            e = (q - p) / lb
            if abs(float(u @ e)) < np.cos(np.radians(DOUBLE_DEG)):
                continue
            nv = np.array([-e[1], e[0]])
            if max(abs(float((a - p) @ nv)), abs(float((b - p) @ nv))) > DOUBLE_GAP_M:
                continue
            s = sorted([float((a - p) @ e), float((b - p) @ e)])
            if min(s[1], lb) - max(s[0], 0.0) > DOUBLE_SHARE * min(la, lb):
                double = True
                break
        if not double or float(walls[i].get("behind_m", 0)) > 0:
            kept.append(i)
    return [walls[i] for i in sorted(kept)]


def plane_rows(structure: o3d.geometry.PointCloud, max_planes: int) -> list[dict]:
    """The plane search's walls as full-length plan lines: what the wall reports measure with. #drawn-walls"""
    return [segment_row(w) for w in extract_walls(structure, level_rotation=None, max_walls=max_planes)]


def drawn_walls(structure: o3d.geometry.PointCloud, floor_z: float, max_planes: int,
                rows: list[dict] | None = None) -> list[dict]:
    """Walls (payload rows: i, x1, y1, x2, y2, len, h, pts, behind_m, kind, z_lo, z_hi) of one scan from its levelled
    cloud with normals. `max_planes`: how many planes the plane search may take (the wall search setting); `rows`:
    plane_rows already found for it."""
    pts = np.asarray(structure.points).copy()
    pts[:, 2] -= floor_z
    walls = wall_runs(pts, plane_rows(structure, max_planes) if rows is None else rows, min_level_pts=1,
                      keep_past=True)
    above = [{**w, "i": -1, "h": 0.0, "pts": 0, "behind_m": 0.0}
             for w in walls_from_above(pts, np.asarray(structure.normals))]
    for w in sorted(wall_runs(pts, above, min_level_pts=1, keep_past=True),
                    key=lambda w: -float(np.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]))):
        if share_on(w, walls) < 0.5:                     # only what the plane search did not already find
            walls.append(w)
    walls = without_past(drop_crossing(drop_doubles(walls, pts[:, :2])))   # crossings between the two finders too
    out = []
    for k, w in enumerate(walls):
        z_lo, z_hi = w.get("z_lo"), w.get("z_hi")
        h = round(float(z_hi - z_lo), 2) if z_lo is not None and z_hi is not None else w.get("h", 0.0)
        out.append({**{key: v for key, v in w.items() if key != "votes"}, "i": k, "h": h,
                    "len": round(float(np.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"])), 2),
                    "pts": _support(w, pts[:, :2])})
    return out
