"""Curtains, and anything else hung in front of a wall, get detected as the wall. The ceiling does not stop at a
curtain but carries on to the real wall, so where the ceiling ends along a detected wall says where the real wall
is. Used by both wall detectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

CEILING_BAND_M = 0.06         # ceiling points: this close to the ceiling height, facing up or down
BEHIND_M = (0.03, 0.30)       # the ceiling reaching this far past the detected wall means a wall behind it
SECTION_M = 0.25              # the wall's length is checked in sections this long
MIN_SECTIONS = 4              # at least this many sections must have ceiling points near the wall
AGREE_M = 0.03                # sections whose ceiling edge is this close to the median agree
AGREE_SHARE = 0.6             # and at least this share of sections must agree
MAX_REFIT_DEG = 15.0          # the refit wall may turn this much from the detected one, no more
FACE_PER_M = 30               # upright points per metre of wall the real wall must show just under the ceiling


@dataclass
class TrueWall:
    normal: np.ndarray        # unit 2D normal (same canonical sign as the detected wall)
    offset: float             # plane: normal . xy = offset
    behind_m: float           # how far behind the detected plane the ceiling ends
    sections: int             # sections that agreed


def find_true_wall(pts: np.ndarray, nz: np.ndarray, normal: np.ndarray, offset: float, wall_pts: np.ndarray,
                   origin_xy: np.ndarray = np.zeros(2), debug: bool = False) -> Optional[TrueWall]:
    """`pts`: the whole levelled cloud and `nz` the |z| of each point's normal; `normal`, `offset`: a detected wall;
    `wall_pts`: its inliers. Returns the real wall where the ceiling ends behind it, or None (the usual case)."""
    n = np.asarray(normal, float)
    along = np.array([-n[1], n[0]])
    t = wall_pts[:, :2] @ along
    t0, t1 = float(t.min()), float(t.max())
    if t1 - t0 < MIN_SECTIONS * SECTION_M:
        return None
    room_side = np.sign(float(origin_xy @ n) - offset) or 1.0       # the scan stood on this side
    s = (pts[:, :2] @ n - offset) * -room_side                      # distance behind the detected plane
    ta = pts[:, :2] @ along
    near = (ta > t0) & (ta < t1) & (s > -0.5) & (s < BEHIND_M[1] + 0.1)
    flat = near & (nz > 0.9)
    if flat.sum() < 100:
        return None
    ceiling = float(np.percentile(pts[flat, 2], 98))
    ceil = flat & (pts[:, 2] > ceiling - CEILING_BAND_M)
    edges, where = [], []
    for a in np.arange(t0, t1 - SECTION_M / 2, SECTION_M):
        sec = ceil & (ta >= a) & (ta < a + SECTION_M)
        if sec.sum() < 10:
            continue
        e = float(np.percentile(s[sec], 97))                        # how far the ceiling reaches in this section
        edges.append(e)
        where.append(a + SECTION_M / 2)
    if len(edges) < MIN_SECTIONS:
        return None
    edges, where = np.array(edges), np.array(where)
    if debug:
        print(f"    ceiling at {ceiling:.2f}; ceiling edge behind the wall per section (cm): "
              + " ".join(f"{e * 100:.0f}" for e in edges))
    # The ceiling edge need not be parallel to the detected surface (curtains hang at an angle): find the straight
    # line most sections agree on, trying the line through every pair of sections.
    best = None
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            slope = (edges[j] - edges[i]) / (where[j] - where[i])
            if abs(np.degrees(np.arctan(slope))) > MAX_REFIT_DEG:
                continue
            res = edges - (edges[i] + slope * (where - where[i]))
            inl = np.abs(res) <= AGREE_M
            if best is None or inl.sum() > best.sum():
                best = inl
    if best is None or best.sum() < MIN_SECTIONS or best.mean() < AGREE_SHARE:
        return None
    a, b = np.polyfit(where[best], edges[best], 1)[::-1]
    mid_t = (t0 + t1) / 2
    behind_mid = float(a + b * mid_t)
    if not (BEHIND_M[0] <= behind_mid <= BEHIND_M[1]):
        return None
    # A real wall shows as an upright surface right at that line, just under the ceiling. A wall that stops short of
    # the ceiling, with the ceiling carrying on past it, has nothing there.
    s_line = a + b * ta
    upper = (nz < 0.3) & (ta > t0) & (ta < t1) & (pts[:, 2] > ceiling - 0.30) & (pts[:, 2] < ceiling - 0.02)
    close = upper & (np.abs(s - s_line) < 0.05)
    if close.sum():                                                  # the wall face itself sets the exact position
        a += float(np.median(s[close] - s_line[close]))
        s_line = a + b * ta
        behind_mid = float(a + b * mid_t)
    face = upper & (np.abs(s - s_line) < 0.025)
    if debug:
        print(f"    edge line {behind_mid * 100:.0f} cm behind at the middle, {int(best.sum())} sections on it, "
              f"{int(face.sum())} upright points at it under the ceiling")
    if face.sum() < FACE_PER_M * (t1 - t0):
        return None
    sd = (a + b * where[best]) * -room_side + offset
    q = np.outer(sd, n) + np.outer(where[best], along)
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n_new = vt[1] if vt[1] @ n >= 0 else -vt[1]
    return TrueWall(normal=n_new, offset=float(c @ n_new), behind_m=behind_mid, sections=int(best.sum()))
