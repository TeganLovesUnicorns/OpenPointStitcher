"""Walls found the way a person finds them in plan view with the ceiling sliced off: 5 cm squares of sideways-facing
points stacked tall vote for a line (direction from the points' facing, place from where they are), the strongest
lines win, and each is cut where its tall squares stop. No random sampling, so the same scan always gives the same
walls. Catches short pieces (returns, reveals, stubs) the plane search misses."""
from __future__ import annotations

import numpy as np

CELL_M = 0.05          # plan squares
LEVEL_M = 0.05         # height levels inside a square
MIN_TALL_M = 0.8       # a square counts when its points span at least this many levels' worth of height
MAX_NZ = 0.3           # a point faces sideways when its surface tips less than this from vertical
ABOVE_FLOOR_M = 0.08   # floor points left out
MIN_AGREE = 0.5        # a square's points must roughly agree on which way they face
ANGLE_BIN_DEG = 1.0    # vote grid: direction ...
SEAM_DEG = 45.0        # ... counted from -45 to 135 degrees, so walls along the grid never straddle its wrap
OFFSET_BIN_M = 0.04    # ... and place across the line
BAND_M = 0.05          # squares this close to a winning line belong to it
ANGLE_TOL_DEG = 6.0    # ... when they face within this of it
MIN_VOTES = 13.0       # weakest line kept (sum of its squares' heights in metres, smoothed; #walls-above-grid)
MAX_LINES = 60
CLOSE_M = 1.0          # gaps along a line shorter than this do not split it
MIN_LEN_M = 0.4
FIT_MIN_M = 1.0       # lines longer than this take their direction from a straight-line fit
SETTLE_M = 0.06       # a found line moves across to the middle of the sideways points this close to it


def _squares(points: np.ndarray, normals: np.ndarray):
    """(centres k x 2, facing angle in degrees 0..180, height spanned) of every plan square, from sideways points."""
    xy, z, n = points[:, :2], points[:, 2], normals[:, :2]
    if len(xy) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    ij = np.floor(xy / CELL_M).astype(np.int64) + 2 ** 20
    key = ij[:, 0] * 2 ** 21 + ij[:, 1]
    levels = np.clip(np.floor(z / LEVEL_M).astype(np.int64), 0, 4095)
    _, tall_count = np.unique(np.unique(key * 4096 + levels) // 4096, return_counts=True)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    doubled = 2 * np.arctan2(n[order, 1], n[order, 0])      # doubled, so a face seen from either side agrees
    c, s = np.add.reduceat(np.cos(doubled), starts), np.add.reduceat(np.sin(doubled), starts)
    agree = np.hypot(c, s) / np.diff(np.r_[starts, len(ks)])
    angle = (np.degrees(np.arctan2(s, c)) / 2) % 180.0
    uk = ks[starts]
    centres = np.column_stack([(uk // 2 ** 21 - 2 ** 20 + 0.5) * CELL_M, (uk % 2 ** 21 - 2 ** 20 + 0.5) * CELL_M])
    tall = tall_count * LEVEL_M
    good = (tall >= MIN_TALL_M) & (agree > MIN_AGREE)
    return centres[good], angle[good], tall[good]


def _settle(a: np.ndarray, b: np.ndarray, side_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move line a-b across onto the middle of the points on it: squares alone place it only to within half a
    square, and wall reports compare walls to 3 cm."""
    length = float(np.linalg.norm(b - a))
    u = (b - a) / length
    n = np.array([-u[1], u[0]])
    rel = side_xy - a
    t, off = rel @ u, rel @ n
    near = (t > 0) & (t < length) & (np.abs(off) < SETTLE_M)
    if near.sum() < 10:
        return a, b
    shift = float(np.median(off[near]))
    return a + n * shift, b + n * shift


def walls_from_above(points: np.ndarray, normals: np.ndarray) -> list[dict]:
    """Wall lines {x1, y1, x2, y2, votes} from levelled points (z above the floor) and their normals. The grid is laid
    out from the scan itself (its middle, turned to its main wall direction), so the same scan moved or turned gives
    the same walls."""
    side = (np.abs(normals[:, 2]) < MAX_NZ) & (points[:, 2] > ABOVE_FLOOR_M)
    if not side.any():
        return []
    p, n = points[side], normals[side]                   # copies: only sideways points are used from here on
    middle = p[:, :2].mean(axis=0)
    four = 4 * np.arctan2(n[:, 1], n[:, 0])              # walls at right angles agree when x4. #walls-above-grid
    turn = float(np.arctan2(np.sin(four).sum(), np.cos(four).sum()) / 4)
    rot = np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]])   # scan -> grid
    p[:, :2] = (p[:, :2] - middle) @ rot.T
    n[:, :2] = n[:, :2] @ rot.T
    out = []
    for w in _lines(p, n):
        a = rot.T @ [w["x1"], w["y1"]] + middle
        b = rot.T @ [w["x2"], w["y2"]] + middle
        out.append({**w, "x1": round(float(a[0]), 3), "y1": round(float(a[1]), 3),
                    "x2": round(float(b[0]), 3), "y2": round(float(b[1]), 3)})
    return out


def _lines(points: np.ndarray, normals: np.ndarray) -> list[dict]:
    """walls_from_above in the grid's own frame, from its sideways points only."""
    centres, angle, weight = _squares(points, normals)
    side_xy = points[:, :2]
    walls: list[dict] = []
    used = np.zeros(len(centres), bool)
    n_angles = int(round(180 / ANGLE_BIN_DEG))
    for _ in range(MAX_LINES):
        live = ~used
        if live.sum() < 3:
            break
        shifted = (angle[live] + SEAM_DEG) % 180.0       # the seam sits between the grid's two wall directions
        ai = np.round(shifted / ANGLE_BIN_DEG).astype(int)
        seam = ai >= n_angles                            # just under the seam: the same as just over it
        ai[seam] = 0
        theta = np.radians(shifted - 180.0 * seam - SEAM_DEG)
        offset = centres[live, 0] * np.cos(theta) + centres[live, 1] * np.sin(theta)
        oi = np.round(offset / OFFSET_BIN_M).astype(int)
        o0 = oi.min()
        votes = np.zeros((n_angles + 2, oi.max() - o0 + 3))           # a border of empty bins: no wrapping round
        np.add.at(votes, (ai + 1, oi - o0 + 1), weight[live])
        smooth = sum(votes[1 + da:n_angles + 1 + da, 1 + do:votes.shape[1] - 1 + do]
                     for da in (-1, 0, 1) for do in (-1, 0, 1))
        best = np.unravel_index(np.argmax(smooth), smooth.shape)
        if smooth[best] < MIN_VOTES:
            break
        theta_deg = best[0] * ANGLE_BIN_DEG - SEAM_DEG
        t = np.radians(theta_deg)
        nv = np.array([np.cos(t), np.sin(t)])
        along = np.array([-nv[1], nv[0]])
        turn = np.abs((angle - theta_deg + 90) % 180 - 90)
        place = (best[1] + o0) * OFFSET_BIN_M
        member = live & (np.abs(centres @ nv - place) < BAND_M) & (turn < ANGLE_TOL_DEG)
        if member.sum() < 3:
            # a peak made only by smoothing: retire its squares so the next pass moves on
            used |= live & (np.abs(centres @ nv - place) < OFFSET_BIN_M) & (turn < ANGLE_BIN_DEG)
            continue
        # direction from where its squares lie (the vote grid's 1 degree steps tie between neighbours; the points'
        # own facing is too noisy on real walls), for lines long enough to fit
        xy = centres[member]
        if np.ptp(xy @ along) > FIT_MIN_M:
            spread = np.cov((xy - np.average(xy, axis=0, weights=weight[member])).T, aweights=weight[member])
            along = np.linalg.eigh(spread)[1][:, 1]
            nv = np.array([along[1], -along[0]])
            theta_deg = float(np.degrees(np.arctan2(nv[1], nv[0]))) % 180.0
            nv = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])
            along = np.array([-nv[1], nv[0]])
            turn = np.abs((angle - theta_deg + 90) % 180 - 90)
        place = float(np.average(centres[member] @ nv, weights=weight[member]))
        member = live & (np.abs(centres @ nv - place) < BAND_M) & (turn < ANGLE_TOL_DEG)
        used |= member
        s = np.sort(centres[member] @ along)
        for piece in np.split(s, np.flatnonzero(np.diff(s) > CLOSE_M) + 1):
            if len(piece) == 0 or piece[-1] - piece[0] + CELL_M < MIN_LEN_M:
                continue
            a = nv * place + along * (piece[0] - CELL_M / 2)
            b = nv * place + along * (piece[-1] + CELL_M / 2)
            a, b = _settle(a, b, side_xy)
            walls.append({"x1": round(float(a[0]), 3), "y1": round(float(a[1]), 3),
                          "x2": round(float(b[0]), 3), "y2": round(float(b[1]), 3), "votes": len(piece)})
    return walls
