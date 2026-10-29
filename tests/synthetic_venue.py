"""A generated two-room venue scanned twice the way a phone scans it, with the true alignment between the scans known
exactly.

A square hall with four columns, a 45 degree wall and a stage riser, and a bar room behind it, joined by an ordinary
door: jambs through a 20 cm wall, casing on both faces, the leaf set in from the hall face. The door is closed in the
hall scan and open in the bar scan, which sees the hall only through the doorway, from the threshold.

Each scan sees only what its walking path has a line of sight to, within the scanner's range, and carries what a phone
scan carries: density falling off with distance, noise mostly along the line of sight, drift over the walk (surfaces
seen twice land twice, a little apart), points smeared off edges toward what is behind them, stray points in mid-air,
a mirror's phantom room, a person walking through, chairs moved between the scans, and lighting and exposure that
differ between them. The bar scan's own frame sits about 28 m and 157 degrees from the hall's, as in the real pair
this stands in for. Same seed, same points, on any machine."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

HALL_M = 10.0               # square, with a square grid of columns: a quarter turn is a plausible wrong answer
HALL_CEILING_M = 4.7
BAR_CEILING_M = 3.0
WALL_Y = (-0.2, 0.0)        # the wall between the rooms: bar face, hall face
DOOR_X = (4.0, 4.95)
DOOR_TOP = 2.1
CASING_W, CASING_PROUD = 0.07, 0.015
LEAF_T, LEAF_SET_IN = 0.045, 0.025
BAR_X = (3.0, 7.2)          # the bar room's side walls; its far wall is at BAR_BACK_Y
BAR_BACK_Y = -4.2
MIRROR = (3.8, 6.4, 1.2, 2.2)  # behind the bar, on the bar room's far wall: x0, x1, z0, z1


@dataclass
class _Surface:
    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    colour: tuple
    holes: tuple = ()          # (a0, a1, b0, b1) in the e1/e2 unit square: no points there
    cut_corner_m: float = 0.0  # the hall's floor and ceiling: nothing where x + y < this (the 45 degree corner)
    edgy: bool = False         # a narrow face whose edges are depth jumps: its edge points smear
    pattern: object = None     # optional: venue points (n x 3) -> their colours (n x 3), in place of colour


def _rect(o, e1, e2, colour, holes=(), cut=0.0, edgy=False) -> _Surface:
    return _Surface(np.asarray(o, float), np.asarray(e1, float), np.asarray(e2, float), colour, holes, cut, edgy)


def _box(x0, x1, y0, y1, z0, z1, colour, faces="tnsew", edgy=True) -> list[_Surface]:
    """The listed faces of a box: t(op), n/s (the y1/y0 sides), e/w (the x1/x0 sides)."""
    out = []
    if "t" in faces:
        out.append(_rect((x0, y0, z1), (x1 - x0, 0, 0), (0, y1 - y0, 0), colour, edgy=edgy))
    if "s" in faces:
        out.append(_rect((x0, y0, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0), colour, edgy=edgy))
    if "n" in faces:
        out.append(_rect((x0, y1, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0), colour, edgy=edgy))
    if "w" in faces:
        out.append(_rect((x0, y0, z0), (0, y1 - y0, 0), (0, 0, z1 - z0), colour, edgy=edgy))
    if "e" in faces:
        out.append(_rect((x1, y0, z0), (0, y1 - y0, 0), (0, 0, z1 - z0), colour, edgy=edgy))
    return out


def _hall() -> list[_Surface]:
    h, n, wall = HALL_CEILING_M, HALL_M, (0.62, 0.58, 0.52)
    door = ((DOOR_X[0] - 2.5) / (n - 2.5), (DOOR_X[1] - 2.5) / (n - 2.5), 0.0, DOOR_TOP / h)
    s = [
        _rect((0, 0, 0), (n, 0, 0), (0, n, 0), (0.30, 0.29, 0.28), cut=2.5),
        _rect((0, 0, h), (n, 0, 0), (0, n, 0), (0.08, 0.08, 0.09), cut=2.5),     # black ceiling
        _rect((2.5, 0, 0), (n - 2.5, 0, 0), (0, 0, h), wall, holes=(door,)),
        _rect((0, n, 0), (n, 0, 0), (0, 0, h), wall),
        _rect((0, 2.5, 0), (0, n - 2.5, 0), (0, 0, h), wall),
        _rect((n, 0, 0), (0, n, 0), (0, 0, h), wall, holes=((0.4, 0.5, 0.0, 2.1 / h),)),   # a fire exit
        _rect((0, 2.5, 0), (2.5, -2.5, 0), (0, 0, h), (0.70, 0.66, 0.60)),        # the 45 degree wall
    ]
    for cx in (3.5, 6.5):
        for cy in (3.5, 6.5):
            s += _box(cx - 0.2, cx + 0.2, cy - 0.2, cy + 0.2, 0, h, (0.75, 0.75, 0.72), faces="nsew")
    s += _box(7.0, n, 7.5, n, 0, 0.6, (0.20, 0.15, 0.12), faces="tsw")
    s += _box(1.2, 2.1, 3.8, 4.7, 0, 0.75, (0.45, 0.30, 0.20))
    return s


def _bar() -> list[_Surface]:
    h, wall = BAR_CEILING_M, (0.50, 0.42, 0.35)
    (x0, x1), y0, y1 = BAR_X, BAR_BACK_Y, WALL_Y[0]
    w = x1 - x0
    door = ((DOOR_X[0] - x0) / w, (DOOR_X[1] - x0) / w, 0.0, DOOR_TOP / h)
    mirror = ((MIRROR[0] - x0) / w, (MIRROR[1] - x0) / w, MIRROR[2] / h, MIRROR[3] / h)
    s = [
        _rect((x0, y0, 0), (w, 0, 0), (0, y1 - y0, 0), (0.25, 0.22, 0.20)),
        _rect((x0, y0, h), (w, 0, 0), (0, y1 - y0, 0), (0.15, 0.15, 0.15)),
        _rect((x0, y1, 0), (w, 0, 0), (0, 0, h), wall, holes=(door,)),
        _rect((x0, y0, 0), (w, 0, 0), (0, 0, h), wall, holes=(mirror,)),         # a mirror reflects, no points
        _rect((x0, y0, 0), (0, y1 - y0, 0), (0, 0, h), wall),
        _rect((x1, y0, 0), (0, y1 - y0, 0), (0, 0, h), wall),
    ]
    s += _box(3.6, 6.6, -3.6, -3.0, 0, 1.1, (0.35, 0.18, 0.10), faces="tnwe")     # bar counter
    s += _box(6.8, 7.2, -2.6, -1.0, 1.2, 2.0, (0.55, 0.55, 0.50), faces="tswn")   # a shelf
    return s


def _doorway(open_: bool) -> list[_Surface]:
    """Jambs and head through the wall, casing proud of both faces, and the leaf: set in from the hall face when
    closed, swung into the bar room when open."""
    (x0, x1), (yb, yh), trim = DOOR_X, WALL_Y, (0.85, 0.84, 0.80)
    s = [_rect((x0, yb, 0), (0, yh - yb, 0), (0, 0, DOOR_TOP), trim, edgy=True),
         _rect((x1, yb, 0), (0, yh - yb, 0), (0, 0, DOOR_TOP), trim, edgy=True),
         _rect((x0, yb, DOOR_TOP), (x1 - x0, 0, 0), (0, yh - yb, 0), trim, edgy=True)]
    top = DOOR_TOP + CASING_W
    for face, out in ((yh, CASING_PROUD), (yb, -CASING_PROUD)):
        lo, hi = sorted((face, face + out))
        s += _box(x0 - CASING_W, x0, lo, hi, 0, top, trim, faces="nsw" if out > 0 else "nse")
        s += _box(x1, x1 + CASING_W, lo, hi, 0, top, trim, faces="nse" if out > 0 else "nsw")
        s += _box(x0 - CASING_W, x1 + CASING_W, lo, hi, DOOR_TOP, top, trim, faces="tns")
    leaf = (0.55, 0.36, 0.22)
    if open_:
        s += _box(x0 - LEAF_T, x0, yb - (x1 - x0), yb, 0, DOOR_TOP - 0.01, leaf, faces="ewts")
    else:
        y_hall = yh - LEAF_SET_IN
        s += _box(x0, x1, y_hall - LEAF_T, y_hall, 0, DOOR_TOP - 0.01, leaf, faces="ns")
    return s


def _chairs(rng, n: int, x, y) -> list[_Surface]:
    out = []
    for cx, cy in zip(rng.uniform(*x, n), rng.uniform(*y, n)):
        out += _box(cx - 0.22, cx + 0.22, cy - 0.22, cy + 0.22, 0, 0.45, (0.10, 0.10, 0.12))
    return out


def _column_boxes():
    return [(cx - 0.2, cx + 0.2, cy - 0.2, cy + 0.2) for cx in (3.5, 6.5) for cy in (3.5, 6.5)]


def _blocked(path_pt: np.ndarray, p: np.ndarray, door_open: bool) -> np.ndarray:
    """True where the straight line from the scanner position to a point is blocked: by the wall between the rooms
    (only the doorway lets a line through, and only while the door is open), or by a column."""
    d = p - path_pt
    blocked = np.zeros(len(p), bool)
    sides = (p[:, 1] - WALL_Y[0] / 2 - WALL_Y[1] / 2) * (path_pt[1] - WALL_Y[0] / 2 - WALL_Y[1] / 2) < 0
    if sides.any():
        through = np.ones(len(p), bool)
        for y in WALL_Y:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (y - path_pt[1]) / d[:, 1]
            x, z = path_pt[0] + t * d[:, 0], path_pt[2] + t * d[:, 2]
            crosses = (t > 0) & (t < 1)
            through &= ~crosses | ((x > DOOR_X[0]) & (x < DOOR_X[1]) & (z > 0) & (z < DOOR_TOP))
        blocked |= sides & ~(through & door_open)
    for x0, x1, y0, y1 in _column_boxes():                     # 2D slab test, full height columns
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.sort(np.stack([(x0 - path_pt[0]) / d[:, 0], (x1 - path_pt[0]) / d[:, 0]]), axis=0)
            ty = np.sort(np.stack([(y0 - path_pt[1]) / d[:, 1], (y1 - path_pt[1]) / d[:, 1]]), axis=0)
        t_in, t_out = np.maximum(tx[0], ty[0]), np.minimum(tx[1], ty[1])
        on_column = (p[:, 0] >= x0 - 0.01) & (p[:, 0] <= x1 + 0.01) & (p[:, 1] >= y0 - 0.01) & (p[:, 1] <= y1 + 0.01)
        blocked |= (t_in < t_out) & (t_in > 0) & (t_in < 0.999) & ~on_column
    return blocked


@dataclass
class _Walk:
    path: np.ndarray           # where the phone was carried, in order (n x 3)
    door_open: bool
    drift_yaw_deg: float       # how far the scan's own idea of its pose wandered by the end of the walk
    drift_tilt_deg: float
    drift_m: tuple
    exposure: float
    stage_light: bool
    stray_share: float         # points floating in mid-air, as a share of the scan
    range_m: float             # nothing farther than this from the phone is captured
    full_density_m: float = 0.8    # points thin out with distance beyond this
    texture: float = 0.03          # point-to-point colour variation (0..1): paint is near 0, carpet and wood more
    blocked: object = None         # optional: (scanner position, points) -> which are hidden; else this venue's walls


def _drift(points: np.ndarray, u: np.ndarray, walk: _Walk) -> np.ndarray:
    """Each point moved by the pose error the scan had when it captured it (u: 0 at the walk's start, 1 at its end)."""
    a, b = np.radians(walk.drift_yaw_deg) * u, np.radians(walk.drift_tilt_deg) * u
    rel = points - walk.path[0]
    x = rel[:, 0] * np.cos(a) - rel[:, 1] * np.sin(a)
    y = rel[:, 0] * np.sin(a) + rel[:, 1] * np.cos(a)
    z = rel[:, 2] + x * np.sin(b)
    x = x * np.cos(b)
    return np.column_stack([x, y, z]) + walk.path[0] + np.outer(u, walk.drift_m)


def _capture_weights(p: np.ndarray, walk: _Walk) -> tuple[np.ndarray, np.ndarray]:
    """(weights, distances), point by walk position: which positions see each point within range (weight 0 where
    not), nearer ones counting more. One of them captures the point."""
    dist = np.linalg.norm(p[:, None, :] - walk.path[None, :, :], axis=2)
    seen = dist < walk.range_m
    for j in range(len(walk.path)):
        seen[:, j] &= ~(walk.blocked(walk.path[j], p) if walk.blocked else _blocked(walk.path[j], p, walk.door_open))
    return np.where(seen, 1.0 / np.maximum(dist, 0.3) ** 2, 0.0), dist


def _scan(surfaces, walk: _Walk, frame, n_target, rng) -> tuple[np.ndarray, np.ndarray]:
    """Points and colours (0..1) of one scan, in its own raw frame."""
    areas = np.array([np.linalg.norm(np.cross(s.e1, s.e2)) for s in surfaces])
    per_m2 = 5.95 * n_target / areas.sum()     # n_target: candidates; about a quarter (hall) or less survive
    k = len(walk.path)
    pts, cols = [], []
    for s, area in zip(surfaces, areas):
        n = int(round(per_m2 * area))
        ab = rng.random((n, 2))
        keep = np.ones(n, bool)
        for a0, a1, b0, b1 in s.holes:
            keep &= ~((ab[:, 0] >= a0) & (ab[:, 0] <= a1) & (ab[:, 1] >= b0) & (ab[:, 1] <= b1))
        p = s.origin + ab[:, :1] * s.e1 + ab[:, 1:] * s.e2
        if s.cut_corner_m:
            keep &= p[:, 0] + p[:, 1] >= s.cut_corner_m
        p, ab = p[keep], ab[keep]
        w, dist = _capture_weights(p, walk)
        total = w.sum(axis=1)
        ok = total > 0
        p, ab, w, dist, total = p[ok], ab[ok], w[ok], dist[ok], total[ok]
        r_near = np.min(np.where(w > 0, dist, np.inf), axis=1)
        keep = rng.random(len(p)) < np.minimum(1.0, (walk.full_density_m / r_near) ** 1.3)
        p, ab, w, total = p[keep], ab[keep], w[keep], total[keep]
        base = s.pattern(p) if s.pattern is not None else np.asarray(s.colour)
        pick = (np.cumsum(w, axis=1) / total[:, None] < rng.random(len(p))[:, None]).sum(axis=1)
        eye = walk.path[pick]
        ray = p - eye
        r = np.linalg.norm(ray, axis=1)
        ray /= r[:, None]
        if s.edgy:     # depth jumps: some edge points smear back along the line of sight
            edge = np.minimum(np.minimum(ab[:, 0], 1 - ab[:, 0]) * np.linalg.norm(s.e1),
                              np.minimum(ab[:, 1], 1 - ab[:, 1]) * np.linalg.norm(s.e2)) < 0.02
            smear = edge & (rng.random(len(p)) < 0.12)
            p[smear] += ray[smear] * rng.uniform(0.02, 0.3, smear.sum())[:, None]
        p = p + ray * rng.normal(0.0, 1.0, len(p))[:, None] * (0.005 + 0.0025 * r)[:, None]
        p = p + rng.normal(0.0, 0.0025, p.shape)
        p = _drift(p, pick / max(k - 1, 1), walk)
        light = 0.75 + 0.25 * np.cos(p[:, 0] * 0.9) * np.cos(p[:, 1] * 0.7)
        c = base * walk.exposure * light[:, None]
        if walk.stage_light:
            pool = np.exp(-((p[:, 0] - 8.5) ** 2 + (p[:, 1] - 8.5) ** 2) / 3.0)
            c = c * (1 - pool[:, None]) + pool[:, None] * np.array([0.9, 0.2, 0.6])
        pts.append(p)
        cols.append(np.clip(c + rng.normal(0.0, walk.texture, c.shape), 0.0, 1.0))
    venue, colour = np.concatenate(pts), np.concatenate(cols)
    # Stray points in mid-air around the walk.
    n_stray = int(len(venue) * walk.stray_share)
    strays = walk.path[rng.integers(0, k, n_stray)] + rng.normal(0.0, 1.2, (n_stray, 3))
    venue = np.concatenate([venue, strays])
    colour = np.concatenate([colour, rng.random((n_stray, 3)) * 0.5])
    local = venue @ frame[:3, :3].T + frame[:3, 3]
    return local, colour


def _mirror_ghost(points: np.ndarray, colours: np.ndarray, walk: _Walk, rng, share: float = 0.02):
    """The bar room seen in the mirror: points reflected behind the far wall, where the line from the phone to the
    reflection passes through the glass."""
    x0, x1, z0, z1 = MIRROR
    take = rng.random(len(points)) < share
    p = points[take].copy()
    p[:, 1] = 2 * BAR_BACK_Y - p[:, 1]
    eye = walk.path[rng.integers(0, len(walk.path), len(p))]
    t = (BAR_BACK_Y - eye[:, 1]) / (p[:, 1] - eye[:, 1])
    x, z = eye[:, 0] + t * (p[:, 0] - eye[:, 0]), eye[:, 2] + t * (p[:, 2] - eye[:, 2])
    seen = (x > x0) & (x < x1) & (z > z0) & (z < z1) & (p[:, 1] < BAR_BACK_Y)
    return p[seen], colours[take][seen] * 0.8


def _person(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Someone walking across the hall during the hall scan: a body-sized blur along their route."""
    along = rng.random(n)
    centre = np.column_stack([2.0 + 5.0 * along, 7.5 - 1.5 * along, np.zeros(n)])
    ang = rng.uniform(0, 2 * np.pi, n)
    body = np.column_stack([0.2 * np.cos(ang), 0.15 * np.sin(ang), rng.uniform(0.0, 1.75, n)])
    return centre + body, np.tile([0.35, 0.25, 0.2], (n, 1))


# Each scan's raw frame from the venue's. BAR_TO_HALL (bar raw -> hall raw) has the yaw, tilt and offset sizes of
# the real confirmed two-room pair it stands in for.
def _transform(yaw_deg: float, pitch_deg: float, roll_deg: float, t) -> np.ndarray:
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    m = np.eye(4)
    m[:3, :3] = (np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
                 @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]))
    m[:3, 3] = t
    return m


HALL_FROM_VENUE = _transform(0.0, 0.2, -0.1, (-6.0, -4.5, -1.4))
BAR_TO_HALL = _transform(157.1, 0.6, -0.8, (5.26, -27.35, 0.88))
BAR_FROM_VENUE = np.linalg.inv(BAR_TO_HALL) @ HALL_FROM_VENUE

HALL_WALK = _Walk(path=np.array([[2.2, 2.8, 1.4], [3.0, 5.2, 1.4], [5.0, 8.0, 1.4], [8.2, 6.0, 1.4],
                                 [8.0, 2.8, 1.4], [5.2, 1.6, 1.4], [2.6, 3.4, 1.4]]),
                  door_open=False, drift_yaw_deg=0.5, drift_tilt_deg=0.5, drift_m=(0.02, -0.015, 0.012),
                  exposure=1.0, stage_light=True, stray_share=0.015, range_m=6.0)
BAR_WALK = _Walk(path=np.array([[6.2, -2.4, 1.4], [5.0, -2.2, 1.4], [3.7, -1.4, 1.4], [4.47, -0.6, 1.4],
                                [4.47, 0.3, 1.45], [3.3, 1.2, 1.45]]),          # one step into the hall
                 door_open=True, drift_yaw_deg=-0.4, drift_tilt_deg=0.4, drift_m=(-0.015, 0.012, -0.008),
                 exposure=1.15, stage_light=False, stray_share=0.003, range_m=4.0)


def build(seed: int = 20261028, hall_points: int = 770_000, bar_points: int = 250_000) -> dict:
    """Both scans' points and colours in their own raw frames, and the true bar -> hall transform."""
    rng = np.random.default_rng(seed)
    shared = _hall() + _bar() + _chairs(rng, 12, (0.8, HALL_M - 0.8), (0.8, 6.5)) + _chairs(rng, 3, (3.4, 6.8), (-2.6, -0.8))
    moved = _chairs(rng, 4, (0.8, 5.0), (0.8, 3.5))
    hall = _scan(shared + _doorway(open_=False) + moved[:10], HALL_WALK, HALL_FROM_VENUE, hall_points, rng)
    person, person_c = _person(rng, hall_points // 200)
    hall = (np.concatenate([hall[0], person @ HALL_FROM_VENUE[:3, :3].T + HALL_FROM_VENUE[:3, 3]]),
            np.concatenate([hall[1], person_c]))
    bar_venue_pts, bar_c = _scan(shared + _doorway(open_=True) + moved[10:], BAR_WALK, np.eye(4), bar_points, rng)
    ghost, ghost_c = _mirror_ghost(bar_venue_pts, bar_c, BAR_WALK, rng)
    bar_all = np.concatenate([bar_venue_pts, ghost])
    bar = (bar_all @ BAR_FROM_VENUE[:3, :3].T + BAR_FROM_VENUE[:3, 3], np.concatenate([bar_c, ghost_c]))
    return {"hall": hall, "bar": bar, "bar_to_hall": _shared_truth(shared, rng)}


def _expected_drift(p: np.ndarray, walk: _Walk) -> np.ndarray:
    w, _ = _capture_weights(p, walk)
    u = w @ (np.arange(len(walk.path)) / max(len(walk.path) - 1, 1)) / w.sum(axis=1)
    return _drift(p, u, walk)


def _shared_truth(surfaces, rng, per_m2: float = 400.0) -> np.ndarray:
    """The right answer once both scans have drifted: the rigid move that best puts the bar scan's view of the
    surfaces both scans saw onto the hall scan's view of them. Each scan bends a little over its walk, so no move
    fits everywhere; this is the one that fits where they overlap, which is what lining them up means."""
    pts = []
    for s in surfaces:
        area = np.linalg.norm(np.cross(s.e1, s.e2))
        ab = rng.random((int(per_m2 * area), 2))
        pts.append(s.origin + ab[:, :1] * s.e1 + ab[:, 1:] * s.e2)
    p = np.concatenate(pts)
    both = (_capture_weights(p, HALL_WALK)[0].sum(axis=1) > 0) & (_capture_weights(p, BAR_WALK)[0].sum(axis=1) > 0)
    p = p[both]
    hall = _expected_drift(p, HALL_WALK) @ HALL_FROM_VENUE[:3, :3].T + HALL_FROM_VENUE[:3, 3]
    bar = _expected_drift(p, BAR_WALK) @ BAR_FROM_VENUE[:3, :3].T + BAR_FROM_VENUE[:3, 3]
    # Least-squares rotation and shift taking bar onto hall (Kabsch).
    cb, ch = bar.mean(axis=0), hall.mean(axis=0)
    u, _, vt = np.linalg.svd((bar - cb).T @ (hall - ch))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, ch - r @ cb
    return m


def write(folder: str, seed: int = 20261028, **sizes) -> dict:
    """Writes "hall.pts" and "bar.pts" (the bar file opens with a GeoData comment line, as newer exports do) and
    "truth.json" into folder; returns the truth."""
    from open_point_stitch.pts_file import write_pts

    venue = build(seed, **sizes)
    os.makedirs(folder, exist_ok=True)
    write_pts(os.path.join(folder, "hall.pts"), *venue["hall"], precision=6)
    bar_path = os.path.join(folder, "bar.pts")
    write_pts(bar_path, *venue["bar"], precision=6)
    with open(bar_path) as fh:
        body = fh.read()
    with open(bar_path, "w", newline="\n") as fh:
        fh.write(';{"GeoData":{"Lat":0.0,"Lon":0.0}}\n' + body)
    truth = {"source": "bar", "target": "hall", "seed": seed,
             "source_points": len(venue["bar"][0]), "target_points": len(venue["hall"][0]),
             "transform": venue["bar_to_hall"].tolist()}
    with open(os.path.join(folder, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2)
    return truth
