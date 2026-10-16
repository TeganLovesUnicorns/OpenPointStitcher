"""Suggest each scan's height, pitch and roll so the floors and flat ceilings of overlapping scans agree. Meant for
every point of each scan (full_resolution_inputs); only returns a suggestion."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .align_anchors import ScanPose
from .placement import tilt_matrix3

log = logging.getLogger(__name__)

CELL_M = 0.25             # plan cell the surfaces are measured in
MIN_CELL_POINTS = 12      # points a scan needs in a cell before its floor or ceiling counts there
MIN_SHARED_CELLS = 40     # shared cells a pair needs before it says anything
LAYER_SPREAD_M = 0.08     # a floor or ceiling cell must be one flat layer, not a spread of points
FLOOR_MAX_H_M = 1.5       # only points this close to the scan's own floor level can be its floor
CEILING_MIN_H_M = 2.5     # only points this far above it can be its ceiling
STEP_WINDOW_M = 0.75      # cells this far from the pair's typical difference are a different surface
MAX_STEP_M = 2.0          # a pair whose typical difference is bigger is not comparing like with like
MAX_SLOPE_DEG = 10.0      # nor is one whose surfaces tilt more than this against each other
MIN_SHARE = 0.35          # share of a pair's cells its plane must explain before the pair counts
FLOOR_WEIGHT = 1.0
CEILING_WEIGHT = 0.3
CELL_SIGMA_M = 0.05       # how far apart two honest floor readings of one cell can be
PRIOR_Z_M = 1.0           # how far a scan may need to move up or down before the prior objects
PRIOR_TILT_RAD = 0.05     # ... and how far to tip (about 3 degrees)
PAIR_CELL_CAP = 150.0     # a pair's total weight never exceeds this many cells
TUKEY_M = 0.10            # residual past which a cell counts for nothing
_STRIDE = 1 << 21
_OFFSET = 1 << 20


@dataclass
class Layer:
    """Clean floor (or ceiling) readings of one scan: the plan cell and the height found there."""
    keys: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass
class SurfaceMap:
    floor: Layer
    ceiling: Layer


@dataclass
class PairReport:
    """How two scans' floors (or ceilings) disagree, as a robust tilted plane."""
    a: str
    b: str
    kind: str                     # "floor" or "ceiling"
    cells: int
    step_m: float                 # most common height difference
    slope_deg: Optional[float]    # relative tilt between the two surfaces, None when no plane fits
    rms_m: Optional[float]        # scatter about that plane
    share: float                  # share of the shared cells the plane explains
    disagree_m: Optional[float] = None   # RMS height difference about ZERO over those cells: step and slope together


@dataclass
class TiltSuggestion:
    scan: str
    pitch_deg: float
    roll_deg: float
    x_m: float
    y_m: float
    z_m: float
    d_pitch_deg: float
    d_roll_deg: float
    d_z_m: float
    cells: int                    # floor and ceiling cells that constrain this scan
    partners: int


@dataclass
class TiltSolveResult:
    suggestions: dict[str, TiltSuggestion] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    before: list[PairReport] = field(default_factory=list)
    after: list[PairReport] = field(default_factory=list)
    iterations: int = 0


def _layer(world: np.ndarray, use: np.ndarray, bottom: bool, cell: float, min_points: int) -> Layer:
    """One height per plan cell from the chosen points: the low end for a floor, the high end for a
    ceiling, kept only where the points there form one flat layer."""
    pts = world[use]
    if len(pts) == 0:
        empty = np.zeros(0)
        return Layer(np.zeros(0, dtype=np.int64), empty, empty, empty)
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] + _OFFSET) * _STRIDE + (ij[:, 1] + _OFFSET)
    order = np.lexsort((pts[:, 2], key))
    key_s, z_s = key[order], pts[order, 2]
    starts = np.flatnonzero(np.r_[True, np.diff(key_s) != 0])
    counts = np.diff(np.r_[starts, len(key_s)])
    big = counts >= min_points
    starts, counts = starts[big], counts[big]

    def pick(q: float) -> np.ndarray:
        return z_s[starts + np.round((counts - 1) * q).astype(np.int64)]

    edge, inner = (pick(0.05), pick(0.30)) if bottom else (pick(0.95), pick(0.70))
    ok = np.abs(inner - edge) < LAYER_SPREAD_M
    keys = key_s[starts][ok]
    ix, iy = keys // _STRIDE - _OFFSET, keys % _STRIDE - _OFFSET
    return Layer(keys, (ix + 0.5) * cell, (iy + 0.5) * cell, edge[ok])


def surface_map(world: np.ndarray, height: np.ndarray, cell: float = CELL_M,
                min_points: int = MIN_CELL_POINTS) -> SurfaceMap:
    """A scan's floor and ceiling heights per plan cell. `height` is each point's height above the scan's
    OWN floor level, which decides what may count: a cell where the scan never saw the floor would
    otherwise hand its ceiling to the floor comparison."""
    return SurfaceMap(floor=_layer(world, height < FLOOR_MAX_H_M, True, cell, min_points),
                      ceiling=_layer(world, height > CEILING_MIN_H_M, False, cell, min_points))


def robust_plane(xy: np.ndarray, d: np.ndarray, min_span_m: float = 0.45):
    """Trimmed plane fit d = a + b x + c y. Returns (coef, kept mask, residuals) or None when the cells
    are too few or lie along a strip, where the slope across the strip is unconstrained."""
    keep = np.abs(d - np.median(d)) < 0.5
    if keep.sum() < 20:
        return None
    design = np.column_stack([np.ones(len(d)), xy])
    coef = np.zeros(3)
    res = d
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(design[keep], d[keep], rcond=None)
        res = d - design @ coef
        mad = np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) < max(0.10, 3.0 * mad)
        if keep.sum() < 20:
            return None
    if np.sqrt(np.linalg.eigvalsh(np.cov(xy[keep].T)).min()) < min_span_m:
        return None
    return coef, keep, res


def _tukey(r: np.ndarray, c: float) -> np.ndarray:
    u = np.clip(np.abs(r) / c, 0.0, 1.0)
    return (1.0 - u * u) ** 2


def _shared(la: Layer, lb: Layer):
    """(x, y, difference a - b) over the plan cells both scans have a clean reading of."""
    _, ia, ib = np.intersect1d(la.keys, lb.keys, return_indices=True)
    return la.x[ia], la.y[ia], la.z[ia] - lb.z[ib]


def _fit_pair(x: np.ndarray, y: np.ndarray, d: np.ndarray):
    """The tilted plane one pair's differences follow, over ALL its cells: (coef, inlier mask, residuals,
    share, slope in degrees), or None when the pair is not comparing the same surface."""
    med = float(np.median(d))
    near = np.abs(d - med) < STEP_WINDOW_M
    if abs(med) > MAX_STEP_M or near.sum() < MIN_SHARED_CELLS:
        return None
    xy = np.column_stack([x, y])
    fit = robust_plane(xy[near], d[near])
    if fit is None:
        return None
    coef, keep_near, _ = fit
    res = d - np.column_stack([np.ones(len(d)), xy]) @ coef
    keep = np.zeros(len(d), dtype=bool)
    keep[np.flatnonzero(near)[keep_near]] = True
    slope = float(np.degrees(np.arctan(np.hypot(coef[1], coef[2]))))
    if slope > MAX_SLOPE_DEG:
        return None
    return coef, keep, res, float(keep.mean()), slope


def _dominant_step(d: np.ndarray) -> float:
    bins = np.round(d / 0.05).astype(int)
    vals, counts = np.unique(bins, return_counts=True)
    best_w, best_v = -1, 0
    for v in vals:
        w = counts[(vals >= v - 2) & (vals <= v + 2)].sum()
        if w > best_w:
            best_w, best_v = w, v
    return float(np.median(d[np.abs(d - best_v * 0.05) <= 0.10]))


def pair_reports(maps: dict[str, SurfaceMap]) -> list[PairReport]:
    """Floor and ceiling disagreement for every pair of scans that share enough clean cells."""
    out = []
    names = list(maps)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for kind in ("floor", "ceiling"):
                x, y, d = _shared(getattr(maps[a], kind), getattr(maps[b], kind))
                if len(d) < MIN_SHARED_CELLS:
                    continue
                fit = _fit_pair(x, y, d)
                if fit is None:
                    out.append(PairReport(a, b, kind, len(d), _dominant_step(d), None, None, 0.0))
                    continue
                _, keep, res, share, slope = fit
                out.append(PairReport(a, b, kind, len(d), _dominant_step(d), slope,
                                      float(np.sqrt(np.mean(res[keep] ** 2))), share,
                                      float(np.sqrt(np.mean(d[keep] ** 2)))))
    return out


@dataclass
class _Rows:
    a: int
    b: int
    x: np.ndarray
    y: np.ndarray
    d: np.ndarray
    w: np.ndarray


def _gather(maps: dict[str, SurfaceMap], names: list[str], surfaces: tuple) -> list[_Rows]:
    """The cell readings the solve fits, weighted by how well each pair's own tilted plane explains them."""
    rows = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            for kind, kw in (("floor", FLOOR_WEIGHT), ("ceiling", CEILING_WEIGHT)):
                if kind not in surfaces:
                    continue
                x, y, d = _shared(getattr(maps[a], kind), getattr(maps[names[j]], kind))
                if len(d) < MIN_SHARED_CELLS:
                    continue
                fit = _fit_pair(x, y, d)
                if fit is None or fit[3] < MIN_SHARE:
                    continue
                _, _, res, share, _ = fit
                w = _tukey(res, 2.0 * TUKEY_M) * share * kw
                cap = min(1.0, PAIR_CELL_CAP / max(float(w.sum()), 1e-9))
                rows.append(_Rows(i, j, x, y, d, w * cap))
    return rows


def _lever(r: _Rows, s: int, centres: np.ndarray) -> np.ndarray:
    """How a small change in height, pitch and roll of scan `s` (about its own middle) moves each cell."""
    return np.column_stack([np.ones(len(r.x)), -(r.x - centres[s, 0]), (r.y - centres[s, 1])])


def _solve(rows: list[_Rows], centres: np.ndarray, fixed: np.ndarray, n: int) -> np.ndarray:
    """Least squares for [dz, dpitch, droll] per scan (metres, radians), reweighted so cells that
    disagree with the rest count for less. A weak pull toward zero keeps unconstrained motion out."""
    prior = np.tile([1.0 / PRIOR_Z_M ** 2, 1.0 / PRIOR_TILT_RAD ** 2, 1.0 / PRIOR_TILT_RAD ** 2], n)
    prior[np.repeat(fixed, 3)] = 1e8
    levers = [(_lever(r, r.a, centres), _lever(r, r.b, centres)) for r in rows]
    theta = np.zeros(3 * n)
    weights = [r.w.copy() for r in rows]
    for _ in range(6):
        h = np.diag(prior)
        g = np.zeros(3 * n)
        for r, w, (ga, gb) in zip(rows, weights, levers):
            sa, sb = slice(3 * r.a, 3 * r.a + 3), slice(3 * r.b, 3 * r.b + 3)
            wa = w[:, None] / CELL_SIGMA_M ** 2
            h[sa, sa] += (ga * wa).T @ ga
            h[sb, sb] += (gb * wa).T @ gb
            cross = (ga * wa).T @ gb
            h[sa, sb] -= cross
            h[sb, sa] -= cross.T
            g[sa] -= (ga * wa).T @ r.d
            g[sb] += (gb * wa).T @ r.d
        theta = np.linalg.solve(h, g)
        weights = [r.w * _tukey(r.d + ga @ theta[3 * r.a:3 * r.a + 3] - gb @ theta[3 * r.b:3 * r.b + 3], TUKEY_M)
                   for r, (ga, gb) in zip(rows, levers)]
    return theta


def suggest_tilt(payloads: dict[str, dict], poses: dict[str, ScanPose], locked: Optional[dict] = None,
                 subset: Optional[set] = None, max_change_deg: float = 6.0, max_lift_m: float = 2.0,
                 iterations: int = 8, surfaces: tuple = ("floor", "ceiling"), settle_after: int = 4) -> TiltSolveResult:
    """Suggested pitch, roll and position per scan. Locked scans and scans outside subset stay put as references. A
    scan whose fix needs more than max_change_deg of tip or max_lift_m of lift is left alone and listed in skipped.
    After settle_after rounds each step is half the last, so the answer settles instead of jittering."""
    locked = locked or {}
    names = [n for n in poses if n in payloads]
    result = TiltSolveResult()
    if len(names) < 2:
        return result

    local, mid = {}, {}
    for n in names:
        p = payloads[n]
        pts = np.column_stack([np.asarray(p["xy"], dtype=float).reshape(-1, 2), np.asarray(p["h"], dtype=float)])
        local[n], mid[n] = pts, pts.mean(axis=0)
    work = {n: ScanPose(**vars(poses[n])) for n in names}
    fixed = np.array([bool(locked.get(n, False)) or (subset is not None and n not in subset) for n in names])

    def world_of(n: str, p: ScanPose) -> np.ndarray:
        return local[n] @ tilt_matrix3(p.pitch_deg, p.roll_deg, p.yaw_deg).T + np.array([p.x_m, p.y_m, p.z_m])

    def maps_of(poses_now: dict) -> dict:
        return {n: surface_map(world_of(n, poses_now[n]), local[n][:, 2]) for n in names}

    maps = maps_of(work)
    result.before = pair_reports(maps)
    support = np.zeros(len(names))
    partners = [set() for _ in names]

    for it in range(iterations):
        rows = _gather(maps, names, surfaces)
        if not rows:
            break
        centres = np.array([tilt_matrix3(work[n].pitch_deg, work[n].roll_deg, work[n].yaw_deg) @ mid[n]
                            + [work[n].x_m, work[n].y_m, work[n].z_m] for n in names])
        theta = _solve(rows, centres, fixed, len(names)) * (0.5 ** max(0, it - settle_after + 1))
        if it == 0:
            for r in rows:
                for s, o in ((r.a, r.b), (r.b, r.a)):
                    support[s] += float((r.w > 0.05).sum())
                    partners[s].add(o)
        biggest = 0.0
        for i, n in enumerate(names):
            dz, dp, dr = theta[3 * i:3 * i + 3]
            p = work[n]
            new_pitch, new_roll = p.pitch_deg + np.degrees(dp), p.roll_deg + np.degrees(dr)
            keep_centre = centres[i] + [0.0, 0.0, dz]
            shifted = keep_centre - tilt_matrix3(new_pitch, new_roll, p.yaw_deg) @ mid[n]
            work[n] = ScanPose(yaw_deg=p.yaw_deg, x_m=float(shifted[0]), y_m=float(shifted[1]), z_m=float(shifted[2]),
                               pitch_deg=float(new_pitch), roll_deg=float(new_roll))
            biggest = max(biggest, abs(np.degrees(dp)), abs(np.degrees(dr)), abs(dz) * 10.0)
        maps = maps_of(work)
        result.iterations = it + 1
        if biggest < 0.02:
            break

    accepted = dict(work)
    for i, n in enumerate(names):
        p0, p1 = poses[n], work[n]
        dp, dr, dz = p1.pitch_deg - p0.pitch_deg, p1.roll_deg - p0.roll_deg, p1.z_m - p0.z_m
        if fixed[i]:
            accepted[n] = p0
        elif support[i] == 0:
            result.skipped[n] = "no other scan shares a clean floor or ceiling with it"
            accepted[n] = p0
        elif max(abs(dp), abs(dr)) > max_change_deg or abs(dz) > max_lift_m:
            result.skipped[n] = (f"the fix would tip it {max(abs(dp), abs(dr)):.1f} degrees or move it "
                                 f"{abs(dz):.2f} m up or down, which is more than a tilt correction; check its placement")
            accepted[n] = p0
        else:
            result.suggestions[n] = TiltSuggestion(
                scan=n, pitch_deg=round(p1.pitch_deg, 4), roll_deg=round(p1.roll_deg, 4), x_m=round(p1.x_m, 5),
                y_m=round(p1.y_m, 5), z_m=round(p1.z_m, 5), d_pitch_deg=round(dp, 4), d_roll_deg=round(dr, 4),
                d_z_m=round(dz, 4), cells=int(support[i]), partners=len(partners[i]))
    result.after = pair_reports(maps_of(accepted))
    return result


def full_resolution_inputs(scans_dir: str, payloads: dict[str, dict], names, workers: int = 4) -> dict[str, dict]:
    """Every point of each named scan, in the levelled frame the preview uses, for suggest_tilt. The solve must not
    run on the thinned preview, which shifts the answer by tenths of a degree."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from .pts_file import find_scans, read_pts
    files = {os.path.splitext(os.path.basename(p))[0]: p for p in find_scans(scans_dir, "*.pts")}

    def load(name: str):
        levelled = read_pts(files[name]).points @ np.asarray(payloads[name]["level_rotation"], dtype=float).T
        return name, {"xy": levelled[:, :2].astype(np.float32),
                      "h": (levelled[:, 2] - float(payloads[name]["floor_z"])).astype(np.float32)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(load, [n for n in names if n in files]))


def summarise(before: list[PairReport], after: list[PairReport]) -> dict:
    """For floors and for ceilings, on the pairs that had a clear plane before: the typical disagreement
    before and after (cm) and how many pairs got better or worse."""
    out = {}
    for kind in ("floor", "ceiling"):
        old = {(r.a, r.b): r for r in before if r.kind == kind and r.disagree_m is not None and r.share >= 0.5}
        new = {(r.a, r.b): r for r in after if r.kind == kind and r.disagree_m is not None and (r.a, r.b) in old}
        keys = [k for k in old if k in new]
        if not keys:
            out[kind] = {"pairs": 0}
            continue
        b = np.array([old[k].disagree_m for k in keys]) * 100.0
        a = np.array([new[k].disagree_m for k in keys]) * 100.0
        out[kind] = {"pairs": len(keys), "median_before_cm": round(float(np.median(b)), 1),
                     "median_after_cm": round(float(np.median(a)), 1),
                     "better": int((a < b - 0.5).sum()), "worse": int((a > b + 0.5).sum())}
    return out


def near_share(payloads: dict[str, dict], poses: dict[str, ScanPose], within_m: float = 0.05) -> dict[str, float]:
    """Per scan, the share of its preview points within `within_m` of any OTHER scan, as the bench draws
    them. The solve does not optimise this, so it is an independent check on the suggestion."""
    from .align_anchors import world_cloud_points_as_drawn
    from .overlap import nearest_distances
    names = [n for n in poses if n in payloads]
    world = {n: world_cloud_points_as_drawn(payloads[n], poses[n]) for n in names}
    out = {}
    for n in names:
        others = [world[m] for m in names if m != n]
        target = np.concatenate(others) if others else np.zeros((0, 3))
        out[n] = float((nearest_distances(world[n], target) <= within_m).mean())
    return out


def apply_suggestions(poses: dict[str, ScanPose], result: TiltSolveResult) -> dict[str, ScanPose]:
    """The poses with every suggestion in place; scans without one keep theirs."""
    return {n: (ScanPose(yaw_deg=p.yaw_deg, x_m=result.suggestions[n].x_m, y_m=result.suggestions[n].y_m,
                         z_m=result.suggestions[n].z_m, pitch_deg=result.suggestions[n].pitch_deg,
                         roll_deg=result.suggestions[n].roll_deg) if n in result.suggestions else p)
            for n, p in poses.items()}
