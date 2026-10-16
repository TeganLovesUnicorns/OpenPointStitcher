"""Gravity-constrained coarse registration for level scans.

Levelled scans differ by yaw plus translation (4 DOF), small enough to search exhaustively: level each cloud,
rasterise walls top-down, sweep yaw with FFT phase correlation, and hand the best candidates to ICP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .limits import single_threaded

log = logging.getLogger(__name__)

# The largest floor tilt estimate_level will straighten unless a caller says otherwise. The bench's Settings change
# it for the whole process (set_default_max_tilt).
DEFAULT_MAX_TILT_DEG = 15.0


def set_default_max_tilt(deg: float) -> None:
    global DEFAULT_MAX_TILT_DEG
    if not 1.0 <= deg <= 45.0:
        raise ValueError("the levelling tilt limit must be between 1 and 45 degrees")
    DEFAULT_MAX_TILT_DEG = float(deg)


@dataclass
class LevelInfo:
    rotation: np.ndarray      # 3x3, rotates the cloud so its floor normal is +Z
    tilt_deg: float           # how far off level the cloud was
    floor_z: float            # floor height after levelling
    inliers: int              # support for the fitted plane
    reliable: bool            # False when no convincing horizontal plane was found


@dataclass
class YawCandidate:
    yaw_deg: float
    translation: np.ndarray   # 3 vector applied after the yaw rotation
    score: float              # phase correlation peak, higher is better


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b. Anti-parallel inputs give a proper 180 degree turn,
    never a reflection."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def estimate_level(pcd: o3d.geometry.PointCloud,
                   distance_threshold: float = 0.03,
                   max_tilt_deg: Optional[float] = None,
                   seed: int = 42) -> LevelInfo:
    """Find the rotation that makes this cloud's floor exactly level, fitting planes only to points with
    near-vertical normals. Falls back to identity, marked unreliable."""
    if max_tilt_deg is None:
        max_tilt_deg = DEFAULT_MAX_TILT_DEG
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 100:
        return LevelInfo(np.eye(3), 0.0, float(np.min(pts[:, 2])), 0, False)

    work_src = o3d.geometry.PointCloud(pcd)
    if not work_src.has_normals():
        work_src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
    nz = np.abs(np.asarray(work_src.normals)[:, 2])
    horiz_idx = np.flatnonzero(nz > np.cos(np.radians(max_tilt_deg)))
    if horiz_idx.size < 200:
        log.warning("almost no horizontal surfaces found; leaving cloud unlevelled")
        return LevelInfo(np.eye(3), 0.0, float(np.percentile(pts[:, 2], 1)), 0, False)

    # See the note in walls.extract_walls: the deterministic fitter is written
    # and tested but deliberately not wired in yet.
    work = work_src.select_by_index(horiz_idx)

    # Single-threaded for the RANSAC plane fit only (its guesses race across threads); seed inside the lock.
    best = None
    with single_threaded():
        try:
            o3d.utility.random.seed(int(seed))
        except Exception:
            pass
        for _ in range(5):
            if len(work.points) < 100:
                break
            try:
                model, inliers = work.segment_plane(
                    distance_threshold=distance_threshold, ransac_n=3,
                    num_iterations=2000
                )
            except RuntimeError:
                break
            if len(inliers) < 100:
                break
            n = np.array(model[:3], dtype=float)
            n /= np.linalg.norm(n)
            if n[2] < 0:
                n = -n
            tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
            if tilt <= max_tilt_deg and (best is None or len(inliers) > best[1]):
                best = (n, len(inliers))
            work = work.select_by_index(inliers, invert=True)

    if best is None:
        log.warning("no horizontal plane found; leaving cloud unlevelled")
        return LevelInfo(np.eye(3), 0.0, float(np.percentile(pts[:, 2], 1)), 0, False)

    n, count = best
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    R = _rotation_between(n, np.array([0.0, 0.0, 1.0]))
    rotated = pts @ R.T
    floor_z = float(np.percentile(rotated[:, 2], 1))
    return LevelInfo(R, tilt, floor_z, int(count), True)


def _wall_mask(pcd: o3d.geometry.PointCloud, max_nz: float = 0.35) -> np.ndarray:
    """Points whose normal is close to horizontal, meaning wall-like surfaces."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
    nz = np.abs(np.asarray(pcd.normals)[:, 2])
    return nz < max_nz


def _rasterise(xy: np.ndarray, size: int, res: float, center: np.ndarray) -> np.ndarray:
    """Top down occupancy image, centred on `center`, `size` cells square."""
    half = size * res / 2.0
    ij = np.floor((xy - center + half) / res).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < size) & (ij[:, 1] >= 0) & (ij[:, 1] < size)
    ij = ij[ok]
    img = np.zeros((size, size), dtype=np.float64)
    if ij.size:
        np.add.at(img, (ij[:, 1], ij[:, 0]), 1.0)
    # Occupancy rather than density: a wall scanned twice as densely should not
    # outvote a wall scanned once.
    np.clip(img, 0.0, 1.0, out=img)
    return img


def _hann2d(size: int) -> np.ndarray:
    w = np.hanning(size)
    return np.outer(w, w)


def _phase_correlate(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Best (dy, dx) shift taking `a` onto `b`, plus the correlation peak."""
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa.conj() * fb
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    corr = np.fft.irfft2(cross / mag, s=a.shape)
    idx = int(np.argmax(corr))
    dy, dx = np.unravel_index(idx, corr.shape)
    peak = float(corr[dy, dx])
    n0, n1 = a.shape
    if dy > n0 // 2:
        dy -= n0
    if dx > n1 // 2:
        dx -= n1
    return int(dy), int(dx), peak


def yaw_sweep(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    res: float = 0.10,
    yaw_step_deg: float = 1.0,
    top_k: int = 8,
    grid: int = 256,
) -> list[YawCandidate]:
    """Exhaustive yaw search with FFT translation search at each angle.

    Both clouds must already be levelled. Returns the best `top_k` candidates,
    each a full yaw plus 3D translation, ordered best first.
    """
    src_pts = np.asarray(src.points)
    tgt_pts = np.asarray(tgt.points)

    src_wall = src_pts[_wall_mask(src)]
    tgt_wall = tgt_pts[_wall_mask(tgt)]
    if src_wall.shape[0] < 200 or tgt_wall.shape[0] < 200:
        log.warning("too few wall points for a yaw sweep, using all points")
        src_wall, tgt_wall = src_pts, tgt_pts

    src_c = src_wall[:, :2].mean(axis=0)
    tgt_c = tgt_wall[:, :2].mean(axis=0)

    window = _hann2d(grid)
    tgt_img = _rasterise(tgt_wall[:, :2], grid, res, tgt_c) * window

    src_rel = src_wall[:, :2] - src_c
    zero = np.zeros(2)

    results: list[YawCandidate] = []
    for yaw in np.arange(0.0, 360.0, yaw_step_deg):
        th = np.radians(yaw)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, -s], [s, c]])
        rotated = src_rel @ rot.T
        src_img = _rasterise(rotated, grid, res, zero) * window
        dy, dx, peak = _phase_correlate(src_img, tgt_img)
        # Shift in metres taking the rotated, centred source onto the target grid.
        t_xy = tgt_c + np.array([dx * res, dy * res]) - rot @ src_c
        results.append(YawCandidate(float(yaw), np.array([t_xy[0], t_xy[1], 0.0]), peak))

    results.sort(key=lambda r: -r.score)

    # Keep candidates that are genuinely distinct in yaw, not 8 samples of one peak.
    kept: list[YawCandidate] = []
    for cand in results:
        if all(min(abs(cand.yaw_deg - k.yaw_deg),
                   360.0 - abs(cand.yaw_deg - k.yaw_deg)) > 8.0 for k in kept):
            kept.append(cand)
        if len(kept) >= top_k:
            break
    return kept


def _solve_dz(src_z: np.ndarray, tgt_z: np.ndarray,
              search: float = 4.0, step: float = 0.02) -> float:
    """Best vertical offset, by 1D correlation of height histograms.

    Floors and ceilings make sharp peaks in a height histogram, so this locks
    on well even when the horizontal overlap is small.
    """
    lo = min(src_z.min(), tgt_z.min()) - search
    hi = max(src_z.max(), tgt_z.max()) + search
    bins = np.arange(lo, hi + step, step)
    hs, _ = np.histogram(src_z, bins=bins)
    ht, _ = np.histogram(tgt_z, bins=bins)
    hs = hs.astype(float)
    ht = ht.astype(float)
    if hs.sum() == 0 or ht.sum() == 0:
        return 0.0
    hs /= hs.sum()
    ht /= ht.sum()
    corr = np.correlate(ht, hs, mode="full")
    shift = int(np.argmax(corr)) - (len(hs) - 1)
    return float(shift * step)


def yaw_transform(yaw_deg: float, t: np.ndarray) -> np.ndarray:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    m = np.eye(4)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m[:3, 3] = t
    return m


def tilt_of(transform: np.ndarray) -> float:
    """How far this transform tips the vertical axis, in degrees."""
    z = transform[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))


def coarse_candidates(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    res: float = 0.10,
    yaw_step_deg: float = 1.0,
    top_k: int = 8,
) -> tuple[list[np.ndarray], LevelInfo, LevelInfo]:
    """Full gravity-constrained coarse search. Returns candidate 4x4 transforms in
    the ORIGINAL (unlevelled) frames, ready for ICP, plus each cloud's levelling
    info."""
    src_level = estimate_level(src)
    tgt_level = estimate_level(tgt)

    src_l = o3d.geometry.PointCloud(src)
    tgt_l = o3d.geometry.PointCloud(tgt)
    src_l.rotate(src_level.rotation, center=(0, 0, 0))
    tgt_l.rotate(tgt_level.rotation, center=(0, 0, 0))

    cands = yaw_sweep(src_l, tgt_l, res=res, yaw_step_deg=yaw_step_deg, top_k=top_k)

    src_lp = np.asarray(src_l.points)
    tgt_lp = np.asarray(tgt_l.points)

    out: list[np.ndarray] = []
    for cand in cands:
        yt = yaw_transform(cand.yaw_deg, cand.translation)
        moved = (yt[:3, :3] @ src_lp.T).T + yt[:3, 3]
        dz = _solve_dz(moved[:, 2], tgt_lp[:, 2])
        yt[2, 3] += dz

        # Compose back into the original frames:
        #   tgt_orig = inv(Rt) . yaw . Rs . src_orig
        rs = np.eye(4); rs[:3, :3] = src_level.rotation
        rt_inv = np.eye(4); rt_inv[:3, :3] = tgt_level.rotation.T
        out.append(rt_inv @ yt @ rs)

    return out, src_level, tgt_level
