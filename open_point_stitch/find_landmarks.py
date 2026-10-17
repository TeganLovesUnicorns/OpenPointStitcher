"""Landmarks: things in a room that are not walls, floor or ceiling (seating, equipment, fixtures), found in each
scan and matched by how far apart they are: two things 3.2 m apart in one scan must be 3.2 m apart in the other.
Two matched landmarks plus level fix the whole alignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

from .lib.clouds import as_cloud
from .lib.poses import fit_yaw_shift_2d, rot2, tilt_of, yaw_gap_deg, yaw_matrix2

log = logging.getLogger(__name__)


@dataclass
class Landmark:
    xy: np.ndarray            # centre in the levelled plan (metres)
    h: float                  # height of its middle above the floor
    length: float             # footprint, longer side
    width: float              # footprint, shorter side
    top: float                # height of its highest part above the floor
    base: float               # height of its lowest part above the floor
    n_points: int
    brightness: float         # relative to this scan's own median, so the lighting on the day does not matter
    weight: float
    # Position in this scan's raw (unlevelled) frame, filled in by build_scan_landmarks; None from find_landmarks.
    # score.landmark_agreement needs it to apply a candidate's transform.
    raw_xyz: Optional[np.ndarray] = None


@dataclass
class LandmarkMatch:
    yaw_deg: float
    translation: np.ndarray                       # (tx, ty) in the levelled plan
    pairs: list = field(default_factory=list)     # (source index, target index) of landmarks that line up
    score: float = 0.0                            # summed weight of those pairs


def _organic_fraction(p3: np.ndarray, k: int = 12, sample_cap: int = 250,
                      scatter_threshold: float = 0.12) -> float:
    """Share of a cluster's points whose neighbourhood scatters in every direction (skin, hair, fabric) rather than
    lying flat or along a line (anything rigid). Seeded, so a cluster always scores the same."""
    n = p3.shape[0]
    if n < k + 1:
        return 0.0   # too few points to say anything - never reject on no evidence
    idx = (np.random.default_rng(0).choice(n, size=sample_cap, replace=False)
           if n > sample_cap else np.arange(n))
    # KDTreeFlann needs a PointCloud; a bare Vector3dVector builds without error but every query fails.
    tree = o3d.geometry.KDTreeFlann(as_cloud(p3))
    scattered = checked = 0
    for i in idx:
        _, nbr, _ = tree.search_knn_vector_3d(p3[i], k + 1)   # includes the point itself
        nbr = np.asarray(nbr)
        if nbr.size < k + 1:
            continue
        local = p3[nbr] - p3[nbr].mean(axis=0)
        cov = local.T @ local
        e = np.sort(np.linalg.eigvalsh(cov))[::-1]            # e[0] >= e[1] >= e[2] >= 0
        if e[0] <= 1e-12:
            continue
        checked += 1
        if (e[2] / e[0]) > scatter_threshold:
            scattered += 1
    return (scattered / checked) if checked else 0.0


def find_landmarks(
    xy: np.ndarray, h: np.ndarray, gray: Optional[np.ndarray] = None,
    cell: float = 0.10, min_points: int = 10, min_height: float = 0.10, max_height: float = 2.6,
    min_size: float = 0.15, max_size: float = 6.0, join: int = 0,
    reject_organic: bool = False, organic_min_points: int = 40,
    organic_min_fraction: float = 0.5, organic_scatter_threshold: float = 0.12,
) -> list[Landmark]:
    """Cluster the points no plane explains (xy levelled plan, h height above the floor, gray 0-1 or None) into
    things: touching occupied cells join, gaps of up to join cells are bridged, tiny and huge clusters are dropped.
    Biggest first.

    reject_organic drops a cluster that reads as organic (_organic_fraction) on at least organic_min_fraction of its
    points: a person, a pet, a draped jacket. Clusters under organic_min_points are never rejected this way."""
    keep = (h >= min_height) & (h <= max_height) & (np.abs(xy - np.median(xy, axis=0)).max(axis=1) < 60.0)
    if keep.sum() < min_points:
        return []
    xy, h = xy[keep], h[keep]
    g = gray[keep] if gray is not None else None
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(int) + 1               # (column, row), one cell of border
    counts = np.zeros((ij[:, 1].max() + 2, ij[:, 0].max() + 2), dtype=np.int32)
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    joined = (counts >= 1).astype(np.uint8)
    if join > 0:
        joined = cv2.dilate(joined, np.ones((2 * join + 1, 2 * join + 1), np.uint8))
    _, labels = cv2.connectedComponents(joined, connectivity=8)
    point_label = labels[ij[:, 1], ij[:, 0]]

    if g is not None:
        g_median = float(np.median(g))
        g_spread = float(1.4826 * np.median(np.abs(g - g_median))) + 1e-3
    order = np.argsort(point_label, kind="stable")
    labs, starts = np.unique(point_label[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    found: list[Landmark] = []
    for lab, s, e in zip(labs, starts, ends):
        if lab == 0 or e - s < min_points:
            continue
        idx = order[s:e]
        p = xy[idx]
        centre = p.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((p - centre).T))
        along, across = (p - centre) @ vecs[:, 1], (p - centre) @ vecs[:, 0]
        length = float(np.percentile(along, 98) - np.percentile(along, 2))
        width = float(np.percentile(across, 98) - np.percentile(across, 2))
        if not (min_size <= length <= max_size):
            continue
        if reject_organic and idx.size >= organic_min_points:
            p3 = np.column_stack([p, h[idx]])
            if _organic_fraction(p3, scatter_threshold=organic_scatter_threshold) >= organic_min_fraction:
                continue
        top, base = float(np.percentile(h[idx], 98)), float(np.percentile(h[idx], 2))
        bright = 0.0 if g is None else float(np.clip((g[idx].mean() - g_median) / g_spread, -3.0, 3.0))
        found.append(Landmark(xy=centre, h=(top + base) / 2.0, length=length, width=width, top=top, base=base,
                              n_points=int(idx.size), brightness=bright, weight=float(np.log1p(idx.size))))
    found.sort(key=lambda lm: -lm.weight)
    return found


def _yaw_of(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[..., 1], v[..., 0])


def _compatible(src: list[Landmark], tgt: list[Landmark], base_tol: float, top_tol: float, size_ratio: float,
                brightness_tol: Optional[float] = None) -> np.ndarray:
    """Which source landmark could be which target landmark: about the same height off the floor, the same size and,
    if `brightness_tol` is given, about the same brightness against its own scan. Loose on purpose: two scans see
    different sides of the same thing."""
    def col(lms, name):
        return np.array([getattr(lm, name) for lm in lms])
    ok = np.abs(col(src, "base")[:, None] - col(tgt, "base")[None, :]) <= base_tol
    ok &= np.abs(col(src, "top")[:, None] - col(tgt, "top")[None, :]) <= top_tol
    for name in ("length", "width"):
        a, b = np.maximum(col(src, name), 0.05)[:, None], np.maximum(col(tgt, name), 0.05)[None, :]
        ok &= np.maximum(a / b, b / a) <= size_ratio
    if brightness_tol is not None:
        ok &= np.abs(col(src, "brightness")[:, None] - col(tgt, "brightness")[None, :]) <= brightness_tol
    return ok


def _lineup(p: np.ndarray, q: np.ndarray, compat: np.ndarray, yaw: float, shift: np.ndarray, tol: float):
    """Source landmark i -> target landmark j pairs that land within `tol` of each other under (yaw, shift), one target each."""
    moved = p @ rot2(yaw).T + shift
    dist = np.linalg.norm(moved[:, None, :] - q[None, :, :], axis=2)
    dist = np.where(compat, dist, np.inf)
    best = dist.argmin(axis=1)
    pairs = [(i, int(j)) for i, j in enumerate(best) if dist[i, j] <= tol]
    seen: dict[int, tuple[int, float]] = {}
    for i, j in pairs:
        if j not in seen or dist[i, j] < seen[j][1]:
            seen[j] = (i, dist[i, j])
    return [(i, j) for j, (i, _d) in seen.items()]


def match_landmarks(
    src: list[Landmark], tgt: list[Landmark], max_landmarks: int = 40, pair_tol_m: float = 0.4, inlier_tol_m: float = 0.6,
    min_separation_m: float = 1.0, max_separation_m: float = 25.0, base_tol_m: float = 0.4, top_tol_m: float = 0.6,
    size_ratio: float = 2.5, min_pairs: int = 3, top_k: int = 10, brightness_tol: Optional[float] = None,
) -> list[LandmarkMatch]:
    """Poses (plan rotation and shift) that line up at least min_pairs landmarks. Every pair of source landmarks is
    tried against every pair of target landmarks at the same separation; each agreeing pair votes for the pose it
    implies, and the best-voted poses are re-scored on all landmarks and refitted."""
    src, tgt = src[:max_landmarks], tgt[:max_landmarks]
    if len(src) < min_pairs or len(tgt) < min_pairs:
        return []
    p = np.array([lm.xy for lm in src])
    q = np.array([lm.xy for lm in tgt])
    compat = _compatible(src, tgt, base_tol_m, top_tol_m, size_ratio, brightness_tol)
    d_src = np.linalg.norm(p[:, None] - p[None], axis=2)
    d_tgt = np.linalg.norm(q[:, None] - q[None], axis=2)

    yaws, shifts = [], []
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            if not (min_separation_m <= d_src[i, j] <= max_separation_m):
                continue
            ok = (np.abs(d_tgt - d_src[i, j]) <= pair_tol_m) & compat[i][:, None] & compat[j][None, :]
            k, l = np.nonzero(ok)
            if k.size == 0:
                continue
            yaw = _yaw_of(q[l] - q[k]) - _yaw_of(p[j] - p[i])
            mid_s, mid_t = (p[i] + p[j]) / 2.0, (q[k] + q[l]) / 2.0
            c, s = np.cos(yaw), np.sin(yaw)
            shift = mid_t - np.column_stack([c * mid_s[0] - s * mid_s[1], s * mid_s[0] + c * mid_s[1]])
            yaws.append(yaw)
            shifts.append(shift)
    if not yaws:
        return []
    yaw_all = np.concatenate(yaws)
    shift_all = np.vstack(shifts)

    # Each agreeing pair votes; keep the most-voted poses, in bins of 3 degrees and 30 cm.
    yb = np.round(np.degrees(np.angle(np.exp(1j * yaw_all))) / 3.0).astype(np.int64)
    xb = np.round(shift_all[:, 0] / 0.3).astype(np.int64)
    zb = np.round(shift_all[:, 1] / 0.3).astype(np.int64)
    key = (yb + 1000) * 4_000_000_000 + (xb + 20000) * 40_000 + (zb + 20000)
    uniq, inverse, votes = np.unique(key, return_inverse=True, return_counts=True)
    strongest = np.argsort(-votes)[:300]

    scored: list[LandmarkMatch] = []
    for b in strongest:
        sel = inverse == b
        yaw = float(np.angle(np.exp(1j * yaw_all[sel]).mean()))
        shift = shift_all[sel].mean(axis=0)
        pairs = _lineup(p, q, compat, yaw, shift, inlier_tol_m)
        for _ in range(2):                                    # refit on what lines up, then look again
            if len(pairs) < 2:
                break
            yaw, shift = fit_yaw_shift_2d(p[[i for i, _ in pairs]], q[[j for _, j in pairs]])
            pairs = _lineup(p, q, compat, yaw, shift, inlier_tol_m)
        if len(pairs) >= min_pairs:
            scored.append(LandmarkMatch(yaw_deg=float(np.degrees(yaw)), translation=shift, pairs=pairs,
                                        score=float(sum(src[i].weight * tgt[j].weight for i, j in pairs))))
    scored.sort(key=lambda m: -m.score)
    kept: list[LandmarkMatch] = []
    for m in scored:
        if all(yaw_gap_deg(m.yaw_deg, k.yaw_deg) > 3.0 or np.linalg.norm(m.translation - k.translation) > 0.4
               for k in kept):
            kept.append(m)
        if len(kept) >= top_k:
            break
    return kept


def build_scan_landmarks(features, down, level_rotation: np.ndarray, floor_z: float, **kwargs) -> None:
    """Fill features.landmarks in place, in the bench's own level (payload level_rotation and floor_z), like
    build_scan_corners; organic shapes rejected; each landmark also gets raw_xyz. Recomputes its own residual for
    this down cloud and never raises: a missing landmark list must never stop an alignment."""
    try:
        from .find_surfaces import extract_scan_features
        idx = np.asarray(extract_scan_features(down).residual_idx, dtype=np.int64)
        if idx.size == 0:
            features.landmarks = []
            return
        raw = np.asarray(down.points)[idx]
        r = np.asarray(level_rotation)
        levelled = raw @ r.T
        gray = None
        if down.has_colors():
            rgb = np.asarray(down.colors)[idx]
            gray = rgb @ np.array([0.299, 0.587, 0.114])
        kwargs.setdefault("reject_organic", True)
        lms = find_landmarks(levelled[:, :2], levelled[:, 2] - float(floor_z), gray, **kwargs)
        for lm in lms:
            # Inverse of `raw @ r.T`: r is a rotation (orthogonal), so its inverse is its transpose,
            # making `levelled @ r` the way back - the same relationship `landmark_candidates` already
            # uses the other direction (`rs[:3,:3] = src_level.rotation`, composed onto raw poses).
            lm.raw_xyz = np.array([lm.xy[0], lm.xy[1], lm.h + float(floor_z)]) @ r
        features.landmarks = lms
    except Exception:  # noqa: BLE001 - a missing voice must never stop an align
        log.exception("could not find landmarks for one scan; it will not vote")
        features.landmarks = []


def scan_landmarks(down, level, features=None) -> list[Landmark]:
    """One scan's landmarks: the points no plane explains, levelled with `level` (a LevelInfo), heights above its floor.
    `down` is the voxel-downsampled cloud; `features` are its ScanFeatures if already computed."""
    from .find_surfaces import extract_scan_features
    if features is None:
        features = extract_scan_features(down)
    idx = np.asarray(features.residual_idx, dtype=np.int64)
    if idx.size == 0:
        return []
    levelled = np.asarray(down.points)[idx] @ np.asarray(level.rotation).T
    gray = None
    if down.has_colors():
        rgb = np.asarray(down.colors)[idx]
        gray = rgb @ np.array([0.299, 0.587, 0.114])
    return find_landmarks(levelled[:, :2], levelled[:, 2] - float(level.floor_z), gray)


def landmark_candidates(src_down, tgt_down, src_features=None, tgt_features=None, top_k: int = 8, **match_kwargs):
    """Starting poses from matched landmarks, as full transforms between the scans' ORIGINAL frames (the same shape
    `wall_candidates` returns), best first, plus each match's lined-up count and both landmark lists for reporting."""
    from .align_gravity import estimate_level
    src_level, tgt_level = estimate_level(src_down), estimate_level(tgt_down)
    src_lms = scan_landmarks(src_down, src_level, src_features)
    tgt_lms = scan_landmarks(tgt_down, tgt_level, tgt_features)
    matches = match_landmarks(src_lms, tgt_lms, top_k=top_k, **match_kwargs)
    rs = np.eye(4)
    rs[:3, :3] = src_level.rotation
    rt_inv = np.eye(4)
    rt_inv[:3, :3] = np.asarray(tgt_level.rotation).T
    out = []
    for m in matches:
        levelled = np.eye(4)
        levelled[:2, :2] = yaw_matrix2(m.yaw_deg)
        levelled[:2, 3] = m.translation
        levelled[2, 3] = tgt_level.floor_z - src_level.floor_z          # heights are floor-relative, so the floors coincide
        full = rt_inv @ levelled @ rs
        if tilt_of(full) <= 5.0:
            out.append(full)
    log.info("landmarks: %d source, %d target, %d matches, %d candidates", len(src_lms), len(tgt_lms), len(matches), len(out))
    return out, matches, src_lms, tgt_lms
