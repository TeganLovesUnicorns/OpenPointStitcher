"""Planar surfaces at any orientation, described and matched between scans.

Every plane is kept, not just near-vertical ones, and matched by size, shape, colour and neighbours. A single
plane is ambiguous; its neighbours make it close to unique.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d

from .lib import geometry
from .footprint import footprint
from .find_walls import find_true_wall
from .find_walls import fit_plane

log = logging.getLogger(__name__)


@dataclass
class Surface:
    """One planar surface, in a scan's own levelled local frame."""

    normal: np.ndarray       # unit 3D normal
    offset: float            # plane: normal . xyz = offset
    centroid: np.ndarray     # 3D centroid of the inliers
    support: int             # inlier point count
    area: float              # footprint area, m^2
    extent_u: float          # in-plane bounding size, long axis
    extent_v: float          # in-plane bounding size, short axis
    corners: int             # vertex count of the simplified footprint outline
    hue: float               # circular mean hue, degrees 0-360 (nan if too grey)
    saturation: float        # mean saturation, 0-1
    value: float             # mean brightness, 0-1
    u_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    v_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    index: int = -1
    neighbors: list = field(default_factory=list)  # filled by build_neighbor_graph

    @property
    def aspect(self) -> float:
        return self.extent_u / self.extent_v if self.extent_v > 1e-6 else 0.0

    @property
    def tilt_deg(self) -> float:
        """0 for a floor or ceiling, 90 for a plumb wall."""
        return geometry.tilt_off_horizontal(self.normal)

    def footprint_slack(self, p: np.ndarray) -> float:
        """How far p sits outside this surface's footprint box in its (u, v) plane; 0.0 inside. Uses the bounding
        extent, so it can pass a point the box reaches but no measured point does."""
        rel = np.asarray(p) - self.centroid
        du = max(0.0, abs(float(rel @ self.u_axis)) - self.extent_u / 2)
        dv = max(0.0, abs(float(rel @ self.v_axis)) - self.extent_v / 2)
        return float(np.hypot(du, dv))

    def describe(self) -> str:
        ang = self.tilt_deg
        hue_s = f"{self.hue:.0f}deg" if not np.isnan(self.hue) else "grey"
        return (f"S{self.index}: tilt {ang:5.1f} deg off horizontal, "
                f"{self.extent_u:4.1f} x {self.extent_v:4.1f} m ({self.corners} corners), "
                f"hue {hue_s} sat {self.saturation:.2f}, {self.support:6,d} pts")


@dataclass
class Neighbor:
    other: int          # index of the friend surface
    angle_deg: float     # angle between the two normals, 0-180
    distance_m: float    # centroid-to-centroid distance
    size_ratio: float    # this surface's area / the friend's area
    adjacent: bool       # footprints come within touching distance in 3D


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[float, float, float]:
    """Circular mean hue (degrees), mean saturation and mean value over 0-1 RGB rows. Hue is nan below a
    saturation floor, where it is only noise."""
    if rgb.size == 0:
        return float("nan"), 0.0, 0.0
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    v = mx
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    mean_sat = float(sat.mean())
    mean_val = float(v.mean())
    if mean_sat < 0.12:  # below this saturation, hue is noise
        return float("nan"), mean_sat, mean_val
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    d = mx - mn
    d_safe = np.maximum(d, 1e-6)
    hue = np.zeros_like(mx)
    is_r = (mx == r) & (d > 1e-6)
    is_g = (mx == g) & (d > 1e-6)
    is_b = (mx == b) & (d > 1e-6)
    hue[is_r] = ((g[is_r] - b[is_r]) / d_safe[is_r]) % 6
    hue[is_g] = (b[is_g] - r[is_g]) / d_safe[is_g] + 2
    hue[is_b] = (r[is_b] - g[is_b]) / d_safe[is_b] + 4
    hue_deg = hue * 60.0
    # circular mean, weighted by saturation so a near-grey point (hue noise
    # even above the floor) barely votes
    w = sat
    if w.sum() < 1e-6:
        return float("nan"), mean_sat, mean_val
    ang = np.radians(hue_deg)
    mean_ang = np.degrees(np.arctan2((np.sin(ang) * w).sum(), (np.cos(ang) * w).sum())) % 360
    return float(mean_ang), mean_sat, mean_val


def segment_surfaces(
    pcd: o3d.geometry.PointCloud,
    level_rotation: np.ndarray | None = None,
    max_surfaces: int = 40,
    distance_threshold: float = 0.03,
    min_support: int = 300,
    min_extent: float = 0.4,
    seed: int = 42,
    find_behind_curtains: bool = True,
) -> tuple[list[Surface], np.ndarray]:
    """Peel off the largest planar surfaces at any orientation. Returns (surfaces, residual_idx), where
    residual_idx is the points no plane explains; nothing is discarded."""
    work_pts = np.asarray(pcd.points)
    if level_rotation is not None:
        work_pts = work_pts @ level_rotation.T
    work_nz = None
    if find_behind_curtains and pcd.has_normals():    # the |z| of each normal, for find_true_wall
        nrm = np.asarray(pcd.normals)
        work_nz = np.abs((nrm @ level_rotation.T)[:, 2] if level_rotation is not None else nrm[:, 2])
    has_color = pcd.has_colors()
    work_rgb = np.asarray(pcd.colors) if has_color else None

    remaining = np.arange(len(work_pts))
    surfaces: list[Surface] = []
    for _ in range(max_surfaces):
        if remaining.size < min_support:
            break
        found = fit_plane(work_pts[remaining], distance_threshold, seed=seed + len(surfaces))
        if found is None:
            break
        plane, local_inliers = found
        if local_inliers.size < min_support:
            break
        inlier_idx = remaining[local_inliers]
        pts = work_pts[inlier_idx]
        normal = plane[:3]
        offset = -plane[3]
        moved_back = False
        if work_nz is not None and abs(normal[2]) < 0.30:    # an upright surface may be a curtain
            n2 = normal[:2] / np.linalg.norm(normal[:2])
            tw = find_true_wall(work_pts, work_nz, n2, float(pts[:, :2].mean(axis=0) @ n2), pts)
            if tw is not None:
                normal = np.array([tw.normal[0], tw.normal[1], 0.0])
                offset = tw.offset
                moved_back = True

        # In-plane axes from PCA, not an arbitrary cross product, because footprint() is not rotation-invariant.
        centroid = pts.mean(axis=0)
        if moved_back:                                       # the surface's centre goes onto the real wall too
            centroid = centroid + (offset - float(normal @ centroid)) * normal
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u0 = np.cross(normal, arbitrary); u0 /= np.linalg.norm(u0)
        v0 = np.cross(normal, u0)
        flat = np.column_stack([(pts - centroid) @ u0, (pts - centroid) @ v0])
        _, _, vh = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
        u = vh[0, 0] * u0 + vh[0, 1] * v0
        v = np.cross(normal, u)
        uv = np.column_stack([(pts - centroid) @ u, (pts - centroid) @ v])
        polys, area = footprint(uv, min_area_frac=0.5)
        if not polys:
            remaining = np.setdiff1d(remaining, inlier_idx, assume_unique=True)
            continue
        poly = polys[0]
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        ext = np.sort(hi - lo)[::-1]  # long axis first
        if ext[0] < min_extent:
            remaining = np.setdiff1d(remaining, inlier_idx, assume_unique=True)
            continue

        hue, sat, val = _rgb_to_hsv(work_rgb[inlier_idx]) if has_color else (float("nan"), 0.0, 0.0)
        # ext[0] is long-axis first but u/v are not sorted, so pick the axis with the larger spread.
        # find_corners needs real (u, v) axes to test whether a point falls inside this footprint.
        u_spread = float(np.abs(uv[:, 0]).max()) if uv.size else 0.0
        v_spread = float(np.abs(uv[:, 1]).max()) if uv.size else 0.0
        u_axis, v_axis = (u, v) if u_spread >= v_spread else (v, u)
        surfaces.append(Surface(
            normal=normal, offset=float(offset), centroid=centroid,
            support=int(inlier_idx.size), area=float(area),
            extent_u=float(ext[0]), extent_v=float(ext[1]),
            corners=len(poly), hue=hue, saturation=sat, value=val,
            u_axis=u_axis, v_axis=v_axis,
            index=len(surfaces),
        ))
        remaining = np.setdiff1d(remaining, inlier_idx, assume_unique=True)

    surfaces.sort(key=lambda s: -s.support)
    for i, s in enumerate(surfaces):
        s.index = i
    return surfaces, remaining


def build_neighbor_graph(surfaces: list[Surface], k: int = 6,
                         adjacency_gap_m: float = 0.5) -> None:
    """For each surface, its k nearest others by centroid distance. A plane plus its neighbours (angle, distance,
    relative size, whether they touch) is close to unique."""
    n = len(surfaces)
    for s in surfaces:
        s.neighbors = []
    if n < 2:
        return
    centroids = np.array([s.centroid for s in surfaces])
    for i, s in enumerate(surfaces):
        d = np.linalg.norm(centroids - s.centroid, axis=1)
        d[i] = np.inf
        order = np.argsort(d)[:k]
        for j in order:
            if not np.isfinite(d[j]):
                continue
            other = surfaces[j]
            ang = geometry.fold_between(s.normal, other.normal)
            ratio = s.area / other.area if other.area > 1e-6 else 0.0
            # cheap stand-in for a true nearest-edge test
            close = float(d[j]) < (s.extent_u + other.extent_u) / 2 + adjacency_gap_m
            s.neighbors.append(Neighbor(other=other.index, angle_deg=ang,
                                        distance_m=float(d[j]), size_ratio=ratio,
                                        adjacent=close))


@dataclass
class SurfaceMatch:
    src: int
    tgt: int
    score: float             # 0-1, size+shape+colour+neighbour combined
    size_score: float
    shape_score: float
    colour_score: Optional[float]   # None when neither surface had a usable colour
    neighbor_score: float


def _hue_circular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _pair_score(s: Surface, t: Surface, angle_deg: float) -> tuple[float, float, float, Optional[float]]:
    """size, shape, angle-derived, colour sub-scores, each 0 (bad) to 1 (good)."""
    size_ratio = min(s.area, t.area) / max(s.area, t.area, 1e-6)
    aspect_diff = abs(s.aspect - t.aspect) / max(s.aspect, t.aspect, 1e-6)
    shape_score = max(0.0, 1.0 - aspect_diff) * max(0.0, 1.0 - abs(s.corners - t.corners) / 6.0)
    colour_score = None
    if not (np.isnan(s.hue) or np.isnan(t.hue)):
        colour_score = max(0.0, 1.0 - _hue_circular_diff(s.hue, t.hue) / 90.0)
    return size_ratio, shape_score, colour_score


def match_surfaces(
    src_surfaces: list[Surface], tgt_surfaces: list[Surface],
    seed_transform: np.ndarray, position_bound_m: float, angle_bound_deg: float = 30.0,
    min_score: float = 0.35, min_shape_score: float = 0.25, min_neighbor_score: float = 0.25,
) -> tuple[list[SurfaceMatch], list[int], list[int]]:
    """Shortlist and score surface correspondences in the seeded frame. position_bound_m must not be
    cfg.local_only_bound_m; the min_*_score arguments are hard gates.
    Returns (matches, unmatched_src, unmatched_tgt)."""
    R = seed_transform[:3, :3]
    t = seed_transform[:3, 3]
    src_centroids_in_tgt = np.array([R @ s.centroid + t for s in src_surfaces]) if src_surfaces else np.zeros((0, 3))
    src_normals_in_tgt = np.array([R @ s.normal for s in src_surfaces]) if src_surfaces else np.zeros((0, 3))

    candidates: list[tuple[float, int, int, float, float, Optional[float]]] = []
    for i, s in enumerate(src_surfaces):
        c_i, n_i = src_centroids_in_tgt[i], src_normals_in_tgt[i]
        for j, tgt in enumerate(tgt_surfaces):
            dist = float(np.linalg.norm(c_i - tgt.centroid))
            if dist > position_bound_m:
                continue
            ang = geometry.fold_between(n_i, tgt.normal)
            if ang > angle_bound_deg:
                continue
            size_score, shape_score, colour_score = _pair_score(s, tgt, ang)
            if shape_score < min_shape_score:
                continue  # hard gate - see the docstring
            angle_score = max(0.0, 1.0 - ang / angle_bound_deg)
            proximity_score = max(0.0, 1.0 - dist / position_bound_m)
            base = 0.30 * size_score + 0.25 * shape_score + 0.25 * angle_score + 0.20 * proximity_score
            candidates.append((base, i, j, size_score, shape_score, colour_score))

    # Greedy one-to-one assignment, not an optimal (Hungarian) match.
    candidates.sort(key=lambda c: -c[0])
    used_src, used_tgt = set(), set()
    provisional: list[tuple[int, int, float, float, float, Optional[float]]] = []
    for base, i, j, size_score, shape_score, colour_score in candidates:
        if i in used_src or j in used_tgt:
            continue
        used_src.add(i); used_tgt.add(j)
        provisional.append((i, j, base, size_score, shape_score, colour_score))

    # Neighbour agreement, second pass: a matched pair's own confidence rises
    # when its friends also matched into the same neighbourhood.
    matched_pairs = {(i, j) for i, j, *_ in provisional}
    src_by_i = {s.index: s for s in src_surfaces}
    tgt_by_j = {t.index: t for t in tgt_surfaces}
    matches: list[SurfaceMatch] = []
    for i, j, base, size_score, shape_score, colour_score in provisional:
        s, tt = src_by_i[i], tgt_by_j[j]
        s_friend_matches = {n.other for n in s.neighbors}
        t_friend_set = {n.other for n in tt.neighbors}
        agree = sum(1 for (mi, mj) in matched_pairs
                    if mi in s_friend_matches and mj in t_friend_set)
        possible = max(1, min(len(s.neighbors), len(tt.neighbors)))
        neighbor_score = min(1.0, agree / possible)
        colour_term = 0.0 if colour_score is None else colour_score
        colour_weight = 0.0 if colour_score is None else 0.15
        combined = ((1 - 0.20 - colour_weight) * base + 0.20 * neighbor_score
                    + colour_weight * colour_term)
        if combined < min_score or neighbor_score < min_neighbor_score:
            continue
        matches.append(SurfaceMatch(src=i, tgt=j, score=combined, size_score=size_score,
                                    shape_score=shape_score, colour_score=colour_score,
                                    neighbor_score=neighbor_score))

    matched_src = {m.src for m in matches}
    matched_tgt = {m.tgt for m in matches}
    unmatched_src = [s.index for s in src_surfaces if s.index not in matched_src]
    unmatched_tgt = [t.index for t in tgt_surfaces if t.index not in matched_tgt]
    return matches, unmatched_src, unmatched_tgt


@dataclass
class ScanFeatures:
    """One scan's surfaces plus neighbour graph in its raw local frame, computed once. corners is filled in by
    find_corners.build_scan_corners; do not import that here."""
    surfaces: list[Surface]
    residual_idx: np.ndarray
    corners: list = field(default_factory=list)
    edges: list = field(default_factory=list)   # find_corners.build_scan_edges
    openings: list = field(default_factory=list)   # find_openings.build_scan_openings
    landmarks: list = field(default_factory=list)   # find_landmarks.build_scan_landmarks


def extract_scan_features(
    pcd: o3d.geometry.PointCloud,
    max_surfaces: int = 40, distance_threshold: float = 0.03,
    min_support: int = 300, min_extent: float = 0.4, seed: int = 42,
) -> ScanFeatures:
    """The per-scan pre-pass: every planar surface (walls, ceiling, floor, anything else) once, in the raw local
    frame with no levelling applied."""
    surfaces, residual_idx = segment_surfaces(
        pcd, level_rotation=None, max_surfaces=max_surfaces,
        distance_threshold=distance_threshold, min_support=min_support,
        min_extent=min_extent, seed=seed)
    build_neighbor_graph(surfaces)
    return ScanFeatures(surfaces=surfaces, residual_idx=residual_idx)


def feature_pair_score(
    src: ScanFeatures, tgt: ScanFeatures, seed_transform: np.ndarray,
    position_bound_m: float, angle_bound_deg: float = 30.0,
) -> tuple[float, int, list[SurfaceMatch]]:
    """How much of what two scans see is the same static structure, by matched surfaces. Use
    cfg.adjacency_feature_position_bound_m for position_bound_m, never local_only_bound_m.
    Returns (frac in [0, 1], n_matches, matches)."""
    if not src.surfaces or not tgt.surfaces:
        return 0.0, 0, []
    matches, _unmatched_src, _unmatched_tgt = match_surfaces(
        src.surfaces, tgt.surfaces, seed_transform,
        position_bound_m=position_bound_m, angle_bound_deg=angle_bound_deg)
    denom = min(len(src.surfaces), len(tgt.surfaces))
    frac = (sum(m.score for m in matches) / denom) if denom else 0.0
    return frac, len(matches), matches


def solve_pose_from_surfaces(
    src_surfaces: list[Surface], tgt_surfaces: list[Surface],
    matches: list[SurfaceMatch], min_matches: int = 3,
) -> Optional[np.ndarray]:
    """Solve a 4-DOF (yaw + translation) transform from matched surfaces of any orientation. None with too few
    matches or a degenerate set (e.g. floor only)."""
    if len(matches) < min_matches:
        return None
    src_by_i = {s.index: s for s in src_surfaces}
    tgt_by_j = {t.index: t for t in tgt_surfaces}

    us, vs, ws = [], [], []
    for m in matches:
        s, t = src_by_i[m.src], tgt_by_j[m.tgt]
        n_s_xy = s.normal[:2]
        # Normal sign is arbitrary; resolve it from the seed.
        n_t = t.normal if float(s.normal @ t.normal) >= 0 else -t.normal
        us.append(n_s_xy); vs.append(n_t[:2]); ws.append(m.score)
    us, vs, ws = np.array(us), np.array(vs), np.array(ws)
    horiz = np.linalg.norm(us, axis=1) * np.linalg.norm(vs, axis=1)
    ws = ws * horiz  # a near-horizontal surface's normal cannot pin yaw - let it vote near zero, not equally
    if ws.sum() < 1e-6:
        return None
    cross = (us[:, 0] * vs[:, 1] - us[:, 1] * vs[:, 0]) * ws
    dot = (us[:, 0] * vs[:, 0] + us[:, 1] * vs[:, 1]) * ws
    yaw = float(np.arctan2(cross.sum(), dot.sum()))
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    A, rhs, w_rows = [], [], []
    for m in matches:
        s, t = src_by_i[m.src], tgt_by_j[m.tgt]
        n_t = t.normal if float(s.normal @ t.normal) >= 0 else -t.normal
        o_t = t.offset if float(s.normal @ t.normal) >= 0 else -t.offset
        A.append(n_t); rhs.append(o_t - s.offset); w_rows.append(m.score)
    A, rhs, w_rows = np.array(A), np.array(rhs), np.array(w_rows)
    Aw = A * w_rows[:, None]
    if np.linalg.matrix_rank(Aw) < 3:
        return None  # matched surfaces do not span 3D - cannot pin translation
    t_xyz, *_ = np.linalg.lstsq(Aw, rhs * w_rows, rcond=None)

    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t_xyz
    return out
