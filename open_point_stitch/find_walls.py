"""Wall-plane matching: coarse registration for rooms that are adjacent rather than overlapping. Two
non-parallel matched walls fully determine the transform (yaw and x/y; z comes from floors)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import open3d as o3d

from .align_gravity import estimate_level, tilt_of
from .find_true_walls import find_true_wall
from .limits import single_threaded

log = logging.getLogger(__name__)


@dataclass
class Wall:
    normal: np.ndarray      # unit 2D normal in the levelled XY plane
    offset: float           # plane is  normal . xy = offset
    support: int            # inlier count
    length: float           # extent along the wall, metres
    height: float           # vertical extent, metres
    centroid: np.ndarray    # 3D centroid of the inliers
    index: int = -1
    behind_curtain_m: float = 0.0   # >0: the detected surface was a curtain; the wall is this far behind it

    @property
    def area(self) -> float:
        return self.length * self.height

    def describe(self) -> str:
        ang = np.degrees(np.arctan2(self.normal[1], self.normal[0])) % 180.0
        return (f"W{self.index}: normal {ang:6.1f} deg, offset {self.offset:7.2f} m, "
                f"{self.length:4.1f} x {self.height:4.1f} m, {self.support:6,d} pts")


def fit_plane(points: np.ndarray, distance_threshold: float,
              iterations: int = 1200, sample_size: int = 4000,
              seed: int = 42):
    """Deterministic numpy RANSAC plane fit. Returns (plane, inlier indices) or None, plane being (a, b, c, d)
    with a unit normal, a*x + b*y + c*z + d = 0.

    Not used by extract_walls on purpose: swapping it in changes alignment results. Keep Open3D's segment_plane."""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return None

    rng = np.random.default_rng(seed)
    tri = rng.integers(0, n, size=(iterations, 3))
    ok = ((tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2])
          & (tri[:, 0] != tri[:, 2]))
    tri = tri[ok]
    if tri.shape[0] == 0:
        return None

    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-9
    if not good.any():
        return None
    normals = normals[good] / lengths[good, None]
    d = -np.einsum("ij,ij->i", normals, a[good])

    # Two-stage scoring: shortlist cheaply on a subsample, then score the shortlist against every point.
    # Scoring on the subsample alone is too noisy.
    m = min(sample_size, n)
    probe = pts[rng.choice(n, size=m, replace=False)] if m < n else pts
    rough = (np.abs(probe @ normals.T + d[None, :]) < distance_threshold).sum(axis=0)

    shortlist = np.argsort(rough)[::-1][:32]
    exact = np.empty(shortlist.size, dtype=np.int64)
    for k, j in enumerate(shortlist):
        exact[k] = int(np.count_nonzero(
            np.abs(pts @ normals[j] + d[j]) < distance_threshold))
    best = int(shortlist[int(np.argmax(exact))])

    plane = np.array([normals[best, 0], normals[best, 1], normals[best, 2], d[best]])
    inliers = np.flatnonzero(np.abs(pts @ plane[:3] + plane[3]) < distance_threshold)
    if inliers.size < 3:
        return None

    # Refit on all inliers so the answer does not hinge on which three points
    # happened to be drawn.
    centre = pts[inliers].mean(axis=0)
    _, _, vh = np.linalg.svd(pts[inliers] - centre, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    plane = np.array([normal[0], normal[1], normal[2], -float(normal @ centre)])
    inliers = np.flatnonzero(np.abs(pts @ plane[:3] + plane[3]) < distance_threshold)
    if inliers.size < 3:
        return None
    return plane, inliers


def _tightness_weights(pts: np.ndarray, n2: np.ndarray, bin_m: float = 0.15, min_bin: int = 5) -> np.ndarray:
    """How much each inlier counts toward the wall's fitted position, by how tightly its small patch scatters across
    the wall. A patch with fewer than min_bin points gets the median weight."""
    along = np.array([-n2[1], n2[0]])
    tangent = pts[:, :2] @ along        # position along the wall
    height = pts[:, 2]                  # position up the wall
    normal_pos = pts[:, :2] @ n2        # position along the wall's own normal - scatter here is the signal

    bt = np.floor(tangent / bin_m).astype(np.int64)
    bh = np.floor(height / bin_m).astype(np.int64)
    # A single sortable key per 2D bin. bh's real range is a few hundred bins at most (a scan is
    # metres tall, bin_m is centimetres), so this cannot collide two different bins in practice.
    keys = bt * 1_000_003 + bh

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_keys)) + 1])
    ends = np.concatenate([starts[1:], [len(sorted_keys)]])

    weights = np.full(len(pts), np.nan)
    for s, e in zip(starts, ends):
        idx = order[s:e]
        if e - s >= min_bin:
            spread = float(np.std(normal_pos[idx]))
            weights[idx] = 1.0 / (spread + 0.003)  # 3mm floor: a suspiciously-perfect patch
            # (near-duplicate points) must not outweigh everything else outright.
    unmeasured = np.isnan(weights)
    if unmeasured.any():
        fallback = float(np.median(weights[~unmeasured])) if (~unmeasured).any() else 1.0
        weights[unmeasured] = fallback
    return weights


def extract_walls(
    pcd: o3d.geometry.PointCloud,
    level_rotation: np.ndarray | None = None,
    max_walls: int = 8,
    distance_threshold: float = 0.04,
    min_support: int = 400,
    min_length: float = 0.8,
    min_height: float = 0.8,
    seed: int = 42,
    tightness_weighted: bool = False,
    find_behind_curtains: bool = True,
) -> list[Wall]:
    """Find the dominant vertical planes in an already levelled cloud, largest first (only points with
    near-horizontal normals are searched). tightness_weighted: weight each inlier's contribution to the wall's
    position by how tightly its local patch scatters, instead of a plain average."""
    work = o3d.geometry.PointCloud(pcd)
    if level_rotation is not None:
        work.rotate(level_rotation, center=(0, 0, 0))
    if not work.has_normals():
        work.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
    nz = np.abs(np.asarray(work.normals)[:, 2])
    all_pts, all_nz = np.asarray(work.points).copy(), nz.copy()     # for find_true_wall, which needs the ceiling
    vertical_idx = np.flatnonzero(nz < 0.30)
    if vertical_idx.size < min_support:
        return []

    # Open3D's segment_plane, not fit_plane(): do not swap them.
    work = work.select_by_index(vertical_idx)

    # Single-threaded for the RANSAC only (its guesses race across threads); set the seed inside the lock.
    walls: list[Wall] = []
    with single_threaded():
        try:
            o3d.utility.random.seed(int(seed))
        except Exception:
            pass
        for _ in range(max_walls * 3):
            if len(work.points) < min_support or len(walls) >= max_walls:
                break
            try:
                model, inliers = work.segment_plane(
                    distance_threshold=distance_threshold, ransac_n=3,
                    num_iterations=2000
                )
            except RuntimeError:
                break
            if len(inliers) < min_support:
                break

            pts = np.asarray(work.points)[inliers]
            a, b, c, d = model
            n3 = np.array([a, b, c], dtype=float)
            inliers = np.asarray(inliers)
            work = work.select_by_index(list(inliers), invert=True)

            horiz = np.linalg.norm(n3[:2])
            if horiz < 1e-6 or abs(n3[2]) / np.linalg.norm(n3) > 0.30:
                continue  # not a vertical plane

            n2 = n3[:2] / horiz
            # offset from the inlier points, not -d/horiz (which assumes c=0)
            if tightness_weighted:
                offset = float(np.average(pts[:, :2] @ n2, weights=_tightness_weights(pts, n2)))
            else:
                offset = float(np.mean(pts[:, :2] @ n2))
            # Canonical sign, so a wall found from either side compares the same way.
            if n2[0] < 0 or (abs(n2[0]) < 1e-9 and n2[1] < 0):
                n2 = -n2
                offset = -offset

            along = np.array([-n2[1], n2[0]])
            length = float(np.ptp(pts[:, :2] @ along))
            height = float(np.ptp(pts[:, 2]))
            if length < min_length or height < min_height:
                continue

            behind = 0.0
            centroid = pts.mean(axis=0)
            if find_behind_curtains:     # a curtain is not the wall: use where the ceiling ends
                tw = find_true_wall(all_pts, all_nz, n2, offset, pts)
                if tw is not None:
                    n2, offset, behind = tw.normal, tw.offset, tw.behind_m
                    centroid = centroid.copy()
                    centroid[:2] += (offset - float(n2 @ centroid[:2])) * n2     # onto the real wall, for drawing
            walls.append(Wall(
                normal=n2, offset=offset, support=int(inliers.size),
                length=length, height=height, centroid=centroid,
                index=len(walls), behind_curtain_m=behind,
            ))

    walls = merge_duplicate_walls(walls)
    walls.sort(key=lambda w: -w.support)
    for i, w in enumerate(walls):
        w.index = i
    return walls


def merge_duplicate_walls(walls: list[Wall],
                          angle_tol_deg: float = 6.0,
                          offset_tol: float = 0.25) -> list[Wall]:
    """Fold together planes that are really the same wall found twice (RANSAC splits long walls)."""
    merged: list[Wall] = []
    for w in sorted(walls, key=lambda x: -x.support):
        hit = None
        for m in merged:
            ang = abs(np.degrees(np.arccos(np.clip(abs(float(w.normal @ m.normal)), -1, 1))))
            if ang > angle_tol_deg:
                continue
            # Compare the candidate's centroid against the kept wall's plane. Raw offsets are unreliable far from
            # the origin, where a fraction of a degree becomes a large offset difference.
            gap = abs(float(m.normal @ w.centroid[:2]) - m.offset)
            if gap <= offset_tol:
                hit = m
                break
        if hit is None:
            merged.append(w)
            continue
        # Keep the strongest detection's plane exactly as fitted. Averaging offsets across slightly different normals
        # lands on neither plane far from the origin. Only support and extent accumulate.
        hit.length = max(hit.length, w.length)
        hit.height = max(hit.height, w.height)
        hit.support = hit.support + w.support
    return merged


def _yaw_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def solve_from_wall_pair(
    src_a: Wall, src_b: Wall, tgt_a: Wall, tgt_b: Wall,
    src_floor: float, tgt_floor: float,
    angle_tolerance_deg: float = 8.0,
    min_wall_angle_deg: float = 20.0,
) -> np.ndarray | None:
    """Transform sending src wall a onto tgt wall a and src b onto tgt b.
    None if the angles between the wall pairs disagree or the walls are near-parallel."""
    def signed_angle(u, v):
        return np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)

    # Walls must be distinct enough in direction to pin down a translation.
    for pair in ((src_a, src_b), (tgt_a, tgt_b)):
        between = abs(np.degrees(signed_angle(pair[0].normal, pair[1].normal)))
        between = min(between, 180.0 - between)
        if between < min_wall_angle_deg:
            return None

    # A wall's normal sign is arbitrary, so try both flips of each target wall.
    best = None
    for fa in (1.0, -1.0):
        for fb in (1.0, -1.0):
            na, oa = tgt_a.normal * fa, tgt_a.offset * fa
            nb, ob = tgt_b.normal * fb, tgt_b.offset * fb

            theta = signed_angle(src_a.normal, na)
            rot = _yaw_matrix(theta)
            # Consistency: the second wall must land on its counterpart too.
            err = abs(np.degrees(signed_angle(rot @ src_b.normal, nb)))
            err = min(err, 360.0 - err)
            if err > angle_tolerance_deg:
                continue

            # n . t = target_offset - source_offset, for both walls.
            A = np.vstack([na, nb])
            rhs = np.array([oa - src_a.offset, ob - src_b.offset])
            if abs(np.linalg.det(A)) < 1e-6:
                continue
            t_xy = np.linalg.solve(A, rhs)

            if best is None or err < best[0]:
                best = (err, theta, t_xy)

    if best is None:
        return None

    _, theta, t_xy = best
    m = np.eye(4)
    m[:2, :2] = _yaw_matrix(theta)
    m[0, 3] = t_xy[0]
    m[1, 3] = t_xy[1]
    m[2, 3] = tgt_floor - src_floor
    return m


def wall_candidates(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    max_walls: int = 6,
    max_candidates: int = 40,
    tightness_weighted: bool = False,
) -> tuple[list[np.ndarray], list[Wall], list[Wall]]:
    """Enumerate transforms implied by every plausible pair of wall matches. Returns candidates in the original
    (unlevelled) frames, plus the wall lists for reporting."""
    src_level = estimate_level(src)
    tgt_level = estimate_level(tgt)

    src_walls = extract_walls(src, src_level.rotation, max_walls=max_walls, tightness_weighted=tightness_weighted)
    tgt_walls = extract_walls(tgt, tgt_level.rotation, max_walls=max_walls, tightness_weighted=tightness_weighted)
    if len(src_walls) < 2 or len(tgt_walls) < 2:
        log.warning("not enough walls found (%d source, %d target)",
                    len(src_walls), len(tgt_walls))
        return [], src_walls, tgt_walls

    scored: list[tuple[float, np.ndarray]] = []
    for sa, sb in permutations(src_walls, 2):
        for ta, tb in permutations(tgt_walls, 2):
            m = solve_from_wall_pair(
                sa, sb, ta, tb, src_level.floor_z, tgt_level.floor_z
            )
            if m is None:
                continue
            # Prefer hypotheses built on large, well supported walls.
            weight = min(sa.area, ta.area) + min(sb.area, tb.area)
            scored.append((weight, m))

    scored.sort(key=lambda x: -x[0])

    rs = np.eye(4); rs[:3, :3] = src_level.rotation
    rt_inv = np.eye(4); rt_inv[:3, :3] = tgt_level.rotation.T

    out: list[np.ndarray] = []
    for _, m in scored:
        full = rt_inv @ m @ rs
        if tilt_of(full) > 5.0:
            continue
        # Drop near duplicates, which are common when two walls are similar.
        if any(np.allclose(full, prev, atol=0.05) for prev in out):
            continue
        out.append(full)
        if len(out) >= max_candidates:
            break

    log.info("wall matching: %d source walls, %d target walls, %d candidates",
             len(src_walls), len(tgt_walls), len(out))
    return out, src_walls, tgt_walls


def solve_from_named_pairs(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    pairs: list[tuple[int, int]],
    max_walls: int = 6,
) -> tuple[list[np.ndarray], list[str], list[Wall], list[Wall]]:
    """Build candidates from wall pairs a person labelled: two pairs give one candidate,
    one pair enumerates the second correspondence."""
    src_level = estimate_level(src)
    tgt_level = estimate_level(tgt)
    src_walls = extract_walls(src, src_level.rotation, max_walls=max_walls)
    tgt_walls = extract_walls(tgt, tgt_level.rotation, max_walls=max_walls)

    def get(walls, i, which):
        if i < 0 or i >= len(walls):
            raise ValueError(
                f"{which} has no wall W{i}. It has W0..W{len(walls) - 1}."
            )
        return walls[i]

    rs = np.eye(4); rs[:3, :3] = src_level.rotation
    rt_inv = np.eye(4); rt_inv[:3, :3] = tgt_level.rotation.T

    combos: list[tuple[Wall, Wall, Wall, Wall, str]] = []
    if len(pairs) >= 2:
        (sa_i, ta_i), (sb_i, tb_i) = pairs[0], pairs[1]
        combos.append((get(src_walls, sa_i, "source"), get(src_walls, sb_i, "source"),
                       get(tgt_walls, ta_i, "target"), get(tgt_walls, tb_i, "target"),
                       f"W{sa_i}=W{ta_i}, W{sb_i}=W{tb_i}"))
    elif len(pairs) == 1:
        sa_i, ta_i = pairs[0]
        sa = get(src_walls, sa_i, "source")
        ta = get(tgt_walls, ta_i, "target")
        for sb in src_walls:
            if sb is sa:
                continue
            for tb in tgt_walls:
                if tb is ta:
                    continue
                combos.append((sa, sb, ta, tb,
                               f"W{sa_i}=W{ta_i}, W{sb.index}=W{tb.index}"))
    else:
        raise ValueError("supply at least one wall pair")

    out: list[np.ndarray] = []
    labels: list[str] = []
    for sa, sb, ta, tb, label in combos:
        m = solve_from_wall_pair(sa, sb, ta, tb, src_level.floor_z, tgt_level.floor_z)
        if m is None:
            continue
        full = rt_inv @ m @ rs
        if tilt_of(full) > 5.0:
            continue
        if any(np.allclose(full, prev, atol=0.05) for prev in out):
            continue
        out.append(full)
        labels.append(label)
    return out, labels, src_walls, tgt_walls
