"""Walls in a levelled scan: the plane search (vertical planes, a curtain moved back to the real wall behind it),
wall-pair matching for coarse registration, where along each wall the scan really has wall (breaks at doorways,
half walls, bulkheads), walls found from above, and the walls the bench draws, which combine the two finders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import permutations
from typing import Optional

import numpy as np
import open3d as o3d

from .align_gravity import estimate_level
from .lib.geometry import fold_between
from .lib.poses import rot2, tilt_of
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
            ang = fold_between(w.normal, m.normal)
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
            rot = rot2(theta)
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
    m[:2, :2] = rot2(theta)
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


# -------------------------------------------------------------------------------------------------------------------
# Curtains, and anything else hung in front of a wall, get detected as the wall. The ceiling does not stop at a
# curtain but carries on to the real wall, so where the ceiling ends along a detected wall says where the real wall
# is. Used by both wall detectors.
# -------------------------------------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------------------------------------
# Where along each detected wall line a scan really has wall, and how tall it is there, found from every point.
# A drawn wall stops where the wall stops, breaks at a doorway and keeps its real bottom and top, so a half wall or a
# bulkhead face is drawn as what it is. Used by the 3D export and the bench's walls.
# -------------------------------------------------------------------------------------------------------------------

NEAR_M = 0.04          # a point this close to the wall line, seen from above, is on the wall
BIN_M = 0.10           # the wall is looked at in 10 cm steps along its length
RUN_LEVEL_M = 0.05     # ... and in 5 cm levels up it
MIN_LEVEL_PTS = 2      # a level counts in a step when it holds this many points
RUN_MIN_TALL_M = 0.4   # a step is wall when its points stand at least this tall, wherever that is; a floor or
                       # ceiling edge alone is one or two levels, so it never counts
REACH_M = 1.0          # a run may reach this far past the detected wall's ends
RUN_CLOSE_M = 0.30     # gaps this short (a pipe, a dark patch) do not break a wall
MIN_RUN_M = 0.5        # shorter runs are left out
STACK_SHARE = 0.2      # a level is part of the run's wall when this share of its steps have it
STACK_GAP = 2          # levels missing this many in a row still join one wall body (a shelf, a dark band)
AROUND_M = 0.6         # floor and ceiling beside a run are read from points this close to it
KIND_M = 0.4           # this close to the floor or ceiling counts as reaching it
HANG_ABOVE_M = 1.5     # a wall starting this far above the floor, and reaching the ceiling, hangs from it
HALF_BELOW_M = 1.6     # a wall standing on the floor and stopping below this is a half wall
CROSS_CLEAR_M = 0.2    # two runs crossing further than this inside both are not two walls


def wall_runs(points_levelled: np.ndarray, walls: list[dict], min_level_pts: int = MIN_LEVEL_PTS,
              keep_past: bool = False) -> list[dict]:
    """The walls (payload rows: x1, y1, x2, y2, h, ...) cut to their solid runs, each with z_lo and z_hi (where the
    wall body starts and stops, scan frame) and kind: "full", "half wall", "hanging" or "behind curtain" (kept
    whole, since its own points are hidden). min_level_pts is 1 on the 3 cm preview grid, where one point per level
    is already a real surface. keep_past leaves _past on each row for a caller that runs drop_crossing again."""
    xy, z = points_levelled[:, :2], points_levelled[:, 2]
    out = []
    for w in walls:
        if float(w.get("behind_m", 0.0)) > 0:
            out.append({**w, "kind": "behind curtain", "z_lo": None, "z_hi": None})
            continue
        a, b = np.array([w["x1"], w["y1"]], float), np.array([w["x2"], w["y2"]], float)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        d = (b - a) / length
        nrm = np.array([-d[1], d[0]])
        rel = xy - a
        t, off = rel @ d, rel @ nrm
        span = (t > -REACH_M) & (t < length + REACH_M)
        on = span & (np.abs(off) < NEAR_M)
        if on.sum() < 10:
            continue
        tb = ((t[on] + REACH_M) / BIN_M).astype(np.int64)
        zmin = float(z[on].min())
        zl = ((z[on] - zmin) / RUN_LEVEL_M).astype(np.int64)
        nl = int(zl.max()) + 1
        nb = int((length + 2 * REACH_M) / BIN_M) + 1
        uniq, counts = np.unique(tb * nl + zl, return_counts=True)
        filled = uniq[counts >= min_level_pts]
        occ = np.zeros((nb, nl), bool)
        occ[filled // nl, filled % nl] = True
        solid = occ.sum(axis=1) * RUN_LEVEL_M >= RUN_MIN_TALL_M
        for start, stop in _runs(_close(solid, int(round(RUN_CLOSE_M / BIN_M)))):
            if (stop - start) * BIN_M < MIN_RUN_M:
                continue
            lo_l, hi_l = _body(occ[start:stop].mean(axis=0) >= STACK_SHARE)
            z_lo, z_hi = zmin + lo_l * RUN_LEVEL_M, zmin + (hi_l + 1) * RUN_LEVEL_M
            p, q = a + d * (start * BIN_M - REACH_M), a + d * (stop * BIN_M - REACH_M)
            near = span & (np.abs(off) < AROUND_M) & (t >= start * BIN_M - REACH_M) & (t < stop * BIN_M - REACH_M)
            floor_here, ceil_here = (np.percentile(z[near], [2, 98]) if near.sum() > 20 else (z_lo, z_hi))
            # Hanging: starts above head height and reaches the ceiling. Half wall: stands on the floor and stops low.
            # A wall whose foot is hidden behind furniture is still a full wall.
            kind = ("hanging" if z_lo - floor_here > HANG_ABOVE_M and ceil_here - z_hi < KIND_M else
                    "half wall" if z_lo - floor_here < KIND_M and z_hi - floor_here < HALF_BELOW_M else "full")
            out.append({**w, "x1": round(float(p[0]), 3), "y1": round(float(p[1]), 3),
                        "x2": round(float(q[0]), 3), "y2": round(float(q[1]), 3),
                        "len": round(float((stop - start) * BIN_M), 2), "kind": kind,
                        "z_lo": round(float(z_lo), 3), "z_hi": round(float(z_hi), 3),
                        "_past": (max(0.0, REACH_M - start * BIN_M), max(0.0, stop * BIN_M - REACH_M - length))})
    return drop_crossing(out) if keep_past else without_past(drop_crossing(out))


def without_past(runs: list[dict]) -> list[dict]:
    """The rows with drop_crossing's working field taken off."""
    return [{k: v for k, v in r.items() if k != "_past"} for r in runs]


def _body(levels: np.ndarray) -> tuple[int, int]:
    """(first, last) level of the tallest stack of levels, gaps of up to STACK_GAP allowed: the wall itself, apart
    from a ceiling or floor just above or below it."""
    best, best_len, start, last, gap = (0, 0), -1, None, None, 0
    for k, on in enumerate(levels):
        if on:
            if start is None:
                start = k
            last, gap = k, 0
        elif start is not None:
            gap += 1
            if gap > STACK_GAP:
                if last - start > best_len:
                    best, best_len = (start, last), last - start
                start, gap = None, 0
    if start is not None and last - start > best_len:
        best = (start, last)
    return best


def _inner_cross(a: dict, b: dict) -> float | None:
    """Where along a (0 to 1) it crosses b, when the crossing is further than CROSS_CLEAR_M inside both; else None."""
    p, r = np.array([a["x1"], a["y1"]]), np.array([a["x2"] - a["x1"], a["y2"] - a["y1"]])
    q, s = np.array([b["x1"], b["y1"]]), np.array([b["x2"] - b["x1"], b["y2"] - b["y1"]])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
    la, lb = np.linalg.norm(r), np.linalg.norm(s)
    inside = CROSS_CLEAR_M / la < t < 1 - CROSS_CLEAR_M / la and CROSS_CLEAR_M / lb < u < 1 - CROSS_CLEAR_M / lb
    return float(t) if inside else None


def _shallow(w: dict, t: float) -> bool:
    """Whether t (0 to 1 along w) lies in the stretch w was given past its found end, plus one step. A row without
    `_past` (a wall behind a curtain, a plain row) was given none, so it is never cut."""
    length = float(np.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]))
    return min(t, 1 - t) * length <= w.get("_past", (0.0, 0.0))[0 if t <= 0.5 else 1] + BIN_M


def _cut(w: dict, t: float) -> dict:
    """w cut back to t (0 to 1 along it) at its nearer end."""
    length = float(np.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]))
    cx, cy = w["x1"] + t * (w["x2"] - w["x1"]), w["y1"] + t * (w["y2"] - w["y1"])
    end = ("x1", "y1") if t <= 0.5 else ("x2", "y2")
    out = {**w, end[0]: round(float(cx), 3), end[1]: round(float(cy), 3),
           "len": round((1 - t if t <= 0.5 else t) * length, 2)}
    if "_past" in w:                                     # what is cut no longer counts as stretch
        lo, hi = w["_past"]
        out["_past"] = (max(0.0, lo - t * length), hi) if t <= 0.5 else (lo, max(0.0, hi - (1 - t) * length))
    return out


def drop_crossing(runs: list[dict]) -> list[dict]:
    """Walls meet at their ends (an L or a T) and never pass through each other. A run poking through a longer one
    within the stretch it was given past its found end ran past a corner and is cut back; where two runs cross
    further inside both, the shorter is something else (an open door, a shelf end) and is dropped. Rows carry _past
    from wall_runs."""
    order = sorted(range(len(runs)), key=lambda i: -float(runs[i].get("len", 0)))
    kept: dict[int, dict] = {}
    for i in order:
        w = runs[i]
        while True:                                      # every pass removes a crossing, so this ends
            hits = [(j, t) for j, k in kept.items() if (t := _inner_cross(w, k)) is not None]
            if not hits:
                kept[i] = w
                break
            if not all(_shallow(w, t) for _, t in hits):   # crossed well inside: not a wall
                break
            w = _cut(w, max((t for _, t in hits), key=lambda t: min(t, 1 - t)))   # the corner furthest in
            if w["len"] < MIN_RUN_M:
                break
    return [kept[i] for i in sorted(kept)]


def _close(solid: np.ndarray, gap: int) -> np.ndarray:
    """Fill runs of False no longer than `gap` that sit between two True steps."""
    out = solid.copy()
    for start, stop in _runs(~solid):
        if stop - start <= gap and start > 0 and stop < len(solid):
            out[start:stop] = True
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each run of True, stop exclusive."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


# -------------------------------------------------------------------------------------------------------------------
# Walls found the way a person finds them in plan view with the ceiling sliced off: 5 cm squares of sideways-facing
# points stacked tall vote for a line (direction from the points' facing, place from where they are), the strongest
# lines win, and each is cut where its tall squares stop. No random sampling, so the same scan always gives the same
# walls. Catches short pieces (returns, reveals, stubs) the plane search misses.
# -------------------------------------------------------------------------------------------------------------------

CELL_M = 0.05          # plan squares
ABOVE_LEVEL_M = 0.05   # height levels inside a square
ABOVE_MIN_TALL_M = 0.8  # a square counts when its points span at least this many levels' worth of height
MAX_NZ = 0.3           # a point faces sideways when its surface tips less than this from vertical
ABOVE_FLOOR_M = 0.08   # floor points left out
MIN_AGREE = 0.5        # a square's points must roughly agree on which way they face
ANGLE_BIN_DEG = 1.0    # vote grid: direction ...
SEAM_DEG = 45.0        # ... counted from -45 to 135 degrees, so walls along the grid never straddle its wrap
OFFSET_BIN_M = 0.04    # ... and place across the line
BAND_M = 0.05          # squares this close to a winning line belong to it
ANGLE_TOL_DEG = 6.0    # ... when they face within this of it
MIN_VOTES = 13.0       # weakest line kept (sum of its squares' heights in metres, smoothed)
MAX_LINES = 60
ABOVE_CLOSE_M = 1.0    # gaps along a line shorter than this do not split it
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
    levels = np.clip(np.floor(z / ABOVE_LEVEL_M).astype(np.int64), 0, 4095)
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
    tall = tall_count * ABOVE_LEVEL_M
    good = (tall >= ABOVE_MIN_TALL_M) & (agree > MIN_AGREE)
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
    four = 4 * np.arctan2(n[:, 1], n[:, 0])              # walls at right angles agree when x4
    turn = float(np.arctan2(np.sin(four).sum(), np.cos(four).sum()) / 4)
    rot = rot2(-turn)                                    # scan -> grid
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
        for piece in np.split(s, np.flatnonzero(np.diff(s) > ABOVE_CLOSE_M) + 1):
            if len(piece) == 0 or piece[-1] - piece[0] + CELL_M < MIN_LEN_M:
                continue
            a = nv * place + along * (piece[0] - CELL_M / 2)
            b = nv * place + along * (piece[-1] + CELL_M / 2)
            a, b = _settle(a, b, side_xy)
            walls.append({"x1": round(float(a[0]), 3), "y1": round(float(a[1]), 3),
                          "x2": round(float(b[0]), 3), "y2": round(float(b[1]), 3), "votes": len(piece)})
    return walls


# -------------------------------------------------------------------------------------------------------------------
# The walls the bench shows and the drawings start from: the plane search's walls cut to where the scan really
# has wall, plus short pieces only the from-above finder sees, with doubles (a second line just in front of a wall)
# removed.
# -------------------------------------------------------------------------------------------------------------------

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
    """The plane search's walls as full-length plan lines: what the wall reports measure with."""
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
