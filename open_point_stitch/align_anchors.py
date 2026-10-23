"""Align from links: solve new poses and write them back; nothing is merged.

Each link is matched on the full-resolution points inside its two boxes; a box with too few points falls back to
the whole scan's preview points. Corrections are 4 DOF (yaw, x, y, z): Open3D finds correspondences,
lib.poses.solve_yaw_translation solves, and every link is then combined in one solve (align_graph.solve_4dof)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from . import align_graph as graph4
from .lib.clouds import as_cloud
from .lib.geometry import points_in_box
from .lib.groups import connected_groups
from .lib.poses import (ScanPose, raw_to_world_point, raw_to_world_points, solve_yaw_translation, tilt_of,
                        world_cloud_points_as_drawn, yaw_gap_deg, yaw_of, yaw_transform)
from .pts_file import find_scans, read_pts

log = logging.getLogger(__name__)

reg = o3d.pipelines.registration

# Boxed matching
MIN_REGION_POINTS = 200     # a box needs this many full-resolution points, on BOTH scans, to be matched on its own
REGION_MARGIN_M = 1.0       # the reference box is grown this much, so the other box's edge does not meet a cut,
REGION_GROW_STEP_M = 0.15   # and a box short of MIN_REGION_POINTS is grown by this much at a time, up to that
REGION_START_M = 2.0        # matching distance for the first round of a boxed match
REGION_FINAL_M = 0.15       # ...shrinking to this
INLIER_M = 0.05             # a boxed point "lines up" when the other scan has a point this close
SEED_KEEP_MARGIN = 0.02     # matching must line up this much MORE of the box than the anchor seed alone to replace it
# A link's starting height comes from floors and ceilings, never from where its boxes were drawn.
FLAT_NZ = 0.9               # a floor or ceiling: its surface faces within about 25 degrees of straight up or down
FLAT_PAIR_M = 0.5           # the two scans' floors are paired when this close
MIN_FLAT_PAIRS = 100        # fewer pairs than this and the link does not change height at the start
PLANE_PAIR_M = 0.3          # along_normal: points paired within this, both on parallel flat surfaces
PLANE_PARALLEL = 0.98       # ... whose normals agree within about 11 degrees


@dataclass
class AlignRow:
    scan: str
    against: str
    fitness: float
    shift_mean_m: float
    shift_max_m: float
    discarded_tilt_deg: float
    applied: bool
    reason: str = ""
    matched_on: str = ""        # what the match ran on: the boxes at full resolution, or the whole-scan preview and why


@dataclass
class AlignResult:
    poses: dict[str, ScanPose]          # updated (or unchanged) current poses
    rows: list[AlignRow] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # anchors that named an unknown scan
    cancelled: bool = False             # stopped by should_cancel before every anchor was tried
    n_anchors_completed: int = 0        # anchors actually processed, whether or not cancelled


def full_res_region(scans_folder: str, member, cache: dict, margin_m: float = 0.0) -> Optional[np.ndarray]:
    """Every full-resolution point (the scan file's raw frame) inside a link member's box, grown by margin_m on
    every side. None if the scan file cannot be read (logged); the caller then uses the preview. cache maps scan
    name to its points, so each file is read once per run."""
    try:
        if member.scan not in cache:
            matches = [p for p in find_scans(scans_folder, "*.pts")
                       if os.path.splitext(os.path.basename(p))[0] == member.scan]
            if not matches:
                log.warning("no scan file named %r in %s, so anchors on it use the preview points",
                            member.scan, scans_folder)
            cache[member.scan] = read_pts(matches[0]).points if matches else None
        pts = cache[member.scan]
        if pts is None:
            return None
        return pts[points_in_box(pts, member.centroid, np.asarray(member.extent, dtype=float) + 2.0 * margin_m)]
    except Exception:
        cache[member.scan] = None   # do not retry, or log, on every anchor that names this scan
        log.exception("could not read the full-resolution points of %s in %s, so anchors on it use the "
                      "preview points instead", member.scan, scans_folder)
        return None


def _agreement_weights(payload: dict) -> np.ndarray:
    """Per-point agreement score; all ones if the payload lacks it."""
    n = len(payload["xy"])
    agree = payload.get("agreement")
    if agree is None or len(agree) != n:
        return np.ones(n, dtype=float)
    return np.asarray(agree, dtype=float)


@dataclass
class _FourDofResult:
    transform: np.ndarray       # 4x4, world frame, src points -> corrected src points
    fitness: float
    rmse: float
    discarded_tilt_deg: float
    n_correspondences: int
    correspondences: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))


def _icp_4dof(
    src_pts: np.ndarray,
    src_weights: np.ndarray,
    tgt_pts: np.ndarray,
    dist_threshold: float,
    max_iteration: int = 30,
    tol_deg: float = 1e-3,
    tol_m: float = 1e-4,
    init: Optional[np.ndarray] = None,
    start_threshold: Optional[float] = None,
) -> _FourDofResult:
    """ICP where Open3D finds correspondences and the closed-form 4 DOF solver solves. Points are in the world
    frame, so the result is a correction on top of the current pose. The threshold narrows to dist_threshold by
    iteration 4, or with start_threshold, shrinks 30% a round from there down to dist_threshold."""
    src_cloud = as_cloud(src_pts)
    tgt_cloud = as_cloud(tgt_pts)
    point_est = reg.TransformationEstimationPointToPoint()
    one_step = reg.ICPConvergenceCriteria(max_iteration=1)

    # init: translation-only seed from a link's region
    current = np.eye(4) if init is None else np.array(init, dtype=float)
    corr = np.zeros((0, 2), dtype=int)
    for i in range(max_iteration):
        thresh = (dist_threshold * max(1, 4 - i) if start_threshold is None
                  else max(dist_threshold, start_threshold * 0.7 ** i))
        found = reg.registration_icp(src_cloud, tgt_cloud, thresh, current, point_est, one_step)
        corr = np.asarray(found.correspondence_set)
        if corr.shape[0] < 2:
            break

        matched_src = src_pts[corr[:, 0]]
        matched_tgt = tgt_pts[corr[:, 1]]
        matched_w = src_weights[corr[:, 0]]

        yaw_deg, t = solve_yaw_translation(matched_src, matched_tgt, matched_w)
        new_current = yaw_transform(yaw_deg, t)

        delta_yaw = yaw_gap_deg(yaw_of(new_current), yaw_of(current))
        delta_t = float(np.linalg.norm(new_current[:3, 3] - current[:3, 3]))
        current = new_current
        if delta_yaw < tol_deg and delta_t < tol_m:
            break

    n_corr = int(corr.shape[0])
    if n_corr >= 2:
        matched_src = src_pts[corr[:, 0]]
        matched_tgt = tgt_pts[corr[:, 1]]
        rmse = float(np.sqrt(np.mean(
            np.sum(((current[:3, :3] @ matched_src.T).T + current[:3, 3] - matched_tgt) ** 2, axis=1)
        )))
        tilt = _free_tilt_deg(matched_src, matched_tgt)
    else:
        rmse = 0.0
        tilt = 0.0

    fitness = n_corr / len(src_pts) if len(src_pts) else 0.0
    return _FourDofResult(transform=current, fitness=fitness, rmse=rmse,
                          discarded_tilt_deg=tilt, n_correspondences=n_corr, correspondences=corr)


def _free_tilt_deg(matched_src: np.ndarray, matched_tgt: np.ndarray) -> float:
    """Diagnostic only, never applied: the tilt an unconstrained fit of these matched points would want."""
    n = len(matched_src)
    free_corr = o3d.utility.Vector2iVector(np.column_stack([np.arange(n), np.arange(n)]))
    src_c = as_cloud(matched_src)
    tgt_c = as_cloud(matched_tgt)
    free = np.asarray(reg.TransformationEstimationPointToPoint().compute_transformation(src_c, tgt_c, free_corr))
    return tilt_of(free)


@dataclass
class _MatchSet:
    src_pts: np.ndarray         # world frame: the scan being corrected
    src_weights: np.ndarray
    tgt_pts: np.ndarray         # world frame: the reference scan
    matched_on: str             # plain words, for the log, the row and the report
    used_regions: bool          # True when these are the boxes' full-resolution points


@dataclass
class _BoxPoints:
    raw: Optional[np.ndarray]   # full-resolution points in the scan file's raw frame; None if the file was not available
    grown_m: float = 0.0        # how far the box had to be grown to reach MIN_REGION_POINTS


def _preview_points_in_box(payload: dict, member, grown_m: float) -> int:
    """How many of the scan's browser PREVIEW points sit in an anchor member's box, grown by `grown_m`."""
    leveled = np.column_stack([np.asarray(payload["xy"], dtype=float).reshape(-1, 2),
                               np.asarray(payload["h"], dtype=float) + payload["floor_z"]])
    return int(points_in_box(leveled @ np.asarray(payload["level_rotation"], dtype=float),
                             member.centroid, np.asarray(member.extent, dtype=float) + 2.0 * grown_m).sum())


def _grown_note(name: str, box: _BoxPoints) -> str:
    return f", the box on {name} grown by {box.grown_m:.2f} m to reach {MIN_REGION_POINTS} points" if box.grown_m else ""


def _points_to_match(a_name, a_pl, a_pose, tgt: _BoxPoints, tgt_context, b_name, b_pl, b_pose, src: _BoxPoints,
                     src_member) -> _MatchSet:
    """What ICP matches for one pair (a is the reference, b is corrected): b's full-resolution box points against
    a's box grown by REGION_MARGIN_M, or the whole-scan preview points if either box has fewer than
    MIN_REGION_POINTS or its scan file is unavailable. Each full-resolution point is weighted like the preview
    points it stands in for, so a huge box does not outvote everything else in the joint solve."""
    n_src = None if src.raw is None else len(src.raw)
    n_tgt = None if tgt.raw is None else len(tgt.raw)
    if n_src is not None and n_tgt is not None and min(n_src, n_tgt) >= MIN_REGION_POINTS:
        n_preview = max(_preview_points_in_box(b_pl, src_member, src.grown_m), 1)
        return _MatchSet(
            src_pts=raw_to_world_points(src.raw, b_pl["level_rotation"], b_pl["floor_z"], b_pose),
            src_weights=np.full(n_src, min(1.0, n_preview / n_src)),
            tgt_pts=raw_to_world_points(tgt_context, a_pl["level_rotation"], a_pl["floor_z"], a_pose),
            matched_on=(f"the boxes at full resolution ({n_src:,} points on {b_name}, {n_tgt:,} on {a_name}"
                        f"{_grown_note(b_name, src)}{_grown_note(a_name, tgt)})"),
            used_regions=True)
    if n_src is None or n_tgt is None:
        why = "the full-resolution scan was not available"
    else:
        why = (f"a box held too few full-resolution points even grown by {REGION_MARGIN_M:.1f} m "
               f"({n_src:,} on {b_name}, {n_tgt:,} on {a_name}, needs {MIN_REGION_POINTS})")
    return _MatchSet(
        src_pts=world_cloud_points_as_drawn(b_pl, b_pose), src_weights=_agreement_weights(b_pl),
        tgt_pts=world_cloud_points_as_drawn(a_pl, a_pose),
        matched_on=f"the whole scan at preview resolution, because {why}", used_regions=False)


@dataclass
class _PairFit:
    transform: np.ndarray               # the correction chosen for this pair, world frame
    fitness: float                      # what the fitness gate reads (see `_fit_pair`)
    discarded_tilt_deg: float
    correspondences: np.ndarray         # (source index, target index) pairs at `transform`
    note: str = ""                      # plain words on how the transform was chosen, for the log and the report
    kept_seed: bool = False             # boxes only: matching did not beat the anchor seed, so the seed stands


def flat_height_gap(src_pts: np.ndarray, tgt_pts: np.ndarray, along_normal: bool = False) -> tuple[int, float]:
    """(pairs, median height of src above tgt in m) over tgt's floors and ceilings; (0, nan) if none.
    along_normal: both points must sit on parallel flat surfaces, and the gap is the height above tgt's local
    surface, so a sloped or raked floor seen a little sideways does not read as a height gap."""
    src = as_cloud(src_pts)
    tgt = as_cloud(tgt_pts)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    if along_normal:
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
        corr = np.asarray(reg.evaluate_registration(src, tgt, PLANE_PAIR_M, np.eye(4)).correspondence_set)
        if not len(corr):
            return 0, float("nan")
        ns, nt = np.asarray(src.normals)[corr[:, 0]], np.asarray(tgt.normals)[corr[:, 1]]
        keep = (np.abs(nt[:, 2]) >= FLAT_NZ) & (np.abs(ns[:, 2]) >= FLAT_NZ) & (np.abs((ns * nt).sum(1)) >= PLANE_PARALLEL)
        corr, nt = corr[keep], nt[keep]
        if not len(corr):
            return 0, float("nan")
        d = src_pts[corr[:, 0]] - tgt_pts[corr[:, 1]]
        return len(corr), float(np.median((d * nt).sum(1) / nt[:, 2]))
    corr = np.asarray(reg.evaluate_registration(src, tgt, FLAT_PAIR_M, np.eye(4)).correspondence_set)
    if len(corr):
        corr = corr[np.abs(np.asarray(tgt.normals)[corr[:, 1], 2]) >= FLAT_NZ]
    if not len(corr):
        return 0, float("nan")
    return len(corr), float(np.median(src_pts[corr[:, 0], 2] - tgt_pts[corr[:, 1], 2]))


def _seed_height(match: "_MatchSet") -> float:
    """The link's starting height change: close the floor/ceiling gap, or none without floors."""
    n, gap = flat_height_gap(match.src_pts, match.tgt_pts)
    return 0.0 if n < MIN_FLAT_PAIRS else -gap


def _settle_height(match: "_MatchSet", transform: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """(transform, has_floors, plain words): the match's height re-set so the floors and ceilings meet;
    with none in the boxes, height is left where it was and the link must not pull on it."""
    out = np.array(transform, dtype=float)
    n, gap = flat_height_gap(match.src_pts @ out[:3, :3].T + out[:3, 3], match.tgt_pts)
    if n < MIN_FLAT_PAIRS:
        out[2, 3] = 0.0
        return out, False, "no floor or ceiling in the boxes, so this link leaves the height alone"
    out[2, 3] -= gap
    return out, True, f"height set where the floors and ceilings in the boxes meet ({n:,} points)"


def _inlier_share(src_pts: np.ndarray, tgt_pts: np.ndarray, transform: np.ndarray, radius_m: float) -> float:
    """Share of `src_pts`, moved by `transform`, that have a point of `tgt_pts` within `radius_m`."""
    src = as_cloud(src_pts)
    tgt = as_cloud(tgt_pts)
    return float(reg.evaluate_registration(src, tgt, radius_m, np.asarray(transform, dtype=float)).fitness)


def _fit_pair(match: _MatchSet, seed: np.ndarray, icp_distance_m: float, icp_max_iteration: int) -> _PairFit:
    """One pair's correction. On preview points it is _icp_4dof from the link seed. On boxes, ICP has nothing to
    hold a flat or small box in place, so its answer is only taken if it lines up more of the box (within INLIER_M)
    than the seed alone, by SEED_KEEP_MARGIN; otherwise the seed stands."""
    if not match.used_regions:
        fit = _icp_4dof(match.src_pts, match.src_weights, match.tgt_pts, icp_distance_m,
                        max_iteration=icp_max_iteration, init=seed)
        return _PairFit(fit.transform, fit.fitness, fit.discarded_tilt_deg, fit.correspondences)
    fit = _icp_4dof(match.src_pts, match.src_weights, match.tgt_pts, REGION_FINAL_M,
                    max_iteration=icp_max_iteration, init=seed, start_threshold=REGION_START_M)
    seed_share = _inlier_share(match.src_pts, match.tgt_pts, seed, INLIER_M)
    icp_share = _inlier_share(match.src_pts, match.tgt_pts, fit.transform, INLIER_M)
    if icp_share >= seed_share + SEED_KEEP_MARGIN:
        return _PairFit(fit.transform, icp_share, fit.discarded_tilt_deg, fit.correspondences,
                        f"matching moved the boxes from {seed_share:.0%} to {icp_share:.0%} lined up within "
                        f"{INLIER_M * 100:.0f} cm")
    src_c = as_cloud(match.src_pts)
    tgt_c = as_cloud(match.tgt_pts)
    corr = np.asarray(reg.evaluate_registration(src_c, tgt_c, REGION_FINAL_M, np.asarray(seed, dtype=float)).correspondence_set)
    tilt = _free_tilt_deg(match.src_pts[corr[:, 0]], match.tgt_pts[corr[:, 1]]) if len(corr) >= 2 else 0.0
    return _PairFit(np.asarray(seed, dtype=float), seed_share, tilt, corr,
                    f"matching did not line the boxes up better than your placement of them "
                    f"({seed_share:.0%} within {INLIER_M * 100:.0f} cm), so that was kept", kept_seed=True)


def _say_how(match: _MatchSet, fit: _PairFit) -> str:
    """`match.matched_on` plus how the correction was chosen, when it was chosen."""
    return f"{match.matched_on}; {fit.note}" if fit.note else match.matched_on


def _summarise_matched_on(edges: list[dict]) -> str:
    """One line for a scan that several anchor pairs touched: what they matched on, keeping every fallback's reason."""
    if len({e["matched_on"] for e in edges}) == 1:
        return edges[0]["matched_on"]
    n_region = sum(1 for e in edges if e["region"])
    reasons = "; ".join(dict.fromkeys(e["matched_on"] for e in edges if not e["region"]))
    if n_region == len(edges):
        kept = sum(1 for e in edges if e["kept_seed"])
        return (f"the boxes at full resolution, in all {len(edges)} anchor pairs"
                + (f"; in {kept} of them matching did not beat your placement of the boxes, so that was kept"
                   if kept else ""))
    if n_region == 0:
        return reasons
    return f"{n_region} of {len(edges)} anchor pairs on the boxes at full resolution; the others on {reasons}"


def _still_scan(group: list[str], reference: Optional[str]) -> str:
    """The scan a free-floating group of linked scans keeps still: the reference scan if it is in the group, else the
    group's first-listed scan."""
    return reference if reference in group else group[0]


def align_from_anchors(
    payloads: dict[str, dict],       # scan name -> the bench preview payload
    poses: dict[str, ScanPose],      # scan name -> its CURRENT pose (mutated copy returned)
    anchors: list,                   # placement.Anchor objects
    locked: dict[str, bool],
    subset: Optional[set[str]],      # None means "everyone eligible", else only these may move
    scans_folder: Optional[str] = None,
    icp_distance_m: float = 0.5,
    max_tilt_deg: float = 3.0,
    min_fitness: float = 0.05,
    max_shift_m: float = 8.0,
    icp_max_iteration: int = 30,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    reference_scan: Optional[str] = None,
) -> AlignResult:
    """Match every link pair (4 DOF ICP seeded from the link's region, on full-resolution box points when
    scans_folder has them) against the current poses, gate on tilt, fitness and a shift limit, then solve all
    accepted pairs together so two links touching one scan both count. Locked scans and scans outside subset
    never move and act as references; a group with none keeps reference_scan, or else its first-listed scan, still.
    One row per scan.

    progress_cb(done, total) runs after each link; should_cancel is checked at the same point. A cancel returns
    with cancelled=True before the next link, and the caller must not write any pose."""
    start = {k: ScanPose(**vars(v)) for k, v in poses.items()}   # do not mutate caller's dict
    skipped: list[str] = []
    appeared: list[str] = []                                     # scans in the order the anchors first name them
    constraints: list[graph4.Constraint] = []
    edge_facts: list[dict] = []                                  # per accepted pair, for the rows
    failures: dict[str, AlignRow] = {}
    locked_partner: dict[str, str] = {}
    scan_points_cache: dict[str, Optional[np.ndarray]] = {}                # scan name -> its full-resolution points
    region_cache: dict[tuple[str, int, float], Optional[np.ndarray]] = {}  # (anchor, member, margin) -> raw points in its box

    def is_fixed(name: str) -> bool:
        return bool(locked.get(name, False)) or (subset is not None and name not in subset)

    def region_of(member, member_index_in_anchor, anchor_id, margin_m=0.0) -> Optional[np.ndarray]:
        key = (anchor_id, member_index_in_anchor, margin_m)
        if key not in region_cache:
            region_cache[key] = (full_res_region(scans_folder, member, scan_points_cache, margin_m)
                                 if scans_folder else None)
        return region_cache[key]

    def grown_box(member, member_index_in_anchor, anchor_id) -> _BoxPoints:
        """The member's box points, grown in `REGION_GROW_STEP_M` steps until they reach MIN_REGION_POINTS."""
        grown = 0.0
        while True:
            pts = region_of(member, member_index_in_anchor, anchor_id, grown)
            if pts is None or len(pts) >= MIN_REGION_POINTS or grown >= REGION_MARGIN_M:
                return _BoxPoints(pts, grown)
            grown = min(grown + REGION_GROW_STEP_M, REGION_MARGIN_M)

    def resolved_centroid(member, member_index_in_anchor, anchor_id):
        region = region_of(member, member_index_in_anchor, anchor_id)
        if region is not None and len(region):
            return region.mean(axis=0)                                     # the full-resolution centroid
        return np.asarray(member.centroid, dtype=float)                   # else the one stored from the preview

    cancelled = False
    n_completed = 0
    for anchor_idx, anchor in enumerate(anchors, start=1):
        if should_cancel and should_cancel():
            cancelled = True
            break
        if progress_cb:
            progress_cb(anchor_idx, len(anchors))
        n_completed = anchor_idx
        members = anchor.members
        if len(members) < 2:
            skipped.append(f"anchor {anchor.id!r} has fewer than 2 members, nothing to align")
            continue
        ref = members[0]
        if ref.scan not in payloads:
            skipped.append(f"anchor {anchor.id!r} names an unknown scan {ref.scan!r}")
            continue
        if ref.scan not in appeared:
            appeared.append(ref.scan)

        for m_idx, mem in enumerate(members[1:], start=1):
            a, b = ref.scan, mem.scan
            if b not in payloads or b == a:
                if b != a:
                    skipped.append(f"anchor {anchor.id!r} names an unknown scan {b!r}")
                continue
            if b not in appeared:
                appeared.append(b)
            for s_name, other in ((a, b), (b, a)):
                if locked.get(s_name, False):
                    locked_partner.setdefault(s_name, other)
            if is_fixed(a) and is_fixed(b):
                continue

            a_pl, b_pl = payloads[a], payloads[b]
            a_centroid_world = raw_to_world_point(
                resolved_centroid(ref, 0, anchor.id), a_pl["level_rotation"], a_pl["floor_z"], start[a])
            b_centroid_world = raw_to_world_point(
                resolved_centroid(mem, m_idx, anchor.id), b_pl["level_rotation"], b_pl["floor_z"], start[b])
            match = _points_to_match(a, a_pl, start[a], grown_box(ref, 0, anchor.id),
                                     region_of(ref, 0, anchor.id, REGION_MARGIN_M),
                                     b, b_pl, start[b], grown_box(mem, m_idx, anchor.id), mem)
            src_pts, tgt_pts, src_weights = match.src_pts, match.tgt_pts, match.src_weights

            # Sideways from the box centres; height from floors and ceilings, never from the box centres.
            seed = np.eye(4)
            seed[:2, 3] = (a_centroid_world - b_centroid_world)[:2]
            seed[2, 3] = _seed_height(match)

            fit = _fit_pair(match, seed, icp_distance_m, icp_max_iteration)
            correction, has_floors, height_note = _settle_height(match, fit.transform)

            moved_pts = (correction[:3, :3] @ src_pts.T).T + correction[:3, 3]
            # The backstop is about how far the WHOLE scan moves, whichever points did the matching.
            whole_b = world_cloud_points_as_drawn(b_pl, start[b])
            disp = np.linalg.norm((correction[:3, :3] @ whole_b.T).T + correction[:3, 3] - whole_b, axis=1)
            shift_mean = float(disp.mean()) if disp.size else 0.0
            shift_max = float(disp.max()) if disp.size else 0.0

            row = AlignRow(
                scan=b, against=a, fitness=round(float(fit.fitness), 4),
                shift_mean_m=round(shift_mean, 4), shift_max_m=round(shift_max, 4),
                discarded_tilt_deg=round(fit.discarded_tilt_deg, 3), applied=False,
                matched_on=f"{_say_how(match, fit)}; {height_note}",
            )
            if fit.discarded_tilt_deg > max_tilt_deg:
                row.reason = (
                    f"the matched geometry wanted to tip {fit.discarded_tilt_deg:.1f} "
                    f"degrees off level, past the {max_tilt_deg:.1f} degree gate. "
                    f"Level scans should give level answers; this looks like no "
                    f"real correspondence, not a slightly-wrong one. Keeping its "
                    f"current pose."
                )
            elif fit.fitness < min_fitness:
                row.reason = (
                    f"fitness {fit.fitness:.3f} is below the {min_fitness:.3f} gate "
                    + (f"(only {fit.fitness:.1%} of the box lines up within {INLIER_M * 100:.0f} cm of the other "
                       f"scan's). Keeping its current pose." if match.used_regions else
                       f"({len(fit.correspondences)} of {len(src_pts)} points found "
                       f"a match within {icp_distance_m:.2f} m). Keeping its current pose.")
                )
            elif shift_max > max_shift_m:
                row.reason = (
                    f"ICP wanted to move this scan's points up to {shift_max:.2f} m, "
                    f"past the generous {max_shift_m:.1f} m backstop, despite "
                    f"passing the tilt and fitness gates. Keeping its current pose."
                )
            log.info("align %s vs %s (anchor %s) on %s: fitness=%.3f shift mean=%.3f max=%.3f m "
                     "discarded tilt=%.2f deg %s", b, a, anchor.id, row.matched_on, fit.fitness, shift_mean,
                     shift_max, fit.discarded_tilt_deg, "accepted" if not row.reason else f"({row.reason})")
            if row.reason:
                failures.setdefault(b, row)
                continue

            # Where the matching put b, as a change to its pose variables. The correction acts in the world
            # frame, so it turns b's position about the world origin as well as moving it.
            p = start[b]
            t_now = np.array([p.x_m, p.y_m, p.z_m])
            t_new = correction[:3, :3] @ t_now + correction[:3, 3]
            yaw_delta = np.radians(yaw_of(correction))
            corr = fit.correspondences
            tgt_cloud = as_cloud(tgt_pts)
            tgt_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
            matched_b, matched_a = moved_pts[corr[:, 0]], tgt_pts[corr[:, 1]]
            normals, weights = np.asarray(tgt_cloud.normals)[corr[:, 1]], src_weights[corr[:, 0]]
            step, info = graph4.robust_edge(matched_b, matched_a, normals, weights, t_new,
                                            np.array([start[a].x_m, start[a].y_m, start[a].z_m]))
            step[3] = 0.0                       # height is already settled on the floors, or not this link's to set
            if not has_floors:
                info[[3, 7], :] = 0.0
                info[:, [3, 7]] = 0.0
            constraints.append(graph4.Constraint(b=b, a=a, target_b=np.concatenate([[yaw_delta], t_new - t_now]) + step,
                                                 info=info))
            edge_facts.append({"a": a, "b": b, "fitness": float(fit.fitness), "tilt": float(fit.discarded_tilt_deg),
                               "region": match.used_regions, "kept_seed": fit.kept_seed, "matched_on": row.matched_on})

    if cancelled:
        # Cancelled: discard everything, since the links are only meaningful solved together. start is the
        # untouched copy of the caller's poses.
        return AlignResult(poses=start, rows=[], skipped=skipped, cancelled=True, n_anchors_completed=n_completed)

    # Who may move. A group of scans joined by accepted anchors with nothing fixed in it keeps the reference scan
    # (or else its first-listed scan) still, so the whole group cannot drift together.
    in_graph = [n for n in appeared if any(n in (c.a, c.b) for c in constraints)]
    fixed_names = {n for n in in_graph if is_fixed(n)}
    at = {n: k for k, n in enumerate(in_graph)}
    for group in connected_groups(len(in_graph), [(at[c.a], at[c.b]) for c in constraints]):
        if not any(in_graph[k] in fixed_names for k in group):
            fixed_names.add(_still_scan([in_graph[k] for k in group], reference_scan))
    free = [n for n in in_graph if n not in fixed_names]
    deltas = graph4.solve_4dof(constraints, free) if constraints else {}

    poses_out = {k: ScanPose(**vars(v)) for k, v in start.items()}
    rows: dict[str, AlignRow] = {}
    for name in free:
        d = deltas[name]
        p0 = start[name]
        new = ScanPose(yaw_deg=float((p0.yaw_deg + np.degrees(d[0])) % 360.0), x_m=p0.x_m + float(d[1]),
                       y_m=p0.y_m + float(d[2]), z_m=p0.z_m + float(d[3]),
                       # the 4 DOF solve never touches pitch/roll; carry them forward
                       pitch_deg=p0.pitch_deg, roll_deg=p0.roll_deg)
        mine = [e for e in edge_facts if name in (e["a"], e["b"])]
        best = max(mine, key=lambda e: e["fitness"])
        moved = np.linalg.norm(world_cloud_points_as_drawn(payloads[name], new)
                               - world_cloud_points_as_drawn(payloads[name], p0), axis=1)
        row = AlignRow(scan=name, against=best["a"] if best["b"] == name else best["b"],
                       fitness=round(best["fitness"], 4), shift_mean_m=round(float(moved.mean()), 4),
                       shift_max_m=round(float(moved.max()), 4), discarded_tilt_deg=round(best["tilt"], 3),
                       applied=True, matched_on=_summarise_matched_on(mine))
        if row.shift_max_m > max_shift_m:
            row.applied = False
            row.reason = (f"solving every anchor together would move this scan's points up to {row.shift_max_m:.2f} m, "
                          f"past the {max_shift_m:.1f} m backstop. Keeping its current pose.")
        else:
            poses_out[name] = new
        rows[name] = row
    for name, row in failures.items():
        if name not in rows and not is_fixed(name):
            rows[name] = row
    for name, partner in locked_partner.items():
        if name not in rows:
            rows[name] = AlignRow(scan=name, against=partner, fitness=0.0, shift_mean_m=0.0, shift_max_m=0.0,
                                  discarded_tilt_deg=0.0, applied=False,
                                  reason="locked; align does not move a locked scan")
    return AlignResult(poses=poses_out, rows=[rows[n] for n in appeared if n in rows], skipped=skipped,
                       n_anchors_completed=n_completed)
