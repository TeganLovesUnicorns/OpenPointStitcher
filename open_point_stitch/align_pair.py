"""Pairwise registration: coarse candidates (FPFH + RANSAC, walls, gravity), then fine ICP.

Every attempt returns a PairResult with fitness, RMSE, correspondence count, an accepted flag and a reason.
Candidates are ranked by score.py's surface_fit; the accept/reject gates use raw fitness and RMSE."""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import threading
from dataclasses import dataclass, replace as _dc_replace
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .align_gravity import coarse_candidates
from .lib.clouds import as_cloud
from .lib.poses import tilt_of, yaw_of, yaw_transform
from .find_walls import wall_candidates
from .find_landmarks import landmark_candidates
from .prepare import Prepared, aabb, density_preserving_thin, thin_rng
from . import score as _verify
from .overlap import direct_overlap
from . import explain as _why
from .point_trust import (_MIDAIR, MIDAIR_MIN_W, THICK_FLOOR_M, midair_weight, point_weight,  # noqa: F401
                          surface_thickness)

log = logging.getLogger(__name__)

reg = o3d.pipelines.registration


@dataclass
class PairResult:
    source: int
    target: int
    source_name: str
    target_name: str
    transform: np.ndarray                  # 4x4, maps source into target frame
    information: np.ndarray                # 6x6, edge confidence for pose graph
    coarse_fitness: float = 0.0
    coarse_rmse: float = 0.0
    fitness: float = 0.0
    rmse: float = 0.0
    n_correspondences: int = 0
    accepted: bool = False
    reason: str = ""
    why: str = ""                          # `reason` in plain words, for the links table (explain.py)
    method: str = ""
    seeded_from: str = "ransac"            # ransac | gravity | manual | identity
    tilt_deg: float = 0.0                  # how far the result tips vertical
    interpenetration: float = 0.0          # volume overlap, see interpenetration()
    n_candidates: int = 0                  # coarse starting poses evaluated
    # Direct 15 cm overlap (smaller of the two directions) at the accepted transform, full resolution.
    # None when the overlap gate could not run (no full-resolution scans).
    overlap_min_frac: Optional[float] = None
    # This pair's ceiling on fitness at the seed pose, at the final ICP radius. None when not measurable.
    achievable_frac: Optional[float] = None
    fitness_over_achievable: Optional[float] = None
    # Winning candidate's detector-vote diagnostics; None when the vote did not run.
    vote_n_planes_considered: Optional[int] = None
    vote_n_corners_considered: Optional[int] = None
    vote_n_edges_considered: Optional[int] = None
    vote_plane_agreement: Optional[float] = None
    vote_corner_agreement: Optional[float] = None
    vote_edge_agreement: Optional[float] = None
    vote_n_openings_considered: Optional[int] = None
    vote_opening_agreement: Optional[float] = None
    vote_n_landmarks_considered: Optional[int] = None
    vote_landmark_agreement: Optional[float] = None
    # The detectors' combined verdict without surface_fit; None when the vote did not run. align_graph.py uses it
    # to weight this edge.
    vote_detector_score: Optional[float] = None
    # Set when the GPU screened this pair's candidates: how many the CPU then re-ran and confirmed, and how far the GPU's own
    # answer for the winner was from the CPU's (cm). None when the pair ran wholly on the CPU.
    gpu_screened: Optional[bool] = None
    gpu_verified: Optional[int] = None
    gpu_vs_cpu_cm: Optional[float] = None
    # What the user's placement scored, and what the best candidate scored against it; None outside seeded runs.
    # declined_no_improvement means no candidate beat the placement, which is a success, not a failure.
    seed_score: Optional[float] = None
    best_candidate_score: Optional[float] = None
    declined_no_improvement: bool = False
    vote_surface_fit: Optional[float] = None   # winning candidate's score BEFORE the vote
    vote_score: Optional[float] = None         # winning candidate's score AFTER the vote
    vote_rejected_count: int = 0               # candidates vetoed by the vote, this pair
    confidence: str = ""                       # score.confidence(): clear, workable, ambiguous, unchecked or none
    confidence_why: str = ""                   # the same, in plain words

    def as_row(self) -> dict:
        # NaN means no coarse stage ran; NaN and Infinity are not valid JSON, so both are refused here.
        def _nan_to_none(v: float, ndigits: int) -> Optional[float]:
            return round(v, ndigits) if np.isfinite(v) else None

        return {
            "source": self.source,
            "target": self.target,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "seeded_from": self.seeded_from,
            "method": self.method,
            "candidates": self.n_candidates,
            "tilt_deg": round(self.tilt_deg, 2),
            "interpenetration": round(self.interpenetration, 3),
            "coarse_fitness": _nan_to_none(self.coarse_fitness, 4),
            "coarse_rmse_m": _nan_to_none(self.coarse_rmse, 4),
            "fitness": _nan_to_none(self.fitness, 4),
            "rmse_m": _nan_to_none(self.rmse, 4),
            "correspondences": self.n_correspondences,
            "overlap_min_frac": (None if self.overlap_min_frac is None
                                 else round(self.overlap_min_frac, 4)),
            "achievable_frac": (None if self.achievable_frac is None
                                else round(self.achievable_frac, 4)),
            "fitness_over_achievable": (None if self.fitness_over_achievable is None
                                        else round(self.fitness_over_achievable, 4)),
            "accepted": self.accepted,
            "reason": self.reason,
            "why": self.why,
            "vote_n_planes_considered": self.vote_n_planes_considered,
            "vote_n_corners_considered": self.vote_n_corners_considered,
            "vote_n_edges_considered": self.vote_n_edges_considered,
            "vote_n_openings_considered": self.vote_n_openings_considered,
            "vote_n_landmarks_considered": self.vote_n_landmarks_considered,
            "gpu_screened": self.gpu_screened,
            "gpu_verified": self.gpu_verified,
            "gpu_vs_cpu_cm": (None if self.gpu_vs_cpu_cm is None else round(self.gpu_vs_cpu_cm, 3)),
            "seed_score": (None if self.seed_score is None else round(self.seed_score, 6)),
            "best_candidate_score": (None if self.best_candidate_score is None
                                     else round(self.best_candidate_score, 6)),
            "declined_no_improvement": self.declined_no_improvement,
            "confidence": self.confidence,
            "confidence_why": self.confidence_why,
            "vote_edge_agreement": (None if self.vote_edge_agreement is None
                                    else round(self.vote_edge_agreement, 4)),
            "vote_plane_agreement": (None if self.vote_plane_agreement is None
                                     else round(self.vote_plane_agreement, 4)),
            "vote_corner_agreement": (None if self.vote_corner_agreement is None
                                      else round(self.vote_corner_agreement, 4)),
            "vote_opening_agreement": (None if self.vote_opening_agreement is None
                                       else round(self.vote_opening_agreement, 4)),
            "vote_landmark_agreement": (None if self.vote_landmark_agreement is None
                                        else round(self.vote_landmark_agreement, 4)),
            "vote_detector_score": (None if self.vote_detector_score is None
                                    else round(self.vote_detector_score, 4)),
            "vote_surface_fit": (None if self.vote_surface_fit is None
                                 else round(self.vote_surface_fit, 6)),
            "vote_score": (None if self.vote_score is None else round(self.vote_score, 6)),
            "vote_rejected_count": self.vote_rejected_count,
            # Where this pair would have put the scan, kept even when rejected, to tell a near miss from nonsense.
            "transform": [[round(float(v), 6) for v in row] for row in self.transform],
        }


def coarse_register(src: Prepared, tgt: Prepared, cfg: PipelineConfig):
    """Global registration with no initial guess, via FPFH feature matching.

    Deliberately not re-seeded per call: Open3D's RNG is a shared stream and re-seeding here changes results."""
    dist = cfg.voxel_size * cfg.ransac_distance_mult
    result = reg.registration_ransac_based_on_feature_matching(
        src.down, tgt.down, src.fpfh, tgt.fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=reg.TransformationEstimationPointToPoint(False),
        ransac_n=cfg.ransac_n,
        checkers=[
            reg.CorrespondenceCheckerBasedOnEdgeLength(cfg.ransac_edge_length_ratio),
            reg.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=reg.RANSACConvergenceCriteria(
            cfg.ransac_max_iteration, cfg.ransac_confidence
        ),
    )
    return result


class _TensorResult:
    """Stand-in for Open3D's legacy RegistrationResult (.transformation, .fitness, .inlier_rmse), so callers
    handle one result type from either ICP path."""
    __slots__ = ("transformation", "fitness", "inlier_rmse", "correspondence_set")

    def __init__(self, T, fitness, rmse, n_corr):
        self.transformation = T
        self.fitness = fitness
        self.inlier_rmse = rmse
        # Callers only ever take len() of this. Cheaper than materialising the
        # real correspondence list off the device just to count it.
        self.correspondence_set = range(int(n_corr))


def fine_register(src: Prepared, tgt: Prepared, init: np.ndarray, cfg: PipelineConfig):
    """Local refinement of an existing alignment.

    Uses the tensor API when cfg.use_gpu is set; on any failure it falls back
    to the legacy CPU path, so a GPU problem costs speed, not results."""
    dist = cfg.voxel_size * cfg.icp_distance_mult

    if cfg.use_gpu:
        try:
            from . import align_gpu as _gpu
            device = _gpu.resolve_device(True)
            gs = _gpu.prepare_on_device(src.down, cfg, device, src.name,
                                        src.index, with_features=False)
            gt = _gpu.prepare_on_device(tgt.down, cfg, device, tgt.name,
                                        tgt.index, with_features=False)
            T, fit, rmse = _gpu.icp_on_device(gs, gt, init, cfg)
            return _TensorResult(T, fit, rmse, fit * gs.n_down)
        except Exception as exc:
            log.warning("tensor/GPU refinement failed (%s), using the CPU path",
                        exc)

    if cfg.icp_method == "point_to_point":
        estimation = reg.TransformationEstimationPointToPoint(False)
    else:
        estimation = reg.TransformationEstimationPointToPlane()

    result = reg.registration_icp(
        src.down, tgt.down, dist, init, estimation,
        reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iteration),
    )
    return result


def fine_register_ladder(src: Prepared, tgt: Prepared, init: np.ndarray, cfg: PipelineConfig):
    """Chained coarse-to-fine ICP for seeded runs: one pass per cfg.local_only_icp_distance_mults entry, each
    starting from the last. CPU only."""
    if cfg.icp_method == "point_to_point":
        estimation = reg.TransformationEstimationPointToPoint(False)
    else:
        estimation = reg.TransformationEstimationPointToPlane()
    # Residuals beyond twice this pair's typical surface thickness count for less, never tighter than a third of
    # the step's reach, so coarse steps can still close a big gap.
    thick = (2.0 * max(float(np.median(surface_thickness(tgt, cfg))), THICK_FLOOR_M / 2)
             if cfg.thickness_trust and cfg.icp_method != "point_to_point" else None)

    current = np.asarray(init, dtype=np.float64)
    result = None
    if (cfg.midair_trust or cfg.learned_trust) and cfg.icp_method != "point_to_point":   # per-point weights
        for mult in cfg.local_only_icp_distance_mults:
            dist = cfg.voxel_size * mult
            current = weighted_point_to_plane(src, tgt, current, dist, None if thick is None else max(thick, dist / 3), cfg)
        return reg.evaluate_registration(src.down, tgt.down, dist, current)
    for mult in cfg.local_only_icp_distance_mults:
        dist = cfg.voxel_size * mult
        if thick is not None:
            estimation = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=max(thick, dist / 3)))
        result = reg.registration_icp(
            src.down, tgt.down, dist, current, estimation,
            reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iteration),
        )
        current = np.asarray(result.transformation)
    return result


def colored_refine(src: Prepared, tgt: Prepared, init: np.ndarray, cfg: PipelineConfig):
    """Multi-scale colored ICP refinement on top of a geometric result.

    Only meaningful when both clouds carry colour. Returns None when they do not.
    """
    if not (src.down.has_colors() and tgt.down.has_colors()):
        return None
    current = np.array(init, copy=True)
    result = None
    for scale, iters in zip(cfg.colored_icp_scales, cfg.colored_icp_iters):
        radius = cfg.voxel_size * scale
        s = src.down.voxel_down_sample(radius)
        t = tgt.down.voxel_down_sample(radius)
        for c in (s, t):
            c.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )
        try:
            result = reg.registration_colored_icp(
                s, t, radius, current,
                reg.TransformationEstimationForColoredICP(),
                reg.ICPConvergenceCriteria(relative_fitness=1e-6,
                                           relative_rmse=1e-6,
                                           max_iteration=iters),
            )
        except RuntimeError as exc:
            log.debug("colored ICP failed at scale %.3f: %s", radius, exc)
            return None
        current = result.transformation
    return result


def full_res_refine(src_scan, tgt_scan, transform: np.ndarray,
                    cfg: PipelineConfig) -> Optional[np.ndarray]:
    """One more ICP pass on full-resolution points cropped to the overlap box.

    src_scan/tgt_scan are Scan objects, not Prepared. Returns None (never raises) on failure or too little overlap;
    the caller keeps its existing transform."""
    try:
        full_src = src_scan.to_o3d()
        full_tgt = tgt_scan.to_o3d()
    except Exception as exc:
        log.debug("full-res refine: could not load full-resolution points (%s)", exc)
        return None

    full_src.transform(transform)  # into the target's frame, so init below is identity

    src_lo, src_hi = aabb(full_src)
    tgt_lo, tgt_hi = aabb(full_tgt)
    lo = np.maximum(src_lo, tgt_lo)
    hi = np.minimum(src_hi, tgt_hi)
    if np.any(lo >= hi):
        log.debug("full-res refine: %s / %s do not overlap in this transform, skipping",
                  src_scan.name, tgt_scan.name)
        return None

    box = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    crop_src = full_src.crop(box)
    crop_tgt = full_tgt.crop(box)
    # Reuse the pipeline's own floor for "enough points to trust a fit".
    if len(crop_src.points) < cfg.min_correspondences or len(crop_tgt.points) < cfg.min_correspondences:
        log.debug("full-res refine: %s / %s overlap crop too sparse (%d/%d pts), skipping",
                  src_scan.name, tgt_scan.name, len(crop_src.points), len(crop_tgt.points))
        return None

    normal_radius = cfg.voxel_size * cfg.normal_radius_mult
    for c in (crop_src, crop_tgt):
        c.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=cfg.normal_max_nn)
        )

    dist = cfg.voxel_size * cfg.icp_distance_mult
    try:
        result = reg.registration_icp(
            crop_src, crop_tgt, dist, np.eye(4),
            reg.TransformationEstimationPointToPlane(),
            reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iteration),
        )
    except RuntimeError as exc:
        log.debug("full-res refine: ICP failed for %s / %s (%s)",
                  src_scan.name, tgt_scan.name, exc)
        return None

    # result is a small correction in the already-transformed frame, so compose it on top of the original.
    return np.asarray(result.transformation) @ transform


def interpenetration(src: Prepared, tgt: Prepared, transform: np.ndarray,
                     cell: float = 0.25) -> float:
    """Fraction of the source cloud that lands inside the target's occupied volume, 0..1.

    Read it alongside fitness: scans of the same room legitimately score high."""
    s_pts = np.asarray(src.down.points)
    t_pts = np.asarray(tgt.down.points)
    if s_pts.size == 0 or t_pts.size == 0:
        return 0.0
    moved = (transform[:3, :3] @ s_pts.T).T + transform[:3, 3]
    s_keys = np.floor(moved / cell).astype(np.int64)
    t_keys = np.floor(t_pts / cell).astype(np.int64)
    t_set = set(map(tuple, t_keys))
    if not t_set:
        return 0.0
    hits = sum(1 for k in map(tuple, s_keys) if k in t_set)
    return float(hits) / float(s_keys.shape[0])


def _weights(p: Prepared, cfg: PipelineConfig) -> np.ndarray:
    w = point_weight(p, cfg)
    return np.ones(len(p.down.points)) if w is None else w


def weighted_point_to_plane(src: Prepared, tgt: Prepared, init: np.ndarray, dist: float,
                            tukey_k: Optional[float], cfg: PipelineConfig) -> np.ndarray:
    """One point-to-plane ICP step, each pairing weighted by Tukey (if tukey_k) times both points' trust.
    Stops like Open3D's: max iterations, or fitness and RMSE change under 1e-6."""
    sp, tp = np.asarray(src.down.points), np.asarray(tgt.down.points)
    tn = np.asarray(tgt.down.normals)
    ws, wt = _weights(src, cfg), _weights(tgt, cfg)
    on_gpu = cfg.trust_search_on_gpu and o3d.core.cuda.is_available() and not card_full_now()
    dev = o3d.core.Device("CUDA:0" if on_gpu else "CPU:0")
    dtype = o3d.core.float32 if dev.get_type() == o3d.core.Device.DeviceType.CUDA else o3d.core.float64
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(tp, dtype=dtype, device=dev))   # once per rung
    nns.hybrid_index(dist)                                             # radius-limited, like Open3D's own ICP
    current = np.asarray(init, dtype=np.float64)
    prev = None
    for _ in range(cfg.icp_max_iteration):
        moved = sp @ current[:3, :3].T + current[:3, 3]
        idx, d2, _cnt = nns.hybrid_search(o3d.core.Tensor(moved, dtype=dtype, device=dev), dist, 1)
        idx, d2 = idx.cpu().numpy().reshape(-1), d2.cpu().numpy().reshape(-1).astype(np.float64)
        keep = np.flatnonzero(idx >= 0)
        if len(keep) < 6:
            break
        fit, rmse = len(keep) / len(sp), float(np.sqrt(d2[keep].mean()))
        if prev is not None and abs(fit - prev[0]) < 1e-6 and abs(rmse - prev[1]) < 1e-6:
            break
        prev = (fit, rmse)
        corr = np.column_stack([keep, idx[keep]])
        p = moved[keep]
        q, n = tp[corr[:, 1]], tn[corr[:, 1]]
        r = np.einsum("ij,ij->i", p - q, n)
        w = ws[corr[:, 0]] * wt[corr[:, 1]]
        if tukey_k is not None:
            w = w * np.where(np.abs(r) < tukey_k, (1 - (r / tukey_k) ** 2) ** 2, 0.0)
        J = np.hstack([np.cross(p, n), n])
        A = (J * w[:, None]).T @ J
        b = -(J * w[:, None]).T @ r
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        step = np.eye(4)
        step[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(x[:3])
        step[:3, 3] = x[3:]
        current = step @ current
    return current


def _weighted_information(src: Prepared, tgt: Prepared, dist: float, transform: np.ndarray,
                          cfg: PipelineConfig) -> np.ndarray:
    """Open3D's information matrix (sum of G^T G over pairings at the target points), each pairing weighted."""
    corr = np.asarray(reg.evaluate_registration(src.down, tgt.down, dist, transform).correspondence_set)
    info = np.zeros((6, 6))
    if not len(corr):
        return info
    q = np.asarray(tgt.down.points)[corr[:, 1]]
    w = _weights(src, cfg)[corr[:, 0]] * _weights(tgt, cfg)[corr[:, 1]]
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    zero, one = np.zeros_like(x), np.ones_like(x)
    G = np.stack([np.stack([zero, z, -y, one, zero, zero], 1),
                  np.stack([-z, zero, x, zero, one, zero], 1),
                  np.stack([y, -x, zero, zero, zero, one], 1)], 1)         # (n, 3, 6)
    return np.einsum("n,nki,nkj->ij", w, G, G)


def _matched_thickness(src: Prepared, tgt: Prepared, transform: np.ndarray, dist: float, cfg: PipelineConfig) -> float:
    """Median thickness of the target surfaces this pair actually matches at `transform`."""
    corr = np.asarray(reg.evaluate_registration(src.down, tgt.down, dist, transform).correspondence_set)
    if not len(corr):
        return THICK_FLOOR_M
    return float(np.median(surface_thickness(tgt, cfg)[corr[:, 1]]))


def _information_matrix(src: Prepared, tgt: Prepared, transform: np.ndarray,
                        cfg: PipelineConfig) -> np.ndarray:
    dist = cfg.voxel_size * cfg.posegraph_max_correspondence_distance_mult
    if cfg.midair_trust or cfg.learned_trust:   # each pairing counts by both points' trust
        info = _weighted_information(src, tgt, dist, transform, cfg)
    else:
        info = reg.get_information_matrix_from_point_clouds(
            src.down, tgt.down, dist, transform
        )
    if cfg.thickness_trust:                 # a pair matched on thick surfaces pins its scans less
        t = _matched_thickness(src, tgt, transform, dist, cfg)
        info = info * (THICK_FLOOR_M / max(t, THICK_FLOOR_M)) ** 2
    return info


def achievable_overlap_frac(src_scan, tgt_scan, transform: np.ndarray,
                             radius_m: float, cfg: PipelineConfig) -> float:
    """Ceiling on fitness for this pair at transform: the fraction of one scan's surface within radius_m of the
    other, full resolution. Thinned the same way as pair_filter.surface_pairs so the two stay comparable."""
    def _thin(scan):
        pcd = as_cloud(scan.points)
        keep_idx, _ = density_preserving_thin(
            pcd, cfg.adjacency_surface_thin_points, thin_rng(scan.name),
            bin_m=cfg.voxel_size, floor_n=cfg.search_thin_floor_n,
        )
        return scan.points[keep_idx.astype(np.int64)]

    src_pts, tgt_pts = _thin(src_scan), _thin(tgt_scan)
    key = f"within_{int(round(radius_m * 100))}cm_frac"
    fwd = direct_overlap(src_pts, transform, tgt_pts, thresholds=(radius_m,))
    bwd = direct_overlap(tgt_pts, np.linalg.inv(transform), src_pts, thresholds=(radius_m,))
    return max(fwd[key], bwd[key])


def _reject_reason_from_gates(
    res: "PairResult", fitness_reject_reason: Optional[str],
    fitness: float, rmse: float, ncorr: int, transform: np.ndarray,
    src_scan, tgt_scan, cfg: PipelineConfig, fitness_why: str = "",
) -> Optional[str]:
    """Reject reason, checked in order: fitness, RMSE, correspondences, overlap gate; None if accepted. On a reject,
    res.why holds the failed check in plain words. Shared with align_gpu.py so both paths use the same
    criteria. The overlap gate is last because it is the most expensive."""
    if fitness_reject_reason is not None:
        res.why = fitness_why
        return fitness_reject_reason
    if rmse > cfg.max_rmse:
        res.why = _why.too_loose(rmse, cfg.max_rmse)
        return f"rmse {rmse:.3f} m > max {cfg.max_rmse} m"
    if ncorr < cfg.min_correspondences:
        res.why = _why.too_few_matched(ncorr, cfg.min_correspondences)
        return f"only {ncorr} correspondences < min {cfg.min_correspondences}"
    if cfg.overlap_gate_enabled and src_scan is not None and tgt_scan is not None:
        fwd = direct_overlap(src_scan.points, transform, tgt_scan.points,
                             thresholds=(cfg.overlap_gate_threshold_m,))
        bwd = direct_overlap(tgt_scan.points, np.linalg.inv(transform), src_scan.points,
                             thresholds=(cfg.overlap_gate_threshold_m,))
        key = f"within_{int(cfg.overlap_gate_threshold_m * 100)}cm_frac"
        res.overlap_min_frac = min(fwd[key], bwd[key])
        if res.overlap_min_frac < cfg.overlap_gate_min_frac:
            res.why = _why.too_little_overlap(res.overlap_min_frac, cfg.overlap_gate_min_frac,
                                              cfg.overlap_gate_threshold_m)
            return (
                f"direct overlap {res.overlap_min_frac:.3f} < min "
                f"{cfg.overlap_gate_min_frac} at {cfg.overlap_gate_threshold_m*100:.0f}cm "
                f"(fitness {fitness:.3f} passed on its own)"
            )
    return None


def _score(src: Prepared, tgt: Prepared, init: np.ndarray, cfg: PipelineConfig,
           local_only: bool = False):
    """Refine one starting pose and return (transform, fitness, rmse, ncorr). Seeded runs use the coarse-to-fine
    ladder instead of a single fixed window."""
    fine = (fine_register_ladder(src, tgt, init, cfg) if local_only
            else fine_register(src, tgt, init, cfg))
    return (np.asarray(fine.transformation),
            float(fine.fitness),
            float(fine.inlier_rmse),
            int(np.asarray(fine.correspondence_set).shape[0]))


def _apply_vote(verdict, transform, src_features, tgt_features, cfg,
                 progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """Run all detectors at transform and fold them into verdict. The candidate loop and the placement baseline
    must be scored identically, so both use this. progress_cb, if given, receives each detector's name."""
    if progress_cb:
        progress_cb("matching planes")
    plane_agreement, colour_agreement, n_planes = _verify.plane_colour_agreement(
        transform, src_features, tgt_features, cfg)
    if progress_cb:
        progress_cb("matching corners")
    corner_agree, corner_colour, n_corners = _verify.corner_agreement(
        transform, src_features, tgt_features, cfg)
    if progress_cb:
        progress_cb("matching edges")
    edge_agree, edge_colour, n_edges = _verify.edge_agreement(
        transform, src_features, tgt_features, cfg)
    if progress_cb:
        progress_cb("matching doors and windows")
    opening_agree, n_openings = _verify.opening_agreement(
        transform, src_features, tgt_features, cfg)
    if progress_cb:
        progress_cb("matching landmarks")
    landmark_agree, n_landmarks = _verify.landmark_agreement(
        transform, src_features, tgt_features, cfg)
    _verify.apply_detector_vote(
        verdict, plane_agreement, colour_agreement, n_planes, cfg,
        corner_agreement=corner_agree, corner_colour_agreement=corner_colour,
        n_corners=n_corners, edge_agreement=edge_agree,
        edge_colour_agreement=edge_colour, n_edges=n_edges,
        opening_agreement=opening_agree, n_openings=n_openings,
        landmark_agreement=landmark_agree, n_landmarks=n_landmarks)


_DUMP_LOCK = threading.Lock()


def _dump_candidate(src: Prepared, tgt: Prepared, origin: str, status: str, verdict,
                    extra: Optional[dict] = None) -> None:
    """Diagnostics: with OPENPOINT_DUMP_CANDIDATES set, append this candidate and every clue it was scored on as
    one JSON line."""
    path = os.environ.get("OPENPOINT_DUMP_CANDIDATES")
    if not path:
        return
    row = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
           for k, v in vars(verdict).items() if k not in ("transform",)}
    row.update(extra or {})
    row.update({"src": src.name, "tgt": tgt.name, "origin": origin, "status": status,
                "run": os.environ.get("OPENPOINT_DUMP_RUN", ""), "T": np.asarray(verdict.transform).tolist(),
                "src_centre": np.asarray(src.down.points).mean(axis=0).tolist()})
    with _DUMP_LOCK, open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")


def _judge_p(verdict, facts: dict) -> float:
    """The seed judge's probability that this candidate is right, from its verdict and its seed facts."""
    return float(_verify.predict(_verify.load_model(), _verify.clues({**vars(verdict), **facts}))[0])


def _pose_delta(seed: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Translation (m) and yaw (deg) of other relative to seed, for the seeded-run bound. Yaw is read from the
    rotation's plane."""
    rel = other @ np.linalg.inv(seed)
    trans_m = float(np.linalg.norm(rel[:3, 3]))
    return trans_m, abs(yaw_of(rel))


def _proposal_for(src_name: str, tgt_name: str, cfg: PipelineConfig) -> Optional[np.ndarray]:
    """An outside proposal for this pair (raw source to raw target), in either direction."""
    if not cfg.proposals:
        return None
    t = cfg.proposals.get(f"{src_name}|{tgt_name}")
    if t is not None:
        return np.asarray(t, dtype=np.float64)
    t = cfg.proposals.get(f"{tgt_name}|{src_name}")
    return None if t is None else np.linalg.inv(np.asarray(t, dtype=np.float64))


def _within_proposal_gate(seed: np.ndarray, other: np.ndarray, cfg: PipelineConfig) -> bool:
    d_m, d_deg = _pose_delta(seed, other)
    return d_m <= cfg.proposal_gate_m and d_deg <= cfg.proposal_gate_deg


def _local_candidates(seed: np.ndarray, cfg: PipelineConfig) -> list[np.ndarray]:
    """Sample poses near a hand-placed seed with a local numpy generator, never Open3D's RNG."""
    rng = np.random.default_rng(cfg.random_seed)
    seed = np.asarray(seed, dtype=np.float64)
    out = []
    for _ in range(cfg.local_search_n):
        dx, dy = rng.uniform(-cfg.local_search_radius_m, cfg.local_search_radius_m, size=2)
        dz = rng.uniform(-cfg.local_search_height_m, cfg.local_search_height_m)
        dyaw = rng.uniform(-cfg.local_search_yaw_deg, cfg.local_search_yaw_deg)
        perturb = yaw_transform(dyaw, np.array([dx, dy, dz]))
        out.append(perturb @ seed)
    return out


@functools.lru_cache(maxsize=1)
def _gpu_ready() -> bool:
    """Whether a CUDA card can be used here, asked once per process."""
    from .limits import gpu_status
    return bool(gpu_status()["gpu_usable"])


# What the card-full check saw this run; cleared by reset_gpu_stop() at the start of each run.
_gpu_busy = {"reason": "", "pairs": 0, "now": False, "used": False}
_gpu_busy_lock = threading.Lock()


def reset_gpu_stop() -> None:
    """Called at the start of each run, so one run's full card is not reported by the next."""
    _gpu_busy.update(reason="", pairs=0, now=False, used=False)


def gpu_stop_reason() -> str:
    """Why some of this run's work moved off the GPU, or "" if none did."""
    return _gpu_busy["reason"]


def gpu_moved_pairs() -> int:
    """How many of this run's pairs ran on the CPU because the card was full."""
    return _gpu_busy["pairs"]


def gpu_busy_now() -> bool:
    """Whether the latest check found the card full."""
    return _gpu_busy["now"]


def card_full_now(pair: bool = False) -> bool:
    """Whether the card is too full to share right now. Checked per pair, so work returns to the GPU once the
    other program lets go; before this run has used the card it must have room for the whole run."""
    from .limits import GPU_FULL_FREE_GB, GPU_RUN_NEEDS_GB, card_too_full
    reason = card_too_full(GPU_FULL_FREE_GB if _gpu_busy["used"] else GPU_RUN_NEEDS_GB)
    with _gpu_busy_lock:
        _gpu_busy["now"] = bool(reason)
        if reason:
            if not _gpu_busy["reason"]:
                log.warning("%s; pairs use the CPU until it frees up", reason)
            _gpu_busy["reason"] = _gpu_busy["reason"] or reason
            _gpu_busy["pairs"] += int(pair)
    return bool(reason)


def _gpu_screen(src: Prepared, tgt: Prepared, candidates: list, cfg: PipelineConfig):
    """Every candidate's ICP ladder on the GPU at once, as (4x4, fitness, rmse, matches) each, or None if the GPU
    cannot be used (the caller then does everything on the CPU)."""
    # A card shared beyond its size makes every program on it crawl.
    if not _gpu_ready() or card_full_now(pair=True):
        return None
    try:
        from .align_gpu import ladder_batch
        got = ladder_batch(src, tgt, [start for _origin, start in candidates], cfg, max_iteration=cfg.gpu_screen_iterations)
        _gpu_busy["used"] = True
        return got
    except Exception as exc:  # noqa: BLE001 - a GPU problem costs speed, never the result
        log.warning("%s -> %s: GPU screen failed (%s), using the CPU for this pair", src.name, tgt.name, exc)
        return None


def pair_search_voxel(src: Prepared, tgt: Prepared, cfg: PipelineConfig) -> Optional[float]:
    """The voxel size for this pair's candidate search: cfg.search_voxel_size, or with adaptive_search_voxel on,
    never finer than adaptive_voxel_spacings point spacings of the sparser scan (rounded up to a centimetre)."""
    configured = cfg.search_voxel_size
    if not cfg.adaptive_search_voxel:
        return configured
    wanted = math.ceil(max(src.spacing_m, tgt.spacing_m) * cfg.adaptive_voxel_spacings * 100 - 1e-9) / 100
    return max(wanted, configured or 0.0) if wanted > cfg.voxel_size else configured


def _search_prepared(base: Prepared, cfg: PipelineConfig) -> Prepared:
    """A coarser copy of a prepared scan, used only to find starting poses. Thinned with density_preserving_thin
    (real points, never voxel averages); fine ICP always uses base.down."""
    target_n = len(base.down.voxel_down_sample(cfg.search_voxel_size).points)
    keep_idx, _thin_frac = density_preserving_thin(
        base.down, target_n, thin_rng(base.name),
        bin_m=cfg.search_voxel_size, floor_n=cfg.search_thin_floor_n,
    )
    down = base.down.select_by_index(keep_idx.tolist())
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.search_voxel_size * cfg.normal_radius_mult,
            max_nn=cfg.normal_max_nn,
        )
    )
    down.orient_normals_consistent_tangent_plane(k=10)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.search_voxel_size * cfg.fpfh_radius_mult,
            max_nn=cfg.fpfh_max_nn,
        ),
    )
    return Prepared(name=base.name, index=base.index, down=down, fpfh=fpfh, n_full=base.n_full)


def register_pair(
    src: Prepared,
    tgt: Prepared,
    cfg: PipelineConfig,
    init: Optional[np.ndarray] = None,
    seeded_from: str = "ransac",
    src_scan=None,
    tgt_scan=None,
    local_only: bool = False,
    src_features=None,
    tgt_features=None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> PairResult:
    """Register one pair: gather starting poses, refine each with ICP, keep the best.

    Candidates tilted past cfg.max_tilt_deg are dropped. src_scan/tgt_scan are needed for cfg.full_res_refine,
    *_features for cfg.detector_vote_enabled, and local_only requires init. progress_cb, if given, receives a short
    name for each stage as it starts."""
    if local_only and init is None:
        raise ValueError("register_pair: local_only=True requires init (a seed pose)")

    def report(stage: str) -> None:
        if progress_cb:
            progress_cb(stage)

    res = PairResult(
        source=src.index, target=tgt.index,
        source_name=src.name, target_name=tgt.name,
        transform=np.eye(4), information=np.eye(6),
        seeded_from=seeded_from,
    )

    # Pairs excluded by configuration are rejected before any expensive work.
    if frozenset((src.name, tgt.name)) in cfg.excluded_pairs:
        res.reason = "manually excluded (cfg.excluded_pairs)"
        res.why = _why.EXCLUDED_PAIR
        return res

    # --- collect coarse starting poses ---------------------------------
    candidates: list[tuple[str, np.ndarray]] = []

    res.coarse_fitness = float("nan")
    res.coarse_rmse = float("nan")

    if init is not None:
        # The seed and local samples around it are always tried; the normal candidate search still runs.
        init = np.asarray(init, dtype=np.float64)
        candidates.append(("manual", init))
        candidates.extend(("manual+local", t) for t in _local_candidates(init, cfg))
        prop = _proposal_for(src.name, tgt.name, cfg)
        if prop is not None and _within_proposal_gate(init, prop, cfg):
            candidates.append(("proposer", prop))

    # "Run from my alignment" is bounded local refinement only: no candidates proposed from scratch.
    if not local_only:
        # A coarser copy for candidate generation only; fine ICP below uses src/tgt.
        search_voxel = pair_search_voxel(src, tgt, cfg)
        if search_voxel and search_voxel > cfg.voxel_size:
            pair_cfg = _dc_replace(cfg, search_voxel_size=search_voxel)
            search_src = _search_prepared(src, pair_cfg)
            search_tgt = _search_prepared(tgt, pair_cfg)
            search_cfg = _dc_replace(cfg, voxel_size=search_voxel)
        else:
            search_src, search_tgt, search_cfg = src, tgt, cfg

        if cfg.coarse_method in ("ransac", "both"):
            report("RANSAC")
            coarse = coarse_register(search_src, search_tgt, search_cfg)
            res.coarse_fitness = float(coarse.fitness)
            res.coarse_rmse = float(coarse.inlier_rmse)
            if res.coarse_fitness > 0.0:
                candidates.append(("ransac", np.asarray(coarse.transformation)))

        if cfg.coarse_method in ("walls", "both") and cfg.assume_level:
            report("wall matching")
            try:
                wall_cands, _, _ = wall_candidates(search_src.down, search_tgt.down,
                                                   tightness_weighted=cfg.wall_tightness_weighting)
                candidates.extend(("walls", t) for t in wall_cands)
            except Exception as exc:
                log.warning("%s -> %s: wall matching failed (%s)",
                            src.name, tgt.name, exc)

        if cfg.coarse_method in ("gravity", "both") and cfg.assume_level:
            report("gravity matching")
            try:
                grav, src_level, tgt_level = coarse_candidates(
                    search_src.down, search_tgt.down,
                    res=cfg.gravity_grid_res,
                    yaw_step_deg=cfg.gravity_yaw_step_deg,
                    top_k=cfg.gravity_top_k,
                )
                if not (src_level.reliable and tgt_level.reliable):
                    log.warning(
                        "%s -> %s: could not level both clouds reliably, so the "
                        "gravity candidates are unreliable too",
                        src.name, tgt.name,
                    )
                candidates.extend(("gravity", t) for t in grav)
            except Exception as exc:
                log.warning("%s -> %s: gravity search failed (%s)",
                            src.name, tgt.name, exc)

        if cfg.use_landmarks and cfg.assume_level:
            # Off by default: landmarks alone rarely find the right placement.
            report("landmark matching")
            try:
                lm_cands = landmark_candidates(search_src.down, search_tgt.down, top_k=cfg.landmark_top_k)[0]
                candidates.extend(("landmarks", t) for t in lm_cands)
            except Exception as exc:
                log.warning("%s -> %s: landmark matching failed (%s)", src.name, tgt.name, exc)

    res.n_candidates = len(candidates)
    if not candidates:
        res.reason = "no coarse starting pose could be found"
        res.why = _why.NO_START
        return res

    # Refine and score every candidate. Ranked by surface_fit, not raw fitness; the target index is built once.
    target_nns = _verify.build_target_index(tgt.down)
    use_vote = bool(cfg.detector_vote_enabled and src_features is not None
                     and tgt_features is not None)
    best = None
    best_verdict = None  # the winning candidate's Verdict, for diagnostics
    best_vote_score = -1.0
    rejected_for_tilt = 0
    rejected_for_bound = 0
    rejected_for_vote = 0
    # The unperturbed seed's own result: the fallback in seeded runs if nothing beats the placement. Computed even
    # if the seed fails the tilt gate, since keeping it is not a claim that it is a new answer.
    seed_outcome = None
    # What the user's placement scores before refinement: the number every candidate has to beat.
    seed_baseline = None
    judge_w = ({"src_w": _weights(src, cfg), "tgt_w": _weights(tgt, cfg)}
               if cfg.learned_trust else {})
    if local_only and init is not None:
        seed_baseline = _verify.score_candidate_fast(
            src.down, tgt.down, target_nns, init, cfg, origin="manual", **judge_w)
        if use_vote:
            _apply_vote(seed_baseline, init, src_features, tgt_features, cfg, progress_cb=progress_cb)
        if cfg.seed_judge:
            seed_baseline.vote_score = _judge_p(seed_baseline, {"move_m": 0.0, "move_deg": 0.0, "fit_gain": 0.0})

    def _seed_facts(origin, transform, verdict) -> dict:
        """What a candidate looks like from a rough start: how far it moves the placement, whether BUFFER-X
        proposed it and how sure it was, and how much better than the placement it fits."""
        if init is None:
            return {}
        d_m, d_deg = _pose_delta(init, transform)
        agree = cfg.proposal_agree.get(f"{src.name}|{tgt.name}", cfg.proposal_agree.get(f"{tgt.name}|{src.name}", 0))
        return {"move_m": d_m, "move_deg": d_deg, "is_proposer": float(origin == "proposer"),
                "proposer_agree": float(agree if origin == "proposer" else 0),
                "fit_gain": verdict.surface_fit - (seed_baseline.surface_fit if seed_baseline is not None else 0.0)}

    def consider(origin, refined):
        """The gates and the score for one refined candidate: ("ok", entry, verdict), or the gate that refused it."""
        transform, fitness, rmse, ncorr = refined
        tilt = tilt_of(transform)
        if cfg.assume_level and tilt > cfg.max_tilt_deg:
            return "tilt", None, None
        if local_only:
            d_m, d_deg = _pose_delta(init, transform)
            if origin == "proposer":
                if not _within_proposal_gate(init, transform, cfg):
                    return "bound", None, None
            elif d_m > cfg.candidate_bound_m or d_deg > cfg.candidate_bound_deg:
                return "bound", None, None
        verdict = _verify.score_candidate_fast(src.down, tgt.down, target_nns, transform, cfg, origin=origin,
                                               **judge_w)
        seed_facts = _seed_facts(origin, transform, verdict)
        if use_vote:
            _apply_vote(verdict, transform, src_features, tgt_features, cfg, progress_cb=progress_cb)
            if verdict.vote_score < 0:
                # Vetoed: excluded from `best` entirely, so "every candidate
                # vetoed" takes the same "nothing survived" path as tilt/bound.
                _dump_candidate(src, tgt, origin, "vote", verdict, seed_facts)
                return "vote", None, verdict
        _dump_candidate(src, tgt, origin, "ok", verdict, seed_facts)
        if cfg.seed_judge and local_only:
            verdict.vote_score = _judge_p(verdict, seed_facts)
        return "ok", (origin, transform, fitness, rmse, ncorr, tilt), verdict

    refusals = {"tilt": 0, "bound": 0, "vote": 0}
    scored = []                                         # every surviving candidate's verdict, for score.confidence()
    use_gpu_screen = local_only and cfg.use_gpu_ladder
    if use_gpu_screen:
        report(f"GPU screening {len(candidates)} candidate(s)")
    screen = _gpu_screen(src, tgt, candidates, cfg) if use_gpu_screen else None
    if screen is None:
        report(f"refining {len(candidates)} candidate(s)")
        for origin, start in candidates:
            refined = _score(src, tgt, start, cfg, local_only=local_only)
            if local_only and origin == "manual":
                seed_outcome = (*refined, tilt_of(refined[0]))
            status, entry, verdict = consider(origin, refined)
            if status != "ok":
                refusals[status] += 1
                continue
            scored.append(verdict)
            if best is None or verdict.vote_score > best_vote_score:
                best, best_verdict, best_vote_score = entry, verdict, verdict.vote_score
    else:
        # The GPU refined every candidate at once; the CPU repeats the ladder for the best few from their original
        # starting poses, so the answer is always the CPU's.
        report(f"scoring {len(candidates)} GPU-screened candidate(s)")
        ranked = []
        for k, (origin, _start) in enumerate(candidates):
            status, _entry, verdict = consider(origin, screen[k])
            if status != "ok":
                refusals[status] += 1
                continue
            scored.append(verdict)
            ranked.append((-verdict.vote_score, k))
        ranked.sort()
        res.gpu_screened, res.gpu_verified, best_k, ladders = True, 0, None, 0
        report(f"verifying the top {cfg.gpu_verify_top} on the CPU")
        for _neg, k in ranked:
            if res.gpu_verified >= cfg.gpu_verify_top or ladders >= 2 * cfg.gpu_verify_top:
                break
            origin, start = candidates[k]
            refined = _score(src, tgt, start, cfg, local_only=local_only)
            ladders += 1
            status, entry, verdict = consider(origin, refined)
            if status != "ok":
                continue                                    # the CPU disagrees with the screen: try the next one
            res.gpu_verified += 1
            if best is None or verdict.vote_score > best_vote_score:
                best, best_verdict, best_vote_score, best_k = entry, verdict, verdict.vote_score, k
                res.gpu_vs_cpu_cm = float(np.linalg.norm(np.asarray(screen[k][0])[:3, 3] - refined[0][:3, 3])) * 100.0
    rejected_for_tilt, rejected_for_bound, rejected_for_vote = refusals["tilt"], refusals["bound"], refusals["vote"]
    res.confidence, res.confidence_why = _verify.confidence(sorted(scored, key=lambda v: -v.vote_score))

    if rejected_for_vote:
        log.info("%s -> %s: %d candidate(s) discarded, a matched surface or "
                  "corner disagreed geometrically",
                  src.name, tgt.name, rejected_for_vote)

    if rejected_for_bound:
        log.info("%s -> %s: %d local candidate(s) discarded, further than "
                  "%.2fm/%.1fdeg from the placement",
                  src.name, tgt.name, rejected_for_bound,
                  cfg.candidate_bound_m, cfg.candidate_bound_deg)

    if best is None:
        if local_only and (seed_baseline is not None or seed_outcome is not None):
            # No eligible candidate: leave the placement exactly at init.
            if seed_baseline is not None:
                transform = init
                fitness, rmse = seed_baseline.fitness, seed_baseline.rmse
                ncorr, tilt = seed_baseline.n_matched, tilt_of(init)
            else:
                transform, fitness, rmse, ncorr, tilt = seed_outcome
            res.transform, res.fitness, res.rmse = transform, fitness, rmse
            res.n_correspondences, res.tilt_deg = ncorr, tilt
            res.interpenetration = interpenetration(src, tgt, transform)
            res.information = _information_matrix(src, tgt, transform, cfg)
            res.method, res.seeded_from = "kept_placement", "kept_placement"
            res.accepted = True
            res.reason = "kept your placement - no local candidate was eligible to replace it"
            res.why = _why.KEPT_NOTHING_ELIGIBLE
            return res
        # Say which gate rejected every candidate.
        if rejected_for_vote and not rejected_for_tilt:
            res.reason = (
                f"all {len(candidates)} coarse candidate(s) had a matched surface "
                f"or corner that disagreed geometrically (see the log for "
                f"which pairs and by how much)."
            )
            res.why = _why.all_disagreed(len(candidates))
        else:
            res.reason = (
                f"all {len(candidates)} coarse candidate(s) came back tilted past "
                f"{cfg.max_tilt_deg} deg, which contradicts the levelled-scan "
                f"assumption. Manual pre-alignment is the fix for this pair."
            )
            res.why = _why.all_tilted(len(candidates), cfg.max_tilt_deg)
        return res

    # A scan only moves if the best candidate beats the user's placement by more than the detectors' noise;
    # otherwise init is returned untouched. Small moves without that margin tend to make things worse.
    if local_only and seed_baseline is not None and best is not None:
        # Judge the move on the detectors, not vote_score: vote_score includes surface_fit, which ICP improves almost
        # by definition.
        evidence = (seed_baseline.n_planes_considered + seed_baseline.n_corners_considered
                    + seed_baseline.n_edges_considered
                    + best_verdict.n_planes_considered + best_verdict.n_corners_considered
                    + best_verdict.n_edges_considered)
        if cfg.seed_judge:                   # the trained judge's probabilities decide
            seed_value, best_value = seed_baseline.vote_score, best_vote_score
        elif evidence > 0:
            seed_value, best_value = seed_baseline.detector_score, best_verdict.detector_score
        else:
            # No detector could see anything at either pose. With the vote on, that is no reason to move on fitness
            # alone; with the vote off there is nothing else to judge by.
            seed_value, best_value = seed_baseline.vote_score, best_vote_score
            if use_vote:
                best_value = seed_value - 1.0
        required = 0.0 if cfg.seed_judge else cfg.min_improvement_frac * max(seed_value, 0.0)
        res.seed_score = seed_value
        res.best_candidate_score = best_value
        # `<=`, not `<`: when both score 0.0 the required margin is also 0.0,
        # and a strict comparison let "no improvement whatsoever" through as
        # a reason to move. That is how a pair travelled 2.53m on nothing.
        if best_value <= seed_value + required or (cfg.seed_judge and best_value < 0.5):
            res.declined_no_improvement = True
            res.transform = init
            res.fitness, res.rmse = seed_baseline.fitness, seed_baseline.rmse
            res.n_correspondences, res.tilt_deg = seed_baseline.n_matched, tilt_of(init)
            res.interpenetration = interpenetration(src, tgt, init)
            res.information = _information_matrix(src, tgt, init, cfg)
            res.method, res.seeded_from = "kept_placement", "kept_placement"
            res.accepted = True
            res.reason = (
                "kept your placement - no wall, corner or edge could be matched "
                "between these two scans at all, so there is no evidence to move it on"
                if evidence == 0 and use_vote else
                f"kept your placement - the best candidate scored "
                f"{best_value:.4f} against your own {seed_value:.4f}, not the "
                f"{cfg.min_improvement_frac:.0%} better it needs to be worth moving"
            )
            res.why = (_why.KEPT_NO_EVIDENCE if evidence == 0 and use_vote else
                       _why.kept_no_better(best_value, seed_value, cfg.min_improvement_frac))
            log.info("%s -> %s: declined, nothing beat the placement (%.4f vs %.4f)",
                     src.name, tgt.name, best_value, seed_value)
            return res
        res.seed_score = seed_baseline.vote_score
        res.best_candidate_score = best_vote_score

    origin, transform, fitness, rmse, ncorr, tilt = best
    if local_only and origin == "manual":
        # The unperturbed seed won on its merits: report it as a kept placement, not a rejection.
        origin = "kept_placement"
    method = cfg.icp_method

    def surface_fit_at(t: np.ndarray) -> float:
        # A refine stage keeps its change only on surface fit: raw fitness rewards one cloud nested inside the other.
        return _verify.score_candidate_fast(src.down, tgt.down, target_nns, t, cfg, **judge_w).surface_fit

    if cfg.use_colored_icp:
        report("colour refine")
        colored = colored_refine(src, tgt, transform, cfg)
        if colored is not None:
            # Re-score at the SAME threshold the geometric run used, so the two
            # are actually comparable.
            ct, cf, cr, cn = _score(src, tgt, np.asarray(colored.transformation), cfg, local_only=local_only)
            ctilt = tilt_of(ct)
            improved = (surface_fit_at(ct) >= surface_fit_at(transform)
                        and not (cfg.assume_level and ctilt > cfg.max_tilt_deg))
            if improved:
                transform, fitness, rmse, ncorr, tilt = ct, cf, cr, cn, ctilt
                method = "colored+" + cfg.icp_method
            else:
                log.info("%s -> %s: colored ICP did not improve, keeping geometric",
                         src.name, tgt.name)

    if cfg.full_res_refine:
        if src_scan is None or tgt_scan is None:
            log.warning("%s -> %s: full_res_refine is on but no full-resolution "
                        "scans were supplied, skipping the stage",
                        src.name, tgt.name)
        else:
            report("full-resolution refine")
            refined = full_res_refine(src_scan, tgt_scan, transform, cfg)
            if refined is not None:
                # Re-score at the same voxel-level threshold as the rest of this function, so "improved" means one thing.
                rt, rf, rr, rn = _score(src, tgt, refined, cfg, local_only=local_only)
                rtilt = tilt_of(rt)
                improved = (surface_fit_at(rt) >= surface_fit_at(transform)
                            and not (cfg.assume_level and rtilt > cfg.max_tilt_deg))
                if improved:
                    transform, fitness, rmse, ncorr, tilt = rt, rf, rr, rn, rtilt
                    method = method + "+full_res"
                else:
                    log.info("%s -> %s: full-resolution refine did not improve, "
                             "keeping the prior result", src.name, tgt.name)

    res.transform = transform
    res.fitness = fitness
    res.rmse = rmse
    res.n_correspondences = ncorr
    res.tilt_deg = tilt
    res.interpenetration = interpenetration(src, tgt, transform)
    res.method = method
    res.seeded_from = origin
    res.vote_rejected_count = rejected_for_vote
    if use_vote and best_verdict is not None:
        # Only when the vote ran: best_verdict's defaults would otherwise read as "ran and found nothing".
        res.vote_n_planes_considered = best_verdict.n_planes_considered
        res.vote_n_corners_considered = best_verdict.n_corners_considered
        res.vote_n_edges_considered = best_verdict.n_edges_considered
        res.vote_n_openings_considered = best_verdict.n_openings_considered
        res.vote_opening_agreement = best_verdict.opening_agreement
        res.vote_n_landmarks_considered = best_verdict.n_landmarks_considered
        res.vote_landmark_agreement = best_verdict.landmark_agreement
        res.vote_plane_agreement = best_verdict.plane_agreement
        res.vote_corner_agreement = best_verdict.corner_agreement
        res.vote_edge_agreement = best_verdict.edge_agreement
        res.vote_surface_fit = best_verdict.surface_fit
        res.vote_score = best_verdict.vote_score
        res.vote_detector_score = best_verdict.detector_score
    if rejected_for_tilt:
        log.info("%s -> %s: %d candidate(s) discarded as tilted",
                 src.name, tgt.name, rejected_for_tilt)

    ratio = None
    if (cfg.relative_fitness_gate_enabled and local_only
            and src_scan is not None and tgt_scan is not None):
        # At the scored transform, full resolution, at the final ladder radius.
        achievable = achievable_overlap_frac(
            src_scan, tgt_scan, transform, cfg.voxel_size * cfg.icp_distance_mult, cfg)
        res.achievable_frac = achievable
        if achievable > 0:
            ratio = fitness / achievable
            res.fitness_over_achievable = ratio

    if ratio is not None:
        fitness_reject_reason = (
            None if ratio >= cfg.relative_fitness_min_ratio else
            f"fitness/achievable {ratio:.3f} < min {cfg.relative_fitness_min_ratio} "
            f"(raw fitness {fitness:.3f}, achievable {achievable:.3f})")
        fitness_why = _why.too_few_of_possible(src.name, tgt.name, ratio, cfg.relative_fitness_min_ratio)
    else:
        fitness_reject_reason = (
            None if fitness >= cfg.min_fitness else
            f"fitness {fitness:.3f} < min {cfg.min_fitness}")
        fitness_why = _why.too_few_on_surface(src.name, tgt.name, fitness, cfg.min_fitness)

    # The overlap gate runs last and is skipped (not failed) without full-resolution scans.
    reject_reason = _reject_reason_from_gates(
        res, fitness_reject_reason, fitness, rmse, ncorr, transform,
        src_scan, tgt_scan, cfg, fitness_why)

    if reject_reason is not None:
        # Seeded runs: failing the gates falls back to the seed; keeping the placement is never a rejection.
        if local_only and seed_outcome is None and screen is not None:
            # The GPU screen skips the seed's own CPU ladder; it is only needed here, so run it now.
            manual_start = next((s for o, s in candidates if o == "manual"), None)
            if manual_start is not None:
                refined_seed = _score(src, tgt, manual_start, cfg, local_only=True)
                seed_outcome = (*refined_seed, tilt_of(refined_seed[0]))
        if local_only and seed_outcome is not None:
            log.info("%s -> %s: local candidate rejected (%s), keeping the placement instead",
                      src.name, tgt.name, reject_reason)
            transform, fitness, rmse, ncorr, tilt = seed_outcome
            res.transform, res.fitness, res.rmse = transform, fitness, rmse
            res.n_correspondences, res.tilt_deg = ncorr, tilt
            res.interpenetration = interpenetration(src, tgt, transform)
            res.information = _information_matrix(src, tgt, transform, cfg)
            res.overlap_min_frac = None  # measured against the rejected transform, not this one
            res.accepted = True
            # transform is the ICP-refined seed, not init; say so when it actually moved.
            d_m, d_deg = _pose_delta(init, transform)
            if d_m < 1e-3 and d_deg < 0.05:
                res.method, res.seeded_from = "kept_placement", "kept_placement"
                res.reason = f"kept your placement - local candidate rejected ({reject_reason})"
                res.why = _why.kept_exactly(res.why)
            else:
                res.method, res.seeded_from = "refined_seed", "manual"
                res.reason = (f"your placement refined by {d_m:.2f} m and {d_deg:.1f} degrees; the search's best "
                              f"other position failed a quality check ({reject_reason}), so check this pair")
                res.why = _why.kept_and_fine_tuned(d_m, d_deg, res.why)
            return res
        res.reason = reject_reason
        res.why = _why.rejected(res.why)
        return res

    res.information = _information_matrix(src, tgt, transform, cfg)
    res.accepted = True
    res.reason = "ok"
    res.why = _why.matched(src.name, tgt.name, fitness, rmse)
    return res
