"""CUDA pair registration: coarse (many on-device hypotheses), fine (multiscale 4 DOF ICP), then scoring via
score.py. Every transform maps a scan's raw, unlevelled points into the other's raw frame; nothing here levels."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .align_gravity import coarse_candidates
from .lib.poses import solve_yaw_translation, yaw_transform
from .prepare import Prepared
from .score import Verdict, build_target_index, score_candidate_fast
from .find_walls import wall_candidates
from . import gpu_stages as _gpu
from . import explain_link as _why
from .align_pair import (
    PairResult, interpenetration, _information_matrix, _reject_reason_from_gates,
)

log = logging.getLogger(__name__)

treg = o3d.t.pipelines.registration


# Device prep reuses gpu_stages.py's existing helper (downsample, normals, FPFH).
resolve_device = _gpu.resolve_device
DevicePrepared = _gpu.DevicePrepared


def prepare_scan(pcd_full_res: o3d.geometry.PointCloud, cfg: PipelineConfig,
                 device: o3d.core.Device, name: str, index: int) -> DevicePrepared:
    """Prepare one scan on the card. Call ONCE per scan and reuse the result
    across every pair that scan appears in - re-calling per pair loses the
    whole saving."""
    return _gpu.prepare_on_device(pcd_full_res, cfg, device, name, index, with_features=True)


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class CudaPairResult:
    transform: np.ndarray            # 4x4, src RAW frame -> tgt RAW frame
    fitness: float
    rmse: float
    surface_fit: float               # score.py's ranking number - the honest score
    tilt_deg: float
    n_correspondences: int
    origin: str                      # which strategy proposed the WINNING candidate
    verified: bool                   # did the winner survive score.py's hard gates
    n_coarse_candidates: int
    n_refined: int
    rounds: int                      # how many hypothesis batches were run (1, unless time-budgeted)
    stages_on_device: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# coarse: many hypotheses from on-device feature matching
# ---------------------------------------------------------------------------

def _feature_hypotheses(src: DevicePrepared, tgt: DevicePrepared,
                        n_hypotheses: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Sample n_hypotheses minimal (two-correspondence) 4 DOF hypotheses from on-device FPFH matches: the GPU
    replacement for RANSAC's sample-and-verify loop."""
    corr = _gpu.feature_matches(src, tgt)  # (n, 2) int array: src_idx, tgt_idx
    if corr.shape[0] < 2:
        return []
    src_pts = src.down.point.positions.cpu().numpy().astype(np.float64)
    tgt_pts = tgt.down.point.positions.cpu().numpy().astype(np.float64)
    n = corr.shape[0]
    pairs = rng.integers(0, n, size=(n_hypotheses, 2))
    hyps: list[np.ndarray] = []
    for a, b in pairs:
        if a == b:
            continue
        s_idx = corr[[a, b], 0]
        t_idx = corr[[a, b], 1]
        try:
            yaw_deg, t = solve_yaw_translation(src_pts[s_idx], tgt_pts[t_idx])
        except ValueError:
            continue  # coincident picks - not a usable hypothesis, skip it
        hyps.append(yaw_transform(yaw_deg, t))
    return hyps


def _build_device_index(tgt: DevicePrepared) -> "o3d.core.nns.NearestNeighborSearch":
    """Build once per pair, not once per hypothesis round."""
    tgt_pts_t = tgt.down.point.positions.to(o3d.core.float64)
    nns = o3d.core.nns.NearestNeighborSearch(tgt_pts_t)
    nns.knn_index()
    return nns


def _score_transforms_batch(
    transforms: list[np.ndarray], src: DevicePrepared, tgt: DevicePrepared,
    cfg: PipelineConfig, rng: np.random.Generator,
    target_nns: "o3d.core.nns.NearestNeighborSearch", sample_n: int = 800,
) -> np.ndarray:
    """Inlier fraction for every transform in transforms, in one batched on-device nearest-neighbour search.
    target_nns comes from _build_device_index, built once per pair."""
    if not transforms:
        return np.zeros(0)
    device = tgt.device
    src_pts = src.down.point.positions.cpu().numpy().astype(np.float64)
    n_src = src_pts.shape[0]
    m = min(sample_n, n_src)
    idx = rng.choice(n_src, size=m, replace=False)
    sample = src_pts[idx]  # (m, 3), the SAME sample for every candidate this call

    stacked = np.empty((len(transforms) * m, 3), dtype=np.float64)
    for k, T in enumerate(transforms):
        stacked[k * m:(k + 1) * m] = (T[:3, :3] @ sample.T).T + T[:3, 3]

    query = o3d.core.Tensor(stacked, dtype=o3d.core.float64, device=device)
    _, sq_dist = target_nns.knn_search(query, 1)
    sq_dist = sq_dist.cpu().numpy().reshape(len(transforms), m)
    window = cfg.voxel_size * cfg.icp_distance_mult
    return (np.sqrt(sq_dist) <= window).mean(axis=1)


# Final polish: CPU, legacy, deliberately. Runs on a handful of survivors, never the raw hypothesis pool.
_legacy_reg = o3d.pipelines.registration


def _legacy_polish(src_down_legacy: o3d.geometry.PointCloud, tgt_down_legacy: o3d.geometry.PointCloud,
                   init: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, float]:
    """One legacy point-to-plane ICP pass on the CPU. Returns the transform and Open3D's fitness; cheap enough to
    rank many candidates."""
    dist = cfg.voxel_size * cfg.icp_distance_mult
    est = _legacy_reg.TransformationEstimationPointToPlane()
    result = _legacy_reg.registration_icp(
        src_down_legacy, tgt_down_legacy, dist, np.asarray(init, dtype=np.float64), est,
        _legacy_reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iteration))
    return np.asarray(result.transformation), float(result.fitness)


# ---------------------------------------------------------------------------
# the whole pair
# ---------------------------------------------------------------------------

def register_pair_cuda(
    src: Prepared, tgt: Prepared,
    src_gpu: DevicePrepared, tgt_gpu: DevicePrepared,
    cfg: PipelineConfig,
    n_hypotheses: int = 300,
    top_k_hypotheses: int = 8,
    seed: int = 42,
    time_budget_s: Optional[float] = None,
) -> CudaPairResult:
    """Register one pair in three tiers: coarse hypotheses, a _legacy_polish pass on each survivor, then
    score.score_candidate_fast on all. src/tgt are CPU Prepared; src_gpu/tgt_gpu are on the GPU. time_budget_s
    (default None, one round) adds more hypothesis rounds."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    # The fine ICP polish is CPU (_legacy_polish) - only these five stages
    # genuinely run on the card in the path this function actually takes.
    stages: list[str] = ["voxel downsample", "normals", "FPFH features",
                         "feature matching", "batched candidate scoring"]

    all_candidates: list[tuple[np.ndarray, str]] = []
    try:
        wall_t, _, _ = wall_candidates(src.down, tgt.down)
        all_candidates += [(t, "walls") for t in wall_t]
    except Exception as exc:  # noqa: BLE001 - one strategy failing must not fail the pair
        log.warning("%s -> %s: wall matching failed (%s)", src.name, tgt.name, exc)
    try:
        grav_t, _src_lvl, _tgt_lvl = coarse_candidates(
            src.down, tgt.down, res=cfg.gravity_grid_res,
            yaw_step_deg=cfg.gravity_yaw_step_deg, top_k=cfg.gravity_top_k)
        all_candidates += [(t, "gravity") for t in grav_t]
    except Exception as exc:  # noqa: BLE001
        log.warning("%s -> %s: gravity search failed (%s)", src.name, tgt.name, exc)

    # Wall and gravity candidates go straight to polish, never through the hypothesis screen.
    to_polish: list[tuple[np.ndarray, str]] = list(all_candidates)
    coarse_nns = _build_device_index(tgt_gpu)

    def add_hypothesis_round() -> int:
        hyps = _feature_hypotheses(src_gpu, tgt_gpu, n_hypotheses, rng)
        if not hyps:
            return 0
        # Score raw hypotheses directly: refining each first is too slow at this count.
        scores = _score_transforms_batch(hyps, src_gpu, tgt_gpu, cfg, rng, coarse_nns)
        order = np.argsort(-scores)[:top_k_hypotheses]
        for i in order:
            to_polish.append((hyps[int(i)], "cuda-feature"))
        all_candidates.extend((h, "cuda-feature") for h in hyps)
        return len(hyps)

    add_hypothesis_round()
    rounds = 1

    if time_budget_s:
        while time.time() - t0 < time_budget_s:
            if add_hypothesis_round() == 0:
                break
            rounds += 1

    # Polish everyone, then score every polished candidate honestly - no
    # top-N cutoff needed before score.py (score_candidate_fast is cheap
    # enough to run on all of them).
    polished = [(*_legacy_polish(src.down, tgt.down, T0, cfg), origin) for T0, origin in to_polish]

    target_nns = build_target_index(tgt.down)
    verdicts: list[Verdict] = []
    for T_polished, _icp_fitness, origin in polished:
        v = score_candidate_fast(src.down, tgt.down, target_nns, T_polished, cfg, origin=origin)
        verdicts.append(v)

    survivors = [v for v in verdicts if v.ok]
    ranked = sorted(survivors, key=lambda v: -v.surface_fit) if survivors else []
    if ranked:
        best = ranked[0]
        verified = True
    else:
        # Nothing passed the hard gates. Report the best-by-fitness candidate marked unverified rather than raise.
        best = max(verdicts, key=lambda v: v.fitness) if verdicts else Verdict(transform=np.eye(4))
        verified = False

    return CudaPairResult(
        transform=best.transform, fitness=best.fitness, rmse=best.rmse,
        surface_fit=best.surface_fit, tilt_deg=best.tilt_deg,
        n_correspondences=best.n_matched, origin=best.origin, verified=verified,
        n_coarse_candidates=len(all_candidates), n_refined=len(polished), rounds=rounds,
        stages_on_device=stages, elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# prepare every scan on the card ONCE, for the whole run
# ---------------------------------------------------------------------------

def prepare_all_gpu(prepared: list[Prepared], cfg: PipelineConfig) -> list[DevicePrepared]:
    """GPU counterpart to prepare.preprocess_all, called once per run and reused for every pair. Works on each
    scan's downsampled .down cloud."""
    device = resolve_device(True)
    return [prepare_scan(p.down, cfg, device, p.name, p.index) for p in prepared]


# ---------------------------------------------------------------------------
# PairResult wrapper: same shape callers of register_pair already expect
# ---------------------------------------------------------------------------

def register_pair_cuda_gated(
    src: Prepared, tgt: Prepared,
    src_gpu: DevicePrepared, tgt_gpu: DevicePrepared,
    cfg: PipelineConfig,
    src_scan=None, tgt_scan=None,
) -> PairResult:
    """register_pair_cuda through the same accept/reject gates as register_pair. Unseeded runs only."""
    res = PairResult(
        source=src.index, target=tgt.index,
        source_name=src.name, target_name=tgt.name,
        transform=np.eye(4), information=np.eye(6),
        seeded_from="ransac",
    )
    if frozenset((src.name, tgt.name)) in cfg.excluded_pairs:
        res.reason = "manually excluded (cfg.excluded_pairs)"
        res.why = _why.EXCLUDED_PAIR
        return res

    cuda_result = register_pair_cuda(src, tgt, src_gpu, tgt_gpu, cfg)

    res.coarse_fitness = float("nan")
    res.coarse_rmse = float("nan")
    res.n_candidates = cuda_result.n_coarse_candidates
    res.transform = cuda_result.transform
    res.fitness = cuda_result.fitness
    res.rmse = cuda_result.rmse
    res.n_correspondences = cuda_result.n_correspondences
    res.tilt_deg = cuda_result.tilt_deg
    res.method = f"cuda+{cfg.icp_method}"
    res.seeded_from = cuda_result.origin

    if cfg.assume_level and cuda_result.tilt_deg > cfg.max_tilt_deg:
        res.reason = f"tips the world {cuda_result.tilt_deg:.1f} deg off level"
        res.why = _why.tips_off_level(cuda_result.tilt_deg)
        return res
    if not cuda_result.verified:
        res.reason = "no candidate survived score.py's hard gates (unverified)"
        res.why = _why.NOTHING_SURVIVED
        return res

    res.interpenetration = interpenetration(src, tgt, cuda_result.transform)

    fitness_reject_reason = (
        None if cuda_result.fitness >= cfg.min_fitness else
        f"fitness {cuda_result.fitness:.3f} < min {cfg.min_fitness}")
    reject_reason = _reject_reason_from_gates(
        res, fitness_reject_reason, cuda_result.fitness, cuda_result.rmse,
        cuda_result.n_correspondences, cuda_result.transform, src_scan, tgt_scan, cfg,
        _why.too_few_on_surface(src.name, tgt.name, cuda_result.fitness, cfg.min_fitness))
    if reject_reason is not None:
        res.reason = reject_reason
        res.why = _why.rejected(res.why)
        return res

    res.information = _information_matrix(src, tgt, cuda_result.transform, cfg)
    res.accepted = True
    res.reason = "ok"
    res.why = _why.matched(src.name, tgt.name, cuda_result.fitness, cuda_result.rmse)
    return res
