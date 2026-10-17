"""Registration on the graphics card: the tensor-API stages (downsample, normals, FPFH, feature matching), the
repeatable point-to-plane ICP that screens a pair's starting poses, and the CUDA pair pipeline for unseeded runs.
RANSAC and the pose graph stay on the CPU."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d

from . import explain as _why
from .align_gravity import coarse_candidates
from .align_pair import PairResult, _information_matrix, _reject_reason_from_gates, interpenetration
from .config import PipelineConfig
from .find_walls import wall_candidates
from .lib.poses import solve_yaw_translation, yaw_transform
from .prepare import Prepared
from .score import Verdict, build_target_index, score_candidate_fast

log = logging.getLogger(__name__)

treg = o3d.t.pipelines.registration


def resolve_device(use_gpu: bool) -> o3d.core.Device:
    """Pick the device to run on, without ever pretending we have a GPU."""
    if not use_gpu:
        return o3d.core.Device("CPU:0")
    from .limits import gpu_status
    st = gpu_status()
    if not st["gpu_usable"]:
        log.warning("GPU asked for but not usable, staying on CPU. %s", st["notes"])
        return o3d.core.Device("CPU:0")
    return o3d.core.Device("CUDA:0")


class DevicePrepared:
    """A scan reduced to what tensor-API registration needs, on one device."""

    __slots__ = ("name", "index", "down", "fpfh", "device", "n_full")

    def __init__(self, name: str, index: int, down, fpfh, device, n_full: int):
        self.name = name
        self.index = index
        self.down = down
        self.fpfh = fpfh
        self.device = device
        self.n_full = n_full

    @property
    def n_down(self) -> int:
        return int(self.down.point.positions.shape[0])


def prepare_on_device(
    pcd_legacy: o3d.geometry.PointCloud,
    cfg: PipelineConfig,
    device: o3d.core.Device,
    name: str = "",
    index: int = -1,
    with_features: bool = True,
) -> DevicePrepared:
    """Downsample, estimate normals and compute FPFH, all on `device`."""
    n_full = len(pcd_legacy.points)

    # float32 is what the tensor API is built around; float64 doubles memory for no gain at millimetre scale.
    t = o3d.t.geometry.PointCloud.from_legacy(
        pcd_legacy, dtype=o3d.core.float32, device=device
    )
    t = t.voxel_down_sample(cfg.voxel_size)

    # Hybrid search on both counts. See the module docstring.
    t.estimate_normals(
        max_nn=cfg.normal_max_nn,
        radius=cfg.voxel_size * cfg.normal_radius_mult,
    )

    fpfh = None
    if with_features:
        fpfh = treg.compute_fpfh_feature(
            t,
            max_nn=cfg.fpfh_max_nn,
            radius=cfg.voxel_size * cfg.fpfh_radius_mult,
        )
    return DevicePrepared(name, index, t, fpfh, device, n_full)


def icp_on_device(
    src: DevicePrepared,
    tgt: DevicePrepared,
    init: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, float, float]:
    """Point-to-plane ICP on whichever device the clouds already live on.

    Returns (4x4 transform, fitness, inlier rmse), matching what the legacy
    path returns so callers do not need to care which ran.
    """
    device = src.down.device
    init_t = o3d.core.Tensor(
        np.ascontiguousarray(np.asarray(init, dtype=np.float64)),
        dtype=o3d.core.float64,
        device=device,
    )
    max_dist = cfg.voxel_size * cfg.icp_distance_mult

    if cfg.icp_method == "point_to_point":
        est = treg.TransformationEstimationPointToPoint()
    elif cfg.icp_method == "colored":
        est = treg.TransformationEstimationForColoredICP()
    else:
        est = treg.TransformationEstimationPointToPlane()

    result = treg.icp(
        source=src.down,
        target=tgt.down,
        max_correspondence_distance=max_dist,
        init_source_to_target=init_t,
        estimation_method=est,
        criteria=treg.ICPConvergenceCriteria(
            max_iteration=cfg.icp_max_iteration,
            relative_fitness=1e-6,
            relative_rmse=1e-6,
        ),
    )
    T = result.transformation.cpu().numpy().astype(np.float64)
    return T, float(result.fitness), float(result.inlier_rmse)


def feature_matches(src: DevicePrepared, tgt: DevicePrepared,
                    mutual: bool = True) -> np.ndarray:
    """Nearest-neighbour matches between FPFH descriptors, on device. A
    building block only: RANSAC has no tensor version in 0.19 and stays on CPU."""
    if src.fpfh is None or tgt.fpfh is None:
        raise ValueError("features were not computed; pass with_features=True")
    corr = treg.correspondences_from_features(src.fpfh, tgt.fpfh, mutual_filter=mutual)
    return corr.cpu().numpy()


# -------------------------------------------------------------------------------------------------------------------
# Point-to-plane ICP on the GPU that gives the same answer every run. Open3D's tensor ICP does not repeat, so this
# uses only operations that do (nearest-neighbour search, elementwise maths, one matrix product) and solves the 6 by 6
# system on the CPU. It follows the CPU ICP step for step.
# -------------------------------------------------------------------------------------------------------------------

_LOCK = threading.Lock()
F64 = o3d.core.float64


class GpuCloud:
    """One scan's downsampled points and normals on the card, in float64 (exact, and these operations are memory bound)."""

    def __init__(self, pcd: o3d.geometry.PointCloud, device: o3d.core.Device):
        pts = np.ascontiguousarray(np.asarray(pcd.points), dtype=np.float64)
        nrm = np.ascontiguousarray(np.asarray(pcd.normals), dtype=np.float64)
        if nrm.shape != pts.shape:
            raise ValueError("a cloud needs normals for point-to-plane ICP")
        self.device = device
        self.n = int(pts.shape[0])
        self.points = o3d.core.Tensor(pts, dtype=F64, device=device)
        self.normals = o3d.core.Tensor(nrm, dtype=F64, device=device)
        self._indexes: dict[float, object] = {}

    def index(self, radius: float):
        """A fixed-radius grid index of this cloud for one search window, built the first time it is needed as a target."""
        with _LOCK:
            nns = self._indexes.get(radius)
            if nns is None:
                nns = o3d.core.nns.NearestNeighborSearch(self.points)
                nns.hybrid_index(radius)
                self._indexes[radius] = nns
        return nns


def cloud_for(prepared, device: o3d.core.Device | None = None) -> GpuCloud:
    """The GPU copy of a prepared scan, made once and kept on it."""
    cloud = getattr(prepared, "_gpu_cloud", None)
    if cloud is None:
        with _LOCK:
            cloud = getattr(prepared, "_gpu_cloud", None)
            if cloud is None:
                cloud = GpuCloud(prepared.down, device or o3d.core.Device("CUDA:0"))
                prepared._gpu_cloud = cloud
    return cloud


def coarse_for(prepared, voxel: float, device: o3d.core.Device | None = None) -> GpuCloud:
    """A coarser GPU copy of a prepared scan (its cloud thinned to `voxel`, normals re-normalised), made once."""
    cloud = getattr(prepared, "_gpu_coarse", None)
    if cloud is None:
        with _LOCK:
            cloud = getattr(prepared, "_gpu_coarse", None)
            if cloud is None:
                thin = prepared.down.voxel_down_sample(voxel)
                nrm = np.asarray(thin.normals)
                thin.normals = o3d.utility.Vector3dVector(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12))
                cloud = GpuCloud(thin, device or o3d.core.Device("CUDA:0"))
                prepared._gpu_coarse = cloud
    return cloud


def _motion(x: np.ndarray) -> np.ndarray:
    """The 4x4 for a small motion (rx, ry, rz, tx, ty, tz), composed the way the CPU ICP composes it (Rz Ry Rx)."""
    cx, sx, cy, sy, cz, sz = np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1]), np.cos(x[2]), np.sin(x[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = x[3:]
    return m


def _evaluate(src: GpuCloud, tgt: GpuCloud, index, T: np.ndarray, radius: float):
    """Where the source lands under T: its matches within the window, and the normal equations for the next step.
    Returns (A 6x6, b 6, match count, sum of squared match distances)."""
    dev = src.device
    rot_t = o3d.core.Tensor(np.ascontiguousarray(T[:3, :3].T), dtype=F64, device=dev)
    shift = o3d.core.Tensor(np.ascontiguousarray(T[:3, 3].reshape(1, 3)), dtype=F64, device=dev)
    moved = src.points.matmul(rot_t) + shift
    idx, _, count = index.hybrid_search(moved, radius, 1)
    found = count.reshape((-1,)) > 0
    idx = idx.reshape((-1,)) * found.to(idx.dtype)                # a point with no match points at row 0, weighted out below
    w = found.to(F64)
    q, n = tgt.points[idx], tgt.normals[idx]
    diff = moved - q
    d2 = (diff * diff).sum(1) * w
    r = (diff * n).sum(1) * w
    mx, my, mz = moved[:, 0], moved[:, 1], moved[:, 2]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    cols = [(my * nz - mz * ny) * w, (mz * nx - mx * nz) * w, (mx * ny - my * nx) * w, nx * w, ny * w, nz * w]
    jac = o3d.core.concatenate([c.reshape((-1, 1)) for c in cols], axis=1)
    jac_t = jac.T().contiguous()
    a = jac_t.matmul(jac)
    b = jac_t.matmul(r.reshape((-1, 1)))
    return (a.cpu().numpy(), b.cpu().numpy().reshape(6), w.sum().item(), d2.sum().item())


_PAIRS = [(i, j) for i in range(6) for j in range(i, 6)]        # the 21 distinct entries of the symmetric 6x6


def _evaluate_batch(src: GpuCloud, tgt: GpuCloud, index, transforms: np.ndarray, radius: float):
    """`_evaluate` for K starting poses at once, so the fixed cost of a GPU call is shared K ways. Rows of every working
    tensor are ordered (source point, candidate). Returns (A K x 6 x 6, b K x 6, counts K, squared distances K)."""
    k, n = transforms.shape[0], src.n
    dev = src.device
    rot = np.ascontiguousarray(np.concatenate([transforms[i, :3, :3].T for i in range(k)], axis=1))       # 3 x 3K
    shift = np.ascontiguousarray(np.concatenate([transforms[i, :3, 3] for i in range(k)]).reshape(1, 3 * k))
    moved = (src.points.matmul(o3d.core.Tensor(rot, dtype=F64, device=dev)) + o3d.core.Tensor(shift, dtype=F64, device=dev)).reshape((n * k, 3))
    idx, _, count = index.hybrid_search(moved, radius, 1)
    found = count.reshape((-1,)) > 0
    idx = idx.reshape((-1,)) * found.to(idx.dtype)
    w = found.to(F64)
    q, nrm = tgt.points[idx], tgt.normals[idx]
    diff = moved - q
    r = (diff * nrm).sum(1) * w
    mx, my, mz = moved[:, 0], moved[:, 1], moved[:, 2]
    nx, ny, nz = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    jac = [(my * nz - mz * ny) * w, (mz * nx - mx * nz) * w, (mx * ny - my * nx) * w, nx * w, ny * w, nz * w]

    def per_candidate(t):
        return t.reshape((n, k)).sum(0).reshape((1, k))

    rows = [per_candidate(jac[i] * jac[j]) for i, j in _PAIRS] + [per_candidate(jac[i] * r) for i in range(6)]
    rows += [per_candidate(w), per_candidate((diff * diff).sum(1) * w)]
    out = o3d.core.concatenate(rows, axis=0).cpu().numpy()                    # 29 x K, one transfer
    a = np.zeros((k, 6, 6))
    for row, (i, j) in enumerate(_PAIRS):
        a[:, i, j] = a[:, j, i] = out[row]
    return a, out[21:27].T.copy(), out[27], out[28]


def icp_point_to_plane_batch(src: GpuCloud, tgt: GpuCloud, inits: list, max_dist: float, max_iteration: int = 60,
                             relative_fitness: float = 1e-6, relative_rmse: float = 1e-6) -> list:
    """`icp_point_to_plane` for several starting poses at once. Each one stops on its own test, exactly as it would alone.
    Returns one (4x4, fitness, inlier rmse, match count) per start."""
    k = len(inits)
    index = tgt.index(max_dist)
    trans = np.array([np.asarray(t, dtype=np.float64) for t in inits])
    a, b, count, sum_d2 = _evaluate_batch(src, tgt, index, trans, max_dist)
    fitness = count / src.n
    rmse = np.where(count > 0, np.sqrt(sum_d2 / np.maximum(count, 1.0)), 0.0)
    active = np.ones(k, dtype=bool)
    for _ in range(max_iteration):
        active &= count >= 6
        for i in np.flatnonzero(active):
            try:
                trans[i] = _motion(np.linalg.solve(a[i], -b[i])) @ trans[i]
            except np.linalg.LinAlgError:
                active[i] = False
        live = np.flatnonzero(active)
        if live.size == 0:
            break
        prev_fitness, prev_rmse = fitness[live].copy(), rmse[live].copy()
        a[live], b[live], count[live], sum_d2[live] = _evaluate_batch(src, tgt, index, trans[live], max_dist)      # only those still moving
        fitness[live] = count[live] / src.n
        rmse[live] = np.where(count[live] > 0, np.sqrt(sum_d2[live] / np.maximum(count[live], 1.0)), 0.0)
        done = (np.abs(prev_fitness - fitness[live]) < relative_fitness) & (np.abs(prev_rmse - rmse[live]) < relative_rmse)
        active[live[done]] = False
    return [(trans[i], float(fitness[i]), float(rmse[i]), int(count[i])) for i in range(k)]


def ladder_batch(src_prepared, tgt_prepared, inits: list, cfg, coarse_from: float = 2.0, max_iteration: int | None = None) -> list:
    """The coarse-to-fine ICP ladder (`cfg.local_only_icp_distance_mults`) for several starting poses at once. Rungs whose
    window is `coarse_from` voxels or wider run on a 2x coarser copy of both clouds, the finer rungs on the full ones.
    Returns one (4x4, fitness, inlier rmse, match count) per start, from the last rung."""
    iterations = cfg.icp_max_iteration if max_iteration is None else max_iteration
    full_src, full_tgt = cloud_for(src_prepared), cloud_for(tgt_prepared)
    coarse_src = coarse_for(src_prepared, 2.0 * cfg.voxel_size)
    coarse_tgt = coarse_for(tgt_prepared, 2.0 * cfg.voxel_size)
    current = [np.asarray(t, dtype=np.float64) for t in inits]
    out = None
    for mult in cfg.local_only_icp_distance_mults:
        wide = mult >= coarse_from
        out = icp_point_to_plane_batch(coarse_src if wide else full_src, coarse_tgt if wide else full_tgt, current,
                                       cfg.voxel_size * mult, iterations)
        current = [o[0] for o in out]
    if cfg.local_only_icp_distance_mults and cfg.local_only_icp_distance_mults[-1] >= coarse_from:
        # every rung was wide: finish on the full clouds so the reported fitness is the same number the CPU reports
        out = icp_point_to_plane_batch(full_src, full_tgt, current, cfg.voxel_size * cfg.local_only_icp_distance_mults[-1],
                                       iterations)
    return out


def icp_point_to_plane(src: GpuCloud, tgt: GpuCloud, init: np.ndarray, max_dist: float, max_iteration: int = 60,
                       relative_fitness: float = 1e-6, relative_rmse: float = 1e-6):
    """Refine `init` (source into target). Returns (4x4, fitness, inlier rmse, match count), with the CPU ICP's definitions:
    fitness is matches over source points, rmse is over the matches' distances."""
    index = tgt.index(max_dist)
    T = np.array(init, dtype=np.float64)
    a, b, count, sum_d2 = _evaluate(src, tgt, index, T, max_dist)
    fitness = count / src.n
    rmse = float(np.sqrt(sum_d2 / count)) if count > 0 else 0.0
    for _ in range(max_iteration):
        if count < 6:
            break
        try:
            x = np.linalg.solve(a, -b)
        except np.linalg.LinAlgError:
            break
        T = _motion(x) @ T
        prev_fitness, prev_rmse = fitness, rmse
        a, b, count, sum_d2 = _evaluate(src, tgt, index, T, max_dist)
        fitness = count / src.n
        rmse = float(np.sqrt(sum_d2 / count)) if count > 0 else 0.0
        if abs(prev_fitness - fitness) < relative_fitness and abs(prev_rmse - rmse) < relative_rmse:
            break
    return T, fitness, rmse, int(count)


# -------------------------------------------------------------------------------------------------------------------
# CUDA pair registration: coarse (many on-device hypotheses), fine (multiscale 4 DOF ICP), then scoring via
# score.py. Every transform maps a scan's raw, unlevelled points into the other's raw frame; nothing here levels.
# -------------------------------------------------------------------------------------------------------------------

def prepare_scan(pcd_full_res: o3d.geometry.PointCloud, cfg: PipelineConfig,
                 device: o3d.core.Device, name: str, index: int) -> DevicePrepared:
    """Prepare one scan on the card. Call ONCE per scan and reuse the result
    across every pair that scan appears in - re-calling per pair loses the
    whole saving."""
    return prepare_on_device(pcd_full_res, cfg, device, name, index, with_features=True)


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
    corr = feature_matches(src, tgt)  # (n, 2) int array: src_idx, tgt_idx
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
