"""Point-to-plane ICP on the GPU that gives the same answer every run. Open3D's tensor ICP does not repeat, so this
uses only operations that do (nearest-neighbour search, elementwise maths, one matrix product) and solves the 6 by 6
system on the CPU. It follows the CPU ICP step for step."""
from __future__ import annotations

import logging
import threading

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)

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
