"""GPU stages via Open3D's tensor API (open3d.t.*).

On the GPU: voxel downsample, normals, FPFH, feature matching, ICP, information matrix. RANSAC and the pose graph
stay on the CPU. Every neighbour search passes both max_nn and radius (hybrid), the only fast form on the GPU."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from .config import PipelineConfig

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

