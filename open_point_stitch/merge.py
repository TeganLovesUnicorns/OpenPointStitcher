"""Apply optimised poses, merge at full resolution, export.

Lossless by default: every point out, with colour and intensity, at full precision. Merges in numpy so the
intensity column survives."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .pts_file import Scan, write_pts

log = logging.getLogger(__name__)


@dataclass
class MergeResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    intensity: Optional[np.ndarray]
    n_before_clean: int
    n_after_clean: int
    used_indices: list[int]

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def to_o3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd

    @property
    def cloud(self) -> o3d.geometry.PointCloud:
        """Kept so callers that expected an Open3D cloud still work."""
        return self.to_o3d()


def merge_scans(
    scans: list[Scan],
    poses: list[np.ndarray],
    cfg: PipelineConfig,
    only_indices: Optional[list[int]] = None,
) -> MergeResult:
    """Transform every scan into the shared frame and concatenate.

    Nothing is discarded unless the config explicitly asks for it.
    """
    indices = sorted(only_indices) if only_indices is not None else list(range(len(scans)))

    any_color = any(scans[i].has_color for i in indices)
    all_color = all(scans[i].has_color for i in indices)
    if any_color and not all_color:
        log.warning("Some scans have no colour. Those points will be written grey.")

    any_intensity = any(scans[i].intensity is not None for i in indices)

    pts_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []

    for i in indices:
        scan = scans[i]
        pose = np.asarray(poses[i], dtype=np.float64)
        pts_parts.append((pose[:3, :3] @ scan.points.T).T + pose[:3, 3])

        if any_color:
            col_parts.append(scan.colors if scan.has_color
                             else np.full((scan.n_points, 3), 0.5))
        if any_intensity:
            # -1 marks an unpopulated intensity field.
            int_parts.append(scan.intensity if scan.intensity is not None
                             else np.full(scan.n_points, -1.0))

    points = np.vstack(pts_parts)
    colors = np.vstack(col_parts) if col_parts else None
    intensity = np.concatenate(int_parts) if int_parts else None
    n_before = int(points.shape[0])

    # Both of the following throw points away. Off by default.
    if cfg.merge_voxel_size:
        log.warning("Downsampling the merged cloud to %.3f m. Points will be lost.",
                    cfg.merge_voxel_size)
        lo = points.min(axis=0)
        key = np.floor((points - lo) / cfg.merge_voxel_size).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        keep = np.sort(first)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
        if intensity is not None:
            intensity = intensity[keep]

    if cfg.outlier_nb_neighbors:
        log.warning("Removing statistical outliers. Points will be lost.")
        work = o3d.geometry.PointCloud()
        work.points = o3d.utility.Vector3dVector(points)
        _, kept = work.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_nb_neighbors,
            std_ratio=cfg.outlier_std_ratio,
        )
        kept = np.asarray(kept)
        points = points[kept]
        if colors is not None:
            colors = colors[kept]
        if intensity is not None:
            intensity = intensity[kept]

    n_after = int(points.shape[0])
    if n_after == n_before:
        log.info("merged %d scan(s): %d points, nothing discarded",
                 len(indices), n_after)
    else:
        log.info("merged %d scan(s): %d -> %d points", len(indices), n_before, n_after)

    return MergeResult(points=points, colors=colors, intensity=intensity,
                       n_before_clean=n_before, n_after_clean=n_after,
                       used_indices=indices)


def export(result: MergeResult, path: str, cfg: PipelineConfig) -> None:
    """Write the merged cloud: .pts via our writer (lossless), other formats via Open3D (loses intensity, warns)."""
    lower = path.lower()
    if lower.endswith(".pts"):
        write_pts(path, result.points, result.colors, result.intensity,
                  precision=cfg.output_precision)
    elif lower.endswith((".ply", ".pcd", ".xyz", ".xyzrgb")):
        if result.intensity is not None:
            log.warning("%s cannot carry the intensity column, it will be dropped. "
                        "Export .pts to keep it.", path)
        if lower.endswith(".xyz") and result.colors is not None:
            # Open3D's .xyz writer has no colour channel (.xyzrgb does), so warn rather than drop colour silently.
            log.warning("%s cannot carry colour, it will be dropped. "
                        "Export .pts, .ply, .pcd or .xyzrgb to keep it.", path)
        if not o3d.io.write_point_cloud(path, result.to_o3d()):
            raise IOError(f"Open3D failed to write {path}")
    else:
        raise ValueError(
            f"Unsupported output extension for {path}. Use .pts, .ply, .pcd or .xyz"
        )
    log.info("wrote %s: %d points, colour=%s, intensity=%s, %d decimal places",
             path, result.n_points,
             "yes" if result.colors is not None else "no",
             "yes" if result.intensity is not None else "no",
             cfg.output_precision)
