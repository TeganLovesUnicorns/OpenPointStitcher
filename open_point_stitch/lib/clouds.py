"""Point arrays to and from Open3D clouds."""

from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d


def as_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None,
             normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """An Open3D cloud holding copies of the given N x 3 arrays; colours in 0..1."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
