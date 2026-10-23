"""Point arrays to and from Open3D clouds."""

from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d


PER_POINT_KEYS = ("xy", "h", "rgb", "agreement")   # a preview payload's lists with one entry per point


def payload_subset(payload: dict, keep: np.ndarray) -> dict:
    """The preview payload with only the points where keep is True: every per-point list is cut the same way and
    marked stairs are renumbered to the points that remain."""
    keep = np.asarray(keep, dtype=bool)
    out = dict(payload)
    for k in PER_POINT_KEYS:
        if payload.get(k) is None:
            continue
        v = np.asarray(payload[k], dtype=float if k in ("xy", "h") else None)
        out[k] = v.reshape(-1, 2)[keep].ravel() if k == "xy" else v[keep]
    if payload.get("stairs"):
        stairs = np.asarray(payload["stairs"], dtype=int)
        out["stairs"] = (np.cumsum(keep) - 1)[stairs[keep[stairs]]].tolist()
    return out


def as_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None,
             normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """An Open3D cloud holding copies of the given N x 3 arrays; colours in 0..1."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
