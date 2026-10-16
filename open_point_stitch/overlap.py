"""Direct overlap: how much two full-resolution clouds share at a given transform, independent of any registration
or scoring in this project."""

from __future__ import annotations

import numpy as np
import open3d as o3d

THRESHOLDS_M = (0.05, 0.15)  # 5 cm and 15 cm


def nearest_distances(src_points: np.ndarray, tgt_points: np.ndarray) -> np.ndarray:
    """Per-point distance from src_points to the nearest tgt_points (same frame), by batched tensor search. Feeds the
    deviation heatmap."""
    src_points = np.asarray(src_points, dtype=np.float64)
    tgt_points = np.asarray(tgt_points, dtype=np.float64)
    if len(tgt_points) == 0 or len(src_points) == 0:
        return np.full(len(src_points), np.inf)
    device = o3d.core.Device("CPU:0")
    tgt_t = o3d.core.Tensor(tgt_points, dtype=o3d.core.float64, device=device)
    nns = o3d.core.nns.NearestNeighborSearch(tgt_t)
    nns.knn_index()
    q = o3d.core.Tensor(src_points, dtype=o3d.core.float64, device=device)
    _, sq = nns.knn_search(q, 1)
    return np.sqrt(sq.cpu().numpy().reshape(-1))


def direct_overlap(src_points: np.ndarray, src_to_tgt: np.ndarray,
                   tgt_points: np.ndarray, thresholds=THRESHOLDS_M) -> dict:
    """How much of `src_points` lands near `tgt_points` at the given transform, via a
    batched tensor search so it stays affordable at full resolution."""
    src_points = np.asarray(src_points, dtype=np.float64)
    tgt_points = np.asarray(tgt_points, dtype=np.float64)
    T = np.asarray(src_to_tgt, dtype=np.float64)
    moved = (T[:3, :3] @ src_points.T).T + T[:3, 3]

    device = o3d.core.Device("CPU:0")
    tgt_t = o3d.core.Tensor(tgt_points, dtype=o3d.core.float64, device=device)
    nns = o3d.core.nns.NearestNeighborSearch(tgt_t)
    nns.knn_index()
    q = o3d.core.Tensor(moved, dtype=o3d.core.float64, device=device)
    _, sq = nns.knn_search(q, 1)
    dist = np.sqrt(sq.cpu().numpy().reshape(-1))

    lo = np.maximum(moved.min(axis=0), tgt_points.min(axis=0))
    hi = np.minimum(moved.max(axis=0), tgt_points.max(axis=0))
    shared_extent = np.maximum(hi - lo, 0.0)

    return {
        "n_points": int(len(moved)),
        **{f"within_{int(t * 100)}cm_frac": float((dist <= t).mean()) for t in thresholds},
        **{f"within_{int(t * 100)}cm_n": int((dist <= t).sum()) for t in thresholds},
        "shared_bbox_m": [round(float(v), 2) for v in shared_extent],
    }
