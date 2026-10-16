"""Manual pre-alignment by point picking: two matching points fix yaw plus translation, and the solve is yaw-only
so it cannot come back tilted. solve_from_correspondences is headless; only pick_points needs a display."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .align_gravity import estimate_level, tilt_of

log = logging.getLogger(__name__)


@dataclass
class ManualAlignment:
    transform: np.ndarray          # 4x4 in the ORIGINAL frames, source -> target
    yaw_deg: float
    residuals: np.ndarray          # per correspondence distance after solving, metres
    n_points: int

    @property
    def rms_residual(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2))) if self.residuals.size else 0.0


def solve_yaw_translation(
    src: np.ndarray, tgt: np.ndarray, weights: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    """Closed-form least-squares yaw and translation mapping src onto tgt, both levelled. Optional weights (one per
    pair) weight every sum and mean; equal weights or None give the unweighted result exactly."""
    src = np.asarray(src, dtype=float).reshape(-1, 3)
    tgt = np.asarray(tgt, dtype=float).reshape(-1, 3)
    if src.shape != tgt.shape or src.shape[0] < 2:
        raise ValueError("need at least 2 matching points in each cloud")

    if weights is None:
        w = np.ones(src.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != src.shape[0]:
            raise ValueError("weights must have one entry per correspondence")
        if np.any(np.isnan(w)):
            # w < 0 below is False for NaN (NaN fails every comparison), so
            # a NaN weight passed the old check silently and produced a NaN
            # transform further down. Caught explicitly here instead.
            raise ValueError("weights must not contain NaN")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    wsum = float(w.sum())
    if wsum <= 0:
        raise ValueError("weights sum to zero; nothing to solve from")

    sc = (w[:, None] * src[:, :2]).sum(axis=0) / wsum
    tc = (w[:, None] * tgt[:, :2]).sum(axis=0) / wsum
    a = src[:, :2] - sc
    b = tgt[:, :2] - tc

    num = float(np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))
    den = float(np.sum(w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    if abs(num) < 1e-12 and abs(den) < 1e-12:
        raise ValueError(
            "picked points are coincident, so yaw is undetermined. Pick two "
            "features that are far apart."
        )
    theta = float(np.arctan2(num, den))

    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    t_xy = tc - rot @ sc
    t_z = float(np.sum(w * (tgt[:, 2] - src[:, 2])) / wsum)
    return float(np.degrees(theta)), np.array([t_xy[0], t_xy[1], t_z])


def solve_from_correspondences(
    src_cloud: o3d.geometry.PointCloud,
    tgt_cloud: o3d.geometry.PointCloud,
    src_points: np.ndarray,
    tgt_points: np.ndarray,
) -> ManualAlignment:
    """Turn picked points (each in its own cloud's coordinates) into a transform in the original frames: level both,
    solve levelled, compose back for ICP."""
    src_level = estimate_level(src_cloud)
    tgt_level = estimate_level(tgt_cloud)

    sp = np.asarray(src_points, dtype=float).reshape(-1, 3) @ src_level.rotation.T
    tp = np.asarray(tgt_points, dtype=float).reshape(-1, 3) @ tgt_level.rotation.T

    yaw_deg, t = solve_yaw_translation(sp, tp)

    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    yt = np.eye(4)
    yt[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    yt[:3, 3] = t

    moved = (yt[:3, :3] @ sp.T).T + yt[:3, 3]
    residuals = np.linalg.norm(moved - tp, axis=1)

    rs = np.eye(4); rs[:3, :3] = src_level.rotation
    rt_inv = np.eye(4); rt_inv[:3, :3] = tgt_level.rotation.T
    transform = rt_inv @ yt @ rs

    return ManualAlignment(
        transform=transform,
        yaw_deg=yaw_deg,
        residuals=residuals,
        n_points=sp.shape[0],
    )


def pick_points(pcd: o3d.geometry.PointCloud, title: str) -> np.ndarray:
    """Open the Open3D editing viewer and return the picked XYZ coordinates (shift+left pick, shift+right undo,
    Q/Escape done). Needs a display."""
    print()
    print("=" * 68)
    print(f"PICK POINTS: {title}")
    print("  shift + left click   pick a point")
    print("  shift + right click  undo the last pick")
    print("  Q or Escape          done, close the window")
    print("=" * 68)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=860)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    idx = vis.get_picked_points()
    pts = np.asarray(pcd.points)
    picked = pts[idx] if len(idx) else np.zeros((0, 3))
    print(f"picked {len(idx)} point(s) in {title}")
    return picked


def interactive_align(
    src_cloud: o3d.geometry.PointCloud,
    tgt_cloud: o3d.geometry.PointCloud,
    src_name: str,
    tgt_name: str,
) -> ManualAlignment:
    """Full picking round trip: pick in target, pick in source, solve."""
    tgt_points = pick_points(tgt_cloud, f"TARGET: {tgt_name}")
    if len(tgt_points) < 2:
        raise ValueError("need at least 2 points picked in the target")

    src_points = pick_points(src_cloud, f"SOURCE: {src_name} (same features, same order)")
    if len(src_points) != len(tgt_points):
        raise ValueError(
            f"picked {len(src_points)} points in the source but "
            f"{len(tgt_points)} in the target. They must match one to one, "
            f"in the same order."
        )

    return solve_from_correspondences(src_cloud, tgt_cloud, src_points, tgt_points)


# --- seed file handling -------------------------------------------------

def seed_key(src_name: str, tgt_name: str) -> str:
    return f"{src_name}|{tgt_name}"


def load_seeds(path: str) -> dict[str, np.ndarray]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    return {k: np.array(v["transform"], dtype=float) for k, v in raw.items()}


def save_seed(path: str, src_name: str, tgt_name: str,
              alignment: ManualAlignment) -> None:
    raw: dict = {}
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
    raw[seed_key(src_name, tgt_name)] = {
        "source": src_name,
        "target": tgt_name,
        "transform": alignment.transform.tolist(),
        "yaw_deg": alignment.yaw_deg,
        "n_points": alignment.n_points,
        "rms_residual_m": alignment.rms_residual,
        "tilt_deg": tilt_of(alignment.transform),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(raw, fh, indent=2)
    log.info("saved seed for %s -> %s in %s", src_name, tgt_name, path)
