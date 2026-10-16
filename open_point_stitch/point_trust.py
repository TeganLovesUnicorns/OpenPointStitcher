"""How much each point of a prepared scan can be trusted when lining scans up: surface thickness, mid-air points
and the trained trust map. Weights only: no point is dropped, and the export never sees any of this. Every array is
per point of Prepared.down, cached per scan."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .prepare import Prepared

_THICKNESS: dict = {}       # (scan name, point count) -> per-point surface thickness of its prepared cloud
THICK_FLOOR_M = 0.01        # surfaces this tight or tighter get full trust


def surface_thickness(p: Prepared, cfg: PipelineConfig) -> np.ndarray:
    """Per point of `p.down`: spread across the local surface (m), from its neighbours. Cached per scan."""
    key = (p.name, len(p.down.points))
    if key not in _THICKNESS:
        pcd = o3d.geometry.PointCloud(p.down.points)
        pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.voxel_size * 3, max_nn=20))
        _THICKNESS[key] = np.sqrt(np.maximum(np.linalg.eigvalsh(np.asarray(pcd.covariances))[:, 0], 0.0))
    return _THICKNESS[key]


_COUNTS: dict = {}          # (scan name, point count) -> neighbours of each point within MIDAIR_RADIUS_VOXELS
_MIDAIR: dict = {}          # (scan name, point count) -> per-point mid-air weight of its prepared cloud
MIDAIR_RADIUS_VOXELS = 3.0  # neighbours counted within this many voxels
MIDAIR_FULL_N = 8           # this many neighbours or more: full weight
MIDAIR_MIN_W = 0.1          # a lone point still counts this much: weights only, never zero


def neighbour_count(p: Prepared, cfg: PipelineConfig) -> np.ndarray:
    key = (p.name, len(p.down.points))
    if key not in _COUNTS:
        tree = o3d.geometry.KDTreeFlann(p.down)
        r = cfg.voxel_size * MIDAIR_RADIUS_VOXELS
        _COUNTS[key] = np.array([tree.search_radius_vector_3d(q, r)[0] - 1 for q in np.asarray(p.down.points)],
                                dtype=float)
    return _COUNTS[key]


def midair_weight(p: Prepared, cfg: PipelineConfig) -> np.ndarray:
    """Per point of `p.down`: 1 on a solid surface, falling to MIDAIR_MIN_W for a point with few neighbours."""
    key = (p.name, len(p.down.points))
    if key not in _MIDAIR:
        _MIDAIR[key] = np.clip(neighbour_count(p, cfg) / MIDAIR_FULL_N, MIDAIR_MIN_W, 1.0)
    return _MIDAIR[key]


POINT_FEATURES = ("log_thickness", "log_distance", "view_cos", "log_neighbours", "upright")
FEATURES = POINT_FEATURES + ("log_points", "log_patch_thickness", "flatness")   # per patch: medians, then the patch's own shape
PATCH_M = 0.5               # trust is learned per patch
PATCH_MIN = 3               # patches with fewer prepared points than this get MIDAIR_MIN_W
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trust_model.json")
_MODEL: Optional[dict] = None
_LEARNED: dict = {}


def point_features(p: Prepared, cfg: PipelineConfig) -> np.ndarray:
    """(n, len(POINT_FEATURES)): what the trust map sees of each point. Distance and view angle are measured from the
    scan's raw origin, a rough stand-in for where the device was."""
    pts = np.asarray(p.down.points)
    nrm = np.asarray(p.down.normals)
    dist = np.linalg.norm(pts, axis=1)
    ray = pts / np.maximum(dist, 1e-6)[:, None]
    return np.column_stack([
        np.log(np.maximum(surface_thickness(p, cfg), 1e-4)),
        np.log(np.maximum(dist, 0.1)),
        np.abs(np.einsum("ij,ij->i", ray, nrm)),
        np.log1p(neighbour_count(p, cfg)),
        np.abs(nrm[:, 2]),
    ])


def patch_table(feats: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(patch index per point, -1 for a patch under PATCH_MIN points; one row of FEATURES per patch). Patches are
    PATCH_M cubes in the scan's own frame. Training and the run both call this, so they cannot disagree."""
    keys = np.floor(raw / PATCH_M).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    pid = np.full(len(raw), -1)
    rows = []
    for g in range(cnt.size):
        if cnt[g] < PATCH_MIN:
            continue
        sel = order[starts[g]:starts[g] + cnt[g]]
        p = raw[sel] - raw[sel].mean(axis=0)
        ev = np.linalg.eigvalsh(np.cov(p.T))
        pid[sel] = len(rows)
        rows.append(np.r_[np.median(feats[sel], axis=0), np.log(cnt[g]),
                          np.log(np.sqrt(max(ev[0], 1e-10))), ev[0] / max(ev[1], 1e-10)])
    return pid, (np.array(rows) if rows else np.zeros((0, len(FEATURES))))


def load_model(path: str = MODEL_PATH) -> dict:
    global _MODEL
    if _MODEL is None or path != MODEL_PATH:
        with open(path) as fh:
            model = json.load(fh)
        if tuple(model["features"]) != FEATURES:
            raise ValueError(f"trust model was trained on {model['features']}, this code computes {FEATURES}")
        if path != MODEL_PATH:
            return model
        _MODEL = model
    return _MODEL


def predicted_misfit(feats: np.ndarray, model: dict) -> np.ndarray:
    """The model's expected distance (m) from this point to the matching surface of an overlapping scan."""
    x = np.column_stack([np.ones(len(feats)), feats])
    return np.exp(x @ np.asarray(model["coef"]))


def learned_weight(p: Prepared, cfg: PipelineConfig) -> np.ndarray:
    """Per point, its patch's (floor / expected misfit)^2, capped at 1: the same shape thickness trust uses, from
    the trained map. A point in a patch too small to judge gets MIDAIR_MIN_W."""
    key = (p.name, len(p.down.points))
    if key not in _LEARNED:
        model = load_model()
        floor = float(model["floor_m"])
        pid, table = patch_table(point_features(p, cfg), np.asarray(p.down.points))
        w_patch = np.minimum(1.0, (floor / np.maximum(predicted_misfit(table, model), floor)) ** 2)
        _LEARNED[key] = np.where(pid >= 0, w_patch[np.maximum(pid, 0)] if len(w_patch) else MIDAIR_MIN_W, MIDAIR_MIN_W)
    return _LEARNED[key]


def point_weight(p: Prepared, cfg: PipelineConfig) -> Optional[np.ndarray]:
    """The per-point weight the switched-on settings ask for, or None when none of them is on."""
    w = None
    if cfg.midair_trust:
        w = midair_weight(p, cfg)
    if cfg.learned_trust:
        lw = learned_weight(p, cfg)
        w = lw if w is None else w * lw
    return w
