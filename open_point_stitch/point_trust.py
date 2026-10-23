"""How much each point of a prepared scan can be trusted when lining scans up: surface thickness, mid-air points and
the trained trust map. Weights only: no point is dropped, and the export never sees any of this. Every array is per
point of Prepared.down, cached per scan. Also each scan's sensor noise from its own flat surfaces, and the link
weight the noise_trust option gives it."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

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


def ladder_tukey_base(tgt: Prepared, cfg: PipelineConfig) -> Optional[float]:
    """Twice the target's typical surface thickness, the base of every ICP step's Tukey window, or None with
    thickness trust off. Each step uses max(this, step reach / 3) so coarse steps can still close a big gap."""
    if not cfg.thickness_trust or cfg.icp_method == "point_to_point":
        return None
    return 2.0 * max(float(np.median(surface_thickness(tgt, cfg))), THICK_FLOOR_M / 2)


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


# -------------------------------------------------------------------------------------------------------------------
# A scan's sensor noise in metres, from surfaces that are flat in real life: 0.5 m patches on a large flat sheet
# (walls, floors, ceilings), scatter about each patch's own plane. Full resolution, read only.
# -------------------------------------------------------------------------------------------------------------------

NOISE_CELL_M = 0.5
GATHER_M = 0.08         # points this far either side of a patch's plane are its own when re-collected
NOISE_MIN_POINTS = 60
PLANAR = 0.2            # a patch is surface when its thinnest spread is under this share of the next
SHEET_DEG = 3.0         # patches whose planes agree this well in angle ...
SHEET_M = 0.03          # ... and offset belong to one sheet
SHEET_MIN = 6           # a sheet needs this many patches (1.5 m2) to count as a flat surface
SLOPED = (0.3, 0.9)     # |normal z| in this band is neither wall nor floor: left out (raked seating, ramps)


def scan_patches(raw: np.ndarray, level: np.ndarray) -> list[dict]:
    """Every planar 0.5 m patch of one scan, in its levelled frame (origin = where the scan started)."""
    pts = raw @ level.T
    keys = np.floor(pts / NOISE_CELL_M).astype(np.int64)
    order = np.lexsort(keys.T)
    keys, pts_s = keys[order], pts[order]
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    ends = np.r_[starts[1:], len(pts_s)]
    tree = cKDTree(pts_s)
    out = []
    for s, e in zip(starts, ends):
        if e - s < NOISE_MIN_POINTS:
            continue
        p = pts_s[s:e]
        c = p.mean(axis=0)
        vals, vecs = np.linalg.eigh(np.cov((p - c).T))
        if vals[0] > (PLANAR ** 2) * vals[1]:
            continue
        n = vecs[:, 0]
        nz = abs(float(n[2]))
        if SLOPED[0] <= nz <= SLOPED[1]:
            continue
        # A surface lying on a cell boundary is split between cells; re-collect it across the boundary.
        q = pts_s[tree.query_ball_point(c, NOISE_CELL_M)]
        rel = q - c
        across = rel @ n
        inside = (np.abs(rel @ vecs[:, 1]) < NOISE_CELL_M / 2) & (np.abs(rel @ vecs[:, 2]) < NOISE_CELL_M / 2) & (np.abs(across) < GATHER_M)
        p = q[inside]
        c = p.mean(axis=0)
        n = np.linalg.eigh(np.cov((p - c).T))[1][:, 0]
        r = (p - c) @ n
        dist = float(np.linalg.norm(c))
        out.append({"c": c, "n": n if n @ c <= 0 else -n, "nz": nz,
                    "sigma": float(1.4826 * np.median(np.abs(r - np.median(r)))), "dist": dist,
                    "cos": abs(float(n @ (c / max(dist, 1e-6)))), "count": int(e - s)})
    return out


def keep_sheets(patches: list[dict]) -> list[dict]:
    """Only patches on a large flat sheet: grouped by agreeing plane, groups under SHEET_MIN patches dropped."""
    if not patches:
        return []
    cos_tol = np.cos(np.radians(SHEET_DEG))
    normals = np.array([p["n"] for p in patches])
    centres = np.array([p["c"] for p in patches])
    label = np.full(len(patches), -1)
    keep = []
    for i in range(len(patches)):
        if label[i] >= 0:
            continue
        members = (label < 0) & (normals @ normals[i] > cos_tol) & (np.abs((centres - centres[i]) @ normals[i]) < SHEET_M)
        label[members] = i
        if members.sum() >= SHEET_MIN:
            keep.extend(np.flatnonzero(members).tolist())
    return [patches[i] for i in sorted(keep)]


_NOISE: dict = {}       # (scan name, point count) -> median flat-sheet scatter (m)


def scan_noise(name: str, raw: np.ndarray, level: np.ndarray) -> float | None:
    """Median scatter of one scan's flat-sheet patches (m), or None when it has fewer than 10. Cached per scan."""
    key = (name, len(raw))
    if key not in _NOISE:
        sheets = keep_sheets(scan_patches(raw, level))
        _NOISE[key] = float(np.median([p["sigma"] for p in sheets])) if len(sheets) >= 10 else None
    return _NOISE[key]


NOISE_REF_M = 0.01      # a pair of scans this noisy each keeps its trust unchanged
WEIGHT_RANGE = (0.25, 4.0)


def pair_weight(noise_a: float | None, noise_b: float | None) -> float:
    """How much more (or less) a link between two scans counts in the final solve: (reference / their noise)^2."""
    if noise_a is None or noise_b is None:
        return 1.0
    sigma = float(np.hypot(noise_a, noise_b) / np.sqrt(2.0))
    return float(np.clip((NOISE_REF_M / sigma) ** 2, *WEIGHT_RANGE))
