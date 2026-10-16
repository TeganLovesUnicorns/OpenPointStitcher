"""A scan's sensor noise in metres, from surfaces that are flat in real life: 0.5 m patches on a large flat sheet
(walls, floors, ceilings), scatter about each patch's own plane. Full resolution, read only."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

CELL_M = 0.5
GATHER_M = 0.08         # points this far either side of a patch's plane are its own when re-collected
MIN_POINTS = 60
PLANAR = 0.2            # a patch is surface when its thinnest spread is under this share of the next
SHEET_DEG = 3.0         # patches whose planes agree this well in angle ...
SHEET_M = 0.03          # ... and offset belong to one sheet
SHEET_MIN = 6           # a sheet needs this many patches (1.5 m2) to count as a flat surface
SLOPED = (0.3, 0.9)     # |normal z| in this band is neither wall nor floor: left out (raked seating, ramps)


def scan_patches(raw: np.ndarray, level: np.ndarray) -> list[dict]:
    """Every planar 0.5 m patch of one scan, in its levelled frame (origin = where the scan started)."""
    pts = raw @ level.T
    keys = np.floor(pts / CELL_M).astype(np.int64)
    order = np.lexsort(keys.T)
    keys, pts_s = keys[order], pts[order]
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    ends = np.r_[starts[1:], len(pts_s)]
    tree = cKDTree(pts_s)
    out = []
    for s, e in zip(starts, ends):
        if e - s < MIN_POINTS:
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
        q = pts_s[tree.query_ball_point(c, CELL_M)]
        rel = q - c
        across = rel @ n
        inside = (np.abs(rel @ vecs[:, 1]) < CELL_M / 2) & (np.abs(rel @ vecs[:, 2]) < CELL_M / 2) & (np.abs(across) < GATHER_M)
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
