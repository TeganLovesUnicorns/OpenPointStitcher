"""Floors and ceilings as pieces: several floor heights and flat, dropped and sloped ceiling sections in one scan.
Every point is used; nothing is thinned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d

from .lib.clouds import as_cloud
from .lib import geometry

CELL_M = 0.05             # plan grid: each 5 cm cell gets at most one floor and one ceiling height
BIN_M = 0.02              # height step used to find the lowest and highest surface in a cell
MIN_BIN_POINTS = 3        # a height step needs this many points to count as a surface
FLAT_NZ = 0.5             # a point is on a floor or ceiling if its surface is within 60 deg of level
MIN_GAP_M = 1.5           # a cell's lowest and highest surfaces this far apart are its floor and its ceiling
PLANE_TOL_M = 0.03        # a cell belongs to a piece when its height is this close to the piece's plane
MIN_PIECE_M2 = 0.5        # pieces smaller than this are furniture or noise, not architecture
SLOPED_DEG = 3.0          # a piece tilted more than this is reported as sloped
MAX_SLOPE_DEG = 60.0      # steeper than this is a wall, not a ceiling
RANSAC_TRIES = 400
NORMAL_RADIUS_M = 0.08
MAX_PIECES = 30
MID_GRID_M = 1.0          # how finely the floor-to-ceiling middle is mapped
FALLBACK_MID_M = 1.5      # the middle when no cell saw both a floor and a ceiling
TOUCH_CELLS = 2           # pieces this many cells apart or closer touch
MERGE_ANGLE_DEG = 3.0     # touching pieces tilted within this of each other, with no step, are one surface
MERGE_RMS_GROWTH = 1.25   # ... unless the joined plane fits their points worse than this times the worse part
SURFACE_MAX_M2 = 2.0      # a raised floor-like piece smaller than this is a counter, table or step, not a floor
SURFACE_MIN_RISE_M = 0.3  # ... when it sits at least this far above the main floor
COARSE_M = 0.20           # second look for drops whose underside the scan barely saw
DROP_MIN_M, DROP_MAX_M = 0.10, 0.80     # how far below the ceiling beside it such a drop sits
DROP_MIN_HEIGHT_M = 1.9   # ... and above head height, so a bookshelf top is not taken for one
DROP_REACH = 5            # ... within this many 20 cm squares of that ceiling
DROP_MIN_M2 = 0.3
DROP_FIT_SPAN_M = 1.2     # a drop at least this wide both ways must also be level (narrower: wall tops tilt it) ...
DROP_MAX_TILT_DEG = 5.0   # ... to within this, judged on its 20 cm squares


@dataclass
class Piece:
    """One floor or ceiling section, in the scan's levelled frame (heights above its detected floor)."""

    kind: str                 # "floor", "ceiling", or "surface" (a counter, table or step: flat, not a floor)
    normal: np.ndarray        # unit, pointing up
    offset: float             # plane: normal . p = offset
    slope_deg: float
    height_m: float           # plane height at the piece's middle
    z_min: float
    z_max: float
    area_m2: float
    n_points: int             # full-resolution points on the piece
    rms_m: float              # how far those points sit from the plane: the piece's own measuring error
    centroid: np.ndarray
    outlines: list = field(default_factory=list)   # plan polygons, each an (n, 2) array
    label: str = ""
    cells: np.ndarray | None = field(default=None, repr=False)   # its grid cells; working data, not saved
    sparse: bool = False      # found on the second, coarse look: the scan saw few of its points

    @property
    def sloped(self) -> bool:
        return self.slope_deg > SLOPED_DEG

    def z_at(self, xy: np.ndarray) -> np.ndarray:
        """The plane's height above the given plan points."""
        n = self.normal
        return (self.offset - xy[:, 0] * n[0] - xy[:, 1] * n[1]) / n[2]

    def to_json(self) -> dict:
        return {
            "kind": self.kind, "label": self.label,
            "normal": [round(float(v), 6) for v in self.normal], "offset": round(float(self.offset), 4),
            "slope_deg": round(float(self.slope_deg), 2), "h": round(float(self.height_m), 3),
            "z_min": round(float(self.z_min), 3), "z_max": round(float(self.z_max), 3),
            "area_m2": round(float(self.area_m2), 2), "n_points": int(self.n_points),
            "rms_m": round(float(self.rms_m), 4), "sparse": bool(self.sparse),
            "outlines": [[[round(float(x), 3), round(float(y), 3)] for x, y in o] for o in self.outlines],
        }


def _flat_mask(points: np.ndarray) -> np.ndarray:
    """Points whose local surface is within 60 deg of level. Normal sign does not matter here."""
    pcd = as_cloud(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
    return np.abs(np.asarray(pcd.normals)[:, 2]) > FLAT_NZ


def _cell_samples(xyz: np.ndarray, origin: np.ndarray, cell_m: float = CELL_M):
    """Per occupied cell: its lowest and highest surface height. Returns (cells (n, 2) int, low z, high z)."""
    ij = np.floor((xyz[:, :2] - origin) / cell_m).astype(np.int64)
    width = int(ij[:, 0].max()) + 1
    zb = np.floor(xyz[:, 2] / BIN_M).astype(np.int64)
    zb -= zb.min()
    nb = int(zb.max()) + 1
    key = (ij[:, 1] * width + ij[:, 0]) * nb + zb
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    zsum = np.bincount(inv, weights=xyz[:, 2])
    near = counts.copy()                                         # a height step and its two neighbours together,
    nxt = (uniq[1:] == uniq[:-1] + 1) & (uniq[1:] // nb == uniq[:-1] // nb)   # so noise across a step edge counts
    near[:-1][nxt] += counts[1:][nxt]
    near[1:][nxt] += counts[:-1][nxt]
    keep = near >= MIN_BIN_POINTS
    uniq, zmean = uniq[keep], zsum[keep] / counts[keep]
    cell = uniq // nb
    starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])    # rows are sorted by cell, then height
    ends = np.r_[starts[1:], len(cell)] - 1
    cells = np.column_stack([cell[starts] % width, cell[starts] // width])
    return cells, zmean[starts], zmean[ends]


def _fit_plane(xyz: np.ndarray):
    """Least-squares plane through the points; normal pointing up. Returns (normal, offset, rms)."""
    c = xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(xyz - c, full_matrices=False)
    n = vh[2] if vh[2, 2] >= 0 else -vh[2]
    d = (xyz - c) @ n
    return n, float(n @ c), float(np.sqrt(np.mean(d * d)))


def _largest_component(cells: np.ndarray) -> np.ndarray:
    """Index (into `cells`) of the biggest 8-connected group of grid cells."""
    lo = cells.min(axis=0)
    rc = cells - lo
    grid = np.zeros((int(rc[:, 1].max()) + 1, int(rc[:, 0].max()) + 1), np.uint8)
    grid[rc[:, 1], rc[:, 0]] = 1
    _, labels = cv2.connectedComponents(grid, connectivity=8)
    lab = labels[rc[:, 1], rc[:, 0]]
    best = np.bincount(lab).argmax()
    return np.flatnonzero(lab == best)


def _ransac_plane(sxyz: np.ndarray, rng: np.random.Generator):
    """The plane, within 60 deg of level, that the most cell samples sit on."""
    tri = sxyz[rng.integers(0, len(sxyz), size=(RANSAC_TRIES, 3))]          # all guesses at once
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n, tri = n[ok] / norm[ok, None], tri[ok]
    n *= np.where(n[:, 2] < 0, -1.0, 1.0)[:, None]                         # pointing up
    ok = n[:, 2] >= np.cos(np.radians(MAX_SLOPE_DEG))
    n, a = n[ok], tri[ok][:, 0]
    if len(n) == 0:
        return None, 0.0
    off = np.einsum("ij,ij->i", n, a)
    counts = np.zeros(len(n), np.int64)
    for s in range(0, len(sxyz), 20_000):                                  # in slices, to bound memory
        counts += (np.abs(sxyz[s:s + 20_000] @ n.T - off) < PLANE_TOL_M).sum(axis=0)
    best = int(np.argmax(counts))
    return n[best], float(off[best])


def _outlines(cells: np.ndarray, origin: np.ndarray, cell_m: float = CELL_M) -> tuple[list[np.ndarray], float]:
    """Plan polygons around a set of grid cells, straightened to within one cell, and the area they cover
    (holes left out), square metres."""
    lo = cells.min(axis=0) - 1
    rc = cells - lo
    grid = np.zeros((int(rc[:, 1].max()) + 2, int(rc[:, 0].max()) + 2), np.uint8)
    grid[rc[:, 1], rc[:, 0]] = 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    found, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in found:
        c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
        if len(c) >= 3 and cv2.contourArea(c.astype(np.float32)) * cell_m ** 2 >= MIN_PIECE_M2 / 2:
            out.append(origin + (c + lo + 0.5) * cell_m)
    return out, float(grid.sum()) * cell_m ** 2


def _mid_heights(cells: np.ndarray, mid: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Halfway between floor and ceiling, from the cells that saw both, read off at the cells `at`.
    Smoothed on a 1 m grid; a 1 m square with none takes its nearest neighbour's."""
    k = int(round(MID_GRID_M / CELL_M))
    lo = np.minimum(cells.min(axis=0), at.min(axis=0)) // k
    hi = np.maximum(cells.max(axis=0), at.max(axis=0)) // k
    shape = (int(hi[1] - lo[1]) + 1, int(hi[0] - lo[0]) + 1)
    g = cells // k - lo
    total = np.zeros(shape)
    count = np.zeros(shape)
    np.add.at(total, (g[:, 1], g[:, 0]), mid)
    np.add.at(count, (g[:, 1], g[:, 0]), 1)
    grid = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    empty = np.isnan(grid).astype(np.uint8)
    if empty.any():
        _, labels = cv2.distanceTransformWithLabels(empty, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        rows, cols = np.nonzero(empty == 0)
        filled = np.zeros(int(labels.max()) + 1)
        filled[labels[rows, cols]] = grid[rows, cols]
        grid = filled[labels]
    a = at // k - lo
    return grid[a[:, 1], a[:, 0]]


def _pieces_of_kind(kind: str, cells: np.ndarray, z: np.ndarray, flat_pts: np.ndarray, flat_cell_key: np.ndarray,
                    width: int, origin: np.ndarray, rng: np.random.Generator) -> list[Piece]:
    """Peel planes off this kind's cell samples, largest first; each piece is one connected area."""
    pieces: list[Piece] = []
    sxyz = np.column_stack([origin + (cells + 0.5) * CELL_M, z])
    pool = np.ones(len(sxyz), bool)
    min_cells = int(MIN_PIECE_M2 / CELL_M ** 2)
    while pool.sum() >= min_cells and len(pieces) < MAX_PIECES:
        idx = np.flatnonzero(pool)
        n, off = _ransac_plane(sxyz[idx], rng)
        if n is None:
            break
        on = idx[np.abs(sxyz[idx] @ n - off) < PLANE_TOL_M]
        if len(on) < 3:
            break
        part = on[_largest_component(cells[on])]
        if len(part) < min_cells:                                # the plane's biggest patch is too small: done with it
            pool[part] = False
            continue
        n, off, _ = _fit_plane(sxyz[part])                       # refit on the connected part, then gather again
        on = idx[np.abs(sxyz[idx] @ n - off) < PLANE_TOL_M]
        part = on[_largest_component(cells[on])]
        pool[part] = False
        if len(part) < min_cells:
            continue
        piece = _measure(kind, cells[part], [(n, off)], flat_pts, flat_cell_key, width, origin)
        if piece is not None:
            pieces.append(piece)
    return _merge_touching(pieces, flat_pts, flat_cell_key, width, origin)


def _measure(kind, piece_cells, planes, flat_pts, flat_cell_key, width, origin) -> Piece | None:
    """Refit the piece's plane on every full-resolution point in its cells that sits on one of `planes`."""
    keys = piece_cells[:, 1] * width + piece_cells[:, 0]
    pts = flat_pts[np.isin(flat_cell_key, keys)]
    near = np.zeros(len(pts), bool)
    for n, off in planes:
        near |= np.abs(pts @ n - off) < PLANE_TOL_M
    pts = pts[near]
    if len(pts) < 10:
        return None
    n, off, rms = _fit_plane(pts)
    centroid = pts.mean(axis=0)
    outlines, area = _outlines(piece_cells, origin)
    if not outlines:
        return None
    return Piece(kind=kind, normal=n, offset=off, slope_deg=geometry.angle_between(n, np.array([0.0, 0.0, 1.0])),
                 height_m=float(centroid[2]), z_min=float(np.percentile(pts[:, 2], 1)),
                 z_max=float(np.percentile(pts[:, 2], 99)), area_m2=area,
                 n_points=int(len(pts)), rms_m=rms, centroid=centroid, outlines=outlines, cells=piece_cells)


def _touching_gap(a: Piece, b: Piece, origin: np.ndarray) -> float | None:
    """How far apart the two planes are where the pieces meet (metres), or None if they do not touch."""
    r = TOUCH_CELLS
    near = {(int(x) + dx, int(y) + dy) for x, y in a.cells for dx in range(-r, r + 1) for dy in range(-r, r + 1)}
    meet = np.array([c for c in b.cells.tolist() if (c[0], c[1]) in near])
    if len(meet) == 0:
        return None
    xy = origin + (meet + 0.5) * CELL_M
    return float(np.median(np.abs(a.z_at(xy) - b.z_at(xy))))


def _merge_touching(pieces: list[Piece], flat_pts, flat_cell_key, width, origin) -> list[Piece]:
    """One piece where the plane search split a single surface: same tilt, touching, no step between them."""
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(pieces):
            for j in range(i + 1, len(pieces)):
                b = pieces[j]
                if geometry.angle_between(a.normal, b.normal) > MERGE_ANGLE_DEG:
                    continue
                gap = _touching_gap(a, b, origin)
                if gap is None or gap > PLANE_TOL_M:
                    continue
                both = _measure(a.kind, np.concatenate([a.cells, b.cells]), [(a.normal, a.offset),
                                (b.normal, b.offset)], flat_pts, flat_cell_key, width, origin)
                if both is None or both.rms_m > max(a.rms_m, b.rms_m) * MERGE_RMS_GROWTH + 0.002:
                    continue                                     # joined, it would be less flat than its parts
                pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [both]
                merged = True
                break
            if merged:
                break
    return pieces


def _m(v: float) -> str:
    """Metres to the centimetre, never "-0.00"."""
    return f"{v + 0.0:.2f}".replace("-0.00", "0.00")


def _label(pieces: list[Piece]) -> None:
    """Plain words for each piece, relative to the biggest flat floor and ceiling. Small raised floor-like pieces
    become "surface" first."""
    floors = [p for p in pieces if p.kind == "floor" and not p.sloped]
    main_floor = max(floors, key=lambda p: p.area_m2) if floors else None
    for p in pieces:
        if (p.kind == "floor" and main_floor is not None and p is not main_floor and p.area_m2 < SURFACE_MAX_M2
                and p.height_m - main_floor.height_m > SURFACE_MIN_RISE_M):
            p.kind = "surface"
            p.label = f"flat surface {_m(p.height_m)} m (a counter, table, step or landing)"
    for kind in ("floor", "ceiling"):
        flat = [p for p in pieces if p.kind == kind and not p.sloped]
        main = max(flat, key=lambda p: p.area_m2) if flat else None
        for p in (q for q in pieces if q.kind == kind):
            if p.sloped:
                p.label = f"sloped {kind}, {p.slope_deg:.0f} deg, {_m(p.z_min)} to {_m(p.z_max)} m"
            elif p is main or main is None or abs(p.height_m - main.height_m) < 0.05:
                p.label = f"{kind} {_m(p.height_m)} m"
            else:
                d = (p.height_m - main.height_m) * 100
                if kind == "ceiling":
                    word = "lowered ceiling" if d < 0 else "raised ceiling"
                else:
                    word = "raised floor" if d > 0 else "lower floor"
                p.label = f"{word} {_m(p.height_m)} m ({abs(d):.0f} cm {'below' if d < 0 else 'above'} the main {kind})"
            if p.sparse:
                p.label += ", from few points: the scan barely saw its underside"


def _sparse_drops(pts: np.ndarray, ceils: list[Piece]) -> list[Piece]:
    """Dropped ceiling sections whose underside the scan barely saw (a bulkhead along a wall): a second, coarser look
    at every point, not only flat ones. A 20 cm square under a drop has its highest points on the drop's underside.
    Only kept beside a found ceiling, 10 to 80 cm below it and above head height."""
    if not ceils:
        return []
    origin = pts[:, :2].min(axis=0)
    cells, _, top = _cell_samples(pts, origin, COARSE_M)
    xy = origin + (cells + 0.5) * COARSE_M
    rows = [c.to_json() for c in ceils]
    seen = on_pieces(xy, top, rows, band_m=2 * PLANE_TOL_M, kinds=("ceiling",))      # already a found ceiling
    lo = cells.min(axis=0)
    shape = tuple((cells.max(axis=0) - lo + 1)[::-1])
    above = np.full(len(cells), np.nan)                     # the found ceiling over each square, from nearby pieces
    nears = []
    for c in ceils:
        grid = np.zeros(shape, np.uint8)
        for o in c.outlines:
            cv2.fillPoly(grid, [np.round((o - origin) / COARSE_M - 0.5 - lo).astype(np.int32)], 1)
        near = cv2.dilate(grid, np.ones((2 * DROP_REACH + 1,) * 2, np.uint8))[cells[:, 1] - lo[1], cells[:, 0] - lo[0]] > 0
        nears.append(near)
        z = c.z_at(xy)
        above = np.where(near & (np.isnan(above) | (z > above)), z, above)
    drop = above - top
    cand = ~seen & (top > DROP_MIN_HEIGHT_M) & (drop >= DROP_MIN_M) & (drop <= DROP_MAX_M)
    if cand.sum() < 3:
        return []
    out = []
    idx = np.flatnonzero(cand)
    rc = cells[idx] - lo
    grid = np.zeros(shape, np.uint8)
    grid[rc[:, 1], rc[:, 0]] = 1
    n_lab, lab = cv2.connectedComponents(grid, connectivity=8)
    ij = np.floor((pts[:, :2] - origin) / COARSE_M).astype(np.int64)
    for k in range(1, n_lab):
        part = idx[lab[rc[:, 1], rc[:, 0]] == k]
        if len(part) * COARSE_M ** 2 < DROP_MIN_M2:
            continue
        tops = dict(zip(map(tuple, cells[part]), top[part]))
        mine = np.array([tuple(c) in tops for c in map(tuple, ij)])     # points in this drop's squares
        sel = pts[mine]
        under = np.abs(sel[:, 2] - np.array([tops[tuple(c)] for c in ij[mine]])) < PLANE_TOL_M
        if under.sum() < 10:
            continue
        # Level, at the middle height: a fitted plane through a strip this narrow and sparse tilts on noise alone.
        u = sel[under]
        h = float(np.median(u[:, 2]))
        u = u[np.abs(u[:, 2] - h) < 2 * PLANE_TOL_M]               # the underside itself, not wall tops beside it
        if len(u) < 10:
            continue
        h = float(np.median(u[:, 2]))
        mid = xy[part].mean(axis=0)
        if any(near[part].any() and not c.sloped and abs(h - float(c.z_at(mid[None])[0])) < DROP_MIN_M
               for c, near in zip(ceils, nears)):
            continue                                        # level with a found ceiling it touches: more of that one
        c2 = xy[part] - xy[part].mean(axis=0)
        span = np.ptp(c2 @ np.linalg.svd(c2, full_matrices=False)[2].T, axis=0)   # along its own long and short ways
        if span.min() >= DROP_FIT_SPAN_M:                   # wide enough to tell: a tilted patch is roof, not a drop
            a = np.column_stack([xy[part], np.ones(len(part))])
            keep = np.ones(len(part), bool)
            for _ in range(3):                              # squares on the edge that see the ceiling beside it are set aside
                g = np.linalg.lstsq(a[keep], top[part][keep], rcond=None)[0]
                keep = np.abs(a @ g - top[part]) < 0.05
                if keep.sum() < 3:
                    break
            if np.degrees(np.arctan(np.hypot(g[0], g[1]))) > DROP_MAX_TILT_DEG:
                continue
        n, off, slope = np.array([0.0, 0.0, 1.0]), h, 0.0
        rms = float(np.sqrt(np.mean((u[:, 2] - h) ** 2)))
        outlines, area = _outlines(cells[part], origin, COARSE_M)
        if not outlines:
            continue
        out.append(Piece(kind="ceiling", normal=n, offset=off, slope_deg=slope, height_m=float(u[:, 2].mean()),
                         z_min=float(np.percentile(u[:, 2], 1)), z_max=float(np.percentile(u[:, 2], 99)),
                         area_m2=area, n_points=int(len(u)), rms_m=rms, centroid=u.mean(axis=0),
                         outlines=outlines, sparse=True))
    return out


def find_floors_and_ceilings(points_levelled: np.ndarray, seed: int = 7) -> list[Piece]:
    """Every floor and ceiling piece in one scan. `points_levelled`: the scan's full-resolution points, levelled,
    with its detected floor at height 0. Pieces are sorted floors first, then by area, largest first."""
    if len(points_levelled) < 100:
        return []
    flat = _flat_mask(points_levelled)
    fp = points_levelled[flat]
    if len(fp) < 100:
        return []
    origin = fp[:, :2].min(axis=0)
    cells, low, high = _cell_samples(fp, origin)
    ij = np.floor((fp[:, :2] - origin) / CELL_M).astype(np.int64)
    width = int(ij[:, 0].max()) + 1
    fkey = ij[:, 1] * width + ij[:, 0]
    two = (high - low) >= MIN_GAP_M                              # both a floor and a ceiling seen in the cell
    one = ~two
    if two.sum() >= 3:                                           # one surface only: floor if below the local middle
        is_floor = low[one] < _mid_heights(cells[two], (low[two] + high[two]) / 2, cells[one])
    else:
        is_floor = low[one] < FALLBACK_MID_M
    f_cells = np.concatenate([cells[two], cells[one][is_floor]])
    f_z = np.concatenate([low[two], low[one][is_floor]])
    c_cells = np.concatenate([cells[two], cells[one][~is_floor]])
    c_z = np.concatenate([high[two], high[one][~is_floor]])
    rng = np.random.default_rng(seed)
    floors = _pieces_of_kind("floor", f_cells, f_z, fp, fkey, width, origin, rng)
    ceils = _pieces_of_kind("ceiling", c_cells, c_z, fp, fkey, width, origin, rng)
    ceils += _sparse_drops(points_levelled, ceils)
    pieces = sorted(floors, key=lambda p: -p.area_m2) + sorted(ceils, key=lambda p: -p.area_m2)
    _label(pieces)
    order = {"floor": 0, "surface": 1, "ceiling": 2}
    return sorted(pieces, key=lambda p: (order[p.kind], -p.area_m2))


def on_pieces(xy: np.ndarray, h: np.ndarray, pieces: list[dict], band_m: float,
              kinds=("floor", "ceiling")) -> np.ndarray:
    """Which points (scan's levelled frame) lie inside a saved piece's outline and within `band_m` of its plane.
    `pieces` are to_json() rows; counters and tables ("surface") are left out by default."""
    keep = np.zeros(len(h), bool)
    for p in pieces:
        if p["kind"] not in kinds:
            continue
        n, off = np.asarray(p["normal"]), p["offset"]
        near = np.abs(h - (off - xy[:, 0] * n[0] - xy[:, 1] * n[1]) / n[2]) <= band_m
        idx = np.flatnonzero(near & ~keep)
        if not len(idx):
            continue
        polys = [np.asarray(o) for o in p["outlines"]]
        lo = np.min([q.min(axis=0) for q in polys], axis=0)
        hi = np.max([q.max(axis=0) for q in polys], axis=0)
        size = np.ceil((hi - lo) / CELL_M).astype(int) + 2
        grid = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(grid, [np.round((q - lo) / CELL_M).astype(np.int32) for q in polys], 1)
        ij = np.round((xy[idx] - lo) / CELL_M).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < size[0]) & (ij[:, 1] < size[1])
        inside = np.zeros(len(idx), bool)
        inside[ok] = grid[ij[ok, 1], ij[ok, 0]] > 0
        keep[idx[inside]] = True
    return keep
