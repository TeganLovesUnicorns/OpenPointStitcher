"""Openings in walls: doors, windows and any other gap bounded by wall, found looking straight at each wall. An
opening is the same physical thing seen from two rooms, so matching openings pins a position along a wall, which two
wall planes alone cannot. No standard size is assumed."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .find_walls import Wall

log = logging.getLogger(__name__)


@dataclass
class Opening:
    kind: str                       # "door", "window", or whatever a registered detector names
    wall_index: int
    along: float                    # centre of the opening, measured along the wall from its first point
    width: float
    height: float                   # top minus sill
    sill: float                     # height of the opening's base above the floor
    centre: np.ndarray              # 3D centre in the levelled frame (z is absolute, floor_z + height above floor)
    normal: np.ndarray              # the wall's unit normal in the levelled XY plane
    clear_fraction: float           # how empty the opening actually is, 0..1
    bounded: float                  # how much wall sits around it (above, and below a window), 0..1
    # Set by scan_openings, in the scan's RAW frame, which is what the vote compares:
    position: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None    # unit vector along the wall
    wall_normal: Optional[np.ndarray] = None  # unit vector across the wall
    cohesion: float = 0.0                     # clear_fraction * bounded

    def describe(self) -> str:
        return (f"{self.kind} on W{self.wall_index}: {self.width:.2f} m wide x {self.height:.2f} m tall, "
                f"{self.along:.2f} m along the wall, sill {self.sill:.2f} m, clear {self.clear_fraction:.2f}, "
                f"bounded {self.bounded:.2f}")


# A detector looks at one wall face-on and returns what it finds. New kinds (alcoves, columns) are added by
# registering another detector.
Detector = Callable[[np.ndarray, Wall, float], "list[Opening]"]
DETECTORS: dict[str, Detector] = {}


def register_detector(name: str, fn: Detector) -> None:
    DETECTORS[name] = fn


def _runs(mask: np.ndarray):
    """(first, last) index of every run of True in a 1-D boolean array."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(idx[0::2], idx[1::2] - 1))


def find_wall_openings(
    points_levelled: np.ndarray,
    wall: Wall,
    floor_z: float,
    slab: float = 0.20,
    cell: float = 0.05,
    min_width: float = 0.50,
    max_width: float = 3.00,
    min_door_height: float = 1.60,
    min_window_height: float = 0.40,
    max_top: float = 3.20,
    min_clear: float = 0.90,
    min_bounded: float = 0.25,
) -> list[Opening]:
    """Doors and windows in one wall, seen face on: a run of columns empty over a band of heights, with wall above it,
    wall on both sides, and for a window wall below it too. A door's band starts at the floor. Ranges only reject nonsense."""
    d = points_levelled[:, :2] @ wall.normal - wall.offset
    near = np.abs(d) < slab
    if near.sum() < 200:
        return []
    along_v = np.array([-wall.normal[1], wall.normal[0]])
    a = points_levelled[near][:, :2] @ along_v
    h = points_levelled[near][:, 2] - floor_z
    a0 = a.min()
    keep = (h > -0.10) & (h < max_top + 1.0)
    a, h = a[keep], h[keep]
    if a.size < 200:
        return []

    n_cols = max(int(np.ceil((a.max() - a0) / cell)), 4)
    n_rows = int(np.ceil((max_top + 1.0) / cell))
    grid = np.zeros((n_rows, n_cols), dtype=bool)
    grid[np.clip((np.clip(h, 0, None) / cell).astype(int), 0, n_rows - 1),
         np.clip(((a - a0) / cell).astype(int), 0, n_cols - 1)] = True
    col_any = grid.any(axis=0)
    if not col_any.any():
        return []
    first, last = np.flatnonzero(col_any)[[0, -1]]
    cum = np.vstack([np.zeros((1, n_cols)), np.cumsum(grid, axis=0)])   # cum[r]: occupied cells in rows below r

    def band(lo_m: float, hi_m: float) -> np.ndarray:
        """Per column, the share of occupied cells between two heights (metres above the floor)."""
        lo, hi = max(int(round(lo_m / cell)), 0), min(int(round(hi_m / cell)), n_rows)
        return (cum[hi] - cum[lo]) / max(hi - lo, 1)

    found: list[tuple[int, int, float, float, float, float, float]] = []
    sills = [0.0] + [round(s, 2) for s in np.arange(0.30, 1.31, 0.10)]
    for sill in sills:
        lowest = min_door_height if sill == 0.0 else min_window_height
        for top in np.arange(sill + lowest, max_top + 1e-9, 0.10):
            inside = band(sill, top)
            clear_cols = inside < 0.10
            above = band(top, top + 0.40)
            below = band(max(sill - 0.40, 0.0), sill) if sill > 0.0 else np.ones(n_cols)
            for i, j in _runs(clear_cols):
                width = (j - i + 1) * cell
                # A run touching either end is where the wall stops, not an opening in it.
                if not (min_width <= width <= max_width and i > first and j < last):
                    continue
                clear = 1.0 - float(inside[i:j + 1].mean())
                head = float(above[i:j + 1].mean())
                foot = float(below[i:j + 1].mean())
                side = np.concatenate([inside[max(i - 3, 0):i], inside[j + 1:j + 4]])
                if clear >= min_clear and head >= min_bounded and foot >= min_bounded and side.size and side.mean() >= min_bounded:
                    found.append((i, j, sill, float(top), clear, min(head, foot), width))

    # Keep the biggest, best-bounded reading of each opening: the same gap is found at several sills and tops.
    found.sort(key=lambda f: (-(f[3] - f[2]) * f[6], -f[5]))
    kept: list[tuple] = []
    for f in found:
        span = f[1] - f[0] + 1
        if all(min(f[1], k[1]) - max(f[0], k[0]) + 1 < 0.6 * min(span, k[1] - k[0] + 1) for k in kept):
            kept.append(f)

    out = []
    for i, j, sill, top, clear, bounded, width in kept:
        along = a0 + (i + j + 1) / 2.0 * cell
        xy = wall.normal * wall.offset + along_v * along
        height = top - sill
        out.append(Opening(
            kind="door" if sill == 0.0 else "window", wall_index=wall.index, along=float(along - a0), width=float(width),
            height=float(height), sill=float(sill), centre=np.array([xy[0], xy[1], floor_z + sill + height / 2.0]),
            normal=np.asarray(wall.normal, dtype=float), clear_fraction=float(clear), bounded=float(bounded)))
    return out


register_detector("doors and windows", find_wall_openings)


def find_openings(points_levelled: np.ndarray, walls: list[Wall], floor_z: float) -> list[Opening]:
    """Everything every registered detector finds, in every wall."""
    out: list[Opening] = []
    for wall in walls:
        for fn in DETECTORS.values():
            out.extend(fn(points_levelled, wall, floor_z))
    return out


def scan_openings(points_raw: np.ndarray, level_rotation: np.ndarray, floor_z: float, walls: list[Wall]) -> list[Opening]:
    """One scan's openings, each also placed in the scan's RAW frame (position, and the directions along and across its wall)."""
    rot = np.asarray(level_rotation, dtype=float)
    found = find_openings(np.asarray(points_raw, dtype=float) @ rot.T, walls, floor_z)
    for o in found:
        o.position = rot.T @ o.centre
        o.direction = rot.T @ np.array([-o.normal[1], o.normal[0], 0.0])
        o.wall_normal = rot.T @ np.array([o.normal[0], o.normal[1], 0.0])
        o.cohesion = o.clear_fraction * o.bounded
    return found


def build_scan_openings(features, points_raw: np.ndarray, down, level_rotation: np.ndarray, floor_z: float) -> None:
    """Fill features.openings for one scan. down is its voxel-downsampled cloud (walls are found on that, openings on
    every raw point). Never raises: a scan whose openings cannot be found just has none."""
    from .find_walls import extract_walls
    try:
        walls = extract_walls(down, level_rotation=np.asarray(level_rotation, dtype=float))
        features.openings = scan_openings(points_raw, level_rotation, floor_z, walls)
    except Exception:  # noqa: BLE001 - a missing voice must never stop an align
        log.exception("could not find openings for one scan; it will not vote")
        features.openings = []
