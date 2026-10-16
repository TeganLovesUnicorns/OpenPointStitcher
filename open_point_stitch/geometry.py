"""Small geometric measurements shared by more than one detector. The abs and no-abs angle forms have separate
names on purpose: mixing them up is a real bug."""

from __future__ import annotations

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two unit vectors, 0-180. Direction matters: two
    normals pointing opposite ways are 180 apart, not 0."""
    return float(np.degrees(np.arccos(np.clip(float(np.asarray(a) @ np.asarray(b)), -1.0, 1.0))))


def fold_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two unit vectors ignoring direction, 0-90. Right for "are these
    planes parallel", wrong for describing a corner (use `angle_between`)."""
    return float(np.degrees(np.arccos(
        np.clip(abs(float(np.asarray(a) @ np.asarray(b))), -1.0, 1.0))))


def tilt_off_horizontal(normal: np.ndarray) -> float:
    """0 for a floor or ceiling, 90 for a plumb wall."""
    return fold_between(normal, UP)


def cohesion_from_slack(slack_m: float, tolerance_m: float) -> float:
    """1.0 inside every footprint, falling to 0 at `tolerance_m` outside. The single
    definition behind `vote_corner_min_cohesion` and `vote_edge_min_cohesion`."""
    return float(np.clip(1.0 - slack_m / tolerance_m, 0.0, 1.0))


# A region built from one clicked point has zero extent; give it a real margin so it catches any points at all.
MIN_HALF_EXTENT_M = 0.05


def points_in_box(points: np.ndarray, centroid, extent) -> np.ndarray:
    """Boolean mask of the points inside a hand-drawn box: `centroid` plus or minus half of `extent`
    on each axis, never narrower than `MIN_HALF_EXTENT_M`. Axis-aligned in whatever frame the points
    and the box are both in. The one definition behind exclusions and anchor regions."""
    half = np.maximum(np.asarray(extent, dtype=float) / 2.0, MIN_HALF_EXTENT_M)
    return np.all(np.abs(points - np.asarray(centroid, dtype=float)) <= half, axis=1)
