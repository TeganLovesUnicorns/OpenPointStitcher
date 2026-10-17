"""Junctions between detected surfaces: corners (three planes) and edges (two planes along a line), derived from
find_surfaces.py's planes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .lib import geometry
from .find_surfaces import ScanFeatures, Surface


@dataclass
class Corner:
    """One three-plane intersection, in a scan's own local frame."""

    position: np.ndarray          # 3D point where the three planes meet
    surfaces: tuple[int, int, int]   # the three Surface.index values, fixed order
    angles_deg: tuple[float, float, float]  # angle between each pair of normals, folded to 0-90
    # The real fold. Never reaches 0 or 180: both mean the two planes are
    # parallel, which is one wall detected twice, not a corner. What
    # survives find_corners' own gate is 20-160.
    interior_angles_deg: tuple[float, float, float]
    cohesion: float               # 0 (scattered) - 1 (all three footprints reach the corner)
    hue: float                    # mean of the three surfaces' own hues, nan if all grey
    saturation: float
    value: float
    index: int = -1

    def describe(self) -> str:
        hue_s = f"{self.hue:.0f}deg" if not np.isnan(self.hue) else "grey"
        a = self.interior_angles_deg
        return (f"C{self.index}: surfaces {self.surfaces}, angles "
                f"{a[0]:.0f}/{a[1]:.0f}/{a[2]:.0f}deg, "
                f"cohesion {self.cohesion:.2f}, hue {hue_s} sat {self.saturation:.2f}")


def fold_between_surfaces(a: Surface, b: Surface) -> float:
    """The real fold between two surfaces, 0-180, found by turning each normal to face the other. The two-surface
    version of corner_folds."""
    na = np.asarray(a.normal, dtype=float)
    nb = np.asarray(b.normal, dtype=float)
    if float(na @ (b.centroid - a.centroid)) < 0:
        na = -na
    if float(nb @ (a.centroid - b.centroid)) < 0:
        nb = -nb
    return 180.0 - geometry.angle_between(na, nb)


def corner_folds(trio) -> tuple[float, float, float]:
    """The real fold (0-180) between each pair of the three surfaces; each normal faces the other two, never the rest
    of the scan."""
    normals = []
    for s, others in ((trio[0], trio[1:]), (trio[1], (trio[0], trio[2])), (trio[2], trio[:2])):
        ref = (others[0].centroid + others[1].centroid) / 2.0
        n = np.asarray(s.normal, dtype=float)
        normals.append(-n if float(n @ (ref - s.centroid)) < 0 else n)
    return tuple(
        180.0 - geometry.angle_between(a, b)
        for a, b in combinations(normals, 2)
    )


def find_corners(
    surfaces: list[Surface],
    cohesion_tolerance_m: float = 0.5,
    min_angle_deg: float = 20.0,
    merge_within_m: float = 0.15,
) -> list[Corner]:
    """Solve where each non-near-parallel triple of surfaces meets, scored by cohesion. min_angle_deg gates on the
    folded angle; interior_angles_deg records the real one. No neighbour prefilter, on purpose. Corners within
    merge_within_m are merged."""
    corners: list[Corner] = []
    for i, j, k in combinations(range(len(surfaces)), 3):
        si, sj, sk = surfaces[i], surfaces[j], surfaces[k]
        normals = (si.normal, sj.normal, sk.normal)
        angles = tuple(
            geometry.fold_between(a, b)
            for a, b in combinations(normals, 2)
        )
        if min(angles) < min_angle_deg:
            continue

        A = np.array(normals)
        b = np.array([si.offset, sj.offset, sk.offset])
        if abs(np.linalg.det(A)) < 1e-3:
            continue  # near-degenerate even after the angle gate above
        p = np.linalg.solve(A, b)

        slack = max(s.footprint_slack(p) for s in (si, sj, sk))
        cohesion = geometry.cohesion_from_slack(slack, cohesion_tolerance_m)
        if cohesion <= 0.0:
            continue

        hues = [s.hue for s in (si, sj, sk) if not np.isnan(s.hue)]
        hue = float(np.mean(hues)) if hues else float("nan")
        sat = float(np.mean([s.saturation for s in (si, sj, sk)]))
        val = float(np.mean([s.value for s in (si, sj, sk)]))
        corners.append(Corner(
            position=p, surfaces=(si.index, sj.index, sk.index),
            angles_deg=angles, interior_angles_deg=corner_folds((si, sj, sk)),
            cohesion=cohesion, hue=hue, saturation=sat, value=val,
        ))

    corners.sort(key=lambda c: -c.cohesion)
    kept: list[Corner] = []
    for c in corners:
        # Best-first, so the survivor of a cluster is its strongest member.
        if any(float(np.linalg.norm(c.position - k.position)) < merge_within_m
               for k in kept):
            continue
        kept.append(c)
    for idx, c in enumerate(kept):
        c.index = idx
    return kept


@dataclass
class Edge:
    """Where two surfaces meet along a line, in a scan's own local frame,
    cut down to the stretch both surfaces actually measured."""

    midpoint: np.ndarray          # centre of that shared stretch
    direction: np.ndarray         # unit vector along the line, sign-canonical
    length_m: float               # how long the shared stretch is
    surfaces: tuple[int, int]     # the two Surface.index values
    interior_angle_deg: float     # the real fold between them, 20-160
    cohesion: float               # 0 (the line misses both) - 1 (it runs through both)
    hue: float                    # mean of the two surfaces' hues, nan if both grey
    saturation: float
    value: float
    index: int = -1

    def describe(self) -> str:
        hue_s = f"{self.hue:.0f}deg" if not np.isnan(self.hue) else "grey"
        return (f"E{self.index}: surfaces {self.surfaces}, {self.length_m:.1f} m long, "
                f"fold {self.interior_angle_deg:.0f}deg, cohesion {self.cohesion:.2f}, "
                f"hue {hue_s} sat {self.saturation:.2f}")


def _canonical_direction(d: np.ndarray) -> np.ndarray:
    """One of the two opposite directions along a line, chosen the same
    way every time - the cross product's sign follows the surfaces' own
    arbitrary normal signs and means nothing."""
    i = int(np.argmax(np.abs(d)))
    return -d if d[i] < 0 else d


def _extent_along(surface: Surface, d: np.ndarray) -> float:
    """Half the footprint's reach in direction `d`."""
    return (abs(float(d @ surface.u_axis)) * surface.extent_u / 2
            + abs(float(d @ surface.v_axis)) * surface.extent_v / 2)


def find_edges(
    surfaces: list[Surface],
    cohesion_tolerance_m: float = 0.5,
    min_angle_deg: float = 20.0,
    min_length_m: float = 0.5,
    merge_within_m: float = 0.15,
) -> list[Edge]:
    """Junction lines between non-parallel surface pairs, cut to the stretch both footprints reach (shorter than
    min_length_m is dropped). Same gates as find_corners."""
    edges: list[Edge] = []
    for si, sj in combinations(surfaces, 2):
        d = np.cross(si.normal, sj.normal)
        d_norm = float(np.linalg.norm(d))
        # |sin| between the normals: the same near-parallel test the corner
        # search makes, and equally the condition number of the line solve.
        if np.degrees(np.arcsin(np.clip(d_norm, 0.0, 1.0))) < min_angle_deg:
            continue
        d = _canonical_direction(d / d_norm)

        # The point on the line closest to the origin: two plane equations
        # plus "no travel along the line" is a well-conditioned 3x3.
        p0 = np.linalg.solve(np.array([si.normal, sj.normal, d]),
                             np.array([si.offset, sj.offset, 0.0]))

        # Where each footprint reaches along the line, as an interval, then
        # the stretch they share.
        spans = []
        for s in (si, sj):
            centre = float((s.centroid - p0) @ d)
            half = _extent_along(s, d)
            spans.append((centre - half, centre + half))
        lo = max(spans[0][0], spans[1][0])
        hi = min(spans[0][1], spans[1][1])
        if hi - lo < min_length_m:
            continue

        # The line has to run THROUGH both footprints, not merely past
        # them: sampled at both ends and the middle, same slack measure the
        # corner search uses.
        samples = [p0 + t * d for t in (lo, (lo + hi) / 2, hi)]
        slack = max(s.footprint_slack(p) for s in (si, sj) for p in samples)
        cohesion = geometry.cohesion_from_slack(slack, cohesion_tolerance_m)
        if cohesion <= 0.0:
            continue

        hues = [s.hue for s in (si, sj) if not np.isnan(s.hue)]
        edges.append(Edge(
            midpoint=p0 + ((lo + hi) / 2) * d, direction=d, length_m=float(hi - lo),
            surfaces=(si.index, sj.index),
            interior_angle_deg=fold_between_surfaces(si, sj),
            cohesion=cohesion,
            hue=float(np.mean(hues)) if hues else float("nan"),
            saturation=float(np.mean([s.saturation for s in (si, sj)])),
            value=float(np.mean([s.value for s in (si, sj)])),
        ))

    edges.sort(key=lambda e: (-e.cohesion, -e.length_m))
    kept = _merge_collinear(edges, merge_within_m)
    for idx, e in enumerate(kept):
        e.index = idx
    return kept


def _merge_collinear(edges: list[Edge], merge_within_m: float) -> list[Edge]:
    """Report one junction line once, however many surface fragments it came from. Stretches sit end to end, so the
    survivor absorbs both."""
    kept: list[Edge] = []
    for e in edges:
        half = e.length_m / 2
        for k in kept:
            if abs(float(e.direction @ k.direction)) <= 0.99:
                continue
            delta = e.midpoint - k.midpoint
            along = float(delta @ k.direction)
            if float(np.linalg.norm(delta - along * k.direction)) >= merge_within_m:
                continue  # parallel, but a different line
            k_half = k.length_m / 2
            lo, hi = min(along - half, -k_half), max(along + half, k_half)
            if along - half > k_half + merge_within_m or along + half < -k_half - merge_within_m:
                continue  # same line, but a genuine gap between the two stretches
            k.midpoint = k.midpoint + ((lo + hi) / 2) * k.direction
            k.length_m = float(hi - lo)
            break
        else:
            kept.append(e)
    return kept


def build_scan_corners(features: ScanFeatures, **kwargs) -> None:
    """Populate `features.corners` in place, after `extract_scan_features` (kept separate to
    avoid a circular import). `**kwargs` forwards to `find_corners`."""
    features.corners = find_corners(features.surfaces, **kwargs)


def build_scan_edges(features: ScanFeatures, **kwargs) -> None:
    """Populate `features.edges` in place, same contract as
    `build_scan_corners`. `**kwargs` forwards to `find_edges`."""
    features.edges = find_edges(features.surfaces, **kwargs)
