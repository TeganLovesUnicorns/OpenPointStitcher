"""Applying exclusion regions to a scan's points.

Excluding is a cut done once here for every consumer; survivors stay full resolution and the scan on disk is never
written."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .lib.geometry import points_in_box
from .placement import Exclusion
from .pts_file import Scan


def _in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test: True where a 2D point is inside the closed polygon `poly` (k x 2)."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < at)
    return inside


def shape_mask(points: np.ndarray, shape: dict) -> np.ndarray:
    """True where a raw-frame point falls inside one outline drawn in the review view: a maps raw to the screen it was
    drawn on (2 x 4), b maps raw to the world it was seen in (3 x 4), poly and circles are the outline in screen
    pixels, clip the slice limits in world metres that were on."""
    a, b = np.asarray(shape["a"], float), np.asarray(shape["b"], float)
    scr = points @ a[:, :3].T + a[:, 3]
    hit = np.zeros(len(points), dtype=bool)
    if shape.get("poly"):
        poly = np.asarray(shape["poly"], float)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        near = np.all((scr >= lo) & (scr <= hi), axis=1)
        hit[near] = _in_polygon(scr[near], poly)
    for cx, cy, r in shape.get("circles") or []:
        hit |= (scr[:, 0] - cx) ** 2 + (scr[:, 1] - cy) ** 2 <= r * r
    clip = shape.get("clip") or {}
    if any(clip.get(ax) for ax in "xyz") and hit.any():
        world = points[hit] @ b[:, :3].T + b[:, 3]
        keep = np.ones(len(world), dtype=bool)
        for k, ax in enumerate("xyz"):
            if clip.get(ax):
                lo, hi = clip[ax]
                keep &= (world[:, k] >= lo) & (world[:, k] <= hi)
        idx = np.flatnonzero(hit)
        hit[idx[~keep]] = False
    return hit


def excluded_mask(points: np.ndarray, regions: list[Exclusion]) -> np.ndarray:
    """True where a point is inside ANY region (its outlines if it has them, else its box); `regions` must already
    be this scan's (`region.scan` is not checked)."""
    n = points.shape[0]
    if not regions:
        return np.zeros(n, dtype=bool)
    excluded = np.zeros(n, dtype=bool)
    for region in regions:
        if region.shapes:
            for shape in region.shapes:
                excluded |= shape_mask(points, shape)
        else:
            excluded |= points_in_box(points, region.centroid, region.extent)
    return excluded


def filter_scan(scan: Scan, regions: list[Exclusion]) -> Scan:
    """A copy of `scan` without points inside its own regions (others ignored); `scan` itself if none."""
    own = [r for r in regions if r.scan == scan.name]
    if not own:
        return scan
    excluded = excluded_mask(scan.points, own)
    if not excluded.any():
        return scan
    keep = ~excluded
    return replace(
        scan,
        points=scan.points[keep],
        colors=None if scan.colors is None else scan.colors[keep],
        intensity=None if scan.intensity is None else scan.intensity[keep],
    )


def for_export(regions: list[Exclusion], keep_excluded: bool = True) -> list[Exclusion]:
    """What an export leaves out: every review removal, plus exclude regions only when keep_excluded is off. Alignment
    always uses all of regions."""
    return [r for r in regions if r.is_removal or not keep_excluded]


def filter_scans(scans: list[Scan], regions: list[Exclusion]) -> list[Scan]:
    """`filter_scan` over a whole list, in scan order - the shape every
    caller here (place_merge, server.py's automatic-align path) already
    has scans in."""
    return [filter_scan(s, regions) for s in scans]
