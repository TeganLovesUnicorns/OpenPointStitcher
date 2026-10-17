"""Settle each scan's height on the floors and ceilings it shares with overlapping scans. Heights only: turn and
position are never touched."""
from __future__ import annotations

import itertools
from typing import Iterable, Optional

import numpy as np

from .align_anchors import flat_height_gap
from .lib.poses import ScanPose, world_cloud_points_as_drawn
from .find_floors_ceilings import on_pieces

MIN_PAIRS = 200         # floor/ceiling point pairs two scans must share before their gap counts
SIGMA_M = 0.03          # how far one pair's median gap may honestly be off
PRIOR_M = 0.20          # the weak pull toward the height the scan already has
FLOOR_M = 0.005         # residuals under this all count the same (the least-absolute solve needs a floor)
ROUNDS = 30


def pair_gaps(world: dict[str, np.ndarray]) -> dict[tuple[str, str], tuple[int, float]]:
    """{(a, b): (pairs, gap m)} for every scan pair sharing floors or ceilings; gap > 0 means b sits higher."""
    out = {}
    for a, b in itertools.combinations(sorted(world), 2):
        n, gap = flat_height_gap(world[b], world[a])
        if n >= MIN_PAIRS:
            out[(a, b)] = (n, gap)
    return out


def solve(gaps: dict[tuple[str, str], tuple[int, float]], names: Iterable[str], fixed: set[str]) -> dict[str, float]:
    """The height change per scan that best closes every gap at once; `fixed` scans do not move."""
    free = [n for n in names if n not in fixed]
    at = {n: i for i, n in enumerate(free)}
    edges = [(a, b, n, g) for (a, b), (n, g) in gaps.items() if a in at or b in at]
    dz = np.zeros(len(free))
    if not edges:
        return {n: 0.0 for n in free}
    robust = np.ones(len(edges))
    for _ in range(ROUNDS):
        h = np.eye(len(free)) / PRIOR_M ** 2
        rhs = np.zeros(len(free))
        for (a, b, n, g), rw in zip(edges, robust):
            w = rw * np.sqrt(n) / SIGMA_M ** 2        # want dz_b - dz_a = -g
            j = np.zeros(len(free))
            if b in at:
                j[at[b]] += 1.0
            if a in at:
                j[at[a]] -= 1.0
            h += w * np.outer(j, j)
            rhs += w * j * (-g)
        dz = np.linalg.solve(h, rhs)
        # Least-absolute weighting: pairs that agree win outright, a pair that disagrees keeps its whole error.
        r = np.array([(dz[at[b]] if b in at else 0.0) - (dz[at[a]] if a in at else 0.0) + g for a, b, _, g in edges])
        robust = FLOOR_M / np.maximum(np.abs(r), FLOOR_M)
    return {n: float(dz[at[n]]) for n in free}


MARK_BAND_M = 0.06      # a scan with a marked floor or ceiling settles on points this close to the marks only


def marked_payload(payload: dict, heights: list[float]) -> dict:
    """The payload's preview points within MARK_BAND_M of any marked height (scan frame), so tables and seats cannot
    pair as floor."""
    h = np.asarray(payload["h"], dtype=float)
    keep = np.zeros(len(h), dtype=bool)
    for m in heights:
        keep |= np.abs(h - m) <= MARK_BAND_M
    xy = np.asarray(payload["xy"], dtype=float).reshape(-1, 2)
    return {**payload, "xy": xy[keep].ravel(), "h": h[keep]}


def pieces_payload(payload: dict, pieces: list[dict]) -> dict:
    """The payload's preview points on the scan's found floor and ceiling pieces only (several floor levels and
    sloped ceilings included, counters and tables not)."""
    h = np.asarray(payload["h"], dtype=float)
    xy = np.asarray(payload["xy"], dtype=float).reshape(-1, 2)
    keep = on_pieces(xy, h, pieces, MARK_BAND_M)
    return {**payload, "xy": xy[keep].ravel(), "h": h[keep]}


def without_stairs(payload: dict) -> dict:
    """The payload without its preview points on marked stairs (stairs indices), so a tread is never paired as a
    floor."""
    if not payload.get("stairs"):
        return payload
    keep = np.ones(len(payload["h"]), dtype=bool)
    keep[np.asarray(payload["stairs"], dtype=int)] = False
    xy = np.asarray(payload["xy"], dtype=float).reshape(-1, 2)
    return {**payload, "xy": xy[keep].ravel(), "h": np.asarray(payload["h"], dtype=float)[keep]}


def settle(payloads: dict[str, dict], poses: dict[str, ScanPose], fixed: set[str],
           only: Optional[set[str]] = None,
           marks: Optional[dict[str, list[float]]] = None,
           pieces: Optional[dict[str, list[dict]]] = None) -> tuple[dict[str, float], dict]:
    """(height change per movable scan, the gaps it was solved from). only, if given, limits which scans may move.
    marks holds per-scan floor and ceiling heights marked in review; those scans use only those. pieces holds each
    scan's found floor and ceiling pieces: an unmarked scan with some uses only points on them."""
    marks = marks or {}
    pieces = pieces or {}

    def usable(n: str) -> dict:
        pl = without_stairs(payloads[n])
        if marks.get(n):
            return marked_payload(pl, marks[n])
        return pieces_payload(pl, pieces[n]) if pieces.get(n) else pl
    world = {n: world_cloud_points_as_drawn(usable(n), p) for n, p in poses.items() if n in payloads}
    gaps = pair_gaps(world)
    hold = set(fixed) | ({n for n in world if n not in only} if only is not None else set())
    return solve(gaps, world, hold), gaps
