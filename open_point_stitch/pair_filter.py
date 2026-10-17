"""Deciding which scan pairs are worth attempting. Modes: all (default), sequential (no loop closures), bbox
(weak heuristic)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from .config import PipelineConfig
from .lib.clouds import as_cloud
from .overlap import direct_overlap
from .prepare import Prepared, aabb, density_preserving_thin, thin_rng
from .pts_file import Scan
from .find_surfaces import ScanFeatures, extract_scan_features, feature_pair_score

log = logging.getLogger(__name__)


def _boxes_overlap(a: tuple[np.ndarray, np.ndarray],
                   b: tuple[np.ndarray, np.ndarray],
                   pad: float) -> bool:
    a_lo, a_hi = a
    b_lo, b_hi = b
    return bool(np.all(a_lo - pad <= b_hi) and np.all(b_lo - pad <= a_hi))


def candidate_pairs(prepared: list[Prepared], cfg: PipelineConfig) -> list[tuple[int, int]]:
    n = len(prepared)
    if n < 2:
        return []

    if cfg.adjacency_mode == "sequential":
        pairs = [(i, i + 1) for i in range(n - 1)]
        log.warning(
            "sequential adjacency gives the pose graph no loop closures, so "
            "accumulated drift will not be redistributed."
        )
        return pairs

    if cfg.adjacency_mode == "bbox":
        boxes = [aabb(p.down) for p in prepared]
        # Centre each box so the comparison is about size/shape overlap rather
        # than the drifted absolute positions, which we do not trust.
        centred = []
        for lo, hi in boxes:
            c = (lo + hi) / 2.0
            centred.append((lo - c, hi - c))
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or _boxes_overlap(centred[i], centred[j], cfg.bbox_padding):
                    pairs.append((i, j))
        return pairs

    # Default: all pairs.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > 120:
        log.warning(
            "%d scans means %d pairwise registrations. Consider --adjacency bbox "
            "or splitting the job.", n, len(pairs)
        )
    return pairs


def placed_pairs(scans: list[Scan], placements: dict, padding_m: float) -> list[tuple[int, int]]:
    """Filter which pairs alignment attempts by the placed boxes, padded. Not candidate_pairs' centred bbox mode.
    A scan with no placement keeps its pairs."""
    n = len(scans)
    boxes: list[tuple[np.ndarray, np.ndarray] | None] = []
    for scan in scans:
        pl = placements.get(scan.name)
        if pl is None:
            boxes.append(None)
            continue
        pts = scan.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        M = pl.matrix()
        world = (M[:3, :3] @ corners.T).T + M[:3, 3]
        boxes.append((world.min(axis=0), world.max(axis=0)))

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = boxes[i], boxes[j]
            if bi is None or bj is None or _boxes_overlap(bi, bj, padding_m):
                pairs.append((i, j))
    return pairs


def surface_pairs(
    scans: list[Scan], prepared: list[Prepared], placements: dict, cfg: PipelineConfig,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict]]:
    """Choose pairs to attempt by measured shared surface within cfg.local_only_bound_m. Unplaced scans are kept.
    Returns (pairs, fractions); fractions covers every placed pair with its radius."""
    n = len(scans)
    thinned_world: list[np.ndarray | None] = [None] * n
    for i, scan in enumerate(scans):
        if placements.get(scan.name) is None:
            continue
        pcd = as_cloud(scan.points)
        keep_idx, _ = density_preserving_thin(
            pcd, cfg.adjacency_surface_thin_points, thin_rng(scan.name),
            bin_m=cfg.voxel_size, floor_n=cfg.search_thin_floor_n,
        )
        local_pts = scan.points[keep_idx.astype(np.int64)]
        M = placements[scan.name].matrix()
        thinned_world[i] = (M[:3, :3] @ local_pts.T).T + M[:3, 3]

    pairs: list[tuple[int, int]] = []
    fractions: dict[tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = thinned_world[i], thinned_world[j]
            if wi is None or wj is None:
                pairs.append((i, j))  # no placement to trust - keep, per placed_pairs' own rule
                continue
            radius = cfg.local_only_bound_m
            key = f"within_{int(radius * 100)}cm_frac"
            fwd = direct_overlap(wi, np.eye(4), wj, thresholds=(radius,))
            bwd = direct_overlap(wj, np.eye(4), wi, thresholds=(radius,))
            frac = max(fwd[key], bwd[key])
            fractions[(i, j)] = {"frac": round(frac, 4), "radius_m": round(radius, 4)}
            if frac >= cfg.adjacency_surface_min_frac:
                pairs.append((i, j))
    return pairs, fractions


def feature_surface_pairs(
    scans: list[Scan], prepared: list[Prepared], placements: dict, cfg: PipelineConfig,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict], list]:
    """Attempt a pair if matched static surfaces or surface_pairs()'s raw-point measure says yes (either one). Each
    fraction records its method. Also returns the per-scan features (None if unplaced) for the detector vote.
    progress_cb, if given, receives which scan is having its walls found."""
    n = len(scans)
    features: list[ScanFeatures | None] = [None] * n
    for i, scan in enumerate(scans):
        if placements.get(scan.name) is None:
            continue
        if progress_cb:
            progress_cb(f"finding walls: {i + 1}/{n}: {scan.name}")
        pcd = as_cloud(scan.points)
        features[i] = extract_scan_features(pcd)

    raw_pairs, raw_fractions = surface_pairs(scans, prepared, placements, cfg)
    raw_pairs_set = set(raw_pairs)

    pairs: list[tuple[int, int]] = []
    fractions: dict[tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = features[i], features[j]
            raw_info = raw_fractions.get((i, j))
            raw_attempted = (i, j) in raw_pairs_set

            if fi is None or fj is None or len(fi.surfaces) < 2 or len(fj.surfaces) < 2:
                # No usable features on at least one side - the raw-point
                # measure decides alone (still the same rule for an
                # entirely-unplaced scan: keep, never silently drop).
                if raw_info is not None:
                    fractions[(i, j)] = {**raw_info, "method": "surface"}
                    if raw_attempted:
                        pairs.append((i, j))
                else:
                    pairs.append((i, j))
                continue

            M_i = placements[scans[i].name].matrix()
            M_j = placements[scans[j].name].matrix()
            seed_transform = np.linalg.inv(M_j) @ M_i
            frac, n_matches, _matches = feature_pair_score(
                fi, fj, seed_transform,
                position_bound_m=cfg.adjacency_feature_position_bound_m)
            feature_attempted = frac >= cfg.adjacency_feature_min_frac

            # "feature" by default; changed to "neither" below if neither signal attempted the pair.
            entry = {"frac": round(frac, 4), "n_matches": n_matches,
                    "method": "feature" if feature_attempted else "neither"}
            if raw_info is not None:
                entry["surface_frac"] = raw_info["frac"]
                if raw_attempted:
                    entry["method"] = "feature+surface" if feature_attempted else "surface"
            fractions[(i, j)] = entry
            if feature_attempted or raw_attempted:
                pairs.append((i, j))
    return pairs, fractions, features
