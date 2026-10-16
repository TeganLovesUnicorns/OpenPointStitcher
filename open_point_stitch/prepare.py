"""Downsampling, normal estimation and feature extraction.

Registration works on the downsampled clouds; full resolution returns only at merge."""

from __future__ import annotations

import contextlib
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .pts_file import Scan

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    """A scan reduced to what registration needs."""

    name: str
    index: int
    down: o3d.geometry.PointCloud     # voxel downsampled, normals estimated
    fpfh: o3d.pipelines.registration.Feature
    n_full: int
    spacing_m: float = 0.0            # median distance to the nearest neighbour

    @property
    def n_down(self) -> int:
        return len(self.down.points)

    @property
    def survival(self) -> float:
        """Fraction of points that survived the voxel downsample."""
        return self.n_down / self.n_full if self.n_full else 0.0


def thin_rng(name: str) -> np.random.Generator:
    """A stable per-scan (or per-pair) generator; never seed Open3D's global RNG."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def density_preserving_thin(pcd: o3d.geometry.PointCloud, target_max_points: int,
                             rng: np.random.Generator,
                             bin_m: float, floor_n: int,
                             iters: int = 24) -> tuple[np.ndarray, float]:
    """Thin to about target_max_points real points, keeping relative density (no averaging). Each bin keeps floor_n
    points plus a searched fraction of its excess. Returns (indices to keep, kept fraction of each bin's excess;
    0.0 means nothing thinned)."""
    pts = np.asarray(pcd.points)
    n_full = pts.shape[0]
    if n_full <= target_max_points:
        return np.arange(n_full), 0.0

    # Bin by XY cell only (this is about footprint density) and fold the two cell coordinates into one key, so
    # grouping is a single sort.
    cells = np.floor(pts[:, :2] / bin_m).astype(np.int64)
    keys = cells[:, 0] * 1_000_003 + cells[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n_full]))
    counts = ends - starts                         # points per occupied bin
    excess = np.maximum(counts - floor_n, 0)        # points above the floor

    def kept_count(frac: float) -> int:
        kept_excess = np.round(excess * frac).astype(np.int64)
        return int(np.minimum(counts, floor_n + kept_excess).sum())

    # Binary-search the excess-keep fraction. best_frac must start at 0.0, not 1.0.
    lo, hi = 0.0, 1.0
    best_frac = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if kept_count(mid) > target_max_points:
            hi = mid   # still too many points: keep less of the excess
        else:
            lo = mid   # fits the budget: remember it, then try keeping more
            best_frac = mid

    keep_n = np.minimum(counts, floor_n + np.round(excess * best_frac).astype(np.int64))
    keep_idx = np.empty(int(keep_n.sum()), dtype=np.int64)
    pos = 0
    for s, e, k in zip(starts, ends, keep_n):
        bin_idx = order[s:e]
        k = int(k)
        if k >= bin_idx.shape[0]:
            chosen = bin_idx
        else:
            # Real points, picked at random from what actually landed in
            # this bin - never an average of them.
            chosen = rng.choice(bin_idx, size=k, replace=False)
        keep_idx[pos:pos + chosen.shape[0]] = chosen
        pos += chosen.shape[0]
    return keep_idx[:pos], best_frac


def measure_spacing(pcd: o3d.geometry.PointCloud, sample: int = 20000,
                    seed: int = 42) -> float:
    """Median distance from a point to its nearest neighbour, in metres."""
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    n = min(sample, pts.shape[0])
    idx = rng.choice(pts.shape[0], size=n, replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in idx:
        k, _, d2 = tree.search_knn_vector_3d(pts[i], 2)
        if k > 1:
            dists.append(d2[1])
    return float(np.sqrt(np.median(dists))) if dists else 0.0


def _compute_prepared(name: str, points: np.ndarray, colors, cfg: PipelineConfig):
    """The heavy Open3D work for one scan, as (down, fpfh, n_full, spacing). Plain arrays in, so it can run in
    another process: several of these calls hold the GIL for seconds."""
    n_full = len(points)
    if cfg.max_range_m:
        near = np.hypot(points[:, 0], points[:, 1]) <= cfg.max_range_m
        if near.sum() >= 100:
            points, colors = points[near], (colors[near] if colors is not None else None)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    down = pcd.voxel_down_sample(cfg.voxel_size)

    if len(down.points) < 10:
        raise ValueError(
            f"{name}: only {len(down.points)} points survive a "
            f"{cfg.voxel_size} m voxel downsample. Lower --voxel-size."
        )

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.voxel_size * cfg.normal_radius_mult,
            max_nn=cfg.normal_max_nn,
        )
    )
    # Consistent normal orientation matters for point-to-plane ICP stability.
    down.orient_normals_consistent_tangent_plane(k=10)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.voxel_size * cfg.fpfh_radius_mult,
            max_nn=cfg.fpfh_max_nn,
        ),
    )
    return down, fpfh, n_full, measure_spacing(pcd)


def _finish_prepared(name: str, index: int, down, fpfh, n_full: int, spacing: float,
                     cfg: PipelineConfig) -> Prepared:
    prepared = Prepared(name=name, index=index, down=down, fpfh=fpfh, n_full=n_full, spacing_m=spacing)

    log.info("prepared %s: %d -> %d points (%.0f%% survived), point spacing %.2f cm",
             name, n_full, len(down.points), 100 * prepared.survival, spacing * 100)

    # Warn when the scan is coarse relative to the settings. This is the closest
    # the tool can get to noticing a decimated file, since nothing in the format
    # records it.
    if spacing > cfg.voxel_size * 0.5:
        log.warning(
            "%s: points sit %.1f cm apart, which is coarse next to the %.0f cm "
            "voxel setting. The downsample is barely reducing anything, so this "
            "scan may have been decimated, or captured at low density. Registration "
            "quality on thin overlaps is sensitive to this.",
            name, spacing * 100, cfg.voxel_size * 100,
        )
    return prepared


def preprocess_scan(scan: Scan, index: int, cfg: PipelineConfig) -> Prepared:
    down, fpfh, n_full, spacing = _compute_prepared(scan.name, scan.points, scan.colors, cfg)
    return _finish_prepared(scan.name, index, down, fpfh, n_full, spacing, cfg)


def preprocess_all(scans: list[Scan], cfg: PipelineConfig) -> list[Prepared]:
    return [preprocess_scan(s, i, cfg) for i, s in enumerate(scans)]


def _prepare_in_worker(task) -> dict:
    """Runs in a worker process: one scan in, the pieces of a `Prepared` out as plain arrays."""
    name, points, colors, cfg, threads = task
    from . import limits
    with (limits.limited_to(threads) if threads else contextlib.nullcontext()):
        down, fpfh, n_full, spacing = _compute_prepared(name, points, colors, cfg)
    return {"points": np.array(down.points), "normals": np.array(down.normals),
            "colors": np.array(down.colors) if down.has_colors() else None,
            "fpfh": np.array(fpfh.data), "n_full": n_full, "spacing": spacing}


def _rebuild_prepared(name: str, index: int, arrays: dict, cfg: PipelineConfig) -> Prepared:
    down = o3d.geometry.PointCloud()
    down.points = o3d.utility.Vector3dVector(arrays["points"])
    down.normals = o3d.utility.Vector3dVector(arrays["normals"])
    if arrays["colors"] is not None:
        down.colors = o3d.utility.Vector3dVector(arrays["colors"])
    fpfh = o3d.pipelines.registration.Feature()
    fpfh.data = arrays["fpfh"]
    return _finish_prepared(name, index, down, fpfh, arrays["n_full"], arrays["spacing"], cfg)


def preprocess_all_offloaded(scans: list[Scan], cfg: PipelineConfig, workers: int = 4,
                             threads: Optional[int] = None, should_cancel=None,
                             progress_cb=None) -> Optional[list[Prepared]]:
    """preprocess_all in worker processes, so the server's threads keep answering, stoppable between scans. Same
    results bit for bit. Returns None if should_cancel() said stop. threads is the run's CPU cap, shared between the
    workers."""
    import concurrent.futures
    import multiprocessing
    import os
    cap = threads if threads else (os.cpu_count() or 2)
    workers = max(1, min(workers, len(scans), cap))
    per_worker = max(1, cap // workers) if threads else None
    ctx = multiprocessing.get_context("spawn")   # never fork a process that already runs OpenMP threads
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    try:
        futures = [pool.submit(_prepare_in_worker, (s.name, s.points, s.colors, cfg, per_worker)) for s in scans]
        while True:
            done, pending = concurrent.futures.wait(futures, timeout=0.5)
            if progress_cb:
                progress_cb(len(done), len(futures))
            if should_cancel and should_cancel():
                pool.shutdown(wait=False, cancel_futures=True)
                for f in futures:
                    f.cancel()
                return None
            if not pending:
                break
        return [_rebuild_prepared(s.name, i, f.result(), cfg) for i, (s, f) in enumerate(zip(scans, futures))]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def aabb(pcd: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points)
    return pts.min(axis=0), pts.max(axis=0)
