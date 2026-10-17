"""Merging scans at full resolution and exporting, lossless by default: every point out, with colour and intensity,
at full precision (merged in numpy so the intensity column survives). Also merging straight from a hand placement,
with no search, optionally refined between linked scans."""
from __future__ import annotations

import logging
import os
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .exclusions import filter_scan, for_export
from .lib.clouds import as_cloud
from .placement import Anchor, load_placements
from .pts_file import Scan, find_scans, read_pts, write_pts

log = logging.getLogger(__name__)

@dataclass
class MergeResult:
    points: np.ndarray
    colors: Optional[np.ndarray]
    intensity: Optional[np.ndarray]
    n_before_clean: int
    n_after_clean: int
    used_indices: list[int]

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def to_o3d(self) -> o3d.geometry.PointCloud:
        return as_cloud(self.points, self.colors)

    @property
    def cloud(self) -> o3d.geometry.PointCloud:
        """Kept so callers that expected an Open3D cloud still work."""
        return self.to_o3d()


def merge_scans(
    scans: list[Scan],
    poses: list[np.ndarray],
    cfg: PipelineConfig,
    only_indices: Optional[list[int]] = None,
) -> MergeResult:
    """Transform every scan into the shared frame and concatenate.

    Nothing is discarded unless the config explicitly asks for it.
    """
    indices = sorted(only_indices) if only_indices is not None else list(range(len(scans)))

    any_color = any(scans[i].has_color for i in indices)
    all_color = all(scans[i].has_color for i in indices)
    if any_color and not all_color:
        log.warning("Some scans have no colour. Those points will be written grey.")

    any_intensity = any(scans[i].intensity is not None for i in indices)

    pts_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []

    for i in indices:
        scan = scans[i]
        pose = np.asarray(poses[i], dtype=np.float64)
        pts_parts.append((pose[:3, :3] @ scan.points.T).T + pose[:3, 3])

        if any_color:
            col_parts.append(scan.colors if scan.has_color
                             else np.full((scan.n_points, 3), 0.5))
        if any_intensity:
            # -1 marks an unpopulated intensity field.
            int_parts.append(scan.intensity if scan.intensity is not None
                             else np.full(scan.n_points, -1.0))

    points = np.vstack(pts_parts)
    colors = np.vstack(col_parts) if col_parts else None
    intensity = np.concatenate(int_parts) if int_parts else None
    n_before = int(points.shape[0])

    # Both of the following throw points away. Off by default.
    if cfg.merge_voxel_size:
        log.warning("Downsampling the merged cloud to %.3f m. Points will be lost.",
                    cfg.merge_voxel_size)
        lo = points.min(axis=0)
        key = np.floor((points - lo) / cfg.merge_voxel_size).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        keep = np.sort(first)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
        if intensity is not None:
            intensity = intensity[keep]

    if cfg.outlier_nb_neighbors:
        log.warning("Removing statistical outliers. Points will be lost.")
        work = as_cloud(points)
        _, kept = work.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_nb_neighbors,
            std_ratio=cfg.outlier_std_ratio,
        )
        kept = np.asarray(kept)
        points = points[kept]
        if colors is not None:
            colors = colors[kept]
        if intensity is not None:
            intensity = intensity[kept]

    n_after = int(points.shape[0])
    if n_after == n_before:
        log.info("merged %d scan(s): %d points, nothing discarded",
                 len(indices), n_after)
    else:
        log.info("merged %d scan(s): %d -> %d points", len(indices), n_before, n_after)

    return MergeResult(points=points, colors=colors, intensity=intensity,
                       n_before_clean=n_before, n_after_clean=n_after,
                       used_indices=indices)


def export(result: MergeResult, path: str, cfg: PipelineConfig) -> None:
    """Write the merged cloud: .pts via our writer (lossless), other formats via Open3D (loses intensity, warns)."""
    lower = path.lower()
    if lower.endswith(".pts"):
        write_pts(path, result.points, result.colors, result.intensity,
                  precision=cfg.output_precision)
    elif lower.endswith((".ply", ".pcd", ".xyz", ".xyzrgb")):
        if result.intensity is not None:
            log.warning("%s cannot carry the intensity column, it will be dropped. "
                        "Export .pts to keep it.", path)
        if lower.endswith(".xyz") and result.colors is not None:
            # Open3D's .xyz writer has no colour channel (.xyzrgb does), so warn rather than drop colour silently.
            log.warning("%s cannot carry colour, it will be dropped. "
                        "Export .pts, .ply, .pcd or .xyzrgb to keep it.", path)
        if not o3d.io.write_point_cloud(path, result.to_o3d()):
            raise IOError(f"Open3D failed to write {path}")
    else:
        raise ValueError(
            f"Unsupported output extension for {path}. Use .pts, .ply, .pcd or .xyz"
        )
    log.info("wrote %s: %d points, colour=%s, intensity=%s, %d decimal places",
             path, result.n_points,
             "yes" if result.colors is not None else "no",
             "yes" if result.intensity is not None else "no",
             cfg.output_precision)


# -------------------------------------------------------------------------------------------------------------------
# Turn a hand placement into a merged cloud.
#
# Applies the saved bench placement to the full-resolution scans and merges them, with no coarse search. Optional
# refine runs ICP between linked scans, bounded by the placement.
# -------------------------------------------------------------------------------------------------------------------

reg = o3d.pipelines.registration


@dataclass
class PlaceMergeOutcome:
    scans: list[Scan]
    names: list[str]
    poses: list[np.ndarray]
    refine_log: list[dict] = field(default_factory=list)
    merged: Optional[MergeResult] = None
    skipped_no_placement: list[str] = field(default_factory=list)


def _load_named_scans(folder: str, names: list[str], pattern: str,
                      should_cancel: Optional[Callable[[], bool]] = None) -> tuple[dict[str, Scan], list[str]]:
    """Read only the .pts files the placement names, at full resolution.

    Also returns the other .pts names in the folder (not read) for the "not merged" report."""
    by_name = {}
    for path in find_scans(folder, pattern):
        n = os.path.splitext(os.path.basename(path))[0]
        by_name[n] = path

    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(
            "The placement file names scan(s) not found in "
            f"{folder}: {missing}. Placement names must match the .pts "
            "filenames (without the extension) exactly."
        )

    scans = {}
    for n in names:
        if should_cancel and should_cancel():
            raise CancelledError(f"cancelled after loading {len(scans)}/{len(names)} scan(s)")
        scans[n] = read_pts(by_name[n])
    unnamed = [n for n in by_name if n not in names]
    return scans, unnamed


def _prepared_down(scan: Scan, cfg: PipelineConfig) -> o3d.geometry.PointCloud:
    pcd = scan.to_o3d()
    down = pcd.voxel_down_sample(cfg.voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.voxel_size * cfg.normal_radius_mult,
            max_nn=cfg.normal_max_nn,
        )
    )
    return down


def _refine_from_anchors(
    anchors: list[Anchor],
    scans: dict[str, Scan],
    poses: dict[str, np.ndarray],
    cfg: PipelineConfig,
    max_shift_m: float,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """Nudge scan poses with ICP, seeded from the placement, in a star around each link's first member. A scan is
    adjusted once, by the first link that lists it."""
    downs: dict[str, o3d.geometry.PointCloud] = {}

    def down_of(name: str) -> o3d.geometry.PointCloud:
        if name not in downs:
            downs[name] = _prepared_down(scans[name], cfg)
        return downs[name]

    dist = cfg.voxel_size * cfg.icp_distance_mult
    log_rows: list[dict] = []
    adjusted: set[str] = set()

    for anchor_idx, anchor in enumerate(anchors):
        if should_cancel and should_cancel():
            raise CancelledError(f"cancelled after refining against {anchor_idx}/{len(anchors)} anchor(s)")
        members = anchor.members
        if len(members) < 2:
            log_rows.append({
                "anchor": anchor.id, "applied": False,
                "reason": "fewer than 2 members, nothing to refine",
            })
            continue
        ref_name = members[0].scan
        for mem in members[1:]:
            mover_name = mem.scan
            if ref_name not in scans or mover_name not in scans:
                log_rows.append({
                    "anchor": f"{anchor.id} ({ref_name} <-> {mover_name})", "applied": False,
                    "reason": "one or both scans in this anchor have no placement entry, skipped",
                })
                continue
            if mover_name in adjusted:
                continue

            src = o3d.geometry.PointCloud(down_of(mover_name)).transform(poses[mover_name])
            tgt = o3d.geometry.PointCloud(down_of(ref_name)).transform(poses[ref_name])
            src_pts_before = np.asarray(src.points)

            # Seed is identity: both clouds are already at their PLACED pose, so
            # ICP only has to find the small correction, not the whole transform.
            result = reg.registration_icp(
                src, tgt, dist, np.eye(4),
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iteration),
            )
            correction = np.asarray(result.transformation)

            # Measure actual point displacement, not the translation column.
            moved_pts = (correction[:3, :3] @ src_pts_before.T).T + correction[:3, 3]
            disp = np.linalg.norm(moved_pts - src_pts_before, axis=1)
            shift_mean = float(disp.mean()) if disp.size else 0.0
            shift_max = float(disp.max()) if disp.size else 0.0

            row = {
                "scan": mover_name, "against": ref_name,
                "icp_fitness": round(float(result.fitness), 4),
                "shift_mean_m": round(shift_mean, 4),
                "shift_max_m": round(shift_max, 4),
            }
            if shift_max > max_shift_m:
                row["applied"] = False
                row["reason"] = (
                    f"ICP wanted to move this scan's points up to {shift_max:.2f} m "
                    f"from where they were placed, past the {max_shift_m:.2f} m "
                    f"guard. Keeping the placed position."
                )
            else:
                poses[mover_name] = correction @ poses[mover_name]
                adjusted.add(mover_name)
                row["applied"] = True
            log_rows.append(row)
            log.info("refine %s vs %s: fitness=%.3f shift mean=%.3f max=%.3f m applied=%s",
                     mover_name, ref_name, result.fitness, shift_mean, shift_max, row["applied"])

    return log_rows


def place_merge(
    scans_folder: str,
    placements_path: str,
    output_path: str,
    cfg: PipelineConfig,
    pattern: str = "*.pts",
    refine: bool = False,
    refine_max_shift_m: float = 0.5,
    progress_cb=None,
    should_cancel: Optional[Callable[[], bool]] = None,
    keep_excluded: bool = True,
) -> PlaceMergeOutcome:
    """progress_cb(phase, detail), if given, runs at coarse checkpoints (the server polls it). should_cancel, if given,
    is checked between scan loads and between refine links, where a large export spends its time.
    keep_excluded: exclude-region points stay in the output (review removals never do); the refine step sees
    neither."""
    def tick(phase, detail=""):
        if progress_cb:
            progress_cb(phase, detail)

    loaded = load_placements(placements_path)
    placements = loaded.placements
    names = list(placements.keys())
    tick("loading", f"reading {len(names)} full-resolution scan(s)")
    all_scans, skipped = _load_named_scans(scans_folder, names, pattern, should_cancel=should_cancel)

    if skipped:
        log.warning(
            "%d scan(s) in %s have no entry in %s and will NOT be merged: %s",
            len(skipped), scans_folder, placements_path, skipped,
        )

    # Cut before anything else touches these scans; survivors stay full resolution and full colour. Refining loses
    # every region, the output only what for_export says.
    export_regions = for_export(loaded.exclusions, keep_excluded)
    align_scans = {n: filter_scan(all_scans[n], loaded.exclusions) for n in names} if refine else None
    if export_regions:
        for n in names:
            before = all_scans[n].n_points
            all_scans[n] = filter_scan(all_scans[n], export_regions)
            cut = before - all_scans[n].n_points
            if cut:
                log.info("%s: left %s of %s point(s) out of the export", n, f"{cut:,}", f"{before:,}")

    poses = {n: placements[n].matrix() for n in names}
    refine_log: list[dict] = []
    if refine:
        if not loaded.anchors:
            log.info("--refine was requested but the placement has no anchors, "
                      "so there is nothing to refine against. Add anchor "
                      "regions in the bench for the scans you want checked.")
        tick("refining", f"{len(loaded.anchors)} anchor(s)")
        refine_log = _refine_from_anchors(loaded.anchors, align_scans, poses, cfg, refine_max_shift_m,
                                          should_cancel=should_cancel)

    ordered_scans = [all_scans[n] for n in names]
    ordered_poses = [poses[n] for n in names]

    tick("merging", f"{sum(s.n_points for s in ordered_scans):,} points")
    merged = merge_scans(ordered_scans, ordered_poses, cfg)
    if output_path:
        tick("writing", output_path)
        export(merged, output_path, cfg)
    tick("done", f"{merged.n_points:,} points")

    return PlaceMergeOutcome(
        scans=ordered_scans, names=names, poses=ordered_poses,
        refine_log=refine_log, merged=merged, skipped_no_placement=skipped,
    )
