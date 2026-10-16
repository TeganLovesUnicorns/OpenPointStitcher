"""Turn a hand placement into a merged cloud.

Applies the saved bench placement to the full-resolution scans and merges them, with no coarse search. Optional
refine runs ICP between linked scans, bounded by the placement."""
from __future__ import annotations

import logging
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .exclusions import filter_scan, for_export
from .merge import MergeResult, merge_scans, export
from .placement import Anchor, load_placements
from .pts_file import Scan, read_pts, find_scans
import os

log = logging.getLogger(__name__)

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
