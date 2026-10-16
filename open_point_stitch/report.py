"""Per-run QA report.

Exists so an alignment that converges but is wrong in space shows up as a row to inspect."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .config import PipelineConfig
from .align_pair import PairResult
from .align_graph import GlobalResult


def _pose_summary(pose: np.ndarray) -> dict:
    t = pose[:3, 3]
    r = pose[:3, :3]
    # Rotation magnitude from the trace, clamped for numerical safety.
    cos_theta = (np.trace(r) - 1.0) / 2.0
    theta = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return {
        "tx_m": round(float(t[0]), 4),
        "ty_m": round(float(t[1]), 4),
        "tz_m": round(float(t[2]), 4),
        "rotation_deg": round(theta, 3),
    }


def transform_error_against(
    transform: np.ndarray, source_name: str, target_name: str,
    reference_transform: np.ndarray, reference_source_name: str, reference_target_name: str,
) -> tuple[float, float]:
    """Translation (m) and rotation (deg) error against a reference, whichever scan is the low index. Raises
    ValueError if the two name pairs are not the same pair."""
    given = {source_name, target_name}
    ref = {reference_source_name, reference_target_name}
    if given != ref:
        raise ValueError(
            f"not the same pair: got {sorted(given)}, reference is {sorted(ref)}"
        )
    candidate = transform if source_name == reference_source_name else np.linalg.inv(transform)
    delta = np.linalg.inv(reference_transform) @ candidate
    t_err = float(np.linalg.norm(delta[:3, 3]))
    cos_theta = (np.trace(delta[:3, :3]) - 1.0) / 2.0
    r_err = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return t_err, r_err


def write_pair_csv(path: str, results: list[PairResult]) -> None:
    if not results:
        return
    rows = [r.as_row() for r in results]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    cfg: PipelineConfig,
    scan_names: list[str],
    scan_sizes: list[int],
    results: list[PairResult],
    global_result: Optional[GlobalResult],
    merged_points: Optional[int],
    output_path: Optional[str],
    elapsed_s: float,
    scan_density: Optional[list[dict]] = None,
) -> dict:
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_s, 2),
        "config": cfg.to_dict(),
        "derived_distances_m": cfg.scaled(),
        "scans": [
            {"index": i, "name": n, "points": s,
             **(scan_density[i] if scan_density and i < len(scan_density) else {})}
            for i, (n, s) in enumerate(zip(scan_names, scan_sizes))
        ],
        "pairs_attempted": len(results),
        "pairs_accepted": len(accepted),
        "pairs_rejected": len(rejected),
        "pairs": [r.as_row() for r in results],
        "merged_points": merged_points,
        "output": output_path,
    }

    if global_result is not None:
        report["pose_graph"] = {
            "odometry_edges": global_result.n_odometry_edges,
            "loop_closure_edges": global_result.n_loop_edges,
            "connected": global_result.connected,
            "components": global_result.components,
            "final_poses": [
                {"index": i, "name": scan_names[i], **_pose_summary(p)}
                for i, p in enumerate(global_result.poses)
            ],
            # Full 4x4 matrices, so a run can be scored or re-applied later
            # without re-registering anything.
            "full_poses": [
                {"index": i, "name": scan_names[i], "matrix": p.tolist()}
                for i, p in enumerate(global_result.poses)
            ],
        }

    # Things a human should look at before trusting the result.
    warnings: list[str] = []
    if rejected:
        warnings.append(
            f"{len(rejected)} pair(s) were rejected. Check the rejected rows: a "
            f"pair that should overlap but failed usually means the two scans "
            f"need manual pre-alignment."
        )
    if global_result is not None and not global_result.connected:
        warnings.append(
            "The scans do not form one connected graph. Scans outside the "
            "largest group were left out of the merge."
        )
    if global_result is not None and global_result.n_loop_edges == 0:
        warnings.append(
            "No loop closure edges were found, so the pose graph could not "
            "redistribute drift. The result is only as good as the chain."
        )
    tilted = [r for r in results if "tilted" in r.reason]
    if tilted:
        warnings.append(
            f"{len(tilted)} pair(s) produced only tilted alignments. Both scans "
            f"are assumed level, so a tilted answer is physically wrong however "
            f"well it scores. Use manual pre-alignment for these pairs: "
            f"open-point-stitch align <folder> --source A --target B"
        )
    suspicious = [r for r in results
                  if r.interpenetration > 0.5 and r.fitness < 0.20]
    if suspicious:
        warnings.append(
            f"{len(suspicious)} pair(s) show heavy volume overlap ({'>'}50%) but "
            f"poor surface agreement. That combination usually means one scan "
            f"has been placed bodily inside the other, which is physically "
            f"impossible for rooms either side of a wall. Treat these as wrong "
            f"and pre-align them by hand."
        )
    # No "looks sparse" warning from voxel survival: point spacing varies too much within real scans.
    weak = [r for r in accepted if r.fitness < cfg.min_fitness * 2]
    if weak:
        warnings.append(
            f"{len(weak)} accepted pair(s) are only marginally above the fitness "
            f"threshold. Inspect these visually."
        )
    report["warnings"] = warnings
    return report


def write_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)


def print_summary(report: dict) -> None:
    print()
    print("=" * 68)
    print("STITCH SUMMARY")
    print("=" * 68)
    print(f"Scans:            {len(report['scans'])}")
    print(f"Pairs attempted:  {report['pairs_attempted']}")
    print(f"Pairs accepted:   {report['pairs_accepted']}")
    print(f"Pairs rejected:   {report['pairs_rejected']}")
    if "pose_graph" in report:
        pg = report["pose_graph"]
        print(f"Pose graph edges: {pg['odometry_edges']} odometry, "
              f"{pg['loop_closure_edges']} loop closure")
        print(f"Connected:        {pg['connected']}")
    if report.get("merged_points") is not None:
        print(f"Merged points:    {report['merged_points']:,}")
    if report.get("output"):
        print(f"Output:           {report['output']}")
    print(f"Elapsed:          {report['elapsed_seconds']} s")

    print()
    print(f"{'pair':<26} {'seed':<8} {'fit':>7} {'rmse(m)':>9} {'corr':>7} {'tilt':>6} {'ipen':>6}  status")
    print("-" * 104)
    for row in report["pairs"]:
        label = f"{row['source_name'][:11]}->{row['target_name'][:11]}"
        status = "OK" if row["accepted"] else f"REJECT: {row['reason']}"
        fit = "-" if row["fitness"] is None else f"{row['fitness']:.3f}"
        rmse = "-" if row["rmse_m"] is None else f"{row['rmse_m']:.4f}"   # None: nothing near enough to measure
        print(f"{label:<26} {row['seeded_from']:<8} {fit:>7} "
              f"{rmse:>9} {row['correspondences']:>7} "
              f"{row.get('tilt_deg', 0.0):>6.2f} "
              f"{row.get('interpenetration', 0.0):>6.2f}  {status}")

    if report.get("warnings"):
        print()
        print("WARNINGS")
        for w in report["warnings"]:
            print(f"  - {w}")
    print()
