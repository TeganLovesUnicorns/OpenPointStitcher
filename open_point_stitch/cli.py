"""Command line entry point for OpenPointStitcher. Local only, no network.

Examples:
    python -m open_point_stitch.cli stitch ./scans -o merged.pts [--threads 12]
    python -m open_point_stitch.cli inspect ./scans
    python -m open_point_stitch.cli gpu-info
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("folder", help="Folder containing .pts scans")
    p.add_argument("--pattern", default="*.pts", help="Glob for scan files")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-point-stitch",
        description="Align and merge .pts scans, locally.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    insp = sub.add_parser("inspect", help="Load scans and print their stats")
    _add_common(insp)

    gpu = sub.add_parser("gpu-info", help="Report what the installed Open3D can do")
    gpu.add_argument("-v", "--verbose", action="store_true")

    st = sub.add_parser("stitch", help="Run the full pipeline")
    _add_common(st)
    st.add_argument("-o", "--output", default="merged.pts",
                    help="Merged output path (.pts, .ply, .pcd, .xyz)")
    st.add_argument("--report", default=None,
                    help="Path for the JSON QA report (default: <output>.report.json)")
    st.add_argument("--pair-csv", default=None,
                    help="Path for the per-pair CSV (default: <output>.pairs.csv)")

    reg = st.add_argument_group("registration")
    reg.add_argument("--voxel-size", type=float, default=0.05,
                     help="Downsample size in metres for registration (default 0.05)")
    reg.add_argument("--landmarks", action="store_true",
                     help="Also propose starting poses by matching things that are not walls, floor or ceiling "
                          "(seating, equipment) between scans; found the right placement for about 1 pair in 10 on the "
                          "one real venue measured, so off by default")
    reg.add_argument("--adaptive-voxel", action="store_true",
                     help="Search for starting poses at a coarser size for a pair whose scans are sparse "
                          "(never finer than the default); dense pairs are unchanged")
    reg.add_argument("--icp-method",
                     choices=["point_to_plane", "point_to_point"],
                     default="point_to_plane")
    reg.add_argument("--colored-icp", action="store_true",
                     help="Add a colour-assisted refinement pass, re-scored at "
                          "the same threshold as the geometric fit")
    reg.add_argument("--min-fitness", type=float, default=0.08)
    reg.add_argument("--max-rmse", type=float, default=0.10)
    reg.add_argument("--min-correspondences", type=int, default=500)
    reg.add_argument("--adjacency", choices=["all", "sequential", "bbox"],
                     default="all")
    reg.add_argument("--coarse", choices=["ransac", "gravity", "walls", "both"],
                     default="both",
                     help="Coarse strategy. 'gravity' assumes both scans are "
                          "levelled and searches yaw only. 'walls' matches "
                          "dominant wall planes, for adjacent rooms.")
    reg.add_argument("--no-assume-level", action="store_true",
                     help="Do not assume the scans are gravity levelled. Disables "
                          "the yaw-only search and the tilt sanity check.")
    reg.add_argument("--max-tilt", type=float, default=6.0,
                     help="Reject any alignment tipping vertical past this, in degrees")
    reg.add_argument("--seed-file", default=None,
                     help="JSON of manual pre-alignments from the 'align' command")

    out = st.add_argument_group("output")
    out.add_argument("--merge-voxel", type=float, default=0.0,
                     help="Downsample the merged cloud, in metres. Default 0, "
                          "meaning keep every point. Only set this if you want "
                          "a smaller file; reduction is usually better done in "
                          "Vectorworks where you can see what you are losing.")
    out.add_argument("--outlier-removal", action="store_true",
                     help="Discard statistical outliers from the merged cloud. "
                          "Off by default; it throws points away.")
    out.add_argument("--precision", type=int, default=6,
                     help="Decimal places for XYZ in .pts output (default 6)")

    res = st.add_argument_group("resources (set once per run, not live adjustable)")
    res.add_argument("--threads", type=int, default=None,
                     help="Cap CPU threads for the numeric libraries")
    res.add_argument("--gpu", action="store_true",
                     help="Use the CUDA path where the installed Open3D supports it")
    res.add_argument("--gpu-pipeline", action="store_true",
                     help="Run the unseeded search on the full CUDA pipeline (coarse, fine and "
                          "scoring on the GPU). Separate from --gpu.")
    res.add_argument("--concurrency", type=int, default=1,
                     help="How many scan pairs to register at once")

    al = sub.add_parser(
        "align",
        help="LEGACY: pre-align one pair by clicking matching points. Needs a display, so it cannot run in "
             "the container; use the placement bench, or --wall-pair, which needs no display",
    )
    _add_common(al)
    al.add_argument("--source", required=True, help="Scan name (filename without .pts)")
    al.add_argument("--target", required=True, help="Scan name (filename without .pts)")
    al.add_argument("--seed-file", default="seeds.json",
                    help="Where to save the resulting pre-alignment")
    al.add_argument("--view-voxel", type=float, default=0.03,
                    help="Downsample for the picking viewer only, in metres")
    al.add_argument("--wall-pair", action="append", default=None, metavar="S:T",
                    help="Align by naming matching walls instead of clicking "
                         "points, e.g. --wall-pair 4:0 --wall-pair 2:1 where S "
                         "is a source wall label and T a target wall label from "
                         "tools/plot_walls.py. Two pairs fully determine the "
                         "alignment; one pair enumerates the second and ranks "
                         "the options. Needs no display.")

    pm = sub.add_parser(
        "place-merge",
        help="Apply a saved placement (from the browser bench) to the full "
             "resolution scans and merge them. No RANSAC, no wall matching, "
             "no gravity search: a person already solved coarse registration "
             "by placing the scans.",
    )
    _add_common(pm)
    pm.add_argument("--placements", required=True,
                    help="Placement JSON saved from the bench (open_point_stitch.placement.v1)")
    pm.add_argument("-o", "--output", default="merged.pts",
                    help="Merged output path (.pts, .ply, .pcd, .xyz)")
    pm.add_argument("--report", default=None,
                    help="Path for a JSON summary (default: <output>.report.json)")
    pm.add_argument("--voxel-size", type=float, default=0.05,
                    help="Downsample size for --refine's ICP pass, in metres")
    pm.add_argument("--refine", action="store_true",
                    help="After applying the placement, run ICP only, seeded "
                         "from the placement, between scans whose PLACED "
                         "bounding boxes overlap. Off by default.")
    pm.add_argument("--refine-max-shift", type=float, default=0.5,
                    help="If --refine would move a scan more than this many "
                         "metres from where it was placed, keep the placed "
                         "position instead (default 0.5 m)")
    pm.add_argument("--precision", type=int, default=6,
                    help="Decimal places for XYZ in .pts output (default 6)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    # Every command logs the build it is, the same identity the server reports at /api/version.
    from . import __version__
    build_time = os.environ.get("OPENPOINT_BUILD_TIME", "unknown (not a built image)")
    mode = "dev (mounted source)" if os.environ.get("OPENPOINT_DEV_MODE") else "release (baked in)"
    logging.info("open-point-stitcher %s, built %s, %s", __version__, build_time, mode)

    # Thread caps must be applied before Open3D and NumPy build their pools.
    from .limits import apply_thread_cap, configure_open3d_threads, gpu_status, resolve_use_gpu
    apply_thread_cap(getattr(args, "threads", None))

    if args.command == "gpu-info":
        print(json.dumps(gpu_status(), indent=2))
        return 0

    configure_open3d_threads(getattr(args, "threads", None))

    from .config import PipelineConfig
    from .pts_file import load_folder

    if args.command == "inspect":
        scans = load_folder(args.folder, args.pattern)
        for s in scans:
            print(s.describe())
        print(f"\n{len(scans)} scan(s), {sum(s.n_points for s in scans):,} points total")
        return 0

    if args.command == "align":
        import numpy as np
        from .align_points import interactive_align, save_seed
        from .align_gravity import tilt_of
        from .pts_file import read_pts

        def load_one(name):
            path = os.path.join(args.folder, name if name.endswith(".pts") else name + ".pts")
            if not os.path.exists(path):
                raise SystemExit(f"No such scan: {path}")
            return read_pts(path)

        s_scan = load_one(args.source)
        t_scan = load_one(args.target)
        s_cloud = s_scan.to_o3d().voxel_down_sample(args.view_voxel)
        t_cloud = t_scan.to_o3d().voxel_down_sample(args.view_voxel)

        if args.wall_pair:
            import open3d as o3d
            from .config import PipelineConfig
            from .limits import apply_seed
            from .find_walls import solve_from_named_pairs
            from .align_pair import interpenetration
            from .prepare import preprocess_all
            from .align_points import ManualAlignment

            cfg = PipelineConfig()
            apply_seed(cfg.random_seed)
            try:
                pairs = [tuple(int(x) for x in p.split(":")) for p in args.wall_pair]
            except ValueError:
                raise SystemExit("--wall-pair must look like 4:0")
            if any(len(p) != 2 for p in pairs):
                raise SystemExit("--wall-pair must look like 4:0")

            prep = preprocess_all([s_scan, t_scan], cfg)
            sp, tp = prep[0], prep[1]
            cands, labels, sw, tw = solve_from_named_pairs(sp.down, tp.down, pairs)
            if not cands:
                raise SystemExit(
                    "Those wall correspondences are not geometrically consistent. "
                    "Check the labels against tools/plot_walls.py output."
                )

            o3dreg = o3d.pipelines.registration
            scored = []
            for T, lab in zip(cands, labels):
                r = o3dreg.registration_icp(
                    sp.down, tp.down, cfg.voxel_size * cfg.icp_distance_mult, T,
                    o3dreg.TransformationEstimationPointToPlane(),
                    o3dreg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iteration))
                Tf = np.asarray(r.transformation)
                scored.append((float(r.fitness), float(r.inlier_rmse),
                               interpenetration(sp, tp, Tf), Tf, lab))
            scored.sort(key=lambda x: -x[0])

            print()
            print(f"{'fit':>7} {'rmse':>8} {'ipen':>6}  walls")
            for f, rm, ip, _, lab in scored:
                print(f"{f:>7.3f} {rm:>8.4f} {ip:>6.2f}  {lab}")
            print()
            print("Read fitness WITH interpenetration. High interpenetration with")
            print("low fitness means one room has been placed inside the other.")

            best = scored[0]
            alignment = ManualAlignment(
                transform=best[3],
                yaw_deg=float(np.degrees(np.arctan2(best[3][1, 0], best[3][0, 0]))),
                residuals=np.zeros(0),
                n_points=0,
            )
            save_seed(args.seed_file, s_scan.name, t_scan.name, alignment)
            print(f"\nsaved best ({best[4]}) to {args.seed_file}")
            return 0

        print()
        print("Pick the SAME features in both scans, in the SAME order.")
        print("Two points is enough, because gravity already fixes roll and pitch.")
        print("Pick features as far apart as you can: a door frame corner and")
        print("something across the room beats two things side by side.")

        alignment = interactive_align(s_cloud, t_cloud, s_scan.name, t_scan.name)
        print()
        print(f"  points used:    {alignment.n_points}")
        print(f"  yaw:            {alignment.yaw_deg:.2f} deg")
        print(f"  tilt:           {tilt_of(alignment.transform):.2f} deg")
        print(f"  rms residual:   {alignment.rms_residual * 100:.1f} cm")
        if alignment.rms_residual > 0.30:
            print("  WARNING: residual is large. The picked points may not "
                  "correspond. Consider redoing this pair.")
        save_seed(args.seed_file, s_scan.name, t_scan.name, alignment)
        print(f"  saved to:       {args.seed_file}")
        print()
        print("Now run the stitch with:")
        print(f"  python -m open_point_stitch.cli stitch {args.folder} "
              f"-o merged.pts --seed-file {args.seed_file}")
        return 0

    if args.command == "stitch":
        cfg = PipelineConfig(
            voxel_size=args.voxel_size,
            adaptive_search_voxel=args.adaptive_voxel,
            use_landmarks=args.landmarks,
            icp_method=args.icp_method,
            use_colored_icp=args.colored_icp,
            min_fitness=args.min_fitness,
            max_rmse=args.max_rmse,
            min_correspondences=args.min_correspondences,
            adjacency_mode=args.adjacency,
            coarse_method=args.coarse,
            assume_level=not args.no_assume_level,
            max_tilt_deg=args.max_tilt,
            merge_voxel_size=(args.merge_voxel if args.merge_voxel > 0 else None),
            outlier_nb_neighbors=(20 if args.outlier_removal else None),
            output_precision=args.precision,
            threads=args.threads,
            use_gpu=resolve_use_gpu(args.gpu),
            use_gpu_pipeline=resolve_use_gpu(args.gpu_pipeline),
            concurrency=max(1, args.concurrency),
        )

        report_path = args.report or (args.output + ".report.json")
        pair_csv = args.pair_csv or (args.output + ".pairs.csv")

        from .pipeline import stitch_folder
        from .report import print_summary

        outcome = stitch_folder(
            folder=args.folder,
            cfg=cfg,
            output_path=args.output,
            report_path=report_path,
            pair_csv_path=pair_csv,
            pattern=args.pattern,
            seed_file=args.seed_file,
        )
        print_summary(outcome.report)
        print(f"Report: {report_path}")
        print(f"Pairs:  {pair_csv}")
        return 1 if outcome.error else 0

    if args.command == "place-merge":
        import time
        from .merge_placed import place_merge

        cfg = PipelineConfig(voxel_size=args.voxel_size, output_precision=args.precision)
        report_path = args.report or (args.output + ".report.json")

        t0 = time.time()
        outcome = place_merge(
            scans_folder=args.folder,
            placements_path=args.placements,
            output_path=args.output,
            cfg=cfg,
            pattern=args.pattern,
            refine=args.refine,
            refine_max_shift_m=args.refine_max_shift,
        )
        elapsed = time.time() - t0

        print()
        print(f"placed and merged {len(outcome.names)} scan(s): {outcome.merged.n_points:,} points")
        if outcome.skipped_no_placement:
            print(f"skipped (no placement entry): {outcome.skipped_no_placement}")
        if args.refine:
            for row in outcome.refine_log:
                mark = "applied" if row["applied"] else "KEPT PLACEMENT"
                if "scan" in row:
                    print(f"  refine {row['scan']} vs {row['against']}: "
                          f"fitness={row['icp_fitness']:.3f} "
                          f"shift mean={row['shift_mean_m']:.3f} "
                          f"max={row['shift_max_m']:.3f}m [{mark}]")
                else:
                    print(f"  refine {row['pin']}: skipped ({row['reason']})")
        print(f"elapsed: {elapsed:.1f}s")
        print(f"wrote: {args.output}")

        report = {
            "scans": outcome.names,
            "skipped_no_placement": outcome.skipped_no_placement,
            "refine_enabled": args.refine,
            "refine_log": outcome.refine_log,
            "merged_points": outcome.merged.n_points,
            "output_path": args.output,
            "elapsed_s": round(elapsed, 2),
        }
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report: {report_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
