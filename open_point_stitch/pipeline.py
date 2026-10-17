"""End-to-end orchestration: load, prepare, pairwise, pose graph, merge.

The report is written even on failure, and each pair registers the smaller cloud as the source."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .pair_filter import candidate_pairs
from .config import PipelineConfig
from .align_points import load_seeds, seed_key
from .merge import MergeResult, merge_scans, export
from .align_pair import PairResult, register_pair
from .align_graph import GlobalResult, build_and_optimize
from .prepare import Prepared, preprocess_all
from .pts_file import Scan, load_folder
from .report import build_report, write_report, write_pair_csv
from .limits import apply_seed

log = logging.getLogger(__name__)


@dataclass
class StitchOutcome:
    scans: list[Scan]
    prepared: list[Prepared]
    pairs: list[PairResult]
    global_result: Optional[GlobalResult]
    merged: Optional[MergeResult]
    report: dict
    error: Optional[str] = None


def _invert(result: PairResult) -> PairResult:
    """Flip a PairResult so it reads source -> target in the opposite direction."""
    inv = np.linalg.inv(result.transform)
    result.transform = inv
    result.source, result.target = result.target, result.source
    result.source_name, result.target_name = result.target_name, result.source_name
    return result


def run_pairwise(
    prepared: list[Prepared],
    pairs: list[tuple[int, int]],
    cfg: PipelineConfig,
    seeds: Optional[dict[str, np.ndarray]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    scans: Optional[list[Scan]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    seeded_local_only: bool = False,
    prepared_gpu: Optional[list] = None,
    features: Optional[list] = None,
) -> list[PairResult]:
    """Register every planned pair; all keyword arguments are optional. progress_cb runs per pair and should_cancel
    between pairs; a pair's current stage (RANSAC, wall matching, corners, ...) is folded into desc as it runs."""
    seeds = seeds or {}
    use_gpu_pipeline = cfg.use_gpu_pipeline and prepared_gpu is not None
    if cfg.use_gpu_pipeline and prepared_gpu is None:
        log.warning("cfg.use_gpu_pipeline is on but no prepared_gpu was "
                    "supplied - falling back to the CPU path for every pair")

    # Read by stage_cb (below) from whichever thread is currently mid-pair;
    # only ever written from the loops at the bottom of this function.
    done_count = 0

    def full_res_pair(prep: Prepared) -> Optional[Scan]:
        return scans[prep.index] if scans is not None else None

    def feature_pair(prep: Prepared):
        return features[prep.index] if features is not None else None

    def make_stage_cb(desc: str):
        if progress_cb is None:
            return None
        def stage_cb(stage: str) -> None:
            progress_cb(done_count, len(pairs), f"{desc}: {stage}")
        return stage_cb

    def work(pair: tuple[int, int]) -> PairResult:
        i, j = pair
        a, b = prepared[i], prepared[j]
        stage_cb = make_stage_cb(f"{a.name} <-> {b.name}")

        key = seed_key(a.name, b.name)
        rev = seed_key(b.name, a.name)
        init = seeds.get(key)
        flipped_seed = None
        if init is None and rev in seeds:
            flipped_seed = np.linalg.inv(seeds[rev])

        if init is not None:
            r = register_pair(a, b, cfg, init=init, seeded_from="manual",
                              src_scan=full_res_pair(a), tgt_scan=full_res_pair(b),
                              local_only=seeded_local_only,
                              src_features=feature_pair(a), tgt_features=feature_pair(b),
                              progress_cb=stage_cb)
            return r
        if flipped_seed is not None:
            r = register_pair(a, b, cfg, init=flipped_seed, seeded_from="manual",
                              src_scan=full_res_pair(a), tgt_scan=full_res_pair(b),
                              local_only=seeded_local_only,
                              src_features=feature_pair(a), tgt_features=feature_pair(b),
                              progress_cb=stage_cb)
            return r

        # No seed: register the smaller cloud into the larger one.
        if use_gpu_pipeline:
            from .align_gpu import register_pair_cuda_gated
            if a.n_down <= b.n_down:
                return register_pair_cuda_gated(a, b, prepared_gpu[i], prepared_gpu[j], cfg,
                                                src_scan=full_res_pair(a), tgt_scan=full_res_pair(b))
            return _invert(register_pair_cuda_gated(b, a, prepared_gpu[j], prepared_gpu[i], cfg,
                                                    src_scan=full_res_pair(b), tgt_scan=full_res_pair(a)))
        if a.n_down <= b.n_down:
            return register_pair(a, b, cfg, src_scan=full_res_pair(a), tgt_scan=full_res_pair(b),
                                 src_features=feature_pair(a), tgt_features=feature_pair(b),
                                 progress_cb=stage_cb)
        return _invert(register_pair(b, a, cfg, src_scan=full_res_pair(b), tgt_scan=full_res_pair(a),
                                     src_features=feature_pair(b), tgt_features=feature_pair(a),
                                     progress_cb=stage_cb))

    results: list[PairResult] = []
    if cfg.concurrency <= 1:
        for k, pair in enumerate(pairs, 1):
            desc = f"{prepared[pair[0]].name} <-> {prepared[pair[1]].name}"
            log.info("[%d/%d] registering %s", k, len(pairs), desc)
            results.append(work(pair))
            done_count = k
            if progress_cb:
                progress_cb(k, len(pairs), desc)
            if should_cancel and should_cancel():
                log.info("cancelled after %d/%d pair(s) - keeping what finished", k, len(pairs))
                break
    else:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures = {pool.submit(work, p): p for p in pairs}
            done = 0
            for fut in as_completed(futures):
                done += 1
                done_count = done
                pair = futures[fut]
                desc = f"{prepared[pair[0]].name} <-> {prepared[pair[1]].name}"
                log.info("[%d/%d] finished %s", done, len(pairs), desc)
                results.append(fut.result())
                if progress_cb:
                    progress_cb(done, len(pairs), desc)
                if should_cancel and should_cancel():
                    # cancel_futures=True so the whole pool drains
                    pool.shutdown(wait=False, cancel_futures=True)
                    log.info("cancelled after %d/%d pair(s) finished - discarding unstarted "
                             "pairs, waiting only on whatever is already running", done, len(pairs))
                    break

    results.sort(key=lambda r: (min(r.source, r.target), max(r.source, r.target)))
    return results


def stitch_folder(
    folder: str,
    cfg: PipelineConfig,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    pair_csv_path: Optional[str] = None,
    pattern: str = "*.pts",
    largest_component_only: bool = True,
    seed_file: Optional[str] = None,
) -> StitchOutcome:
    t0 = time.time()
    apply_seed(cfg.random_seed)

    scans = load_folder(folder, pattern)
    log.info("loaded %d scan(s)", len(scans))
    for s in scans:
        log.info("  %s", s.describe())

    if len(scans) < 2:
        raise ValueError("Need at least 2 scans to stitch.")

    prepared = preprocess_all(scans, cfg)
    pairs = candidate_pairs(prepared, cfg)
    seeds = load_seeds(seed_file) if seed_file else {}
    if seeds:
        log.info("loaded %d manual seed(s) from %s", len(seeds), seed_file)
    log.info("attempting %d pairwise registration(s)", len(pairs))

    prepared_gpu = None
    if cfg.use_gpu_pipeline:
        from .align_gpu import prepare_all_gpu
        prepared_gpu = prepare_all_gpu(prepared, cfg)
        log.info("prepared %d scan(s) on the GPU for the unseeded path", len(prepared_gpu))

    results = run_pairwise(prepared, pairs, cfg, seeds, scans=scans, prepared_gpu=prepared_gpu)

    global_result: Optional[GlobalResult] = None
    merged: Optional[MergeResult] = None
    error: Optional[str] = None

    # From here on, failures must still produce a report.
    try:
        global_result = build_and_optimize(len(scans), results, cfg)
        indices = None
        if largest_component_only and not global_result.connected:
            indices = global_result.components[0]
        merged = merge_scans(scans, global_result.poses, cfg, only_indices=indices)
        if output_path:
            export(merged, output_path, cfg)
    except Exception as exc:
        error = str(exc)
        log.error("stitch could not complete: %s", exc)

    report = build_report(
        cfg=cfg,
        scan_names=[s.name for s in scans],
        scan_sizes=[s.n_points for s in scans],
        scan_density=[
            {"point_spacing_m": round(p.spacing_m, 5),
             "voxel_survival": round(p.survival, 4)}
            for p in prepared
        ],
        results=results,
        global_result=global_result,
        merged_points=merged.n_after_clean if merged else None,
        output_path=output_path if merged else None,
        elapsed_s=time.time() - t0,
    )
    if error:
        report["error"] = error
        report.setdefault("warnings", []).insert(0, f"RUN FAILED: {error}")

    if report_path:
        write_report(report_path, report)
        log.info("report written to %s", report_path)
    if pair_csv_path:
        write_pair_csv(pair_csv_path, results)

    return StitchOutcome(scans, prepared, results, global_result, merged, report, error)
