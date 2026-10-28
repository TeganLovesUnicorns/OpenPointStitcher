"""Regression test on a generated two-room venue with an exactly known answer (tests/synthetic_venue.py): needs no
scans of anyone's. The small scan is registered into the large one with the default settings, the way the real-scan
regression test does it, and the answer must land on the truth. A second, separate process must then give the
bit-identical answer, since differences between processes never show inside one."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

# Judged where the bar scan is: how far its points land from where the truth puts them, on average, and the turn
# between the two answers. The truth is exact here, so these are tighter than the real pair's 10 cm and 1 degree.
MAX_TRANSLATION_ERROR_M = 0.03
MAX_ROTATION_ERROR_DEG = 0.3


def _register(folder: str) -> dict:
    """One registration of bar into hall, as the real-scan regression test runs it."""
    from open_point_stitch.config import PipelineConfig
    from open_point_stitch.prepare import preprocess_all
    from open_point_stitch.align_pair import register_pair
    from open_point_stitch.pts_file import read_pts
    from open_point_stitch.limits import apply_seed

    cfg = PipelineConfig()
    apply_seed(cfg.random_seed)
    src, tgt = read_pts(os.path.join(folder, "bar.pts")), read_pts(os.path.join(folder, "hall.pts"))
    t0 = time.time()
    prep = preprocess_all([src, tgt], cfg)
    result = register_pair(prep[0], prep[1], cfg)
    transform = np.asarray(result.transform, dtype=np.float64)
    return {"source": result.source_name, "target": result.target_name, "transform": transform.tolist(),
            "digest": hashlib.sha256(transform.tobytes()).hexdigest()[:16], "fitness": float(result.fitness),
            "rmse": float(result.rmse), "tilt_deg": float(result.tilt_deg), "strategy": result.seeded_from,
            "points": [src.n_points, tgt.n_points], "seconds": round(time.time() - t0, 1)}


def _errors(run: dict, truth: dict, bar_points: np.ndarray) -> tuple[float, float]:
    """(mean distance in metres between where the answer and the truth put the bar scan's points, turn in degrees)."""
    from open_point_stitch.lib.poses import rotation_angle_deg
    got = np.array(run["transform"])
    if run["source"] != truth["source"]:
        got = np.linalg.inv(got)
    want = np.array(truth["transform"])
    gap = bar_points @ (got[:3, :3] - want[:3, :3]).T + (got[:3, 3] - want[:3, 3])
    return float(np.linalg.norm(gap, axis=1).mean()), rotation_angle_deg(want[:3, :3].T @ got[:3, :3])


def main() -> int:
    import synthetic_venue
    from open_point_stitch.pts_file import read_pts

    with tempfile.TemporaryDirectory() as folder:
        truth = synthetic_venue.write(folder)
        bar_points = read_pts(os.path.join(folder, "bar.pts")).points
        print(f"venue: bar {truth['source_points']:,} points into hall {truth['target_points']:,} points")
        runs = [_register(folder)]
        if os.environ.get("OPENPOINT_TEST_CROSS_PROCESS", "1") != "0":
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", folder],
                                 capture_output=True, text=True, check=False)
            if out.returncode != 0:
                print(out.stdout + out.stderr)
                print("FAIL: the second process could not register the pair")
                return 1
            runs.append(json.loads(out.stdout.strip().splitlines()[-1]))

    print(f"{'run':>4} {'fitness':>8} {'rmse':>8} {'tilt':>6} {'off by':>11} {'turn':>9} {'time':>7}  strategy")
    failed = False
    for i, run in enumerate(runs):
        t_err, r_err = _errors(run, truth, bar_points)
        ok = t_err <= MAX_TRANSLATION_ERROR_M and r_err <= MAX_ROTATION_ERROR_DEG
        failed |= not ok
        print(f"{i + 1:>4} {run['fitness']:>8.4f} {run['rmse']:>8.4f} {run['tilt_deg']:>6.2f} {t_err * 100:>9.2f}cm "
              f"{r_err:>8.3f}d {run['seconds']:>6.1f}s  {run['strategy']}  {run['digest']}"
              f"{'' if ok else '   <-- WRONG'}")
    if failed:
        print(f"FAIL: the answer is more than {MAX_TRANSLATION_ERROR_M * 100:.0f} cm or {MAX_ROTATION_ERROR_DEG} "
              f"degrees from the known truth")
        return 1
    if len({r["digest"] for r in runs}) > 1:
        print("FAIL: two separate processes gave different answers on the same files with the same code")
        return 1
    print("PASS" + (" (both processes bit-identical)" if len(runs) > 1 else " (one process only)"))
    return 0


def test_regression_synthetic():
    assert main() == 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(_register(sys.argv[2])))
        raise SystemExit(0)
    raise SystemExit(main())
