"""Launch N separate processes (--worker: a real run; --probe: upstream stage fingerprints only) and compare what
each produced. Writes hunt_log.jsonl, one line per process.
Usage: python tools/hunt_nondeterminism.py --scans "C:\\path\\to\\scans" --runs 40 [--hashseed 0]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "ampbar_confirmed.json")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _digest(arr) -> str:
    """A short stable fingerprint of an array's exact bytes."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def find_scans(folder: str):
    """Locate the confirmed pair, matching the regression test's logic."""
    src = tgt = None
    for f in sorted(glob.glob(os.path.join(folder, "*.pts"))):
        low = os.path.basename(f).lower()
        if "bar" in low:
            src = f
        elif "amp" in low:
            tgt = f
    return src, tgt


def env_fingerprint() -> dict:
    import numpy as np
    import open3d as o3d
    return {
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "open3d": o3d.__version__,
        "hashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        # PYTHONHASHSEED unset means Python randomises per process while the
        # variable stays unset, so the variable alone cannot tell you whether
        # hashing differed. This does.
        "hash_probe": hash("open-point-stitcher"),
        "omp": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "mkl": os.environ.get("MKL_NUM_THREADS", "<unset>"),
        "openblas": os.environ.get("OPENBLAS_NUM_THREADS", "<unset>"),
    }


# --------------------------------------------------------------------------
# mode 1: the pristine worker
# --------------------------------------------------------------------------

def run_worker(src_path: str, tgt_path: str) -> dict:
    """One real registration. Do not add stage probes here: touching the global random stream first changes the
    answer."""
    import numpy as np
    from open_point_stitch.config import PipelineConfig
    from open_point_stitch.prepare import preprocess_all
    from open_point_stitch.align_pair import register_pair
    from open_point_stitch.pts_file import read_pts
    from open_point_stitch.limits import apply_seed

    out = {"mode": "worker", "env": env_fingerprint()}

    cfg = PipelineConfig()
    apply_seed(cfg.random_seed)

    src = read_pts(src_path)
    tgt = read_pts(tgt_path)
    out["load"] = {
        "src_points": src.n_points,
        "tgt_points": tgt.n_points,
        "src_digest": _digest(src.points),
        "tgt_digest": _digest(tgt.points),
    }

    prep = preprocess_all([src, tgt], cfg)
    out["preprocess"] = {
        "src_down": len(prep[0].down.points),
        "tgt_down": len(prep[1].down.points),
        "src_down_digest": _digest(np.asarray(prep[0].down.points)),
        "tgt_down_digest": _digest(np.asarray(prep[1].down.points)),
        "src_normal_digest": _digest(np.asarray(prep[0].down.normals)),
        "tgt_normal_digest": _digest(np.asarray(prep[1].down.normals)),
    }

    result = register_pair(prep[0], prep[1], cfg)
    out["result"] = {
        "transform": np.asarray(result.transform).tolist(),
        "transform_digest": _digest(result.transform),
        "fitness": round(float(result.fitness), 6),
        "rmse": round(float(result.rmse), 6),
        "coarse_fitness": round(float(result.coarse_fitness), 6),
        "coarse_rmse": round(float(result.coarse_rmse), 6),
        "tilt_deg": round(float(result.tilt_deg), 4),
        "n_candidates": int(result.n_candidates),
        "seeded_from": result.seeded_from,
        "accepted": bool(result.accepted),
        "reason": result.reason,
    }

    if os.path.exists(FIXTURE):
        truth = json.load(open(FIXTURE))
        expected = np.array(truth["transform"])
        delta = np.linalg.inv(expected) @ np.asarray(result.transform)
        cos_theta = (np.trace(delta[:3, :3]) - 1.0) / 2.0
        out["error"] = {
            "translation_m": round(float(np.linalg.norm(delta[:3, 3])), 4),
            "rotation_deg": round(
                float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))), 4),
        }
    return out


# --------------------------------------------------------------------------
# mode 2: the probe
# --------------------------------------------------------------------------

def run_probe(src_path: str, tgt_path: str) -> dict:
    """Fingerprint everything upstream of registration; never registers, so the RNG re-seeding in
    estimate_level and extract_walls is safe here."""
    import numpy as np
    from open_point_stitch.config import PipelineConfig
    from open_point_stitch.prepare import preprocess_all
    from open_point_stitch.pts_file import read_pts
    from open_point_stitch.limits import apply_seed
    from open_point_stitch.align_gravity import estimate_level
    from open_point_stitch.find_walls import extract_walls

    out = {"mode": "probe", "env": env_fingerprint()}

    cfg = PipelineConfig()
    apply_seed(cfg.random_seed)

    src = read_pts(src_path)
    tgt = read_pts(tgt_path)
    out["load"] = {
        "src_points": src.n_points,
        "tgt_points": tgt.n_points,
        "src_digest": _digest(src.points),
        "tgt_digest": _digest(tgt.points),
    }

    prep = preprocess_all([src, tgt], cfg)
    out["preprocess"] = {
        "src_down": len(prep[0].down.points),
        "tgt_down": len(prep[1].down.points),
        "src_down_digest": _digest(np.asarray(prep[0].down.points)),
        "tgt_down_digest": _digest(np.asarray(prep[1].down.points)),
        "src_normal_digest": _digest(np.asarray(prep[0].down.normals)),
        "tgt_normal_digest": _digest(np.asarray(prep[1].down.normals)),
        "src_fpfh_digest": _digest(np.asarray(prep[0].fpfh.data))
                           if getattr(prep[0], "fpfh", None) is not None else None,
        "tgt_fpfh_digest": _digest(np.asarray(prep[1].fpfh.data))
                           if getattr(prep[1], "fpfh", None) is not None else None,
    }

    lvl = {}
    for label, p in (("src", prep[0]), ("tgt", prep[1])):
        info = estimate_level(p.down)
        lvl[label] = {
            "rotation_digest": _digest(info.rotation),
            "floor_z": round(float(getattr(info, "floor_z", float("nan"))), 6),
            "tilt_deg": round(float(getattr(info, "tilt_deg", float("nan"))), 6),
            "reliable": bool(getattr(info, "reliable", False)),
        }
    out["level"] = lvl

    wl = {}
    for label, p in (("src", prep[0]), ("tgt", prep[1])):
        walls = extract_walls(p.down)
        planes = [[float(w.normal[0]), float(w.normal[1]), float(w.offset)]
                  for w in walls]
        wl[label] = {
            "count": len(walls),
            "planes_digest": _digest(planes) if planes else None,
            "planes": [[round(v, 5) for v in row] for row in planes],
            "support": [int(w.support) for w in walls],
        }
    out["walls"] = wl
    return out


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

STAGE_KEYS = [
    ("load", ["src_digest", "tgt_digest", "src_points", "tgt_points"]),
    ("preprocess", ["src_down", "tgt_down", "src_down_digest", "tgt_down_digest",
                    "src_normal_digest", "tgt_normal_digest",
                    "src_fpfh_digest", "tgt_fpfh_digest"]),
    ("level", None),
    ("walls", None),
    ("result", ["transform_digest", "fitness", "seeded_from", "n_candidates"]),
]


def _stage_values(record: dict, stage: str, keys):
    block = record.get(stage)
    if block is None:
        return None
    if keys is None:
        return json.dumps(block, sort_keys=True)
    return json.dumps({k: block.get(k) for k in keys if k in block}, sort_keys=True)


def summarise(records: list[dict], label: str) -> None:
    if not records:
        print(f"\n{label}: no runs completed.")
        return

    print(f"\n{'=' * 72}")
    print(f"{label}: {len(records)} processes")
    print("=" * 72)

    first_unstable = None
    for stage, keys in STAGE_KEYS:
        vals = [_stage_values(r, stage, keys) for r in records]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        distinct = len(set(vals))
        mark = "stable" if distinct == 1 else f"{distinct} DIFFERENT VALUES"
        print(f"  {stage:<12} {mark}")
        if distinct > 1 and first_unstable is None:
            first_unstable = stage

    if first_unstable:
        print(f"\n  --> first stage that disagrees between processes: {first_unstable}")
    else:
        print("\n  --> every stage identical across all processes")

    # environment differences, which is where a per-process cause would show
    env_keys = ["hashseed", "hash_probe", "omp", "mkl", "openblas",
                "python", "numpy", "open3d"]
    varying = [k for k in env_keys
               if len({r["env"].get(k) for r in records}) > 1]
    if varying:
        print(f"  environment values that varied: {', '.join(varying)}")
    else:
        print("  environment identical across all processes")

    results = [r for r in records if "result" in r]
    if not results:
        return

    print(f"\n  {'error':>10}  {'fitness':>8}  {'strategy':>9}  count")
    buckets = defaultdict(list)
    for r in results:
        err = r.get("error", {}).get("translation_m")
        key = ("n/a" if err is None
               else "under 10cm" if err <= 0.10
               else f"{err:.2f} m")
        buckets[key].append(r)
    for key, rs in sorted(buckets.items(),
                          key=lambda kv: (kv[0] != "under 10cm", kv[0])):
        fits = [r["result"]["fitness"] for r in rs]
        strat = Counter(r["result"]["seeded_from"] for r in rs).most_common(1)[0][0]
        print(f"  {key:>10}  {sum(fits) / len(fits):>8.4f}  {strat:>9}  {len(rs)}")

    good = sum(1 for r in results
               if r.get("error", {}).get("translation_m", 9e9) <= 0.10)
    print(f"\n  {good} of {len(results)} landed within 10 cm of the confirmed answer")

    if good and good < len(results):
        # The most useful comparison available: what differs between a run that
        # got it right and one that did not.
        right = next(r for r in results
                     if r.get("error", {}).get("translation_m", 9e9) <= 0.10)
        wrong = next(r for r in results
                     if r.get("error", {}).get("translation_m", 9e9) > 0.10)
        print("\n  one right run vs one wrong run:")
        for stage, keys in STAGE_KEYS:
            a = _stage_values(right, stage, keys)
            b = _stage_values(wrong, stage, keys)
            if a is None or b is None:
                continue
            print(f"    {stage:<12} {'same' if a == b else 'DIFFERENT'}")
        for k in env_keys:
            a, b = right["env"].get(k), wrong["env"].get(k)
            if a != b:
                print(f"    env.{k:<8} {a} vs {b}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", default=os.environ.get("OPENPOINT_TEST_SCANS"),
                    help="Folder holding the full resolution amp room and small "
                         "bar scans. Defaults to $OPENPOINT_TEST_SCANS.")
    ap.add_argument("--runs", type=int, default=20,
                    help="How many separate processes per mode (default 20)")
    ap.add_argument("--mode", choices=["both", "worker", "probe"], default="both")
    ap.add_argument("--hashseed", default=None,
                    help="Force PYTHONHASHSEED in every child, to test whether "
                         "Python's per-process hash randomisation is the cause")
    ap.add_argument("--threads", default=None,
                    help="Force OMP_NUM_THREADS in every child, to test whether "
                         "thread count or scheduling is the cause")
    ap.add_argument("--log", default="hunt_log.jsonl")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--src", help=argparse.SUPPRESS)
    ap.add_argument("--tgt", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    # child process paths
    if args.worker or args.probe:
        fn = run_worker if args.worker else run_probe
        try:
            print(json.dumps(fn(args.src, args.tgt)))
            return 0
        except Exception as exc:
            import traceback
            print(json.dumps({"mode": "worker" if args.worker else "probe",
                              "env": env_fingerprint(),
                              "error_text": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()}))
            return 1

    # orchestrator path
    if not args.scans or not os.path.isdir(args.scans):
        print("Point --scans at the folder holding the full resolution amp room "
              "and small bar scans, or set OPENPOINT_TEST_SCANS.")
        return 2

    src_path, tgt_path = find_scans(args.scans)
    if not src_path or not tgt_path:
        print(f"Could not find both scans in {args.scans}. Need one file with "
              f"'bar' in the name and one with 'amp'.")
        return 2

    print(f"source: {os.path.basename(src_path)}")
    print(f"target: {os.path.basename(tgt_path)}")
    print(f"{args.runs} processes per mode. This is meant to be slow; that is "
          f"what the desktop is for.\n")

    modes = ["worker", "probe"] if args.mode == "both" else [args.mode]
    collected = {m: [] for m in modes}

    with open(args.log, "w") as log:
        for mode in modes:
            for i in range(args.runs):
                env = dict(os.environ)
                if args.hashseed is not None:
                    env["PYTHONHASHSEED"] = str(args.hashseed)
                if args.threads is not None:
                    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                        env[var] = str(args.threads)

                cmd = [sys.executable, os.path.abspath(__file__),
                       f"--{mode}", "--src", src_path, "--tgt", tgt_path]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

                line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
                try:
                    rec = json.loads(line)
                except Exception:
                    rec = {"mode": mode, "env": {"pid": -1},
                           "error_text": "child produced no JSON",
                           "stderr_tail": proc.stderr.strip()[-800:]}

                rec["run_index"] = i
                log.write(json.dumps(rec) + "\n")
                log.flush()

                if "error_text" in rec:
                    print(f"  {mode} {i + 1:>3}/{args.runs}  FAILED: "
                          f"{rec['error_text'][:90]}")
                elif mode == "worker":
                    err = rec.get("error", {}).get("translation_m")
                    print(f"  {mode} {i + 1:>3}/{args.runs}  "
                          f"fitness {rec['result']['fitness']:.4f}  "
                          f"err {'n/a' if err is None else f'{err * 100:.1f}cm'}  "
                          f"{rec['result']['seeded_from']}")
                else:
                    print(f"  {mode} {i + 1:>3}/{args.runs}  "
                          f"walls {rec['walls']['src']['count']}/"
                          f"{rec['walls']['tgt']['count']}  "
                          f"level {rec['level']['src']['rotation_digest']}")

                collected[mode].append(rec)

    for mode in modes:
        summarise([r for r in collected[mode] if "error_text" not in r],
                  f"{mode} population")

    print(f"\nfull log: {os.path.abspath(args.log)}")
    print("\nHow to read this:")
    print("  probe stable + worker varying -> the variation is inside coarse")
    print("     registration and the global random stream it draws from.")
    print("  probe varying -> it is upstream, and the first disagreeing stage")
    print("     named above is where to look.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
