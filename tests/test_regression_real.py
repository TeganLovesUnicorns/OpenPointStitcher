"""Regression test against a real alignment confirmed by hand. Needs the full-resolution scans: set
OPENPOINT_TEST_SCANS to their folder. Skips when the variable is unset; fails when it points somewhere wrong."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

FIXTURE = os.path.join(HERE, "fixtures", "ampbar_confirmed.json")

# Tolerances. The pair has very little shared surface, so a few centimetres of
# wobble between library versions is acceptable. Metres of movement is not.
MAX_TRANSLATION_ERROR_M = 0.10
MAX_ROTATION_ERROR_DEG = 1.0


def _find_scans(folder: str) -> tuple[str, str]:
    """Locate the two scans, tolerating renames and thinning suffixes."""
    import glob
    # Sorted, matching tools/hunt_nondeterminism.py, so both tools always pick the same files.
    files = sorted(glob.glob(os.path.join(folder, "*.pts")))
    src = tgt = None
    for f in files:
        low = os.path.basename(f).lower()
        if "bar" in low:
            src = f
        elif "amp" in low:
            tgt = f
    return src, tgt


def main() -> int:
    folder = os.environ.get("OPENPOINT_TEST_SCANS")
    if not folder:
        print("SKIP: set OPENPOINT_TEST_SCANS to a folder holding the full "
              "resolution AMP room and small bar scans.")
        return 0
    if not os.path.isdir(folder):
        # The variable is set, so this test was meant to run: a wrong path must fail, not skip.
        print(f"FAIL: OPENPOINT_TEST_SCANS points at {folder}, which does not exist.")
        return 1

    src_path, tgt_path = _find_scans(folder)
    if not src_path or not tgt_path:
        # An existing folder missing the pair is a failure, not a skip.
        print(f"FAIL: OPENPOINT_TEST_SCANS points at {folder}, which exists, but it "
              f"does not contain both scans. Need one file with 'bar' in the "
              f"name and one with 'amp'.")
        return 1

    truth = json.load(open(FIXTURE))
    expected = np.array(truth["transform"])

    from open_point_stitch.config import PipelineConfig
    from open_point_stitch.prepare import preprocess_all
    from open_point_stitch.align_pair import register_pair
    from open_point_stitch.pts_file import read_pts
    from open_point_stitch.limits import apply_seed
    from open_point_stitch.report import transform_error_against

    cfg = PipelineConfig()
    apply_seed(cfg.random_seed)

    src = read_pts(src_path)
    tgt = read_pts(tgt_path)

    # Guard against being handed thinned copies, which is how this went wrong
    # once already: thinning changed the answer by nearly three metres.
    for scan, expected_n, label in ((src, truth["source_points"], "source"),
                                    (tgt, truth["target_points"], "target")):
        if scan.n_points < expected_n * 0.9:
            print(f"SKIP: the {label} scan has {scan.n_points:,} points but the "
                  f"confirmed answer was measured on {expected_n:,}. This looks "
                  f"like a thinned copy. Thinning changes the answer on this "
                  f"pair, so the test would be meaningless. Use the originals.")
            return 0

    # In-process repeats are bit identical; cross-process checks need separate invocations.
    repeats = int(os.environ.get("OPENPOINT_TEST_REPEATS", "1"))
    prep = preprocess_all([src, tgt], cfg)

    print(f"{'run':>4} {'fitness':>8} {'rmse':>8} {'tilt':>6} {'trans err':>11} "
          f"{'rot err':>9}  strategy")
    outcomes = []
    for attempt in range(repeats):
        result = register_pair(prep[0], prep[1], cfg)
        # transform_error_against matches by scan name, so source/target order does not matter.
        t_err, r_err = transform_error_against(
            result.transform, result.source_name, result.target_name,
            expected, truth["source"], truth["target"])
        ok = (t_err <= MAX_TRANSLATION_ERROR_M
              and r_err <= MAX_ROTATION_ERROR_DEG
              and result.fitness >= truth["expected_fitness_at_least"])
        outcomes.append((ok, t_err, r_err, result.fitness))
        print(f"{attempt + 1:>4} {result.fitness:>8.4f} {result.rmse:>8.4f} "
              f"{result.tilt_deg:>6.2f} {t_err * 100:>9.1f}cm {r_err:>8.2f}d  "
              f"{result.seeded_from}{'' if ok else '   <-- WRONG'}")

    good = sum(1 for o in outcomes if o[0])
    spread = max(o[1] for o in outcomes) - min(o[1] for o in outcomes)
    print()
    print(f"{good} of {repeats} runs landed on the confirmed answer")
    print(f"spread between best and worst run: {spread * 100:.1f} cm")

    if good == repeats:
        print("PASS")
        return 0
    if good == 0:
        print("FAIL: every run missed the confirmed answer. Something is wrong, "
              "not merely unstable.")
        return 1
    print("FAIL: the result is unstable. Same input, same code, different "
          "answers. This is the known nondeterminism in plane fitting, and it "
          "is the thing to fix before trusting any output.")
    return 1


def test_regression():
    """Thin pytest wrapper around main()."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
