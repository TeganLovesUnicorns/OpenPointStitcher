"""End-to-end acceptance test: runs the real pipeline and checks the output read back from disk."""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

# Three synthetic scans with known ground truth.
FIXTURE_DIR = os.path.join(HERE, "fixtures", "acceptance_synth")


def _fixture_files() -> list[str]:
    return sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.pts")))


def _validate_output(path: str, expected_n_points: int) -> list[str]:
    """Check a merged file: exact point count, colour present, index-aligned, not uniform.
    Returns problem strings (empty means clean) so the canary test can reuse it."""
    from open_point_stitch.pts_file import read_pts

    problems = []
    try:
        merged = read_pts(path)
    except Exception as exc:  # noqa: BLE001 - "does not parse" is itself a result to report
        return [f"did not parse back through our own reader: {type(exc).__name__}: {exc}"]

    if merged.n_points != expected_n_points:
        problems.append(
            f"{merged.n_points:,} points, expected exactly {expected_n_points:,} - "
            f"something thinned or dropped points, or added some that should not be there"
        )
    if merged.colors is None:
        problems.append("no colour in the output at all")
    elif merged.colors.shape[0] != merged.n_points:
        problems.append(
            f"colour array has {merged.colors.shape[0]} rows for {merged.n_points} points "
            f"- not index-aligned to the coordinates"
        )
    elif merged.colors.std() < 0.01:
        # Scan.colors is 0..1 float; near-zero spread means stripped or zeroed colour.
        problems.append(
            f"colour has almost no variance (std={merged.colors.std():.4f}) - "
            f"looks stripped or zeroed, not merely coincidentally uniform"
        )
    return problems


def test_full_pipeline_output_is_correct_on_disk():
    """Runs the real `stitch` CLI end to end and checks the file it wrote."""
    from open_point_stitch.cli import main as cli_main
    from open_point_stitch.pts_file import read_pts

    files = _fixture_files()
    assert len(files) == 3, f"expected the 3-scan fixture, found {len(files)}: {files}"
    total_in = sum(read_pts(p).n_points for p in files)

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "merged.pts")
        report_path = os.path.join(d, "report.json")
        rc = cli_main(["stitch", FIXTURE_DIR, "-o", out_path, "--report", report_path])
        assert rc == 0, f"stitch exited {rc}, expected 0"
        assert os.path.exists(out_path), "stitch exited 0 but wrote no output file"

        problems = _validate_output(out_path, total_in)
        assert not problems, "acceptance check(s) failed: " + "; ".join(problems)

        with open(report_path) as fh:
            report = json.load(fh)
        assert report["merged_points"] == read_pts(out_path).n_points, (
            "the report's own point count disagrees with the file it just wrote"
        )

        # All three viewpoints share one room, so they must form ONE component.
        pg = report["pose_graph"]
        assert pg["connected"] is True, f"pose graph did not connect: {pg['components']}"
        largest = max(pg["components"], key=len)
        assert len(largest) == 3, (
            f"expected all 3 scans in the largest connected component, got {pg['components']}"
        )


def test_export_keeps_exclude_regions_and_cuts_exactly_the_removals():
    """On the written file: an exclude region stays in by default and is cut exactly when the setting says so; a
    review removal is always cut exactly. Counted independently with numpy; identity placement on purpose."""
    from open_point_stitch.config import PipelineConfig
    from open_point_stitch.merge_placed import place_merge
    from open_point_stitch.placement import SCHEMA
    from open_point_stitch.pts_file import read_pts

    files = _fixture_files()
    names = [os.path.splitext(os.path.basename(p))[0] for p in files]
    target_name, target_path = names[0], files[0]
    raw = read_pts(target_path)
    total_in = sum(read_pts(p).n_points for p in files)

    # Box over the lower third of the scan's x-range, measured from its own points.
    lo, hi = raw.points.min(axis=0), raw.points.max(axis=0)
    x_cut = lo[0] + (hi[0] - lo[0]) / 3.0
    centroid = [(lo[0] + x_cut) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0]
    extent = [x_cut - lo[0], (hi[1] - lo[1]) + 1.0, (hi[2] - lo[2]) + 1.0]
    half = np.maximum(np.asarray(extent) / 2.0, 0.05)
    expected_excluded = int(np.all(np.abs(raw.points - np.asarray(centroid)) <= half, axis=1).sum())
    assert 0 < expected_excluded < raw.n_points, (
        f"setup: the exclusion box should cut some but not all of {target_name}'s "
        f"{raw.n_points:,} points; it cut {expected_excluded}"
    )

    region = {"id": "acceptance-test-region", "scan": target_name, "points": [], "centroid": centroid,
              "extent": extent, "n_points": expected_excluded}
    cases = (("exclude region, default", "", {}, total_in),
             ("exclude region, left out", "", {"keep_excluded": False}, total_in - expected_excluded),
             ("review removal", "review", {}, total_in - expected_excluded))
    with tempfile.TemporaryDirectory() as d:
        for label, source, kwargs, expected_out in cases:
            placement_path = os.path.join(d, "placement.json")
            with open(placement_path, "w") as fh:
                json.dump({"schema": SCHEMA, "scans": [{"name": n} for n in names],
                           "exclusions": [dict(region, source=source)]}, fh)
            out_path = os.path.join(d, "merged.pts")
            place_merge(FIXTURE_DIR, placement_path, out_path, PipelineConfig(), **kwargs)
            problems = _validate_output(out_path, expected_out)
            assert not problems, f"{label}: acceptance check(s) failed: " + "; ".join(problems)
            print(f"    {label}: {expected_out:,} points written, as expected")


def test_canary_detects_a_broken_output():
    """_validate_output must catch three deliberately broken files built from a real merge."""
    from open_point_stitch.cli import main as cli_main
    from open_point_stitch.pts_file import read_pts, write_pts

    files = _fixture_files()
    total_in = sum(read_pts(p).n_points for p in files)

    with tempfile.TemporaryDirectory() as d:
        good_path = os.path.join(d, "merged.pts")
        rc = cli_main(["stitch", FIXTURE_DIR, "-o", good_path,
                       "--report", os.path.join(d, "report.json")])
        assert rc == 0, f"setup: stitch exited {rc}"
        assert not _validate_output(good_path, total_in), (
            "setup: the real merge should already be clean - fix the pipeline "
            "before trusting this canary's negative results"
        )
        good = read_pts(good_path)

        # Fault 1: a colour column dropped.
        no_colour_path = os.path.join(d, "no_colour.pts")
        write_pts(no_colour_path, good.points, colors=None)
        assert _validate_output(no_colour_path, total_in), (
            "CANARY FAILED: a merge with no colour at all passed validation"
        )

        # Fault 2: an exclusion region that should have removed points did
        # not - simulated directly as a merge silently missing points it
        # should have written.
        dropped_path = os.path.join(d, "dropped.pts")
        write_pts(dropped_path, good.points[:-500], good.colors[:-500])
        assert _validate_output(dropped_path, total_in), (
            "CANARY FAILED: a merge silently short 500 points passed validation"
        )

        # Fault 3: colour present but zeroed - the right shape, wrong
        # values, the kind of fault an index-alignment-only check would miss.
        zeroed_path = os.path.join(d, "zeroed_colour.pts")
        write_pts(zeroed_path, good.points, np.zeros_like(good.colors))
        assert _validate_output(zeroed_path, total_in), (
            "CANARY FAILED: a merge with every point coloured identically passed validation"
        )


# The real amp room and small bar pair (about 42 MB, not in the repository).
REGRESSION_SCANS_DIR = os.environ.get("OPENPOINT_REGRESSION_SCANS", "/scans-regression")


def _find_ampbar(folder: str) -> tuple[str | None, str | None]:
    amp_path = bar_path = None
    for p in glob.glob(os.path.join(folder, "*.pts")):
        low = os.path.basename(p).lower()
        if "amp" in low:
            amp_path = p
        elif "bar" in low:
            bar_path = p
    return amp_path, bar_path


def test_confirmed_pair_direct_overlap_matches_the_stated_band():
    """Direct overlap at the confirmed transform stays in its expected band."""
    amp_path, bar_path = _find_ampbar(REGRESSION_SCANS_DIR)
    if not amp_path or not bar_path:
        print(f"SKIP: {REGRESSION_SCANS_DIR} does not have both the real AMP room "
              f"and small bar scans - this machine's local scans-regression/ folder "
              f"only, never committed to the repo (too large).")
        return

    from open_point_stitch.overlap import direct_overlap
    from open_point_stitch.pts_file import read_pts

    amp = read_pts(amp_path)
    bar = read_pts(bar_path)
    truth_path = os.path.join(HERE, "fixtures", "ampbar_confirmed.json")
    with open(truth_path) as fh:
        truth = json.load(fh)
    transform = np.array(truth["transform"])  # bar RAW -> amp RAW, confirmed correct

    result = direct_overlap(bar.points, transform, amp.points)
    frac_15cm = result["within_15cm_frac"]
    LOW, HIGH = 0.40, 0.50
    assert LOW <= frac_15cm <= HIGH, (
        f"direct overlap at the confirmed transform is {frac_15cm*100:.1f}% within "
        f"15cm, expected {LOW*100:.0f}-{HIGH*100:.0f}%. Outside this band means "
        f"either the confirmed transform, the scan files, or direct_overlap() "
        f"itself changed - a real correctness question, not noise."
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
