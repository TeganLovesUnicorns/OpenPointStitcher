"""Tests for merge_placed.py's should_cancel: checked between full-resolution scan loads and between refine links,
where a large export spends its time."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import CancelledError

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.merge_placed import place_merge  # noqa: E402
from open_point_stitch.placement import SCHEMA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "fixtures", "acceptance_synth")
NAMES = ["scan_00", "scan_01", "scan_02"]  # the three real .pts files that fixture holds


def _placement_file(d: str) -> str:
    path = os.path.join(d, "placement.json")
    with open(path, "w") as fh:
        json.dump({"schema": SCHEMA, "scans": [{"name": n} for n in NAMES]}, fh)
    return path


def test_should_cancel_stops_between_scan_loads_and_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        placement_path = _placement_file(d)
        out_path = os.path.join(d, "merged.pts")
        calls = {"n": 0}

        def should_cancel():
            calls["n"] += 1
            return calls["n"] > 1  # let the first scan load through, stop before the second

        try:
            place_merge(FIXTURE_DIR, placement_path, out_path, PipelineConfig(),
                       should_cancel=should_cancel)
            raise AssertionError("expected CancelledError, place_merge returned normally")
        except CancelledError as exc:
            assert "1/3" in str(exc), f"should say how far it got: {exc}"
        assert not os.path.exists(out_path), (
            "a lossless export must write nothing at all on cancel, not a partial file")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
