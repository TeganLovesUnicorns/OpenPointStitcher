"""Re-processing one scan with its removals: the preview is rebuilt without the removed points, cached apart from
the plain one, and what changed is reported in plain words."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import build_bench  # noqa: E402
from open_point_stitch.placement import Exclusion  # noqa: E402
from open_point_stitch.server import _describe_reprocess, _reprocessed_exclusions  # noqa: E402

# A plan view at 100 px per metre (as in test_exclusions.py): raw (x, y, z) -> screen px, raw -> world.
PLAN_A = [[100, 0, 0, 0], [0, -100, 0, 1000]]
PLAN_B = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]


def _write_room_with_sofa(path: str) -> None:
    """A 6 x 4 m room (floor, four walls) with a 2.5 m long, 1 m high sofa back standing across the middle."""
    rng = np.random.default_rng(5)
    n = 6000
    parts = [np.column_stack([rng.uniform(0, 6, n), rng.uniform(0, 4, n), rng.normal(0, 0.004, n)])]
    for (x0, y0, x1, y1) in ((0, 0, 6, 0), (0, 4, 6, 4), (0, 0, 0, 4), (6, 0, 6, 4)):
        t = rng.uniform(0, 1, n)
        parts.append(np.column_stack([x0 + t * (x1 - x0) + rng.normal(0, .004, n),
                                      y0 + t * (y1 - y0) + rng.normal(0, .004, n), rng.uniform(0, 2.6, n)]))
    parts.append(np.column_stack([rng.uniform(1.75, 4.25, n), 2.0 + rng.normal(0, .004, n), rng.uniform(0, 1.0, n)]))
    pts = np.vstack(parts)
    with open(path, "w") as fh:
        fh.write(f"{len(pts)}\n")
        for p in pts:
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} -1 128 128 128\n")


def _sofa_removal() -> Exclusion:
    """A lasso round the sofa back in plan view (x 1.6 to 4.4 m, y 1.8 to 2.2 m), as the review view stores it."""
    poly = [[160, 820], [440, 820], [440, 780], [160, 780]]
    return Exclusion.from_dict({"id": "review-1", "scan": "room", "points": [], "centroid": [3, 2, .5],
                                "extent": [3, .4, 1], "n_points": 0, "source": "review",
                                "shapes": [{"a": PLAN_A, "b": PLAN_B, "poly": poly}]})


def _walls_near_y2(payload) -> int:
    return sum(1 for w in payload["walls"] if abs((w["y1"] + w["y2"]) / 2 - 2.0) < 0.3 and abs(w["y1"] - w["y2"]) < 0.3)


def test_re_processing_finds_the_walls_again_without_the_removed_points():
    with tempfile.TemporaryDirectory() as tmp:
        path, cache = os.path.join(tmp, "room.pts"), os.path.join(tmp, ".bench_cache")
        _write_room_with_sofa(path)
        plain, _ = build_bench.build_scan_payload_cached(path, cache)
        assert _walls_near_y2(plain) == 1, f"the sofa back should be found as a wall first: {plain['walls']}"
        edited, cached = build_bench.build_scan_payload_cached(path, cache, [_sofa_removal()])
        assert not cached and _walls_near_y2(edited) == 0, edited["walls"]
        removed = plain["points"] - edited["points"]                  # the sofa, and the floor under it (plan lasso)
        assert 6000 < removed < 6000 + 400, (plain["points"], edited["points"])
        again, cached = build_bench.build_scan_payload_cached(path, cache, [_sofa_removal()])
        assert cached and again == edited, "the re-processed preview was not served from its own cache"
        plain_again, cached = build_bench.build_scan_payload_cached(path, cache)
        assert cached and plain_again == plain, "re-processing disturbed the plain preview's cache"
        assert sorted(os.listdir(cache)) == ["edited", "room.json"], os.listdir(cache)

        changes = _describe_reprocess(plain, edited)
        assert changes["walls_gone"] == 1 and changes["walls_new"] == 0, changes
        text = " ".join(changes["lines"])
        assert f"{removed:,} left out" in text and "1 no longer found (2.5 m long)" in text, text


def _write_room_with_stairs(path: str) -> None:
    """A 6 x 4 m room (floor, four walls) with eight 20 cm steps going DOWN (a stair to a basement) along x from 1.0
    to 3.0 m, y 0.5 to 1.5 m."""
    rng = np.random.default_rng(6)
    n = 6000
    parts = [np.column_stack([rng.uniform(0, 6, n), rng.uniform(0, 4, n), rng.normal(0, 0.004, n)])]
    for (x0, y0, x1, y1) in ((0, 0, 6, 0), (0, 4, 6, 4), (0, 0, 0, 4), (6, 0, 6, 4)):
        t = rng.uniform(0, 1, n)
        parts.append(np.column_stack([x0 + t * (x1 - x0) + rng.normal(0, .004, n),
                                      y0 + t * (y1 - y0) + rng.normal(0, .004, n), rng.uniform(0, 2.6, n)]))
    for k in range(8):
        x = 1.0 + 0.25 * k
        parts.append(np.column_stack([rng.uniform(x, x + 0.25, 1500), rng.uniform(0.5, 1.5, 1500),
                                      np.full(1500, -0.2 * (k + 1)) + rng.normal(0, .003, 1500)]))
    pts = np.vstack(parts)
    with open(path, "w") as fh:
        fh.write(f"{len(pts)}\n")
        for p in pts:
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} -1 128 128 128\n")


def _stairs_mark() -> Exclusion:
    """A plan-view outline round the staircase (x 0.9 to 3.1 m, y 0.4 to 1.6 m)."""
    poly = [[90, 960], [310, 960], [310, 840], [90, 840]]
    return Exclusion.from_dict({"id": "stairs-1", "scan": "room", "points": [], "centroid": [2, 1, .8],
                                "extent": [0, 0, 0], "n_points": 0, "source": "stairs",
                                "shapes": [{"a": PLAN_A, "b": PLAN_B, "poly": poly}]})


def test_marked_stairs_stay_in_the_scan_but_are_not_taken_for_floors():
    from open_point_stitch.find_floors_ceilings import without_stairs
    with tempfile.TemporaryDirectory() as tmp:
        path, cache = os.path.join(tmp, "room.pts"), os.path.join(tmp, ".bench_cache")
        _write_room_with_stairs(path)
        plain, _ = build_bench.build_scan_payload_cached(path, cache)
        assert plain["floor_z"] < -0.5, f"unmarked, the stair down should drag the floor down: {plain['floor_z']}"
        marked, _ = build_bench.build_scan_payload_cached(path, cache, stairs=[_stairs_mark()])
        assert marked["points"] == plain["points"] and marked["preview_points"] == plain["preview_points"]
        assert abs(marked["floor_z"]) < 0.05, f"marked, the floor should be the room's own: {marked['floor_z']}"
        assert marked["zRange"][0] < -1.4, "the stairs should still be inside the scan's height range"
        idx = marked["stairs"]
        xy = np.asarray(marked["xy"])[idx]
        assert len(idx) > 100 and np.all((xy[:, 0] > 0.85) & (xy[:, 0] < 3.15)), (len(idx), xy.min(0), xy.max(0))
        settled = without_stairs(marked)
        assert len(settled["h"]) == len(marked["h"]) - len(idx), "height settle still sees the stairs"
        assert np.asarray(settled["h"]).min() > -0.1, "a tread is still among the points height settle uses"


def test_a_scan_is_rebuilt_with_the_removals_it_was_re_processed_with_only():
    saved = {"scans": [{"name": "room", "reprocessed_with": ["review-1"]}, {"name": "other"}],
             "exclusions": [vars(_sofa_removal()),
                            {"id": "review-2", "scan": "room", "points": [], "centroid": [0, 0, 0], "extent": [1, 1, 1]}]}
    got = _reprocessed_exclusions(saved, "room")
    assert [e.id for e in got] == ["review-1"] and got[0].shapes, got
    assert _reprocessed_exclusions(saved, "other") is None


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
