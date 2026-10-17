"""Tests for measure_walls.py: scans that agree about a wall read as a thin sheet; scans that do not read as a band,
and the report says by how much and which scan is off."""
from __future__ import annotations

import os
import sys
from concurrent.futures import CancelledError

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.measure_walls import cluster_walls, measure_walls, summarise, _segments  # noqa: E402
from open_point_stitch.placement import Placement  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402

INCH = 0.0254


def _wall_scan(name: str, wall_y_local: float, seed: int, along=(0.0, 10.0)) -> Scan:
    """A wall along x at local y = wall_y_local, 0.5 cm noise, and a floor, seen from the origin."""
    rng = np.random.default_rng(seed)
    n = 30000
    wall = np.column_stack([rng.uniform(*along, n), wall_y_local + rng.normal(0, 0.005, n), rng.uniform(0.0, 3.0, n)])
    floor = np.column_stack([rng.uniform(*along, 3000), rng.uniform(-3, 3, 3000), np.zeros(3000)])
    return Scan(name=name, path=f"{name}.pts", points=np.vstack([wall, floor]))


def _payload(wall_y_local: float, along=(0.0, 10.0)) -> dict:
    return {"walls": [{"i": 0, "x1": along[0], "y1": wall_y_local, "x2": along[1], "y2": wall_y_local, "len": along[1] - along[0], "h": 3.0, "pts": 30000}]}


def _scene(offsets_and_sides):
    """Scans each standing 3 m from the wall (on the side given) whose face lands `offset` metres from y = 0."""
    scans, placements, payloads = [], {}, {}
    for k, (name, offset, side) in enumerate(offsets_and_sides):
        # A scan on the -y side stands at y = -3 and sees the wall 3 m ahead; on the +y side, at y = +3 and 3 m behind.
        stand = -3.0 * side
        wall_local = (offset - stand)
        scans.append(_wall_scan(name, wall_local, k))
        placements[name] = Placement(name=name, ty=stand)
        payloads[name] = _payload(wall_local)
    return scans, placements, payloads


def test_should_cancel_stops_between_scans_and_raises_cancelled_error():
    # Checked once per scan (placing every point at full resolution, before clustering) and once per wall cluster.
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1)])
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 1  # let scan 0 through, stop before scan 1

    try:
        measure_walls(scans, placements, payloads, should_cancel=should_cancel)
        raise AssertionError("expected CancelledError, measure_walls returned normally")
    except CancelledError as exc:
        assert "1/2" in str(exc), f"should say how far it got: {exc}"


def test_scans_that_agree_read_as_a_thin_sheet():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1)])
    reports = measure_walls(scans, placements, payloads)
    assert len(reports) == 1 and reports[0].thickness_in < 1.0, [(r.thickness_in, r.length_m) for r in reports]
    assert all(abs(s.offset_in) < 0.3 for s in reports[0].scans)


def test_scans_that_disagree_read_as_a_band_and_the_report_says_by_how_much_each():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.08, -1)])
    r = measure_walls(scans, placements, payloads)[0]
    assert 3.0 < r.thickness_in < 4.0, r.thickness_in                   # 8 cm apart, in inches
    offsets = {s.scan: s.offset_in for s in r.scans}
    assert abs(offsets["A"] + 0.04 / INCH) < 0.3 and abs(offsets["B"] - 0.04 / INCH) < 0.3, offsets
    assert r.body_in is None and abs(r.length_m - 9.6) < 0.2


def test_scans_on_opposite_sides_report_the_wall_body_not_a_thick_sheet():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1), ("C", 0.15, 1)])
    r = measure_walls(scans, placements, payloads)[0]
    assert r.thickness_in < 1.0, f"the two scans on the same side agree: {r.thickness_in}"
    assert r.body_in is not None and abs(r.body_in - 0.15 / INCH) < 1.0, r.body_in


def test_a_wall_across_the_first_is_a_different_wall():
    a = {"scan": "A", "a": np.array([0.0, 0.0]), "b": np.array([10.0, 0.0]), "u": np.array([1.0, 0.0]), "n": np.array([0.0, 1.0]), "length": 10.0}
    b = {"scan": "B", "a": np.array([5.0, -2.0]), "b": np.array([5.0, 2.0]), "u": np.array([0.0, 1.0]), "n": np.array([-1.0, 0.0]), "length": 4.0}
    c = {"scan": "C", "a": np.array([0.0, 0.1]), "b": np.array([10.0, 0.1]), "u": np.array([1.0, 0.0]), "n": np.array([0.0, 1.0]), "length": 10.0}
    groups = sorted(len(g) for g in cluster_walls([a, b, c]))
    assert groups == [1, 2], groups


def test_a_scan_with_too_few_points_on_the_wall_is_left_out():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1)])
    scans[1] = Scan(name="B", path="B.pts", points=scans[1].points[:50])
    r = measure_walls(scans, placements, payloads)[0]
    assert [s.scan for s in r.scans] == ["A"]


def test_no_detected_walls_gives_an_empty_report_and_summary():
    scans, placements, _ = _scene([("A", 0.0, -1)])
    reports = measure_walls(scans, placements, {"A": {"walls": []}})
    assert reports == [] and summarise(reports) == {"walls": 0}


def test_the_summary_counts_walls_too_thick_to_draw_from():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.08, -1)])
    s = summarise(measure_walls(scans, placements, payloads))
    assert s["walls"] == 1 and s["over_2_in"] == 1 and s["over_6_in"] == 0, s


def test_segments_are_placed_in_the_world_frame():
    payloads = {"A": _payload(3.0)}
    seg = _segments(payloads, {"A": Placement(name="A", ty=-3.0, tx=2.0)})[0]
    assert abs(seg["a"][1]) < 1e-9 and abs(seg["a"][0] - 2.0) < 1e-9 and abs(seg["length"] - 10.0) < 1e-9


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
