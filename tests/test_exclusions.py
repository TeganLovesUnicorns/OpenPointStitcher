"""Tests for open_point_stitch.exclusions: an exclusion region cuts exactly the right points from a scan, and
leaves everything else untouched.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.exclusions import (excluded_mask, filter_scan, filter_scans, for_export,  # noqa: E402
                                          payload_without_excluded)
from open_point_stitch.lib.poses import levelled_points, tilt_matrix3  # noqa: E402
from open_point_stitch.placement import Exclusion  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402


def _scan(name, points, colors=None, intensity=None):
    return Scan(name=name, path=f"/fake/{name}.pts", points=np.asarray(points, dtype=float),
               colors=None if colors is None else np.asarray(colors, dtype=float),
               intensity=None if intensity is None else np.asarray(intensity, dtype=float))


def _region(scan, centroid, extent, n_points=1):
    return Exclusion(id="e1", scan=scan, points=[], centroid=tuple(centroid),
                     extent=tuple(extent), n_points=n_points)


def test_no_regions_masks_nothing():
    pts = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    mask = excluded_mask(pts, [])
    assert mask.tolist() == [False, False]


def test_point_inside_box_is_excluded_point_outside_is_not():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    region = _region("s", centroid=[0, 0, 0], extent=[1, 1, 1])
    mask = excluded_mask(pts, [region])
    assert mask.tolist() == [True, False]


def test_zero_size_extent_still_gets_a_real_margin():
    """A region built from a single click has extent [0,0,0] - the same
    minimum-half-extent floor `geometry.points_in_box` gives anchors, so
    a point a few centimetres from the exact clicked point is still caught."""
    pts = np.array([[0.03, 0.0, 0.0], [1.0, 0.0, 0.0]])
    region = _region("s", centroid=[0, 0, 0], extent=[0, 0, 0])
    mask = excluded_mask(pts, [region])
    assert mask.tolist() == [True, False]


def test_two_regions_union_not_just_the_last_one():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    regions = [_region("s", [0, 0, 0], [1, 1, 1]), _region("s", [10, 10, 10], [1, 1, 1])]
    mask = excluded_mask(pts, regions)
    assert mask.tolist() == [True, True, False]


def test_filter_scan_removes_the_right_points_keeps_colours_aligned():
    scan = _scan("s", points=[[0, 0, 0], [10, 10, 10]],
                colors=[[1, 0, 0], [0, 1, 0]], intensity=[-1, -1])
    out = filter_scan(scan, [_region("s", [0, 0, 0], [1, 1, 1])])
    assert out.n_points == 1
    assert out.points.tolist() == [[10, 10, 10]]
    assert out.colors.tolist() == [[0, 1, 0]]
    assert out.intensity.tolist() == [-1]
    # the input is untouched - filter_scan returns a COPY
    assert scan.n_points == 2


def test_filter_scan_with_no_matching_region_returns_the_same_object():
    scan = _scan("s", points=[[0, 0, 0]])
    other_scans_region = _region("other scan", [0, 0, 0], [1, 1, 1])
    out = filter_scan(scan, [other_scans_region])
    assert out is scan, "no exclusion applies to this scan - should not even copy it"


def test_filter_scan_with_no_regions_at_all_returns_the_same_object():
    scan = _scan("s", points=[[0, 0, 0]])
    out = filter_scan(scan, [])
    assert out is scan


def test_region_covering_every_point_leaves_an_empty_but_valid_scan():
    scan = _scan("s", points=[[0, 0, 0], [0.1, 0, 0]])
    out = filter_scan(scan, [_region("s", [0, 0, 0], [5, 5, 5])])
    assert out.n_points == 0
    assert out.points.shape == (0, 3)


def test_filter_scans_only_touches_the_named_scan():
    a = _scan("a", points=[[0, 0, 0], [10, 10, 10]])
    b = _scan("b", points=[[0, 0, 0], [10, 10, 10]])
    out = filter_scans([a, b], [_region("a", [0, 0, 0], [1, 1, 1])])
    assert out[0].n_points == 1  # a: one point cut
    assert out[1].n_points == 2  # b: untouched


# Outlines drawn in the review view: a plan view at 100 px per metre, y up the screen.
PLAN_A = [[100, 0, 0, 500], [0, -100, 0, 400]]             # raw (x, y, z) -> screen px
PLAN_B = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]        # raw -> world: the scan placed at the origin


def _review(shapes):
    return Exclusion.from_dict({"id": "review-1", "scan": "a", "points": [], "centroid": [0, 0, 0],
                                "extent": [100, 100, 100], "n_points": 0, "source": "review", "shapes": shapes})


def test_a_lasso_cuts_everything_inside_it_top_to_bottom_and_nothing_outside():
    pts = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.2], [1.0, 1.0, 2.5], [3.0, 1.0, 1.2], [1.4, 1.0, 1.0]])
    poly = [[580, 280], [630, 280], [630, 330], [580, 330]]      # x 0.8 to 1.3 m, y 0.7 to 1.2 m
    m = excluded_mask(pts, [_review([{"a": PLAN_A, "b": PLAN_B, "poly": poly}])])
    assert m.tolist() == [True, True, True, False, False], m


def test_the_slice_limits_that_were_on_keep_the_floor_and_ceiling():
    pts = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.2], [1.0, 1.0, 2.5]])
    poly = [[580, 280], [630, 280], [630, 330], [580, 330]]
    shape = {"a": PLAN_A, "b": PLAN_B, "poly": poly, "clip": {"z": [0.1, 2.2]}}
    assert excluded_mask(pts, [_review([shape])]).tolist() == [False, True, False]


def test_a_brush_stroke_and_a_concave_outline_cut_only_what_they_cover():
    pts = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.3, 0.0, 1.0],      # brush at the origin, 10 px
                    [2.5, 2.5, 1.0], [1.5, 2.5, 1.0], [1.5, 1.0, 1.0]])      # inside the L, its arm, the notch
    brush = {"a": PLAN_A, "b": PLAN_B, "circles": [[500, 400, 10]]}
    ell = [[700, 400], [800, 400], [800, 100], [600, 100], [600, 200], [700, 200]]   # an L, notch at x 1-2, y 0-2
    m = excluded_mask(pts, [_review([brush, {"a": PLAN_A, "b": PLAN_B, "poly": ell}])])
    assert m.tolist() == [True, True, False, True, True, False], m


def test_outlines_decide_and_the_drawing_box_does_not_and_survivors_keep_their_colours():
    pts = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 1.0], [6.0, 6.0, 1.0]])
    s = _scan("a", pts, colors=[[1, 1, 1], [2, 2, 2], [3, 3, 3]], intensity=[-1, -1, -1])
    region = _review([{"a": PLAN_A, "b": PLAN_B, "circles": [[600, 300, 5]]}])   # the 100 m box covers all three
    out = filter_scan(s, [region])
    assert out.points.tolist() == [[5.0, 5.0, 1.0], [6.0, 6.0, 1.0]], out.points
    assert out.colors.tolist() == [[2, 2, 2], [3, 3, 3]] and out.intensity.tolist() == [-1, -1]


def test_an_export_keeps_exclude_regions_unless_told_and_always_drops_removals():
    box = _region("a", (0, 0, 0), (1, 1, 1))
    removal = Exclusion(id="r1", scan="a", points=[], centroid=(0, 0, 0), extent=(1, 1, 1), n_points=1, source="review")
    assert not box.is_removal and removal.is_removal
    assert for_export([box, removal]) == [removal]
    assert for_export([box, removal], keep_excluded=False) == [box, removal]
    assert for_export([]) == []


def test_preview_points_lose_exactly_the_raw_points_a_region_holds():
    """A tilted level and a raised floor: the preview's points inside a raw-frame region are cut, every per-point list
    stays in step, and marked stairs are renumbered."""
    rng = np.random.default_rng(4)
    raw = rng.uniform(-3, 3, (400, 3))
    level, floor_z = tilt_matrix3(3.0, -2.0, 0.0), 0.4
    lv = levelled_points(raw, level, floor_z)
    payload = {"name": "A", "level_rotation": level.tolist(), "floor_z": floor_z,
               "xy": lv[:, :2].ravel().tolist(), "h": lv[:, 2].tolist(), "rgb": list(range(400)),
               "agreement": (np.arange(400) / 400).tolist(), "stairs": [0, 5, 6, 399]}
    region = _region("A", (0.0, 0.0, 0.0), (2.0, 2.0, 6.0))
    inside = excluded_mask(raw, [region])
    got = payload_without_excluded(payload, [region])
    assert len(got["h"]) == int((~inside).sum()) and len(got["rgb"]) == len(got["h"])
    assert np.array_equal(got["rgb"], np.arange(400)[~inside]), "colours fell out of step with the points"
    assert np.allclose(np.asarray(got["xy"]).reshape(-1, 2), lv[~inside, :2])
    kept_stairs = [i for i in payload["stairs"] if not inside[i]]
    assert [int(got["rgb"][j]) for j in got["stairs"]] == kept_stairs, "stairs point at the wrong points"
    assert payload_without_excluded(payload, [_region("B", (0, 0, 0), (9, 9, 9))]) is payload


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
