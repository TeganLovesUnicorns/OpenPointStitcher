"""Tests for overlap.direct_overlap on synthetic clouds."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.overlap import direct_overlap, nearest_distances  # noqa: E402


def test_identical_clouds_fully_overlap():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, (500, 3))
    r = direct_overlap(pts, np.eye(4), pts)
    assert r["within_5cm_frac"] == 1.0
    assert r["within_15cm_frac"] == 1.0
    assert r["n_points"] == 500


def test_far_apart_clouds_do_not_overlap():
    rng = np.random.default_rng(1)
    src = rng.uniform(-1, 1, (300, 3))
    tgt = rng.uniform(-1, 1, (300, 3)) + np.array([1000.0, 1000.0, 1000.0])
    r = direct_overlap(src, np.eye(4), tgt)
    assert r["within_5cm_frac"] == 0.0
    assert r["within_15cm_frac"] == 0.0
    assert r["shared_bbox_m"] == [0.0, 0.0, 0.0]


def test_transform_is_applied_before_measuring():
    """A cloud offset by exactly `t` should overlap perfectly once shifted
    back by `src_to_tgt` - if the transform were ignored, this would report
    zero overlap instead."""
    rng = np.random.default_rng(2)
    tgt = rng.uniform(-2, 2, (400, 3))
    t = np.array([5.0, -3.0, 1.0])
    src = tgt - t  # src + t == tgt
    T = np.eye(4)
    T[:3, 3] = t
    r = direct_overlap(src, T, tgt)
    assert r["within_5cm_frac"] > 0.99


def test_partial_overlap_is_between_zero_and_one():
    rng = np.random.default_rng(3)
    shared = rng.uniform(-1, 1, (200, 3))
    src_only = rng.uniform(-1, 1, (200, 3)) + np.array([50.0, 0.0, 0.0])
    tgt_only = rng.uniform(-1, 1, (200, 3)) + np.array([-50.0, 0.0, 0.0])
    src = np.vstack([shared, src_only])
    tgt = np.vstack([shared, tgt_only])
    r = direct_overlap(src, np.eye(4), tgt)
    # Exactly the 200 shared points should land within 5cm (identical
    # coordinates in both clouds); the other 200 should not.
    assert r["within_5cm_n"] == 200
    assert 0.0 < r["within_5cm_frac"] < 1.0


def test_thresholds_are_configurable():
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1, 1, (100, 3))
    r = direct_overlap(pts, np.eye(4), pts, thresholds=(0.01, 1.0, 2.0))
    assert "within_1cm_frac" in r
    assert "within_100cm_frac" in r
    assert "within_200cm_frac" in r


def test_nearest_distances_zero_for_shared_points():
    """The deviation heatmap's maths: a point present in both clouds is exactly 0 m from its nearest neighbour."""
    rng = np.random.default_rng(5)
    pts = rng.uniform(-2, 2, (200, 3))
    d = nearest_distances(pts, pts)
    assert np.allclose(d, 0.0)


def test_nearest_distances_matches_known_offset():
    """A single target point at a known distance should be picked up exactly
    - this is the number the heatmap colours by, so it has to be right, not
    just non-crashing."""
    src = np.array([[0.0, 0.0, 0.0]])
    tgt = np.array([[0.0, 0.0, 0.30]])  # 30cm straight up
    d = nearest_distances(src, tgt)
    assert abs(d[0] - 0.30) < 1e-9


def test_nearest_distances_empty_target_is_infinite():
    """No overlapping scan to compare against - the heatmap's own "grey,
    nothing to compare" case, not a crash or a false zero."""
    src = np.zeros((5, 3))
    tgt = np.zeros((0, 3))
    d = nearest_distances(src, tgt)
    assert len(d) == 5
    assert np.all(np.isinf(d))


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
