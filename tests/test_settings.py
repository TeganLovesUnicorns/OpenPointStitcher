"""Tests for the per-stage settings: the levelling tilt limit and the alignment-only range limit, and that both
leave the defaults exactly as they were."""
from __future__ import annotations

import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from open_point_stitch import align_gravity  # noqa: E402
from open_point_stitch.align_gravity import estimate_level, set_default_max_tilt  # noqa: E402
from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.prepare import _compute_prepared  # noqa: E402


def _tilted_floor(deg: float, seed: int = 0) -> o3d.geometry.PointCloud:
    """A large floor tipped `deg` about x, with a wall, so it is the dominant horizontal-ish surface."""
    rng = np.random.default_rng(seed)
    n = 20000
    x, y = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)
    floor = np.column_stack([x, y * np.cos(np.radians(deg)), y * np.sin(np.radians(deg))])
    wall = np.column_stack([rng.uniform(-5, 5, 3000), np.full(3000, 5.0), rng.uniform(0, 3, 3000)])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([floor, wall])))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
    return pcd


def test_the_default_tilt_limit_is_unchanged_at_fifteen():
    assert align_gravity.DEFAULT_MAX_TILT_DEG == 15.0


def test_a_lower_tilt_limit_refuses_a_floor_steeper_than_it():
    steep = _tilted_floor(8.0)
    try:
        set_default_max_tilt(15.0)
        found = estimate_level(steep)
        set_default_max_tilt(3.0)
        refused = estimate_level(steep)
    finally:
        set_default_max_tilt(15.0)
    assert found.reliable and abs(found.tilt_deg - 8.0) < 1.0, (found.reliable, found.tilt_deg)
    assert not refused.reliable, f"an 8 degree floor was accepted under a 3 degree limit (tilt {refused.tilt_deg:.1f})"


def test_an_explicit_limit_beats_the_default_and_nonsense_is_refused():
    try:
        set_default_max_tilt(3.0)
        assert estimate_level(_tilted_floor(8.0), max_tilt_deg=15.0).reliable
    finally:
        set_default_max_tilt(15.0)
    for bad in (0.0, 90.0):
        try:
            set_default_max_tilt(bad)
        except ValueError:
            continue
        raise AssertionError(f"a tilt limit of {bad} degrees should be refused")


def test_previews_built_under_one_tilt_limit_are_not_reused_under_another():
    import build_bench
    here = os.path.abspath(__file__)
    try:
        set_default_max_tilt(15.0)
        a = build_bench._cache_key(here)
        set_default_max_tilt(3.0)
        b = build_bench._cache_key(here)
    finally:
        set_default_max_tilt(15.0)
    assert a != b and a["level_tilt_deg"] == 15.0 and b["level_tilt_deg"] == 3.0


def _two_clusters():
    rng = np.random.default_rng(3)
    near = np.column_stack([rng.uniform(-3, 3, 4000), rng.uniform(-3, 3, 4000), rng.uniform(0, 2, 4000)])
    far = np.column_stack([rng.uniform(9, 12, 4000), rng.uniform(9, 12, 4000), rng.uniform(0, 2, 4000)])
    return np.vstack([near, far])


def test_a_range_limit_leaves_out_far_points_when_aligning_and_off_keeps_them():
    pts = _two_clusters()
    everything, _, n_all, _ = _compute_prepared("t", pts, None, PipelineConfig())
    limited, _, n_limited, _ = _compute_prepared("t", pts, None, PipelineConfig(max_range_m=6.0))
    reach_all = np.hypot(*np.asarray(everything.points)[:, :2].T).max()
    reach_limited = np.hypot(*np.asarray(limited.points)[:, :2].T).max()
    assert reach_all > 12 and reach_limited < 6.5, (reach_all, reach_limited)
    assert n_all == n_limited == len(pts), "the recorded full-resolution count is the scan's own, filter or not"


def test_a_range_limit_that_would_leave_nothing_is_ignored_rather_than_emptying_the_scan():
    pts = _two_clusters()[4000:]                       # only the far cluster
    down, _, _, _ = _compute_prepared("t", pts, None, PipelineConfig(max_range_m=2.0))
    assert len(down.points) > 100


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
