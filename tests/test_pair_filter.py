"""Tests for pair_filter.feature_surface_pairs on synthetic rooms."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402
from open_point_stitch.placement import Placement  # noqa: E402
from open_point_stitch.pair_filter import feature_surface_pairs, surface_pairs  # noqa: E402
from test_surfaces import _room_points  # noqa: E402


def _scan(name, pts, rgb):
    return Scan(name=name, path=f"/fake/{name}.pts", points=pts, colors=rgb)


def test_two_scans_of_the_same_room_are_attempted_via_features():
    rng = np.random.default_rng(20)
    pts_a, rgb_a = _room_points(rng)
    true_yaw_deg, true_t = 6.0, np.array([0.4, -0.3, 0.0])
    yaw = np.radians(true_yaw_deg)
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    pts_b = (R @ pts_a.T).T + true_t

    scans = [_scan("a", pts_a, rgb_a), _scan("b", pts_b, rgb_a.copy())]
    placements = {
        "a": Placement(name="a"),
        "b": Placement(name="b", yaw_deg=true_yaw_deg, tx=true_t[0], ty=true_t[1], tz=true_t[2]),
    }
    cfg = PipelineConfig()
    pairs, fractions, _features = feature_surface_pairs(scans, [], placements, cfg)
    assert (0, 1) in pairs, f"expected the pair attempted, fractions={fractions}"
    # Either signal attempting the pair is a pass; "feature+surface" means both agreed.
    assert fractions[(0, 1)]["method"] in ("feature", "feature+surface")
    assert fractions[(0, 1)]["frac"] >= cfg.adjacency_feature_min_frac


def test_two_unrelated_scans_are_not_attempted():
    rng = np.random.default_rng(21)
    pts_a, rgb_a = _room_points(rng)
    pts_b = rng.uniform(-1, 1, (2000, 3)) + np.array([80.0, 80.0, 80.0])
    rgb_b = rng.uniform(0, 1, (2000, 3))

    scans = [_scan("a", pts_a, rgb_a), _scan("b", pts_b, rgb_b)]
    placements = {"a": Placement(name="a"), "b": Placement(name="b")}
    cfg = PipelineConfig()
    pairs, fractions, _features = feature_surface_pairs(scans, [], placements, cfg)
    assert (0, 1) not in pairs, f"expected the pair skipped, fractions={fractions}"


def test_falls_back_to_raw_point_method_when_a_scan_has_too_few_surfaces():
    """A scan of a single wall gives match_surfaces too little to go on, so the pair falls back to the raw-point
    measure and the report says so."""
    rng = np.random.default_rng(22)
    pts_room, rgb_room = _room_points(rng)
    # A single flat wall, offset so it overlaps part of the room's own
    # wall_x plane (x=-3) - just one plane, well under the 2-surface floor.
    n = 3000
    single_wall = np.column_stack([
        np.full(n, -3.0) + rng.normal(0, 0.003, n),
        rng.uniform(-3, 3, n),
        rng.uniform(0, 2.5, n),
    ])
    rgb_wall = np.tile([0.2, 0.25, 0.7], (n, 1)) + rng.normal(0, 0.02, (n, 3))
    rgb_wall = np.clip(rgb_wall, 0, 1)

    scans = [_scan("room", pts_room, rgb_room), _scan("wall_only", single_wall, rgb_wall)]
    placements = {"room": Placement(name="room"), "wall_only": Placement(name="wall_only")}
    cfg = PipelineConfig()
    pairs, fractions, _features = feature_surface_pairs(scans, [], placements, cfg)
    assert fractions[(0, 1)]["method"] == "surface", (
        f"expected the raw-point fallback for a single-surface scan, got {fractions[(0, 1)]}"
    )


def test_union_never_excludes_a_pair_the_raw_point_method_alone_would_attempt():
    """The union never drops a pair surface_pairs() alone caught."""
    rng = np.random.default_rng(25)
    pts_a, rgb_a = _room_points(rng)
    true_yaw_deg, true_t = 4.0, np.array([0.2, -0.15, 0.0])
    yaw = np.radians(true_yaw_deg)
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    pts_b = (R @ pts_a.T).T + true_t

    scans = [_scan("a", pts_a, rgb_a), _scan("b", pts_b, rgb_a.copy())]
    placements = {
        "a": Placement(name="a"),
        "b": Placement(name="b", yaw_deg=true_yaw_deg, tx=true_t[0], ty=true_t[1], tz=true_t[2]),
    }
    cfg = PipelineConfig()
    raw_pairs, _raw_fractions = surface_pairs(scans, [], placements, cfg)
    union_pairs, _fractions, _features = feature_surface_pairs(scans, [], placements, cfg)
    assert set(raw_pairs) <= set(union_pairs), (
        f"union dropped a pair the raw-point method alone attempted: "
        f"raw={raw_pairs} union={union_pairs}"
    )


def test_unplaced_scan_pair_is_always_kept():
    rng = np.random.default_rng(23)
    pts_a, rgb_a = _room_points(rng)
    pts_b, rgb_b = _room_points(np.random.default_rng(24))
    scans = [_scan("a", pts_a, rgb_a), _scan("b", pts_b, rgb_b)]
    placements = {"a": Placement(name="a")}  # "b" has no saved placement at all
    cfg = PipelineConfig()
    pairs, fractions, _features = feature_surface_pairs(scans, [], placements, cfg)
    assert (0, 1) in pairs
    assert (0, 1) not in fractions  # nothing to measure - kept, not scored


def test_feature_surface_pairs_reports_progress_per_placed_scan():
    """The per-scan wall search reports progress, so a long run never looks stuck."""
    rng = np.random.default_rng(20)
    pts_a, rgb_a = _room_points(rng)
    pts_b, rgb_b = _room_points(np.random.default_rng(24))
    scans = [_scan("a", pts_a, rgb_a), _scan("b", pts_b, rgb_b)]
    placements = {"a": Placement(name="a"), "b": Placement(name="b")}
    cfg = PipelineConfig()

    stages = []
    feature_surface_pairs(scans, [], placements, cfg, progress_cb=stages.append)

    assert stages == ["finding walls: 1/2: a", "finding walls: 2/2: b"], stages


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
