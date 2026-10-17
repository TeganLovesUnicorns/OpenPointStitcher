"""Tests for find_surfaces.py: planar-surface detection at any orientation, the neighbour graph, cross-scan
matching and the pose solve. Synthetic geometry only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_surfaces import (  # noqa: E402
    segment_surfaces, build_neighbor_graph, match_surfaces, solve_pose_from_surfaces,
    extract_scan_features, feature_pair_score, ScanFeatures,
)


def _room_points(rng, n_per_surface=3000, grey=False):
    """Floor (z=0) and two walls at 90 degrees (x=-3, y=-3): three independent normals. The floor is deliberately not
    square (4 x 6)."""
    floor = np.column_stack([rng.uniform(-3, 3, n_per_surface),
                             rng.uniform(-2, 2, n_per_surface),
                             rng.normal(0, 0.003, n_per_surface)])
    wall_x = np.column_stack([np.full(n_per_surface, -3.0) + rng.normal(0, 0.003, n_per_surface),
                              rng.uniform(-3, 3, n_per_surface),
                              rng.uniform(0, 2.5, n_per_surface)])
    wall_y = np.column_stack([rng.uniform(-3, 3, n_per_surface),
                              np.full(n_per_surface, -3.0) + rng.normal(0, 0.003, n_per_surface),
                              rng.uniform(0, 2.5, n_per_surface)])
    pts = np.vstack([floor, wall_x, wall_y])
    if grey:
        rgb = np.full((len(pts), 3), 0.4)
    else:
        rgb = np.vstack([
            np.tile([0.75, 0.70, 0.60], (n_per_surface, 1)),   # floor: warm tan
            np.tile([0.20, 0.25, 0.70], (n_per_surface, 1)),   # wall_x: blue
            np.tile([0.70, 0.20, 0.20], (n_per_surface, 1)),   # wall_y: red
        ])
    rgb = rgb + rng.normal(0, 0.02, rgb.shape)
    rgb = np.clip(rgb, 0, 1)
    return pts, rgb


def _pcd(pts, rgb):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    p.colors = o3d.utility.Vector3dVector(rgb)
    return p


def test_segment_surfaces_finds_floor_and_both_walls_any_orientation():
    rng = np.random.default_rng(1)
    pts, rgb = _room_points(rng)
    surfaces, _residual = segment_surfaces(_pcd(pts, rgb), min_support=200)
    assert len(surfaces) >= 3, f"expected floor + 2 walls, got {len(surfaces)}"

    normals = sorted((tuple(np.round(np.abs(s.normal), 2)) for s in surfaces))
    # One surface normal to Z (the floor), two normal to X/Y (the walls) -
    # found with NO horizontal/vertical bias, unlike walls.extract_walls.
    have_z = any(abs(abs(s.normal[2]) - 1.0) < 0.05 for s in surfaces)
    have_x = any(abs(abs(s.normal[0]) - 1.0) < 0.05 for s in surfaces)
    have_y = any(abs(abs(s.normal[1]) - 1.0) < 0.05 for s in surfaces)
    assert have_z and have_x and have_y, f"normals found: {normals}"


def test_segment_surfaces_keeps_a_raked_surface_no_level_assumption():
    """A raked floor is a distinctive feature: a tilted plane is found and described at its real tilt, not discarded
    or snapped."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-3, 3, 4000)
    y = rng.uniform(-3, 3, 4000)
    z = 0.08 * y + rng.normal(0, 0.003, 4000)  # a real rake, a few degrees
    pts = np.column_stack([x, y, z])
    rgb = np.tile([0.5, 0.5, 0.5], (len(pts), 1))
    surfaces, _residual = segment_surfaces(_pcd(pts, rgb), min_support=200)
    assert len(surfaces) >= 1
    s = surfaces[0]
    tilt_deg = np.degrees(np.arccos(np.clip(abs(float(s.normal[2])), -1, 1)))
    assert 2.0 < tilt_deg < 8.0, f"raked surface's own tilt measured at {tilt_deg:.2f}deg"


def test_grey_surface_has_no_hue_saturation_floor():
    rng = np.random.default_rng(3)
    pts, rgb = _room_points(rng, grey=True)
    surfaces, _residual = segment_surfaces(_pcd(pts, rgb), min_support=200)
    assert surfaces, "expected at least one surface"
    for s in surfaces:
        assert np.isnan(s.hue), f"grey surface reported a real hue: {s.hue}"


def test_neighbor_graph_flags_touching_surfaces_not_distant_ones():
    rng = np.random.default_rng(4)
    pts, rgb = _room_points(rng)
    surfaces, _residual = segment_surfaces(_pcd(pts, rgb), min_support=200)
    build_neighbor_graph(surfaces, k=len(surfaces) - 1)
    # Floor and each wall meet at a real edge (z=0, x=-3 or y=-3) - both
    # should be adjacent. The two walls only meet at one far corner line,
    # so they need not be (not asserted either way here, kept honest).
    by_axis = {}
    for s in surfaces:
        axis = int(np.argmax(np.abs(s.normal)))
        by_axis.setdefault(axis, s)
    if 2 in by_axis and 0 in by_axis:
        floor, wall_x = by_axis[2], by_axis[0]
        nb = {n.other: n for n in floor.neighbors}
        assert wall_x.index in nb and nb[wall_x.index].adjacent, (
            "floor and a wall meeting at a real edge were not flagged adjacent"
        )


def test_match_surfaces_and_solve_pose_recover_a_known_transform():
    rng = np.random.default_rng(5)
    pts_a, rgb_a = _room_points(rng)

    true_yaw_deg, true_t = 12.0, np.array([1.5, -0.8, 0.2])
    yaw = np.radians(true_yaw_deg)
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = true_t
    pts_b = (R @ pts_a.T).T + true_t
    rgb_b = rgb_a.copy()  # same physical surfaces, same colour up to noise

    src_surfaces, _res_a = segment_surfaces(_pcd(pts_a, rgb_a), min_support=200)
    tgt_surfaces, _res_b = segment_surfaces(_pcd(pts_b, rgb_b), min_support=200)
    build_neighbor_graph(src_surfaces)
    build_neighbor_graph(tgt_surfaces)

    # A seed close to, but not exactly, the true transform - matching how a
    # real hand placement is a strong hint, not an exact answer.
    seed = np.eye(4)
    seed_yaw = np.radians(true_yaw_deg + 3.0)
    cs, ss = np.cos(seed_yaw), np.sin(seed_yaw)
    seed[:3, :3] = np.array([[cs, -ss, 0.0], [ss, cs, 0.0], [0.0, 0.0, 1.0]])
    seed[:3, 3] = true_t + np.array([0.15, -0.1, 0.05])

    matches, unmatched_src, unmatched_tgt = match_surfaces(
        src_surfaces, tgt_surfaces, seed, position_bound_m=0.5)
    assert len(matches) >= 3, f"expected floor + 2 walls to match, got {len(matches)}"
    assert not unmatched_src and not unmatched_tgt, (
        f"unmatched: src={unmatched_src} tgt={unmatched_tgt}"
    )
    # Each surface should match itself across scans (same axis), not a
    # different one - the actual correctness check, not just "3 matches".
    src_by_i = {s.index: s for s in src_surfaces}
    tgt_by_j = {t.index: t for t in tgt_surfaces}
    for m in matches:
        s, t = src_by_i[m.src], tgt_by_j[m.tgt]
        assert np.argmax(np.abs(s.normal)) == np.argmax(np.abs(t.normal)), (
            f"matched surfaces on different axes: {s.normal} vs {t.normal}"
        )

    solved = solve_pose_from_surfaces(src_surfaces, tgt_surfaces, matches)
    assert solved is not None, "pose solve returned None on a well-determined synthetic case"

    solved_yaw = np.degrees(np.arctan2(solved[1, 0], solved[0, 0]))
    assert abs(solved_yaw - true_yaw_deg) < 0.5, (
        f"solved yaw {solved_yaw:.3f}deg, expected {true_yaw_deg}deg"
    )
    assert np.linalg.norm(solved[:3, 3] - true_t) < 0.05, (
        f"solved translation {solved[:3,3]}, expected {true_t}"
    )


def test_solve_pose_needs_at_least_min_matches():
    rng = np.random.default_rng(6)
    pts, rgb = _room_points(rng)
    surfaces, _residual = segment_surfaces(_pcd(pts, rgb), min_support=200)
    build_neighbor_graph(surfaces)
    assert solve_pose_from_surfaces(surfaces, surfaces, [], min_matches=3) is None


def test_residual_is_the_non_planar_landmark_set():
    """Whatever no plane explains (a small blob off the three planes) comes back in the residual, not absorbed into a
    surface."""
    rng = np.random.default_rng(7)
    pts, rgb = _room_points(rng)
    # Fewer points than min_support: no subset of this blob can ever reach
    # min_support as a spurious "surface" of its own, by construction - it
    # MUST end up in the residual, not absorbed into a real plane.
    n_blob = 150
    blob = np.column_stack([rng.normal(0.0, 0.05, n_blob), rng.normal(0.0, 0.05, n_blob),
                            rng.normal(1.2, 0.05, n_blob)])  # floats in mid-air, on no plane
    pts_with_blob = np.vstack([pts, blob])
    rgb_with_blob = np.vstack([rgb, np.tile([0.9, 0.9, 0.1], (n_blob, 1))])
    surfaces, residual = segment_surfaces(_pcd(pts_with_blob, rgb_with_blob), min_support=200)
    residual_pts = pts_with_blob[residual]
    near_blob = np.linalg.norm(residual_pts - np.array([0, 0, 1.2]), axis=1) < 0.3
    assert near_blob.sum() >= n_blob * 0.9, (
        f"expected the {n_blob}-point off-plane blob in the residual, "
        f"found {near_blob.sum()} of it in {len(residual)} residual pts"
    )


def test_feature_pair_score_high_for_the_same_room_seeded_close_to_true():
    """The per-scan pre-pass, scored between two scans of the same room a known transform apart: the case this filter
    exists to accept."""
    rng = np.random.default_rng(8)
    pts_a, rgb_a = _room_points(rng)
    true_yaw_deg, true_t = 5.0, np.array([0.3, -0.2, 0.0])
    yaw = np.radians(true_yaw_deg)
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    pts_b = (R @ pts_a.T).T + true_t
    rgb_b = rgb_a.copy()

    src = extract_scan_features(_pcd(pts_a, rgb_a), min_support=200)
    tgt = extract_scan_features(_pcd(pts_b, rgb_b), min_support=200)
    assert len(src.surfaces) >= 3 and len(tgt.surfaces) >= 3

    seed = np.eye(4)
    seed[:3, :3] = R
    seed[:3, 3] = true_t
    frac, n_matches, matches = feature_pair_score(src, tgt, seed, position_bound_m=2.0)
    assert n_matches >= 3, f"expected all 3 planes to match, got {n_matches}"
    assert frac > 0.7, f"expected a high feature-overlap score for the same room, got {frac:.3f}"


def test_feature_pair_score_low_for_unrelated_scans():
    """A scan and a physically unrelated point cloud (nothing in common)
    must score near zero, not accidentally match on coincidence."""
    rng = np.random.default_rng(9)
    pts_a, rgb_a = _room_points(rng)
    # An unrelated cluster of small, randomly oriented blobs - no coherent
    # large planes, so segment_surfaces should find little to nothing.
    pts_b = rng.uniform(-1, 1, (2000, 3)) + np.array([50.0, 50.0, 50.0])
    rgb_b = rng.uniform(0, 1, (2000, 3))

    src = extract_scan_features(_pcd(pts_a, rgb_a), min_support=200)
    tgt = extract_scan_features(_pcd(pts_b, rgb_b), min_support=200)

    frac, n_matches, _matches = feature_pair_score(src, tgt, np.eye(4), position_bound_m=2.0)
    assert frac < 0.1, f"expected near-zero overlap for unrelated scans, got {frac:.3f}"


def test_feature_pair_score_zero_with_no_surfaces_either_side():
    """A scan with zero detected surfaces scores 0.0 rather than raising or dividing by zero."""
    empty = ScanFeatures(surfaces=[], residual_idx=np.array([], dtype=np.int64))
    rng = np.random.default_rng(10)
    pts, rgb = _room_points(rng)
    real = extract_scan_features(_pcd(pts, rgb), min_support=200)
    frac, n_matches, matches = feature_pair_score(empty, real, np.eye(4), position_bound_m=2.0)
    assert frac == 0.0 and n_matches == 0 and matches == []


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
