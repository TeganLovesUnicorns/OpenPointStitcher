"""Tests for find_corners.py: corners derived from find_surfaces.py's planes and neighbour graph. Synthetic
geometry only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_surfaces import (  # noqa: E402
    Surface, Neighbor, segment_surfaces, build_neighbor_graph,
)
from open_point_stitch.find_corners import find_corners, find_edges  # noqa: E402


def _room_points(rng, n_per_surface=3000):
    """Floor (z=0) and two walls at 90 degrees meeting at (-3, -3, 0). The floor is not square (6.5 x 6)."""
    floor = np.column_stack([rng.uniform(-3, 3.5, n_per_surface),
                             rng.uniform(-3, 3, n_per_surface),
                             rng.normal(0, 0.003, n_per_surface)])
    wall_x = np.column_stack([np.full(n_per_surface, -3.0) + rng.normal(0, 0.003, n_per_surface),
                              rng.uniform(-3, 3, n_per_surface),
                              rng.uniform(0, 2.5, n_per_surface)])
    wall_y = np.column_stack([rng.uniform(-3, 3, n_per_surface),
                              np.full(n_per_surface, -3.0) + rng.normal(0, 0.003, n_per_surface),
                              rng.uniform(0, 2.5, n_per_surface)])
    pts = np.vstack([floor, wall_x, wall_y])
    rgb = np.vstack([
        np.tile([0.75, 0.70, 0.60], (n_per_surface, 1)),
        np.tile([0.20, 0.25, 0.70], (n_per_surface, 1)),
        np.tile([0.70, 0.20, 0.20], (n_per_surface, 1)),
    ])
    rgb = np.clip(rgb + rng.normal(0, 0.02, rgb.shape), 0, 1)
    return pts, rgb


def _pcd(pts, rgb):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    p.colors = o3d.utility.Vector3dVector(rgb)
    return p


def _flat_surface(index, normal, offset, centroid, u_axis, v_axis,
                   extent_u=2.0, extent_v=2.0, hue=float("nan")):
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)
    return Surface(
        normal=n, offset=float(offset), centroid=np.array(centroid, dtype=float),
        support=1000, area=extent_u * extent_v, extent_u=extent_u, extent_v=extent_v,
        corners=4, hue=hue, saturation=0.0 if np.isnan(hue) else 0.5, value=0.5,
        u_axis=np.array(u_axis, dtype=float), v_axis=np.array(v_axis, dtype=float),
        index=index,
    )


def test_find_corners_recovers_the_known_room_corner():
    rng = np.random.default_rng(11)
    pts, rgb = _room_points(rng)
    surfaces, _residual = segment_surfaces(_pcd(pts, rgb), min_support=200)
    build_neighbor_graph(surfaces, k=len(surfaces) - 1)
    assert len(surfaces) >= 3, f"expected floor + 2 walls, got {len(surfaces)}"

    corners = find_corners(surfaces)
    assert corners, "expected the floor/wall_x/wall_y intersection to be found"
    best = corners[0]
    assert np.linalg.norm(best.position - np.array([-3.0, -3.0, 0.0])) < 0.05, (
        f"solved corner at {best.position}, expected near (-3, -3, 0)"
    )
    for a in best.angles_deg:
        assert 80.0 < a < 100.0, f"expected three ~90deg angles, got {best.angles_deg}"
    assert best.cohesion > 0.5, f"a real, well-supported corner scored low cohesion: {best.cohesion}"


def test_find_corners_rejects_near_parallel_planes():
    """Three surfaces whose normals are all close to the same direction -
    the intersection is numerically unstable and must be dropped, not
    reported with false confidence."""
    a = _flat_surface(0, [1.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], [0, 1, 0], [0, 0, 1])
    b = _flat_surface(1, [0.98, 0.15, 0.0], 0.1, [0.1, 0.0, 0.0], [0, 1, 0], [0, 0, 1])
    c = _flat_surface(2, [0.97, -0.17, 0.05], -0.1, [-0.1, 0.0, 0.0], [0, 1, 0], [0, 0, 1])
    for s, others in ((a, (1, 2)), (b, (0, 2)), (c, (0, 1))):
        s.neighbors = [Neighbor(other=o, angle_deg=5.0, distance_m=0.1,
                                size_ratio=1.0, adjacent=True) for o in others]
    assert find_corners([a, b, c]) == []


def test_find_corners_rejects_an_intersection_far_from_any_real_footprint():
    """Three perpendicular planes whose measured footprints are centred far from where they cross (scattered
    points): rejected on cohesion, not accepted on angles alone."""
    a = _flat_surface(0, [1, 0, 0], 0.0, [0.0, 8.0, 8.0], [0, 1, 0], [0, 0, 1],
                       extent_u=1.0, extent_v=1.0)
    b = _flat_surface(1, [0, 1, 0], 0.0, [8.0, 0.0, 8.0], [1, 0, 0], [0, 0, 1],
                       extent_u=1.0, extent_v=1.0)
    c = _flat_surface(2, [0, 0, 1], 0.0, [8.0, 8.0, 0.0], [1, 0, 0], [0, 1, 0],
                       extent_u=1.0, extent_v=1.0)
    for s, others in ((a, (1, 2)), (b, (0, 2)), (c, (0, 1))):
        s.neighbors = [Neighbor(other=o, angle_deg=90.0, distance_m=0.1,
                                size_ratio=1.0, adjacent=True) for o in others]
    # The three planes (x=0, y=0, z=0) meet at the origin - none of the three
    # footprints (each centred 8m away, 1m across) reaches anywhere near it.
    assert find_corners([a, b, c], cohesion_tolerance_m=0.5) == []


def test_three_planes_that_do_not_all_meet_are_not_a_corner():
    """Two surfaces reach their shared intersection and the third does not: caught by cohesion."""
    a = _flat_surface(0, [1, 0, 0], -3.0, [-3.0, 0.0, 1.0], [0, 1, 0], [0, 0, 1])
    b = _flat_surface(1, [0, 1, 0], -3.0, [0.0, -3.0, 1.0], [1, 0, 0], [0, 0, 1])
    c = _flat_surface(2, [0, 0, 1], 0.0, [-3.0, -3.0, 0.0], [1, 0, 0], [0, 1, 0])
    assert find_corners([a, b, c]) == []


def _corner_room_surfaces():
    """Three perpendicular surfaces whose footprints all reach their intersection at (-3, -3, 0): a real corner."""
    a = _flat_surface(0, [1, 0, 0], -3.0, [-3.0, -1.5, 1.0], [0, 1, 0], [0, 0, 1],
                      extent_u=4.0, extent_v=2.5)
    b = _flat_surface(1, [0, 1, 0], -3.0, [-1.5, -3.0, 1.0], [1, 0, 0], [0, 0, 1],
                      extent_u=4.0, extent_v=2.5)
    c = _flat_surface(2, [0, 0, 1], 0.0, [-1.5, -1.5, 0.0], [1, 0, 0], [0, 1, 0],
                      extent_u=4.0, extent_v=4.0)
    return a, b, c


def test_find_corners_does_not_need_the_neighbour_graph():
    """No k-nearest neighbour prefilter: a wall's nearest surfaces by centroid are usually fragments of the same
    wall, so the prefilter threw away real corners."""
    a, b, c = _corner_room_surfaces()
    assert a.neighbors == [] and b.neighbors == [] and c.neighbors == []
    corners = find_corners([a, b, c])
    assert len(corners) == 1, f"expected the one real corner, got {len(corners)}"
    assert np.linalg.norm(corners[0].position - np.array([-3.0, -3.0, 0.0])) < 1e-6
    assert corners[0].cohesion > 0.9


def test_two_coplanar_surfaces_are_one_wall_not_a_corner():
    """A wall detected as two pieces sits at 180 degrees to itself: with the floor that is three planes but no corner,
    and the same just short of 180."""
    floor = _flat_surface(0, [0, 0, 1], 0.0, [0.0, -1.5, 0.0], [1, 0, 0], [0, 1, 0],
                          extent_u=8.0, extent_v=6.0)
    for offset_deg in (0.0, 0.01, 1.0):
        tip = np.radians(offset_deg)
        left = _flat_surface(1, [0, 1, 0], -3.0, [-2.0, -3.0, 1.5], [1, 0, 0], [0, 0, 1],
                             extent_u=4.0, extent_v=3.0)
        right = _flat_surface(2, [np.sin(tip), np.cos(tip), 0.0], -3.0,
                              [2.0, -3.0, 1.5], [1, 0, 0], [0, 0, 1],
                              extent_u=4.0, extent_v=3.0)
        right.offset = float(right.normal @ np.array([2.0, -3.0, 1.5]))
        assert find_corners([floor, left, right]) == [], (
            f"two near-coplanar wall pieces {offset_deg} deg apart were reported as a corner"
        )


def test_find_corners_handles_a_vaulted_ceiling_not_just_square_rooms():
    """Two roof slopes meeting at a ridge, closed off by an end wall. The ridge is a ~114 degree fold and must be
    described as one, not as the 66 degree fold an unsigned normal implies."""
    pitch = np.radians(33.0)
    # Both slopes' normals point down into the room, tipped toward the ridge.
    left = _flat_surface(0, [np.sin(pitch), 0.0, -np.cos(pitch)], 0.0,
                         [-3.0, 0.0, 5.0 - 3.0 * np.tan(pitch)], [0, 1, 0],
                         [np.cos(pitch), 0, np.sin(pitch)], extent_u=12.0, extent_v=7.0)
    right = _flat_surface(1, [-np.sin(pitch), 0.0, -np.cos(pitch)], 0.0,
                          [3.0, 0.0, 5.0 - 3.0 * np.tan(pitch)], [0, 1, 0],
                          [np.cos(pitch), 0, -np.sin(pitch)], extent_u=12.0, extent_v=7.0)
    # Offsets put both planes through the ridge line at (0, y, 5).
    for s in (left, right):
        s.offset = float(s.normal @ np.array([0.0, 0.0, 5.0]))
        s.centroid = np.array([np.sign(-s.normal[0]) * 3.0, 0.0,
                               5.0 - 3.0 * np.tan(pitch)])
    end_wall = _flat_surface(2, [0, 1, 0], -6.0, [0.0, -6.0, 3.5], [1, 0, 0], [0, 0, 1],
                             extent_u=8.0, extent_v=7.0)

    corners = find_corners([left, right, end_wall])
    assert corners, "the ridge meeting the end wall is a real corner and was not found"
    ridge = corners[0]
    assert np.linalg.norm(ridge.position - np.array([0.0, -6.0, 5.0])) < 1e-6, (
        f"solved the ridge/end-wall corner at {ridge.position}, expected (0, -6, 5)"
    )
    folded = sorted(ridge.angles_deg)
    interior = sorted(ridge.interior_angles_deg)
    assert abs(folded[0] - 66.0) < 1.0, f"expected a 66 deg folded ridge, got {folded}"
    assert abs(interior[-1] - 114.0) < 1.0, (
        f"the ridge is a 114 deg fold and must be recorded as one, got {interior}"
    )


def test_find_corners_reports_one_physical_corner_once():
    """A wall broken into two detected fragments makes two triples that
    solve to the same point. One corner, not two, or a matching count
    turns into a count of how fragmented the segmentation was."""
    a, b, c = _corner_room_surfaces()
    fragment = _flat_surface(3, [0, 0, 1], 0.0, [-2.5, -2.5, 0.0], [1, 0, 0], [0, 1, 0],
                             extent_u=2.0, extent_v=2.0)
    corners = find_corners([a, b, c, fragment])
    assert len(corners) == 1, f"expected one merged corner, got {len(corners)}"
    assert len(find_corners([a, b, c, fragment], merge_within_m=0.0)) == 2, (
        "sanity: both triples really are found, merging is what differs"
    )


def test_find_edges_recovers_a_wall_meeting_the_floor():
    """Edges: two surfaces meeting along a line. Floor at z=0 and a wall at y=-3 meet along y=-3, z=0, for as long
    as the shorter footprint."""
    floor = _flat_surface(0, [0, 0, 1], 0.0, [0.0, -1.0, 0.0], [1, 0, 0], [0, 1, 0],
                          extent_u=8.0, extent_v=4.0)
    wall = _flat_surface(1, [0, 1, 0], -3.0, [0.0, -3.0, 1.25], [1, 0, 0], [0, 0, 1],
                         extent_u=6.0, extent_v=2.5)
    edges = find_edges([floor, wall])
    assert len(edges) == 1, f"expected the one wall/floor junction, got {len(edges)}"
    e = edges[0]
    assert abs(abs(float(e.direction @ np.array([1.0, 0.0, 0.0]))) - 1.0) < 1e-9, (
        f"the junction runs along X, got direction {e.direction}"
    )
    assert np.linalg.norm(e.midpoint - np.array([0.0, -3.0, 0.0])) < 1e-9
    assert abs(e.length_m - 6.0) < 1e-9, (
        f"the shared stretch is the wall's own 6 m, got {e.length_m}"
    )
    assert abs(e.interior_angle_deg - 90.0) < 1e-6
    assert e.cohesion > 0.9


def test_find_edges_ignores_planes_that_would_only_cross_far_away():
    """Two surfaces on opposite sides of the room are not a junction just
    because their planes meet somewhere out past the building."""
    floor = _flat_surface(0, [0, 0, 1], 0.0, [0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0],
                          extent_u=4.0, extent_v=4.0)
    high_wall = _flat_surface(1, [0, 1, 0], -20.0, [0.0, -20.0, 6.0], [1, 0, 0], [0, 0, 1],
                              extent_u=4.0, extent_v=2.0)
    assert find_edges([floor, high_wall]) == []


def test_find_edges_reports_one_junction_per_physical_line():
    """A wall detected as two fragments meets the floor along the same
    line twice. One edge, not two."""
    floor = _flat_surface(0, [0, 0, 1], 0.0, [0.0, -1.0, 0.0], [1, 0, 0], [0, 1, 0],
                          extent_u=8.0, extent_v=4.0)
    left = _flat_surface(1, [0, 1, 0], -3.0, [-1.5, -3.0, 1.25], [1, 0, 0], [0, 0, 1],
                         extent_u=3.0, extent_v=2.5)
    right = _flat_surface(2, [0, 1, 0], -3.0, [1.5, -3.0, 1.25], [1, 0, 0], [0, 0, 1],
                          extent_u=3.0, extent_v=2.5)
    assert len(find_edges([floor, left, right])) == 1


def test_find_edges_rejects_a_coplanar_pair():
    """Same rule as corners: 180 degrees apart is one wall, not a junction."""
    a = _flat_surface(0, [0, 1, 0], -3.0, [-2.0, -3.0, 1.5], [1, 0, 0], [0, 0, 1])
    b = _flat_surface(1, [0, 1, 0], -3.0, [2.0, -3.0, 1.5], [1, 0, 0], [0, 0, 1])
    assert find_edges([a, b]) == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
