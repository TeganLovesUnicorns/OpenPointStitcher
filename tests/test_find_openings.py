"""Tests for find_openings.py (doors and windows seen face-on), the openings vote in score.py, and that it can raise a
candidate but never reject one. Synthetic walls only."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import find_openings  # noqa: E402
from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.find_openings import Opening, find_wall_openings, scan_openings  # noqa: E402
from open_point_stitch.find_surfaces import ScanFeatures  # noqa: E402
from open_point_stitch.find_walls import Wall  # noqa: E402
from open_point_stitch.score import Verdict, apply_detector_vote, opening_agreement  # noqa: E402

CFG = PipelineConfig()


def _wall(length=6.0):
    return Wall(normal=np.array([1.0, 0.0]), offset=0.0, support=50000, length=length, height=3.0,
                centroid=np.array([0.0, length / 2, 1.5]), index=0)


def _wall_points(length=6.0, holes=(), floor=True, seed=0):
    """A wall in the plane x = 0, y from 0 to `length`, z from 0 to 3, with rectangular holes (y0, y1, z0, z1)."""
    rng = np.random.default_rng(seed)
    y, z = np.meshgrid(np.arange(0.0, length, 0.02), np.arange(0.0, 3.0, 0.02))
    y, z = y.ravel(), z.ravel()
    keep = np.ones(y.size, dtype=bool)
    for y0, y1, z0, z1 in holes:
        keep &= ~((y >= y0) & (y < y1) & (z >= z0) & (z < z1))
    pts = np.column_stack([rng.normal(0.0, 0.004, keep.sum()), y[keep], z[keep]])
    if floor:
        fy, fx = np.meshgrid(np.arange(0.0, length, 0.03), np.arange(-0.5, 0.5, 0.03))
        pts = np.vstack([pts, np.column_stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)])])
    return pts


DOOR = (2.5, 3.4, 0.0, 2.1)
WINDOW = (4.4, 5.4, 0.9, 2.1)


def test_a_door_and_a_window_are_found_with_their_sizes_and_places():
    found = find_wall_openings(_wall_points(holes=[DOOR, WINDOW]), _wall(), 0.0)
    kinds = sorted(o.kind for o in found)
    assert kinds == ["door", "window"], [o.describe() for o in found]
    door = next(o for o in found if o.kind == "door")
    window = next(o for o in found if o.kind == "window")
    assert abs(door.width - 0.9) <= 0.11 and abs(door.height - 2.1) <= 0.11 and door.sill == 0.0, door.describe()
    assert abs(door.along - 2.95) <= 0.12, door.describe()
    assert abs(window.width - 1.0) <= 0.11 and abs(window.sill - 0.9) <= 0.11 and abs(window.height - 1.2) <= 0.16, window.describe()
    assert abs(window.along - 4.9) <= 0.12, window.describe()
    assert door.clear_fraction > 0.9 and door.bounded > 0.5


def test_a_solid_wall_has_no_openings():
    assert find_wall_openings(_wall_points(), _wall(), 0.0) == []


def test_where_the_wall_simply_stops_is_not_an_opening():
    pts = _wall_points(holes=[(0.0, 0.9, 0.0, 3.0)])           # the wall starts at y = 0.9
    assert find_wall_openings(pts, _wall(), 0.0) == []


def test_a_gap_too_narrow_to_be_a_door_is_ignored():
    assert find_wall_openings(_wall_points(holes=[(2.5, 2.8, 0.0, 2.1)]), _wall(), 0.0) == []


def test_a_hole_with_no_wall_around_it_is_not_a_window():
    pts = _wall_points(holes=[(2.0, 4.0, 0.9, 3.0)])            # open to the ceiling above, so nothing bounds it
    assert [o for o in find_wall_openings(pts, _wall(), 0.0) if o.kind == "window"] == []


def test_openings_are_also_placed_in_the_scans_raw_frame():
    tilt = np.radians(3.0)
    rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    levelled = _wall_points(holes=[DOOR])
    raw = levelled @ rot                                        # levelled = raw @ rot.T
    found = scan_openings(raw, rot, 0.0, [_wall()])
    assert len(found) == 1
    o = found[0]
    expected = np.array([0.0, o.centre[1], o.centre[2]]) @ rot
    assert np.allclose(o.position, expected, atol=1e-6), (o.position, expected)
    assert abs(abs(float(o.direction @ (np.array([0.0, 1.0, 0.0]) @ rot))) - 1.0) < 1e-6
    assert abs(abs(float(o.wall_normal @ (np.array([1.0, 0.0, 0.0]) @ rot))) - 1.0) < 1e-6
    assert abs(o.cohesion - o.clear_fraction * o.bounded) < 1e-9


def test_another_kind_of_feature_can_be_registered_without_touching_anything_else():
    def alcoves(points, wall, floor_z):
        return [Opening(kind="alcove", wall_index=wall.index, along=1.0, width=1.0, height=2.0, sill=0.0,
                        centre=np.array([0.0, 1.0, 1.0]), normal=wall.normal, clear_fraction=1.0, bounded=1.0)]
    find_openings.register_detector("alcoves", alcoves)
    try:
        kinds = {o.kind for o in find_openings.find_openings(_wall_points(holes=[DOOR]), [_wall()], 0.0)}
    finally:
        del find_openings.DETECTORS["alcoves"]
    assert kinds == {"door", "alcove"}, kinds


def _opening(kind="door", position=(0.0, 0.0, 1.0), along=(0.0, 1.0, 0.0), across=(1.0, 0.0, 0.0), width=0.9, height=2.1, cohesion=0.9):
    return Opening(kind=kind, wall_index=0, along=0.0, width=width, height=height, sill=0.0, centre=np.array(position),
                   normal=np.array(across[:2]), clear_fraction=1.0, bounded=cohesion, position=np.array(position, dtype=float),
                   direction=np.array(along, dtype=float), wall_normal=np.array(across, dtype=float), cohesion=cohesion)


def _features(*openings):
    return ScanFeatures(surfaces=[], residual_idx=np.array([], dtype=np.int64), openings=list(openings))


def _yaw(deg, t=(0.0, 0.0, 0.0)):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:3, 3] = t
    return m


def test_the_same_door_seen_from_two_scans_agrees_at_the_true_transform():
    truth = _yaw(30.0, (4.0, -2.0, 0.0))
    a = _opening(position=(1.0, 2.0, 1.0))
    p = truth[:3, :3] @ a.position + truth[:3, 3]
    b = _opening(position=tuple(p), along=tuple(truth[:3, :3] @ a.direction), across=tuple(truth[:3, :3] @ a.wall_normal))
    agree, n = opening_agreement(truth, _features(a), _features(b), CFG)
    assert n == 1 and agree > 0.99, (agree, n)


def test_a_wall_thickness_across_the_wall_is_not_held_against_a_candidate():
    a = _opening(position=(0.0, 2.0, 1.0))
    b = _opening(position=(0.25, 2.0, 1.0))                     # 25 cm across the wall: the other side of it
    agree, n = opening_agreement(np.eye(4), _features(a), _features(b), CFG)
    assert n == 1 and agree > 0.99, (agree, n)


def test_being_off_along_the_wall_lowers_the_agreement_and_far_off_is_not_counted():
    a = _opening(position=(0.0, 2.0, 1.0))
    near, n_near = opening_agreement(np.eye(4), _features(a), _features(_opening(position=(0.0, 2.2, 1.0))), CFG)
    far, n_far = opening_agreement(np.eye(4), _features(a), _features(_opening(position=(0.0, 3.5, 1.0))), CFG)
    assert n_near == 1 and near < 0.5, near
    assert n_far == 0 and far == 0.0


def test_a_half_turn_puts_the_door_somewhere_else():
    a = _opening(position=(0.0, 2.0, 1.0))
    agree, n = opening_agreement(_yaw(180.0), _features(a), _features(_opening(position=(0.0, 2.0, 1.0))), CFG)
    assert n == 0, "the flipped candidate must not find its door where the true one does"


def test_a_window_does_not_match_a_door_and_a_different_size_does_not_match():
    a = _opening(position=(0.0, 2.0, 1.0))
    assert opening_agreement(np.eye(4), _features(a), _features(_opening(kind="window", position=(0.0, 2.0, 1.0))), CFG)[1] == 0
    assert opening_agreement(np.eye(4), _features(a), _features(_opening(position=(0.0, 2.0, 1.0), width=1.6)), CFG)[1] == 0


def test_a_weak_opening_is_ignored():
    a = _opening(position=(0.0, 2.0, 1.0), cohesion=0.2)
    assert opening_agreement(np.eye(4), _features(a), _features(_opening(position=(0.0, 2.0, 1.0))), CFG)[1] == 0


def test_no_openings_at_all_says_nothing():
    assert opening_agreement(np.eye(4), _features(), _features(), CFG) == (0.0, 0)


def _verdict():
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    return v


def test_openings_alone_raise_the_ranking_and_can_never_reject():
    good, bad = _verdict(), _verdict()
    apply_detector_vote(good, 0.0, 0.0, 0, CFG, opening_agreement=1.0, n_openings=3)
    apply_detector_vote(bad, 0.0, 0.0, 0, CFG, opening_agreement=0.0, n_openings=3)
    assert good.rejected_by == "" and bad.rejected_by == ""
    assert good.vote_score > good.surface_fit and bad.vote_score < good.vote_score
    assert good.n_openings_considered == 3 and good.opening_agreement == 1.0


def test_openings_cannot_rescue_a_candidate_the_building_rejects_or_sink_one_it_accepts():
    rejected = _verdict()
    apply_detector_vote(rejected, 0.05, 0.0, 3, CFG, corner_agreement=0.05, n_corners=2, opening_agreement=1.0, n_openings=4)
    assert rejected.rejected_by != "", "a strong door match must not overrule planes and corners that disagree"
    accepted = _verdict()
    apply_detector_vote(accepted, 0.9, 0.0, 3, CFG, corner_agreement=0.9, n_corners=2, opening_agreement=0.0, n_openings=4)
    assert accepted.rejected_by == "", "doors that disagree must not reject what planes and corners accept"


def test_with_no_openings_the_vote_is_exactly_what_it_was():
    plain, with_zero = _verdict(), _verdict()
    apply_detector_vote(plain, 0.6, 0.2, 3, CFG, corner_agreement=0.7, n_corners=2, edge_agreement=0.5, n_edges=3)
    apply_detector_vote(with_zero, 0.6, 0.2, 3, CFG, corner_agreement=0.7, n_corners=2, edge_agreement=0.5, n_edges=3,
                        opening_agreement=0.0, n_openings=0)
    assert plain.vote_score == with_zero.vote_score and plain.detector_score == with_zero.detector_score


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
