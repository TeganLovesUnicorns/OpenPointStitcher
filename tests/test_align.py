"""Tests for align_from_anchors: pose composition maths and the link-seeded bootstrap."""

from __future__ import annotations

import logging
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.align_anchors import align_from_anchors, ScanPose, pose_shift_m  # noqa: E402
from open_point_stitch.placement import Anchor, AnchorMember  # noqa: E402


def _room_points(rng, n=800):
    """A crude room: floor, two walls, so ICP has real geometry to lock onto."""
    floor = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), np.zeros(n)])
    wall1 = np.column_stack([np.full(n // 2, -3.0), rng.uniform(-3, 3, n // 2),
                              rng.uniform(0, 2.5, n // 2)])
    wall2 = np.column_stack([rng.uniform(-3, 3, n // 2), np.full(n // 2, -3.0),
                              rng.uniform(0, 2.5, n // 2)])
    return np.vstack([floor, wall1, wall2])


def _payload_from_points(pts):
    return {
        "xy": pts[:, :2].tolist(), "h": pts[:, 2].tolist(),
        "level_rotation": np.eye(3).tolist(), "floor_z": 0.0,
    }


def _yaw_matrix2(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _world_points(payload, pose):
    xy = np.asarray(payload["xy"])
    h = np.asarray(payload["h"])
    r = _yaw_matrix2(pose.yaw_deg)
    wxy = xy @ r.T + [pose.x_m, pose.y_m]
    return np.column_stack([wxy, h + pose.z_m])


def _anchor(a_name, a_point_raw, b_name, b_point_raw, anchor_id="anchor-1"):
    """A minimal 2-member region anchor: one raw-frame point per scan, zero extent."""
    def member(name, pt):
        p = tuple(float(v) for v in pt)
        return AnchorMember(scan=name, points=[p], centroid=p,
                            extent=(0.0, 0.0, 0.0), n_points=1)
    return Anchor(id=anchor_id, members=[member(a_name, a_point_raw), member(b_name, b_point_raw)])


def _true_and_wrong_pair(true_rel_yaw=10.0, true_rel_x=5.0, true_rel_y=-3.0,
                         error_yaw=4.0, error_m=0.35, seed=3):
    """Build a same-room pair with a real relative pose to recover (B's frame is independent of A's). Do not derive
    local_b from the wrong pose."""
    rng = np.random.default_rng(seed)
    local = _room_points(rng)
    pose_a = ScanPose(yaw_deg=25.0, x_m=40.0, y_m=-18.0, z_m=0.3)
    true_world = _world_points(_payload_from_points(local), pose_a)

    pose_b_true = ScanPose(yaw_deg=pose_a.yaw_deg + true_rel_yaw,
                           x_m=pose_a.x_m + true_rel_x, y_m=pose_a.y_m + true_rel_y,
                           z_m=pose_a.z_m)
    r_true = _yaw_matrix2(pose_b_true.yaw_deg)
    local_b_xy = (true_world[:, :2] - [pose_b_true.x_m, pose_b_true.y_m]) @ r_true
    local_b = np.column_stack([local_b_xy, true_world[:, 2] - pose_b_true.z_m])

    pose_b_wrong = ScanPose(yaw_deg=pose_b_true.yaw_deg + error_yaw,
                            x_m=pose_b_true.x_m + error_m, y_m=pose_b_true.y_m,
                            z_m=pose_b_true.z_m)

    payloads = {"A": _payload_from_points(local), "B": _payload_from_points(local_b)}
    poses = {"A": pose_a, "B": pose_b_wrong}
    anchors = [_anchor("A", local[0], "B", local_b[0])]
    return payloads, poses, anchors, true_world, error_m


def test_align_recovers_a_known_offset_far_from_origin():
    payloads, poses, anchors, true_world, error_m = _true_and_wrong_pair()

    # Confirm the test actually starts from a real mismatch, not the
    # degenerate zero-correction-needed case the old version of this test
    # silently had.
    starting_err = np.linalg.norm(
        _world_points(payloads["B"], poses["B"]) - true_world, axis=1)
    assert starting_err.mean() > 0.1, "test setup starts already-aligned; proves nothing"

    result = align_from_anchors(
        payloads, poses, anchors, locked={}, subset=None,
        icp_distance_m=0.3, max_shift_m=2.0,
    )

    assert result.rows[0].applied, result.rows[0].reason
    assert result.rows[0].shift_max_m > 0.5 * error_m, (
        "a real correction should have been reported, not a trivial one")
    # B's corrected pose, applied to B's own local points, should now land
    # very close to A's frame (the room is the same room).
    b_world_after = _world_points(payloads["B"], result.poses["B"])
    err = np.linalg.norm(b_world_after - true_world, axis=1)
    assert err.mean() < 0.05, f"mean error {err.mean():.3f} m, expected well under 5 cm"


def test_locked_scan_is_never_moved():
    rng = np.random.default_rng(1)
    local = _room_points(rng)
    payloads = {"A": _payload_from_points(local), "B": _payload_from_points(local)}
    poses = {"A": ScanPose(), "B": ScanPose(x_m=5.0)}
    anchors = [_anchor("A", local[0], "B", local[0])]

    result = align_from_anchors(payloads, poses, anchors, locked={"B": True}, subset=None)
    by_scan = {r.scan: r for r in result.rows}
    assert not by_scan["B"].applied and "locked" in by_scan["B"].reason
    assert result.poses["B"].x_m == 5.0
    # A is free, and the anchor says the two are the same place, so A comes to B.
    assert by_scan["A"].applied and abs(result.poses["A"].x_m - 5.0) < 0.05, result.poses["A"]


def test_subset_restricts_which_scan_may_move():
    rng = np.random.default_rng(2)
    local = _room_points(rng)
    payloads = {"A": _payload_from_points(local), "B": _payload_from_points(local)}
    poses = {"A": ScanPose(), "B": ScanPose(x_m=5.0, yaw_deg=3.0)}
    anchors = [_anchor("A", local[0], "B", local[0])]

    result = align_from_anchors(payloads, poses, anchors, locked={}, subset={"A"})
    # B is not in the subset, so it gets no row and stays; A, which is, moves to it.
    assert [r.scan for r in result.rows] == ["A"] and result.rows[0].applied
    assert result.poses["B"].x_m == 5.0 and abs(result.poses["A"].x_m - 5.0) < 0.1 and abs(result.poses["A"].yaw_deg - 3.0) < 0.5


def test_unknown_scan_in_anchor_is_skipped_not_fatal():
    payloads = {"A": _payload_from_points(_room_points(np.random.default_rng(0)))}
    poses = {"A": ScanPose()}
    anchors = [_anchor("A", (0, 0, 0), "Nonexistent", (0, 0, 0))]
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None)
    assert result.rows == []
    assert len(result.skipped) == 1
    assert "Nonexistent" in result.skipped[0]


def test_anchor_with_a_single_member_is_skipped_not_fatal():
    """A link needs two or more members. One member alone has nothing to align against and must not crash the run."""
    payloads = {"A": _payload_from_points(_room_points(np.random.default_rng(0)))}
    poses = {"A": ScanPose()}
    anchors = [Anchor(id="lonely", members=[
        AnchorMember(scan="A", points=[(0.0, 0.0, 0.0)], centroid=(0.0, 0.0, 0.0),
                    extent=(0.0, 0.0, 0.0), n_points=1),
    ])]
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None)
    assert result.rows == []
    assert len(result.skipped) == 1


def _good_pair(offset_m=0.35, offset_deg=4.0):
    """A known-GOOD case (real mismatch, cleanly recoverable), for the gate
    tests below to reuse and only vary the gate under test."""
    payloads, poses, anchors, _, _ = _true_and_wrong_pair(error_yaw=offset_deg, error_m=offset_m)
    return payloads, poses, anchors


def test_large_but_confident_correction_is_applied_not_rejected_by_distance():
    """A large, confident, level correction from a rough hand placement is the normal case and must not be rejected
    just for being a few metres."""
    payloads, poses, anchors = _good_pair(offset_m=3.0, offset_deg=6.0)
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None,
                                icp_distance_m=1.0)
    assert result.rows[0].applied, result.rows[0].reason
    assert result.rows[0].shift_max_m > 1.0  # genuinely a multi-metre correction


def test_tilt_gate_rejects_even_a_good_fit():
    """Deterministic exercise of the tilt-gate rejection branch: an
    impossible threshold must reject, and say why."""
    payloads, poses, anchors = _good_pair()
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None,
                                max_tilt_deg=-1.0)
    assert not result.rows[0].applied
    assert "tip" in result.rows[0].reason


def test_fitness_gate_rejects_even_a_good_fit():
    payloads, poses, anchors = _good_pair()
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None,
                                min_fitness=2.0)
    assert not result.rows[0].applied
    assert "fitness" in result.rows[0].reason


def test_distance_backstop_still_rejects_an_implausible_shift():
    payloads, poses, anchors = _good_pair()
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None,
                                max_shift_m=0.01)
    assert not result.rows[0].applied
    assert "backstop" in result.rows[0].reason


def test_no_real_overlap_is_rejected_on_fitness():
    """Two rooms with no shared geometry: the fitness gate must catch it rather than return a confident-looking but
    meaningless answer."""
    rng = np.random.default_rng(9)
    local_a = _room_points(rng)
    local_b = _room_points(np.random.default_rng(10))
    payloads = {"A": _payload_from_points(local_a), "B": _payload_from_points(local_b)}
    # Placed 500 m apart: no plausible correspondence within any sane threshold.
    poses = {"A": ScanPose(), "B": ScanPose(x_m=500.0)}
    anchors = [_anchor("A", local_a[0], "B", local_b[0])]
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None)
    assert not result.rows[0].applied
    assert "fitness" in result.rows[0].reason
    assert result.poses["B"].x_m == 500.0


def test_duplicate_anchors_naming_the_same_mover_produce_one_row_not_two():
    """Two links naming the same rejected scan still produce exactly one row."""
    payloads, poses, anchors = _good_pair()
    local_a_pt, local_b_pt = anchors[0].members[0].centroid, anchors[0].members[1].centroid
    anchors = [anchors[0], _anchor("A", local_a_pt, "B", local_b_pt, anchor_id="anchor-2")]
    result = align_from_anchors(payloads, poses, anchors, locked={}, subset=None,
                                max_tilt_deg=-1.0)
    assert len(result.rows) == 1


def test_agreement_weights_change_the_result():
    """Per-point agreement weights change the solve."""
    payloads, poses, anchors = _good_pair()
    true_pose_b = ScanPose(yaw_deg=poses["B"].yaw_deg - 4.0,
                           x_m=poses["B"].x_m - 0.35, y_m=poses["B"].y_m,
                           z_m=poses["B"].z_m)
    true_world_b = _world_points(payloads["B"], true_pose_b)

    n = len(payloads["B"]["xy"])
    rng = np.random.default_rng(5)
    half = n // 2
    noisy_local = dict(payloads["B"])
    xy = np.asarray(noisy_local["xy"], dtype=float).copy()
    xy[half:] += rng.uniform(-2.0, 2.0, size=(n - half, 2))  # scramble the noisy half
    noisy_local["xy"] = xy.tolist()
    payloads_noisy = {"A": payloads["A"], "B": noisy_local}

    uniform = align_from_anchors(payloads_noisy, poses, anchors, locked={}, subset=None,
                                 icp_distance_m=1.0)

    payloads_weighted = {"A": payloads["A"], "B": dict(noisy_local)}
    agreement = np.ones(n)
    agreement[half:] = 0.001  # de-emphasise the scrambled half, do not drop it
    payloads_weighted["B"]["agreement"] = agreement.tolist()
    weighted = align_from_anchors(payloads_weighted, poses, anchors, locked={}, subset=None,
                                  icp_distance_m=1.0)

    assert uniform.rows[0].applied and weighted.rows[0].applied
    err_uniform = pose_shift_m(payloads_noisy["B"], uniform.poses["B"], true_pose_b)
    err_weighted = pose_shift_m(payloads_weighted["B"], weighted.poses["B"], true_pose_b)
    assert err_weighted < err_uniform, (
        f"weighting the clean half higher should recover the true pose better: "
        f"uniform error {err_uniform:.3f} m, weighted error {err_weighted:.3f} m"
    )


def test_points_as_drawn_include_hand_tilt_and_match_the_placement_matrix():
    """The heatmap's points sit where the bench draws them: tilt included, identical to Placement.matrix()."""
    from open_point_stitch.align_anchors import _world_cloud_points, world_cloud_points_as_drawn
    from open_point_stitch.placement import Placement

    rng = np.random.default_rng(3)
    payload = {"xy": rng.uniform(-6, 6, (400, 2)).tolist(), "h": rng.uniform(0, 4, 400).tolist()}
    pose = ScanPose(yaw_deg=37.0, x_m=2.5, y_m=-1.0, z_m=0.4, pitch_deg=2.0, roll_deg=-3.5)

    got = world_cloud_points_as_drawn(payload, pose)
    m = Placement(name="t", yaw_deg=37.0, tx=2.5, ty=-1.0, tz=0.4, pitch_deg=2.0, roll_deg=-3.5).matrix()
    local = np.column_stack([np.asarray(payload["xy"]), np.asarray(payload["h"])])
    assert np.allclose(got, local @ m[:3, :3].T + m[:3, 3], atol=1e-9), "not where Placement.matrix() puts them"

    flat = ScanPose(yaw_deg=37.0, x_m=2.5, y_m=-1.0, z_m=0.4)
    assert np.allclose(world_cloud_points_as_drawn(payload, flat), _world_cloud_points(payload, flat), atol=1e-9), (
        "with no tilt the two helpers must agree, or something else about the heatmap moved")
    off = float(np.abs(got - _world_cloud_points(payload, pose)).max())
    assert off > 0.05, f"the untilted helper is only {off:.3f} m off for a 2/-3.5 degree tilt: the test proves nothing"


def test_the_placement_matrix_puts_raw_points_where_the_bench_draws_them_with_a_floor_level():
    """The export and the seeded align use Placement.matrix(), and the bench draws height above the scan's own floor;
    the two must agree when the floor level is not zero."""
    from open_point_stitch.align_anchors import world_cloud_points_as_drawn
    from open_point_stitch.placement import Placement

    rng = np.random.default_rng(11)
    level = np.linalg.qr(np.eye(3) + 0.03 * rng.normal(size=(3, 3)))[0]
    raw = np.column_stack([rng.uniform(-6, 6, 300), rng.uniform(-6, 6, 300), rng.uniform(-2.8, 1.5, 300)])
    floor_z = -1.6
    levelled = raw @ level.T
    payload = {"xy": levelled[:, :2].tolist(), "h": (levelled[:, 2] - floor_z).tolist()}
    pose = ScanPose(yaw_deg=37.0, x_m=2.5, y_m=-1.0, z_m=0.4, pitch_deg=2.0, roll_deg=-3.5)
    bench = world_cloud_points_as_drawn(payload, pose)
    m = Placement(name="t", level_rotation=level, floor_z=floor_z, yaw_deg=37.0, tx=2.5, ty=-1.0, tz=0.4,
                  pitch_deg=2.0, roll_deg=-3.5).matrix()
    exported = raw @ m[:3, :3].T + m[:3, 3]
    assert np.allclose(bench, exported, atol=1e-9), f"export and bench differ by up to {np.abs(bench - exported).max():.3f} m"


def _true_local(world, pose):
    """A scan's own local points for world points seen from its TRUE pose (yaw only, as these tests use)."""
    xy = (world[:, :2] - [pose.x_m, pose.y_m]) @ _yaw_matrix2(pose.yaw_deg)
    return np.column_stack([xy, world[:, 2] - pose.z_m])


def _wall_room(rng, ox=40.0, oy=-18.0, n=3000, half=3.0):
    """World points of two walls (x = ox-3 and y = oy-3) and a floor, each its own array, `half` m either side."""
    wall1 = np.column_stack([np.full(n, ox - 3.0), rng.uniform(oy - half, oy + half, n), rng.uniform(0, 2.5, n)])
    wall2 = np.column_stack([rng.uniform(ox - half, ox + half, n), np.full(n, oy - 3.0), rng.uniform(0, 2.5, n)])
    floor = np.column_stack([rng.uniform(ox - half, ox + half, n), rng.uniform(oy - half, oy + half, n), np.zeros(n)])
    return wall1, wall2, floor


def test_two_anchors_from_two_references_fix_a_scan_in_both_directions():
    """A scan named by two links must satisfy both. A sees only the x wall and B only the y wall, so each alone fixes
    one direction of C; together, with A and B locked, C comes right."""
    rng = np.random.default_rng(21)
    ox, oy = 40.0, -18.0
    big = _wall_room(rng, ox, oy, n=6000, half=8.0)     # what the reference scans see: more wall than C does
    wall1, wall2, floor = _wall_room(rng, ox, oy)
    truth = {"A": ScanPose(yaw_deg=25.0, x_m=ox, y_m=oy, z_m=0.3), "B": ScanPose(yaw_deg=100.0, x_m=ox + 3.0, y_m=oy + 4.0, z_m=0.3),
             "C": ScanPose(yaw_deg=-15.0, x_m=ox - 2.0, y_m=oy - 2.0, z_m=0.3)}
    seen = {"A": np.vstack([big[0], big[2]]), "B": np.vstack([big[1], big[2]]), "C": np.vstack([wall1, wall2, floor])}
    local = {n: _true_local(seen[n], truth[n]) for n in truth}
    payloads = {n: _payload_from_points(local[n]) for n in truth}
    poses = {"A": truth["A"], "B": truth["B"],
             "C": ScanPose(yaw_deg=truth["C"].yaw_deg + 3.0, x_m=truth["C"].x_m + 0.35, y_m=truth["C"].y_m - 0.30, z_m=0.3)}

    def raw(name, world_xyz):
        return _true_local(np.array([world_xyz]), truth[name])[0]

    # Each region on C sits 0.4 m along the wall from the region on the other scan: the seed error a wall cannot correct.
    anchor_x = _anchor("A", raw("A", (ox - 3.0, oy + 1.0, 1.0)), "C", raw("C", (ox - 3.0, oy + 1.4, 1.0)), "x-wall")
    anchor_y = _anchor("B", raw("B", (ox + 1.0, oy - 3.0, 1.0)), "C", raw("C", (ox + 0.6, oy - 3.0, 1.0)), "y-wall")
    locked = {"A": True, "B": True}
    kw = dict(icp_distance_m=0.6, max_shift_m=3.0)

    def error(result):
        return float(np.linalg.norm(_world_points(payloads["C"], result.poses["C"]) - _world_points(payloads["C"], truth["C"]), axis=1).mean())

    only_x = align_from_anchors(payloads, poses, [anchor_x], locked=locked, subset=None, **kw)
    both = align_from_anchors(payloads, poses, [anchor_x, anchor_y], locked=locked, subset=None, **kw)
    assert error(only_x) > 0.15, f"one wall cannot fix the other direction; error {error(only_x):.3f} m proves nothing"
    assert error(both) < 0.05, f"both anchors should place C: error {error(both):.3f} m (one anchor alone {error(only_x):.3f} m)"
    assert [r.scan for r in both.rows if r.applied] == ["C"], [(r.scan, r.applied) for r in both.rows]
    assert both.poses["A"] == poses["A"] and both.poses["B"] == poses["B"]


def test_a_chain_of_free_scans_is_recovered_when_the_anchors_agree():
    """A links B and B links C, nothing locked: C is matched against B's current (wrong) pose, so the joint solve
    has to move B and carry C with it."""
    rng = np.random.default_rng(4)
    ox, oy = 40.0, -18.0
    world = np.vstack(_wall_room(rng, ox, oy, n=6000, half=8.0))
    truth = {"A": ScanPose(25.0, ox, oy, 0.3), "B": ScanPose(100.0, ox + 3.0, oy + 4.0, 0.3),
             "C": ScanPose(-15.0, ox - 2.0, oy - 2.0, 0.3)}
    payloads = {n: _payload_from_points(_true_local(world, truth[n])) for n in truth}
    wrong = {"A": truth["A"],
             "B": ScanPose(truth["B"].yaw_deg + 3.0, truth["B"].x_m + 0.35, truth["B"].y_m - 0.2, truth["B"].z_m),
             "C": ScanPose(truth["C"].yaw_deg - 2.0, truth["C"].x_m - 0.4, truth["C"].y_m + 0.3, truth["C"].z_m)}

    def anchor(i, a, b, w):
        def member(n):
            p = tuple(float(v) for v in _true_local(np.array([w]), truth[n])[0])
            return AnchorMember(scan=n, points=[p], centroid=p, extent=(0.0, 0.0, 0.0), n_points=1)
        return Anchor(id=f"chain-{i}", members=[member(a), member(b)])

    anchors = [anchor(1, "A", "B", (ox - 3.0, oy + 1.0, 1.0)), anchor(2, "A", "B", (ox + 1.0, oy - 3.0, 1.0)),
               anchor(3, "B", "C", (ox - 3.0, oy - 1.0, 1.2)), anchor(4, "B", "C", (ox + 2.0, oy - 3.0, 0.8))]
    assert pose_shift_m(payloads["B"], wrong["B"], truth["B"]) > 0.5 and pose_shift_m(payloads["C"], wrong["C"], truth["C"]) > 0.3
    result = align_from_anchors(payloads, wrong, anchors, locked={}, subset=None, icp_distance_m=0.6, max_shift_m=3.0)
    for name in ("B", "C"):
        err = pose_shift_m(payloads[name], result.poses[name], truth[name])
        assert err < 0.03, f"{name} is still {err:.3f} m out after the chain was solved"
    assert result.poses["A"] == wrong["A"], "the first-listed scan of a free group must stay where it is"


def test_a_group_of_free_scans_keeps_its_first_listed_scan_still():
    rng = np.random.default_rng(5)
    local = _room_points(rng)
    payloads = {"A": _payload_from_points(local), "B": _payload_from_points(local)}
    poses = {"A": ScanPose(x_m=1.0), "B": ScanPose(x_m=5.0)}
    result = align_from_anchors(payloads, poses, [_anchor("A", local[0], "B", local[0])], locked={}, subset=None)
    assert result.poses["A"].x_m == 1.0 and abs(result.poses["B"].x_m - 1.0) < 0.05, (result.poses["A"], result.poses["B"])


def _two_independent_anchors():
    """Two unrelated room pairs (A<->B, C<->D) in one call, so cancelling after the first
    anchor cannot be confused with the second one simply never having been reachable."""
    payloads1, poses1, anchors1, _, _ = _true_and_wrong_pair(seed=11)
    payloads2, poses2, anchors2, _, _ = _true_and_wrong_pair(seed=12)
    payloads = {"A": payloads1["A"], "B": payloads1["B"], "C": payloads2["A"], "D": payloads2["B"]}
    poses = {"A": poses1["A"], "B": poses1["B"], "C": poses2["A"], "D": poses2["B"]}
    m1a, m1b = anchors1[0].members
    m2a, m2b = anchors2[0].members
    anchor2 = _anchor("C", m2a.points[0], "D", m2b.points[0], anchor_id="anchor-2")
    return payloads, poses, [anchors1[0], anchor2]


def test_progress_cb_runs_once_per_anchor_in_order():
    payloads, poses, anchors = _two_independent_anchors()
    calls = []
    align_from_anchors(payloads, poses, anchors, locked={}, subset=None,
                       icp_distance_m=0.3, max_shift_m=2.0,
                       progress_cb=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 2), (2, 2)], calls


def test_should_cancel_stops_before_the_next_anchor_and_discards_everything():
    # Cancel discards everything, not just the unfinished part.
    payloads, poses, anchors = _two_independent_anchors()
    calls = []
    result = align_from_anchors(
        payloads, poses, anchors, locked={}, subset=None,
        icp_distance_m=0.3, max_shift_m=2.0,
        progress_cb=lambda done, total: calls.append((done, total)),
        should_cancel=lambda: len(calls) >= 1,  # true only once anchor 1's own progress has fired
    )
    assert calls == [(1, 2)], "cancelled before anchor 2's progress_cb, not after it"
    assert result.cancelled is True
    assert result.n_anchors_completed == 1
    assert result.rows == []
    for name in ("A", "B", "C", "D"):
        assert vars(result.poses[name]) == vars(poses[name]), (
            f"{name} moved on a cancelled run, even though anchor 1 (A<->B) finished its own ICP solve "
            "before the cancel landed - the pose graph was never solved, so nothing should be applied")


def _write_pts(folder, name, raw_points):
    """A .pts file (count line, then X Y Z I R G B), read the same way as a real scan."""
    with open(os.path.join(folder, f"{name}.pts"), "w") as fh:
        fh.write(f"{len(raw_points)}\n")
        for x, y, z in raw_points:
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} -1 128 128 128\n")


def _small_tilt(deg, axis=(1.0, 0.5, 0.0)):
    """A scan's own levelling: a small rotation about a horizontal axis (Rodrigues)."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * kx + (1 - np.cos(t)) * (kx @ kx)


def _shared_corner_scene(folder, box_extent=(2.4, 2.4, 2.4), thin=12):
    """Two scans sharing one corner (two walls and a floor patch) and otherwise seeing different rooms, each with a
    hand-drawn box round the corner. Both are tilted in their raw frames with their own floor level, and B is placed
    wrong. Preview points are every thin-th full-resolution point. Returns payloads, poses, the link and B's true
    pose."""
    rng = np.random.default_rng(31)
    ox, oy = 40.0, -18.0
    fx, fy = ox + 2.0, oy + 1.0
    n = 2500
    corner = np.vstack([
        np.column_stack([rng.uniform(fx, fx + 1.5, n), np.full(n, fy), rng.uniform(0, 1.2, n)]),
        np.column_stack([np.full(n, fx), rng.uniform(fy, fy + 1.5, n), rng.uniform(0, 1.2, n)]),
        np.column_stack([rng.uniform(fx, fx + 1.5, n), rng.uniform(fy, fy + 1.5, n), np.zeros(n)]),
    ])

    def floor(x0, x1, y0, y1, m):
        return np.column_stack([rng.uniform(x0, x1, m), rng.uniform(y0, y1, m), np.zeros(m)])

    def wall_x(x, m):       # a long wall facing +x, and one facing +y below
        return np.column_stack([np.full(m, x), rng.uniform(oy - 8, oy + 8, m), rng.uniform(0, 2.5, m)])

    def wall_y(y, m):
        return np.column_stack([rng.uniform(ox - 8, ox + 8, m), np.full(m, y), rng.uniform(0, 2.5, m)])

    # Decoys: each scan sees a DIFFERENT wall in the same direction 35 to 40 cm from the other scan's (a wall and the
    # partition beside it). Matching whole scans pairs them up and is dragged toward closing that gap.
    only_a = np.vstack([floor(ox - 8, ox + 8, oy - 8, oy + 8, 6000), wall_x(ox - 6.0, 5000), wall_y(oy + 7.0, 5000)])
    only_b = np.vstack([floor(ox - 8, ox + 8, oy - 8, oy + 8, 6000), wall_x(ox - 6.4, 5000), wall_y(oy + 7.35, 5000)])
    world = {"A": np.vstack([corner, only_a]), "B": np.vstack([corner, only_b])}
    truth = {"A": ScanPose(yaw_deg=25.0, x_m=ox, y_m=oy, z_m=0.3), "B": ScanPose(yaw_deg=100.0, x_m=ox + 3.0, y_m=oy + 4.0, z_m=0.3)}
    level = {"A": _small_tilt(2.0), "B": _small_tilt(-1.5, axis=(0.3, 1.0, 0.0))}
    floor_z = {"A": -1.5, "B": -1.2}
    centre = np.array([[fx + 0.75, fy + 0.75, 0.6]])            # the middle of the corner, in the world
    drawn_off = {"A": np.array([0.10, -0.10, 0.0]), "B": np.array([-0.15, 0.10, 0.05])}   # hand-drawn, not exact

    payloads, members = {}, []
    for name in ("A", "B"):
        local = _true_local(world[name], truth[name])           # leveled frame, height above this scan's floor
        raw = np.column_stack([local[:, :2], local[:, 2] + floor_z[name]]) @ level[name]
        _write_pts(folder, name, raw)
        payloads[name] = {"xy": local[::thin, :2].tolist(), "h": local[::thin, 2].tolist(),
                          "level_rotation": level[name].tolist(), "floor_z": floor_z[name]}
        c_local = _true_local(centre, truth[name])[0] + drawn_off[name]
        c_raw = np.array([c_local[0], c_local[1], c_local[2] + floor_z[name]]) @ level[name]
        members.append(AnchorMember(scan=name, points=[tuple(c_raw)], centroid=tuple(c_raw), extent=tuple(box_extent), n_points=1))
    poses = {"A": truth["A"], "B": ScanPose(yaw_deg=truth["B"].yaw_deg + 4.0, x_m=truth["B"].x_m + 0.35,
                                            y_m=truth["B"].y_m - 0.30, z_m=truth["B"].z_m)}
    return payloads, poses, [Anchor(id="corner", members=members)], truth["B"]


def _error_m(payloads, result, true_pose):
    return pose_shift_m(payloads["B"], result.poses["B"], true_pose)


def _row_b(result):
    """B's row (A is locked in these scenes, so its own "locked" row comes first)."""
    return next(r for r in result.rows if r.scan == "B")


def test_the_match_runs_on_the_boxes_full_resolution_points_and_ignores_the_rest_of_each_scan():
    """The link boxes are the region that gets matched. The corner is all the two scans share; everything else each
    sees must not pull the answer."""
    with tempfile.TemporaryDirectory() as folder:
        payloads, poses, anchors, truth_b = _shared_corner_scene(folder)
        kw = dict(locked={"A": True}, subset=None, max_shift_m=2.0)
        start_err = pose_shift_m(payloads["B"], poses["B"], truth_b)
        boxed = align_from_anchors(payloads, poses, anchors, scans_folder=folder, **kw)
        whole = align_from_anchors(payloads, poses, anchors, **kw)          # no scans folder: the old whole-scan match
        assert start_err > 0.2, f"B starts only {start_err:.3f} m out; proves nothing"
        assert _error_m(payloads, whole, truth_b) > 0.2, (
            "the whole-scan match also gets this scene right, so it does not show the boxes doing anything")
        assert _row_b(boxed).applied, _row_b(boxed).reason
        assert "full resolution" in _row_b(boxed).matched_on, _row_b(boxed).matched_on
        assert "matching moved the boxes" in _row_b(boxed).matched_on, _row_b(boxed).matched_on
        err = _error_m(payloads, boxed, truth_b)
        assert err < 0.03, f"boxed match left B {err:.3f} m out (started {start_err:.3f} m)"


def test_matching_is_only_taken_over_the_anchor_seed_when_it_lines_more_of_the_box_up():
    """Two flat patches of the same floor: the anchor seed already puts every point on the other patch, so no
    matching answer can beat it by the margin and the seed must stand, exactly. A patch of a WALL against the
    same wall 30 cm off is the opposite: the seed leaves it 30 cm out and matching must take over."""
    from open_point_stitch.align_anchors import SEED_KEEP_MARGIN, _MatchSet, _fit_pair
    rng = np.random.default_rng(12)
    floor = np.column_stack([rng.uniform(-2, 2, 30000), rng.uniform(-2, 2, 30000), np.zeros(30000)])
    seed = np.eye(4)
    flat = _fit_pair(_MatchSet(floor[:8000], np.ones(8000), floor, "boxes", True), seed, 0.5, 30)
    assert flat.kept_seed and np.array_equal(flat.transform, seed) and flat.fitness > 0.95, (flat.note, flat.fitness)

    wall = np.column_stack([rng.uniform(-2, 2, 20000), np.zeros(20000), rng.uniform(0, 2.5, 20000)])
    wall_near = wall[:6000] + [0.0, 0.3, 0.0]                  # the same wall, 30 cm off in the direction it faces
    off_seed = np.eye(4)
    fixed = _fit_pair(_MatchSet(wall_near, np.ones(6000), wall, "boxes", True), off_seed, 0.5, 30)
    assert not fixed.kept_seed and abs(fixed.transform[1, 3] + 0.3) < 0.02, (fixed.note, fixed.transform[:3, 3])
    assert SEED_KEEP_MARGIN > 0


def test_flat_boxes_fix_the_height_and_the_run_says_the_placement_was_kept():
    """Two boxes on the same flat floor pin down height and nothing else."""
    with tempfile.TemporaryDirectory() as folder:
        rng = np.random.default_rng(8)
        ox, oy = 40.0, -18.0
        n = 40000
        floor = np.column_stack([rng.uniform(ox - 3, ox + 3, n), rng.uniform(oy - 3, oy + 3, n), np.zeros(n)])
        truth = {"A": ScanPose(yaw_deg=25.0, x_m=ox, y_m=oy, z_m=0.3), "B": ScanPose(yaw_deg=100.0, x_m=ox + 3.0, y_m=oy + 4.0, z_m=0.3)}
        floor_z, payloads, members = -1.5, {}, []
        for name in ("A", "B"):
            local = _true_local(floor, truth[name])
            _write_pts(folder, name, np.column_stack([local[:, :2], local[:, 2] + floor_z]))
            payloads[name] = {"xy": local[::12, :2].tolist(), "h": local[::12, 2].tolist(),
                              "level_rotation": np.eye(3).tolist(), "floor_z": floor_z}
            c = _true_local(np.array([[ox, oy, 0.0]]), truth[name])[0]
            c_raw = (float(c[0]), float(c[1]), float(c[2]) + floor_z)
            members.append(AnchorMember(scan=name, points=[c_raw], centroid=c_raw, extent=(2.0, 2.0, 0.3), n_points=1))
        poses = {"A": truth["A"], "B": ScanPose(yaw_deg=truth["B"].yaw_deg, x_m=truth["B"].x_m,
                                                y_m=truth["B"].y_m, z_m=truth["B"].z_m + 0.2)}
        result = align_from_anchors(payloads, poses, [Anchor(id="floor", members=members)], locked={"A": True},
                                    subset=None, scans_folder=folder, max_shift_m=2.0)
        row = _row_b(result)
        assert row.applied, row.reason
        assert "did not line the boxes up better" in row.matched_on, row.matched_on
        assert abs(result.poses["B"].z_m - truth["B"].z_m) < 0.02, (result.poses["B"].z_m, truth["B"].z_m)


def _drawn_local(world_pts, pose):
    """Inverse of `world_cloud_points_as_drawn`: where world points sit in a tilted scan's own frame."""
    from open_point_stitch.placement import tilt_matrix3
    rot = tilt_matrix3(pose.pitch_deg, pose.roll_deg, pose.yaw_deg)
    return (np.asarray(world_pts, dtype=float) - [pose.x_m, pose.y_m, pose.z_m]) @ rot


def test_links_on_hand_tilted_scans_land_where_the_scans_are_drawn():
    """The link solve uses hand pitch/roll, so link ends meet where the bench draws them."""
    from open_point_stitch.align_anchors import world_cloud_points_as_drawn
    for preview_only in (False, True):
        with tempfile.TemporaryDirectory() as folder:
            rng = np.random.default_rng(9)
            ox, oy, n = 40.0, -18.0, 40000
            floor = np.column_stack([rng.uniform(ox - 3, ox + 3, n), rng.uniform(oy - 3, oy + 3, n), np.zeros(n)])
            truth = {"A": ScanPose(yaw_deg=25.0, x_m=ox - 6.0, y_m=oy, z_m=0.3, roll_deg=2.0),
                     "B": ScanPose(yaw_deg=100.0, x_m=ox + 3.0, y_m=oy + 4.0, z_m=0.3, pitch_deg=-1.5)}
            floor_z, payloads, members = -1.5, {}, []
            for name in ("A", "B"):
                local = _drawn_local(floor, truth[name])
                _write_pts(folder, name, np.column_stack([local[:, :2], local[:, 2] + floor_z]))
                payloads[name] = {"xy": local[::12, :2].tolist(), "h": local[::12, 2].tolist(),
                                  "level_rotation": np.eye(3).tolist(), "floor_z": floor_z}
                c = _drawn_local([[ox, oy, 0.0]], truth[name])[0]
                c_raw = (float(c[0]), float(c[1]), float(c[2]) + floor_z)
                members.append(AnchorMember(scan=name, points=[c_raw], centroid=c_raw, extent=(2.0, 2.0, 0.3), n_points=1))
            poses = {"A": truth["A"], "B": ScanPose(**{**vars(truth["B"]), "z_m": truth["B"].z_m + 0.2})}
            result = align_from_anchors(payloads, poses, [Anchor(id="floor", members=members)], locked={"A": True},
                                        subset=None, scans_folder=None if preview_only else folder, max_shift_m=2.0)
            assert _row_b(result).applied, _row_b(result).reason
            got = world_cloud_points_as_drawn(payloads["B"], result.poses["B"])[:, 2]
            assert abs(float(np.median(got))) < 0.02, (preview_only, float(np.median(got)))


def _two_scans_one_link(folder, world, truth, centres, start_dz=0.0, extent=(2.0, 2.0, 2.0)):
    """Both scans see `world`; one link whose box centres sit at `centres` (world), B starting `start_dz` high."""
    payloads, members = {}, []
    for name in ("A", "B"):
        local = _drawn_local(world, truth[name])
        _write_pts(folder, name, np.column_stack([local[:, :2], local[:, 2] - 1.5]))
        payloads[name] = {"xy": local[::12, :2].tolist(), "h": local[::12, 2].tolist(),
                          "level_rotation": np.eye(3).tolist(), "floor_z": -1.5}
        c = _drawn_local([centres[name]], truth[name])[0]
        c_raw = (float(c[0]), float(c[1]), float(c[2]) - 1.5)
        members.append(AnchorMember(scan=name, points=[c_raw], centroid=c_raw, extent=extent, n_points=1))
    poses = {"A": truth["A"], "B": ScanPose(**{**vars(truth["B"]), "z_m": truth["B"].z_m + start_dz})}
    return payloads, poses, [Anchor(id="link", members=members)]


def test_boxes_drawn_at_different_heights_do_not_move_a_scan_up_or_down():
    """Two boxes round the same wall, one drawn 40 cm higher: only a floor or ceiling may set height, not the box
    centres."""
    rng = np.random.default_rng(21)
    ox, oy, n = 40.0, -18.0, 20000
    walls = np.vstack([np.column_stack([np.full(n, ox), rng.uniform(oy - 3, oy + 3, n), rng.uniform(0.5, 2.5, n)]),
                       np.column_stack([rng.uniform(ox, ox + 3, n), np.full(n, oy + 3), rng.uniform(0.5, 2.5, n)])])
    truth = {"A": ScanPose(yaw_deg=25.0, x_m=ox + 2, y_m=oy, z_m=0.3), "B": ScanPose(yaw_deg=100.0, x_m=ox + 2, y_m=oy + 1, z_m=0.3)}
    with tempfile.TemporaryDirectory() as folder:
        payloads, poses, anchors = _two_scans_one_link(folder, walls, truth, {"A": [ox, oy + 2, 1.0], "B": [ox, oy + 2, 1.4]})
        result = align_from_anchors(payloads, poses, anchors, locked={"A": True}, subset=None, scans_folder=folder,
                                    max_shift_m=2.0)
        row = _row_b(result)
        assert "no floor or ceiling in the boxes" in row.matched_on, row.matched_on
        assert abs(result.poses["B"].z_m - truth["B"].z_m) < 0.02, (result.poses["B"].z_m, truth["B"].z_m)


def test_a_floor_in_the_boxes_sets_the_height_even_when_the_boxes_were_drawn_at_different_heights():
    rng = np.random.default_rng(22)
    ox, oy, n = 40.0, -18.0, 20000
    room = np.vstack([np.column_stack([np.full(n, ox), rng.uniform(oy - 3, oy + 3, n), rng.uniform(0.0, 2.5, n)]),
                      np.column_stack([rng.uniform(ox, ox + 3, n), rng.uniform(oy - 3, oy + 3, n), np.zeros(n)])])
    truth = {"A": ScanPose(yaw_deg=25.0, x_m=ox + 2, y_m=oy, z_m=0.3), "B": ScanPose(yaw_deg=100.0, x_m=ox + 2, y_m=oy + 1, z_m=0.3)}
    with tempfile.TemporaryDirectory() as folder:
        payloads, poses, anchors = _two_scans_one_link(folder, room, truth, {"A": [ox, oy, 0.4], "B": [ox, oy, 1.0]},
                                                       start_dz=0.15)
        result = align_from_anchors(payloads, poses, anchors, locked={"A": True}, subset=None, scans_folder=folder,
                                    max_shift_m=2.0)
        row = _row_b(result)
        assert "where the floors and ceilings in the boxes meet" in row.matched_on, row.matched_on
        assert abs(result.poses["B"].z_m - truth["B"].z_m) < 0.02, (result.poses["B"].z_m, truth["B"].z_m)


def test_whole_scan_preview_match_is_still_what_you_get_without_the_scan_files():
    """No scans folder (or a box too thin to trust) keeps the old whole-scan preview match, and says so."""
    with tempfile.TemporaryDirectory() as folder:
        payloads, poses, anchors, truth_b = _shared_corner_scene(folder)
        result = align_from_anchors(payloads, poses, anchors, locked={"A": True}, subset=None, max_shift_m=2.0)
        assert _row_b(result).matched_on.startswith("the whole scan at preview resolution"), _row_b(result).matched_on
        assert "not available" in _row_b(result).matched_on


def test_a_small_box_is_grown_until_it_holds_enough_points_and_says_so():
    """A 15 cm cube round the middle of the corner holds a few dozen of its 7,500 points, under the minimum. It is
    grown into its neighbourhood and still matched at full resolution, not dropped for the whole scan."""
    with tempfile.TemporaryDirectory() as folder:
        payloads, poses, anchors, truth_b = _shared_corner_scene(folder, box_extent=(0.15, 0.15, 0.15))
        result = align_from_anchors(payloads, poses, anchors, locked={"A": True}, subset=None, scans_folder=folder,
                                    max_shift_m=2.0)
        said = _row_b(result).matched_on
        assert said.startswith("the boxes at full resolution") and "grown by" in said, said
        assert _row_b(result).applied, _row_b(result).reason
        assert _error_m(payloads, result, truth_b) < 0.05, said


def test_a_spot_with_almost_no_points_even_grown_falls_back_to_the_whole_scan_and_says_how_few():
    from open_point_stitch.align_anchors import MIN_REGION_POINTS, _BoxPoints, _points_to_match
    payload = _payload_from_points(_room_points(np.random.default_rng(0)))
    member = AnchorMember(scan="B", points=[], centroid=(0.0, 0.0, 0.0), extent=(0.2, 0.2, 0.2), n_points=0)
    got = _points_to_match("A", payload, ScanPose(), _BoxPoints(np.zeros((5000, 3))), np.zeros((5000, 3)),
                           "B", payload, ScanPose(), _BoxPoints(np.zeros((12, 3)), grown_m=1.0), member)
    assert not got.used_regions
    assert got.matched_on.startswith("the whole scan at preview resolution"), got.matched_on
    assert "too few full-resolution points even grown" in got.matched_on and f"needs {MIN_REGION_POINTS}" in got.matched_on


def test_a_missing_scan_file_falls_back_and_is_logged_not_silent():
    with tempfile.TemporaryDirectory() as folder:
        payloads, poses, anchors, _ = _shared_corner_scene(folder)
        os.remove(os.path.join(folder, "B.pts"))
        result, logged = _capture_log(lambda: align_from_anchors(payloads, poses, anchors, locked={"A": True}, subset=None,
                                                                  scans_folder=folder, max_shift_m=2.0))
        assert "not available" in _row_b(result).matched_on, _row_b(result).matched_on
        assert any(r.levelno >= logging.WARNING and "'B'" in r.getMessage() for r in logged), [r.getMessage() for r in logged]


def test_an_unreadable_scan_file_falls_back_and_is_logged_once_however_many_anchors_name_it():
    with tempfile.TemporaryDirectory() as folder:
        payloads, poses, anchors, _ = _shared_corner_scene(folder)
        with open(os.path.join(folder, "B.pts"), "w") as fh:
            fh.write("this is not a point cloud\n")
        m_a, m_b = anchors[0].members
        two = anchors + [Anchor(id="corner-2", members=[m_a, m_b])]
        result, logged = _capture_log(lambda: align_from_anchors(payloads, poses, two, locked={"A": True}, subset=None,
                                                                  scans_folder=folder, max_shift_m=2.0))
        assert "not available" in _row_b(result).matched_on, _row_b(result).matched_on
        errors = [r for r in logged if r.levelno >= logging.ERROR]
        assert len(errors) == 1, [r.getMessage() for r in logged]
        assert "B" in errors[0].getMessage() and "preview" in errors[0].getMessage()


def _capture_log(run):
    """Run `run()` and return (its result, the log records `open_point_stitch.align_anchors` emitted meanwhile)."""
    records: list[logging.LogRecord] = []

    class _Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("open_point_stitch.align_anchors")
    handler, old_level = _Grab(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        return run(), records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def test_full_res_region_returns_exactly_the_points_in_the_box_and_reads_each_file_once():
    from open_point_stitch.align_anchors import full_res_region
    with tempfile.TemporaryDirectory() as folder:
        pts = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, -0.5, 0.0], [5.0, 5.0, 5.0]])
        _write_pts(folder, "S", pts)
        member = AnchorMember(scan="S", points=[], centroid=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), n_points=0)
        cache: dict = {}
        got = full_res_region(folder, member, cache)
        assert got.tolist() == [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, -0.5, 0.0]], got   # 0.5 m half-width is inclusive
        os.remove(os.path.join(folder, "S.pts"))                                            # gone from disk, still in the cache
        assert len(full_res_region(folder, member, cache, margin_m=0.2)) == 4               # 0.6 m is inside once grown by 0.2
        assert full_res_region(folder, AnchorMember(scan="nope", points=[], centroid=(0, 0, 0), extent=(1, 1, 1), n_points=0), cache) is None


def test_raw_to_world_points_agrees_with_the_preview_convention_and_the_single_point_form():
    """The full-resolution points are moved to the world by the same rule the preview points are, or the two paths
    would disagree about where a scan sits."""
    from open_point_stitch.align_anchors import raw_to_world_point, raw_to_world_points, world_cloud_points_as_drawn
    rng = np.random.default_rng(4)
    level, floor_z = _small_tilt(2.5, axis=(0.2, 1.0, 0.0)), -1.6
    raw = np.column_stack([rng.uniform(-6, 6, 50), rng.uniform(-6, 6, 50), rng.uniform(-2.8, 1.5, 50)])
    leveled = raw @ level.T
    payload = {"xy": leveled[:, :2].tolist(), "h": (leveled[:, 2] - floor_z).tolist()}
    pose = ScanPose(yaw_deg=37.0, x_m=2.5, y_m=-1.0, z_m=0.4, pitch_deg=1.5, roll_deg=-2.0)
    many = raw_to_world_points(raw, level, floor_z, pose)
    assert np.allclose(many, world_cloud_points_as_drawn(payload, pose), atol=1e-9)
    assert np.allclose(many[7], raw_to_world_point(raw[7], level, floor_z, pose), atol=1e-12)


def test_points_in_box_is_inclusive_and_never_narrower_than_the_floor():
    from open_point_stitch.geometry import MIN_HALF_EXTENT_M, points_in_box
    pts = np.array([[0.5, 0.0, 0.0], [0.5001, 0.0, 0.0], [MIN_HALF_EXTENT_M, 0.0, 0.0], [MIN_HALF_EXTENT_M + 0.01, 0.0, 0.0]])
    assert points_in_box(pts, (0, 0, 0), (1.0, 1.0, 1.0)).tolist() == [True, False, True, True]
    assert points_in_box(pts, (0, 0, 0), (0.0, 0.0, 0.0)).tolist() == [False, False, True, False]   # a click still gets a margin


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
