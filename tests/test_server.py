"""Tests for server.py's pose-graph landing maths (not the HTTP layer)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.placement import Placement  # noqa: E402
from open_point_stitch.lib.poses import yaw_matrix3  # noqa: E402
from open_point_stitch.server import (  # noqa: E402
    _decompose_pose, _yaw_and_translation_only, _layout_bound_check, _IdleStoppingPool,
)


def test_uncorrected_correction_leaks_the_root_tilt_into_the_child():
    """Applying the full correction (root's tilt included) to a child with zero relative tilt leaves the root's tilt
    as a residual on the child. Confirms the failure the fix addresses."""
    root_target = Placement(name="root", pitch_deg=8.0, roll_deg=-5.0, yaw_deg=30.0,
                            tx=1.0, ty=2.0, tz=0.3).matrix()
    # The pose graph's own arbitrary solved frame for the root - not the same
    # as root_target, which is exactly the point of needing a correction.
    root_solved = np.eye(4)
    root_solved[:3, 3] = [0.4, -0.6, 0.1]

    # Child's TRUE relative pose to the root is pure yaw + translation - no
    # tilt at all, matching the "4 DOF, not 6" architecture.
    child_rel = np.eye(4)
    child_rel[:3, :3] = yaw_matrix3(15.0)
    child_rel[:3, 3] = [3.0, -1.0, 0.0]
    child_solved = root_solved @ child_rel

    correction = root_target @ np.linalg.inv(root_solved)
    world_child_uncorrected = correction @ child_solved

    _, _, _, _, tilt_deg = _decompose_pose(world_child_uncorrected, np.eye(3))
    root_tilt_magnitude = np.degrees(np.arccos(np.clip(root_target[2, 2], -1.0, 1.0)))
    assert tilt_deg > 5.0, "expected the child to inherit a large residual tilt from the root"
    assert abs(tilt_deg - root_tilt_magnitude) < 1.0, (
        "the leaked tilt should track the root's own tilt magnitude"
    )


def test_yaw_only_correction_gives_the_child_a_near_zero_residual():
    """The fix: applying only the yaw+translation part of `correction` to the
    child leaves its reported residual tilt near zero, matching its true
    (zero-tilt) relative pose - not the root's tilt."""
    root_target = Placement(name="root", pitch_deg=8.0, roll_deg=-5.0, yaw_deg=30.0,
                            tx=1.0, ty=2.0, tz=0.3).matrix()
    root_solved = np.eye(4)
    root_solved[:3, 3] = [0.4, -0.6, 0.1]

    child_rel = np.eye(4)
    child_rel[:3, :3] = yaw_matrix3(15.0)
    child_rel[:3, 3] = [3.0, -1.0, 0.0]
    child_solved = root_solved @ child_rel

    correction = root_target @ np.linalg.inv(root_solved)
    correction_yaw_only = _yaw_and_translation_only(correction)
    world_child_fixed = correction_yaw_only @ child_solved

    _, _, _, _, tilt_deg = _decompose_pose(world_child_fixed, np.eye(3))
    assert tilt_deg < 1e-6, f"child's true-zero tilt should survive the fix, got {tilt_deg}deg"


def test_root_itself_still_lands_exactly_on_its_placed_pose():
    """The root needs no correction at all - it is already exactly its own
    placed pose by construction, with or without the fix."""
    root_target = Placement(name="root", pitch_deg=8.0, roll_deg=-5.0, yaw_deg=30.0,
                            tx=1.0, ty=2.0, tz=0.3).matrix()
    root_solved = np.eye(4)
    root_solved[:3, 3] = [0.4, -0.6, 0.1]

    correction = root_target @ np.linalg.inv(root_solved)
    world_root = correction @ root_solved
    assert np.allclose(world_root, root_target, atol=1e-9)


def test_yaw_and_translation_only_preserves_pure_yaw_transforms():
    """When a transform already carries no tilt, stripping tilt is a no-op -
    the helper must not perturb an already-clean yaw+translation matrix."""
    t = np.eye(4)
    t[:3, :3] = yaw_matrix3(42.0)
    t[:3, 3] = [1.0, 2.0, 3.0]
    out = _yaw_and_translation_only(t)
    assert np.allclose(out, t, atol=1e-9)


def test_deviation_heatmap_uses_the_saved_hand_tilt():
    """Two scans whose drawn positions coincide read as zero deviation. Scan B only reaches scan A's plane through its
    saved pitch/roll, so a heatmap that ignored tilt would see it far off."""
    import json
    import tempfile
    import threading
    from types import SimpleNamespace

    from open_point_stitch.lib.poses import tilt_matrix3
    from open_point_stitch.server import _compute_deviation

    gx, gy = np.meshgrid(np.arange(-3.0, 3.01, 0.25), np.arange(-3.0, 3.01, 0.25))
    plane = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])   # scan A: a flat plate at z=0
    rot = tilt_matrix3(2.0, 3.0, 0.0)
    local_b = plane @ rot                                                   # rot.T applied to the world points
    payloads = {
        "A": {"xy": plane[:, :2].tolist(), "h": plane[:, 2].tolist()},
        "B": {"xy": local_b[:, :2].tolist(), "h": local_b[:, 2].tolist()},
    }
    with tempfile.TemporaryDirectory() as out_dir:
        scans = [{"name": "A", "yaw_deg": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
                 {"name": "B", "yaw_deg": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
                  "pitch_deg": 2.0, "roll_deg": 3.0}]
        with open(os.path.join(out_dir, "placement.json"), "w") as fh:
            json.dump({"scans": scans}, fh)
        app = SimpleNamespace(out_dir=out_dir, _lock=threading.Lock(), payloads=payloads)
        got = _compute_deviation(app)
    assert got["ready"], got
    worst = max(v for v in got["deviation_m"]["B"] if v is not None)
    assert worst < 1e-6, f"B is drawn exactly on A's plane, yet the heatmap puts it up to {worst:.3f} m off: tilt ignored"


def test_decompose_pose_is_the_inverse_of_the_matrix_with_a_floor_level():
    """The landing writes back what Placement.matrix() will read, floor level included."""
    level = np.linalg.qr(np.eye(3) + 0.03 * np.random.default_rng(5).normal(size=(3, 3)))[0]
    p = Placement(name="a", level_rotation=level, floor_z=-2.1, yaw_deg=33.0, tx=4.0, ty=-3.0, tz=0.9)
    yaw, x, y, z, _ = _decompose_pose(p.matrix(), level, p.floor_z)
    assert np.allclose([yaw, x, y, z], [33.0, 4.0, -3.0, 0.9], atol=1e-6), (yaw, x, y, z)


def test_a_dropped_tilt_keeps_the_scans_centre_not_its_origin():
    """The solve tipped a hand-tilted scan one more degree; only yaw and position are kept, so its centre, 10 m from
    its origin, must stay where the solve put it."""
    from open_point_stitch.lib.poses import tilt_matrix3
    from open_point_stitch.server import _centre_shift_for_dropped_tilt
    level = np.eye(3)
    hand = Placement(name="a", level_rotation=level, floor_z=-1.5, yaw_deg=40.0, tx=2.0, ty=1.0, tz=0.2,
                     pitch_deg=1.0, roll_deg=2.0)
    extra = np.eye(4)
    extra[:3, :3] = tilt_matrix3(1.0, 0.0, 0.0)            # one more degree of pitch about the world origin
    world = extra @ hand.matrix()
    centre = np.array([10.0, 0.0, 1.0])
    yaw, x, y, z, _ = _decompose_pose(world, level, hand.floor_z)
    kept = Placement(name="a", level_rotation=level, floor_z=hand.floor_z, yaw_deg=yaw, tx=x, ty=y, tz=z,
                     pitch_deg=1.0, roll_deg=2.0).matrix()
    shift, dropped = _centre_shift_for_dropped_tilt(world, kept, centre)
    assert abs(dropped - 1.0) < 0.05, dropped
    fixed = Placement(name="a", level_rotation=level, floor_z=hand.floor_z, yaw_deg=yaw, tx=x + shift[0],
                      ty=y + shift[1], tz=z + shift[2], pitch_deg=1.0, roll_deg=2.0).matrix()
    solved_c = world[:3, :3] @ centre + world[:3, 3]
    assert np.allclose(fixed[:3, :3] @ centre + fixed[:3, 3], solved_c, atol=1e-9)
    assert np.linalg.norm(kept[:3, :3] @ centre + kept[:3, 3] - solved_c) > 0.1, "keeping the origin should miss by more"
    same, none = _centre_shift_for_dropped_tilt(hand.matrix(), hand.matrix(), centre)
    assert np.allclose(same, 0.0) and none < 1e-6


def test_layout_bound_trips_when_the_mean_seed_shift_is_too_large():
    """Several scans each moving a little can be a systemic problem even when none breaks its per-scan bound."""
    shifts = {"a": 0.30, "b": 0.25, "c": 0.35, "d": 0.0}  # d: root, or already capped
    tripped, mean_m = _layout_bound_check(shifts, bound_mean_m=0.20)
    assert tripped is True
    assert abs(mean_m - 0.225) < 1e-9


def test_layout_bound_does_not_trip_under_the_mean():
    """A handful of genuinely small moves must not decline the whole run - the
    bound is a real bound, not a de facto always-off switch."""
    shifts = {"a": 0.05, "b": 0.02, "c": 0.10, "d": 0.0}
    tripped, mean_m = _layout_bound_check(shifts, bound_mean_m=0.20)
    assert tripped is False
    assert abs(mean_m - 0.0425) < 1e-9


def test_layout_bound_zero_scans_never_trips():
    """No candidate scans (e.g. every scan in the run was locked) - nothing to
    average, so the bound cannot fire on an empty run."""
    tripped, mean_m = _layout_bound_check({}, bound_mean_m=0.20)
    assert tripped is False
    assert mean_m == 0.0


def test_layout_bound_already_capped_scans_pull_the_mean_down_not_out():
    """A capped scan contributes its real 0.0 to the mean rather than being dropped from it."""
    shifts = {"wild": 0.60, "steady_1": 0.0, "steady_2": 0.0, "steady_3": 0.0}
    tripped, mean_m = _layout_bound_check(shifts, bound_mean_m=0.20)
    assert abs(mean_m - 0.15) < 1e-9
    assert tripped is False  # 0.15 < 0.20: diluted by the steady scans, as intended


def test_preview_workers_stop_when_idle_and_start_again_for_new_work():
    import time
    pool = _IdleStoppingPool(2)
    pool.IDLE_S = 0.5
    try:
        assert pool.submit(pow, 2, 10).result(timeout=60) == 1024
        assert pool.workers() >= 1, "no worker ran the job"
        time.sleep(1.5)
        assert pool.workers() == 0, "idle workers were kept"
        assert pool.submit(pow, 3, 2).result(timeout=60) == 9, "the pool did not start again for new work"
    finally:
        pool.shutdown(wait=True)
    try:
        pool.submit(pow, 2, 2)
        raise AssertionError("a closed pool accepted work")
    except RuntimeError:
        pass


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
