"""The per-scan and whole-layout movement limits of a seeded run, measured from where each scan started (the default)
or only against the other scans (relative), so a layout that slides as one piece reads as no movement."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.lib.poses import ScanPose, carry_pose, common_motion, world_cloud_points, yaw_matrix2  # noqa: E402
from open_point_stitch.server import MoveCheck, _limit_moves  # noqa: E402


def _payload(seed: int, centre=(0.0, 0.0)) -> dict:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-3, 3, size=(400, 2)) + np.asarray(centre)
    return {"xy": xy.tolist(), "h": rng.uniform(0, 3, size=400).tolist()}


def _slid(pose: ScanPose, yaw_deg: float, shift) -> ScanPose:
    """pose carried by one whole-layout turn about the origin and shift, worked out by hand."""
    xy = yaw_matrix2(yaw_deg) @ np.array([pose.x_m, pose.y_m]) + np.asarray(shift[:2])
    return ScanPose(yaw_deg=pose.yaw_deg + yaw_deg, x_m=float(xy[0]), y_m=float(xy[1]), z_m=pose.z_m + shift[2])


START = {"a": ScanPose(0.0, 0.0, 0.0, 0.0), "b": ScanPose(30.0, 6.0, 1.0, 0.1), "c": ScanPose(-80.0, -4.0, 5.0, -0.2)}
PAYLOADS = {"a": _payload(1), "b": _payload(2), "c": _payload(3)}


def _checks(proposed: dict, fixed=(), locked=()) -> list[MoveCheck]:
    return [MoveCheck(name=n, start=START[n], proposed=proposed[n], payload=PAYLOADS[n], anchor=n in fixed,
                      locked=n in locked, fitness=0.5) for n in START]


def test_common_motion_recovers_a_whole_layout_turn_and_shift():
    before = [world_cloud_points(PAYLOADS[n], START[n]) for n in START]
    after = [world_cloud_points(PAYLOADS[n], _slid(START[n], 12.0, (3.0, -2.0, 0.4))) for n in START]
    yaw, shift = common_motion(before, after)
    assert abs(yaw - 12.0) < 1e-6 and np.allclose(shift, [3.0, -2.0, 0.4], atol=1e-6), (yaw, shift)


def test_carry_pose_moves_a_scan_with_the_layout():
    moved = carry_pose(START["b"], 12.0, np.array([3.0, -2.0, 0.4]))
    want = _slid(START["b"], 12.0, (3.0, -2.0, 0.4))
    assert np.allclose(world_cloud_points(PAYLOADS["b"], moved), world_cloud_points(PAYLOADS["b"], want), atol=1e-9)


def test_a_layout_sliding_three_metres_as_one_piece_reads_as_zero_when_relative():
    proposed = {n: _slid(START[n], 7.0, (3.0, 0.5, 0.2)) for n in START}
    out = _limit_moves(_checks(proposed), 0.5, 5.0, 0.20, relative=True)
    assert not out.capped and not out.layout_tripped, out
    assert max(m for m, _ in out.shifts.values()) < 1e-6, out.shifts
    assert out.final == proposed


def test_the_same_slide_trips_both_limits_when_measured_from_the_start():
    proposed = {n: _slid(START[n], 7.0, (3.0, 0.5, 0.2)) for n in START}
    out = _limit_moves(_checks(proposed), 0.5, 5.0, 0.20, relative=False)
    assert out.capped == {"a", "b", "c"}, out.capped
    assert all(out.final[n] == START[n] for n in START)


def test_one_scan_moved_against_the_rest_is_still_caught_and_carried_with_the_layout():
    proposed = {n: _slid(START[n], 2.0, (1.0, 0.0, 0.0)) for n in START}
    proposed["c"] = ScanPose(proposed["c"].yaw_deg, proposed["c"].x_m + 0.9, proposed["c"].y_m, proposed["c"].z_m)
    out = _limit_moves(_checks(proposed), 0.5, 5.0, 0.20, relative=True)
    assert out.capped == {"c"}, out.capped
    assert not out.layout_tripped
    kept = world_cloud_points(PAYLOADS["c"], out.final["c"])
    with_layout = world_cloud_points(PAYLOADS["c"], _slid(START["c"], 2.0, (1.0, 0.0, 0.0)))
    assert np.abs(kept - with_layout).max() < 1e-6, "a refused scan keeps its place in the layout, not its old spot"
    assert max(out.shifts[n][0] for n in ("a", "b")) < 1e-6, "the odd scan out must not drag the layout's own move"
    row = next(r for r in out.rejected if r["scan"] == "c")
    assert row["limit"] == "per scan" and row["measured"] == "against the other scans"
    assert row["correction_m"] > 0.5 and row["fitness"] == 0.5 and row["limit_m"] == 0.5


def test_a_fixed_scan_out_of_step_with_the_rest_declines_the_run():
    proposed = {n: _slid(START[n], 0.0, (2.0, 0.0, 0.0)) for n in START}
    proposed["a"] = START["a"]                       # the reference scan never moves
    out = _limit_moves(_checks(proposed, fixed=("a",)), 0.5, 5.0, 0.20, relative=True)
    assert out.layout_tripped, out
    assert all(out.final[n] == START[n] for n in START)
    assert any(r["limit"] == "fixed scan" and r["scan"] == "a" for r in out.rejected), out.rejected


def test_measured_from_the_start_matches_the_old_rules():
    """Default mode: per-scan limit against the start pose, whole-layout mean with refused scans counted as 0, locked
    scans left out of both."""
    proposed = dict(START)
    proposed["b"] = ScanPose(START["b"].yaw_deg, START["b"].x_m + 0.3, START["b"].y_m, START["b"].z_m)
    proposed["c"] = ScanPose(START["c"].yaw_deg, START["c"].x_m + 0.8, START["c"].y_m, START["c"].z_m)
    out = _limit_moves(_checks(proposed, locked=("a",)), 0.5, 5.0, 0.20, relative=False)
    assert out.capped == {"c"} and out.final["c"] == START["c"] and out.final["b"] == proposed["b"]
    assert "a" not in out.shifts and out.shifts["c"] == (0.0, 0.0)
    assert abs(out.mean_shift_m - 0.15) < 1e-9 and not out.layout_tripped
    tight = _limit_moves(_checks(proposed, locked=("a",)), 0.5, 5.0, 0.10, relative=False)
    assert tight.layout_tripped and tight.final["b"] == START["b"]
    assert any(r["limit"] == "whole layout" for r in tight.rejected)


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
