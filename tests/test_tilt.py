"""Tests for align_tilt.py: suggested height, pitch and roll from floor and ceiling agreement, on scenes with a
known answer."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.align_anchors import ScanPose  # noqa: E402
from open_point_stitch.align_tilt import robust_plane, suggest_tilt, surface_map  # noqa: E402
from open_point_stitch.placement import tilt_matrix3  # noqa: E402

CEILING_Z = 4.0


def _room(x0, x1, y0, y1, rake=0.0, step_region=None, seed=0):
    """World points on a floor (optionally raked), a flat ceiling and a little furniture."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(x0, x1, 0.05), np.arange(y0, y1, 0.05))
    x, y = gx.ravel(), gy.ravel()
    floor = np.column_stack([x, y, rake * x])
    if step_region is not None:
        lo, hi, rise = step_region
        floor[(x >= lo) & (x < hi), 2] += rise
    sel = rng.random(len(x)) < 0.5
    ceiling = np.column_stack([x[sel], y[sel], np.full(sel.sum(), CEILING_Z)])
    pick = rng.random(len(x)) < 0.05
    clutter = np.column_stack([x[pick], y[pick], rng.uniform(0.4, 1.8, pick.sum())])
    return np.vstack([floor, ceiling, clutter])


def _payload(world, pose: ScanPose):
    """What the bench stores for a scan whose TRUE pose is `pose`: its points in its own local frame."""
    rot = tilt_matrix3(pose.pitch_deg, pose.roll_deg, pose.yaw_deg)
    local = (world - np.array([pose.x_m, pose.y_m, pose.z_m])) @ rot
    return {"xy": local[:, :2].tolist(), "h": local[:, 2].tolist()}


def _scene(rake=0.0, step_region=None):
    truth = {
        "A": ScanPose(yaw_deg=0.0, x_m=0.0, y_m=0.0, z_m=0.0),
        "B": ScanPose(yaw_deg=35.0, x_m=9.0, y_m=1.0, z_m=0.02, pitch_deg=0.4, roll_deg=-0.3),
        "C": ScanPose(yaw_deg=-20.0, x_m=4.0, y_m=-1.0, z_m=-0.03),
    }
    worlds = {"A": _room(0, 10, 0, 10, rake, seed=1),
              "B": _room(6, 16, 0, 10, rake, step_region, seed=2),
              "C": _room(3, 13, 0, 10, rake, seed=3)}
    payloads = {n: _payload(worlds[n], truth[n]) for n in truth}
    return truth, payloads


def _wrong(truth, **errors):
    out = {n: ScanPose(**vars(p)) for n, p in truth.items()}
    for n, (dp, dr, dz) in errors.items():
        out[n].pitch_deg += dp
        out[n].roll_deg += dr
        out[n].z_m += dz
    return out


def test_recovers_a_tilt_and_height_error_against_a_locked_reference():
    truth, payloads = _scene()
    start = _wrong(truth, B=(1.2, -0.8, 0.06), C=(0.0, 2.0, -0.05))
    got = suggest_tilt(payloads, start, locked={"A": True})
    for n, tol_deg in (("B", 0.08), ("C", 0.08)):
        s = got.suggestions[n]
        assert abs(s.pitch_deg - truth[n].pitch_deg) < tol_deg, (n, s.pitch_deg, truth[n].pitch_deg)
        assert abs(s.roll_deg - truth[n].roll_deg) < tol_deg, (n, s.roll_deg, truth[n].roll_deg)
    # Height at the scan's own middle is what the solve pins down; the pose z also carries the tilt lever.
    world_mid_err = {}
    for n in ("B", "C"):
        s = got.suggestions[n]
        pts = np.array(payloads[n]["xy"]), np.array(payloads[n]["h"])
        local = np.column_stack([pts[0], pts[1]])
        want = local @ tilt_matrix3(truth[n].pitch_deg, truth[n].roll_deg, truth[n].yaw_deg).T + [truth[n].x_m, truth[n].y_m, truth[n].z_m]
        have = local @ tilt_matrix3(s.pitch_deg, s.roll_deg, truth[n].yaw_deg).T + [s.x_m, s.y_m, s.z_m]
        world_mid_err[n] = float(np.abs(want[:, 2] - have[:, 2]).mean())
    assert max(world_mid_err.values()) < 0.01, world_mid_err


def test_pair_reports_show_the_disagreement_shrink():
    truth, payloads = _scene()
    got = suggest_tilt(payloads, _wrong(truth, B=(1.2, -0.8, 0.06)), locked={"A": True})
    before = max(abs(r.slope_deg or 0.0) for r in got.before if r.kind == "floor")
    after = max(abs(r.slope_deg or 0.0) for r in got.after if r.kind == "floor")
    assert before > 0.8 and after < 0.15, (before, after)


def test_a_raked_floor_is_not_flattened():
    """Both scans see the same rake, so it must not show up as a disagreement to correct away."""
    truth, payloads = _scene(rake=np.tan(np.radians(4.0)))
    start = _wrong(truth, B=(0.0, 1.5, 0.04))
    got = suggest_tilt(payloads, start, locked={"A": True, "C": True})
    s = got.suggestions["B"]
    assert abs(s.roll_deg - truth["B"].roll_deg) < 0.1 and abs(s.pitch_deg - truth["B"].pitch_deg) < 0.1, s


def test_a_step_in_one_scans_floor_does_not_drag_the_answer():
    """A platform seen by one scan only is a wrong reading for those cells, not a tilt."""
    truth, payloads = _scene(step_region=(12.0, 16.0, 0.3))
    got = suggest_tilt(payloads, _wrong(truth, B=(0.8, 0.0, 0.0)), locked={"A": True, "C": True})
    s = got.suggestions["B"]
    assert abs(s.pitch_deg - truth["B"].pitch_deg) < 0.15 and abs(s.roll_deg - truth["B"].roll_deg) < 0.15, s


def test_hand_tilt_already_right_stays_put():
    truth, payloads = _scene()
    got = suggest_tilt(payloads, truth, locked={"A": True})
    for n in ("B", "C"):
        s = got.suggestions[n]
        assert abs(s.d_pitch_deg) < 0.05 and abs(s.d_roll_deg) < 0.05 and abs(s.d_z_m) < 0.01, s


def test_a_scan_that_shares_nothing_is_left_alone_and_named():
    truth, payloads = _scene()
    far = ScanPose(x_m=200.0, y_m=200.0)
    payloads["D"] = _payload(_room(200, 210, 200, 210, seed=4), far)
    start = _wrong(truth, B=(1.0, 0.0, 0.0))
    start["D"] = far
    got = suggest_tilt(payloads, start, locked={"A": True})
    assert "D" in got.skipped and "D" not in got.suggestions, got.skipped


def test_a_fix_bigger_than_a_tilt_correction_is_refused_not_applied():
    truth, payloads = _scene()
    got = suggest_tilt(payloads, _wrong(truth, C=(0.0, 3.0, 0.0)), locked={"A": True}, max_change_deg=1.0)
    assert "C" in got.skipped and "C" not in got.suggestions, (got.skipped, list(got.suggestions))


def test_locked_and_out_of_subset_scans_never_move():
    truth, payloads = _scene()
    start = _wrong(truth, B=(1.0, 1.0, 0.05), C=(0.5, 0.0, 0.0))
    got = suggest_tilt(payloads, start, locked={"A": True}, subset={"B"})
    assert set(got.suggestions) == {"B"}, list(got.suggestions)


def test_robust_plane_ignores_a_minority_step():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 10, (400, 2))
    d = 0.02 + 0.01 * xy[:, 0] - 0.005 * xy[:, 1] + rng.normal(0, 0.005, 400)
    d[:80] += 0.4
    coef, keep, _ = robust_plane(xy, d)
    assert abs(coef[1] - 0.01) < 0.002 and abs(coef[2] + 0.005) < 0.002, coef
    assert keep.sum() > 300


def test_surface_map_needs_a_clean_layer():
    """A cell holding only a spread of heights (no floor seen) must not count as a floor."""
    rng = np.random.default_rng(6)
    spread = np.column_stack([rng.uniform(0, 0.24, 60), rng.uniform(0, 0.24, 60), np.linspace(0.0, 3.0, 60)])
    m = surface_map(spread, spread[:, 2])
    assert len(m.floor.keys) == 0 and len(m.ceiling.keys) == 0


def test_a_scan_that_never_saw_the_floor_gives_no_floor_reading():
    """Points only at ceiling height must not become a floor reading, or a ceiling gets compared to a floor."""
    rng = np.random.default_rng(7)
    ceiling_only = np.column_stack([rng.uniform(0, 0.24, 80), rng.uniform(0, 0.24, 80), rng.normal(4.0, 0.01, 80)])
    m = surface_map(ceiling_only, ceiling_only[:, 2])
    assert len(m.floor.keys) == 0 and len(m.ceiling.keys) == 1


def test_a_pair_comparing_a_floor_to_a_ceiling_is_left_out_of_the_solve():
    """Two scans whose readings differ by about four metres are not the same surface; neither may move."""
    truth, payloads = _scene()
    payloads["Z"] = _payload(_room(0, 10, 0, 10, seed=9), ScanPose(z_m=-4.0))
    start = dict(truth)
    start["Z"] = ScanPose(z_m=-4.0)
    got = suggest_tilt(payloads, start, locked={"A": True, "B": True, "C": True}, surfaces=("floor",))
    assert "Z" in got.skipped and "Z" not in got.suggestions, (got.skipped, list(got.suggestions))


def test_settling_after_a_few_rounds_keeps_a_clean_answer_and_shrinks_later_steps():
    """Half-sized steps after `settle_after` rounds stop the answer jittering on real data; on a clean scene they must not
    change where it lands, and turned up hard (settle from round 0) the answer stays nearer the start."""
    truth, payloads = _scene()
    start = _wrong(truth, B=(2.5, -1.8, 0.08), C=(-1.0, 2.2, -0.06))
    plain = suggest_tilt(payloads, start, locked={"A": True}, settle_after=99)
    settled = suggest_tilt(payloads, start, locked={"A": True})
    for n in ("B", "C"):
        assert abs(settled.suggestions[n].pitch_deg - plain.suggestions[n].pitch_deg) < 0.1, n
        assert abs(settled.suggestions[n].roll_deg - plain.suggestions[n].roll_deg) < 0.1, n
    hard = suggest_tilt(payloads, start, locked={"A": True}, settle_after=0)
    moved = lambda r: sum(abs(r.suggestions[n].d_pitch_deg) + abs(r.suggestions[n].d_roll_deg) for n in ("B", "C"))
    assert moved(hard) < moved(plain), (moved(hard), moved(plain))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
