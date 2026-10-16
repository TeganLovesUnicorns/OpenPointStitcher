"""Tests for align_graph_4dof.py: what the solve does with constraints that pin down some directions and not
others."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.align_graph_4dof import Constraint, edge_information, robust_edge, solve  # noqa: E402

ORIGIN = np.zeros(3)


def _plane_pairs(normal, n=1500, seed=0):
    """Matched points sitting on a plane through the origin with this normal: b on it, a on it, both spread out."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=float)
    u = np.cross(normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    pts = rng.uniform(-4, 4, (n, 1)) * u + rng.uniform(-4, 4, (n, 1)) * v
    return pts, pts.copy(), np.tile(normal, (n, 1)), np.ones(n)


def _constraint(normal, target, b="b", a="a"):
    pts_b, pts_a, nrm, w = _plane_pairs(normal)
    return Constraint(b=b, a=a, target_b=np.asarray(target, dtype=float),
                      info=edge_information(pts_b, pts_a, nrm, w, ORIGIN, ORIGIN))


def test_a_wall_pins_down_across_it_and_leaves_sliding_alone():
    """The whole reason for carrying information: a wall says nothing about where along itself a scan sits, so a
    sliding error in the target must not leak into the answer."""
    got = solve([_constraint([1, 0, 0], [0.0, 0.3, 0.7, 0.2])], free=["b"])["b"]
    assert abs(got[1] - 0.3) < 0.01, got
    assert abs(got[2]) < 0.01 and abs(got[3]) < 0.01, f"sliding along the wall (y {got[2]:.3f}, z {got[3]:.3f}) leaked in"


def test_two_walls_from_two_references_pin_down_both_directions():
    cons = [_constraint([1, 0, 0], [0.0, 0.3, 0.7, 0.0], a="a1"), _constraint([0, 1, 0], [0.0, -0.4, -0.2, 0.0], a="a2")]
    got = solve(cons, free=["b"])["b"]
    assert abs(got[1] - 0.3) < 0.01 and abs(got[2] + 0.2) < 0.01, got


def test_a_scan_not_listed_as_free_does_not_move_and_is_not_returned():
    got = solve([_constraint([1, 0, 0], [0.0, 0.3, 0.0, 0.0])], free=["b"])
    assert list(got) == ["b"]
    got = solve([_constraint([1, 0, 0], [0.0, 0.3, 0.0, 0.0])], free=[])
    assert got == {}


def test_a_relative_constraint_between_two_free_scans_is_shared_out_evenly():
    got = solve([_constraint([1, 0, 0], [0.0, 0.3, 0.0, 0.0])], free=["a", "b"])
    assert abs(got["b"][1] - 0.15) < 0.01 and abs(got["a"][1] + 0.15) < 0.01, got


def test_yaw_is_solved_too():
    pts_b, pts_a, nrm, w = _plane_pairs([1, 0, 0])
    c = Constraint("b", "a", np.array([np.radians(2.0), 0.0, 0.0, 0.0]), edge_information(pts_b, pts_a, nrm, w, ORIGIN, ORIGIN))
    got = solve([c], free=["b"])["b"]
    assert abs(np.degrees(got[0]) - 2.0) < 0.05, np.degrees(got[0])


def test_information_is_blind_along_the_wall():
    pts_b, pts_a, nrm, w = _plane_pairs([1, 0, 0])
    info = edge_information(pts_b, pts_a, nrm, w, ORIGIN, ORIGIN)
    assert info[1, 1] > 1e5 and info[2, 2] < 1e-6 and info[3, 3] < 1e-6, np.diag(info)


def test_a_wall_matched_to_a_floor_does_not_bias_the_step():
    """Scan b sees a wall the other scan never saw; its low points match a's FLOOR, a large distance the same way every
    time. The step must still be the floor's 7 cm, not a compromise with those pairs."""
    rng = np.random.default_rng(3)
    n = 2000
    a = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), np.zeros(n)])
    b = a.copy()
    b[:, 2] -= 0.07
    stray_b = np.column_stack([rng.uniform(-3, 3, 600), np.full(600, -3.0), rng.uniform(0.05, 0.6, 600)])
    stray_a = stray_b.copy()
    stray_a[:, 2] = 0.0
    up = np.tile([0.0, 0.0, 1.0], (n + 600, 1))
    step, _ = robust_edge(np.vstack([b, stray_b]), np.vstack([a, stray_a]), up, np.ones(n + 600), ORIGIN, ORIGIN)
    assert abs(step[3] - 0.07) < 0.01, f"step {step[3]:.3f} m instead of 0.07"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
