"""Height settle on shared floors and ceilings."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.align_anchors import ScanPose  # noqa: E402
from open_point_stitch.height_settle import settle, solve  # noqa: E402


def _room(rng, x0, n=6000):
    """A floor at z=0 and a ceiling at z=3 over a 6 x 6 m patch starting at x0, as preview payload rows."""
    floor = np.column_stack([rng.uniform(x0, x0 + 6, n), rng.uniform(0, 6, n), np.zeros(n)])
    ceiling = np.column_stack([rng.uniform(x0, x0 + 6, n), rng.uniform(0, 6, n), np.full(n, 3.0)])
    return np.vstack([floor, ceiling])


def _payload(world_pts, pose):
    local = np.asarray(world_pts) - [pose.x_m, pose.y_m, pose.z_m]
    return {"xy": local[:, :2].tolist(), "h": local[:, 2].tolist()}


def test_known_height_errors_are_closed_and_the_locked_scan_stays():
    rng = np.random.default_rng(1)
    true = {"A": ScanPose(), "B": ScanPose(), "C": ScanPose()}
    payloads = {"A": _payload(_room(rng, 0.0), true["A"]), "B": _payload(_room(rng, 3.0), true["B"]),
                "C": _payload(_room(rng, 6.0), true["C"])}
    placed = {"A": ScanPose(), "B": ScanPose(z_m=0.10), "C": ScanPose(z_m=-0.05)}
    dz, gaps = settle(payloads, placed, fixed={"A"})
    assert "A" not in dz and len(gaps) >= 2, (dz, gaps)
    assert abs(dz["B"] + 0.10) < 0.01 and abs(dz["C"] - 0.05) < 0.01, dz


def test_scans_outside_the_run_are_not_moved():
    rng = np.random.default_rng(2)
    payloads = {n: _payload(_room(rng, x), ScanPose()) for n, x in (("A", 0.0), ("B", 3.0), ("C", 6.0))}
    placed = {"A": ScanPose(), "B": ScanPose(z_m=0.10), "C": ScanPose(z_m=-0.05)}
    dz, _ = settle(payloads, placed, fixed={"A"}, only={"B"})
    assert set(dz) == {"B"}, dz


def test_one_pair_that_disagrees_with_the_rest_counts_for_less():
    """Four scans in a ring agree; one pair's gap reads 40 cm off (raked seating seen sideways)."""
    gaps = {("A", "B"): (1000, 0.0), ("B", "C"): (1000, 0.0), ("C", "D"): (1000, 0.0),
            ("A", "D"): (1000, 0.0), ("A", "C"): (1000, 0.40), ("B", "D"): (1000, 0.0)}
    dz = solve(gaps, ["A", "B", "C", "D"], fixed={"A"})
    assert max(abs(v) for v in dz.values()) < 0.05, dz


def test_no_shared_floor_means_no_change():
    dz = solve({}, ["A", "B"], fixed={"A"})
    assert dz == {"B": 0.0}, dz


def test_a_scan_with_a_marked_floor_settles_on_that_floor_only():
    # A and B share only a table top at 0.8 m; B's floor is out of A's sight.
    rng = np.random.default_rng(3)
    table = lambda: np.column_stack([rng.uniform(3, 6, 4000), rng.uniform(0, 6, 4000), np.full(4000, 0.8)])
    a_pts = np.vstack([_room(rng, 0.0), table()])
    b_pts = np.vstack([table(), np.column_stack([rng.uniform(7, 9, 4000), rng.uniform(0, 6, 4000), np.zeros(4000)])])
    payloads = {"A": _payload(a_pts, ScanPose()), "B": _payload(b_pts, ScanPose())}
    placed = {"A": ScanPose(), "B": ScanPose(z_m=0.05)}
    dz, gaps = settle(payloads, placed, fixed={"A"})
    assert abs(dz["B"] + 0.05) < 0.01 and gaps, (dz, gaps)            # unmarked: the table counts as a floor
    dz, gaps = settle(payloads, placed, fixed={"A"}, marks={"B": [0.0]})
    assert dz == {"B": 0.0} and not gaps, (dz, gaps)                  # marked: only B's floor counts, unshared


def test_with_found_pieces_a_table_is_not_a_floor_but_a_second_floor_level_is():
    # B's found pieces are its floor at 0 (x 7-9) and a raised floor at 0.8 (x 3-6); A shares only the 0.8 m surface.
    # As a found floor it counts; left out of B's pieces it does not.
    rng = np.random.default_rng(4)
    raised = lambda: np.column_stack([rng.uniform(3, 6, 4000), rng.uniform(0, 6, 4000), np.full(4000, 0.8)])
    a_pts = np.vstack([_room(rng, 0.0), raised()])
    b_pts = np.vstack([raised(), np.column_stack([rng.uniform(7, 9, 4000), rng.uniform(0, 6, 4000), np.zeros(4000)])])
    payloads = {"A": _payload(a_pts, ScanPose()), "B": _payload(b_pts, ScanPose())}
    placed = {"A": ScanPose(), "B": ScanPose(z_m=0.05)}
    square = lambda x0, x1: [[[x0, 0.0], [x1, 0.0], [x1, 6.0], [x0, 6.0]]]
    floor = {"kind": "floor", "normal": [0, 0, 1], "offset": 0.0, "outlines": square(7, 9)}
    level2 = {"kind": "floor", "normal": [0, 0, 1], "offset": 0.8, "outlines": square(3, 6)}
    dz, gaps = settle(payloads, placed, fixed={"A"}, pieces={"B": [floor, level2]})
    assert abs(dz["B"] + 0.05) < 0.01 and gaps, (dz, gaps)            # a found second floor level counts
    dz, gaps = settle(payloads, placed, fixed={"A"}, pieces={"B": [floor, {**level2, "kind": "surface"}]})
    assert dz == {"B": 0.0} and not gaps, (dz, gaps)                  # a found table does not
    dz, _ = settle(payloads, placed, fixed={"A"}, marks={"B": [0.0]}, pieces={"B": [floor, level2]})
    assert dz == {"B": 0.0}, dz                                       # hand marks come before found pieces


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
