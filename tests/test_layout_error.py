"""Tests for layout_error: per-scan distance between two layouts, with the whole-layout move removed."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.layout_error import per_scan_error, summarise  # noqa: E402


def _layout():
    """Five scans spread over a venue-sized area, each with its own turn and height."""
    return {f"s{i}": {"name": f"s{i}", "x_m": x, "y_m": y, "z_m": z, "yaw_deg": yaw}
            for i, (x, y, z, yaw) in enumerate([(0, 0, 0, 0), (8, 1, 0.3, 40), (3, 9, -0.2, 170),
                                                (-6, 4, 0.1, -95), (5, -7, 0.0, 12)])}


def _moved_as_one(layout, turn_deg, shift, lift):
    th = math.radians(turn_deg)
    out = {}
    for n, s in layout.items():
        x, y = s["x_m"], s["y_m"]
        out[n] = dict(s, x_m=math.cos(th) * x - math.sin(th) * y + shift[0],
                      y_m=math.sin(th) * x + math.cos(th) * y + shift[1],
                      z_m=s["z_m"] + lift, yaw_deg=s["yaw_deg"] + turn_deg)
    return out


def test_identical_layouts_have_zero_error():
    s = summarise(per_scan_error(_layout(), _layout()))
    assert s["n_scans"] == 5 and s["max_xy_m"] < 1e-9 and s["max_z_m"] < 1e-9 and s["max_yaw_deg"] < 1e-9


def test_a_layout_moved_as_one_piece_counts_as_no_error():
    gold = _layout()
    moved = _moved_as_one(gold, 33.0, (10.5, -4.0), 0.8)
    s = summarise(per_scan_error(moved, gold))
    assert s["max_xy_m"] < 1e-6 and s["max_z_m"] < 1e-6 and s["max_yaw_deg"] < 1e-6, s
    raw = summarise(per_scan_error(moved, gold, remove_layout_move=False))
    assert raw["max_xy_m"] > 5.0, "without removing the move, the same layout looks metres out"


def test_one_scan_out_of_place_is_the_one_blamed():
    gold = _layout()
    bad = {n: dict(s) for n, s in gold.items()}
    bad["s2"]["x_m"] += 1.0
    bad["s2"]["z_m"] += 0.4
    rows = {r["scan"]: r for r in per_scan_error(bad, gold)}
    assert summarise(list(rows.values()))["worst_xy_scan"] == "s2"
    # A plain least-squares fit spreads one bad scan's error onto the good ones.
    others = max(r["xy_m"] for n, r in rows.items() if n != "s2")
    assert abs(rows["s2"]["xy_m"] - 1.0) < 0.02 and others < 0.02, (rows["s2"], others)
    assert abs(rows["s2"]["z_m"] - 0.4) < 1e-9, rows["s2"]
    assert max(r["z_m"] for n, r in rows.items() if n != "s2") < 1e-9


def test_turn_error_wraps_across_180():
    gold = _layout()
    bad = {n: dict(s) for n, s in gold.items()}
    bad["s2"]["yaw_deg"] = -178.0   # golden is 170, so the real difference is 12 degrees
    rows = {r["scan"]: r for r in per_scan_error(bad, gold, remove_layout_move=False)}
    assert abs(rows["s2"]["yaw_deg"] - 12.0) < 1e-9, rows["s2"]


def test_scans_missing_from_either_side_are_skipped():
    gold = _layout()
    partial = {n: s for n, s in gold.items() if n != "s4"}
    assert [r["scan"] for r in per_scan_error(partial, gold)] == ["s0", "s1", "s2", "s3"]
    assert summarise([]) == {"n_scans": 0}


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
