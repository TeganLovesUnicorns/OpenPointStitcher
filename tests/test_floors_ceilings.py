"""find_floors_ceilings.py: floor and ceiling pieces on a made-up room, and the height settle on them."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_floors_ceilings import find_floors_and_ceilings, settle, solve_heights  # noqa: E402
from open_point_stitch.lib.poses import ScanPose  # noqa: E402


# --- floor and ceiling pieces --------------------------------------------------------------------------------------


DENSITY = 2500          # points per square metre on each surface
NOISE_M = 0.005


def _patch(rng, x0, x1, y0, y1, z_of_x):
    n = int((x1 - x0) * (y1 - y0) * DENSITY)
    x = rng.uniform(x0, x1, n)
    y = rng.uniform(y0, y1, n)
    return np.column_stack([x, y, z_of_x(x) + rng.normal(0, NOISE_M, n)])


def _room(seed=1):
    """8 x 6 m. Floor 0.0 for x < 5, 0.6 for x >= 5. Counter 0.9 m at x 1-2, y 1-2 (floor hidden under it).
    Ceiling 3.0 for x < 3, a duct at 2.7 for x 3-4, a slope from 3.0 at x 4 to 4.0 at x 6, then flat 4.0."""
    rng = np.random.default_rng(seed)
    parts = [
        _patch(rng, 0, 5, 2, 6, lambda x: np.zeros_like(x)), _patch(rng, 2, 5, 0, 2, lambda x: np.zeros_like(x)),
        _patch(rng, 0, 1, 0, 2, lambda x: np.zeros_like(x)),
        _patch(rng, 1, 2, 1, 2, lambda x: np.full_like(x, 0.9)), _patch(rng, 1, 2, 0, 1, lambda x: np.zeros_like(x)),
        _patch(rng, 5, 8, 0, 6, lambda x: np.full_like(x, 0.6)),
        _patch(rng, 0, 3, 0, 6, lambda x: np.full_like(x, 3.0)),
        _patch(rng, 3, 4, 0, 6, lambda x: np.full_like(x, 2.7)),
        _patch(rng, 4, 6, 0, 6, lambda x: 3.0 + (x - 4) * 0.5),
        _patch(rng, 6, 8, 0, 6, lambda x: np.full_like(x, 4.0)),
    ]
    return np.vstack(parts)


def _one(pieces, kind, h, tol=0.02):
    hits = [p for p in pieces if p.kind == kind and not p.sloped and abs(p.height_m - h) < tol]
    assert len(hits) == 1, f"expected one {kind} at {h} m, got {[(p.kind, p.label) for p in pieces]}"
    return hits[0]


def test_two_floors_a_counter_and_three_kinds_of_ceiling_are_found():
    pieces = find_floors_and_ceilings(_room())
    low = _one(pieces, "floor", 0.0)
    high = _one(pieces, "floor", 0.6)
    # 29 and 18 m2 drawn; a piece's edge cells count whole, so a little over is expected
    assert 27 < low.area_m2 < 31.5 and 17 < high.area_m2 < 19.5, (low.area_m2, high.area_m2)
    assert "raised floor" in high.label, high.label
    counter = _one(pieces, "surface", 0.9)
    assert "flat surface" in counter.label, counter.label
    flat = _one(pieces, "ceiling", 3.0)
    duct = _one(pieces, "ceiling", 2.7)
    assert "lowered ceiling" in duct.label and "30 cm below" in duct.label, duct.label
    upper = _one(pieces, "ceiling", 4.0)
    assert "raised ceiling" in upper.label, upper.label
    slopes = [p for p in pieces if p.kind == "ceiling" and p.sloped]
    assert len(slopes) == 1 and abs(slopes[0].slope_deg - 26.57) < 1.0, [p.label for p in slopes]
    assert 2.98 <= slopes[0].z_min < 3.2 and 3.8 < slopes[0].z_max <= 4.02, slopes[0].label
    for p in (low, high, flat, duct, upper, slopes[0]):
        assert p.rms_m < 0.01, (p.label, p.rms_m)
    print(f"    {len(pieces)} pieces: " + "; ".join(p.label for p in pieces))


def test_a_bulkhead_whose_underside_was_barely_scanned_is_still_found():
    # A 33 cm drop along one wall, 0.5 m deep, its underside a few points per square foot. The ordinary pass needs
    # denser points; the coarse pass finds it and says it is from few points.
    rng = np.random.default_rng(4)
    dense = lambda x0, x1, z: _patch(rng, x0, x1, 0, 4, lambda x: np.full_like(x, z))
    face = np.column_stack([np.full(6000, 5.5), rng.uniform(0, 4, 6000), rng.uniform(2.07, 2.4, 6000)])
    under = np.column_stack([rng.uniform(5.5, 6.0, 300), rng.uniform(0, 4, 300), 2.07 + rng.normal(0, 0.005, 300)])
    pts = np.vstack([dense(0, 6, 0.0), dense(0, 5.5, 2.4), face, under])
    pieces = find_floors_and_ceilings(pts)
    drops = [p for p in pieces if p.kind == "ceiling" and abs(p.height_m - 2.07) < 0.03]
    assert len(drops) == 1 and drops[0].sparse and "33 cm below" in drops[0].label, [p.label for p in pieces]
    xs = np.concatenate([o[:, 0] for o in drops[0].outlines])
    assert xs.min() > 5.2 and xs.max() < 6.3, (xs.min(), xs.max())


def test_the_same_scan_gives_the_same_pieces_every_time():
    pts = _room(seed=2)
    a = json.dumps([p.to_json() for p in find_floors_and_ceilings(pts)])
    b = json.dumps([p.to_json() for p in find_floors_and_ceilings(pts.copy())])
    assert a == b


def test_a_scan_with_no_flat_surfaces_gives_no_pieces():
    rng = np.random.default_rng(3)
    wall = np.column_stack([np.zeros(20_000), rng.uniform(0, 5, 20_000), rng.uniform(0, 3, 20_000)])
    assert find_floors_and_ceilings(wall) == []


# --- the height settle ---------------------------------------------------------------------------------------------


def _settle_room(rng, x0, n=6000):
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
    payloads = {"A": _payload(_settle_room(rng, 0.0), true["A"]), "B": _payload(_settle_room(rng, 3.0), true["B"]),
                "C": _payload(_settle_room(rng, 6.0), true["C"])}
    placed = {"A": ScanPose(), "B": ScanPose(z_m=0.10), "C": ScanPose(z_m=-0.05)}
    dz, gaps = settle(payloads, placed, fixed={"A"})
    assert "A" not in dz and len(gaps) >= 2, (dz, gaps)
    assert abs(dz["B"] + 0.10) < 0.01 and abs(dz["C"] - 0.05) < 0.01, dz


def test_scans_outside_the_run_are_not_moved():
    rng = np.random.default_rng(2)
    payloads = {n: _payload(_settle_room(rng, x), ScanPose()) for n, x in (("A", 0.0), ("B", 3.0), ("C", 6.0))}
    placed = {"A": ScanPose(), "B": ScanPose(z_m=0.10), "C": ScanPose(z_m=-0.05)}
    dz, _ = settle(payloads, placed, fixed={"A"}, only={"B"})
    assert set(dz) == {"B"}, dz


def test_one_pair_that_disagrees_with_the_rest_counts_for_less():
    """Four scans in a ring agree; one pair's gap reads 40 cm off (raked seating seen sideways)."""
    gaps = {("A", "B"): (1000, 0.0), ("B", "C"): (1000, 0.0), ("C", "D"): (1000, 0.0),
            ("A", "D"): (1000, 0.0), ("A", "C"): (1000, 0.40), ("B", "D"): (1000, 0.0)}
    dz = solve_heights(gaps, ["A", "B", "C", "D"], fixed={"A"})
    assert max(abs(v) for v in dz.values()) < 0.05, dz


def test_no_shared_floor_means_no_change():
    dz = solve_heights({}, ["A", "B"], fixed={"A"})
    assert dz == {"B": 0.0}, dz


def test_a_scan_with_a_marked_floor_settles_on_that_floor_only():
    # A and B share only a table top at 0.8 m; B's floor is out of A's sight.
    rng = np.random.default_rng(3)
    table = lambda: np.column_stack([rng.uniform(3, 6, 4000), rng.uniform(0, 6, 4000), np.full(4000, 0.8)])
    a_pts = np.vstack([_settle_room(rng, 0.0), table()])
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
    a_pts = np.vstack([_settle_room(rng, 0.0), raised()])
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
    from support import run_tests
    raise SystemExit(run_tests(globals()))
