"""Floor and ceiling pieces in one scan: two floor heights, a counter, and a flat, a dropped and a sloped ceiling."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_floors_ceilings import find_floors_and_ceilings  # noqa: E402

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


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
