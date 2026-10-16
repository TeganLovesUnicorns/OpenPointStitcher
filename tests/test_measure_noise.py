"""measure_noise: a scan's flat-wall scatter recovers a known noise, and the link weight follows it."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import measure_noise as mn  # noqa: E402


def room(noise_m: float, seed: int = 0) -> np.ndarray:
    """A 6 x 4 m floor and one 6 x 2.5 m wall, dense, with Gaussian scatter of `noise_m` across each surface."""
    rng = np.random.default_rng(seed)
    n = 200000
    floor = np.column_stack([rng.uniform(1, 7, n), rng.uniform(-2, 2, n), rng.normal(0, noise_m, n)])
    wall = np.column_stack([rng.uniform(1, 7, n), rng.normal(2.0, noise_m, n), rng.uniform(0.1, 2.5, n)])
    return np.vstack([floor, wall])


def test_scan_noise_recovers_a_known_scatter():
    for sigma in (0.005, 0.015):
        got = mn.scan_noise(f"room{sigma}", room(sigma), np.eye(3))
        assert got is not None and abs(got - sigma) / sigma < 0.15, (sigma, got)
    print("    5 mm and 15 mm of scatter on a flat floor and wall are measured back within 15%")


def test_clutter_is_not_a_flat_sheet():
    rng = np.random.default_rng(3)
    blob = rng.normal(0, 0.4, (50000, 3)) + [3, 0, 1]
    assert mn.scan_noise("blob", blob, np.eye(3)) is None
    print("    a scan with no flat sheet reports no noise rather than a made-up one")


def test_pair_weight_trusts_quiet_scans_more():
    assert mn.pair_weight(0.01, 0.01) == 1.0
    assert mn.pair_weight(0.005, 0.005) > 1.0 > mn.pair_weight(0.02, 0.02)
    assert mn.pair_weight(None, 0.01) == 1.0
    assert mn.pair_weight(0.0001, 0.0001) == mn.WEIGHT_RANGE[1]
    print("    a quiet pair counts more, a noisy pair less, unknown noise changes nothing, and the weight is capped")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
