"""Tests for the gravity-constrained two-point manual alignment maths.

No display needed: the picking GUI is separate from the solve.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.lib.poses import solve_yaw_translation, tilt_of, yaw_transform  # noqa: E402


def _random_yaw_transform(rng):
    yaw = rng.uniform(-180, 180)
    t = rng.uniform(-6, 6, 3)
    return yaw, t, yaw_transform(yaw, t)


def test_two_points_recover_exact_transform():
    rng = np.random.default_rng(0)
    for _ in range(25):
        yaw, t, T = _random_yaw_transform(rng)
        src = rng.uniform(-5, 5, (2, 3))
        tgt = (T[:3, :3] @ src.T).T + t
        got_yaw, got_t = solve_yaw_translation(src, tgt)
        dy = abs((got_yaw - yaw + 180) % 360 - 180)
        assert dy < 1e-6, dy
        assert np.allclose(got_t, t, atol=1e-9)


def test_more_points_are_least_squares():
    rng = np.random.default_rng(1)
    yaw, t, T = _random_yaw_transform(rng)
    src = rng.uniform(-5, 5, (6, 3))
    tgt = (T[:3, :3] @ src.T).T + t + rng.normal(0, 0.02, (6, 3))
    got_yaw, got_t = solve_yaw_translation(src, tgt)
    assert abs((got_yaw - yaw + 180) % 360 - 180) < 2.0
    assert np.linalg.norm(got_t - t) < 0.05


def test_result_is_never_tilted():
    rng = np.random.default_rng(2)
    for _ in range(10):
        src = rng.uniform(-5, 5, (3, 3))
        tgt = rng.uniform(-5, 5, (3, 3))   # deliberately nonsense correspondences
        yaw, t = solve_yaw_translation(src, tgt)
        assert tilt_of(yaw_transform(yaw, t)) < 1e-9


def test_coincident_points_rejected():
    p = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    try:
        solve_yaw_translation(p, p.copy())
    except ValueError as exc:
        assert "far apart" in str(exc)
    else:
        raise AssertionError("expected a ValueError for coincident points")


def test_too_few_points_rejected():
    try:
        solve_yaw_translation(np.zeros((1, 3)), np.zeros((1, 3)))
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a single point")


def test_uniform_weights_reproduce_unweighted_result():
    """Equal weights change nothing."""
    rng = np.random.default_rng(3)
    yaw, t, T = _random_yaw_transform(rng)
    src = rng.uniform(-5, 5, (7, 3))
    tgt = (T[:3, :3] @ src.T).T + t + rng.normal(0, 0.02, (7, 3))
    plain = solve_yaw_translation(src, tgt)
    weighted = solve_yaw_translation(src, tgt, weights=np.full(7, 3.7))
    assert abs(plain[0] - weighted[0]) < 1e-9
    assert np.allclose(plain[1], weighted[1], atol=1e-9)


def test_high_weight_on_the_true_correspondences_beats_outliers():
    """A tight wall (high weight) wins over scattered noise (low weight) that outnumbers it."""
    rng = np.random.default_rng(4)
    yaw, t, T = _random_yaw_transform(rng)
    good_src = rng.uniform(-5, 5, (4, 3))
    good_tgt = (T[:3, :3] @ good_src.T).T + t  # exact, noiseless correspondences

    # Ten times as many "outlier" correspondences pointing at a wrong answer.
    bad_src = rng.uniform(-5, 5, (40, 3))
    bad_tgt = rng.uniform(-5, 5, (40, 3))  # unrelated to any consistent transform

    src = np.vstack([good_src, bad_src])
    tgt = np.vstack([good_tgt, bad_tgt])
    weights = np.concatenate([np.full(4, 50.0), np.full(40, 0.01)])

    got_yaw, got_t = solve_yaw_translation(src, tgt, weights=weights)
    assert abs((got_yaw - yaw + 180) % 360 - 180) < 1.0
    assert np.linalg.norm(got_t - t) < 0.1


def test_negative_weight_rejected():
    p = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    try:
        solve_yaw_translation(p, p.copy(), weights=np.array([1.0, -1.0]))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a negative weight")


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
