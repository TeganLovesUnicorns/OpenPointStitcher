"""Tests for report.transform_error_against, independent of pair direction."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.report import transform_error_against  # noqa: E402


def _yaw_transform(deg, t):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    m[:3, 3] = t
    return m


def test_same_direction_matches_exactly():
    T = _yaw_transform(12.0, [1.0, -0.5, 0.0])
    t_err, r_err = transform_error_against(T, "a", "b", T, "a", "b")
    assert t_err < 1e-9 and r_err < 1e-9


def test_opposite_direction_is_resolved_correctly_not_flagged_as_error():
    """The exact bug this function fixes: a transform reported as b->a
    (source_name='b', target_name='a') must be recognised as the SAME
    real-world alignment as a confirmed a->b transform, not a huge error."""
    T_ab = _yaw_transform(12.0, [1.0, -0.5, 0.0])  # a -> b
    T_ba = np.linalg.inv(T_ab)                      # b -> a, same real alignment

    t_err, r_err = transform_error_against(T_ba, "b", "a", T_ab, "a", "b")
    assert t_err < 1e-9, f"expected ~0 error once direction is resolved, got {t_err*100:.2f}cm"
    assert r_err < 1e-9, f"expected ~0 error once direction is resolved, got {r_err:.2f}deg"


def test_a_real_directional_mismatch_still_reads_as_a_real_error():
    """Direction-correction must not hide an ACTUAL wrong answer - only
    resolve which way the reported transform points, nothing more."""
    T_ab = _yaw_transform(12.0, [1.0, -0.5, 0.0])
    wrong = _yaw_transform(90.0, [5.0, 5.0, 0.0])
    t_err, r_err = transform_error_against(wrong, "a", "b", T_ab, "a", "b")
    assert t_err > 1.0 or r_err > 1.0, "a genuinely wrong answer must not read as ~0 error"


def test_mismatched_pair_names_raise_rather_than_silently_compare():
    T = _yaw_transform(0.0, [0.0, 0.0, 0.0])
    try:
        transform_error_against(T, "a", "b", T, "a", "c")
        assert False, "expected a ValueError for comparing unrelated pairs"
    except ValueError:
        pass


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
