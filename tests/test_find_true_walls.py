"""find_true_walls: a curtain hung in front of a wall is moved back to where the ceiling ends; a plain wall and a
half-height wall the ceiling carries on past are left alone."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_true_walls import find_true_wall  # noqa: E402

CEIL = 2.4
RNG = np.random.default_rng(7)


def sheet(n, x=None, y=None, z=None):
    """n points on an axis-aligned sheet; give the fixed coordinate a number and the others a (lo, hi) range."""
    cols = []
    for v in (x, y, z):
        cols.append(np.full(n, v) + RNG.normal(0, 0.004, n) if np.isscalar(v) else RNG.uniform(*v, n))
    return np.column_stack(cols)


def room(curtain=False, half_height=False):
    """A room from x = -3 to +3, the wall under test at x = -3. Returns (points, |normal z|, wall inliers)."""
    ceiling = sheet(30000, x=(-3.0 if not half_height else -4.5, 3.0), y=(-2, 2), z=CEIL)
    floor = sheet(30000, x=(-3.0, 3.0), y=(-2, 2), z=0.0)
    back = sheet(8000, x=3.0, y=(-2, 2), z=(0, CEIL))
    if half_height:
        wall = sheet(8000, x=-3.0, y=(-2, 2), z=(0, 1.2))            # a partition: the ceiling carries on past it
        parts, nz = [ceiling, floor, back, wall], [1, 1, 0, 0]
        inliers = wall
    elif curtain:
        wall = sheet(8000, x=-3.10, y=(-2, 2), z=(0, CEIL))            # the real wall, 10 cm behind
        curt = sheet(9000, x=-3.0, y=(-2, 2), z=(0.1, CEIL - 0.06))   # curtain stops 6 cm short of the ceiling
        ceiling = sheet(30000, x=(-3.10, 3.0), y=(-2, 2), z=CEIL)
        parts, nz = [ceiling, floor, back, wall, curt], [1, 1, 0, 0, 0]
        inliers = curt
    else:
        wall = sheet(8000, x=-3.0, y=(-2, 2), z=(0, CEIL))
        parts, nz = [ceiling, floor, back, wall], [1, 1, 0, 0]
        inliers = wall
    pts = np.vstack(parts)
    nzs = np.concatenate([np.full(len(p), v, float) for p, v in zip(parts, nz)])
    return pts, nzs, inliers


N = np.array([1.0, 0.0])          # the wall under test faces +x (into the room); plane x = -3 is n.xy = -3


def test_a_curtain_is_moved_back_to_the_real_wall():
    pts, nz, inl = room(curtain=True)
    tw = find_true_wall(pts, nz, N, -3.0, inl)
    assert tw is not None, "the curtain was taken for the wall"
    assert abs(tw.behind_m - 0.10) < 0.03, tw
    assert abs(tw.offset * np.sign(tw.normal[0]) + 3.10) < 0.03, tw
    print(f"    curtain 10 cm in front: wall found {tw.behind_m * 100:.0f} cm behind it, on {tw.sections} sections")


def test_a_plain_wall_is_left_alone():
    pts, nz, inl = room()
    assert find_true_wall(pts, nz, N, -3.0, inl) is None
    print("    a plain wall that meets the ceiling does not move")


def test_wall_detection_itself_puts_a_curtained_wall_behind_the_curtain():
    # Through the detector, not only the helper: switching the step off in extract_walls must fail here.
    import open3d as o3d
    from open_point_stitch.find_walls import extract_walls
    pts, _, _ = room(curtain=True)
    walls = extract_walls(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    near = [w for w in walls if abs(abs(w.normal[0]) - 1) < 0.05 and abs(abs(w.offset) - 3.05) < 0.1]
    assert near and max(w.behind_curtain_m for w in near) > 0.05, [(str(w), w.behind_curtain_m) for w in walls]


def test_a_half_height_wall_is_left_alone():
    pts, nz, inl = room(half_height=True)
    assert find_true_wall(pts, nz, N, -3.0, inl) is None
    print("    a half-height wall with the ceiling carrying on past it does not move")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
