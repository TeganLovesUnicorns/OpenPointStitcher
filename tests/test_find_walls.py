"""find_walls.py: tightness-weighted wall position. Run with: python tests/test_find_walls.py"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_walls import _tightness_weights, extract_walls  # noqa: E402


def _make_wall(n2, tangent_lo, tangent_hi, height_lo, height_hi, scatter, seed):
    """A synthetic vertical wall's points, at normal offset 0, along `n2`, with `scatter` metres of
    Gaussian noise along the normal. `n2` must be unit length."""
    rng = np.random.default_rng(seed)
    along = np.array([-n2[1], n2[0]])
    tangent = rng.uniform(tangent_lo, tangent_hi, size=4000)
    height = rng.uniform(height_lo, height_hi, size=4000)
    normal_pos = rng.normal(0.0, scatter, size=4000)
    xy = tangent[:, None] * along[None, :] + normal_pos[:, None] * n2[None, :]
    return np.column_stack([xy, height])


def test_tightness_weights_favour_a_tighter_patch_over_a_scattered_one():
    """Two patches of the same synthetic wall, one tight (2mm scatter), one loose (8cm scatter -
    a real occluded/glancing-angle stretch). The loose patch must get lower weight on average."""
    n2 = np.array([1.0, 0.0])
    tight = _make_wall(n2, 0.0, 2.0, 0.0, 2.5, scatter=0.002, seed=1)
    loose = _make_wall(n2, 3.0, 5.0, 0.0, 2.5, scatter=0.08, seed=2)
    pts = np.vstack([tight, loose])
    w = _tightness_weights(pts, n2)
    w_tight, w_loose = w[:len(tight)], w[len(tight):]
    assert w_tight.mean() > 5 * w_loose.mean(), (
        f"the tight patch should weigh several times more than the loose one: "
        f"tight mean {w_tight.mean():.1f}, loose mean {w_loose.mean():.1f}")
    print(f"    tight patch mean weight {w_tight.mean():.1f}, loose patch mean weight {w_loose.mean():.1f}")


def test_weighted_offset_pulls_toward_the_tighter_patch_when_they_disagree():
    """Two patches at different normal offsets (tight one on 0, loose one 0.05 m off) and different tangent ranges,
    so each bin sees one patch's scatter. A plain average splits the difference by patch size; the weighted one lands
    closer to the tight patch."""
    n2 = np.array([0.0, 1.0])
    tight = _make_wall(n2, 0.0, 2.0, 0.0, 2.5, scatter=0.002, seed=3)
    loose = _make_wall(n2, 3.0, 5.0, 0.0, 2.5, scatter=0.08, seed=4)
    loose[:, :2] += 0.05 * n2  # the loose patch's own points sit 5cm off from the tight patch's plane
    pts = np.vstack([tight, loose])

    plain = float(np.mean(pts[:, :2] @ n2))
    w = _tightness_weights(pts, n2)
    weighted = float(np.average(pts[:, :2] @ n2, weights=w))
    assert abs(plain - 0.025) < 0.003, f"test setup check: plain average should be near 0.025m, was {plain:.4f}m"
    assert abs(weighted) < 0.2 * abs(plain), (
        f"the weighted offset ({weighted:.4f}) should sit within a fifth of the tight patch's true "
        f"0.0, not just somewhat closer than the plain average ({plain:.4f})")
    print(f"    plain average {plain:.4f}m, tightness-weighted {weighted:.4f}m (true position 0.0000m)")


def test_extract_walls_off_by_default_matches_the_old_offset_exactly():
    """tightness_weighted defaults to False and gives exactly the unweighted answer."""
    n2 = np.array([1.0, 0.0])
    pts = _make_wall(n2, 0.0, 4.0, 0.0, 2.5, scatter=0.01, seed=5)
    pcd_points = np.zeros((len(pts) + 1, 3))
    pcd_points[:-1] = pts
    pcd_points[-1] = [0.5, 10.0, 1.0]  # a floor point (near-horizontal normal), filtered out below
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points))
    pcd.normals = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (len(pcd_points), 1)))  # pre-set so extract_walls skips its own estimate

    walls_off = extract_walls(pcd, min_support=100, min_length=1.0, min_height=1.0)
    walls_default = extract_walls(pcd, min_support=100, min_length=1.0, min_height=1.0, tightness_weighted=False)
    assert len(walls_off) == 1 and len(walls_default) == 1, f"expected one wall each: {walls_off}, {walls_default}"
    assert walls_off[0].offset == walls_default[0].offset, "tightness_weighted=False must be a true no-op"
    print(f"    offset unchanged at the explicit default: {walls_off[0].offset:.6f}m")


def test_merging_one_wall_found_twice_keeps_the_stronger_plane_exactly():
    # Averaging offsets across two normals would put the plane on neither detection.
    from open_point_stitch.find_walls import Wall, merge_duplicate_walls
    p = np.array([30.0, 0.5])                                         # one real wall, 30 m from the origin
    a, b = np.radians(0.0), np.radians(2.0)
    n1, n2 = np.array([np.cos(a), np.sin(a)]), np.array([np.cos(b), np.sin(b)])
    strong = Wall(normal=n1, offset=float(n1 @ p), support=900, length=4, height=2.5, centroid=np.array([*p, 1.2]))
    weak = Wall(normal=n2, offset=float(n2 @ p), support=300, length=3, height=2.5, centroid=np.array([*p, 1.2]))
    fitted = strong.offset                                            # the merge updates `strong` in place
    merged = merge_duplicate_walls([weak, strong])
    assert len(merged) == 1, merged
    assert merged[0].offset == fitted and merged[0].support == 1200, (merged[0].offset, fitted)


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
