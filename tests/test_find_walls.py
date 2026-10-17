"""find_walls.py: the plane search, curtains moved back to the real wall, and the walls the bench draws."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_walls import (  # noqa: E402
    _tightness_weights, drawn_walls, drop_doubles, extract_walls, find_true_wall, share_on, walls_from_above)


# --- the plane search: tightness-weighted wall position ------------------------------------------------------------


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


# --- curtains: moved back to where the ceiling ends; a plain wall and a half wall left alone -----------------------


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


# --- the drawn walls on a made-up room: a 60 cm return, a doorway, a double line -----------------------------------


FLOOR_Z = 1.3          # the scan's frame has its floor here, as a levelled scan does


def _sheet(rng, a, b, z0, z1, n):
    """Points on a vertical sheet over the plan line a-b, normals facing out of it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, z = rng.uniform(0, 1, n), rng.uniform(z0, z1, n)
    xy = a + t[:, None] * (b - a)
    d = (b - a) / np.linalg.norm(b - a)
    return np.column_stack([xy, z + FLOOR_Z]), np.tile([-d[1], d[0], 0.0], (n, 1))


def _room():
    rng = np.random.default_rng(3)
    parts = [_sheet(rng, (0, 0), (8, 0), 0, 2.6, 40000), _sheet(rng, (8, 0), (8, 5), 0, 2.6, 25000),
             _sheet(rng, (0, 5), (3, 5), 0, 2.6, 15000), _sheet(rng, (4, 5), (8, 5), 0, 2.6, 20000),   # 1 m doorway
             _sheet(rng, (0, 0), (0, 5), 0, 2.6, 25000),
             _sheet(rng, (5, 0), (5, 0.6), 0, 2.6, 4000)]                                         # a 60 cm return
    floor = np.column_stack([rng.uniform(0, 8, 60000), rng.uniform(0, 5, 60000), np.full(60000, FLOOR_Z)])
    pts = np.vstack([p for p, _ in parts] + [floor])
    nrm = np.vstack([n for _, n in parts] + [np.tile([0, 0, 1.0], (60000, 1))])
    return pts, nrm


def _cloud(pts, nrm):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.normals = o3d.utility.Vector3dVector(nrm)
    return pc


def _line(a, b):
    return {"x1": a[0], "y1": a[1], "x2": b[0], "y2": b[1]}


TRUTH = [_line((0, 0), (8, 0)), _line((8, 0), (8, 5)), _line((0, 5), (3, 5)), _line((4, 5), (8, 5)),
         _line((0, 0), (0, 5)), _line((5, 0), (5, 0.6))]


def test_from_above_finds_every_wall_and_the_short_return_and_repeats_exactly():
    pts, nrm = _room()
    lifted = pts.copy()
    lifted[:, 2] -= FLOOR_Z
    walls = walls_from_above(lifted, nrm)
    for t in TRUTH:
        assert share_on(t, walls) > 0.85, (t, walls)
    assert all(share_on(w, TRUTH) > 0.9 for w in walls), walls
    assert walls == walls_from_above(lifted, nrm), "two runs on the same points differ"


def test_from_above_gives_the_same_walls_when_the_scan_is_moved_or_turned():
    # On a grid fixed to the origin, a 2.5 cm shift changed up to a quarter of a scan's walls.
    pts, nrm = _room()
    pts[:, 2] -= FLOOR_Z
    walls = walls_from_above(pts, nrm)
    for deg, dx, dy in [(0, 0.025, 0.0), (23, 1.013, -0.027), (90, -3.0, 2.0)]:
        a = math.radians(deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p, n = pts.copy(), nrm.copy()
        p[:, :2], n[:, :2] = pts[:, :2] @ rot.T + [dx, dy], nrm[:, :2] @ rot.T
        back = []
        for w in walls_from_above(p, n):
            e1, e2 = (np.array([w["x1"], w["y1"]]) - [dx, dy]) @ rot, (np.array([w["x2"], w["y2"]]) - [dx, dy]) @ rot
            back.append(np.r_[e1, e2])
        assert len(back) == len(walls), (deg, walls, back)
        for w in walls:
            e = np.array([w["x1"], w["y1"], w["x2"], w["y2"]])
            assert min(min(np.abs(b - e).max(), np.abs(b[[2, 3, 0, 1]] - e).max()) for b in back) < 0.01, (deg, w, back)


def test_drawn_walls_keep_the_doorway_as_a_gap_and_find_the_return():
    pts, nrm = _room()
    walls = drawn_walls(_cloud(pts, nrm), FLOOR_Z, max_planes=4)          # 4 planes: the return is left to the other
    for t in TRUTH:
        assert share_on(t, walls) > 0.85, (t, walls)
    doorway = _line((3.2, 5), (3.8, 5))
    assert share_on(doorway, walls) == 0.0, walls
    assert [w["i"] for w in walls] == list(range(len(walls))), walls
    for w in walls:
        assert w["kind"] == "full" and abs(w["h"] - 2.6) < 0.15 and w["pts"] > 0, w
        assert abs(w["len"] - math.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"])) < 0.01, w


def _poke_through():
    """Drawn walls of a long wall only the plane search has (its points face up; given as `rows`) and a return
    that clutter past the long wall stretches through it."""
    rng = np.random.default_rng(4)
    wall, ret = _sheet(rng, (0, 0), (8, 0), 0, 2.6, 40000), _sheet(rng, (5.025, 0), (5.025, 2.5), 0, 2.6, 20000)
    clutter = _sheet(rng, (5.025, -0.5), (5.025, 0), 0, 2.0, 5000)                 # mid-square: no tilt
    pts = np.vstack([wall[0], ret[0], clutter[0]])
    nrm = np.vstack([np.tile([0, 0, 1.0], (40000, 1)), ret[1], np.tile([0, 0, 1.0], (5000, 1))])
    row = {"i": 0, **_line((0, 0), (8, 0)), "len": 8.0, "h": 2.6, "pts": 40000, "behind_m": 0.0}
    return drawn_walls(_cloud(pts, nrm), FLOOR_Z, max_planes=4, rows=[row])


FINGERPRINT = (11, "4e233670e862")   # (WALL_FINDER_SINCE it was taken on, the made-up rooms' walls)


def test_wall_finder_fingerprint():
    # Changing the walls without raising WALL_FINDER_SINCE would put hand wall edits on the wrong walls.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import build_bench
    pts, nrm = _room()
    walls = drawn_walls(_cloud(pts, nrm), FLOOR_Z, max_planes=4) + _poke_through()
    got = hashlib.sha1(json.dumps([[round(w[k] / 0.02) for k in ("x1", "y1", "x2", "y2")] for w in walls])
                       .encode()).hexdigest()[:12]
    assert got == FINGERPRINT[1], (f"the walls found changed: raise BUILDER_VERSION and WALL_FINDER_SINCE (now "
                                   f"{build_bench.WALL_FINDER_SINCE}), then set FINGERPRINT to (that, {got!r})")
    assert FINGERPRINT[0] <= build_bench.WALL_FINDER_SINCE, FINGERPRINT


def test_a_from_above_wall_poking_through_a_plane_search_wall_is_cut_back_to_it():
    # Code review 2026-10-16: crossings were only checked inside each finder's own walls.
    walls = _poke_through()
    across = [w for w in walls if abs(w["x1"] - 5) < 0.1 and abs(w["x2"] - 5) < 0.1]
    assert len(across) == 1, walls
    assert min(across[0]["y1"], across[0]["y2"]) > -0.05 and max(across[0]["y1"], across[0]["y2"]) > 2.4, across


def test_a_second_line_just_in_front_of_a_wall_is_dropped_and_the_better_supported_stays():
    xy = np.column_stack([np.linspace(0, 6, 3000), np.zeros(3000)])
    wall, ghost = _line((0, 0), (6, 0)), _line((0.5, 0.2), (5, 0.2))
    assert drop_doubles([ghost, wall], xy) == [wall]
    apart = _line((0, 0.6), (6, 0.6))                                    # 60 cm away: a different wall
    assert len(drop_doubles([wall, apart], xy)) == 2
    behind = {**_line((0, 0.2), (6, 0.2)), "behind_m": 0.2}             # behind a curtain: never dropped
    assert behind in drop_doubles([wall, behind], xy)


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
