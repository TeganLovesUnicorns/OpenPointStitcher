"""The bench's walls (find_drawn_walls) and the from-above finder (find_walls_above) on a made-up room: four walls,
a 60 cm return the plane search misses, a doorway, and a double line in front of a wall."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_drawn_walls import drawn_walls, drop_doubles, share_on  # noqa: E402
from open_point_stitch.find_walls_above import walls_from_above  # noqa: E402

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
