"""Tests for find_bent_scans.py: one wall off by the same amount at every height while the scan's other walls agree is
a bent scan; every wall off together is a placement; off at some heights only is a curtain or recess and says
nothing."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_bent_scans import as_report, find_bent_scans  # noqa: E402
from open_point_stitch.placement import Placement  # noqa: E402

ROOM = {"south": ((0, 0), (6, 0)), "north": ((0, 4), (6, 4)), "west": ((0, 0), (0, 4)), "east": ((6, 0), (6, 4))}


def _scan(seed: int, stand, east_x=6.0, shift=(0.0, 0.0), east_low_only=False, unseen=(), floor=False):
    """World points and payload walls (in the scan's own frame) for a scan standing at `stand` in a 6 x 4 m room.
    `floor` adds the room's floor, which is what lets each wall's face be read from the points."""
    rng = np.random.default_rng(seed)
    pts, walls = [], []
    if floor:
        pts.append(np.column_stack([rng.uniform(0, 6, (40000, 2)) * [1, 4 / 6] + shift, rng.normal(0, 0.005, 40000)]))
    for i, (name, (a, b)) in enumerate(ROOM.items()):
        if name in unseen:
            continue
        a, b = np.array(a, float), np.array(b, float)
        if name == "east":
            a[0] = b[0] = east_x
        a, b = a + shift, b + shift
        t = rng.uniform(0, 1, 20000)
        z = rng.uniform(0.0, 3.0, 20000)
        xy = a + np.outer(t, b - a)
        if name == "east" and east_low_only:                 # only up to 2.2 m sits off; above it, the true wall
            xy[z > 2.2, 0] = 6.0 + shift[0]
        xy += rng.normal(0, 0.004, xy.shape)
        pts.append(np.column_stack([xy, z]))
        la, lb = a - stand, b - stand
        walls.append({"i": i, "x1": la[0], "y1": la[1], "x2": lb[0], "y2": lb[1]})
    return np.vstack(pts), {"walls": walls}


def _run(floor=False, **c_kwargs):
    world, placements, payloads = {}, {}, {}
    for k, (name, stand) in enumerate((("A", (2.0, 2.0)), ("B", (3.0, 2.0)), ("C", (4.0, 2.0)))):
        stand = np.array(stand)
        world[name], payloads[name] = _scan(k, stand, floor=floor, **(c_kwargs if name == "C" else {}))
        placements[name] = Placement(name=name, tx=float(stand[0]), ty=float(stand[1]))
    return {v.scan: v for v in find_bent_scans(world, placements, payloads)}


def _wall(rng, a, b, z_top=3.0, n=20000):
    t = rng.uniform(0, 1, n)
    xy = np.array(a, float) + np.outer(t, np.subtract(b, a)) + rng.normal(0, 0.004, (n, 2))
    return np.column_stack([xy, rng.uniform(0.0, z_top, n)])


def _two_rooms(thick=0.11):
    """Room A (x 0 to 6) and room B (x 6+thick to 10) share one wall. Scan "west" saw room A. Scan "east" STARTED in
    room A too (its origin is west of the wall), walked through and saw room B's side of that wall, plus room A's
    north and south walls on the way. Returns the run's verdicts by scan and the info dict."""
    rng = np.random.default_rng(7)
    xb = 6.0 + thick
    floor_a = np.column_stack([rng.uniform(0, 6, 40000), rng.uniform(0, 4, 40000), rng.normal(0, .005, 40000)])
    floor_b = np.column_stack([rng.uniform(xb, 10, 40000), rng.uniform(0, 4, 40000), rng.normal(0, .005, 40000)])
    lines = {"west": [((0, 0), (6, 0)), ((0, 4), (6, 4)), ((0, 0), (0, 4)), ((6, 0), (6, 4))],
             "east": [((0, 0), (6, 0)), ((0, 4), (6, 4)), ((xb, 0), (xb, 4)), ((10, 0), (10, 4))]}
    stands = {"west": np.array([2.0, 2.0]), "east": np.array([4.0, 2.0])}
    world, placements, payloads = {}, {}, {}
    for name, ls in lines.items():
        floors = [floor_a] if name == "west" else [floor_a[floor_a[:, 0] < 4.5], floor_b]   # east crossed A's middle only
        world[name] = np.vstack([_wall(rng, a, b) for a, b in ls] + floors)
        s = stands[name]
        payloads[name] = {"walls": [{"x1": a[0] - s[0], "y1": a[1] - s[1], "x2": b[0] - s[0], "y2": b[1] - s[1]}
                                    for a, b in ls]}
        placements[name] = Placement(name=name, tx=float(s[0]), ty=float(s[1]))
    info: dict = {}
    return {v.scan: v for v in find_bent_scans(world, placements, payloads, info=info)}, info


def test_one_wall_off_at_every_height_with_the_rest_agreeing_is_bent():
    v = _run(east_x=6.15)
    assert v["C"].kind == "bent", vars(v["C"])
    assert "15 cm off on its east wall" in v["C"].text, v["C"].text
    assert "drifted" in v["C"].text and "—" not in v["C"].text
    assert v["A"].kind == "" and v["B"].kind == "", (v["A"].text, v["B"].text)


def test_a_whole_scan_placed_wrong_is_not_called_bent():
    v = _run(shift=(0.15, 0.0))
    assert v["C"].kind == "placed off", vars(v["C"])
    assert "placed wrong" in v["C"].text


def test_with_no_agreeing_wall_running_the_same_way_it_says_it_cannot_tell_bent_from_shifted():
    v = _run(east_x=6.15, unseen=("west",))
    assert v["C"].kind == "bent or shifted", vars(v["C"])
    assert "15 cm out of place east-west" in v["C"].text and "needs moving" in v["C"].text, v["C"].text


def test_off_at_some_heights_only_says_nothing():
    v = _run(east_x=6.15, east_low_only=True)
    assert v["C"].kind == "", v["C"].text


def test_scans_that_agree_say_nothing_and_the_report_lists_only_flags():
    v = _run()
    assert all(x.kind == "" for x in v.values())
    rep = as_report(list(v.values()))
    assert rep == {"checked": 3, "flags": []}, rep
    rep = as_report(list(_run(east_x=6.15).values()))
    assert [f["scan"] for f in rep["flags"]] == ["C"] and abs(rep["flags"][0]["walls"][0]["off_cm"]) > 10, rep


def test_two_faces_of_one_wall_are_not_compared_when_the_floor_shows_which_face_each_saw():
    # Both scans started west of the wall, but their floors say otherwise.
    v, info = _two_rooms(thick=0.11)
    assert v["east"].kind == "" and v["west"].kind == "", (v["east"].text, v["west"].text)
    assert any(abs(t - 11.0) < 1.0 for t in info["wall_thickness_cm"]), info


def test_an_unreadable_face_one_wall_thick_off_says_maybe_the_other_side_not_bent():
    v = _run(east_x=6.11)                                     # no floor: faces fall back to where each scan started
    assert v["C"].kind == "other face", vars(v["C"])
    assert "other side of the same wall" in v["C"].text and "compared with A, B" in v["C"].text, v["C"].text
    assert "drifted" not in v["C"].text and "—" not in v["C"].text


def test_a_real_bend_one_wall_thick_stays_bent_when_the_floor_shows_the_faces():
    # 13 cm off, inside the thickness band, with faces readable: bent.
    v = _run(floor=True, east_x=6.13)
    assert v["C"].kind == "bent", vars(v["C"])
    assert "13 cm off on its east wall (compared with A, B)" in v["C"].text, v["C"].text


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
