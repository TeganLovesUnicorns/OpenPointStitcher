"""wall_checks.py: wall thickness, bent scans, and walls corrected by hand."""
from __future__ import annotations

import os
import sys
from concurrent.futures import CancelledError

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.placement import Placement  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402
from open_point_stitch.wall_checks import (  # noqa: E402
    _segments, apply_wall_edits, as_report, cluster_walls, edits_by_scan, find_bent_scans, find_wall, frozen,
    measure_walls, summarise, with_wall_edits)


# --- wall thickness: scans that agree read as a sheet, scans that do not as a band ---------------------------------


INCH = 0.0254


def _wall_scan(name: str, wall_y_local: float, seed: int, along=(0.0, 10.0)) -> Scan:
    """A wall along x at local y = wall_y_local, 0.5 cm noise, and a floor, seen from the origin."""
    rng = np.random.default_rng(seed)
    n = 30000
    wall = np.column_stack([rng.uniform(*along, n), wall_y_local + rng.normal(0, 0.005, n), rng.uniform(0.0, 3.0, n)])
    floor = np.column_stack([rng.uniform(*along, 3000), rng.uniform(-3, 3, 3000), np.zeros(3000)])
    return Scan(name=name, path=f"{name}.pts", points=np.vstack([wall, floor]))


def _payload(wall_y_local: float, along=(0.0, 10.0)) -> dict:
    return {"walls": [{"i": 0, "x1": along[0], "y1": wall_y_local, "x2": along[1], "y2": wall_y_local, "len": along[1] - along[0], "h": 3.0, "pts": 30000}]}


def _scene(offsets_and_sides):
    """Scans each standing 3 m from the wall (on the side given) whose face lands `offset` metres from y = 0."""
    scans, placements, payloads = [], {}, {}
    for k, (name, offset, side) in enumerate(offsets_and_sides):
        # A scan on the -y side stands at y = -3 and sees the wall 3 m ahead; on the +y side, at y = +3 and 3 m behind.
        stand = -3.0 * side
        wall_local = (offset - stand)
        scans.append(_wall_scan(name, wall_local, k))
        placements[name] = Placement(name=name, ty=stand)
        payloads[name] = _payload(wall_local)
    return scans, placements, payloads


def test_should_cancel_stops_between_scans_and_raises_cancelled_error():
    # Checked once per scan (placing every point at full resolution, before clustering) and once per wall cluster.
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1)])
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 1  # let scan 0 through, stop before scan 1

    try:
        measure_walls(scans, placements, payloads, should_cancel=should_cancel)
        raise AssertionError("expected CancelledError, measure_walls returned normally")
    except CancelledError as exc:
        assert "1/2" in str(exc), f"should say how far it got: {exc}"


def test_scans_that_agree_read_as_a_thin_sheet():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1)])
    reports = measure_walls(scans, placements, payloads)
    assert len(reports) == 1 and reports[0].thickness_in < 1.0, [(r.thickness_in, r.length_m) for r in reports]
    assert all(abs(s.offset_in) < 0.3 for s in reports[0].scans)


def test_scans_that_disagree_read_as_a_band_and_the_report_says_by_how_much_each():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.08, -1)])
    r = measure_walls(scans, placements, payloads)[0]
    assert 3.0 < r.thickness_in < 4.0, r.thickness_in                   # 8 cm apart, in inches
    offsets = {s.scan: s.offset_in for s in r.scans}
    assert abs(offsets["A"] + 0.04 / INCH) < 0.3 and abs(offsets["B"] - 0.04 / INCH) < 0.3, offsets
    assert r.body_in is None and abs(r.length_m - 9.6) < 0.2


def test_scans_on_opposite_sides_report_the_wall_body_not_a_thick_sheet():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1), ("C", 0.15, 1)])
    r = measure_walls(scans, placements, payloads)[0]
    assert r.thickness_in < 1.0, f"the two scans on the same side agree: {r.thickness_in}"
    assert r.body_in is not None and abs(r.body_in - 0.15 / INCH) < 1.0, r.body_in


def test_a_wall_across_the_first_is_a_different_wall():
    a = {"scan": "A", "a": np.array([0.0, 0.0]), "b": np.array([10.0, 0.0]), "u": np.array([1.0, 0.0]), "n": np.array([0.0, 1.0]), "length": 10.0}
    b = {"scan": "B", "a": np.array([5.0, -2.0]), "b": np.array([5.0, 2.0]), "u": np.array([0.0, 1.0]), "n": np.array([-1.0, 0.0]), "length": 4.0}
    c = {"scan": "C", "a": np.array([0.0, 0.1]), "b": np.array([10.0, 0.1]), "u": np.array([1.0, 0.0]), "n": np.array([0.0, 1.0]), "length": 10.0}
    groups = sorted(len(g) for g in cluster_walls([a, b, c]))
    assert groups == [1, 2], groups


def test_a_scan_with_too_few_points_on_the_wall_is_left_out():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.0, -1)])
    scans[1] = Scan(name="B", path="B.pts", points=scans[1].points[:50])
    r = measure_walls(scans, placements, payloads)[0]
    assert [s.scan for s in r.scans] == ["A"]


def test_no_detected_walls_gives_an_empty_report_and_summary():
    scans, placements, _ = _scene([("A", 0.0, -1)])
    reports = measure_walls(scans, placements, {"A": {"walls": []}})
    assert reports == [] and summarise(reports) == {"walls": 0}


def test_the_summary_counts_walls_too_thick_to_draw_from():
    scans, placements, payloads = _scene([("A", 0.0, -1), ("B", 0.08, -1)])
    s = summarise(measure_walls(scans, placements, payloads))
    assert s["walls"] == 1 and s["over_2_in"] == 1 and s["over_6_in"] == 0, s


def test_segments_are_placed_in_the_world_frame():
    payloads = {"A": _payload(3.0)}
    seg = _segments(payloads, {"A": Placement(name="A", ty=-3.0, tx=2.0)})[0]
    assert abs(seg["a"][1]) < 1e-9 and abs(seg["a"][0] - 2.0) < 1e-9 and abs(seg["length"] - 10.0) < 1e-9


# --- bent scans: one wall off at every height while the others agree -----------------------------------------------


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


# --- walls corrected by hand in review -----------------------------------------------------------------------------


WALLS = [{"i": 0, "x1": 0.0, "y1": 0.0, "x2": 6.0, "y2": 0.0}, {"i": 1, "x1": 6.0, "y1": 0.0, "x2": 6.0, "y2": 4.0},
         {"i": 2, "x1": 2.0, "y1": 2.0, "x2": 3.0, "y2": 2.0}]          # wall 2: a false wall (a sofa back)


def test_delete_move_and_add_apply_in_order():
    edits = [{"op": "delete", "at": [2.0, 2.0, 3.0, 2.0]},
             {"op": "move", "at": [6.0, 0.0, 6.0, 4.0], "by": -0.12},     # left-hand normal of +y is -x: 12 cm out
             {"op": "add", "wall": [0.0, 4.0, 6.0, 4.0]}]
    out, missed = apply_wall_edits(WALLS, edits)
    assert missed == 0 and len(out) == 3, (missed, out)
    assert [w["i"] for w in out] == [0, 1, -1], out
    assert abs(out[1]["x1"] - 6.12) < 1e-9 and abs(out[1]["x2"] - 6.12) < 1e-9 and out[1]["source"] == "moved", out[1]
    assert out[2]["source"] == "hand" and out[2]["len"] == 6.0, out[2]
    assert WALLS[1]["x1"] == 6.0, "the detected walls themselves were changed"


def test_an_edit_still_finds_its_wall_after_the_scan_is_re_detected():
    redetected = [{"i": 7, "x1": 0.1, "y1": 0.05, "x2": 5.8, "y2": 0.05}]     # renumbered, 5 cm over, a bit shorter
    assert find_wall(redetected, [0.0, 0.0, 6.0, 0.0]) == 0
    assert find_wall(redetected, [0.0, 0.3, 6.0, 0.3]) is None                # 30 cm over: a different wall
    assert find_wall(redetected, [0.0, 0.0, 0.0, 6.0]) is None                # crosswise: a different wall


def test_setting_a_walls_ends_moves_both_ends_and_a_later_edit_finds_it_there():
    # End points dragged in X and Y.
    edits = [{"op": "ends", "at": [0.0, 0.0, 6.0, 0.0], "wall": [-0.5, 0.1, 4.0, 0.1]},
             {"op": "move", "at": [-0.5, 0.1, 4.0, 0.1], "by": 0.2}]
    out, missed = apply_wall_edits(WALLS, edits)
    assert missed == 0, out
    w = out[0]
    assert (w["x1"], w["x2"]) == (-0.5, 4.0) and abs(w["y1"] - 0.3) < 1e-9 and abs(w["y2"] - 0.3) < 1e-9, w
    assert w["len"] == 4.5 and w["source"] == "moved", w


def test_an_edit_whose_wall_is_gone_is_counted_not_guessed():
    out, missed = apply_wall_edits(WALLS, [{"op": "delete", "at": [10.0, 10.0, 12.0, 10.0]}])
    assert missed == 1 and len(out) == 3


def test_payloads_get_the_corrected_walls_and_unedited_scans_are_untouched():
    data = {"scans": [{"name": "a", "wall_edits": [{"op": "delete", "at": [2.0, 2.0, 3.0, 2.0]}]}, {"name": "b"}]}
    payloads = {"a": {"walls": WALLS, "xy": [1]}, "b": {"walls": WALLS}}
    out = with_wall_edits(payloads, edits_by_scan(data))
    assert len(out["a"]["walls"]) == 2 and out["a"]["xy"] == [1] and out["b"] is payloads["b"], out
    assert len(payloads["a"]["walls"]) == 3, "the stored payload was changed"


def test_frozen_edits_give_the_same_walls_on_a_different_wall_finder_and_later_edits_still_apply():
    edits = [{"op": "delete", "at": [2.0, 2.0, 3.0, 2.0]}, {"op": "add", "wall": [0.0, 4.0, 6.0, 4.0]}]
    before, _ = apply_wall_edits(WALLS, edits)
    kept = frozen(WALLS, edits, "test")
    assert len(kept) == 1 and kept[0]["op"] == "set", kept
    newer = [{"i": 0, "x1": 0.0, "y1": 0.3, "x2": 2.0, "y2": 0.3}]          # a different finder, different walls
    after, missed = apply_wall_edits(newer, kept + [{"op": "delete", "at": [6.0, 0.0, 6.0, 4.0]}])
    assert missed == 0 and len(after) == 2, after
    assert [(w["x1"], w["y1"], w["x2"], w["y2"]) for w in after] == \
        [(w["x1"], w["y1"], w["x2"], w["y2"]) for w in before if w["x1"] != 6.0 or w["x2"] != 6.0], after
    assert [w.get("source") for w in after] == [None, "hand"], after      # a detected wall stays unlabelled


def test_a_tag_says_what_a_wall_contains_follows_it_through_a_move_and_survives_freezing():
    edits = [{"op": "tag", "at": [6.0, 0.0, 6.0, 4.0], "contains": ["door", "window"]},
             {"op": "move", "at": [6.0, 0.0, 6.0, 4.0], "by": -0.12}]
    out, missed = apply_wall_edits(WALLS, edits)
    assert missed == 0 and out[1]["contains"] == ["door", "window"] and out[1]["source"] == "moved", out[1]
    cleared, _ = apply_wall_edits(WALLS, edits + [{"op": "tag", "at": [6.12, 0.0, 6.12, 4.0], "contains": []}])
    assert "contains" not in cleared[1], "an empty tag did not clear the wall's list"
    kept = frozen(WALLS, edits, "test")
    again, _ = apply_wall_edits([{"i": 0, "x1": 0.0, "y1": 0.3, "x2": 2.0, "y2": 0.3}], kept)
    assert [w.get("contains") for w in again] == [None, ["door", "window"], None], again


def test_opening_a_project_freezes_edits_made_on_an_older_wall_finder_and_writes_them_once():
    """The server's side (`_freeze_wall_edits`, on open): edits on a scan whose preview came from an older wall finder
    are written to the project as the walls they gave; a scan on the current finder, and a second open, change nothing;
    the file before is kept as a saved version."""
    import glob
    import json
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import build_bench
    from open_point_stitch.server import _freeze_wall_edits
    edits = [{"op": "delete", "at": [2.0, 2.0, 3.0, 2.0]}, {"op": "add", "wall": [0.0, 4.0, 6.0, 4.0]}]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        paths = {}
        for name, version in (("old", build_bench.WALL_FINDER_SINCE - 1), ("new", build_bench.WALL_FINDER_SINCE)):
            paths[name] = os.path.join(tmp, f"{name}.pts")
            with open(paths[name], "w") as fh:
                fh.write("1\n0 0 0 -1 0 0 0\n")
            _, cpath = build_bench._cache_key_and_path(paths[name], cache)
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            with open(cpath, "w") as fh:
                json.dump({"_cache_key": {"version": version}, "payload": {"walls": WALLS}}, fh)
        saved = {"schema": "open_point_stitch.placement.v1", "scans": [{"name": n, "wall_edits": edits} for n in paths]}
        with open(os.path.join(tmp, "placement.json"), "w") as fh:
            json.dump(saved, fh)
        written = _freeze_wall_edits(tmp, saved, paths, cache)
        by_name = {s["name"]: s["wall_edits"] for s in written["scans"]}
        assert by_name["old"] == frozen(WALLS, edits, by_name["old"][0]["why"]), by_name["old"]
        assert by_name["new"] == edits, "a scan already on the current wall finder was frozen"
        with open(os.path.join(tmp, "placement.json")) as fh:
            assert json.load(fh)["scans"][0]["wall_edits"][0]["op"] == "set", "the frozen walls were not written"
        assert glob.glob(os.path.join(tmp, "backups", "*")), "the file before freezing was not kept"
        again = _freeze_wall_edits(tmp, written, paths, cache)
        assert again is written, "a second open froze again"


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
