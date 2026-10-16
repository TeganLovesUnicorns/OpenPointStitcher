"""Tests for wall_edits.py: walls corrected by hand in the review view."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.wall_edits import apply_wall_edits, edits_by_scan, find_wall, frozen, with_wall_edits  # noqa: E402

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
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
