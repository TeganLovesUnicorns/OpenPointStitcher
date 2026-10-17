"""Tests that _write_placement keeps a backup of placement.json."""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.server import (_backup_file_path, _list_backups, _placement_path,  # noqa: E402
                                 _write_placement)


def _backups(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, "backups", "placement.backup-*.json")))


def test_first_write_makes_no_backup():
    """Nothing existed yet, so there is nothing to protect."""
    with tempfile.TemporaryDirectory() as d:
        _write_placement(d, {"schema": "x", "scans": [{"name": "a"}]})
        assert _backups(d) == [], "a first-ever save should not create a backup file"
        assert os.path.exists(_placement_path(d))


def test_second_write_backs_up_the_first():
    with tempfile.TemporaryDirectory() as d:
        first = {"schema": "x", "scans": [{"name": "a", "x_m": 1.0}]}
        second = {"schema": "x", "scans": [{"name": "a", "x_m": 2.0}]}
        _write_placement(d, first)
        _write_placement(d, second)

        backups = _backups(d)
        assert len(backups) == 1, f"expected exactly 1 backup after the second write, got {backups}"
        with open(backups[0]) as fh:
            backed_up = json.load(fh)
        assert backed_up == first, "the backup does not hold the PREVIOUS content"

        with open(_placement_path(d)) as fh:
            current = json.load(fh)
        assert current == second, "the live file should hold the NEW content, not the backup"


def test_clear_links_style_write_is_backed_up_too():
    """Clear-all-links is, from _write_placement's point of view, just
    another Save (this time with anchors: []) - it must get exactly the
    same protection as everything else that overwrites the file."""
    with tempfile.TemporaryDirectory() as d:
        with_anchors = {"schema": "x", "scans": [{"name": "a"}],
                        "anchors": [{"id": "anchor-1", "members": []}]}
        cleared = {"schema": "x", "scans": [{"name": "a"}], "anchors": []}
        _write_placement(d, with_anchors)
        _write_placement(d, cleared)

        backups = _backups(d)
        assert len(backups) == 1
        with open(backups[0]) as fh:
            backed_up = json.load(fh)
        assert backed_up["anchors"], "the backup should still hold the anchors that were about to be cleared"


def test_three_writes_keep_two_distinct_backups():
    """Each overwrite backs up whatever was live at that moment - a THIRD
    write must not silently clobber the backup made for the SECOND write."""
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            _write_placement(d, {"schema": "x", "scans": [{"name": "a", "x_m": float(i)}]})
        backups = _backups(d)
        assert len(backups) == 2, f"expected 2 backups after 3 writes, got {backups}"
        contents = []
        for b in backups:
            with open(b) as fh:
                contents.append(json.load(fh)["scans"][0]["x_m"])
        assert sorted(contents) == [0.0, 1.0], (
            f"backups should hold the FIRST and SECOND versions (x_m 0.0 and 1.0), got {contents}"
        )


def test_backups_go_in_their_own_folder_and_an_unchanged_save_adds_none():
    """Backups go in backups/, and a save that changes nothing makes no new copy."""
    with tempfile.TemporaryDirectory() as d:
        same = {"schema": "x", "scans": [{"name": "a", "x_m": 1.0}]}
        for _ in range(4):
            _write_placement(d, same)
        assert not glob.glob(os.path.join(d, "placement.backup-*.json")), "a backup landed at the top level"
        assert len(_backups(d)) == 1, f"4 identical saves should leave 1 backup, got {_backups(d)}"
        _write_placement(d, {"schema": "x", "scans": [{"name": "a", "x_m": 2.0}]})
        assert len(_backups(d)) == 1, "the identical content was already backed up; still nothing new"
        _write_placement(d, {"schema": "x", "scans": [{"name": "a", "x_m": 3.0}]})
        assert len(_backups(d)) == 2


def test_old_top_level_backups_still_list_and_restore():
    """A project whose tidy-up was skipped keeps working: flat backups are listed and readable."""
    with tempfile.TemporaryDirectory() as d:
        old = "placement.backup-20260901T120000.000Z.json"
        with open(os.path.join(d, old), "w") as fh:
            json.dump({"schema": "x", "scans": []}, fh)
        _write_placement(d, {"schema": "x", "scans": [{"name": "a"}]})
        _write_placement(d, {"schema": "x", "scans": [{"name": "b"}]})
        ids = [r["id"] for r in _list_backups(d)]
        assert old in ids and len(ids) == 2, ids
        assert ids[-1] == old, "newest first, so the September one is last"
        assert _backup_file_path(d, old) == os.path.join(d, old)
        assert _backup_file_path(d, ids[0]) == os.path.join(d, "backups", ids[0])


def test_a_backup_id_can_never_reach_outside_the_project():
    with tempfile.TemporaryDirectory() as d:
        _write_placement(d, {"schema": "x"})
        _write_placement(d, {"schema": "y"})
        for bad in ["../placement.json", "backups/../../x.json", "placement.json",
                    "..\\placement.backup-20260901T120000.000Z.json", "/etc/passwd"]:
            assert _backup_file_path(d, bad) is None, bad


def test_a_save_keeps_the_projects_settings_and_last_run():
    """The page never sends settings or last_run; a Save must keep both."""
    from open_point_stitch.server import _write_whole_placement
    with tempfile.TemporaryDirectory() as d:
        _write_placement(d, {"schema": "x", "scans": [], "settings": {"gpu": False}, "last_run": {"kind": "align"}})
        _write_whole_placement(d, {"schema": "x", "scans": [{"name": "a"}]})
        with open(_placement_path(d)) as fh:
            now = json.load(fh)
        assert now["settings"] == {"gpu": False} and now["last_run"] == {"kind": "align"}, now
        assert now["scans"] == [{"name": "a"}]


def _plant_old_versions(d, n):
    folder = os.path.join(d, "backups")
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        with open(os.path.join(folder, f"placement.backup-20260101T10{i:02d}00.000Z.json"), "w") as fh:
            fh.write("{}")


def test_a_save_tidies_old_versions_unless_the_setting_is_off():
    """The automatic tidy after a save."""
    with tempfile.TemporaryDirectory() as d:
        _plant_old_versions(d, 4)                                   # one old day: keep 1
        _write_placement(d, {"schema": "x", "settings": {"backup_auto_tidy": False}, "scans": [{"x_m": 1}]})
        _write_placement(d, {"schema": "x", "settings": {"backup_auto_tidy": False}, "scans": [{"x_m": 2}]})
        assert len(_backups(d)) == 5, "tidy ran with the setting off"
        _write_placement(d, {"schema": "x", "scans": [{"x_m": 3}]})  # default: on
        left = _backups(d)
        assert len(left) == 3 and any("20260101T100300" in p for p in left), left


def test_region_coordinates_are_rounded_on_disk_and_scan_poses_are_not():
    """Link and exclusion coordinates are rounded, keeping the file small."""
    with tempfile.TemporaryDirectory() as d:
        pt = [1.234567891234, -2.000049999, 0.333333333333]
        data = {"schema": "x", "scans": [{"name": "a", "x_m": 1.23456789, "level_rotation": [[0.123456789, 0, 0]]}],
                "anchors": [{"id": "a1", "members": [{"scan": "a", "points": [pt, pt], "centroid": pt, "extent": pt, "n_points": 2}]}],
                "exclusions": [{"id": "e1", "scan": "a", "points": [pt], "centroid": pt, "extent": pt, "n_points": 1}]}
        _write_placement(d, data)
        with open(_placement_path(d)) as fh:
            text = fh.read()
        got = json.loads(text)
        assert got["anchors"][0]["members"][0]["points"][0] == [1.2346, -2.0, 0.3333]
        assert got["exclusions"][0]["centroid"] == [1.2346, -2.0, 0.3333]
        assert got["scans"][0]["x_m"] == 1.23456789 and got["scans"][0]["level_rotation"][0][0] == 0.123456789
        assert "\n" not in text, "the file should be written compact"
        assert data["anchors"][0]["members"][0]["points"][0] == pt, "the caller's data must not be changed"


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
