"""project.py: project folders, imports and uploads, saved versions and their tidy, and the hand-move log."""
from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "measurements"))

from open_point_stitch import project as pj  # noqa: E402
from open_point_stitch.project import BackupInfo, backups_to_remove, dry_run_summary  # noqa: E402


# --- project folders, imports and uploads --------------------------------------------------------------------------


def _make_pts(path: str, n: int = 3) -> None:
    with open(path, "w") as fh:
        fh.write(f"{n}\n")
        for i in range(n):
            fh.write(f"{i}.0 {i}.0 0.0 -1 200 200 200\n")


def test_sanitize_rejects_traversal_and_empty():
    for bad in ("", "   ", "../evil", "/etc/passwd", "a" * 65, "..", "."):
        try:
            pj.sanitize_project_name(bad)
            raise AssertionError(f"expected {bad!r} to be rejected")
        except pj.InvalidProjectName:
            pass


def test_sanitize_accepts_ordinary_names():
    assert pj.sanitize_project_name("  My Venue  ") == "My Venue"
    assert pj.sanitize_project_name("theatre_2026-09") == "theatre_2026-09"


def test_create_project_makes_scans_raw_and_empty_file():
    with tempfile.TemporaryDirectory() as root:
        pdir = pj.create_project(root, "My Venue")
        assert os.path.isdir(pj.scans_raw_dir(pdir))
        with open(pj.project_file_path(pdir)) as fh:
            data = json.load(fh)
        assert data["schema"] == "open_point_stitch.placement.v1"
        assert data["scans"] == []
        assert data["exclusions"] == []


def test_create_project_twice_refuses():
    with tempfile.TemporaryDirectory() as root:
        pj.create_project(root, "My Venue")
        try:
            pj.create_project(root, "My Venue")
            raise AssertionError("expected FileExistsError on the second create")
        except FileExistsError:
            pass


def test_import_copies_without_touching_source():
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "scans")
        os.makedirs(source)
        _make_pts(os.path.join(source, "Point Cloud A.pts"))
        _make_pts(os.path.join(source, "Point Cloud B.pts"))

        pdir = pj.create_project(root, "Venue")
        imported = pj.import_scans(source, pdir)

        assert sorted(imported) == ["Point Cloud A", "Point Cloud B"]
        raw = pj.scans_raw_dir(pdir)
        assert os.path.exists(os.path.join(raw, "Point Cloud A.pts"))
        assert os.path.exists(os.path.join(raw, "Point Cloud B.pts"))
        # the source files are untouched - still there, still readable
        assert os.path.exists(os.path.join(source, "Point Cloud A.pts"))
        assert os.path.exists(os.path.join(source, "Point Cloud B.pts"))


def test_import_can_be_scoped_to_names():
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "scans")
        os.makedirs(source)
        _make_pts(os.path.join(source, "A.pts"))
        _make_pts(os.path.join(source, "B.pts"))

        pdir = pj.create_project(root, "Venue")
        imported = pj.import_scans(source, pdir, names=["A"])

        assert imported == ["A"]
        raw = pj.scans_raw_dir(pdir)
        assert os.path.exists(os.path.join(raw, "A.pts"))
        assert not os.path.exists(os.path.join(raw, "B.pts"))


def test_list_projects_reports_scan_and_placed_counts():
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "scans")
        os.makedirs(source)
        _make_pts(os.path.join(source, "A.pts"))

        pdir = pj.create_project(root, "Venue")
        pj.import_scans(source, pdir)
        with open(pj.project_file_path(pdir), "w") as fh:
            json.dump({"schema": "open_point_stitch.placement.v1",
                      "scans": [{"name": "A"}], "anchors": []}, fh)

        listed = pj.list_projects(root)
        assert len(listed) == 1
        assert listed[0]["name"] == "Venue"
        assert listed[0]["n_scans_raw"] == 1
        assert listed[0]["n_placed"] == 1


def test_list_projects_on_missing_root_is_empty_not_an_error():
    assert pj.list_projects("/does/not/exist/anywhere") == []


def test_list_import_source_reports_point_counts():
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "scans")
        os.makedirs(source)
        _make_pts(os.path.join(source, "A.pts"), n=7)

        listed = pj.list_import_source(source)
        assert listed == [{"name": "A", "points": 7}]


def test_list_import_source_counts_past_a_leading_geodata_line():
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "G.pts"), "w") as fh:
            fh.write(';{"GeoData":{"Lat":45.0,"Lon":-93.1}}\n2\n0 0 0\n1 1 1\n')
        assert pj.list_import_source(root) == [{"name": "G", "points": 2}]


def test_a_project_with_no_scans_says_so_and_stays_closed():
    """The page offers its import pane on this answer, so it must be told apart from a missing project."""
    from open_point_stitch.server import ProjectHasNoScans, Session
    with tempfile.TemporaryDirectory() as root:
        pj.create_project(root, "empty")
        s = Session(root, root, root)
        try:
            s.open_project("empty")
        except ProjectHasNoScans:
            pass
        else:
            raise AssertionError("an empty project opened")
        assert s.app is None and s.active_project is None
        try:
            s.open_project("missing")
        except ProjectHasNoScans:
            raise AssertionError("a missing project was reported as an empty one")
        except FileNotFoundError:
            pass


def test_migrate_project_layout_moves_flat_backups_reports_and_exports():
    """An old-layout project's loose files move into their subfolders; nothing is deleted, and files of no known shape
    are left alone."""
    with tempfile.TemporaryDirectory() as root:
        pdir = pj.create_project(root, "Venue")
        flat = {
            "placement.backup-20260930T010000.000Z.json": "backups",
            "align_auto_pairs-20260930T010000.000Z.json": "reports",
            "align_anchors_pairs-20260930T010000.000Z.json": "reports",
            "merged.pts": "exports",
            "plan_view.dxf": "exports",
            "plan_view.png": "exports",
        }
        for name in flat:
            with open(os.path.join(pdir, name), "w") as fh:
                fh.write("x")
        with open(os.path.join(pdir, "placement.json"), "w") as fh:
            fh.write("{}")  # the live file - must NOT be moved anywhere

        moved = pj.migrate_project_layout(pdir)
        assert moved == {"backups": 1, "reports": 2, "exports": 3}

        for name, kind in flat.items():
            dest_dir = {"backups": pj.backups_dir, "reports": pj.reports_dir,
                       "exports": pj.exports_dir}[kind](pdir)
            assert os.path.isfile(os.path.join(dest_dir, name)), f"{name} not moved to {kind}/"
            assert not os.path.isfile(os.path.join(pdir, name)), f"{name} still sitting flat"
        assert os.path.isfile(pj.project_file_path(pdir)), "placement.json must stay at the project root"


def test_migrate_project_layout_is_idempotent_and_never_overwrites():
    with tempfile.TemporaryDirectory() as root:
        pdir = pj.create_project(root, "Venue")
        name = "placement.backup-20260930T010000.000Z.json"
        with open(os.path.join(pdir, name), "w") as fh:
            fh.write("first")

        pj.migrate_project_layout(pdir)
        # A second file lands flat under the same name after the first migrated -
        # migration must not clobber what is already in backups/.
        with open(os.path.join(pdir, name), "w") as fh:
            fh.write("second")
        moved = pj.migrate_project_layout(pdir)

        assert moved == {"backups": 0, "reports": 0, "exports": 0}
        with open(os.path.join(pj.backups_dir(pdir), name)) as fh:
            assert fh.read() == "first"
        with open(os.path.join(pdir, name)) as fh:
            assert fh.read() == "second"  # left flat, not silently lost


def test_migrate_project_layout_on_missing_project_is_a_noop():
    assert pj.migrate_project_layout("/does/not/exist/anywhere") == {
        "backups": 0, "reports": 0, "exports": 0}


def test_reopening_the_open_project_keeps_it_loaded_until_a_scan_file_changes():
    from open_point_stitch.server import Session
    # A preview cache write can land as the folder is removed; removing it is not what this tests.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        pdir = pj.create_project(root, "p")
        raw = pj.scans_raw_dir(pdir)
        _make_pts(os.path.join(raw, "a.pts"))
        s = Session(root, root, root)
        try:
            s.open_project("p")
            first = s.app
            first.wait_for_all()
            s.open_project("p")
            assert s.app is first, "re-opening the open project rebuilt it"
            _make_pts(os.path.join(raw, "a.pts"), n=5)          # a different file: size changes
            s.open_project("p")
            assert s.app is not first, "a changed scan file was not picked up on re-open"
        finally:
            if s.app is not None:
                s.app.wait_for_all()                            # no build may outlive the temporary folder
                s.app.shutdown()


def test_opening_reads_saved_previews_without_a_worker_and_makes_no_saved_version():
    from open_point_stitch.server import Session
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        pdir = pj.create_project(root, "p")
        _make_pts(os.path.join(pj.scans_raw_dir(pdir), "a.pts"))
        backups = pj.backups_dir(pdir)
        n_saved = lambda: len(os.listdir(backups)) if os.path.isdir(backups) else 0
        s = Session(root, root, root)
        s.open_project("p")
        s.app.wait_for_all()
        assert s.app.build_cached == {"a": False}, s.app.build_cached
        s.app.shutdown()
        before = n_saved()
        s2 = Session(root, root, root)                           # a restart: nothing loaded
        try:
            s2.open_project("p")
            s2.app.wait_for_all()
            assert s2.app.build_cached == {"a": True}, s2.app.build_cached
            assert not s2.app._preview_pool.workers(), "a saved preview started a worker process"
            assert "a" not in s2.app.quick_bbox, "the rough box ran although the saved preview was valid"
            s2.open_project("p")
            assert n_saved() == before, "opening made a saved version"
        finally:
            s2.app.wait_for_all()
            s2.app.shutdown()


def _pts_bytes(n: int = 4) -> bytes:
    return (f"{n}\n" + "".join(f"{i}.5 {i}.25 1.0 -1 10 20 30\n" for i in range(n))).encode()


def test_an_uploaded_scan_is_copied_byte_for_byte_and_counted():
    # A scan from anywhere on the computer, sent by the page.
    import io
    with tempfile.TemporaryDirectory() as root:
        pdir = pj.create_project(root, "Venue")
        body = _pts_bytes(4)
        got = pj.import_upload(pdir, "Lobby.pts", io.BytesIO(body), len(body))
        assert got == {"name": "Lobby", "file": "Lobby.pts", "points": 4, "bytes": len(body)}, got
        with open(os.path.join(pj.scans_raw_dir(pdir), "Lobby.pts"), "rb") as fh:
            assert fh.read() == body, "the copy is not byte for byte"
        assert os.listdir(os.path.join(pdir, pj.INCOMING_DIRNAME)) == [], "a half-received file was left behind"


def test_a_duplicate_upload_is_refused_and_keep_both_uses_a_new_name():
    import io
    with tempfile.TemporaryDirectory() as root:
        pdir = pj.create_project(root, "Venue")
        first, second = _pts_bytes(4), _pts_bytes(6)
        pj.import_upload(pdir, "Lobby.pts", io.BytesIO(first), len(first))
        try:
            pj.import_upload(pdir, "Lobby.pts", io.BytesIO(second), len(second))
            raise AssertionError("a second Lobby.pts was accepted")
        except pj.DuplicateScan as exc:
            assert exc.suggested == "Lobby (2).pts" and "already has a scan" in str(exc), (exc.suggested, str(exc))
        with open(os.path.join(pj.scans_raw_dir(pdir), "Lobby.pts"), "rb") as fh:
            assert fh.read() == first, "the scan already there was overwritten"
        got = pj.import_upload(pdir, "Lobby.pts", io.BytesIO(second), len(second), keep_both=True)
        assert got["file"] == "Lobby (2).pts" and got["points"] == 6, got


def test_an_unreadable_or_misnamed_upload_leaves_nothing_behind():
    import io
    with tempfile.TemporaryDirectory() as root:
        pdir = pj.create_project(root, "Venue")
        for name, body in (("Broken.pts", b"this is not a scan\nat all\n"), ("notes.txt", _pts_bytes(3)),
                           ("../escape.pts", b"")):
            try:
                pj.import_upload(pdir, name, io.BytesIO(body), len(body))
                raise AssertionError(f"{name!r} was accepted")
            except ValueError as exc:
                assert not isinstance(exc, pj.DuplicateScan), exc
        raw = pj.scans_raw_dir(pdir)
        assert os.listdir(raw) == [], os.listdir(raw)
        incoming = os.path.join(pdir, pj.INCOMING_DIRNAME)
        assert not os.path.isdir(incoming) or os.listdir(incoming) == [], os.listdir(incoming)


def test_importing_from_the_folder_never_overwrites_a_scan_already_there():
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "scans")
        os.makedirs(source)
        _make_pts(os.path.join(source, "A.pts"), n=3)
        pdir = pj.create_project(root, "Venue")
        pj.import_scans(source, pdir)
        _make_pts(os.path.join(source, "A.pts"), n=7)          # the source changed since
        skipped: list = []
        assert pj.import_scans(source, pdir, skipped=skipped) == [] and skipped == ["A"], skipped
        with open(os.path.join(pj.scans_raw_dir(pdir), "A.pts")) as fh:
            assert fh.readline().strip() == "3", "the project's own copy was overwritten"


# --- which saved versions a tidy keeps -----------------------------------------------------------------------------


NOW = datetime.datetime(2026, 9, 30, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _at(hours_ago: float) -> datetime.datetime:
    return NOW - datetime.timedelta(hours=hours_ago)


def test_everything_within_the_recent_window_is_kept():
    backups = [BackupInfo(id=f"b{i}", at=_at(i * 0.1), size=100) for i in range(10)]
    assert backups_to_remove(backups, NOW) == []


def test_newest_is_always_kept_even_if_very_old():
    backups = [BackupInfo(id="only", at=_at(500), size=100)]
    assert backups_to_remove(backups, NOW) == []


def test_pinned_is_never_removed_regardless_of_age():
    # keeper, pinned_old and unpinned_mid share one day bucket, so bucketing alone keeps only keeper; the pin must
    # save pinned_old.
    backups = [
        BackupInfo(id="newest", at=_at(0), size=100),
        BackupInfo(id="keeper", at=_at(200), size=100),
        BackupInfo(id="pinned_old", at=_at(200.3), size=100, pinned=True),
        BackupInfo(id="unpinned_mid", at=_at(200.6), size=100),
    ]
    removed = {b.id for b in backups_to_remove(backups, NOW)}
    assert "pinned_old" not in removed
    assert "unpinned_mid" in removed


def test_hourly_bucket_keeps_one_and_drops_the_rest():
    # Five backups inside the same hour (09:xx UTC), all older than the 1h
    # recent window. 2.0h ago lands in the PREVIOUS hour bucket - start at 2.05.
    backups = [BackupInfo(id="newest", at=_at(0), size=100)]
    backups += [BackupInfo(id=f"h{i}", at=_at(2.05 + i * 0.05), size=100) for i in range(5)]
    removed = backups_to_remove(backups, NOW)
    kept_from_bucket = [b.id for b in backups if b.id.startswith("h")
                        and b.id not in {r.id for r in removed}]
    assert len(kept_from_bucket) == 1, kept_from_bucket
    assert len(removed) == 4


def test_daily_bucket_keeps_one_per_day_past_the_hourly_window():
    backups = [BackupInfo(id="newest", at=_at(0), size=100)]
    # Three backups on the same calendar day, all past the 1-day hourly window.
    backups += [BackupInfo(id=f"d{i}", at=_at(30 + i), size=100) for i in range(3)]
    removed = backups_to_remove(backups, NOW)
    kept = [b.id for b in backups if b.id.startswith("d")
           and b.id not in {r.id for r in removed}]
    assert len(kept) == 1, kept
    assert len(removed) == 2


def test_dry_run_summary_reports_bytes_freed_and_counts():
    backups = [
        BackupInfo(id="newest", at=_at(0), size=1000),
        BackupInfo(id="old1", at=_at(200), size=500),
        BackupInfo(id="old2", at=_at(200.5), size=700),  # same day bucket as old1
    ]
    summary = dry_run_summary(backups, NOW)
    assert summary["n_total"] == 3
    assert summary["n_remove"] == 1
    assert summary["n_keep"] == 2
    assert summary["bytes_freed"] in (500, 700)  # whichever of old1/old2 sorts later is kept


def test_empty_list_is_a_noop():
    assert backups_to_remove([], NOW) == []
    assert dry_run_summary([], NOW) == {
        "n_total": 0, "n_remove": 0, "n_keep": 0, "bytes_freed": 0, "remove_ids": []}


# --- the tidy: pins and what other files name ----------------------------------------------------------------------




def _make(pdir, hours_ago_list):
    """One backup file per age in `hours_ago_list`; returns the listing, newest first, as the server gives it."""
    folder = os.path.join(pdir, "backups")
    os.makedirs(folder, exist_ok=True)
    rows = []
    for h in hours_ago_list:
        at = NOW - datetime.timedelta(hours=h)
        name = f"placement.backup-{at:%Y%m%dT%H%M%S}.000Z.json"
        with open(os.path.join(folder, name), "w") as fh:
            fh.write("{}" + " " * 98)
        rows.append({"id": name, "at": f"{at:%Y-%m-%dT%H:%M:%S}.000Z", "size": 100})
    rows.sort(key=lambda r: r["at"], reverse=True)
    return rows


def _resolve(pdir):
    return lambda i: os.path.join(pdir, "backups", i) if os.path.exists(os.path.join(pdir, "backups", i)) else None


def test_tidy_removes_the_extra_old_versions_and_keeps_the_rest():
    with tempfile.TemporaryDirectory() as pdir, tempfile.TemporaryDirectory() as golden:
        rows = _make(pdir, [0.1, 0.5, 5.1, 5.2, 5.4, 50.0, 50.5, 51.0])   # 3 in one hour, 3 on one old day
        got = pj.tidy_backups(pdir, rows, _resolve(pdir), NOW, 1.0, 1.0, golden_dir=golden)
        assert got["n_removed"] == 4 and got["bytes_removed"] == 400, got
        left = sorted(os.listdir(os.path.join(pdir, "backups")))
        assert len(left) == 4 and rows[0]["id"] in left, left


def test_pinned_and_named_versions_are_never_removed():
    with tempfile.TemporaryDirectory() as pdir, tempfile.TemporaryDirectory() as golden:
        rows = _make(pdir, [0.1, 50.0, 50.2, 50.4, 50.6, 50.8])
        old = [r["id"] for r in rows[1:]]
        pj.set_pinned(pdir, old[0], True)
        os.makedirs(os.path.join(pdir, "reports"))
        with open(os.path.join(pdir, "reports", "align_auto_pairs-x.json"), "w") as fh:
            json.dump({"started_from": old[1]}, fh)
        with open(os.path.join(golden, old[2]), "w") as fh:
            fh.write("{}")
        plan = pj.plan_tidy(pdir, rows, NOW, 1.0, 1.0, golden_dir=golden)
        assert plan["remove_ids"] == [old[4]], plan   # old[3] stays as that day's one kept version
        assert plan["n_pinned"] == 1 and plan["n_named_elsewhere"] == 2, plan


def test_pins_survive_a_reread_and_can_be_undone():
    with tempfile.TemporaryDirectory() as pdir:
        name = "placement.backup-20260930T010151.870Z.json"
        assert pj.set_pinned(pdir, name, True) == {name}
        assert pj.read_pins(pdir) == {name}
        assert pj.set_pinned(pdir, name, False) == set() and pj.read_pins(pdir) == set()


def test_a_path_outside_the_project_is_never_removed():
    with tempfile.TemporaryDirectory() as pdir, tempfile.TemporaryDirectory() as elsewhere, \
            tempfile.TemporaryDirectory() as golden:
        rows = _make(pdir, [0.1, 50.0, 50.2])
        outside = os.path.join(elsewhere, rows[-1]["id"])
        with open(outside, "w") as fh:
            fh.write("{}")
        got = pj.tidy_backups(pdir, rows, lambda i: outside, NOW, 1.0, 1.0, golden_dir=golden)
        assert got["n_removed"] == 0 and os.path.exists(outside), got


def test_a_dry_run_changes_nothing():
    with tempfile.TemporaryDirectory() as pdir, tempfile.TemporaryDirectory() as golden:
        rows = _make(pdir, [0.1, 50.0, 50.2, 50.4])
        before = sorted(os.listdir(os.path.join(pdir, "backups")))
        plan = pj.plan_tidy(pdir, rows, NOW, 1.0, 1.0, golden_dir=golden)
        assert plan["n_remove"] == 2 and sorted(os.listdir(os.path.join(pdir, "backups"))) == before


# --- the hand-move log ---------------------------------------------------------------------------------------------


try:
    from hand_moves_summary import summarise  # noqa: E402  (a developer tool, not in the public repository)
except ImportError:
    summarise = None


def pose(x=0.0, yaw=0.0):
    return {"x": x, "y": 1.0, "z": 0.0, "yaw": yaw, "pitch": 0.0, "roll": 0.0}


def move(scan="kitchen", x=0.5, kind="nudge", source="hand"):
    return {"scan": scan, "kind": kind, "how": "drag", "view": "plan", "before": pose(0.0), "after": pose(x, 2.0),
            "before_source": {"kind": source, "run": "align_auto at 2026-10-02T20:02:38Z" if source != "hand" else ""}}


def test_valid_moves_are_appended_one_line_each_and_bad_ones_dropped():
    d = tempfile.mkdtemp()
    try:
        still = {**move(), "after": pose(0.0)}
        n = pj.append_hand_moves(d, [move(), still, {"scan": "x"}, "nonsense", move(source="automatic run")])
        assert n == 2, n                                        # unchanged pose, missing poses and junk are dropped
        lines = [json.loads(l) for l in open(os.path.join(d, pj.HAND_MOVES_FILE))]
        assert [l["before_source"]["kind"] for l in lines] == ["hand", "automatic run"]
        assert set(lines[0]) == {"at", "scan", "kind", "how", "view", "before", "after", "before_source"}
    finally:
        shutil.rmtree(d)
    print("    two good moves written as two lines; an unchanged pose and malformed input are not written")


def test_a_full_log_rolls_over_instead_of_growing():
    d = tempfile.mkdtemp()
    saved = pj.HAND_MOVES_MAX_BYTES
    try:
        line = len(json.dumps(pj.clean_hand_move(move()))) + 1
        pj.HAND_MOVES_MAX_BYTES = 2 * line                         # two lines fill it
        for _ in range(3):
            pj.append_hand_moves(d, [move()])
        assert os.path.exists(os.path.join(d, pj.HAND_MOVES_OLD_FILE))
        assert len(pj.read_hand_moves(d)) == 3                     # nothing lost at the first roll-over
        for _ in range(5):
            pj.append_hand_moves(d, [move()])
        assert os.path.getsize(os.path.join(d, pj.HAND_MOVES_FILE)) <= 2 * line   # never grows past the cap
    finally:
        pj.HAND_MOVES_MAX_BYTES = saved
        shutil.rmtree(d)
    print("    past the cap the file is renamed .old and a new one starts")


def test_reader_counts_moves_corrections_and_undos():
    if summarise is None:
        print("    SKIP: tools/measurements/hand_moves_summary.py is not in this checkout")
        return
    lines = [pj.clean_hand_move(m) for m in (move(), move(source="automatic run"), move(kind="undo"))]
    s = summarise(lines)
    assert s["kitchen"]["moves"] == 2 and s["kitchen"]["corrections"] == 1 and s["kitchen"]["undone"] == 1
    assert abs(s["kitchen"]["mean_m"] - 0.5) < 1e-6 and abs(s["kitchen"]["mean_deg"] - 2.0) < 1e-6
    print("    the reader counts 2 moves, 1 of them fixing an automatic run, and 1 undo kept apart")


def test_tidying_saved_versions_never_touches_the_log():
    d = tempfile.mkdtemp()
    try:
        pj.append_hand_moves(d, [move()])
        open(os.path.join(d, pj.HAND_MOVES_OLD_FILE), "w").write("{}\n")
        now = datetime.datetime(2026, 10, 2, 22, 0, tzinfo=datetime.timezone.utc)
        old = "placement.backup-20260101T000000.000Z.json"
        open(os.path.join(d, old), "w").write("{}")
        listing = [{"id": pj.HAND_MOVES_FILE, "at": "2026-01-01T00:00:00.000Z", "size": 10},
                   {"id": pj.HAND_MOVES_OLD_FILE, "at": "2026-01-01T00:00:01.000Z", "size": 10},
                   {"id": old, "at": "2026-01-01T00:00:02.000Z", "size": 2},
                   {"id": "placement.backup-20260101T000003.000Z.json", "at": "2026-01-01T00:00:03.000Z", "size": 2},
                   {"id": "placement.backup-20261002T215959.000Z.json", "at": "2026-10-02T21:59:59.000Z", "size": 2}]
        for b in listing[3:]:
            open(os.path.join(d, b["id"]), "w").write("{}")
        planned = pj.plan_tidy(d, listing, now, 1.0, 1.0)["remove_ids"]
        assert pj.HAND_MOVES_FILE in planned        # the retention rules alone WOULD pick it; the guard is what saves it
        result = pj.tidy_backups(d, listing, lambda i: os.path.join(d, i), now, 1.0, 1.0)
        assert os.path.exists(os.path.join(d, pj.HAND_MOVES_FILE)) and os.path.exists(os.path.join(d, pj.HAND_MOVES_OLD_FILE))
        assert pj.HAND_MOVES_FILE not in result["removed_ids"] and pj.HAND_MOVES_OLD_FILE not in result["removed_ids"]
        assert old in result["removed_ids"], result["removed_ids"]   # the tidy still does its own job
    finally:
        shutil.rmtree(d)
    print("    the tidy-up of saved versions leaves hand-moves.jsonl and its .old file alone, even if handed them")


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
