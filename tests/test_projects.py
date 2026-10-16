"""Tests for project.py: creating, listing and importing into project folders. Plain calls against a temporary
directory; no server.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import project as pj  # noqa: E402


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


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
