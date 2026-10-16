"""Tests for tidying saved placement versions on disk."""
from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import backup_cleaner as bc  # noqa: E402

NOW = datetime.datetime(2026, 9, 30, 12, 0, 0, tzinfo=datetime.timezone.utc)


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
        got = bc.tidy(pdir, rows, _resolve(pdir), NOW, 1.0, 1.0, golden_dir=golden)
        assert got["n_removed"] == 4 and got["bytes_removed"] == 400, got
        left = sorted(os.listdir(os.path.join(pdir, "backups")))
        assert len(left) == 4 and rows[0]["id"] in left, left


def test_pinned_and_named_versions_are_never_removed():
    with tempfile.TemporaryDirectory() as pdir, tempfile.TemporaryDirectory() as golden:
        rows = _make(pdir, [0.1, 50.0, 50.2, 50.4, 50.6, 50.8])
        old = [r["id"] for r in rows[1:]]
        bc.set_pinned(pdir, old[0], True)
        os.makedirs(os.path.join(pdir, "reports"))
        with open(os.path.join(pdir, "reports", "align_auto_pairs-x.json"), "w") as fh:
            json.dump({"started_from": old[1]}, fh)
        with open(os.path.join(golden, old[2]), "w") as fh:
            fh.write("{}")
        plan = bc.plan(pdir, rows, NOW, 1.0, 1.0, golden_dir=golden)
        assert plan["remove_ids"] == [old[4]], plan   # old[3] stays as that day's one kept version
        assert plan["n_pinned"] == 1 and plan["n_named_elsewhere"] == 2, plan


def test_pins_survive_a_reread_and_can_be_undone():
    with tempfile.TemporaryDirectory() as pdir:
        name = "placement.backup-20260930T010151.870Z.json"
        assert bc.set_pinned(pdir, name, True) == {name}
        assert bc.read_pins(pdir) == {name}
        assert bc.set_pinned(pdir, name, False) == set() and bc.read_pins(pdir) == set()


def test_a_path_outside_the_project_is_never_removed():
    with tempfile.TemporaryDirectory() as pdir, tempfile.TemporaryDirectory() as elsewhere, \
            tempfile.TemporaryDirectory() as golden:
        rows = _make(pdir, [0.1, 50.0, 50.2])
        outside = os.path.join(elsewhere, rows[-1]["id"])
        with open(outside, "w") as fh:
            fh.write("{}")
        got = bc.tidy(pdir, rows, lambda i: outside, NOW, 1.0, 1.0, golden_dir=golden)
        assert got["n_removed"] == 0 and os.path.exists(outside), got


def test_a_dry_run_changes_nothing():
    with tempfile.TemporaryDirectory() as pdir, tempfile.TemporaryDirectory() as golden:
        rows = _make(pdir, [0.1, 50.0, 50.2, 50.4])
        before = sorted(os.listdir(os.path.join(pdir, "backups")))
        plan = bc.plan(pdir, rows, NOW, 1.0, 1.0, golden_dir=golden)
        assert plan["n_remove"] == 2 and sorted(os.listdir(os.path.join(pdir, "backups"))) == before


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
