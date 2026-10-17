"""Tests for the backup retention decision: pure logic, plain calls."""
from __future__ import annotations

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.backup_retention import BackupInfo, backups_to_remove, dry_run_summary  # noqa: E402

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


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
