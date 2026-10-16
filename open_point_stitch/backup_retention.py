"""Which placement backups a cleaner would remove. Pure decision logic with no filesystem access, so it can be
dry-run, tested and reused by both the automatic and the on-demand tidy.

Rule: always keep the newest backup and anything pinned. Otherwise keep every backup from the last hour, then the
newest per hour for a day, then the newest per day after that."""
from __future__ import annotations

import dataclasses
import datetime


@dataclasses.dataclass
class BackupInfo:
    id: str
    at: datetime.datetime  # backup timestamp, timezone-aware (UTC)
    size: int
    pinned: bool = False


def backups_to_remove(
    backups: list[BackupInfo],
    now: datetime.datetime,
    recent_hours: float = 1.0,
    hourly_days: float = 1.0,
) -> list[BackupInfo]:
    """Every backup the retention rule would delete, oldest first. Never
    includes the single newest backup or anything pinned."""
    if not backups:
        return []
    ordered = sorted(backups, key=lambda b: b.at, reverse=True)
    newest_id = ordered[0].id
    recent_cutoff = now - datetime.timedelta(hours=recent_hours)
    hourly_cutoff = now - datetime.timedelta(days=hourly_days)

    kept_hour_buckets: set[tuple] = set()
    kept_day_buckets: set[tuple] = set()
    remove = []
    for b in ordered:
        if b.id == newest_id or b.pinned:
            continue
        if b.at >= recent_cutoff:
            continue  # inside the always-keep window
        if b.at >= hourly_cutoff:
            bucket = (b.at.year, b.at.month, b.at.day, b.at.hour)
            if bucket in kept_hour_buckets:
                remove.append(b)
            else:
                kept_hour_buckets.add(bucket)
            continue
        bucket = (b.at.year, b.at.month, b.at.day)
        if bucket in kept_day_buckets:
            remove.append(b)
        else:
            kept_day_buckets.add(bucket)
    remove.sort(key=lambda b: b.at)
    return remove


def dry_run_summary(backups: list[BackupInfo], now: datetime.datetime,
                    recent_hours: float = 1.0, hourly_days: float = 1.0) -> dict:
    """What a real run would do, without touching anything - the preview the
    UI shows before Delete is ever pressed."""
    remove = backups_to_remove(backups, now, recent_hours, hourly_days)
    return {
        "n_total": len(backups),
        "n_remove": len(remove),
        "n_keep": len(backups) - len(remove),
        "bytes_freed": sum(b.size for b in remove),
        "remove_ids": [b.id for b in remove],
    }
