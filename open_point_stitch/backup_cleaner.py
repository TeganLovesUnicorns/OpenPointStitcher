"""Tidy a project's saved placement versions. backup_retention decides; this reads pins and references and
removes files. Never removes the newest, a pinned one, or one a report, job, export or the golden folder names."""
from __future__ import annotations

import datetime
import json
import os
import re
from typing import Callable, Optional

from . import project as project_mod
from .backup_retention import BackupInfo, dry_run_summary

PINS_FILE = "pins.json"
_NAME_IN_TEXT = re.compile(r"placement\.backup-\d{8}T\d{6}\.\d{3}Z(?:-\d+)?\.json")
_MAX_TEXT_BYTES = 5 * 1024 * 1024   # reference scan reads JSON/text files up to this size


def default_golden_dir() -> str:
    """OPENPOINT_GOLDEN_DIR if set, else the repo's own golden/ next to the package."""
    return os.environ.get("OPENPOINT_GOLDEN_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden")


def read_pins(pdir: str) -> set[str]:
    try:
        with open(os.path.join(project_mod.backups_dir(pdir), PINS_FILE)) as fh:
            return {p for p in json.load(fh).get("pinned", []) if project_mod.BACKUP_RE.match(p)}
    except (OSError, ValueError, AttributeError):
        return set()


def set_pinned(pdir: str, backup_id: str, pinned: bool) -> set[str]:
    """Pin or unpin one saved version; returns every pinned id."""
    pins = read_pins(pdir)
    (pins.add if pinned else pins.discard)(backup_id)
    folder = project_mod.backups_dir(pdir)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, PINS_FILE + ".tmp")
    with open(tmp, "w") as fh:
        json.dump({"pinned": sorted(pins)}, fh, indent=1)
    os.replace(tmp, os.path.join(folder, PINS_FILE))
    return pins


def _names_in_tree(root: str) -> set[str]:
    """Backup names under `root`: file names, and names written inside small JSON/text files."""
    found: set[str] = set()
    if not os.path.isdir(root):
        return found
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if project_mod.BACKUP_RE.match(name):
                found.add(name)
                continue
            path = os.path.join(folder, name)
            if not name.endswith((".json", ".txt", ".md")) or os.path.getsize(path) > _MAX_TEXT_BYTES:
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    found.update(_NAME_IN_TEXT.findall(fh.read()))
            except OSError:
                continue
    return found


def referenced_ids(pdir: str, golden_dir: Optional[str] = None) -> set[str]:
    """Every saved version a report, job, export or the golden folder names."""
    found: set[str] = set()
    for sub in (project_mod.reports_dir(pdir), project_mod.exports_dir(pdir), os.path.join(pdir, "jobs")):
        found |= _names_in_tree(sub)
    return found | _names_in_tree(golden_dir if golden_dir is not None else default_golden_dir())


def _parse_at(at: str) -> datetime.datetime:
    return datetime.datetime.strptime(at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=datetime.timezone.utc)


def plan(pdir: str, listing: list[dict], now: datetime.datetime, keep_all_hours: float, hourly_days: float,
         golden_dir: Optional[str] = None) -> dict:
    """What a tidy would remove, without touching anything. `listing` is the server's backup list."""
    pins = read_pins(pdir)
    infos = [BackupInfo(id=b["id"], at=_parse_at(b["at"]), size=int(b["size"]), pinned=b["id"] in pins)
             for b in listing]
    summary = dry_run_summary(infos, now, keep_all_hours, hourly_days)
    if summary["n_remove"]:                        # only pay for the reference scan when it matters
        refs = referenced_ids(pdir, golden_dir)
        for b in infos:
            b.pinned = b.pinned or b.id in refs
        summary = dry_run_summary(infos, now, keep_all_hours, hourly_days)
        summary["n_named_elsewhere"] = len(refs & {b.id for b in infos})
    summary.setdefault("n_named_elsewhere", 0)
    summary["n_pinned"] = len(pins & {b.id for b in infos})
    return summary


def tidy(pdir: str, listing: list[dict], resolve: Callable[[str], Optional[str]], now: datetime.datetime,
         keep_all_hours: float, hourly_days: float, golden_dir: Optional[str] = None) -> dict:
    """Remove what `plan` says. `resolve` maps an id to its file inside this project, or None."""
    summary = plan(pdir, listing, now, keep_all_hours, hourly_days, golden_dir)
    removed, freed = [], 0
    project_root = os.path.realpath(pdir)
    for backup_id in summary["remove_ids"]:
        if not project_mod.BACKUP_RE.match(backup_id):   # only ever a saved version: never the hand log or anything else
            continue
        path = resolve(backup_id)
        if not path or os.path.commonpath([os.path.realpath(path), project_root]) != project_root:
            continue
        try:
            size = os.path.getsize(path)
            os.remove(path)
        except OSError:
            continue
        removed.append(backup_id)
        freed += size
    summary.update(removed_ids=removed, n_removed=len(removed), bytes_removed=freed)
    return summary
