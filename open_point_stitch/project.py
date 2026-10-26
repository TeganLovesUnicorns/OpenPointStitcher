"""Projects. A project is a folder: scans-raw/ holds copies of imported .pts files (the source is never written
to) and placement.json is the project file. Also its saved placement versions (which a tidy keeps, and the tidy
itself) and the hand-move log."""
from __future__ import annotations

import dataclasses
import datetime
import glob
import json
import os
import re
import shutil
import time
from typing import Callable, Optional

from open_point_stitch.lib.files import write_json_atomic
from open_point_stitch.placement import SCHEMA

SCANS_RAW_DIRNAME = "scans-raw"
PROJECT_FILE_NAME = "placement.json"  # same file server.py already reads/writes; see module docstring
BACKUPS_DIRNAME = "backups"
EXPORTS_DIRNAME = "exports"
REPORTS_DIRNAME = "reports"

# Allowed characters for a project name that becomes a Windows-safe folder.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")

# The exact file name shapes server.py's writers produce; the layout migration and backup cleaner match on them.
BACKUP_RE = re.compile(r"^placement\.backup-\d{8}T\d{6}\.\d{3}Z(?:-\d+)?\.json$")
REPORT_RE = re.compile(r"^(align_auto_pairs|align_anchors_pairs)-\d{8}T\d{6}\.\d{3}Z(?:-\d+)?\.json$")
REPORT_BASENAMES = ("wall_thickness.json",)
EXPORT_BASENAMES = ("merged.pts", "plan_view.dxf", "plan_view.png")


class InvalidProjectName(ValueError):
    pass


def sanitize_project_name(name: str) -> str:
    """Return the name if it is safe as a folder name; raise InvalidProjectName otherwise (never escaped)."""
    name = (name or "").strip()
    if not name:
        raise InvalidProjectName("project name cannot be empty")
    if not _SAFE_NAME_RE.match(name):
        raise InvalidProjectName(
            "project name must start with a letter or digit and contain only "
            "letters, digits, spaces, '-' or '_' (max 64 characters)"
        )
    return name


def project_dir(projects_root: str, name: str) -> str:
    return os.path.join(projects_root, sanitize_project_name(name))


def scans_raw_dir(pdir: str) -> str:
    return os.path.join(pdir, SCANS_RAW_DIRNAME)


def project_file_path(pdir: str) -> str:
    return os.path.join(pdir, PROJECT_FILE_NAME)


# server.py adds settings and last_run on first write.
_DEFAULT_PROJECT_FILE = {
    "schema": SCHEMA,
    "note": "per-scan yaw and position from the placement bench; scans are gravity levelled",
    "scans": [],
    "anchors": [],
    "exclusions": [],
}


def create_project(projects_root: str, name: str) -> str:
    """Make `<projects_root>/<name>/scans-raw/` and an empty project file; FileExistsError if the name is taken."""
    pdir = project_dir(projects_root, name)
    if os.path.exists(pdir):
        raise FileExistsError(f"a project named {sanitize_project_name(name)!r} already exists")
    os.makedirs(scans_raw_dir(pdir))
    write_json_atomic(project_file_path(pdir), _DEFAULT_PROJECT_FILE, indent=2)
    return pdir


def list_projects(projects_root: str) -> list[dict]:
    """Every subfolder with scans-raw/, including empty or unparseable ones (reported with an `error`, never skipped)."""
    out = []
    if not os.path.isdir(projects_root):
        return out
    for entry in sorted(os.listdir(projects_root)):
        pdir = os.path.join(projects_root, entry)
        raw_dir = scans_raw_dir(pdir)
        if not os.path.isdir(raw_dir):
            continue
        n_scans_raw = len(glob.glob(os.path.join(raw_dir, "*.pts")))
        fpath = project_file_path(pdir)
        n_placed = 0
        last_modified = None
        last_opened = None
        error = None
        if os.path.exists(fpath):
            last_modified = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(fpath)))
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
                n_placed = len(data.get("scans", []))
                # written by Session.open_project; None if never opened
                last_opened = data.get("last_opened")
            except (json.JSONDecodeError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        out.append({
            "name": entry, "n_scans_raw": n_scans_raw, "n_placed": n_placed,
            "last_modified": last_modified, "last_opened": last_opened, "error": error,
        })
    return out


def import_scans(source_dir: str, pdir: str, names: list[str] | None = None,
                 skipped: list[str] | None = None) -> list[str]:
    """Copy .pts files from source_dir into scans-raw/ (never a move). A name already in the project is left alone
    and listed in skipped, never overwritten."""
    available = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(source_dir, "*.pts"))
    }
    wanted = names if names is not None else sorted(available)
    dest_dir = scans_raw_dir(pdir)
    os.makedirs(dest_dir, exist_ok=True)
    imported = []
    for name in wanted:
        src = available.get(name)
        if src is None:
            continue  # asked for a name that is not (or no longer) in the import source
        dest = os.path.join(dest_dir, os.path.basename(src))
        if os.path.exists(dest):
            if skipped is not None:
                skipped.append(name)
            continue
        shutil.copy2(src, dest)
        imported.append(name)
    return imported


INCOMING_DIRNAME = ".incoming"        # half-received uploads; never inside scans-raw/, so never loaded as a scan
_UPLOAD_NAME_RE = re.compile(r"^[^/\\:*?\"<>|\x00-\x1f]{1,120}\.pts$", re.IGNORECASE)


class DuplicateScan(FileExistsError):
    """A scan with this file name is already in the project. `suggested` is a free name to keep both under."""
    def __init__(self, name: str, suggested: str):
        super().__init__(f"this project already has a scan called {name!r}")
        self.suggested = suggested


def free_scan_name(pdir: str, filename: str) -> str:
    """`filename` if scans-raw/ has no such file, else "stem (2).pts", "stem (3).pts", ..."""
    raw = scans_raw_dir(pdir)
    stem, ext = os.path.splitext(filename)
    cand, k = filename, 2
    while os.path.exists(os.path.join(raw, cand)):
        cand, k = f"{stem} ({k}){ext}", k + 1
    return cand


def upload_name(pdir: str, filename: str, keep_both: bool = False) -> str:
    """The file name an upload will be saved under. ValueError for a name that is not a .pts file; DuplicateScan for
    a name already in the project, unless `keep_both`, which picks the next free name."""
    filename = os.path.basename(filename.replace("\\", "/"))
    if not _UPLOAD_NAME_RE.match(filename):
        raise ValueError(f"{filename!r} is not a .pts file name")
    if os.path.exists(os.path.join(scans_raw_dir(pdir), filename)):
        if not keep_both:
            raise DuplicateScan(filename, free_scan_name(pdir, filename))
        filename = free_scan_name(pdir, filename)
    return filename


def import_upload(pdir: str, filename: str, stream, length: int, keep_both: bool = False) -> dict:
    """Write an uploaded .pts into scans-raw/ byte for byte, after checking it reads as a scan. Only the copy the
    browser sent is written; see upload_name for naming."""
    filename = upload_name(pdir, filename, keep_both)
    raw = scans_raw_dir(pdir)
    os.makedirs(raw, exist_ok=True)
    incoming = os.path.join(pdir, INCOMING_DIRNAME)
    os.makedirs(incoming, exist_ok=True)
    tmp = os.path.join(incoming, f"{int(time.time() * 1000)}-{os.getpid()}-{filename}")
    try:
        left = length
        with open(tmp, "wb") as fh:
            while left > 0:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    raise ValueError(f"the upload of {filename!r} stopped early ({length - left} of {length} bytes)")
                fh.write(chunk)
                left -= len(chunk)
        from .pts_file import read_pts
        try:
            points = len(read_pts(tmp).points)
        except Exception as exc:  # noqa: BLE001 - any parse failure is the same answer to the person
            raise ValueError(f"{filename!r} could not be read as a scan: {exc}") from None
        if points == 0:
            raise ValueError(f"{filename!r} has no points in it")
        dest = os.path.join(raw, filename)
        if os.path.exists(dest):                        # another upload took the name while this one was reading
            raise DuplicateScan(filename, free_scan_name(pdir, filename))
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"name": os.path.splitext(filename)[0], "file": filename, "points": points, "bytes": length}


def backups_dir(pdir: str) -> str:
    return os.path.join(pdir, BACKUPS_DIRNAME)


def exports_dir(pdir: str) -> str:
    return os.path.join(pdir, EXPORTS_DIRNAME)


def reports_dir(pdir: str) -> str:
    return os.path.join(pdir, REPORTS_DIRNAME)


def migrate_project_layout(pdir: str) -> dict:
    """Move backup, report and export files left at the project's top level into backups/, reports/ and exports/.
    Only moves, never deletes, and safe to call on every open. A name already at the destination stays put."""
    moved = {"backups": 0, "reports": 0, "exports": 0}
    if not os.path.isdir(pdir):
        return moved
    for name in os.listdir(pdir):
        src = os.path.join(pdir, name)
        if not os.path.isfile(src):
            continue
        if BACKUP_RE.match(name):
            dest_dir, key = backups_dir(pdir), "backups"
        elif REPORT_RE.match(name) or name in REPORT_BASENAMES:
            dest_dir, key = reports_dir(pdir), "reports"
        elif name in EXPORT_BASENAMES:
            dest_dir, key = exports_dir(pdir), "exports"
        else:
            continue
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest):
            continue
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(src, dest)
        moved[key] += 1
    return moved


def list_import_source(source_dir: str) -> list[dict]:
    """What is available to import right now, for the picker's checklist -
    name and a fast, untrusted point count (pts_file.quick_point_count, the header-only read the server uses too)."""
    from .pts_file import quick_point_count
    return [{"name": os.path.splitext(os.path.basename(p))[0], "points": quick_point_count(p)}
            for p in sorted(glob.glob(os.path.join(source_dir, "*.pts")))]


# -------------------------------------------------------------------------------------------------------------------
# Which placement backups a cleaner would remove. Pure decision logic with no filesystem access, so it can be
# dry-run, tested and reused by both the automatic and the on-demand tidy.
#
# Rule: always keep the newest backup and anything pinned. Otherwise keep every backup from the last hour, then the
# newest per hour for a day, then the newest per day after that.
# -------------------------------------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------------------------------------
# Tidy a project's saved placement versions. backups_to_remove decides; this reads pins and references and
# removes files. Never removes the newest, a pinned one, or one a report, job, export or the golden folder names.
# -------------------------------------------------------------------------------------------------------------------

PINS_FILE = "pins.json"
_NAME_IN_TEXT = re.compile(r"placement\.backup-\d{8}T\d{6}\.\d{3}Z(?:-\d+)?\.json")
_MAX_TEXT_BYTES = 5 * 1024 * 1024   # reference scan reads JSON/text files up to this size


def default_golden_dir() -> str:
    """OPENPOINT_GOLDEN_DIR if set, else the repo's own golden/ next to the package."""
    return os.environ.get("OPENPOINT_GOLDEN_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden")


def read_pins(pdir: str) -> set[str]:
    try:
        with open(os.path.join(backups_dir(pdir), PINS_FILE)) as fh:
            return {p for p in json.load(fh).get("pinned", []) if BACKUP_RE.match(p)}
    except (OSError, ValueError, AttributeError):
        return set()


def set_pinned(pdir: str, backup_id: str, pinned: bool) -> set[str]:
    """Pin or unpin one saved version; returns every pinned id."""
    pins = read_pins(pdir)
    (pins.add if pinned else pins.discard)(backup_id)
    write_json_atomic(os.path.join(backups_dir(pdir), PINS_FILE), {"pinned": sorted(pins)}, indent=1)
    return pins


def _names_in_tree(root: str) -> set[str]:
    """Backup names under `root`: file names, and names written inside small JSON/text files."""
    found: set[str] = set()
    if not os.path.isdir(root):
        return found
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if BACKUP_RE.match(name):
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
    for sub in (reports_dir(pdir), exports_dir(pdir), os.path.join(pdir, "jobs")):
        found |= _names_in_tree(sub)
    return found | _names_in_tree(golden_dir if golden_dir is not None else default_golden_dir())


def _parse_at(at: str) -> datetime.datetime:
    return datetime.datetime.strptime(at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=datetime.timezone.utc)


def plan_tidy(pdir: str, listing: list[dict], now: datetime.datetime, keep_all_hours: float, hourly_days: float,
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


def tidy_backups(pdir: str, listing: list[dict], resolve: Callable[[str], Optional[str]], now: datetime.datetime,
         keep_all_hours: float, hourly_days: float, golden_dir: Optional[str] = None) -> dict:
    """Remove what `plan_tidy` says. `resolve` maps an id to its file inside this project, or None."""
    summary = plan_tidy(pdir, listing, now, keep_all_hours, hourly_days, golden_dir)
    removed, freed = [], 0
    project_root = os.path.realpath(pdir)
    for backup_id in summary["remove_ids"]:
        if not BACKUP_RE.match(backup_id):   # only ever a saved version: never the hand log or anything else
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


# -------------------------------------------------------------------------------------------------------------------
# The hand-move log: one line per finished hand move, appended to a plain text file in the project folder.
# Local only.
# -------------------------------------------------------------------------------------------------------------------

HAND_MOVES_FILE = "hand-moves.jsonl"
HAND_MOVES_OLD_FILE = "hand-moves.old.jsonl"  # the previous file, kept once when the current one is full
HAND_MOVES_MAX_BYTES = 2_000_000       # about 5,000 moves; then it rolls over and the oldest file is replaced
HAND_MOVE_POSE_KEYS = ("x", "y", "z", "yaw", "pitch", "roll")
HAND_MOVE_KINDS = ("nudge", "undo", "redo", "revert")
HAND_MOVE_HOWS = ("drag", "arrow keys", "typed number", "nudge button", "reset to as-imported",
                  "revert to placement", "undo", "redo", "other")
HAND_MOVE_SOURCES = ("hand", "automatic run", "as imported", "unknown")


def _hand_pose(p) -> Optional[dict]:
    if not isinstance(p, dict) or not all(isinstance(p.get(k), (int, float)) for k in HAND_MOVE_POSE_KEYS):
        return None
    return {k: round(float(p[k]), 5) for k in HAND_MOVE_POSE_KEYS}


def clean_hand_move(move: dict) -> Optional[dict]:
    """One line as it will be written, or None when it is not a valid move. Unknown fields are dropped."""
    if not isinstance(move, dict) or not isinstance(move.get("scan"), str) or not move["scan"]:
        return None
    before, after = _hand_pose(move.get("before")), _hand_pose(move.get("after"))
    if before is None or after is None or before == after:
        return None
    src = move.get("before_source") if isinstance(move.get("before_source"), dict) else {}
    return {
        "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scan": move["scan"][:200],
        "kind": move.get("kind") if move.get("kind") in HAND_MOVE_KINDS else "nudge",
        "how": move.get("how") if move.get("how") in HAND_MOVE_HOWS else "other",
        "view": str(move.get("view", ""))[:40],
        "before": before, "after": after,
        "before_source": {"kind": src.get("kind") if src.get("kind") in HAND_MOVE_SOURCES else "unknown",
                          "run": str(src.get("run", ""))[:80]},
    }


def append_hand_moves(pdir: str, moves: list) -> int:
    """Append every valid move as one line; roll the file over first when it is full. Returns lines written."""
    lines = [m for m in (clean_hand_move(x) for x in (moves if isinstance(moves, list) else [])) if m is not None]
    if not lines:
        return 0
    path = os.path.join(pdir, HAND_MOVES_FILE)
    if os.path.exists(path) and os.path.getsize(path) >= HAND_MOVES_MAX_BYTES:
        os.replace(path, os.path.join(pdir, HAND_MOVES_OLD_FILE))
    with open(path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    return len(lines)


def read_hand_moves(pdir: str) -> list[dict]:
    """Every line, oldest first (the rolled-over file, then the current one); unreadable lines are skipped."""
    out = []
    for name in (HAND_MOVES_OLD_FILE, HAND_MOVES_FILE):
        path = os.path.join(pdir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    continue
    return out
