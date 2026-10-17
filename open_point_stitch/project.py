"""Projects. A project is a folder: scans-raw/ holds copies of imported .pts files (the source is never written
to) and placement.json is the project file."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import time

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
    name and a fast, untrusted point count (same header-only read
    server.py's own `_quick_point_count` uses for the same purpose)."""
    out = []
    for p in sorted(glob.glob(os.path.join(source_dir, "*.pts"))):
        name = os.path.splitext(os.path.basename(p))[0]
        points = 0
        try:
            with open(p, "r", errors="replace") as fh:
                first = fh.readline().split()
            if len(first) == 1:
                points = int(first[0])
        except (OSError, ValueError):
            pass
        out.append({"name": name, "points": points})
    return out
