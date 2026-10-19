"""The local web server behind the placement bench: serves the page and a JSON API.

Long jobs run in their own process; the page polls for progress. Binds 0.0.0.0 inside the container; what keeps
it off the network is publishing the port on 127.0.0.1 only.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import dataclasses
import datetime
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

try:
    import fcntl                     # the container is Linux; on Windows the thread lock alone is used
except ImportError:
    fcntl = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from open_point_stitch import __version__
from open_point_stitch.config import PipelineConfig
from open_point_stitch.jobs import JobStore, run_job_in_process
from open_point_stitch.align_anchors import align_from_anchors
from open_point_stitch.align_tilt import apply_suggestions, full_resolution_inputs, near_share, suggest_tilt, summarise
from open_point_stitch.overlap import nearest_distances
from open_point_stitch.merge import place_merge
from open_point_stitch.placement import world_boxes
from open_point_stitch.placement import (SCHEMA as PLACEMENT_SCHEMA, Anchor, AnchorMember, Exclusion, Placement,
                                    anchors_from_landmarks, is_placement_schema, load_placements,
                                    relative_transform)
from open_point_stitch.lib.files import write_json_atomic
from open_point_stitch.lib.poses import (ScanPose, carry_pose, common_motion, pose_shift_m, rotation_angle_deg, tilt_of,
                                        world_cloud_points, world_cloud_points_as_drawn, yaw_gap_deg, yaw_matrix2,
                                        yaw_matrix3, yaw_of, yaw_transform)
from open_point_stitch.export_plan import default_band, footprints_from_payloads, plan_cut, write_plan
from open_point_stitch.wall_checks import measure_walls, summarise as summarise_walls
from open_point_stitch.exclusions import filter_scans, for_export
from open_point_stitch.align_points import seed_key
from open_point_stitch.pair_filter import candidate_pairs
from open_point_stitch import pair_filter
from open_point_stitch import find_corners, find_openings, find_landmarks
from open_point_stitch.pipeline import run_pairwise
from open_point_stitch.align_graph import build_and_optimize
from open_point_stitch.prepare import preprocess_all_offloaded
from open_point_stitch.pts_file import load_folder
from open_point_stitch import align_gravity, limits
from open_point_stitch import project as project_mod
from open_point_stitch import find_floors_ceilings
from open_point_stitch import export_3d, find_walls
from open_point_stitch import explain
from open_point_stitch import wall_checks
from open_point_stitch import point_trust
import build_bench  # tools/build_bench.py: the one place a preview payload is built

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Build and version identity
# ---------------------------------------------------------------------------

def build_info() -> dict:
    return {
        "version": __version__,
        "build_time": os.environ.get("OPENPOINT_BUILD_TIME", "unknown (not a built image)"),
        "mode": "dev (mounted source)" if os.environ.get("OPENPOINT_DEV_MODE") else "release (baked in)",
    }


# ---------------------------------------------------------------------------
# job bookkeeping
# ---------------------------------------------------------------------------

# Payloads stay nested in memory; only the HTTP response flattens them (build_bench.flatten_xy_for_wire).


# ---------------------------------------------------------------------------
# placement file helpers: raw JSON on purpose, not via placement.Placement
# ---------------------------------------------------------------------------

def _placement_path(out_dir: str) -> str:
    return os.path.join(out_dir, "placement.json")


def _read_placement(out_dir: str):
    path = _placement_path(out_dir)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _stamped_path(folder: str, prefix: str, ext: str) -> str:
    """A free <prefix>-<UTC stamp>.<ms>Z[-n].<ext> path in folder, so two fast writes never share a name."""
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f".{int((now % 1) * 1000):03d}Z"
    path, n = os.path.join(folder, f"{prefix}-{stamp}.{ext}"), 1
    while os.path.exists(path):
        path, n = os.path.join(folder, f"{prefix}-{stamp}-{n}.{ext}"), n + 1
    return path


def _same_bytes(a: str, b: Optional[str]) -> bool:
    if not b or os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def _backup_placement(out_dir: str) -> Optional[str]:
    """Copy placement.json into backups/ before any overwrite; called from _write_placement so every writer is
    covered. None on a first save, or when the newest backup already holds these exact bytes."""
    path = _placement_path(out_dir)
    if not os.path.exists(path):
        return None
    newest = _list_backups(out_dir)[:1]
    if newest and _same_bytes(path, _backup_file_path(out_dir, newest[0]["id"])):
        return None
    folder = project_mod.backups_dir(out_dir)
    os.makedirs(folder, exist_ok=True)
    backup_path = _stamped_path(folder, "placement.backup", "json")
    shutil.copy2(path, backup_path)
    return backup_path


def _backup_folders(out_dir: str) -> list[str]:
    """backups/ first, then the project top level, where older projects kept them."""
    return [project_mod.backups_dir(out_dir), out_dir]


def _list_backups(out_dir: str) -> list[dict]:
    """Every backup for this project, newest first. The id is the file name; its stamp is when that save happened."""
    rows, seen, pins = [], set(), project_mod.read_pins(out_dir)
    for folder in _backup_folders(out_dir):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name in seen or not project_mod.BACKUP_RE.match(name):
                continue
            m = re.match(r"^placement\.backup-(\d{8}T\d{6})\.(\d{3})Z", name)
            try:
                at = time.strftime("%Y-%m-%dT%H:%M:%S", time.strptime(m.group(1), "%Y%m%dT%H%M%S")) + f".{m.group(2)}Z"
            except ValueError:
                continue
            seen.add(name)
            rows.append({"id": name, "at": at, "size": os.path.getsize(os.path.join(folder, name)),
                         "pinned": name in pins})
    rows.sort(key=lambda r: (r["at"], r["id"]), reverse=True)
    return rows


def _backup_file_path(out_dir: str, backup_id: str) -> Optional[str]:
    """The full path for a backup id from a URL, or None if it is not a real backup of THIS project.
    The id must match the exact backup filename shape, so it can never reach outside the project."""
    if not project_mod.BACKUP_RE.match(backup_id):
        return None
    for folder in _backup_folders(out_dir):
        path = os.path.join(folder, backup_id)
        if os.path.isfile(path):
            return path
    return None


def _tidy_rules(settings: Optional[dict]) -> tuple[float, float]:
    s = settings or {}
    return float(s.get("backup_keep_all_hours", 1.0)), float(s.get("backup_hourly_days", 1.0))


def _tidy_backups(out_dir: str, dry_run: bool, settings: Optional[dict] = None) -> dict:
    """Plan (dry run) or carry out a tidy of saved versions by the project's rules."""
    if settings is None:
        settings = (_read_placement(out_dir) or {}).get("settings", {})
    keep_all, hourly = _tidy_rules(settings)
    now = datetime.datetime.now(datetime.timezone.utc)
    listing = _list_backups(out_dir)
    got = (project_mod.plan_tidy(out_dir, listing, now, keep_all, hourly) if dry_run else
           project_mod.tidy_backups(out_dir, listing, lambda i: _backup_file_path(out_dir, i), now, keep_all, hourly))
    return dict(got, keep_all_hours=keep_all, hourly_days=hourly)


def _auto_tidy_backups(out_dir: str, settings: Optional[dict] = None) -> None:
    """The automatic tidy after a save or on open, when the setting is on. Never raises into its caller."""
    try:
        if settings is None:
            settings = (_read_placement(out_dir) or {}).get("settings", {})
        if not settings.get("backup_auto_tidy", True):
            return
        got = _tidy_backups(out_dir, dry_run=False, settings=settings)
        if got["n_removed"]:
            log.info("tidied %d older saved versions (%.1f MB) in %s", got["n_removed"],
                     got["bytes_removed"] / 1048576, out_dir)
    except Exception:  # noqa: BLE001
        log.exception("automatic tidy of saved versions failed in %s (nothing else affected)", out_dir)


def _write_timestamped_report(out_dir: str, prefix: str, result: dict) -> str:
    """Write a job's full result to a timestamped file in reports/."""
    folder = project_mod.reports_dir(out_dir)
    os.makedirs(folder, exist_ok=True)
    path = _stamped_path(folder, prefix, "json")
    write_json_atomic(path, result, indent=2)
    log.info("wrote %s report to %s", prefix, path)
    return path


def _export_path(out_dir: str, stem: str, ext: str) -> str:
    """A fresh timestamped file in exports/, so a new export never overwrites an earlier one."""
    folder = project_mod.exports_dir(out_dir)
    os.makedirs(folder, exist_ok=True)
    return _stamped_path(folder, stem, ext)


def _note_latest_export(out_dir: str, **paths: str) -> None:
    """Record the newest export of each kind in exports/latest.json (a pointer, not a copy)."""
    path = os.path.join(project_mod.exports_dir(out_dir), "latest.json")
    latest = {}
    if os.path.exists(path):
        with open(path) as fh:
            latest = json.load(fh)
    latest.update({k: os.path.basename(v) for k, v in paths.items() if v})
    write_json_atomic(path, latest, indent=1)


def _write_align_auto_report(out_dir: str, result: dict) -> str:
    return _write_timestamped_report(out_dir, "align_auto_pairs", result)


def _write_align_anchors_report(out_dir: str, result: dict) -> str:
    """Write the per-pair report of an align-from-links job. A different file prefix from automatic-align reports,
    since they are a different job kind."""
    return _write_timestamped_report(out_dir, "align_anchors_pairs", result)


def _stamp_align(data: dict, kind: str) -> None:
    """Stamp the placement with the run kind ('align' or 'align_auto'), time, and how many scans moved."""
    n_moved = sum(1 for s in data.get("scans", []) if s.get("align_moved_m", 0) > 0.001)
    data["align_stamp"] = {
        "kind": kind,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_moved": n_moved,
        "n_scans": len(data.get("scans", [])),
    }


def _stamp_last_run(data: dict, kind: str, summary: dict) -> None:
    """Write the short last-run summary into the project file so the bench can show it on open."""
    data["last_run"] = {"kind": kind, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        **summary}


def _record_as_imported(data: dict, scan: dict) -> None:
    """Record the pose a scan had when first built (gravity levelled only), once and never overwritten. x/y/yaw
    are the scan's current grid position, pitch/roll zero."""
    table = data.setdefault("as_imported", {})
    table.setdefault(scan["name"], {
        "yaw_deg": float(scan.get("yaw_deg", 0.0)), "pitch_deg": 0.0, "roll_deg": 0.0,
        "x_m": float(scan.get("x_m", 0.0)), "y_m": float(scan.get("y_m", 0.0)),
        "z_m": float(scan.get("z_m", 0.0)),
    })


def _apply_process_settings(settings: dict) -> None:
    """Make the project's levelling tilt limit the process default, so previews and alignments use it."""
    try:
        tilt = float(settings.get("level_tilt_deg", 15.0))
        if tilt != 0:                       # 0: no scan is levelled to its floor; previews handle that themselves
            align_gravity.set_default_max_tilt(tilt)
    except (TypeError, ValueError):
        log.warning("ignoring an unusable levelling tilt setting: %r", settings.get("level_tilt_deg"))


def _attach_openings(app: "App", features, scan, prepared) -> None:
    """Attach doors and windows to one scan's features, found in the levelled frame the bench shows. A scan without
    a built preview gets none and does not vote."""
    with app._lock:
        payload = app.payloads.get(scan.name)
    if payload is None:
        return
    find_openings.build_scan_openings(features, scan.points, prepared.down, payload["level_rotation"], float(payload["floor_z"]))


def _attach_landmarks(app: "App", features, scan, prepared) -> None:
    """Attach landmarks (furniture and fixtures; organic shapes already rejected) to one scan's features, in the
    same levelled frame as _attach_openings."""
    with app._lock:
        payload = app.payloads.get(scan.name)
    if payload is None:
        return
    find_landmarks.build_scan_landmarks(features, prepared.down, payload["level_rotation"], float(payload["floor_z"]))


def _compute_deviation(app: "App") -> dict:
    """Deviation heatmap: distance from each preview point to the nearest point of any other scan, in the world
    frame, from the saved placement."""
    data = _read_placement(app.out_dir)
    if data is None:
        return {"ready": False, "reason": "no saved placement yet"}

    poses: dict[str, ScanPose] = {}
    for s in data.get("scans", []):
        poses[s["name"]] = ScanPose.from_dict(s)

    with app._lock:
        payloads = {n: p for n, p in app.payloads.items() if n in poses}

    names = list(payloads.keys())
    if len(names) < 2:
        return {"ready": False, "reason": "need at least two placed, built scans"}

    # World-frame preview points per scan, hand tilt included, so the heatmap matches what the bench draws.
    world = {n: world_cloud_points_as_drawn(payloads[n], poses[n]) for n in names}

    deviation_m: dict[str, list] = {}
    for n in names:
        others = [world[m] for m in names if m != n]
        target = np.concatenate(others, axis=0) if others else np.zeros((0, 3))
        dist = nearest_distances(world[n], target)
        deviation_m[n] = [None if not np.isfinite(v) else round(float(v), 3) for v in dist]

    return {"ready": True, "deviation_m": deviation_m}


def _tilt_inputs(app: "App", payloads: dict) -> dict:
    """Every point of each scan, not the thinned preview, loaded once per open project. The tilt solve needs full
    resolution."""
    with app._tilt_lock:
        missing = [n for n in payloads if n not in app._tilt_inputs]
        if missing:
            app._tilt_inputs.update(full_resolution_inputs(app.scans_dir, payloads, missing))
        return {n: app._tilt_inputs[n] for n in payloads if n in app._tilt_inputs}


def _compute_tilt_suggestion(app: "App", body: dict) -> dict:
    """Suggested height, pitch and roll per scan from floor and ceiling agreement, starting from the poses the page
    holds now (body["scans"]) or else the saved placement. Writes nothing."""
    given = body.get("scans")
    if not given:
        data = _read_placement(app.out_dir)
        given = (data or {}).get("scans", [])
    poses: dict[str, ScanPose] = {}
    locked: dict[str, bool] = {}
    for s in given:
        poses[s["name"]] = ScanPose.from_dict(s)
        locked[s["name"]] = bool(s.get("locked", False))

    with app._lock:
        payloads = {n: p for n, p in app.payloads.items() if n in poses}
        failed = [n for n in poses if n in app.build_errors]
    waiting = [n for n in poses if n not in payloads and n not in failed]
    if waiting:
        return {"ready": False, "reason": f"{len(waiting)} scan preview(s) are still building; try again in a moment"}
    if len(payloads) < 2:
        return {"ready": False, "reason": "need at least two placed, built scans"}

    picked = body.get("scan_names")
    result = suggest_tilt(_tilt_inputs(app, payloads), poses, locked=locked, subset=set(picked) if picked else None)
    near_before = near_share(payloads, poses)
    near_after = near_share(payloads, apply_suggestions(poses, result))
    return {
        "ready": True,
        "suggestions": [vars(s) for s in result.suggestions.values()],
        "skipped": result.skipped,
        "agreement": summarise(result.before, result.after),
        "near_5cm": {"mean_before": round(float(np.mean(list(near_before.values()))), 4),
                     "mean_after": round(float(np.mean(list(near_after.values()))), 4),
                     "per_scan": {n: [round(near_before[n], 4), round(near_after[n], 4)] for n in near_before}},
    }


# One module-level lock serialises every placement writer, so _write_placement stays a plain function.
_placement_write_lock = threading.Lock()


@contextlib.contextmanager
def _placement_lock(out_dir: str):
    """_placement_write_lock plus a file lock in the project folder, because jobs write from their own process."""
    with _placement_write_lock:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, ".placement.lock"), "a") as fh:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl:
                    fcntl.flock(fh, fcntl.LOCK_UN)


def _write_placement(out_dir: str, data: dict) -> None:
    """Locked, atomic (temp file plus os.replace) overwrite of the whole placement, for Save, Clear links and
    Revert. Alignment results must use _write_placement_merged instead."""
    path = _placement_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with _placement_lock(out_dir):
        _write_placement_locked(out_dir, data)


_REGION_KEYS = ("points", "centroid", "extent")


def _round_floats(v, places: int = 4):
    if isinstance(v, float):
        return round(v, places)
    if isinstance(v, (list, tuple)):
        return [_round_floats(x, places) for x in v]
    return v


def _for_disk(data: dict) -> dict:
    """A copy with link and exclusion coordinates rounded to 0.1 mm (hand-drawn boxes, never scan points)."""
    out = dict(data)
    for key in ("anchors", "exclusions", "landmarks"):
        if not isinstance(data.get(key), list):
            continue
        items = []
        for item in data[key]:
            item = dict(item)
            if "members" in item:
                item["members"] = [{k: (_round_floats(v) if k in _REGION_KEYS else v) for k, v in m.items()}
                                   for m in item["members"]]
            for k in _REGION_KEYS:
                if k in item:
                    item[k] = _round_floats(item[k])
            items.append(item)
        out[key] = items
    return out


def _write_placement_locked(out_dir: str, data: dict, backup_first: bool = True) -> None:
    """The backup and atomic-replace write. The caller must already hold _placement_write_lock, so
    _write_placement_merged can read, modify and write in one critical section."""
    path = _placement_path(out_dir)
    backup = _backup_placement(out_dir) if backup_first else None
    if backup:
        log.info("backed up the previous placement to %s before overwriting it", backup)
    write_json_atomic(path, _for_disk(data), separators=(",", ":"))     # compact
    if backup:
        _auto_tidy_backups(out_dir, data.get("settings", {}))


def _write_placement_merged(out_dir: str, apply, backup_first: bool = True) -> dict:
    """Re-read the placement under the write lock, change it with apply(fresh) and write it back as one step, so
    an alignment landing cannot undo a Save made meanwhile. Returns the data written. backup_first=False is only for
    bookkeeping that changes no layout (the open stamp)."""
    path = _placement_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with _placement_lock(out_dir):
        fresh = None
        if os.path.exists(path):
            with open(path) as fh:
                fresh = json.load(fh)
        if fresh is None:
            fresh = {}
        apply(fresh)
        _write_placement_locked(out_dir, fresh, backup_first)
    return fresh


_SERVER_OWNED_KEYS = ("settings", "last_run", "excluded_pairs")   # the page never sends these


def _project_excluded_pairs(data: dict) -> frozenset:
    """The scan pairs this project never tries to match, from its file's "excluded_pairs" list of [name, name]."""
    return frozenset(frozenset(pair) for pair in data.get("excluded_pairs", []) if len(set(pair)) == 2)


def _write_whole_placement(out_dir: str, data: dict) -> dict:
    """A Save or a restore: data replaces the layout, but the settings and last run already on disk are kept,
    since the page never sends them."""
    def _apply(fresh: dict) -> None:
        kept = {k: fresh[k] for k in _SERVER_OWNED_KEYS if k in fresh}
        fresh.clear()
        fresh.update(data)
        fresh.update(kept)
    return _write_placement_merged(out_dir, _apply)


def _normalise_incoming_placement(data: dict) -> dict:
    """Fill in human_* on first save, leave it alone after; the backstop for a fresh page or an
    old bench build that never sends it."""
    for s in data.get("scans", []):
        if "human_yaw_deg" not in s:
            s["human_yaw_deg"] = s.get("yaw_deg", 0.0)
            s["human_x_m"] = s.get("x_m", 0.0)
            s["human_y_m"] = s.get("y_m", 0.0)
            s["human_z_m"] = s.get("z_m", 0.0)
        s.setdefault("align_moved_m", 0.0)
        s.setdefault("align_moved_deg", 0.0)
        # Older files and pages without these load at zero hand-set tilt.
        s.setdefault("pitch_deg", 0.0)
        s.setdefault("roll_deg", 0.0)
        # as_imported is filled in on Save, never from _build_one.
        _record_as_imported(data, s)
    return data


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def _thread_cap(params: dict):
    """A one-shot thread cap for one stage when the page asked for one; otherwise a context that does nothing."""
    threads = params.get("threads")
    return limits.limited_to(int(threads)) if threads else contextlib.nullcontext()


def _finish_cancelled(app: "App", job_id: str, what: str, exc: BaseException) -> None:
    log.info("%s cancelled: %s", what, exc)
    app.jobs.finish_cancelled(job_id, {"cancelled": True, "reason": str(exc)})


def _job(failed: str, needs: Optional[str] = None):
    """The frame every job function shares: when `needs` names the action, a saved placement must exist first; any
    error reaches the job, not the log alone. Cancelling stays inside each job, where it knows what is written."""
    def wrap(fn):
        @functools.wraps(fn)
        def run(app: "App", job_id: str, params: dict) -> None:
            try:
                if needs and not os.path.exists(_placement_path(app.out_dir)):
                    app.jobs.fail(job_id, f"No saved placement yet. Save a placement before {needs}.")
                    return
                fn(app, job_id, params)
            except Exception as exc:  # noqa: BLE001 - job errors must reach the job, not the log alone
                log.exception(failed)
                app.jobs.fail(job_id, f"{type(exc).__name__}: {exc}")
        return run
    return wrap


def _anchors_from_placement(data: dict) -> list:
    """The saved anchors as Anchor objects, exactly as stored (raw local frame). Shared by align-from-links and the
    links stage of a combined run, so the two never drift apart."""
    return [
        Anchor(
            id=a.get("id", f"anchor-{i}"),
            label=a.get("label", ""),
            members=[
                AnchorMember(
                    scan=m["scan"],
                    points=[tuple(pt) for pt in m.get("points", [])],
                    centroid=tuple(m["centroid"]),
                    extent=tuple(m.get("extent", (0.0, 0.0, 0.0))),
                    n_points=int(m.get("n_points", len(m.get("points", [])))),
                )
                for m in a.get("members", [])
            ],
        )
        for i, a in enumerate(data.get("anchors", []))
    ] + anchors_from_landmarks(data.get("landmarks", []))     # same name in two scans


@_job("align job failed")
def _run_align(app: "App", job_id: str, params: dict) -> None:
    data = _read_placement(app.out_dir)
    if data is None:
        app.jobs.fail(job_id, "No saved placement yet. Save a placement before aligning.")
        return

    # Previews build in the background, so wait for them before aligning.
    app.jobs.set_progress(job_id, "waiting for scan previews", "")
    app.wait_for_all()
    if app.build_errors:
        log.warning("align: %d scan(s) never got a preview and will be "
                    "treated as unknown: %s", len(app.build_errors),
                    list(app.build_errors))

    subset = params.get("scan_names")
    subset_set = set(subset) if subset else None
    gpu = bool(params.get("gpu", False))  # accepted for interface completeness;
    # align runs on the browser-preview point clouds, a few thousand
    # points per scan, which is not the workload the GPU tensor path
    # (open_point_stitch/align_gpu.py) was built for. Not wired up.
    max_shift = float(params.get("max_shift_m", 1.0))

    poses = {}
    locked = {}
    for s in data["scans"]:
        poses[s["name"]] = ScanPose.from_dict(s, strict=True)
        locked[s["name"]] = bool(s.get("locked", False))

    anchors = _anchors_from_placement(data)
    if not anchors:
        app.jobs.fail(job_id, "No anchors in the saved placement. Draw anchors with the Anchor tool, or name "
                                "the same landmark in two scans in review, for the scans you want aligned.")
        return

    app.jobs.set_progress(job_id, "aligning", f"0/{len(anchors)} anchor(s)")
    cap = _thread_cap(params)

    def progress_cb(done: int, total: int) -> None:
        app.jobs.set_progress(job_id, "aligning", f"{done}/{total} anchor(s)")

    with cap:
        result = align_from_anchors(app.payloads, poses, anchors, locked, subset_set,
                                    scans_folder=app.scans_dir, max_shift_m=max_shift,
                                    reference_scan=params.get("reference_scan") or data.get("reference_scan"),
                                    progress_cb=progress_cb,
                                    should_cancel=lambda: app.jobs.is_cancel_requested(job_id))

    if result.cancelled:
        # Cancelled: no pose written; whatever ran is kept only in the report.
        report = {"rows": [], "skipped_anchors": result.skipped, "cancelled": True,
                  "n_anchors": len(anchors), "n_anchors_completed": result.n_anchors_completed}
        _write_align_anchors_report(app.out_dir, report)
        app.jobs.finish_cancelled(job_id, report)
        return

    by_name = {s["name"]: s for s in data["scans"]}
    pose_updates: dict[str, dict] = {}
    for name, pose in result.poses.items():
        s = by_name[name]
        update = {
            "yaw_deg": round(pose.yaw_deg, 4),
            "x_m": round(pose.x_m, 5),
            "y_m": round(pose.y_m, 5),
            "z_m": round(pose.z_m, 5),
            # align_from_anchors carries pitch/roll forward unchanged; written explicitly anyway.
            "pitch_deg": round(pose.pitch_deg, 4),
            "roll_deg": round(pose.roll_deg, 4),
        }
        human = ScanPose.from_dict(s, strict=True, prefix="human_")
        if name in app.payloads:
            update["align_moved_m"] = round(pose_shift_m(app.payloads[name], human, pose), 4)
            update["align_moved_deg"] = round(yaw_gap_deg(pose.yaw_deg, s["human_yaw_deg"]), 3)
        # else: this scan's preview never built, so align_from_anchors
        # could not have moved it either (it can only move a scan whose
        # payload it has); leave whatever align_moved_* it already had.
        pose_updates[name] = update

    def _apply(fresh: dict) -> None:
        # Merge these pose updates onto what is on disk now, not the snapshot read at job start.
        fresh_by_name = {s["name"]: s for s in fresh.get("scans", [])}
        for name, update in pose_updates.items():
            s = fresh_by_name.get(name)
            if s is None:
                continue  # scan no longer in the placement
            s.update(update)
        _stamp_align(fresh, "align")
        _stamp_last_run(fresh, "align", {
            "n_anchors": len(anchors), "n_moved": len(pose_updates),
            "n_skipped_anchors": len(result.skipped),
        })

    app.jobs.commit(job_id)                     # writing results from here; Cancel waits
    _write_placement_merged(app.out_dir, _apply)

    report = {"rows": [vars(r) for r in result.rows], "skipped_anchors": result.skipped,
              "n_anchors": len(anchors), "n_anchors_completed": result.n_anchors_completed}
    _write_align_anchors_report(app.out_dir, report)
    app.jobs.finish(job_id, report)


def _decompose_pose(world_matrix: np.ndarray, level_rotation: np.ndarray,
                    floor_z: float = 0.0) -> tuple[float, float, float, float, float]:
    """Inverse of Placement.matrix(): recover yaw_deg, x, y, z and the residual tilt (degrees) left after removing
    the scan's own levelling. floor_z is the scan's floor level, which matrix() takes off before rotating."""
    level_rotation = np.asarray(level_rotation, dtype=float)
    r_yaw_only = world_matrix[:3, :3] @ level_rotation.T
    yaw_deg = yaw_of(r_yaw_only)
    tilt_deg = tilt_of(r_yaw_only)
    t = world_matrix[:3, 3] + r_yaw_only @ np.array([0.0, 0.0, floor_z])
    return yaw_deg % 360.0, float(t[0]), float(t[1]), float(t[2]), tilt_deg


def _centre_shift_for_dropped_tilt(world: np.ndarray, kept: np.ndarray,
                                   centre_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """(shift, dropped tilt in degrees): the move that puts the scan's centre where world put it, once its tilt is
    replaced by the hand-set one in kept."""
    c = np.asarray(centre_raw, dtype=float)
    shift = (world[:3, :3] @ c + world[:3, 3]) - (kept[:3, :3] @ c + kept[:3, 3])
    turn = world[:3, :3] @ kept[:3, :3].T
    dropped = rotation_angle_deg(turn)
    return shift, dropped


def _yaw_and_translation_only(transform: np.ndarray) -> np.ndarray:
    """Keep only the yaw and translation of a 4x4 rigid transform, so the root's tilt is not applied to every scan."""
    return yaw_transform(yaw_of(transform), transform[:3, 3])


def _pick_root(largest: list[int], names: list[str], reference: Optional[str]) -> int:
    """The scan a run is pinned to: the reference scan when it is in the joined-up group, else that group's first."""
    if reference in names and names.index(reference) in largest:
        return names.index(reference)
    return largest[0]


def _layout_bound_check(seed_shifts_m: dict[str, float], bound_mean_m: float) -> tuple[bool, float]:
    """Whole-layout bound: the mean seed-relative shift of every candidate scan this run, and whether it clears
    bound_mean_m. Zeros count."""
    if not seed_shifts_m:
        return False, 0.0
    mean_shift_m = sum(seed_shifts_m.values()) / len(seed_shifts_m)
    return mean_shift_m > bound_mean_m, mean_shift_m


@dataclasses.dataclass
class MoveCheck:
    """One scan's proposed move in a seeded run. anchor: the scan the run is pinned to; locked scans never move."""
    name: str
    start: ScanPose
    proposed: ScanPose
    payload: dict
    anchor: bool = False
    locked: bool = False
    fitness: Optional[float] = None      # best accepted pair fit touching this scan, for the report


@dataclasses.dataclass
class LimitOutcome:
    final: dict                          # name -> ScanPose to keep
    capped: set
    layout_tripped: bool
    mean_shift_m: float
    shifts: dict                         # name -> (m, deg) as judged; 0 for a refused scan; locked scans absent
    rejected: list                       # one row per correction a limit threw away
    common: Optional[tuple] = None       # relative only: the whole layout's (yaw deg, [dx, dy, dz])


def _limit_moves(checks: list[MoveCheck], bound_m: float, bound_deg: float, layout_bound_m: float,
                 relative: bool) -> LimitOutcome:
    """Per-scan and whole-layout limits. relative=True first removes the turn and shift all scans made together; a
    refused scan is then carried with the layout, and a fixed scan out of step declines the run."""
    measured = "against the other scans" if relative else "from where it started"
    final = {c.name: c.proposed for c in checks}
    shifts, capped, rejected = {}, set(), []

    def refuse(c: MoveCheck, limit: str, m: float, d: float, lim_m: float, lim_deg: Optional[float]) -> None:
        rejected.append({"scan": c.name, "limit": limit, "measured": measured, "correction_m": m, "correction_deg": d,
                         "limit_m": lim_m, "limit_deg": lim_deg, "fitness": c.fitness})

    common = None
    if relative:
        before = {c.name: world_cloud_points(c.payload, c.start) for c in checks}
        after = {c.name: world_cloud_points(c.payload, c.proposed) for c in checks}
        names = [c.name for c in checks]
        fit_on = names
        yaw, shift = common_motion([before[n] for n in fit_on], [after[n] for n in fit_on])
        for _ in range(3):              # refit without the scans moving against the rest, so they cannot drag it
            r = yaw_matrix3(yaw)
            rel = {n: float(np.linalg.norm(before[n] @ r.T + shift - after[n], axis=1).mean()) for n in names}
            inliers = [n for n in names if rel[n] <= bound_m]
            if len(inliers) < 2 or inliers == fit_on:
                break
            fit_on = inliers
            yaw, shift = common_motion([before[n] for n in fit_on], [after[n] for n in fit_on])
        r = yaw_matrix3(yaw)
        common = (round(float(yaw), 4), [round(float(v), 4) for v in shift])
        judged = {c.name: (round(float(np.linalg.norm(before[c.name] @ r.T + shift - after[c.name], axis=1).mean()), 4),
                           round(yaw_gap_deg(c.proposed.yaw_deg, c.start.yaw_deg + yaw), 3)) for c in checks}
    else:
        judged = {c.name: (round(pose_shift_m(c.payload, c.start, c.proposed), 4),
                           round(yaw_gap_deg(c.proposed.yaw_deg, c.start.yaw_deg), 3)) for c in checks if not c.locked}

    fixed_out = False
    for c in checks:
        if c.name not in judged:
            continue
        m, d = judged[c.name]
        if not c.locked:
            shifts[c.name] = (m, d)
        if m <= bound_m and d <= bound_deg:
            continue
        if relative and (c.anchor or c.locked):
            refuse(c, "fixed scan", m, d, bound_m, bound_deg)
            fixed_out = True
            continue
        if c.locked:
            continue
        refuse(c, "per scan", m, d, bound_m, bound_deg)
        capped.add(c.name)
        shifts[c.name] = (0.0, 0.0)
        final[c.name] = carry_pose(c.start, common[0], np.asarray(common[1])) if relative else c.start

    tripped, mean_m = _layout_bound_check({n: s[0] for n, s in shifts.items()}, layout_bound_m)
    if tripped or fixed_out:
        for c in checks:
            moved = c.name in shifts and shifts[c.name][0] != 0.0
            if tripped and moved:
                refuse(c, "whole layout", shifts[c.name][0], shifts[c.name][1], layout_bound_m, None)
            if moved or relative:       # relative: a refused scan was carried with the layout, so it goes back too
                final[c.name] = c.start
    return LimitOutcome(final=final, capped=capped, layout_tripped=tripped or fixed_out, mean_shift_m=mean_m,
                        shifts=shifts, rejected=rejected, common=common)


_SEARCH_KNOBS = {"local_search_n": int, "local_search_radius_m": float, "local_search_yaw_deg": float,
                 "local_search_height_m": float, "local_only_bound_m": float, "local_only_bound_deg": float,
                 "min_improvement_frac": float, "scan_bound_m": float, "scan_bound_deg": float,
                 "proposal_gate_m": float, "proposal_gate_deg": float}


def _final_placements(placements: dict, pose_updates: dict) -> dict[str, Placement]:
    """Each scan's placement with this run's pose updates applied."""
    out = {}
    for n, p in placements.items():
        u = pose_updates.get(n, {})
        out[n] = Placement(name=n, level_rotation=p.level_rotation, floor_z=p.floor_z,
                           yaw_deg=u.get("yaw_deg", p.yaw_deg), tx=u.get("x_m", p.tx), ty=u.get("y_m", p.ty),
                           tz=u.get("z_m", p.tz), pitch_deg=p.pitch_deg, roll_deg=p.roll_deg)
    return out


def _reprocessed_exclusions(saved: dict, name: str, key: str = "exclusions") -> Optional[list]:
    """The exclusions (or, with key="stairs", the marked stairs) a scan was last re-processed with (its
    `reprocessed_with` ids), or None for a plain build."""
    ids = next((s.get("reprocessed_with") for s in saved.get("scans", []) if s.get("name") == name), None)
    if not ids:
        return None
    return [Exclusion.from_dict(e) for e in saved.get(key, []) if e.get("scan") == name and e["id"] in ids] or None


def _preview_options_for(saved: dict) -> dict:
    """The preview options a project file asks for: its settings plus the scans that keep the scanner's level."""
    return build_bench.preview_options(saved.get("settings", {}),
                                       [s["name"] for s in saved.get("scans", []) if s.get("keep_scanner_level")])


def _freeze_wall_edits(out_dir: str, saved: dict, paths: dict, cache_dir: str) -> dict:
    """Freeze hand wall edits made on an older wall finder's walls (wall_checks.frozen) before the preview is rebuilt,
    so the user's walls come back exactly. Returns the placement as written, or saved unchanged."""
    options = _preview_options_for(saved)
    frozen = {}
    for s in saved.get("scans", []):
        edits = s.get("wall_edits") or []
        if not edits or edits[0].get("op") == "set" or s["name"] not in paths:
            continue
        _, cpath = build_bench._cache_key_and_path(paths[s["name"]], cache_dir, _reprocessed_exclusions(saved, s["name"]),
                                                   _reprocessed_exclusions(saved, s["name"], "stairs"), options)
        try:
            with open(cpath) as fh:
                cached = json.load(fh)
        except (OSError, ValueError):
            continue
        if cached.get("_cache_key", {}).get("version", 0) >= build_bench.WALL_FINDER_SINCE:
            continue
        frozen[s["name"]] = (edits, wall_checks.frozen(cached["payload"].get("walls", []), edits,
                                                      f"wall finder changed in {__version__}"))
    if not frozen:
        return saved

    def _apply(fresh: dict) -> None:
        for s in fresh.get("scans", []):
            if s["name"] in frozen and s.get("wall_edits") == frozen[s["name"]][0]:
                s["wall_edits"] = frozen[s["name"]][1]
    log.info("kept the hand-corrected walls of %d scan(s) as they were, before the new wall finder runs", len(frozen))
    return _write_placement_merged(out_dir, _apply)


def _describe_reprocess(before: dict, after: dict) -> dict:
    """What re-processing changed, as numbers and as plain sentences."""
    b_walls, a_walls = before.get("walls", []), after.get("walls", [])
    gone = [w for w in b_walls if wall_checks.find_wall(a_walls, [w["x1"], w["y1"], w["x2"], w["y2"]]) is None]
    new = [w for w in a_walls if wall_checks.find_wall(b_walls, [w["x1"], w["y1"], w["x2"], w["y2"]]) is None]
    out = {"points_before": before.get("points"), "points_after": after.get("points"),
           "walls_before": len(b_walls), "walls_after": len(a_walls), "walls_gone": len(gone), "walls_new": len(new),
           "floor_moved_cm": round((after.get("floor_z", 0.0) - before.get("floor_z", 0.0)) * 100, 1),
           "tilt_before_deg": before.get("tilt"), "tilt_after_deg": after.get("tilt")}
    lines = []
    if out["points_before"] and out["points_after"] is not None:
        lines.append(f"{out['points_after']:,} of {out['points_before']:,} points kept "
                     f"({out['points_before'] - out['points_after']:,} left out).")
    def lengths(ws: list) -> str:
        return ", ".join(f"{w.get('len', 0):.1f} m long" for w in ws)
    wall_line = f"walls: {len(b_walls)} before, {len(a_walls)} now"
    if gone:
        wall_line += f"; {len(gone)} no longer found ({lengths(gone)})"
    if new:
        wall_line += f"; {len(new)} newly found ({lengths(new)})"
    lines.append(wall_line + ".")
    if abs(out["floor_moved_cm"]) >= 0.5:
        lines.append(f"the floor it found moved {abs(out['floor_moved_cm']):.0f} cm "
                     f"{'up' if out['floor_moved_cm'] > 0 else 'down'}.")
    if out["tilt_before_deg"] is not None and out["tilt_after_deg"] is not None \
            and abs(out["tilt_after_deg"] - out["tilt_before_deg"]) >= 0.05:
        lines.append(f"its levelling changed from {out['tilt_before_deg']:.2f} to {out['tilt_after_deg']:.2f} degrees.")
    out["lines"] = lines
    return out


class _CountedReader:
    """A request body that remembers how much has been read, so the rest can be drained after a refusal."""
    def __init__(self, f):
        self.f, self.n = f, 0

    def read(self, k: int) -> bytes:
        b = self.f.read(k)
        self.n += len(b)
        return b

    def drain(self, length: int) -> None:
        while self.n < length and self.read(min(length - self.n, 1 << 20)):
            pass


def _bent_scan_report(scans: list, placements: dict[str, Placement], payloads: dict) -> dict:
    """The bent-scan check on placed full-resolution scans. Reports only; a failure here never fails the job."""
    try:
        world = {}
        for s in scans:
            p = placements.get(s.name)
            if p is not None:
                m = p.matrix()
                world[s.name] = (s.points @ m[:3, :3].T + m[:3, 3]).astype(np.float32)
        info: dict = {}
        verdicts = wall_checks.find_bent_scans(world, placements, payloads, info=info)
        return wall_checks.as_report(verdicts, info)
    except Exception as exc:  # noqa: BLE001 - a report section must not cost the user their run
        log.exception("bent-scan check failed")
        return {"checked": 0, "flags": [], "error": f"{type(exc).__name__}: {exc}"}


def _explain_run_moves(placements: dict, pose_updates: dict, prepared: list, results: list, guard_on: bool) -> dict:
    """Moves, link fits and (with the guard) the scans kept at their starting pose, for the run report."""
    start = {n: p.matrix() for n, p in placements.items()}
    final = {n: p.matrix() for n, p in _final_placements(placements, pose_updates).items()}
    points = {pr.name: np.asarray(pr.down.points) for pr in prepared if pr.name in placements}
    outcomes = [r.as_row() for r in results]
    locked = {n for n, p in placements.items() if p.locked}
    if guard_on:
        _, moves, links, offered = explain.guard(points, start, final, outcomes, locked)
    else:
        links = explain.measure_links(points, start, final, outcomes)
        moves, offered = explain.explain(points, start, final, links, locked), {}
    return {"moves": [{"scan": m.scan, "text": m.text, "moved_m": round(m.moved_m, 4), "turned_deg": round(m.turned_deg, 3),
                       "pulled_by": m.pulled_by, "held_by": m.held_by, "toward": m.toward,
                       "toward_is_stronger": m.toward_is_stronger, "guarded": m.guarded,
                       "hurt": [[o, round(b, 3), round(a, 3)] for o, b, a in m.hurt],
                       "helped": [[o, round(b, 3), round(a, 3)] for o, b, a in m.helped]} for m in moves],
            "summary": {**explain.summary(moves, links), "guard_on": guard_on},
            "kept_back": sorted(offered),
            "suggestions": {n: {k: pose_updates[n][k] for k in ("yaw_deg", "x_m", "y_m", "z_m")} for n in offered}}


def _run_align_auto(app: "App", job_id: str, params: dict) -> None:
    """Automatic align: solve from full-resolution geometry with the same registration as the stitch CLI, then
    carry the result so the root scan keeps its placed pose. With use_anchors set, a second stage runs
    align_from_anchors on this stage's result before anything is written, as one combined run."""
    if os.environ.get("OPENPOINT_DUMP_CANDIDATES"):          # measurement only: label recorded candidates by run
        os.environ["OPENPOINT_DUMP_RUN"] = job_id
    try:
        data = _read_placement(app.out_dir)
        if data is None:
            app.jobs.fail(job_id, "No saved placement yet. Save a placement before aligning.")
            return
        # Re-check one scan's links: only pairs with this scan; every other scan held for this run only.
        focus_scan = params.get("focus_scan")
        if focus_scan:
            for s in data["scans"]:
                if s["name"] != focus_scan:
                    s["locked"] = True

        placements: dict[str, Placement] = {}
        for s in data["scans"]:
            placements[s["name"]] = Placement.from_dict(s, strict=True)   # hand tilt kept for the root's target

        # The exclusion list the bench draws, parsed the same way as load_placements.
        exclusions = [Exclusion.from_dict(e) for e in data.get("exclusions", [])]

        # For the "moved from where you put it" diagnostic only - see below.
        app.jobs.set_progress(job_id, "waiting for scan previews", "")
        app.wait_for_all()

        app.jobs.set_progress(job_id, "loading full-resolution scans", "")
        scans = load_folder(app.scans_dir)
        if len(scans) < 2:
            app.jobs.fail(job_id, "Need at least 2 scans to align automatically.")
            return
        if exclusions:
            scans = filter_scans(scans, exclusions)
            log.info("automatic align: %d exclusion region(s) applied across %d scan(s)",
                     len(exclusions), len({e.scan for e in exclusions}))

        threads = params.get("threads")
        concurrency = max(1, int(params.get("concurrency", 1)))
        # The search voxel can be overridden per job.
        search_voxel = params.get("search_voxel_size")
        cfg_kwargs = {"concurrency": concurrency}
        if float(params.get("range_limit_m") or 0) > 0:
            cfg_kwargs["max_range_m"] = float(params["range_limit_m"])
        if search_voxel is not None:
            cfg_kwargs["search_voxel_size"] = float(search_voxel)
        if params.get("wall_tightness_weighting"):
            cfg_kwargs["wall_tightness_weighting"] = True
        cfg_kwargs["thickness_trust"] = bool(params.get("thickness_trust", True))
        cfg_kwargs["midair_trust"] = bool(params.get("midair_trust", False))
        cfg_kwargs["learned_trust"] = bool(params.get("learned_trust", False))
        cfg_kwargs["seed_judge"] = bool(params.get("seed_judge", False))
        # The weighted step's search runs on the GPU whenever the run has the GPU on.
        cfg_kwargs["trust_search_on_gpu"] = bool(params.get("gpu", False)) and limits.resolve_use_gpu(True)
        if params.get("layout_bound_mean_m") is not None:   # diagnostic override
            cfg_kwargs["local_only_layout_bound_mean_m"] = float(params["layout_bound_mean_m"])
        # Only used with seed_from_placement, the one branch that builds per-scan ScanFeatures. A job that does not
        # say takes PipelineConfig's default.
        cfg_kwargs["detector_vote_enabled"] = bool(
            params.get("detector_vote_enabled", PipelineConfig.detector_vote_enabled))
        cfg_kwargs["adaptive_search_voxel"] = bool(
            params.get("adaptive_search_voxel", PipelineConfig.adaptive_search_voxel))
        cfg_kwargs["use_landmarks"] = bool(params.get("use_landmarks", PipelineConfig.use_landmarks))
        # Unseeded runs use the CUDA pipeline (two pairs at once at most, since they share one card); seeded runs
        # screen their candidates on the GPU and let the CPU confirm the best.
        gpu_wanted = bool(params.get("gpu", False)) and limits.resolve_use_gpu(True)
        seeded_job = bool(params.get("seed_from_placement", False))
        use_gpu_pipeline = gpu_wanted and not seeded_job
        if use_gpu_pipeline:
            cfg_kwargs["use_gpu_pipeline"] = True
            cfg_kwargs["concurrency"] = min(concurrency, 2)
        if gpu_wanted and seeded_job:
            cfg_kwargs["use_gpu_ladder"] = True
            for knob in ("gpu_verify_top", "gpu_screen_iterations"):          # measurement knobs; the page never sets them
                if params.get(knob) is not None:
                    cfg_kwargs[knob] = int(params[knob])
        for knob, cast in _SEARCH_KNOBS.items():                               # search experiments
            if params.get(knob) is not None:
                cfg_kwargs[knob] = cast(params[knob])
        if params.get("proposals_path"):                                       # diagnostics only
            with open(params["proposals_path"]) as fh:
                loaded = json.load(fh)
            cfg_kwargs["proposals"] = {k: v["transform"] for k, v in loaded.items()}
            cfg_kwargs["proposal_agree"] = {k: int(v.get("agree", 0)) for k, v in loaded.items()}
        cfg_kwargs["excluded_pairs"] = _project_excluded_pairs(data)
        # The reference scan (top of the page's list) is the one the solve holds still and the result is pinned to.
        reference_scan = params.get("reference_scan") or data.get("reference_scan")
        scan_names = [sc.name for sc in scans]
        if reference_scan in scan_names:
            cfg_kwargs["posegraph_reference_node"] = scan_names.index(reference_scan)
        cfg = PipelineConfig(**cfg_kwargs)
        # Each with needs a fresh limited_to() (one-shot), and the cap must not wrap the concurrent pairwise stage
        # or it deadlocks.
        def _cap():
            return _thread_cap(params)

        t_align_start = time.time()
        marks = [("start", t_align_start)]       # stage timings for the report
        # Prepare in worker processes: several Open3D calls hold the GIL for seconds, which would freeze the bench.
        # Stoppable between scans.
        app.jobs.set_progress(job_id, "preparing scans", f"0/{len(scans)}")
        prepared = preprocess_all_offloaded(
            scans, cfg, threads=int(threads) if threads else None,
            should_cancel=lambda: app.jobs.is_cancel_requested(job_id),
            progress_cb=lambda done, total: app.jobs.set_progress(job_id, "preparing scans", f"{done}/{total}"))
        if prepared is None:
            result = {"pairs": [], "cancelled": True, "cancelled_during": "preparing scans",
                      "n_pairs_before_filter": 0, "n_pairs_after_filter": 0, "n_pairs_completed": 0,
                      "n_placement_seeds": 0, "adjacency_method": None,
                      "adjacency_padding_m": float(params.get("adjacency_padding_m", 3.0)),
                      "use_gpu_pipeline": cfg.use_gpu_pipeline, "use_gpu_ladder": cfg.use_gpu_ladder, "detector_vote_enabled": cfg.detector_vote_enabled, "adjacency_fractions": [],
                      "pairwise_elapsed_s": 0.0, "total_elapsed_s": round(time.time() - t_align_start, 1)}
            _write_align_auto_report(app.out_dir, result)
            app.jobs.finish_cancelled(job_id, result)
            return
        with _cap():
            all_pairs = candidate_pairs(prepared, cfg)
            # "Run from my alignment": bounded local refinement only.
            seed_from_placement = bool(params.get("seed_from_placement", False))
            # With use_anchors, a second stage runs align_from_anchors on this result (see below).
            use_anchors = bool(params.get("use_anchors", False))
            padding_m = float(params.get("adjacency_padding_m", 3.0))
            adjacency_fraction_rows: list[dict] = []
            features = None
            if seed_from_placement:
                # Attempt pairs matched by static surfaces, plus those found by the raw-point measure.
                surf, fractions, features = pair_filter.feature_surface_pairs(
                    scans, prepared, placements, cfg,
                    progress_cb=lambda detail: app.jobs.set_progress(job_id, "finding features", detail))
                if cfg.detector_vote_enabled:
                    # The same per-scan pass, also given corners, with its own progress line.
                    n_placed = sum(1 for f in features if f is not None)
                    k = 0
                    for i, f in enumerate(features):
                        if f is not None:
                            k += 1
                            app.jobs.set_progress(job_id, "finding features",
                                                  f"corners, edges, doors and landmarks: {k}/{n_placed}: {scans[i].name}")
                            find_corners.build_scan_corners(f)
                            find_corners.build_scan_edges(f)
                            _attach_openings(app, f, scans[i], prepared[i])
                            _attach_landmarks(app, f, scans[i], prepared[i])
                surf_set = set(surf)
                pairs = [p for p in all_pairs if p in surf_set]
                adjacency_method = "feature"
                attempted_set = set(pairs)
                for (i, j), f in fractions.items():
                    adjacency_fraction_rows.append({
                        "a": scans[i].name, "b": scans[j].name,
                        "frac": f["frac"], "method": f.get("method", "feature"),
                        "radius_m": f.get("radius_m"), "n_matches": f.get("n_matches"),
                        "attempted": (i, j) in attempted_set,
                    })
                log.info("automatic align: %d candidate pair(s) (all), %d after feature-surface "
                         "filtering (union: >=%.1f%% matched OR >=%.1f%% raw-point within %.2fm)",
                         len(all_pairs), len(pairs), cfg.adjacency_feature_min_frac * 100,
                         cfg.adjacency_surface_min_frac * 100, cfg.local_only_bound_m)
            else:
                # Decides which pairs are attempted (placed boxes plus padding), never the search within a pair.
                placed = pair_filter.placed_pairs(scans, placements, padding_m)
                pairs = [p for p in all_pairs if p in set(placed)]
                adjacency_method = "bbox"
                log.info("automatic align: %d candidate pair(s) (all), %d after placed-position "
                         "filtering at %.1fm padding", len(all_pairs), len(pairs), padding_m)

            if focus_scan:
                pairs = [p for p in pairs if focus_scan in (scans[p[0]].name, scans[p[1]].name)]
                log.info("automatic align: %d pair(s) with %s, the scan being re-checked", len(pairs), focus_scan)
            seeds: dict[str, np.ndarray] = {}
            if seed_from_placement:
                for i, j in pairs:
                    name_i, name_j = scans[i].name, scans[j].name
                    if name_i in placements and name_j in placements:
                        seeds[seed_key(name_i, name_j)] = relative_transform(
                            placements[name_i], placements[name_j])
                log.info("automatic align: seeded %d of %d attempted pair(s) from the current placement",
                         len(seeds), len(pairs))

        prepared_gpu = None
        if cfg.use_gpu_pipeline:
            from open_point_stitch.align_gpu import prepare_all_gpu
            app.jobs.set_progress(job_id, "preparing scans on the GPU", f"{len(prepared)} scan(s)")
            prepared_gpu = prepare_all_gpu(prepared, cfg)
        marks.append(("preparing scans and finding features", time.time()))
        app.jobs.set_progress(job_id, "registering pairs", f"0/{len(pairs)}")

        def progress_cb(done: int, total: int, desc: str) -> None:
            app.jobs.set_progress(job_id, "registering pairs", f"{done}/{total}: {desc}")

        t_pairwise_start = time.time()
        results = run_pairwise(prepared, pairs, cfg, seeds=seeds, progress_cb=progress_cb, scans=scans,
                               should_cancel=lambda: app.jobs.is_cancel_requested(job_id),
                               seeded_local_only=seed_from_placement, features=features,
                               prepared_gpu=prepared_gpu)
        pairwise_elapsed_s = round(time.time() - t_pairwise_start, 1)
        marks.append(("matching pairs", time.time()))

        if len(results) < len(pairs):
            # Cancelled: return before any pose is touched, keep finished pairs in the report.
            result = {
                "pairs": [r.as_row() for r in results],
                "cancelled": True,
                "n_pairs_before_filter": len(all_pairs),
                "n_pairs_after_filter": len(pairs),
                "n_pairs_completed": len(results),
                "n_placement_seeds": len(seeds),
                "adjacency_method": adjacency_method,
                "adjacency_padding_m": padding_m,
                # Record whether the detector vote was on for this run.
                "use_gpu_pipeline": cfg.use_gpu_pipeline, "use_gpu_ladder": cfg.use_gpu_ladder, "detector_vote_enabled": cfg.detector_vote_enabled,
                "adjacency_fractions": adjacency_fraction_rows,
                "pairwise_elapsed_s": pairwise_elapsed_s,
                "total_elapsed_s": round(time.time() - t_align_start, 1),
            }
            _write_align_auto_report(app.out_dir, result)
            app.jobs.finish_cancelled(job_id, result)
            return

        if focus_scan and not any(r.accepted for r in results):
            # A re-check where no link held is an answer, not a failure: report it and move nothing.
            app.jobs.commit(job_id)                 # writing results from here; Cancel waits
            result = {
                "pairs": [r.as_row() for r in results],
                "no_link_held": focus_scan,
                "n_pairs_before_filter": len(all_pairs),
                "n_pairs_after_filter": len(pairs),
                "pairwise_elapsed_s": pairwise_elapsed_s,
                "total_elapsed_s": round(time.time() - t_align_start, 1),
            }
            _write_align_auto_report(app.out_dir, result)
            app.jobs.finish(job_id, result)
            return

        noise_stage = None
        if params.get("noise_trust"):                   # diagnostics option
            app.jobs.set_progress(job_id, "measuring each scan's noise on flat walls", "")
            noise = {s.name: point_trust.scan_noise(s.name, s.points, placements[s.name].level_rotation)
                     for s in scans if s.name in placements}
            for r in results:
                if r.accepted and r.information is not None:
                    r.information = r.information * point_trust.pair_weight(noise.get(r.source_name),
                                                                               noise.get(r.target_name))
            noise_stage = {n: (None if v is None else round(v * 1000, 2)) for n, v in noise.items()}

        with _cap():
            app.jobs.set_progress(job_id, "solving pose graph", f"{len(scans)} scan(s)")
            # Trusted edges are the placement-adjacent pairs.
            global_result = build_and_optimize(len(scans), results, cfg, trusted_pairs=set(pairs))
        marks.append(("solving the layout", time.time()))

        largest = global_result.components[0]
        largest_set = set(largest)
        unregistered = [scans[i].name for i in range(len(scans)) if i not in largest_set]

        root_idx = _pick_root(largest, scan_names, reference_scan)
        root_name = scans[root_idx].name
        root_target = placements[root_name].matrix() if root_name in placements else np.eye(4)
        # Rigid whole-shape correction so the root scan lands exactly on its current placed pose. The root's own
        # hand-set pitch/roll must not be handed to every other scan, so they get only the yaw and translation.
        correction = root_target @ np.linalg.inv(global_result.poses[root_idx])
        correction_yaw_only = _yaw_and_translation_only(correction)

        by_name = {s["name"]: s for s in data["scans"]}
        pose_rows = []
        pose_updates: dict[str, dict] = {}
        proposals: dict[str, tuple] = {}
        best_fit: dict[int, float] = {}
        for r in results:
            if r.accepted:
                for i in (r.source, r.target):
                    best_fit[i] = max(best_fit.get(i, 0.0), float(r.fitness))
        for idx in largest:
            name = scans[idx].name
            if name not in placements or name not in by_name:
                continue
            if placements[name].locked:
                pose_rows.append({"scan": name, "applied": False,
                                  "reason": "locked; automatic align does not move a locked scan"})
                continue
            # The root lands exactly on its placed pose by construction; every other scan gets only the yaw and
            # translation of the correction.
            world = root_target if idx == root_idx else correction_yaw_only @ global_result.poses[idx]
            yaw_deg, x_m, y_m, z_m, tilt_deg = _decompose_pose(
                world, placements[name].level_rotation, placements[name].floor_z)
            placed = placements[name]
            kept = Placement(name=name, level_rotation=placed.level_rotation, floor_z=placed.floor_z,
                             yaw_deg=yaw_deg, tx=x_m, ty=y_m, tz=z_m,
                             pitch_deg=placed.pitch_deg, roll_deg=placed.roll_deg).matrix()
            centre_shift, dropped_tilt = _centre_shift_for_dropped_tilt(world, kept, scans[idx].points.mean(axis=0))
            if params.get("tilt_drop_keeps_centre", True):    # the solved tilt is dropped; keep its centre, not its origin
                x_m, y_m, z_m = x_m + centre_shift[0], y_m + centre_shift[1], z_m + centre_shift[2]
            s = by_name[name]
            new_pose = ScanPose(yaw_deg=round(yaw_deg, 4), x_m=round(x_m, 5),
                                y_m=round(y_m, 5), z_m=round(z_m, 5))
            update = {"yaw_deg": new_pose.yaw_deg, "x_m": new_pose.x_m,
                      "y_m": new_pose.y_m, "z_m": new_pose.z_m}
            # human is the user's original hand placement, which may be much older and is never changed by an alignment.
            human = ScanPose.from_dict(s, strict=True, prefix="human_")
            if name in app.payloads:
                update["align_moved_m"] = round(pose_shift_m(app.payloads[name], human, new_pose), 4)
                update["align_moved_deg"] = round(yaw_gap_deg(new_pose.yaw_deg, s["human_yaw_deg"]), 3)
            pose_updates[name] = update
            row = {"scan": name, "applied": True, "residual_tilt_deg": round(tilt_deg, 3),
                   "dropped_tilt_deg": round(dropped_tilt, 3),
                   "centre_shift_m": [round(float(v), 4) for v in centre_shift]}
            pose_rows.append(row)
            proposals[name] = (new_pose, idx, row)

        # Movement limits, against the pose this run was seeded from (never the possibly stale hand placement): per
        # scan, then the whole layout, since moves each inside the per-scan limit can still add up.
        limits_relative = bool(params.get("limits_relative", False))
        limit_outcome = None
        if seed_from_placement:
            checks = [MoveCheck(name=n, start=ScanPose.from_dict(by_name[n], strict=True), proposed=p[0],
                                payload=app.payloads[n], anchor=p[1] == root_idx, fitness=best_fit.get(p[1]))
                      for n, p in proposals.items() if n in app.payloads]
            if limits_relative:
                checks += [MoveCheck(name=s["name"], start=ScanPose.from_dict(s, strict=True),
                                     proposed=ScanPose.from_dict(s, strict=True), payload=app.payloads[s["name"]],
                                     locked=True) for s in data["scans"]
                           if s.get("locked") and s["name"] in app.payloads and s["name"] in placements]
            limit_outcome = _limit_moves(
                checks, cfg.scan_bound_m if cfg.scan_bound_m is not None else cfg.local_only_bound_m,
                cfg.scan_bound_deg if cfg.scan_bound_deg is not None else cfg.local_only_bound_deg,
                cfg.local_only_layout_bound_mean_m, relative=limits_relative)
            for n, (new_pose, _, row) in proposals.items():
                kept = limit_outcome.final.get(n, new_pose)
                if kept == new_pose:
                    continue
                s = by_name[n]
                human = ScanPose.from_dict(s, strict=True, prefix="human_")
                if n not in limit_outcome.capped or limit_outcome.layout_tripped or not limits_relative:
                    kept = ScanPose.from_dict(s, strict=True)       # back to where this run started
                else:                                               # carried with the layout
                    kept = ScanPose(yaw_deg=round(kept.yaw_deg, 4), x_m=round(kept.x_m, 5), y_m=round(kept.y_m, 5),
                                    z_m=round(kept.z_m, 5))
                # align_moved_m/deg keep reporting real distance from the ORIGINAL hand placement, so the
                # "moved by align" readout and Revert still mean what they always have.
                pose_updates[n] = {"yaw_deg": kept.yaw_deg, "x_m": kept.x_m, "y_m": kept.y_m, "z_m": kept.z_m,
                                   "align_moved_m": round(pose_shift_m(app.payloads[n], human, kept), 4),
                                   "align_moved_deg": round(yaw_gap_deg(kept.yaw_deg, s["human_yaw_deg"]), 3)}
                row["residual_tilt_deg"] = None     # the solved pose was discarded
                if n in limit_outcome.capped:
                    row["kept_placement_bound_exceeded"] = True
                if limit_outcome.layout_tripped and (n not in limit_outcome.capped or limits_relative):
                    row["layout_bound_exceeded"] = True
            for r in limit_outcome.rejected:
                log.info("%s: %s limit refused a %.2fm/%.1fdeg correction (measured %s, fit %s)", r["scan"],
                         r["limit"], r["correction_m"], r["correction_deg"] or 0.0, r["measured"], r["fitness"])
        layout_bound_tripped = bool(limit_outcome and limit_outcome.layout_tripped)
        mean_shift_m = limit_outcome.mean_shift_m if limit_outcome else 0.0

        # Second stage: the user's links refine this job's result (align_from_anchors), so a link can move a scan the
        # geometry search moved or left alone.
        anchor_stage = None
        if use_anchors:
            stage2_anchors = _anchors_from_placement(data)
            if not stage2_anchors:
                anchor_stage = {"n_anchors": 0}
            else:
                app.jobs.set_progress(job_id, "applying your links", f"0/{len(stage2_anchors)} anchor(s)")
                anchor_poses: dict[str, ScanPose] = {}
                anchor_locked: dict[str, bool] = {}
                for s in data["scans"]:
                    u = pose_updates.get(s["name"], {})
                    anchor_poses[s["name"]] = ScanPose.from_dict({**s, **u}, strict=True)
                    anchor_locked[s["name"]] = bool(s.get("locked", False))
                # max_shift_m=1.0 rather than the 8.0 default: this refines an already-seeded position, and 8.0 allowed
                # runaway moves. Matches _run_align's job-level default.
                anchor_result = align_from_anchors(
                    app.payloads, anchor_poses, stage2_anchors, anchor_locked, None,
                    scans_folder=app.scans_dir, max_shift_m=1.0, reference_scan=root_name,
                    progress_cb=lambda done, total: app.jobs.set_progress(
                        job_id, "applying your links", f"{done}/{total} anchor(s)"),
                    should_cancel=lambda: app.jobs.is_cancel_requested(job_id))
                if anchor_result.cancelled:
                    # The geometry stage above already finished and is kept below - only this
                    # stage's own partial work is discarded, same reasoning align_from_anchors
                    # already gives for why a partial graph.solve() answers nothing.
                    anchor_stage = {"cancelled": True, "n_anchors": len(stage2_anchors),
                                    "n_anchors_completed": anchor_result.n_anchors_completed}
                else:
                    for name, pose in anchor_result.poses.items():
                        update = pose_updates.setdefault(name, {})
                        update.update({"yaw_deg": round(pose.yaw_deg, 4), "x_m": round(pose.x_m, 5),
                                      "y_m": round(pose.y_m, 5), "z_m": round(pose.z_m, 5)})
                        s = by_name.get(name)
                        if s is not None and name in app.payloads:
                            human = ScanPose.from_dict(s, strict=True, prefix="human_")
                            update["align_moved_m"] = round(pose_shift_m(app.payloads[name], human, pose), 4)
                            update["align_moved_deg"] = round(
                                yaw_gap_deg(pose.yaw_deg, s["human_yaw_deg"]), 3)
                    anchor_stage = {"rows": [vars(r) for r in anchor_result.rows],
                                    "skipped_anchors": anchor_result.skipped,
                                    "n_anchors": len(stage2_anchors),
                                    "n_anchors_completed": anchor_result.n_anchors_completed}

        # Last: settle heights on floors and ceilings that overlapping scans share, for scans this run could move;
        # skipped when the whole run was declined.
        marks.append(("limits and links stage", time.time()))
        height_stage = None
        if params.get("height_settle", True) and not layout_bound_tripped:
            app.jobs.set_progress(job_id, "settling heights on shared floors and ceilings", "")
            settle_poses: dict[str, ScanPose] = {}
            for s in data["scans"]:
                u = pose_updates.get(s["name"], {})
                settle_poses[s["name"]] = ScanPose.from_dict({**s, **u}, strict=True)
            locked_names = {s["name"] for s in data["scans"] if s.get("locked")}
        # Floors and ceilings marked in review.
            floor_marks = {s["name"]: [float(s[k]) for k in ("floor_h_m", "ceiling_h_m") if s.get(k) is not None]
                           for s in data["scans"]}
            floor_marks = {n: v for n, v in floor_marks.items() if v}
            settle_pieces = None
            if params.get("settle_on_pieces", False):            # off by default
                settle_pieces = find_floors_ceilings.pieces_for_scans(
                    app.scans_dir, os.path.join(app.out_dir, build_bench.CACHE_DIRNAME), app.payloads, data,
                    [s["name"] for s in data["scans"]],
                    progress=lambda i, n, name: app.jobs.set_progress(
                        job_id, "finding each scan's floors and ceilings", f"{i + 1}/{n}: {name}"))
            settle_dz, settle_gaps = find_floors_ceilings.settle(app.payloads, settle_poses, locked_names,
                                                          only=set(pose_updates) - locked_names, marks=floor_marks,
                                                          pieces=settle_pieces)
            for name, d in settle_dz.items():
                pose_updates[name]["z_m"] = round(settle_poses[name].z_m + d, 5)
            height_stage = {"n_pairs": len(settle_gaps), "dz_m": {n: round(d, 4) for n, d in settle_dz.items()},
                            "marked": sorted(floor_marks),
                            "on_pieces": sorted(n for n, v in (settle_pieces or {}).items() if v and n not in floor_marks)}

        # Why each scan moved and what that did to its other links; the guard keeps back a scan whose move made a
        # good link worse.
        marks.append(("height settle", time.time()))
        app.jobs.set_progress(job_id, "explaining each move", "")
        move_report = _explain_run_moves(placements, pose_updates, prepared, results,
                                         bool(params.get("collateral_guard", False)))
        for name in move_report["kept_back"]:
            s = by_name[name]
            pose_updates[name] = {**pose_updates[name], "yaw_deg": s["yaw_deg"], "x_m": s["x_m"],
                                  "y_m": s["y_m"], "z_m": s["z_m"]}

        def _apply(fresh: dict) -> None:
            # Merge onto the current on-disk placement, including the links stage above, if any.
            fresh_by_name = {s["name"]: s for s in fresh.get("scans", [])}
            for name, update in pose_updates.items():
                s = fresh_by_name.get(name)
                if s is None:
                    continue
                s.update(update)
            # Persist each link's outcome and both sides' poses at landing time, so a reload can show them.
            def _pose_at_landing(name: str) -> Optional[list]:
                s = fresh_by_name.get(name)
                if s is None:
                    return None
                return [s.get("x_m", 0.0), s.get("y_m", 0.0), s.get("yaw_deg", 0.0)]

            fresh["last_align_pairs"] = [
                {**r.as_row(),
                 "a_pose_at_align": _pose_at_landing(r.source_name),
                 "b_pose_at_align": _pose_at_landing(r.target_name)}
                for r in results
            ]
            # "align_auto_anchors" only when the links stage actually applied something.
            stamp_kind = "align_auto_anchors" if (anchor_stage and anchor_stage.get("n_anchors")) else "align_auto"
            _stamp_align(fresh, stamp_kind)
            # Persist the filter and seed counts in last_run.
            _stamp_last_run(fresh, stamp_kind, {
                "n_pairs_attempted": len(pairs), "n_accepted": sum(1 for r in results if r.accepted),
                "connected": global_result.connected, "n_unregistered": len(unregistered),
                "elapsed_s": round(time.time() - t_align_start, 1),
                "n_pairs_before_filter": len(all_pairs),
                "n_placement_seeds": len(seeds),
                "adjacency_method": adjacency_method,
                "adjacency_padding_m": padding_m,
                "use_gpu_pipeline": cfg.use_gpu_pipeline, "use_gpu_ladder": cfg.use_gpu_ladder, "detector_vote_enabled": cfg.detector_vote_enabled,  # handoff 2026-09-14 item G
                "use_anchors": use_anchors,
                "n_anchors": (anchor_stage or {}).get("n_anchors", 0),
                "layout_bound_tripped": layout_bound_tripped,
                "layout_mean_shift_m": round(mean_shift_m, 4),
                "bent_scans": [{"scan": f["scan"], "kind": f["kind"], "text": f["text"]} for f in bent_report["flags"]],
            })

        marks.append(("explaining moves", time.time()))
        app.jobs.set_progress(job_id, "checking for bent scans", "")
        with app._lock:
            payloads_now = wall_checks.with_wall_edits(dict(app.payloads), wall_checks.edits_by_scan(data))
        bent_report = _bent_scan_report(scans, _final_placements(placements, pose_updates), payloads_now)
        marks.append(("bent-scan check", time.time()))
        app.jobs.commit(job_id)                     # writing results from here; Cancel waits
        _write_placement_merged(app.out_dir, _apply)

        result = {
            "pairs": [r.as_row() for r in results],
            "poses": pose_rows,
            "n_odometry_edges": global_result.n_odometry_edges,
            "n_loop_edges": global_result.n_loop_edges,
            "connected": global_result.connected,
            "unregistered": unregistered,
            # Counts before and after filtering, plus wall clock.
            "n_pairs_before_filter": len(all_pairs),
            "n_pairs_after_filter": len(pairs),
            "n_placement_seeds": len(seeds),
            "adjacency_method": adjacency_method,
            "adjacency_padding_m": padding_m,
            "use_gpu_pipeline": cfg.use_gpu_pipeline, "use_gpu_ladder": cfg.use_gpu_ladder, "detector_vote_enabled": cfg.detector_vote_enabled,  # handoff 2026-09-14 item G
            "adjacency_fractions": adjacency_fraction_rows,
            "pairwise_elapsed_s": pairwise_elapsed_s,
            "total_elapsed_s": round(time.time() - t_align_start, 1),
            "stage_seconds": {name: round(t - marks[k][1], 1) for k, (name, t) in enumerate(marks[1:])},
            # The links stage's report, or None when use_anchors was not set. {"n_anchors": 0} means it was set but
            # nothing was saved to apply.
            "use_anchors": use_anchors,
            "anchor_stage": anchor_stage,
            "height_stage": height_stage,        # None when turned off or the run was declined
            "noise_mm": noise_stage,             # each scan's flat-wall noise when noise_trust was on, else None
            "moves": move_report["moves"],       # one plain-language account per moved scan
            "move_summary": move_report["summary"],
            "move_suggestions": move_report["suggestions"],   # solved poses the guard kept back
            "bent_scans": bent_report,           # reports only; moves nothing
            # The whole-layout bound's verdict, always reported, so a saved report says whether the geometry search was
            # declined.
            "layout_bound_tripped": layout_bound_tripped,
            "layout_mean_shift_m": round(mean_shift_m, 4),
            # Every correction a movement limit threw away: which limit, how big, and the best fit behind it.
            "limits_measured": "against the other scans" if limits_relative else "from where each scan started",
            "rejected_corrections": limit_outcome.rejected if limit_outcome else [],
            "layout_common_move": limit_outcome.common if limit_outcome else None,
            "reference_scan": root_name,         # the scan this run was pinned to
        }
        _write_align_auto_report(app.out_dir, result)
        app.jobs.finish(job_id, result)
    except Exception as exc:  # noqa: BLE001 - job errors must reach the job, not the log alone
        log.exception("automatic align job failed")
        app.jobs.fail(job_id, f"{type(exc).__name__}: {exc}")


def _export_keeps_excluded(out_dir: str) -> bool:
    """The project's "Keep exclude-region points in the export" setting, on unless turned off."""
    return bool((_read_placement(out_dir) or {}).get("settings", {}).get("export_keeps_excluded", True))


def _landmarks_for_drawings(out_dir: str) -> list[dict]:
    """The named landmarks the drawings show: all of them, unless the project's "Named landmarks in the drawings"
    setting is off."""
    data = _read_placement(out_dir) or {}
    return data.get("landmarks", []) if data.get("settings", {}).get("landmarks_in_drawings", True) else []


@_job("export job failed", needs="exporting")
def _run_export(app: "App", job_id: str, params: dict) -> None:
    placement_path = _placement_path(app.out_dir)
    gpu_requested = bool(params.get("gpu", False))
    cfg = PipelineConfig(use_gpu=limits.resolve_use_gpu(gpu_requested),
                         outlier_nb_neighbors=(20 if params.get("remove_strays") else None),
                         outlier_std_ratio=float(params.get("stray_std_ratio") or 2.0))

    def progress_cb(phase, detail):
        app.jobs.set_progress(job_id, phase, detail)

    output_path = _export_path(app.out_dir, "merged", "pts")
    # Written under a .partial name and renamed only when complete, so a stopped export never leaves a file that
    # looks finished.
    partial_path = output_path[:-len(".pts")] + ".partial.pts"
    cap = _thread_cap(params)
    try:
        with cap:
            outcome = place_merge(
                scans_folder=app.scans_dir, placements_path=placement_path,
                output_path=partial_path, cfg=cfg,
                refine=bool(params.get("refine", False)),
                refine_max_shift_m=float(params.get("refine_max_shift_m", 0.5)),
                progress_cb=progress_cb,
                should_cancel=lambda: app.jobs.is_cancel_requested(job_id),
                keep_excluded=_export_keeps_excluded(app.out_dir),
            )
    except concurrent.futures.CancelledError as exc:
        # Nothing has been written yet: place_merge writes output_path only after merging finishes, and
        # should_cancel is only checked before that.
        _finish_cancelled(app, job_id, "export", exc)
        return

    def _apply(fresh: dict) -> None:
        _stamp_last_run(fresh, "export", {
            "points": outcome.merged.n_points,
            "n_skipped_no_placement": len(outcome.skipped_no_placement),
        })

    app.jobs.commit(job_id)                     # writing results from here; Cancel waits
    os.replace(partial_path, output_path)
    _write_placement_merged(app.out_dir, _apply)
    _note_latest_export(app.out_dir, merged=output_path)

    app.jobs.finish(job_id, {
        "output_path": output_path,
        "points": outcome.merged.n_points,
        "skipped_no_placement": outcome.skipped_no_placement,
        "refine_log": outcome.refine_log,
    })


@_job("plan export failed", needs="exporting a plan view")
def _run_export_plan(app: "App", job_id: str, params: dict) -> None:
    """The plan-view drawing: a horizontal cut through the placed scans at full resolution, as a DXF and a preview
    picture in the project folder. The band is the bench's Z clip if the page sends one, otherwise 1.0 to 1.6 m
    above the typical floor."""
    placement_path = _placement_path(app.out_dir)
    loaded = load_placements(placement_path)
    z_lo, z_hi = default_band(loaded.placements)
    if params.get("z_lo") is not None and params.get("z_hi") is not None:
        z_lo, z_hi = float(params["z_lo"]), float(params["z_hi"])
    crop = None
    if params.get("crop"):
        c = params["crop"]
        crop = (float(c.get("x_lo", -1e9)), float(c.get("x_hi", 1e9)), float(c.get("y_lo", -1e9)), float(c.get("y_hi", 1e9)))

    app.jobs.set_progress(job_id, "loading scans", "full resolution")
    app.wait_for_all()
    scans = load_folder(app.scans_dir)
    try:
        with _thread_cap(params):
            cut = plan_cut(scans, loaded.placements, for_export(loaded.exclusions, _export_keeps_excluded(app.out_dir)),
                           z_lo, z_hi, crop,
                           progress_cb=lambda phase, detail: app.jobs.set_progress(job_id, phase, detail),
                           should_cancel=lambda: app.jobs.is_cancel_requested(job_id))
    except concurrent.futures.CancelledError as exc:
        _finish_cancelled(app, job_id, "plan export", exc)
        return
    with app._lock:
        payloads = dict(app.payloads)
    cut.footprints = footprints_from_payloads(payloads, loaded.placements)
    cut.landmarks = world_boxes(_landmarks_for_drawings(app.out_dir), loaded.placements)
    stem = os.path.basename(_export_path(app.out_dir, "plan_view", "dxf"))[:-len(".dxf")]
    app.jobs.commit(job_id)                     # writing results from here; Cancel waits
    info = write_plan(cut, project_mod.exports_dir(app.out_dir), note="cropped" if crop else "", stem=stem)
    _note_latest_export(app.out_dir, plan_view_dxf=info["dxf_path"], plan_view_png=info["png_path"])
    app.jobs.finish(job_id, info)


@_job("3D export failed", needs="exporting")
def _run_export_3d(app: "App", job_id: str, params: dict) -> None:
    """Floors, ceilings and walls of the placed scans as 3D planes in one DXF (and a top-down picture) in the
    project's exports folder."""
    placement_path = _placement_path(app.out_dir)
    loaded = load_placements(placement_path)
    with open(placement_path) as fh:
        data = json.load(fh)
    app.wait_for_all()
    with app._lock:
        payloads = wall_checks.with_wall_edits(dict(app.payloads), wall_checks.edits_by_scan(data))
    names = [n for n in loaded.placements if n in payloads]
    pieces = find_floors_ceilings.pieces_for_scans(
        app.scans_dir, os.path.join(app.out_dir, build_bench.CACHE_DIRNAME), payloads, data, names,
        progress=lambda i, n, name: app.jobs.set_progress(job_id, "finding floors and ceilings",
                                                          f"{i + 1}/{n}: {name}"))
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in build_bench.find_scans(app.scans_dir)}
    walls = {}
    for i, n in enumerate(names):                       # each wall cut to where the scan really has wall
        app.jobs.set_progress(job_id, "finding where each wall starts and stops", f"{i + 1}/{len(names)}: {n}")
        pts = find_floors_ceilings.levelled_scan(paths[n], payloads[n]["level_rotation"], payloads[n]["floor_z"],
                                                  find_floors_ceilings.scan_exclusions(data, n))
        walls[n] = find_walls.wall_runs(pts, payloads[n].get("walls", []))
    app.jobs.set_progress(job_id, "joining what several scans saw", "")
    per_scan = {n: export_3d.scan_surfaces(n, pieces.get(n, []), walls[n], loaded.placements[n]) for n in names}
    surfaces = export_3d.venue_surfaces(per_scan)
    stem = os.path.basename(_export_path(app.out_dir, "surfaces_3d", "dxf"))[:-len(".dxf")]
    app.jobs.commit(job_id)                     # writing results from here; Cancel waits
    boxes = world_boxes(_landmarks_for_drawings(app.out_dir), loaded.placements)
    info = export_3d.write_3d(surfaces, project_mod.exports_dir(app.out_dir), stem=stem, landmarks=boxes)
    _note_latest_export(app.out_dir, surfaces_3d_dxf=info["dxf_path"], surfaces_3d_png=info["png_path"])
    app.jobs.finish(job_id, {**info, "scans_used": len(names), "scans_with_pieces": len(pieces)})


@_job("wall thickness report failed", needs="measuring walls")
def _run_wall_report(app: "App", job_id: str, params: dict) -> None:
    """How thick each wall reads on the placed scans, in inches, and which scan is off by how much."""
    placement_path = _placement_path(app.out_dir)
    loaded = load_placements(placement_path)
    app.jobs.set_progress(job_id, "loading scans", "full resolution")
    app.wait_for_all()
    scans = load_folder(app.scans_dir)
    with open(placement_path) as fh:
        hand_walls = wall_checks.edits_by_scan(json.load(fh))
    with app._lock:
        payloads = wall_checks.with_wall_edits(dict(app.payloads), hand_walls)
    try:
        with _thread_cap(params):
            reports = measure_walls(scans, loaded.placements, payloads,
                                    progress_cb=lambda phase, detail: app.jobs.set_progress(job_id, phase, detail),
                                    should_cancel=lambda: app.jobs.is_cancel_requested(job_id))
    except concurrent.futures.CancelledError as exc:
        _finish_cancelled(app, job_id, "wall report", exc)
        return
    rows = [dataclasses.asdict(r) for r in reports]
    app.jobs.set_progress(job_id, "checking for bent scans", "")
    bent = _bent_scan_report(scans, loaded.placements, payloads)
    app.jobs.commit(job_id)                     # writing results from here; Cancel waits
    path = os.path.join(project_mod.reports_dir(app.out_dir), "wall_thickness.json")
    write_json_atomic(path, {"summary": summarise_walls(reports), "walls": rows, "bent_scans": bent}, indent=1)
    app.jobs.finish(job_id, {"summary": summarise_walls(reports), "walls": rows[:40], "total_walls": len(rows),
                             "bent_scans": bent, "path": path})


# ---------------------------------------------------------------------------
# the app: holds the built page and payloads, owns the HTTP handler
# ---------------------------------------------------------------------------

def _build_page_html() -> bytes:
    """Build the bench page once at startup (nothing in it is project-specific)."""
    with open(build_bench.TEMPLATE, "r", encoding="utf-8") as fh:
        template = fh.read()
    # count=1: the marker text also appears in a JS comment further down the template.
    template = template.replace(
        "<!--BUILD_INFO-->",
        "<script>window.BUILD_INFO=" + json.dumps(build_info()) + ";</script>",
        1,
    )
    return template.encode("utf-8")


def _scans_fingerprint(scans_dir: str, pattern: str) -> tuple:
    """Each scan file's name, size and time: what a project's loaded previews were built from."""
    out = []
    for p in sorted(build_bench.find_scans(scans_dir, pattern)):
        st = os.stat(p)
        out.append((os.path.basename(p), st.st_size, int(st.st_mtime)))
    return tuple(out)


def _quick_point_count(path: str) -> int:
    """Untrusted header point count for the first paint of a still-building row only;
    replaced when the real preview arrives."""
    try:
        with open(path, "r", errors="replace") as fh:
            first = fh.readline().split()
        if len(first) == 1:
            return int(first[0])
    except (OSError, ValueError):
        pass
    return 0


THUMBNAIL_FILENAME = "thumbnail.png"


def _render_project_thumbnail(app: "App", out_dir: str) -> bool:
    """Write a small top-down thumbnail from the preview arrays (the one render allowed to be thinned; never touches
    exports). True if written."""
    payloads = dict(app.payloads)  # snapshot: builds may still be running for other scans
    if not payloads:
        return False
    saved = _read_placement(out_dir) or {}
    poses = {s["name"]: s for s in saved.get("scans", [])}

    W, H = 480, 360
    canvas = np.full((H, W, 3), (17, 20, 26), dtype=np.uint8)  # matches --bg

    pts_all, rgb_all = [], []
    for name, payload in payloads.items():
        xy = np.asarray(payload.get("xy") or [], dtype=float)
        if xy.size == 0:
            continue
        rgb = np.asarray(payload.get("rgb") or [], dtype=float)
        pose = poses.get(name)
        if pose:
            xy = xy @ yaw_matrix2(float(pose.get("yaw_deg", 0.0))).T + [float(pose.get("x_m", 0.0)),
                                                                         float(pose.get("y_m", 0.0))]
        pts_all.append(xy)
        rgb_all.append(rgb)
    if not pts_all:
        return False
    pts = np.concatenate(pts_all, axis=0)
    rgb = np.concatenate(rgb_all, axis=0)

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 0.1)
    margin = 0.92  # fraction of the canvas the data spans, leaving a quiet border
    scale = min(W * margin / span[0], H * margin / span[1])
    ox = W / 2 - (lo[0] + hi[0]) / 2 * scale
    oy = H / 2 + (lo[1] + hi[1]) / 2 * scale  # +row is DOWN in image space, so Y flips

    px = np.round(pts[:, 0] * scale + ox).astype(int)
    py = np.round(oy - pts[:, 1] * scale).astype(int)
    # A 3x3 stamp per point, not a bare 1px dot - at 480x360 for a
    # multi-scan venue a single pixel per point is too faint to read as
    # anything but noise.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = px + dx, py + dy
            m = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
            canvas[ny[m], nx[m]] = np.clip(rgb[m], 0, 255).astype(np.uint8)

    try:
        import open3d as o3d
        o3d.io.write_image(os.path.join(out_dir, THUMBNAIL_FILENAME), o3d.geometry.Image(canvas))
    except Exception:
        log.exception("could not write project thumbnail to %s", out_dir)
        return False
    return True


class _IdleStoppingPool:
    """Worker processes started when work arrives and stopped after IDLE_S idle: an idle worker holds about 400 MB
    and opening a project starts a dozen. Same submit/shutdown interface as ProcessPoolExecutor."""
    IDLE_S = 30.0

    def __init__(self, max_workers: int):
        self._n = max_workers
        self._pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._busy = 0
        self._timer: Optional[threading.Timer] = None
        self._closed = False
        self._lock = threading.Lock()

    def workers(self) -> int:
        """How many worker processes are running now."""
        with self._lock:
            return len(self._pool._processes or {}) if self._pool is not None else 0

    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._pool is None:
                self._pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self._n, mp_context=multiprocessing.get_context("spawn"))
            self._busy += 1
            fut = self._pool.submit(fn, *args, **kwargs)
        fut.add_done_callback(self._one_done)
        return fut

    def _one_done(self, _fut) -> None:
        with self._lock:
            self._busy -= 1
            if self._busy == 0 and self._pool is not None and self._timer is None:
                self._timer = threading.Timer(self.IDLE_S, self._stop_if_idle)
                self._timer.daemon = True
                self._timer.start()

    def _stop_if_idle(self) -> None:
        with self._lock:
            self._timer = None
            if self._busy or self._pool is None:
                return
            pool, self._pool = self._pool, None
        pool.shutdown(wait=False)

    def shutdown(self, wait: bool = False, cancel_futures: bool = False) -> None:
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class App:
    """One open project: builds scan previews in the background while the page is served at once."""

    def __init__(self, scans_dir: str, out_dir: str, pattern: str = "*.pts"):
        self.scans_dir = scans_dir
        self.out_dir = out_dir
        self.jobs = JobStore(os.path.join(out_dir, "jobs"))
        self._lock = threading.Lock()
        self.payloads: dict[str, dict] = {}       # name -> full payload, once built
        self.wire_cache: dict[str, tuple] = {}    # name -> (that payload, its encoded reply)
        self.build_errors: dict[str, str] = {}    # name -> error message, if a build failed
        self.build_events: dict[str, threading.Event] = {}
        self.build_cached: dict[str, bool] = {}   # name -> True if served from cache
        self._tilt_inputs: dict[str, dict] = {}   # name -> every point, levelled, for the tilt solve
        self._tilt_lock = threading.Lock()
        # Near-instant raw bounding box per scan, shown before the real preview is ready.
        self.quick_bbox: dict[str, dict] = {}
        self._first_pass_done: dict[str, threading.Event] = {}   # saved preview read, or rough box done

        saved = _read_placement(out_dir) or {}
        _apply_process_settings(saved.get("settings", {}))   # before any preview builds
        paths = build_bench.find_scans(scans_dir, pattern)
        self._cache_dir = os.path.join(out_dir, build_bench.CACHE_DIRNAME)
        self._paths = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
        if not paths:
            raise FileNotFoundError(f"No .pts files in {scans_dir}")
        saved = _freeze_wall_edits(out_dir, saved, self._paths, self._cache_dir)
        self.scans_fingerprint = _scans_fingerprint(scans_dir, pattern)

        self.scan_meta = []  # stable order for the initial list
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            self.scan_meta.append({"name": name, "points": _quick_point_count(p)})
            self.build_events[name] = threading.Event()
            self._first_pass_done[name] = threading.Event()

        log.info("serving instantly: %d scan(s) found in %s, previews building in the background",
                 len(paths), scans_dir)

        cache_dir = os.path.join(out_dir, build_bench.CACHE_DIRNAME)
        # One worker per scan up to the CPU count; BENCH_PREVIEW_BUILD_WORKERS caps it for QA.
        env_cap = os.environ.get("BENCH_PREVIEW_BUILD_WORKERS")
        n_workers = (max(1, int(env_cap)) if env_cap
                     else max(1, min(len(paths), os.cpu_count() or 4)))
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="bench-build")
        # Real builds (cache misses) run in worker processes, not these threads, because Open3D calls hold the GIL
        # for seconds. spawn, so a worker never inherits a running OpenMP pool.
        self._preview_pool = _IdleStoppingPool(n_workers)
        # The fast pass is submitted first so every scan gets a box early. It reads a valid saved preview on this
        # thread, and computes the rough box only on a miss.
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            self._executor.submit(self._cached_or_quick_bbox_one, p, cache_dir,
                                  _reprocessed_exclusions(saved, name), _reprocessed_exclusions(saved, name, "stairs"))
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            self._executor.submit(self._build_one, p, cache_dir, _reprocessed_exclusions(saved, name),
                                  _reprocessed_exclusions(saved, name, "stairs"))
        # The thumbnail renders once, after every build, on the same pool.
        self._executor.submit(self._render_thumbnail_when_ready)

    def _render_thumbnail_when_ready(self) -> None:
        self.wait_for_all()
        try:
            _render_project_thumbnail(self, self.out_dir)
        except Exception:  # noqa: BLE001 - a missing thumbnail is not worth failing anything over
            log.exception("project thumbnail render failed (not fatal)")

    def _cached_or_quick_bbox_one(self, path: str, cache_dir: str, exclusions: Optional[list] = None,
                                  stairs: Optional[list] = None) -> None:
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        try:
            payload = build_bench.load_cached_payload(path, cache_dir, exclusions, stairs, self.preview_options())
        except Exception:  # noqa: BLE001 - the real build below still runs
            log.exception("%s: reading the saved preview failed, building it instead", name)
            payload = None
        try:
            if payload is None:
                self._build_quick_bbox_one(path)
            else:
                self._record_payload(name, payload, True, t0)
        finally:
            self._first_pass_done[name].set()

    def _record_payload(self, name: str, payload: dict, was_cached: bool, t0: float) -> None:
        with self._lock:
            self.payloads[name] = payload
            self.build_cached[name] = was_cached
        # Log per-scan footprint: 7 floats (28 bytes) per preview point, matching STRIDE in glRebuildPoints.
        n_preview = len(payload.get("xy", []))
        log.info(
            "%s: %s in %.1fs - %s points full-res, %s preview points sent to the "
            "browser (~%.1f KB in the WebGL buffer)",
            name, "loaded from cache" if was_cached else "built", time.time() - t0,
            f"{payload.get('points', 0):,}", f"{n_preview:,}", n_preview * 28 / 1024,
        )
        self.build_events[name].set()

    def _build_quick_bbox_one(self, path: str) -> None:
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        try:
            payload = build_bench.quick_bbox_payload(path)
            with self._lock:
                self.quick_bbox[name] = payload
            # Log the full-resolution point count here too, the first thing known about a scan's size.
            log.info("%s: quick bbox in %.2fs - %s points full-res",
                      name, time.time() - t0, f"{payload.get('points', 0):,}")
        except Exception:  # noqa: BLE001 - the real build is what actually matters
            log.exception("%s: quick bbox failed (not fatal, the real preview is still coming)", name)

    def reprocess(self, name: str) -> dict:
        """Rebuild one scan's preview with all its exclusions cut first (re-finding its level, walls and bands),
        record that in the project file so the next open does the same, and say what changed."""
        if name not in self._paths:
            raise KeyError(name)
        self.build_events[name].wait()
        saved = _read_placement(self.out_dir) or {}
        ids = [e["id"] for key in ("exclusions", "stairs") for e in saved.get(key, []) if e.get("scan") == name]

        def _record(fresh: dict) -> None:
            for s in fresh.get("scans", []):
                if s.get("name") == name:
                    s["reprocessed_with"] = ids
        _write_placement_merged(self.out_dir, _record)
        before = self.payloads.get(name) or {}
        now = {**saved, "scans": [{"name": name, "reprocessed_with": ids}]}
        payload, _ = self._preview_pool.submit(build_bench.build_scan_payload_cached, self._paths[name],
                                               self._cache_dir, _reprocessed_exclusions(now, name),
                                               _reprocessed_exclusions(now, name, "stairs"),
                                               self.preview_options()).result()
        with self._lock:
            self.payloads[name] = payload
        with self._tilt_lock:
            self._tilt_inputs.pop(name, None)      # its levelling may have changed
        return {"name": name, "reprocessed_with": ids, "changes": _describe_reprocess(before, payload)}

    def preview_options(self) -> dict:
        """The project's preview settings, read from its file each time so a changed setting reaches the next build in
        its worker process."""
        return _preview_options_for(_read_placement(self.out_dir) or {})

    def rebuild(self, name: str) -> dict:
        """Rebuild one scan's preview with the project's current settings (for example after its levelling switch
        changed), keeping its re-processed state as it is."""
        if name not in self._paths:
            raise KeyError(name)
        self.build_events[name].wait()
        saved = _read_placement(self.out_dir) or {}
        before = self.payloads.get(name) or {}
        payload, _ = self._preview_pool.submit(build_bench.build_scan_payload_cached, self._paths[name],
                                               self._cache_dir, _reprocessed_exclusions(saved, name),
                                               _reprocessed_exclusions(saved, name, "stairs"),
                                               self.preview_options()).result()
        with self._lock:
            self.payloads[name] = payload
        with self._tilt_lock:
            self._tilt_inputs.pop(name, None)      # its levelling may have changed
        return {"name": name, "tilt_before": before.get("tilt"), "tilt": payload.get("tilt"),
                "levelled_to_floor": not np.allclose(payload["level_rotation"], np.eye(3))}

    def floors_ceilings(self, name: str) -> Optional[list]:
        """This scan's floor and ceiling pieces (computed in a worker process the first time, then read from the cache).
        None while its preview is still building."""
        if name not in self._paths:
            raise KeyError(name)
        with self._lock:
            payload = self.payloads.get(name)
        if payload is None:
            return None
        saved = _read_placement(self.out_dir) or {}
        return self._preview_pool.submit(
            find_floors_ceilings.cached_pieces, self._paths[name], self._cache_dir, payload["level_rotation"],
            payload["floor_z"], find_floors_ceilings.scan_exclusions(saved, name)).result()

    def _build_one(self, path: str, cache_dir: str, exclusions: Optional[list] = None,
                   stairs: Optional[list] = None) -> None:
        name = os.path.splitext(os.path.basename(path))[0]
        # Queued after every first pass, so this wait never blocks on a task that has not started.
        self._first_pass_done[name].wait()
        if self.build_events[name].is_set():
            return                                  # the first pass already loaded its saved preview
        t0 = time.time()
        try:
            # Runs in self._preview_pool, not this thread.
            payload, was_cached = self._preview_pool.submit(
                build_bench.build_scan_payload_cached, path, cache_dir, exclusions, stairs,
                self.preview_options()).result()
            self._record_payload(name, payload, was_cached, t0)
        except Exception as exc:  # noqa: BLE001 - one bad scan must not take the others down
            log.exception("%s: preview build failed", name)
            with self._lock:
                self.build_errors[name] = f"{type(exc).__name__}: {exc}"
        finally:
            self.build_events[name].set()

    def wait_for_all(self, timeout: Optional[float] = None) -> None:
        """Block until every scan has built or failed. Alignment needs every preview; export does not call this,
        since place_merge reads full-resolution scans from disk."""
        for ev in self.build_events.values():
            ev.wait(timeout)

    def shutdown(self) -> None:
        """Drop unstarted preview builds when a different project is opened; do not wait for running ones."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._preview_pool.shutdown(wait=False, cancel_futures=True)


class Session:
    """Holds whichever project is open (self.app) plus the project routes' roots: projects_root, import_source
    (/scans) and legacy_out_dir."""

    def __init__(self, projects_root: str, import_source: str, legacy_out_dir: str,
                pattern: str = "*.pts"):
        self.projects_root = projects_root
        self.import_source = import_source
        self.legacy_out_dir = legacy_out_dir
        self.pattern = pattern
        self._page_html_cached = _build_page_html()
        self.app: Optional[App] = None
        self.active_project: Optional[str] = None
        self._lock = threading.Lock()
        os.makedirs(projects_root, exist_ok=True)

    @property
    def page_html(self) -> bytes:
        # Dev mode rebuilds per request so edits show on reload; release serves the startup copy.
        if os.environ.get("OPENPOINT_DEV_MODE"):
            return _build_page_html()
        return self._page_html_cached

    def legacy_status(self) -> dict:
        path = _placement_path(self.legacy_out_dir)
        return {"available": os.path.exists(path), "path": path}

    def open_project(self, name: str) -> None:
        """Swap in `name` as the active project; refuses while a job is running."""
        pdir = project_mod.project_dir(self.projects_root, name)
        raw_dir = project_mod.scans_raw_dir(pdir)
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"no project named {name!r} in {self.projects_root}")
        with self._lock:
            # Re-opening the open project with the same scan files keeps what is loaded, so a page reload does not
            # reload every preview. No tidy here; the first open already did it.
            if (self.app is not None and self.active_project == name and not self.app.build_errors
                    and self.app.scans_fingerprint == _scans_fingerprint(raw_dir, self.pattern)):
                self._stamp_opened(pdir, name)
                return
            if self.app is not None and self.app.jobs.any_running():
                raise RuntimeError("a job is running in the current project; "
                                    "wait for it to finish before switching projects")
            try:  # moves only, never deletes; a failure leaves the old flat layout, which still loads
                moved = project_mod.migrate_project_layout(pdir)
                if any(moved.values()):
                    log.info("%s: moved into subfolders: %s", name, moved)
            except OSError:
                log.exception("%s: folder tidy-up failed, keeping the old layout (not fatal)", name)
            _auto_tidy_backups(pdir)
            try:
                new_app = App(raw_dir, pdir, self.pattern)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"project {name!r} has no imported scans yet - import some "
                    f"before opening it") from None
            old_app = self.app
            self.app = new_app
            self.active_project = name
        if old_app is not None:
            old_app.shutdown()
        self._stamp_opened(pdir, name)

    @staticmethod
    def _stamp_opened(pdir: str, name: str) -> None:
        """Merge-write an open-time stamp; a failure here must not fail the open. Only the stamp changes, so no saved
        version is made first."""
        try:
            _write_placement_merged(pdir, lambda fresh: fresh.__setitem__(
                "last_opened", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), backup_first=False)
        except Exception:  # noqa: BLE001 - see the docstring
            log.exception("%s: could not record last_opened (not fatal)", name)


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.info("%s - %s", self.address_string(), fmt % args)

        def _json(self, status: int, payload) -> None:
            # Raise here on a stray NaN rather than send invalid JSON.
            self._json_bytes(status, json.dumps(payload, allow_nan=False).encode("utf-8"))

        def _json_bytes(self, status: int, body: bytes) -> None:
            self._reply(status, body, "application/json")

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def _open_app(self) -> Optional[App]:
            """Every route below that needs a project open goes through
            this - one place that writes the 409, instead of nine copies of
            the same three lines."""
            app = session.app
            if app is None:
                self._json(409, {"error": "no project open",
                                 "hint": "GET /api/projects, then POST /api/projects/<name>/open"})
                return None
            return app

        def do_GET(self):
            try:
                # Route on the path without its query string.
                _split = urllib.parse.urlsplit(self.path)
                path, query = _split.path, urllib.parse.parse_qs(_split.query)
                if path == "/" or path == "/index.html":
                    page = session.page_html  # one read: dev mode rebuilds this per access
                    self._reply(200, page, "text/html; charset=utf-8")
                elif path == "/api/projects":
                    # Needs no open project; reports any running job for the switch-project dialog.
                    running = session.app.jobs.running_job() if session.app else None
                    if running is not None:
                        running["project"] = session.active_project
                        running["elapsed_s"] = round(time.time() - running.get("started_at", time.time()), 1)
                    self._json(200, {
                        "projects": project_mod.list_projects(session.projects_root),
                        "active": session.active_project,
                        "legacy": session.legacy_status(),
                        "running_job": running,
                    })
                elif path == "/splash.png":
                    # Start-screen brand image; a file swap replaces it, no code change.
                    splash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splash.png")
                    if not os.path.isfile(splash_path):
                        self._json(404, {"error": "no splash.png shipped"})
                        return
                    with open(splash_path, "rb") as fh:
                        body = fh.read()
                    self._reply(200, body, "image/png")
                elif path.startswith("/api/projects/") and path.endswith("/" + THUMBNAIL_FILENAME):
                    # Per-project preview for the Open-existing modal; 404 if none yet.
                    name = urllib.parse.unquote(path[len("/api/projects/"):-len("/" + THUMBNAIL_FILENAME)])
                    tpath = os.path.join(project_mod.project_dir(session.projects_root, name),
                                         THUMBNAIL_FILENAME)
                    if not os.path.isfile(tpath):
                        self._json(404, {"error": "no thumbnail yet"})
                        return
                    with open(tpath, "rb") as fh:
                        body = fh.read()
                    self._reply(200, body, "image/png")
                elif path == "/api/import-source":
                    self._json(200, {"scans": project_mod.list_import_source(session.import_source)})
                elif path == "/api/scans":
                    app = self._open_app()
                    if app is None:
                        return
                    # Always fast: the scan list, which the page also polls to see which previews are ready. Never
                    # blocks on a build.
                    with app._lock:
                        rows = []
                        for meta in app.scan_meta:
                            name = meta["name"]
                            row = {"name": name, "points": meta["points"],
                                  "ready": name in app.payloads,
                                  # null until that scan's fast pass lands
                                  "quickBbox": app.quick_bbox.get(name)}
                            if name in app.build_errors:
                                row["error"] = app.build_errors[name]
                            rows.append(row)
                    self._json(200, {"scans": rows})
                elif path.startswith("/api/scans/") and path.endswith("/floors-ceilings"):
                    app = self._open_app()
                    if app is None:
                        return
                    name = urllib.parse.unquote(path[len("/api/scans/"):-len("/floors-ceilings")])
                    try:
                        pieces = app.floors_ceilings(name)
                    except KeyError:
                        self._json(404, {"error": f"no scan named {name!r} in this project"})
                        return
                    if pieces is None:
                        self._json(202, {"ready": False})       # its preview is still building
                    else:
                        self._json(200, {"name": name, "pieces": pieces})
                elif path.startswith("/api/scans/"):
                    app = self._open_app()
                    if app is None:
                        return
                    # One scan's full preview payload, fetched once the list
                    # above says it is ready. 202 means "still building, ask
                    # again"; the page's poll loop already expects that.
                    name = urllib.parse.unquote(path[len("/api/scans/"):])
                    with app._lock:
                        payload = app.payloads.get(name)
                        error = app.build_errors.get(name)
                        kept = app.wire_cache.get(name)
                    if payload is not None:
                        if kept is None or kept[0] is not payload:     # encoded once per preview
                            kept = (payload, json.dumps(build_bench.flatten_xy_for_wire(payload),
                                                        allow_nan=False).encode("utf-8"))
                            with app._lock:
                                app.wire_cache[name] = kept
                        self._json_bytes(200, kept[1])
                    elif error is not None:
                        self._json(500, {"error": error})
                    else:
                        self._json(202, {"ready": False})
                elif path == "/api/version":
                    self._json(200, build_info())
                elif path == "/api/gpu-status":
                    # The Settings GPU checkbox reads this.
                    self._json(200, limits.gpu_status())
                elif path == "/api/placement":
                    app = self._open_app()
                    if app is None:
                        return
                    data = _read_placement(app.out_dir)
                    if data is None:
                        self._json(404, {"error": "no saved placement yet"})
                    else:
                        self._json(200, data)
                elif path == "/api/placement/backups":
                    # List only: the page already holds the current placement and can compare a backup itself.
                    app = self._open_app()
                    if app is None:
                        return
                    self._json(200, {"backups": _list_backups(app.out_dir)})
                elif path == "/api/placement/backups-tidy":
                    app = self._open_app()
                    if app is None:
                        return
                    self._json(200, _tidy_backups(app.out_dir, dry_run=True))
                elif path.startswith("/api/placement/backups/"):
                    app = self._open_app()
                    if app is None:
                        return
                    backup_id = urllib.parse.unquote(path[len("/api/placement/backups/"):])
                    backup_path = _backup_file_path(app.out_dir, backup_id)
                    if backup_path is None:
                        self._json(404, {"error": f"no such backup {backup_id!r}"})
                        return
                    with open(backup_path) as fh:
                        self._json(200, json.load(fh))
                elif path == "/api/settings":
                    # Per-project settings, stored in the project file; {} when none have been changed.
                    app = self._open_app()
                    if app is None:
                        return
                    data = _read_placement(app.out_dir) or {}
                    self._json(200, data.get("settings", {}))
                elif path == "/api/deviation":
                    app = self._open_app()
                    if app is None:
                        return
                    # Computed on demand: it needs the saved placement and every preview, and is too slow to redo on
                    # every poll.
                    self._json(200, _compute_deviation(app))
                elif path == "/api/jobs":
                    app = self._open_app()
                    if app is None:
                        return
                    self._json(200, {"jobs": app.jobs.recent()})
                elif path.startswith("/api/jobs/"):
                    app = self._open_app()
                    if app is None:
                        return
                    job_id = path.rsplit("/", 1)[-1]
                    job = app.jobs.get(job_id)
                    if job is None:
                        self._json(404, {"error": f"no such job {job_id}"})
                    else:
                        self._json(200, job)
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                log.exception("GET %s failed", self.path)
                self._json(500, {"error": str(exc)})

        def do_POST(self):
            try:
                # Route on the path without its query string, as in do_GET.
                _split = urllib.parse.urlsplit(self.path)
                path, query = _split.path, urllib.parse.parse_qs(_split.query)
                if path == "/api/projects":
                    body = self._body()
                    try:
                        pdir = project_mod.create_project(session.projects_root, body.get("name", ""))
                    except project_mod.InvalidProjectName as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    except FileExistsError as exc:
                        self._json(409, {"error": str(exc)})
                        return
                    self._json(200, {"created": True, "name": os.path.basename(pdir)})
                elif path.startswith("/api/projects/") and path.endswith("/import"):
                    name = urllib.parse.unquote(path[len("/api/projects/"):-len("/import")])
                    body = self._body()
                    pdir = project_mod.project_dir(session.projects_root, name)
                    if not os.path.isdir(pdir):
                        self._json(404, {"error": f"no project named {name!r}"})
                        return
                    skipped: list = []
                    imported = project_mod.import_scans(
                        session.import_source, pdir, body.get("names"), skipped)
                    self._json(200, {"imported": imported, "n": len(imported), "skipped_already_there": skipped})
                elif path.startswith("/api/scans/") and path.endswith("/reprocess"):
                    # Rebuild one scan's preview with its saved exclusions.
                    app = self._open_app()
                    if app is None:
                        return
                    if app.jobs.any_running():
                        self._json(409, {"error": "a run is in progress; re-process this scan when it has finished"})
                        return
                    name = urllib.parse.unquote(path[len("/api/scans/"):-len("/reprocess")])
                    try:
                        self._json(200, app.reprocess(name))
                    except KeyError:
                        self._json(404, {"error": f"no scan named {name!r} in this project"})
                elif path.startswith("/api/scans/") and path.endswith("/rebuild"):
                    # Rebuild one scan's preview with the project's current settings (its levelling switch).
                    app = self._open_app()
                    if app is None:
                        return
                    if app.jobs.any_running():
                        self._json(409, {"error": "a run is in progress; change this scan's levelling when it has finished"})
                        return
                    name = urllib.parse.unquote(path[len("/api/scans/"):-len("/rebuild")])
                    try:
                        self._json(200, app.rebuild(name))
                    except KeyError:
                        self._json(404, {"error": f"no scan named {name!r} in this project"})
                elif path.startswith("/api/projects/") and path.endswith("/upload"):
                    # One .pts per request, the raw file as the body.
                    name = urllib.parse.unquote(path[len("/api/projects/"):-len("/upload")])
                    pdir = project_mod.project_dir(session.projects_root, name)
                    if not os.path.isdir(pdir):
                        self._json(404, {"error": f"no project named {name!r}"})
                        return
                    length = int(self.headers.get("Content-Length", 0))
                    filename = (query.get("file") or [""])[0]
                    keep_both = (query.get("keep_both") or ["0"])[0] == "1"
                    body_in = _CountedReader(self.rfile)
                    try:
                        filename = project_mod.upload_name(pdir, filename, keep_both)
                        got = project_mod.import_upload(pdir, filename, body_in, length)
                    except project_mod.DuplicateScan as exc:
                        got = {"error": str(exc), "duplicate": True, "suggested": exc.suggested}
                    except ValueError as exc:
                        got = {"error": str(exc)}
                    body_in.drain(length)                          # a refused upload is still read, or the browser sees a reset
                    self._json(409 if got.get("duplicate") else 400 if "error" in got else 200, got)
                elif path.startswith("/api/projects/") and path.endswith("/open"):
                    name = urllib.parse.unquote(path[len("/api/projects/"):-len("/open")])
                    try:
                        session.open_project(name)
                    except FileNotFoundError as exc:
                        self._json(404, {"error": str(exc)})
                        return
                    except RuntimeError as exc:
                        self._json(409, {"error": str(exc)})
                        return
                    self._json(200, {"opened": name})
                elif path == "/api/projects/legacy-import":
                    body = self._body()
                    status = session.legacy_status()
                    if not status["available"]:
                        self._json(404, {"error": "no legacy placement.json to import"})
                        return
                    try:
                        pdir = project_mod.create_project(session.projects_root, body.get("name", ""))
                    except project_mod.InvalidProjectName as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    except FileExistsError as exc:
                        self._json(409, {"error": str(exc)})
                        return
                    imported = project_mod.import_scans(session.import_source, pdir)
                    # Same file, same schema (see project.py's module
                    # docstring) - carrying it forward is a straight copy,
                    # not a field-by-field migration.
                    shutil.copy2(_placement_path(session.legacy_out_dir), _placement_path(pdir))
                    self._json(200, {"created": True, "name": os.path.basename(pdir),
                                     "imported": imported, "n_imported": len(imported)})
                elif path == "/api/placement":
                    app = self._open_app()
                    if app is None:
                        return
                    data = self._body()
                    if not is_placement_schema(data.get("schema")):
                        self._json(400, {"error": f"expected schema {PLACEMENT_SCHEMA!r}"})
                        return
                    data["schema"] = PLACEMENT_SCHEMA
                    data = _normalise_incoming_placement(data)
                    _write_whole_placement(app.out_dir, data)
                    self._json(200, {"saved": True, "path": _placement_path(app.out_dir)})
                elif path.startswith("/api/placement/backups/") and path.endswith("/pin"):
                    app = self._open_app()
                    if app is None:
                        return
                    backup_id = urllib.parse.unquote(path[len("/api/placement/backups/"):-len("/pin")])
                    if _backup_file_path(app.out_dir, backup_id) is None:
                        self._json(404, {"error": f"no such saved version {backup_id!r}"})
                        return
                    pins = project_mod.set_pinned(app.out_dir, backup_id, bool(self._body().get("pinned", True)))
                    self._json(200, {"pinned": backup_id in pins})
                elif path == "/api/placement/backups-tidy":
                    app = self._open_app()
                    if app is None:
                        return
                    self._json(200, _tidy_backups(app.out_dir, dry_run=False))
                elif path.startswith("/api/placement/backups/") and path.endswith("/restore"):
                    # Restoring is a save through the ordinary path, so whatever was current is backed up first and the
                    # restore can be undone like any save.
                    app = self._open_app()
                    if app is None:
                        return
                    if app.jobs.any_running():
                        self._json(409, {"error": "a job is running; wait for it to finish before restoring a backup"})
                        return
                    backup_id = urllib.parse.unquote(
                        path[len("/api/placement/backups/"):-len("/restore")])
                    backup_path = _backup_file_path(app.out_dir, backup_id)
                    if backup_path is None:
                        self._json(404, {"error": f"no such backup {backup_id!r}"})
                        return
                    with open(backup_path) as fh:
                        data = json.load(fh)
                    if not is_placement_schema(data.get("schema")):
                        self._json(400, {"error": f"backup {backup_id!r} has an unrecognised schema, refusing to restore it"})
                        return
                    data["schema"] = PLACEMENT_SCHEMA
                    _write_whole_placement(app.out_dir, data)
                    self._json(200, {"restored": True, "path": _placement_path(app.out_dir)})
                elif path == "/api/tilt-suggestion":
                    app = self._open_app()
                    if app is None:
                        return
                    self._json(200, _compute_tilt_suggestion(app, self._body()))
                elif path == "/api/hand-moves":
                    # Append-only log of finished hand moves.
                    app = self._open_app()
                    if app is None:
                        return
                    written = project_mod.append_hand_moves(app.out_dir, self._body().get("moves", []))
                    self._json(200, {"written": written, "file": project_mod.HAND_MOVES_FILE})
                elif path == "/api/settings":
                    app = self._open_app()
                    if app is None:
                        return
                    incoming = self._body()

                    def _apply(fresh: dict) -> None:
                        fresh["settings"] = incoming

                    _write_placement_merged(app.out_dir, _apply)
                    _apply_process_settings(incoming)
                    self._json(200, {"saved": True})
                elif path == "/api/jobs":
                    app = self._open_app()
                    if app is None:
                        return
                    body = self._body()
                    kind = body.get("kind")
                    if kind not in ("align", "align_auto", "export", "export_plan", "export_3d", "wall_report"):
                        self._json(400, {"error": "kind must be 'align', 'align_auto', 'export', 'export_plan' or 'wall_report'"})
                        return
                    # One lock acquisition for the check and the create (try_start), so two starts cannot race.
                    job_id = app.jobs.try_start(kind)
                    if job_id is None:
                        self._json(409, {"error": "a job is already running; wait for it to finish"})
                        return
                    fn = {"align": _run_align, "align_auto": _run_align_auto,
                          "export": _run_export, "export_plan": _run_export_plan,
                          "export_3d": _run_export_3d, "wall_report": _run_wall_report}[kind]
                    try:
                        # The job runs in its own process so Cancel can stop it anywhere.
                        threading.Thread(target=run_job_in_process,
                                         args=(app, job_id, f"open_point_stitch.server:{fn.__name__}", body),
                                         daemon=True).start()
                    except Exception:
                        # Fail the job so no phantom "running" job remains.
                        app.jobs.fail(job_id, "could not start the job thread")
                        raise
                    self._json(202, {"id": job_id})
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    # Sets the flag only: the job checks it between steps, so it stops soon, not instantly.
                    app = self._open_app()
                    if app is None:
                        return
                    job_id = urllib.parse.unquote(path[len("/api/jobs/"):-len("/cancel")])
                    if not app.jobs.request_cancel(job_id):
                        self._json(404, {"error": "no running job with that id"})
                        return
                    self._json(202, {"cancelling": True})
                else:
                    self._json(404, {"error": "not found"})
            except json.JSONDecodeError as exc:
                # Malformed JSON is the caller's error: answer 400, not 500.
                log.warning("POST %s: malformed JSON body (%s)", self.path, exc)
                self._json(400, {"error": f"malformed JSON body: {exc}"})
            except Exception as exc:  # noqa: BLE001
                log.exception("POST %s failed", self.path)
                self._json(500, {"error": str(exc)})

    return Handler


def serve(projects_root: str, import_source: str, legacy_out_dir: str,
         host: str = "0.0.0.0", port: int = 8080, pattern: str = "*.pts") -> None:
    info = build_info()
    log.info("open-point-stitcher %s, built %s, %s", info["version"], info["build_time"], info["mode"])
    session = Session(projects_root, import_source, legacy_out_dir, pattern)
    httpd = ThreadingHTTPServer((host, port), make_handler(session))
    log.info("serving on http://%s:%d (open http://localhost:%d in a browser)",
             host, port, port)
    log.info("%d project(s) found in %s%s", len(project_mod.list_projects(projects_root)),
             projects_root,
             " (legacy placement.json available to import)" if session.legacy_status()["available"] else "")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scans", default="/scans",
                    help="import SOURCE only - .pts files here are copied into a "
                         "project's own scans-raw/ by Import, never read directly otherwise")
    ap.add_argument("--out", default="/out",
                    help="legacy single-folder placement.json location, kept only so an "
                         "existing session's work can be imported into a project once "
                         "(see /api/projects/legacy-import) - a project's real output lives "
                         "inside its own folder under --projects, not here")
    ap.add_argument("--projects", default="/projects",
                    help="where project folders live")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--pattern", default="*.pts")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    serve(args.projects, args.scans, args.out, args.host, args.port, args.pattern)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
