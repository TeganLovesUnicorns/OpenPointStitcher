"""Jobs for the bench server: what is running, what has finished, and a small record on disk so a restart does not
make a job vanish; and running each job in its own process so Cancel can stop it at any point."""
from __future__ import annotations

import copy
import glob
import importlib
import json
import logging
import multiprocessing
import os
import queue
import signal
import threading
import time
import uuid
from typing import Optional

from .lib.files import write_json_atomic

log = logging.getLogger(__name__)

KEEP_FINISHED = 25   # finished jobs kept in memory and on disk; a 91-pair result is large

# Diagnostics: OPENPOINT_CANCEL_TRACE=<file> appends one line per cancel check and per stage change, to find the
# longest stretch a stage goes without checking Cancel.
_TRACE = os.environ.get("OPENPOINT_CANCEL_TRACE")


def _trace(job: dict, what: str) -> None:
    try:
        with open(_TRACE, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "job": job["id"], "kind": job["kind"], "what": what,
                                 "phase": job["progress"]["phase"], "detail": job["progress"]["detail"]}) + "\n")
    except OSError:
        pass


class JobStore:
    def __init__(self, persist_dir: Optional[str] = None, keep: int = KEEP_FINISHED):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._dir = persist_dir
        self._keep = keep
        if persist_dir:
            self._load_previous()

    # -- on disk ----------------------------------------------------------------------------------
    def _path(self, job_id: str) -> str:
        return os.path.join(self._dir, f"{job_id}.json")

    def _save(self, job: dict) -> None:
        """Write one job's record. Best effort: a job must never fail because its record could not be written."""
        if not self._dir:
            return
        try:
            write_json_atomic(self._path(job["id"]), job, default=str)
        except OSError:
            log.exception("could not write the record of job %s", job.get("id"))

    def _load_previous(self) -> None:
        """Bring back what an earlier server process left. A job recorded as running was running when that
        process stopped, so it did not finish: say so, once, instead of letting it look like it never existed."""
        if not os.path.isdir(self._dir):
            return
        for name in os.listdir(self._dir):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._dir, name)) as fh:
                    job = json.load(fh)
                if job.get("state") == "running":
                    job["state"] = "interrupted"
                    job["error"] = "the server restarted while this job was running; it did not finish"
                    job["finished_at"] = time.time()
                    self._save(job)
                self._jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                log.warning("ignoring an unreadable job record %s", name)
        self._evict()

    def _evict(self) -> None:
        """Keep the newest `keep` finished jobs; forget the rest, in memory and on disk. Called with the lock held
        or before anything else can see the store."""
        finished = sorted((j for j in self._jobs.values() if j["state"] != "running"),
                          key=lambda j: j.get("finished_at") or j.get("started_at") or 0.0, reverse=True)
        for job in finished[self._keep:]:
            self._jobs.pop(job["id"], None)
            if self._dir:
                try:
                    os.remove(self._path(job["id"]))
                except OSError:
                    pass

    def _end(self, job_id: str, state: str, result=None, error: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["state"] = state
            job["finished_at"] = time.time()
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
            self._save(job)
            self._evict()

    # -- starting ---------------------------------------------------------------------------------
    @staticmethod
    def _new(kind: str) -> dict:
        # `started_at` is top level, not inside `progress`, which set_progress() replaces wholesale.
        return {"id": uuid.uuid4().hex[:12], "kind": kind, "state": "running",
                "progress": {"phase": "starting", "detail": ""}, "result": None, "error": None,
                "started_at": time.time(), "finished_at": None, "cancel_requested": False}

    def create(self, kind: str) -> str:
        job = self._new(kind)
        with self._lock:
            self._jobs[job["id"]] = job
            self._save(job)
        return job["id"]

    def any_running(self) -> bool:
        """Server-side guard against a double click starting two runs."""
        with self._lock:
            return any(j["state"] == "running" for j in self._jobs.values())

    def try_start(self, kind: str) -> Optional[str]:
        """Check and create under one lock, so two POSTs cannot both start. None if a job is already running."""
        job = self._new(kind)
        with self._lock:
            if any(j["state"] == "running" for j in self._jobs.values()):
                return None
            self._jobs[job["id"]] = job
            self._save(job)
        return job["id"]

    def set_progress(self, job_id: str, phase: str, detail: str = "") -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                new_phase = job["progress"]["phase"] != phase
                job["progress"] = {"phase": phase, "detail": detail}
                if new_phase:                      # so a record left by a stopped server names the stage it reached
                    self._save(job)
                if _TRACE:
                    _trace(self._jobs[job_id], "phase")

    # -- ending -----------------------------------------------------------------------------------
    def commit(self, job_id: str) -> None:
        """A job is about to write its results. Here, in the server's own thread, nothing to do; in a job process
        (_ChildJobs) it is the point after which Cancel waits for the job to finish."""

    def set_committing(self, job_id: str) -> None:
        """The job is writing its results; Cancel can no longer stop it, and the page says so."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["committing"] = True

    def finish(self, job_id: str, result) -> None:
        self._end(job_id, "done", result=result)

    def fail(self, job_id: str, error: str) -> None:
        self._end(job_id, "error", error=error)

    def finish_cancelled(self, job_id: str, result) -> None:
        """Distinct from finish()/fail(): the job did real, keepable work and was stopped on request, not
        because anything went wrong, so the interface can say so rather than present it as a failure."""
        self._end(job_id, "cancelled", result=result)

    # -- asking -----------------------------------------------------------------------------------
    def running_job(self) -> Optional[dict]:
        """A job running server-side, described well enough that the picker's banner can name it."""
        with self._lock:
            for j in self._jobs.values():
                if j["state"] == "running":
                    return copy.deepcopy(j)
            return None

    def request_cancel(self, job_id: str) -> bool:
        """Sets the flag only; the job checks it between units of work, so a stop lands after the current
        one. False if missing or finished."""
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None or j["state"] != "running":
                return False
            j["cancel_requested"] = True
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(job_id)
            if _TRACE and j:                       # diagnostics only
                _trace(j, "check")
            return bool(j and j.get("cancel_requested"))

    def get(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def recent(self, limit: int = 10) -> list[dict]:
        """The newest jobs, without their (large) results, newest first."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.get("started_at") or 0.0, reverse=True)[:limit]
            return [{k: v for k, v in copy.deepcopy(j).items() if k != "result"} for j in jobs]


# -------------------------------------------------------------------------------------------------------------------
# Each job runs in its own process so Cancel can stop it at any point: the server stops the job's whole process
# group, which also frees its workers and GPU memory. A job says when it starts writing results (commit); from then on
# it finishes rather than stops, so nothing is left half-written.
# -------------------------------------------------------------------------------------------------------------------

RUNNING, COMMITTING, STOPPED = 0, 1, 2
POLL_S = 0.2


class _ChildJobs:
    """Stands in for the server's JobStore inside the job process: everything is sent to the server."""

    def __init__(self, q, state, lock):
        self._q, self._state, self._lock = q, state, lock

    def set_progress(self, job_id: str, phase: str, detail: str = "") -> None:
        self._q.put(("progress", phase, detail))

    def is_cancel_requested(self, job_id: str) -> bool:
        return False                                  # the server stops the process instead

    def commit(self, job_id: str) -> None:
        with self._lock:
            if self._state.value == STOPPED:          # stopped a moment ago; never start writing
                os._exit(0)
            self._state.value = COMMITTING
        self._q.put(("commit",))

    def finish(self, job_id: str, result) -> None:
        self._q.put(("finish", result))

    def fail(self, job_id: str, error: str) -> None:
        self._q.put(("fail", error))

    def finish_cancelled(self, job_id: str, result) -> None:
        self._q.put(("cancelled", result))


class _ChildApp:
    """What a job reads from the open project, copied in when it starts."""

    def __init__(self, snapshot: dict, jobs: _ChildJobs):
        self.scans_dir = snapshot["scans_dir"]
        self.out_dir = snapshot["out_dir"]
        self.payloads = snapshot["payloads"]
        self.build_errors = snapshot["build_errors"]
        self._lock = threading.Lock()
        self.jobs = jobs

    def wait_for_all(self, timeout=None) -> None:
        pass                                          # the server waited before starting this process


def _child_main(fn_name: str, job_id: str, params: dict, snapshot: dict, q, state, lock) -> None:
    try:
        os.setpgid(0, 0)                              # its own group, so stopping it stops its workers too
    except (AttributeError, OSError):
        pass
    logging.basicConfig(level=snapshot.get("log_level", logging.INFO), format="%(levelname)-7s %(message)s")
    if snapshot.get("level_tilt_deg") is not None:     # a fresh process: the project's levelling setting, as the server has it
        from . import align_gravity
        align_gravity.set_default_max_tilt(float(snapshot["level_tilt_deg"]))
    module, _, name = fn_name.partition(":")          # "open_point_stitch.server:_run_export"
    try:
        fn = getattr(importlib.import_module(module), name)
        fn(_ChildApp(snapshot, _ChildJobs(q, state, lock)), job_id, params)
    except BaseException as exc:  # noqa: BLE001 - whatever happens, the server must hear about it
        q.put(("fail", f"{type(exc).__name__}: {exc}"))
    q.put(("exit",))
    q.close()
    q.join_thread()


def _stop_group(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        proc.kill()


def remove_partial_files(out_dir: str) -> list[str]:
    """Files a stopped job may have been half way through: `*.partial.*` exports and `*.tmp` writes."""
    gone = []
    for pattern in ("*.tmp", "exports/*.partial.*", "exports/*.tmp", "reports/*.tmp"):
        for path in glob.glob(os.path.join(out_dir, pattern)):
            try:
                os.remove(path)
                gone.append(path)
            except OSError:
                pass
    return gone


def run_job_in_process(app, job_id: str, fn_name: str, params: dict) -> None:
    """Run `fn_name` ("module:function", called as fn(app, job_id, params)) in a child process, relaying its progress
    and result into the server's JobStore; stop it at once when Cancel is pressed, unless it is writing its results."""
    t0 = time.time()
    while not all(ev.is_set() for ev in app.build_events.values()):     # previews first, still stoppable
        if app.jobs.is_cancel_requested(job_id):
            app.jobs.finish_cancelled(job_id, {"cancelled": True, "stopped_during": "waiting for scan previews",
                                               "stopped_after_s": round(time.time() - t0, 1)})
            return
        time.sleep(POLL_S)
    with app._lock:
        from . import align_gravity
        snapshot = {"scans_dir": app.scans_dir, "out_dir": app.out_dir, "payloads": dict(app.payloads),
                    "build_errors": dict(app.build_errors), "log_level": logging.getLogger().level,
                    "level_tilt_deg": align_gravity.DEFAULT_MAX_TILT_DEG}
    ctx = multiprocessing.get_context("spawn")
    q, state, lock = ctx.Queue(), ctx.Value("i", RUNNING), ctx.Lock()
    proc = ctx.Process(target=_child_main, args=(fn_name, job_id, params, snapshot, q, state, lock), daemon=False)
    proc.start()
    phase, ended, said_saving = "starting", False, False
    while True:
        try:
            msg = q.get(timeout=POLL_S)
        except queue.Empty:
            msg = None
        if msg:
            kind = msg[0]
            if kind == "progress":
                phase = msg[1]
                app.jobs.set_progress(job_id, msg[1], msg[2])
            elif kind == "commit":
                app.jobs.set_committing(job_id)
            elif kind in ("finish", "fail", "cancelled"):
                {"finish": app.jobs.finish, "fail": app.jobs.fail, "cancelled": app.jobs.finish_cancelled}[kind](
                    job_id, msg[1])
                ended = True
            elif kind == "exit":
                break
        if not ended and app.jobs.is_cancel_requested(job_id):
            with lock:
                stop = state.value == RUNNING
                if stop:
                    state.value = STOPPED
                    _stop_group(proc)
            if stop:
                proc.join(10)
                gone = remove_partial_files(app.out_dir)
                log.info("job %s stopped during %r (%d half-written file(s) removed)", job_id, phase, len(gone))
                app.jobs.finish_cancelled(job_id, {"cancelled": True, "stopped_during": phase,
                                                   "stopped_after_s": round(time.time() - t0, 1)})
                return
            if not said_saving:
                said_saving = True
                app.jobs.set_progress(job_id, phase, "saving the results now; this part cannot be stopped")
        if not proc.is_alive():
            if q.empty():
                break
    proc.join(10)
    if not ended:
        app.jobs.fail(job_id, f"the job stopped unexpectedly (exit code {proc.exitcode}) during {phase!r}")
