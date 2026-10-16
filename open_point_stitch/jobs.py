"""Job bookkeeping for the bench server: what is running, what has finished, and a small record on disk so a
restart does not make a job vanish."""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from typing import Optional

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
            os.makedirs(self._dir, exist_ok=True)
            tmp = self._path(job["id"]) + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(job, fh, default=str)
            os.replace(tmp, self._path(job["id"]))
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
                self._jobs[job_id]["progress"] = {"phase": phase, "detail": detail}
                if _TRACE:
                    _trace(self._jobs[job_id], "phase")

    # -- ending -----------------------------------------------------------------------------------
    def commit(self, job_id: str) -> None:
        """A job is about to write its results. Here, in the server's own thread, nothing to do; in a job process
        (job_process._ChildJobs) it is the point after which Cancel waits for the job to finish."""

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
