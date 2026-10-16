"""Each job runs in its own process so Cancel can stop it at any point: the server stops the job's whole process
group, which also frees its workers and GPU memory. A job says when it starts writing results (commit); from then on
it finishes rather than stops, so nothing is left half-written."""
from __future__ import annotations

import glob
import importlib
import logging
import multiprocessing
import os
import queue
import signal
import threading
import time

log = logging.getLogger(__name__)

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
