"""Tests for jobs.py's job processes: a job runs in its own process and Cancel stops it at any point, worker processes and
half-written files included, except once it is saving its results."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.jobs import run_job_in_process  # noqa: E402
from open_point_stitch.jobs import JobStore  # noqa: E402

ME = "test_job_process"            # this file, as the job processes import it


# ---- stand-in jobs, run in the job's own process --------------------------------------------------------------
def stuck_job(app, job_id, params):
    """One long call that never looks at Cancel, plus a worker process of its own."""
    app.jobs.set_progress(job_id, "one long call", "")
    worker = subprocess.Popen(["sleep", "300"])
    with open(os.path.join(app.out_dir, "worker.pid"), "w") as fh:
        fh.write(str(worker.pid))
    time.sleep(60)
    app.jobs.finish(job_id, {"never": True})


def half_written_export(app, job_id, params):
    os.makedirs(os.path.join(app.out_dir, "exports"), exist_ok=True)
    app.jobs.set_progress(job_id, "writing", "")
    with open(os.path.join(app.out_dir, "exports", "merged-1.partial.pts"), "w") as fh:
        fh.write("3\n")
        fh.flush()
        time.sleep(60)


def saving_job(app, job_id, params):
    app.jobs.set_progress(job_id, "saving", "")
    app.jobs.commit(job_id)
    time.sleep(2)
    app.jobs.finish(job_id, {"saved": True})


def broken_job(app, job_id, params):
    raise ValueError("no such scan")


# ---- the server side ----------------------------------------------------------------------------------------------
class _App:
    def __init__(self, out_dir):
        self.scans_dir, self.out_dir = out_dir, out_dir
        self.payloads, self.build_errors, self.build_events = {"a": {"xy": [1, 2]}}, {}, {}
        self._lock = threading.Lock()
        self.jobs = JobStore()


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _start(app, fn, params=None):
    job = app.jobs.create("test")
    th = threading.Thread(target=run_job_in_process, args=(app, job, f"{ME}:{fn}", params or {}))
    th.start()
    return job, th


def _wait_phase(app, job, phase, limit=60):
    t0 = time.time()
    while app.jobs.get(job)["progress"]["phase"] != phase:
        assert time.time() - t0 < limit, app.jobs.get(job)
        time.sleep(0.05)


def test_cancel_stops_a_job_stuck_in_one_long_call_and_its_worker_at_once():
    with tempfile.TemporaryDirectory() as tmp:
        app = _App(tmp)
        job, th = _start(app, "stuck_job")
        _wait_phase(app, job, "one long call")
        time.sleep(1.0)
        worker = int(open(os.path.join(tmp, "worker.pid")).read())
        t0 = time.time()
        app.jobs.request_cancel(job)
        th.join(10)
        took = time.time() - t0
        j = app.jobs.get(job)
        assert j["state"] == "cancelled" and j["result"]["stopped_during"] == "one long call", j
        assert took < 2.0, f"took {took:.1f} s to stop"
        time.sleep(0.2)
        assert not _alive(worker), "the job's own worker process was left running"


def test_a_stopped_export_leaves_no_half_written_file():
    with tempfile.TemporaryDirectory() as tmp:
        app = _App(tmp)
        job, th = _start(app, "half_written_export")
        _wait_phase(app, job, "writing")
        time.sleep(0.5)
        assert os.listdir(os.path.join(tmp, "exports")) == ["merged-1.partial.pts"]
        app.jobs.request_cancel(job)
        th.join(10)
        assert app.jobs.get(job)["state"] == "cancelled"
        assert os.listdir(os.path.join(tmp, "exports")) == [], "a half-written export was left behind"


def test_a_job_already_saving_its_results_finishes_and_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        app = _App(tmp)
        job, th = _start(app, "saving_job")
        _wait_phase(app, job, "saving")
        t0 = time.time()
        while not app.jobs.get(job).get("committing"):
            assert time.time() - t0 < 10
            time.sleep(0.05)
        app.jobs.request_cancel(job)
        time.sleep(0.6)
        assert "cannot be stopped" in app.jobs.get(job)["progress"]["detail"], app.jobs.get(job)
        th.join(15)
        j = app.jobs.get(job)
        assert j["state"] == "done" and j["result"] == {"saved": True}, j


def test_a_job_that_fails_is_reported_as_failed():
    with tempfile.TemporaryDirectory() as tmp:
        app = _App(tmp)
        job, th = _start(app, "broken_job")
        th.join(60)
        j = app.jobs.get(job)
        assert j["state"] == "error" and "no such scan" in j["error"], j


def test_cancel_while_previews_are_still_building_stops_before_any_process():
    import threading as _t
    with tempfile.TemporaryDirectory() as tmp:
        app = _App(tmp)
        app.build_events = {"a": _t.Event()}
        job, th = _start(app, "stuck_job")
        time.sleep(0.5)
        app.jobs.request_cancel(job)
        th.join(5)
        j = app.jobs.get(job)
        assert j["state"] == "cancelled" and j["result"]["stopped_during"] == "waiting for scan previews", j
        assert not os.path.exists(os.path.join(tmp, "worker.pid"))


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
