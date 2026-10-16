"""Tests for jobs.JobStore: one job at a time, old finished jobs forgotten, and a restart that reports what
happened to a job it interrupted."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.jobs import JobStore  # noqa: E402


def test_only_one_job_runs_at_a_time_and_a_finished_one_frees_the_slot():
    jobs = JobStore()
    first = jobs.try_start("align_auto")
    assert first and jobs.try_start("export") is None
    jobs.finish(first, {"ok": True})
    assert jobs.try_start("export") is not None


def test_finished_jobs_beyond_the_keep_limit_are_forgotten_oldest_first():
    jobs = JobStore(keep=3)
    ids = []
    for i in range(6):
        job = jobs.try_start("export")
        jobs.finish(job, {"n": i})
        ids.append(job)
    assert [jobs.get(i) is not None for i in ids] == [False, False, False, True, True, True]


def test_a_running_job_is_never_evicted():
    jobs = JobStore(keep=1)
    running = jobs.create("align_auto")          # create() skips the one-at-a-time guard, which try_start() applies
    for other in [jobs.create("export") for _ in range(3)]:
        jobs.finish(other, {})
    assert jobs.get(running)["state"] == "running"


def test_records_are_kept_on_disk_and_old_ones_are_deleted_with_the_job():
    with tempfile.TemporaryDirectory() as d:
        jobs = JobStore(d, keep=2)
        made = []
        for i in range(4):
            job = jobs.try_start("export")
            jobs.finish(job, {"n": i})
            made.append(job)
        assert sorted(os.listdir(d)) == sorted(f"{j}.json" for j in made[-2:])


def test_a_job_running_when_the_server_stopped_comes_back_as_interrupted():
    with tempfile.TemporaryDirectory() as d:
        before = JobStore(d)
        lost = before.try_start("align_auto")
        after = JobStore(d)                       # the "restart"
        job = after.get(lost)
        assert job["state"] == "interrupted" and "restarted" in job["error"], job
        assert after.try_start("export") is not None, "an interrupted job must not block the next one"
        assert after.recent()[0]["kind"] in ("export", "align_auto")


def test_finished_results_survive_a_restart_and_recent_leaves_the_result_out():
    with tempfile.TemporaryDirectory() as d:
        jobs = JobStore(d)
        job = jobs.try_start("export")
        jobs.finish(job, {"points": 123})
        again = JobStore(d)
        assert again.get(job)["result"] == {"points": 123} and again.get(job)["state"] == "done"
        assert "result" not in again.recent()[0]


def test_a_cancelled_job_keeps_its_partial_result_and_is_not_an_error():
    jobs = JobStore()
    job = jobs.try_start("align_auto")
    assert jobs.request_cancel(job) and jobs.is_cancel_requested(job)
    jobs.finish_cancelled(job, {"n_pairs_completed": 3})
    got = jobs.get(job)
    assert got["state"] == "cancelled" and got["result"]["n_pairs_completed"] == 3 and got["error"] is None
    assert not jobs.request_cancel(job), "a finished job cannot be cancelled"


def test_an_unwritable_folder_does_not_fail_the_job():
    with tempfile.TemporaryDirectory() as d:
        blocker = os.path.join(d, "blocker")
        open(blocker, "w").close()
        jobs = JobStore(os.path.join(blocker, "jobs"))     # a file where the folder should be
        job = jobs.try_start("export")
        jobs.finish(job, {"ok": 1})
        assert jobs.get(job)["state"] == "done"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
