"""The bench's "Export merged .pts" job, run to the end on the three synthetic acceptance scans: the file it writes,
the project's export setting, the record it leaves, and a plain failure with nothing saved. The merge itself is
tested in test_acceptance.py."""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from open_point_stitch.jobs import JobStore  # noqa: E402
from open_point_stitch.placement import SCHEMA  # noqa: E402
from open_point_stitch.pts_file import read_pts  # noqa: E402
from open_point_stitch.server import _run_export  # noqa: E402
from test_acceptance import FIXTURE_DIR, _fixture_files, _validate_output  # noqa: E402


def _project(tmp: str, keep_excluded: bool | None) -> tuple[SimpleNamespace, int, int]:
    """A project on the fixture scans, identity placement, one exclude region over part of the first scan.
    Returns the app the job needs, every point in, and the points inside the region."""
    files = _fixture_files()
    names = [os.path.splitext(os.path.basename(p))[0] for p in files]
    raw = read_pts(files[0])
    lo, hi = raw.points.min(axis=0), raw.points.max(axis=0)
    centroid = ((lo + hi) / 2).tolist()
    extent = ((hi - lo) / 2).tolist()
    inside = int(np.all(np.abs(raw.points - np.asarray(centroid)) <= np.maximum(np.asarray(extent) / 2, 0.05), axis=1).sum())
    data = {"schema": SCHEMA, "scans": [{"name": n} for n in names],
            "exclusions": [{"id": "r1", "scan": names[0], "points": [], "centroid": centroid, "extent": extent,
                            "n_points": inside}]}
    if keep_excluded is not None:
        data["settings"] = {"export_keeps_excluded": keep_excluded}
    out_dir = os.path.join(tmp, "project")
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "placement.json"), "w") as fh:
        json.dump(data, fh)
    total = sum(read_pts(p).n_points for p in files)
    return SimpleNamespace(out_dir=out_dir, scans_dir=FIXTURE_DIR, jobs=JobStore()), total, inside


def _export(app, params=None) -> dict:
    job_id = app.jobs.create("export")
    _run_export(app, job_id, params or {"threads": 2, "gpu": False})
    return app.jobs.get(job_id)


def test_export_job_writes_every_point_and_leaves_a_record():
    with tempfile.TemporaryDirectory() as tmp:
        app, total, inside = _project(tmp, None)
        assert 0 < inside < total
        job = _export(app)
        assert job["state"] == "done", job.get("error")
        path = job["result"]["output_path"]
        assert os.path.dirname(path).endswith("exports") and path.endswith(".pts"), path
        problems = _validate_output(path, total)                    # exclude regions stay in by default
        assert not problems, problems
        assert job["result"]["points"] == total
        assert not glob.glob(os.path.join(os.path.dirname(path), "*.partial.pts")), "a .partial file was left behind"
        with open(os.path.join(app.out_dir, "placement.json")) as fh:
            last = json.load(fh)["last_run"]
        assert last["kind"] == "export" and last["points"] == total, last
        with open(os.path.join(os.path.dirname(path), "latest.json")) as fh:
            assert json.load(fh)["merged"] == os.path.basename(path)          # a file name, not a path
        print(f"    {total:,} points written to exports/, seven columns, recorded in the project")


def test_export_job_follows_the_projects_setting_to_leave_exclude_regions_out():
    with tempfile.TemporaryDirectory() as tmp:
        app, total, inside = _project(tmp, False)
        job = _export(app)
        assert job["state"] == "done", job.get("error")
        problems = _validate_output(job["result"]["output_path"], total - inside)
        assert not problems, problems
        print(f"    setting off: {inside:,} points in the exclude region left out, {total - inside:,} written")


def test_export_job_with_no_saved_placement_fails_in_plain_words():
    with tempfile.TemporaryDirectory() as tmp:
        app = SimpleNamespace(out_dir=tmp, scans_dir=FIXTURE_DIR, jobs=JobStore())
        job = _export(app)
        assert job["state"] == "error" and "Save a placement" in job["error"], job
        assert not os.path.exists(os.path.join(tmp, "exports")) or not os.listdir(os.path.join(tmp, "exports"))
        print("    no placement: the job fails and says to save one first")


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
