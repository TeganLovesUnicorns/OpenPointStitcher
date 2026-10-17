"""Which test file runs at which QA speed, and which can run side by side. The one list tools/qa.py and
docker-entrypoint.sh test both read. A test file in tests/ that is in no layer fails QA."""
from __future__ import annotations

import ast
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# QUICK: pure checks on synthetic data. No real scans, no browser, no port, no shared folder: safe side by side.
QUICK = [
    "tests/test_ptsio.py", "tests/test_manual.py", "tests/test_placement.py", "tests/test_placement_backup.py",
    "tests/test_projects.py", "tests/test_exclusions.py", "tests/test_explain_link.py", "tests/test_overlap.py", "tests/test_resources.py",
    "tests/test_align.py", "tests/test_pairwise.py", "tests/test_server.py", "tests/test_surfaces.py",
    "tests/test_corners.py", "tests/test_colour.py", "tests/test_score_vote.py", "tests/test_pair_filter.py",
    "tests/test_report.py", "tests/test_tilt.py", "tests/test_prepare_offload.py", "tests/test_build_bench_offload.py",
    "tests/test_align_graph_4dof.py", "tests/test_align_graph.py", "tests/test_plan_export.py", "tests/test_jobs.py",
    "tests/test_wall_checks.py", "tests/test_find_walls.py", "tests/test_merge_placed.py", "tests/test_settings.py",
    "tests/test_find_openings.py", "tests/test_adaptive_voxel.py", "tests/test_find_landmarks.py",
    "tests/test_find_elevations.py", "tests/test_offline_export.py", "tests/test_gpu_icp.py", "tests/test_check_docs.py",
    "tests/test_backup_retention.py", "tests/test_backup_cleaner.py", "tests/test_layout_error.py",
    "tests/test_height_settle.py", "tests/test_explain_moves.py", "tests/test_measure_noise.py",
    "tests/test_hand_moves.py",
    "tests/test_unix_line_endings.py", "tests/test_bug_library.py",
    "tests/test_reprocess.py", "tests/test_job_process.py", "tests/test_floors_ceilings.py",
    "tests/test_export_3d.py", "tests/test_named_landmarks.py",
    "tests/test_score_walls.py", "tests/test_export_job.py", "tests/test_no_undefined_names.py",
]
# Quick files that share something and so keep a lane of their own, never side by side with each other.
OWN_LANE = {"tests/test_gpu_icp.py": "the graphics card"}
# Long quick files split into this many processes (every n-th test each; tools/qa_run_part.py).
SPLIT = {"tests/test_pairwise.py": 4, "tests/test_adaptive_voxel.py": 2, "tests/test_find_landmarks.py": 2,
         "tests/test_prepare_offload.py": 2}

# MEDIUM: the browser suite, against a bench on a copy of a project.
MEDIUM = ["tests/test_browser_bench.py"]

# SLOW: the real scans, each in its own process (cross-process differences are invisible otherwise).
SLOW = ["tests/test_regression_real.py", "tests/test_acceptance.py", "tests/test_cancel_stages.py"]

# Tests that need the developer's own venue data or docs; they are not in the public repository and are skipped
# when absent.
LOCAL_ONLY = {"tests/test_check_docs.py", "tests/test_bug_library.py", "tests/test_cancel_stages.py",
              "tests/test_score_walls.py"}


def _present(files: list[str]) -> list[str]:
    return [f for f in files if f not in LOCAL_ONLY or os.path.exists(os.path.join(ROOT, f))]


QUICK, SLOW = _present(QUICK), _present(SLOW)


def unlisted() -> list[str]:
    """Test files in tests/ that no layer runs, and listed files that do not exist."""
    on_disk = {f"tests/{f}" for f in os.listdir(os.path.join(ROOT, "tests")) if f.startswith("test_") and f.endswith(".py")}
    listed = set(QUICK) | set(MEDIUM) | set(SLOW)
    return sorted(on_disk - listed) + [f"{f} (listed, but no such file)" for f in sorted(listed - on_disk)]


def _imports(path: str) -> set[str]:
    """Repo modules a Python file imports, as repo-relative paths (open_point_stitch/x.py, tools/x.py)."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if node.level and path.startswith("open_point_stitch/"):
                mod = "open_point_stitch." + mod
            names.add(mod)
            names.update(f"{mod}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    out = set()
    for n in names:
        for cand in (n.replace(".", "/") + ".py", "tools/" + n.replace(".", "/") + ".py"):
            if os.path.exists(os.path.join(ROOT, cand)):
                out.add(cand)
    return out


def touched_tests(changed: list[str]) -> list[str] | None:
    """The quick test files a change can reach, following imports through the repo. None means "cannot tell, run
    everything" (a change outside Python source, such as the Dockerfile or the requirements)."""
    reach: dict[str, set[str]] = {}

    def closure(path: str) -> set[str]:
        if path not in reach:
            reach[path] = {path}
            for dep in _imports(path):
                reach[path] |= closure(dep)
        return reach[path]

    out = set()
    for c in changed:
        c = c.replace("\\", "/")
        if c in QUICK:
            out.add(c)
        elif c.endswith(".py") and (c.startswith("open_point_stitch/") or c.startswith("tools/")):
            out.update(t for t in QUICK if c in closure(t))
        elif c.startswith(("notes/", "runs/", "golden/")) or c.endswith(".md"):
            continue                                        # records, not code; check_docs still runs
        else:
            return None
    return sorted(out | set(_present(["tests/test_check_docs.py", "tests/test_bug_library.py"])))
