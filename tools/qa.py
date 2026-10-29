"""One command that answers "is the tool working", at three speeds:
  --quick   pure checks, run side by side; no real scans, no browser
  --medium  the browser suite, split across two benches on two copies of the QA project
  --slow    the real scans: regression, acceptance and (when present) the venue canaries
Ends with one plain verdict: safe to update the bench, or not and why."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_layers  # noqa: E402
from bench_api import call, request, wait_for_http  # noqa: E402

QA_PROJECT_NAME = "qa"

HERE = "/app"  # matches WORKDIR in the Dockerfile; qa.py always runs inside the container
_print_lock = threading.Lock()
COVERAGE_RC = """[run]
parallel = True
data_file = {data}/cov
source = /app/open_point_stitch, /app/tools
concurrency = multiprocessing, thread
sigterm = True
"""


def _start_coverage(out: str) -> str:
    """Every Python process this run starts records what it runs (the QA image's zz_coverage.pth reads
    COVERAGE_PROCESS_START), and the browser suite records the page's own calls."""
    folder = os.path.join(out, "coverage")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(os.path.join(folder, "data"))
    rc = os.path.join(folder, "coveragerc")
    with open(rc, "w") as fh:
        fh.write(COVERAGE_RC.format(data=os.path.join(folder, "data")))
    os.environ["COVERAGE_PROCESS_START"] = rc
    os.environ["JS_COVERAGE_DIR"] = os.path.join(folder, "js")
    return folder


def _run(cmd: list[str], env: dict | None = None, stream: bool = True, prefix: str = "") -> tuple[int, str]:
    """Run a subprocess, streaming its combined output live (each line after prefix), or only collecting it."""
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    for line in proc.stdout:
        if stream:
            with _print_lock:
                print(prefix + line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def _last_meaningful_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return "(no output)"


class Check:
    def __init__(self, name: str, layer: str = ""):
        self.name = name
        self.layer = layer
        self.ok = False
        self.detail = ""
        self.seconds = 0.0


def count_declared_tests(path: str) -> int:
    """How many `test_` functions the file DEFINES, read as text rather than imported - importing it would run it."""
    with open(os.path.join(HERE, path) if not os.path.isabs(path) else path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.startswith("def test_"))


def count_reported(out: str) -> int:
    """PASS/FAIL lines in a test file's output; a bare "PASS" counts (test_regression_real.py prints one)."""
    return sum(1 for line in (ln.strip() for ln in out.splitlines())
               if line in ("PASS", "FAIL") or line.startswith("PASS ") or line.startswith("FAIL "))


def run_python_test_file(path: str, layer: str, stream: bool = True) -> Check:
    """Run one test file in its own process. Fails unless every test it declares reported a result."""
    c = Check(path, layer)
    if stream:
        print(f"--- {path} (running) ---", flush=True)
    t0 = time.time()
    parts = qa_layers.SPLIT.get(path, 1)
    if parts == 1:
        rc, out = _run([sys.executable, path], stream=stream)
    else:                                       # every n-th test per process; the count below adds the parts up
        with ThreadPoolExecutor(max_workers=parts) as pool:
            got = list(pool.map(lambda k: _run([sys.executable, "tools/qa_run_part.py", path, str(k), str(parts)],
                                               stream=False), range(parts)))
        rc, out = max(r for r, _ in got), "".join(o for _, o in got)
        if stream:
            print(out, end="", flush=True)
    c.seconds = time.time() - t0
    c.ok = rc == 0
    first_fail = next((ln.strip() for ln in out.splitlines() if ln.strip().startswith("FAIL")), "")
    c.detail = first_fail or _last_meaningful_line(out)
    declared, reported = count_declared_tests(path), count_reported(out)
    if declared == 0:
        c.ok = False
        c.detail = "no `def test_` found in this file, so QA cannot tell whether anything in it ran\n" + c.detail
    elif reported < declared:
        c.ok = False
        c.detail = (f"{reported} of {declared} declared test(s) actually ran - "
                    f"{declared - reported} never executed\n" + c.detail)
    if not c.ok:
        c.detail += "\n" + out[-2000:]  # enough to see WHY, not the whole log
    footer = f"--- {path} done in {c.seconds:.1f}s ({reported}/{declared} tests ran) ---"
    with _print_lock:
        if not stream:
            print(f"--- {path} ---\n{out.rstrip()}", flush=True)
        print(footer, flush=True)
    return c


def run_files_side_by_side(files: list[str], layer: str, lanes: int) -> list[Check]:
    """Shared-nothing files on `lanes` workers at once; the OWN_LANE files one after another on one more lane, so
    two of them never share what they share (the graphics card)."""
    shared = [f for f in files if f not in qa_layers.OWN_LANE]
    own = [f for f in files if f in qa_layers.OWN_LANE]
    print(f"{len(shared)} file(s) side by side on {lanes} lanes; {len(own)} on a lane of their own "
          f"({', '.join(f'{f}: {qa_layers.OWN_LANE[f]}' for f in own) or 'none'})", flush=True)
    longest_first = sorted(shared, key=lambda f: -_last_seconds().get(f, 0.0))
    with ThreadPoolExecutor(max_workers=max(1, lanes) + 1) as pool:
        own_f = pool.submit(lambda: [run_python_test_file(f, layer, stream=False) for f in own])
        shared_f = [pool.submit(run_python_test_file, f, layer, False) for f in longest_first]
        done = [f.result() for f in shared_f] + own_f.result()
    order = {f: i for i, f in enumerate(files)}
    return sorted(done, key=lambda c: order[c.name])


_TIMINGS_PATH = ""


def _last_seconds() -> dict:
    """Each file's time on the last run, so the longest start first. Only an ordering hint."""
    try:
        with open(_TIMINGS_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def ensure_qa_project(base_url: str, name: str = QA_PROJECT_NAME) -> None:
    """Create (or reopen), import into and open a QA project. Idempotent."""
    status, body = request(base_url, "POST", "/api/projects", {"name": name}, timeout=30)
    if status not in (200, 409):
        raise RuntimeError(f"could not create the QA project {name!r}: {status} {body}")
    status, body = request(base_url, "POST", f"/api/projects/{name}/import", {}, timeout=30)
    if status != 200:
        raise RuntimeError(f"could not import scans into the QA project {name!r}: {status} {body}")
    print(f"QA project {name!r}: imported {body.get('n', 0)} scan(s)")
    status, body = request(base_url, "POST", f"/api/projects/{name}/open", {}, timeout=30)
    if status != 200:
        raise RuntimeError(f"could not open the QA project {name!r}: {status} {body}")


def wait_for_all_scans_ready(base_url: str, timeout_s: float) -> tuple[float, int]:
    """Poll /api/scans until every row is ready (or failed, which still counts as not pending)."""
    t0 = time.time()
    n = 0
    rows: list = []
    while time.time() - t0 < timeout_s:
        data = call(base_url, "GET", "/api/scans", timeout=5)
        rows = data.get("scans", [])
        n = len(rows)
        if rows and all(r.get("ready") or r.get("error") for r in rows):
            return time.time() - t0, n
        time.sleep(1.0)
    raise TimeoutError(
        f"scan previews never finished building within {timeout_s:.0f}s "
        f"({sum(1 for r in rows if r.get('ready') or r.get('error'))}/{n} ready)"
    )


# The browser layer runs on generated scans (tests/synthetic_apartment.py), the same on every machine, so it needs no
# real scans. Bump the version when the scan set changes: a new folder means no project or cache from the old set.
BROWSER_SET = "browser-v6"


def _apartment():
    if os.path.join(HERE, "tests") not in sys.path:
        sys.path[:0] = [HERE, os.path.join(HERE, "tests")]
    import synthetic_apartment
    return synthetic_apartment


def place_where_they_belong(placement: dict) -> None:
    """Every generated scan's pose set to where it really is in the venue, as a placed project's would be. Solved
    from the scan's known frame through the bench's own levelling, so the flat's floor lands at height 0."""
    import numpy as np
    walks = _apartment().QA_WALKS   # first: it puts the app on the path
    from open_point_stitch.lib.poses import levelled_points, solve_yaw_translation

    g = np.linspace(0.0, 10.0, 6)
    venue = np.array([[x, y, z] for x in g for y in g for z in (0.0, 1.5, 3.0)])
    for s in placement["scans"]:
        frame = walks[s["name"]][1]                                  # venue -> the scan's raw frame
        raw = venue @ frame[:3, :3].T + frame[:3, 3]
        local = levelled_points(raw, s["level_rotation"], s.get("floor_z_local_m"))
        yaw, t = solve_yaw_translation(local, venue)
        for prefix in ("", "human_"):   # rounded as the page saves them, or its first save would move every scan
            s.update({prefix + "yaw_deg": round(yaw, 3), prefix + "x_m": round(float(t[0]), 4),
                      prefix + "y_m": round(float(t[1]), 4), prefix + "z_m": round(float(t[2]), 4)})


def browser_scans(args) -> str:
    """The generated QA scans under <out>, written once and kept: the same seed gives the same files, so later runs
    reuse them and their warm preview cache."""
    folder = os.path.join(args.out, BROWSER_SET, "scans")
    if not any(f.endswith(".pts") for f in (os.listdir(folder) if os.path.isdir(folder) else [])):
        apartment = _apartment()
        t0 = time.time()
        tmp = folder + ".part"
        shutil.rmtree(tmp, ignore_errors=True)
        apartment.write_qa_set(tmp)
        shutil.rmtree(folder, ignore_errors=True)
        os.replace(tmp, folder)
        print(f"generated the browser QA scans in {time.time() - t0:.1f}s: {', '.join(sorted(os.listdir(folder)))}",
              flush=True)
    return folder


def shard_projects_root(args, k: int) -> str:
    """Each shard its own projects folder, so nothing a bench keeps about "the last project" is shared between
    benches."""
    return os.path.join(args.out, BROWSER_SET, f"lane-{k + 1}")


def seed_shard_project(args, root: str) -> None:
    """A later shard's copy starts warm with the first shard's preview cache. The cache is keyed on each scan's own
    size and time, so a stale entry is rebuilt rather than served. Scans are imported by the server as usual."""
    src = os.path.join(shard_projects_root(args, 0), QA_PROJECT_NAME, ".bench_cache")
    dst = os.path.join(root, QA_PROJECT_NAME, ".bench_cache")
    if os.path.isdir(src) and not os.path.isdir(dst):
        shutil.copytree(src, dst)


def declared_browser_tests(only: str, skip: str) -> list[str]:
    with open(os.path.join(HERE, qa_layers.MEDIUM[0]), encoding="utf-8") as fh:
        names = [ln[4:ln.index("(")] for ln in fh if ln.startswith("def test_")]
    only_set = {n for n in only.split(",") if n}
    skip_set = {n for n in skip.split(",") if n}
    return [n for n in names if (not only_set or n in only_set) and n not in skip_set]


def start_bench(args, root: str, port: int, log_path: str):
    """A bench on the generated scans with root as its projects folder. Returns (process, its log file)."""
    bench_env = dict(os.environ, BENCH_PREVIEW_BUILD_WORKERS="4")
    # Server output goes to a file, never an unread pipe, which would fill and hang the server.
    log_fh = open(log_path, "w", encoding="utf-8")
    bench = subprocess.Popen([sys.executable, "-m", "open_point_stitch.server", "--scans", browser_scans(args),
                              "--out", args.out, "--projects", root, "--port", str(port)],
                             cwd=HERE, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=bench_env)
    return bench, log_fh


def save_start_layout(args, start: str) -> None:
    """The first run on a new scan set imports it once, saves the page's own layout of it and puts every scan where it
    belongs, so every test starts from a placed project, as a real one is. Kept at start, reused after."""
    from playwright.sync_api import sync_playwright

    root = os.path.join(args.out, BROWSER_SET, "layout")   # a fresh project: nothing an earlier run left can reach it
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    base_url = f"http://127.0.0.1:{args.port}"
    bench, log_fh = start_bench(args, root, args.port, os.path.join(args.out, "qa_bench_server_layout.log"))
    try:
        wait_for_http(f"{base_url}/api/version", 60.0, "bench server startup")
        ensure_qa_project(base_url)
        wait_for_all_scans_ready(base_url, args.scan_build_timeout)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"{base_url}/?project={QA_PROJECT_NAME}", wait_until="domcontentloaded")
            page.wait_for_function("() => Array.isArray(S) && S.length && S.every(s => s.ready)",
                                   timeout=120_000)
            page.evaluate("() => savePlacement()")
            browser.close()
    finally:
        bench.terminate()
        try:
            bench.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bench.kill()
        log_fh.close()
    with open(os.path.join(root, QA_PROJECT_NAME, "placement.json"), encoding="utf-8") as fh:
        placement = json.load(fh)
    if not placement.get("scans"):
        raise RuntimeError("saving the QA project's starting layout wrote no scans")
    place_where_they_belong(placement)
    with open(start, "w", encoding="utf-8") as fh:
        json.dump(placement, fh, indent=2)
    print(f"saved the browser QA starting layout: {len(placement['scans'])} scans", flush=True)


def run_browser_shard(args, k: int, n: int, port: int, results: dict) -> None:
    """Start a bench on this shard's copy of the QA project, wait for its previews, run this shard's tests on it."""
    root = shard_projects_root(args, k)
    project = QA_PROJECT_NAME
    base_url = f"http://127.0.0.1:{port}"
    log_path = os.path.join(args.out, f"qa_bench_server{'' if k == 0 else f'_{k + 1}'}.log")
    bench, log_fh = start_bench(args, root, port, log_path)
    timing = {}
    try:
        try:
            t0 = time.time()
            timing["server start"] = wait_for_http(f"{base_url}/api/version", 60.0, "bench server startup")
            if k > 0:
                seed_shard_project(args, root)
            ensure_qa_project(base_url, project)
            timing["scan previews"], n_scans = wait_for_all_scans_ready(base_url, args.scan_build_timeout)
            print(f"[shard {k + 1}] bench on {root}/{project} ready after {time.time() - t0:.1f}s ({n_scans} scans)", flush=True)
        except (TimeoutError, RuntimeError) as exc:
            c = Check(f"bench server readiness (shard {k + 1})", "medium")
            c.detail = str(exc)
            if bench.poll() is not None:
                log_fh.flush()
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    c.detail += "\nbench server output:\n" + fh.read()[-4000:]
            results[k] = ([c], "", 1, timing)
            return
        t0 = time.time()
        env = {**os.environ, "BENCH_URL": base_url, "ONLY_TESTS": args.only, "SKIP_TESTS": args.skip,
               "BENCH_PROJECT": project, "PROJECT_DIR": os.path.join(root, project)}
        if n > 1:
            env["BROWSER_SHARD"] = f"{k}/{n}"
        rc, out = _run([sys.executable, qa_layers.MEDIUM[0]], env=env, prefix=f"[shard {k + 1}] " if n > 1 else "")
        timing["browser tests"] = time.time() - t0
        checks = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("PASS ") or line.startswith("FAIL "):
                c = Check(line.split(" ", 1)[1].split(":")[0].replace(" (isolated)", ""), "medium")
                c.ok = line.startswith("PASS ")
                c.detail = line
                try:
                    c.seconds = float(line.rsplit(" ", 1)[1].rstrip("s"))
                except ValueError:
                    pass
                checks.append(c)
        results[k] = (checks, out, rc, timing)
    finally:
        bench.terminate()
        try:
            bench.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bench.kill()
        log_fh.close()


def run_browser_layer(args) -> tuple[list[Check], dict]:
    """Every selected browser test must report exactly once across the shards, and each shard must exit cleanly."""
    n = max(1, args.browser_shards)
    # Every bench starts from the same saved layout, so whatever one run leaves behind cannot reach the next.
    browser_scans(args)   # before the shards start, so two of them never write it at once
    start = os.path.join(args.out, BROWSER_SET, "start-placement.json")
    if not os.path.exists(start):
        save_start_layout(args, start)
    for k in range(n):
        pdir = os.path.join(shard_projects_root(args, k), QA_PROJECT_NAME)
        os.makedirs(pdir, exist_ok=True)
        shutil.copyfile(start, os.path.join(pdir, "placement.json"))
    results: dict = {}
    threads = [threading.Thread(target=run_browser_shard, args=(args, k, n, args.port + k, results))
               for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    checks: list[Check] = []
    timings = {}
    for k in range(n):
        ch, out, rc, timing = results[k]
        checks += ch
        timings[f"shard {k + 1}"] = timing
        if rc != 0 and all(c.ok for c in ch):
            c = Check(f"browser shard {k + 1} of {n}", "medium")
            c.detail = f"exited {rc} with no failing test line: it crashed or stopped early\n" + "\n".join(
                l for l in out.strip().splitlines()[-15:])
            checks.append(c)
    expected = declared_browser_tests(args.only, args.skip)
    seen = [c.name for c in checks if c.name.startswith("test_")]
    missing = sorted(set(expected) - set(seen))
    twice = sorted({s for s in seen if seen.count(s) > 1})
    c = Check("every browser test reported once", "medium")
    c.ok = not missing and not twice
    c.detail = (f"{len(set(seen))} of {len(expected)} browser tests reported" +
                (f"; never reported: {', '.join(missing)}" if missing else "") +
                (f"; reported twice: {', '.join(twice)}" if twice else ""))
    checks.append(c)
    return checks, timings


def run_canaries(args) -> list[Check]:
    """Both venues: one run each on a copy, alarms past a stated band. Skipped with a FAILING check when the venue
    projects are not mounted, so a slow pass can never claim canaries it did not run."""
    script = os.path.join(HERE, "tools", "qa_canary.py")
    if not os.path.exists(script):
        print("venue canaries: not part of this checkout (they need the developer's venue data); skipped", flush=True)
        return []
    c = Check("venue canaries (both venues)", "slow")
    if not os.path.isdir(args.venues):
        c.detail = f"no venue projects mounted at {args.venues}; the canaries did not run"
        return [c]
    t0 = time.time()
    rc, out = _run([sys.executable, script, "--venues", args.venues, "--golden", args.golden, "--work", args.canary_work])
    c.seconds = time.time() - t0
    checks = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("PASS ") or line.startswith("FAIL "):
            k = Check(line.split(" ", 1)[1].split(":")[0], "slow")
            k.ok = line.startswith("PASS ")
            k.detail = line
            checks.append(k)
    if rc != 0 and all(k.ok for k in checks):
        c.detail = f"exited {rc} without a failing line\n" + out[-2000:]
        checks.append(c)
    if not checks:
        c.detail = "printed no results\n" + out[-2000:]
        checks.append(c)
    checks[-1].seconds = c.seconds
    return checks


def verdict(checks: list[Check], ran: dict, partial_reasons: list[str], elapsed: float) -> tuple[str, int]:
    """The one page: safe to update the bench, or not and why, naming the failed layer and the first thing to read."""
    names = {"quick": "quick (pure checks)", "medium": "medium (browser)", "slow": "slow (real scans)"}
    lines = ["=" * 70, "VERDICT", "=" * 70]
    failed = [c for c in checks if not c.ok]
    for layer in ("quick", "medium", "slow"):
        mine = [c for c in checks if c.layer == layer]
        if not ran.get(layer):
            lines.append(f"  {names[layer]}: NOT RUN")
        else:
            bad = [c for c in mine if not c.ok]
            lines.append(f"  {names[layer]}: {len(mine) - len(bad)} of {len(mine)} passed, "
                         f"{sum(c.seconds for c in mine if c.seconds):.0f}s of work")
    lines.append(f"  wall clock {elapsed:.0f}s")
    lines.append("")
    if failed:
        first = failed[0]
        lines.append(f"NOT SAFE to update the bench: the {names.get(first.layer, first.layer)} layer failed "
                     f"({len(failed)} check(s)).")
        lines.append(f"First thing to look at: {first.name}: {first.detail.splitlines()[0] if first.detail else ''}")
        lines.append("QA FAILED")
        code = 1
    elif partial_reasons or not all(ran.get(k) for k in ("quick", "medium", "slow")):
        lines.append("NOT A FULL CHECK: everything that ran passed, but " +
                     "; ".join(partial_reasons or ["not every layer ran"]) +
                     ". Run `.\\stitch.ps1 qa` before updating the bench.")
        lines.append("QA PARTIAL")
        code = 0
    else:
        lines.append("SAFE to update the bench: all three layers ran and passed.")
        lines.append("QA PASSED")
        code = 0
    return "\n".join(lines), code


def main() -> int:
    global _TIMINGS_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", default="/scans")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--projects", default="/projects")
    ap.add_argument("--venues", default="/venues", help="the two venue projects, mounted READ ONLY (slow canaries)")
    ap.add_argument("--golden", default="/golden")
    ap.add_argument("--canary-work", default="/out/canary", help="the canaries' copies, kept warm between runs")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scan-build-timeout", type=float, default=900.0)
    ap.add_argument("--quick", action="store_true", help="pure checks only")
    ap.add_argument("--medium", action="store_true", help="the browser suite")
    ap.add_argument("--slow", action="store_true", help="the real scans: regression, acceptance, canaries")
    ap.add_argument("--lanes", type=int, default=12, help="quick-layer files run side by side")
    ap.add_argument("--browser-shards", type=int, default=2, help="benches the browser suite is split across")
    ap.add_argument("--touched", default="", help="comma-separated changed files: quick runs only what they reach (PARTIAL)")
    ap.add_argument("--no-canaries", action="store_true", help="slow layer without the venue canaries (PARTIAL)")
    ap.add_argument("--fast", action="store_true", help="skip the real-venue regression test (PARTIAL)")
    ap.add_argument("--browser-only", action="store_true", help="same as --medium (PARTIAL)")
    ap.add_argument("--only", default="", help="comma-separated browser test names to run (PARTIAL)")
    ap.add_argument("--skip", default="", help="comma-separated browser test names to skip (PARTIAL)")
    ap.add_argument("--coverage", action="store_true", help="also record which code no test reaches (slower)")
    args = ap.parse_args()
    _TIMINGS_PATH = os.path.join(args.out, "qa_timings.json")
    if args.browser_only:
        args.medium = True
    run = {"quick": args.quick, "medium": args.medium, "slow": args.slow}
    if not any(run.values()):
        run = {"quick": True, "medium": True, "slow": True}
    # A run that left anything out must never end in "safe".
    partial = []
    if args.only or args.skip:
        partial.append("browser tests were filtered with --only/--skip")
    if args.fast:
        partial.append("the regression test was skipped (--fast)")
    if args.touched:
        partial.append("the quick layer ran only the tests the changed files reach (--touched)")
    if args.no_canaries:
        partial.append("the venue canaries were skipped (--no-canaries)")

    cov_folder = _start_coverage(args.out) if args.coverage else None
    t_start = time.time()
    checks: list[Check] = []

    c = Check("every test file is in a QA layer", "quick" if run["quick"] else ("medium" if run["medium"] else "slow"))
    gaps = qa_layers.unlisted()
    c.ok = not gaps
    c.detail = "all listed" if not gaps else "never run by any layer: " + ", ".join(gaps)
    checks.append(c)

    if run["quick"]:
        print("=" * 70 + "\nQA QUICK: pure checks\n" + "=" * 70, flush=True)
        files = qa_layers.QUICK
        if args.touched:
            reach = qa_layers.touched_tests([f for f in args.touched.split(",") if f])
            if reach is None:
                print("a change outside Python source: running every quick test", flush=True)
            else:
                files = reach
                print(f"--touched: {len(files)} of {len(qa_layers.QUICK)} quick files reach the change", flush=True)
        t0 = time.time()
        checks += run_files_side_by_side(files, "quick", args.lanes)
        print(f"quick layer: {time.time() - t0:.1f}s wall clock", flush=True)

    # Slow and medium share nothing (own processes, own folders, the benches on their own ports): run together.
    def slow_layer() -> list[Check]:
        print("=" * 70 + "\nQA SLOW: real scans (each test its own process)\n" + "=" * 70, flush=True)
        slow = [f for f in qa_layers.SLOW if not (args.fast and f.endswith("test_regression_real.py"))]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(slow) + 1) as pool:
            parts = [pool.submit(run_python_test_file, f, "slow", False) for f in slow]
            canary = pool.submit(run_canaries, args) if not args.no_canaries else None
            out = [p.result() for p in parts] + (canary.result() if canary else [])
        print(f"slow layer: {time.time() - t0:.1f}s wall clock", flush=True)
        return out

    def medium_layer() -> list[Check]:
        print("=" * 70 + f"\nQA MEDIUM: browser tests on {args.browser_shards} bench(es), each on its own project "
              "copy\n" + "=" * 70, flush=True)
        t0 = time.time()
        out, timings = run_browser_layer(args)
        print(f"medium layer: {time.time() - t0:.1f}s wall clock; "
              f"{json.dumps(timings, default=lambda x: round(x, 1))}", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow_f = pool.submit(slow_layer) if run["slow"] else None
        medium_f = pool.submit(medium_layer) if run["medium"] else None
        checks += (slow_f.result() if slow_f else []) + (medium_f.result() if medium_f else [])

    elapsed = time.time() - t_start
    print("=" * 70 + "\nQA SUMMARY\n" + "=" * 70)
    for c in checks:
        first_line = c.detail.splitlines()[0] if c.detail else ""
        print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}  {first_line}")
    n_fail = sum(1 for c in checks if not c.ok)
    print("-" * 70)
    print(f"{len(checks) - n_fail}/{len(checks)} checks passed, {elapsed:.1f}s total")
    timed = sorted((c for c in checks if c.seconds), key=lambda c: -c.seconds)
    work = sum(c.seconds for c in timed) or 1.0
    print("\nslowest ten (seconds, share of all test work):")
    for c in timed[:10]:
        print(f"  {c.seconds:7.1f}s  {100 * c.seconds / work:4.1f}%  {c.name}")
    try:
        prev = _last_seconds()
        prev.update({c.name: round(c.seconds, 1) for c in checks if c.seconds and c.name.startswith("tests/")})
        with open(_TIMINGS_PATH, "w") as fh:
            json.dump(prev, fh, indent=1)
    except OSError:
        pass
    if n_fail:
        print("\nFAILURES, in full:")
        for c in checks:
            if not c.ok:
                print(f"--- {c.name} ---\n{c.detail}")
    if cov_folder and os.path.exists(os.path.join(HERE, "tools", "measurements", "test_coverage.py")):
        os.environ.pop("COVERAGE_PROCESS_START", None)
        print("\n" + "=" * 70 + "\nCOVERAGE: what no test reaches\n" + "=" * 70, flush=True)
        _run([sys.executable, os.path.join(HERE, "tools", "measurements", "test_coverage.py"),
              "--py-data", os.path.join(cov_folder, "data"), "--rc", os.path.join(cov_folder, "coveragerc"),
              "--js", os.path.join(cov_folder, "js"), "--html", os.path.join(HERE, "placement_bench.html"),
              "--out", os.path.join(cov_folder, "report.json")])
    text, code = verdict(checks, run, partial, elapsed)
    print("\n" + text)
    try:
        with open(os.path.join(args.out, "qa_verdict.txt"), "w") as fh:
            fh.write(text + "\n")
    except OSError:
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
