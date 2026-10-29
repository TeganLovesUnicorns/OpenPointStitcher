"""Browser tests for the placement bench, driven with real Chromium.

Run against a running bench (tools/qa.py starts one) at BENCH_URL (default http://127.0.0.1:8080):
    python tests/test_browser_bench.py
Tests take a Playwright page, supplied by the run loop at the bottom. Each test starts from a fresh page load and
compares a before/after it records itself, so tests do not depend on what an earlier one saved.
"""
from __future__ import annotations

import base64
import glob
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from bench_api import call  # noqa: E402

BENCH_URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8080")
# The QA project's folder: every save lands inside the open project. _goto opens it with ?project=. Checks that need
# the folder skip without it.
BENCH_PROJECT = os.environ.get("BENCH_PROJECT", "qa")
PROJECT_DIR = os.environ.get("PROJECT_DIR")
LOAD_TIMEOUT_MS = 60_000


def _goto(page):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    # Real 404 responses are tracked (the console text never has the URL). Expected: /favicon.ico and /api/placement
    # on a project with no save yet.
    page.expected_404 = False
    def _on_response(r):
        path = r.url.split("?", 1)[0].rstrip("/")
        if r.status == 404 and (path.endswith("/favicon.ico") or path.endswith("/api/placement")):
            page.expected_404 = True
    page.on("response", _on_response)
    t0 = time.time()
    page.goto(f"{BENCH_URL}/?project={urllib.parse.quote(BENCH_PROJECT)}",
              wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    t1 = time.time()
    page.wait_for_function(
        """() => {
            const rows = document.querySelectorAll('#scanlist .scan');
            if (!rows.length) return false;
            return [...rows].every(r => !r.textContent.includes('building preview'));
        }""",
        timeout=LOAD_TIMEOUT_MS,
    )
    PAGE_LOADS.append((t1 - t0, time.time() - t1))   # (page itself, every scan's preview) for the timing line
    # The one-time "review first?" prompt would hold back every run a test starts; only its own test turns it on.
    page.evaluate("() => { try { localStorage.setItem(nudgeOffKey(), '1'); } catch (e) {} }")
    return errors


PAGE_LOADS: list = []

# Opt-in (JS_COVERAGE_DIR set): Chrome's call count for every function of the page, one file per test. Off in normal
# QA.
JS_COVERAGE_DIR = os.environ.get("JS_COVERAGE_DIR")


def _js_coverage_start(page):
    if not JS_COVERAGE_DIR:
        return
    session = page.context.new_cdp_session(page)
    session.send("Profiler.enable")
    session.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": False})
    page._js_cov = (session, {})
    for method in ("goto", "reload"):                   # a navigation drops the counts so far: take them first
        def counted(*a, _original=getattr(page, method), **k):
            _js_coverage_take(page)
            return _original(*a, **k)
        setattr(page, method, counted)


def _js_coverage_take(page):
    """Adds the counts since the last take (a take resets them)."""
    cov = getattr(page, "_js_cov", None)
    if not cov:
        return
    try:
        scripts = cov[0].send("Profiler.takePreciseCoverage")["result"]
    except Exception:  # noqa: BLE001 - a closed page has nothing left to count
        return
    for script in scripts:
        if not script["url"].startswith(BENCH_URL):
            continue
        for f in script["functions"]:
            r = f["ranges"][0]
            key = f"{f['functionName']}@{r['startOffset']}-{r['endOffset']}"
            cov[1][key] = cov[1].get(key, 0) + r["count"]


def _js_coverage_save(page, name):
    cov = getattr(page, "_js_cov", None)
    if not cov:
        return
    _js_coverage_take(page)
    os.makedirs(JS_COVERAGE_DIR, exist_ok=True)
    with open(os.path.join(JS_COVERAGE_DIR, f"{name}.json"), "w") as fh:
        json.dump(cov[1], fh)


def _saved_settings(page, timeout_s: float = 20.0, **want) -> dict:
    """The project's settings on the server, once every want value has arrived (compared as text). Saves are
    debounced, so poll rather than wait a fixed time."""
    deadline = time.time() + timeout_s
    while True:
        page.wait_for_timeout(500)
        got = call(f"{BENCH_URL}/api/settings", "GET", timeout=10)
        if all(str(got.get(k)) == str(v) for k, v in want.items()) or time.time() > deadline:
            return got


def _assert_finite(label, **coords):
    """Fail loudly, at the point a coordinate is computed, if it is not finite: a NaN sent to Playwright's mouse API
    crashes the driver with no useful error."""
    bad = {k: v for k, v in coords.items() if v is None or not math.isfinite(v)}
    assert not bad, f"{label}: non-finite coordinate(s) {bad} in {coords}"


def _canvas_rect(page):
    return page.evaluate("() => { const r = cv.getBoundingClientRect(); return {x:r.x, y:r.y}; }")


def _to_viewport(rect, xy):
    return rect["x"] + xy[0], rect["y"] + xy[1]


def _scan_point_screen_pos(page, scan_i, k_frac=0.5):
    """Screen (canvas-local) position of an actual rendered point of scan scan_i, using the page's own place()/proj()
    maths (a bbox centroid can land in empty space). Also returns the point's height h. Reads points with the page's
    xyAt(s, k)/nPoints(s), since s.xy is a flat array."""
    pos = page.evaluate(
        """(args) => {
            const s = S.find(x => x.i === args.i);
            const n = nPoints(s);
            const k = Math.max(0, Math.min(n - 1, Math.floor(n * args.frac)));
            const g = place(s, xyAt(s, k));
            const p = proj(g[0], g[1], s.h[k] + s.dz);
            return {x: p[0], y: p[1], h: s.h[k]};
        }""",
        {"i": scan_i, "frac": k_frac},
    )
    _assert_finite(f"_scan_point_screen_pos(scan_i={scan_i})", x=pos["x"], y=pos["y"], h=pos["h"])
    return pos


def _hit_testable_point(page, scan_i):
    """A screen position the page's own hit() confirms resolves to scan_i. hit() samples every 3rd point and points
    are sorted by agreement, not position, so candidates are checked through hit() itself. Also skips anywhere
    gizmoHit() would catch first (a spin, not a move) and anywhere a real click would not reach #cv (checked with
    elementFromPoint)."""
    pt = page.evaluate(
        """(i) => {
            const s = S.find(x => x.i === i);
            const rect = cv.getBoundingClientRect();
            const n = nPoints(s);
            const step = Math.max(1, Math.floor(n / 60));
            for (let k = 0; k < n; k += step) {
                const g = place(s, xyAt(s, k));
                const raw = proj(g[0], g[1], s.h[k] + s.dz);
                // Round to whole CSS pixels BEFORE checking anything: a real
                // mouse event only ever lands on an integer pixel, so this is
                // what a real click will actually test, not the sub-pixel
                // float proj() happens to return. Checking the float and
                // clicking the rounded pixel let a DIFFERENT scan's point -
                // one fractionally closer once rounded - win hit()'s
                // closest-point search instead, found live (mousedown
                // started a real 'move' drag, just on the wrong scan).
                const p = [Math.round(raw[0]), Math.round(raw[1])];
                if (p[0] < 4 || p[0] > cv.width - 4 || p[1] < 4 || p[1] > cv.height - 4) continue;
                if (hit(p[0], p[1]) !== i) continue;
                if (i === sel && gizmoHit(p[0], p[1])) continue;
                if (document.elementFromPoint(rect.left + p[0], rect.top + p[1]) !== cv) continue;
                return {x: p[0], y: p[1]};
            }
            return null;
        }""",
        scan_i,
    )
    # None (no hit-testable point found) is a legitimate outcome callers
    # already handle - only a returned-but-broken point needs to fail loud.
    if pt is not None:
        _assert_finite(f"_hit_testable_point(scan_i={scan_i})", x=pt["x"], y=pt["y"])
    return pt


def _click_empty_props_area(page):
    """Click inside the right sidebar on something that is neither an input nor the resize handle (which spans the
    panel's left edge): the "Selected" heading."""
    page.locator('h2:has-text("Selected")').click()


def _pose(page, scan_i):
    return page.evaluate(
        "(i) => { const s = S.find(x => x.i === i); "
        "return {x: s.x, y: s.y, yaw: s.yaw, dz: s.dz, pitch: s.pitch, roll: s.roll}; }",
        scan_i,
    )


def _world_centre(page, scan_i):
    """The scan's 3D centre in world space, via the page's own worldCentre3()/localCentre3(), the functions setPose()'s
    pivot compensation uses."""
    return page.evaluate(
        "(i) => { const s = S.find(x => x.i === i); "
        "return worldCentre3(localCentre3(s), s.yaw, s.pitch, s.roll, s.x, s.y, s.dz); }",
        scan_i,
    )


def _keyboard_step_m(page):
    return page.evaluate("() => granularityM()")


def _rot_step_deg(page):
    return page.evaluate("() => rotStepDeg()")


def _select_scan_row(page, pos=0):
    """Click the sidebar row at display position `pos` (0 = top of the list).
    Returns the scan's fixed `s.i`, since that (not `pos`) is what identifies
    it everywhere else in the page's own state."""
    row = page.locator("#scanlist .scan").nth(pos)
    row.click()
    return int(row.get_attribute("data-i"))


def _pick_run(page, kind):
    """Choose the kind of run the next _press_run starts (the Run window's list)."""
    page.evaluate("(k) => { $('runKind').value = k; }", kind)


def _press_run(page):
    """Open the top bar's Run window and click the chosen kind, the way a person starts a run."""
    kind = page.evaluate("() => $('runKind').value")
    page.locator("#bRunMenu").click()
    page.locator(f'[data-run-kind="{kind}"]').click()


def _show_bottom_tab(page, tab):
    """Bring one of the bottom panel's tabs (links, anchors, exclusions, removals) to the front."""
    page.locator(f"#bottomTabs-tab-{tab}").click()


# --------------------------------------------------------------------------

def test_bookmarked_project_url_opens_that_project(page):
    """A ?project=<name> bookmark opens the bench with that project, not a raw JSON error: the server routes on the
    path without its query string. Uses whichever project tools/qa.py opened."""
    active = call(f"{BENCH_URL}/api/projects", "GET", timeout=10).get("active")
    assert active, "no project is open server-side - nothing real to bookmark against"

    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"{BENCH_URL}/?project={urllib.parse.quote(active)}",
             wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        """() => {
            const rows = document.querySelectorAll('#scanlist .scan');
            if (!rows.length) return false;
            return [...rows].every(r => !r.textContent.includes('building preview'));
        }""",
        timeout=LOAD_TIMEOUT_MS,
    )
    gate_visible = page.evaluate("() => getComputedStyle($('projectGate')).display !== 'none'")
    assert not gate_visible, "the picker is still showing - the bookmark URL did not open the bench"
    title = page.locator("#topbarTitle").inner_text()
    assert active in title, f"topbar shows {title!r}, expected it to name the opened project {active!r}"
    real_errors = [e for e in errors if not e.startswith("Failed to load resource")]
    assert not real_errors, f"console error(s) loading a bookmarked project URL: {real_errors}"
    print(f"    /?project={active!r} opened the bench directly, no picker, no console errors")


def test_bookmarked_url_for_missing_project_shows_readable_message(page):
    """A ?project= naming a project that does not exist lands on the start screen with a readable message, and never
    falls back to whatever else is open."""
    missing_name = "Definitely Not A Real Project " + str(int(time.time()))
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"{BENCH_URL}/?project={urllib.parse.quote(missing_name)}",
             wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        "() => getComputedStyle($('projectGate')).display !== 'none'",
        timeout=LOAD_TIMEOUT_MS,
    )
    body_text = page.evaluate("() => document.body.innerText")
    assert '{"error"' not in body_text, (
        f"the page body looks like raw JSON, not the picker: {body_text[:200]!r}"
    )
    error_text = page.locator("#gateError").inner_text()
    assert missing_name in error_text, (
        f"#gateError reads {error_text!r}, expected it to name the missing project {missing_name!r}"
    )
    gate_visible = page.evaluate("() => getComputedStyle($('projectGate')).display !== 'none'")
    assert gate_visible, "the picker should be showing after a failed bookmark, and is not"
    real_errors = [e for e in errors if not e.startswith("Failed to load resource")]
    assert not real_errors, f"console error(s) on a missing-project bookmark URL: {real_errors}"
    print(f"    /?project=<missing> landed on the picker with a readable message, not raw JSON")


def test_start_screen_no_silent_resume_and_actions_present(page):
    """A bare load with no ?project= shows the start screen with real, clickable actions, never silently resuming the
    server's open project."""
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(BENCH_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        "() => getComputedStyle($('projectGate')).display !== 'none'",
        timeout=LOAD_TIMEOUT_MS,
    )
    app_open = page.evaluate("() => $('app').classList.contains('open')")
    assert not app_open, "a bare load opened a project directly - silent resume is back"
    assert "?project=" not in page.url, f"bare load somehow carries a project in the URL: {page.url}"

    # Continue and Open existing always show here: tools/qa.py opened the QA project once before this suite, so it has
    # a last_opened.
    continue_btn = page.locator("#startContinue")
    assert continue_btn.is_visible(), "Continue-with-X did not show even though a project was just opened"
    assert BENCH_PROJECT in continue_btn.inner_text(), (
        f"Continue button reads {continue_btn.inner_text()!r}, expected it to name {BENCH_PROJECT!r}"
    )
    assert page.locator("#startOpenExisting").is_visible()
    assert page.locator("#startNewProject").is_visible()
    assert page.locator("#startSplash").count() == 1, "the splash <img> is missing from the start screen"

    real_errors = [e for e in errors if not e.startswith("Failed to load resource")]
    assert not real_errors, f"console error(s) on the start screen: {real_errors}"
    print(f"    bare load: start screen shown, no silent resume, Continue names {BENCH_PROJECT!r}")


def test_start_new_project_overlay_opens_and_cancels(page):
    """Start a new project opens the overlay; Cancel closes it without creating anything."""
    page.goto(BENCH_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        "() => getComputedStyle($('projectGate')).display !== 'none'", timeout=LOAD_TIMEOUT_MS)
    page.locator("#startNewProject").click()
    overlay = page.locator("#newProjectOverlay")
    page.wait_for_function("() => getComputedStyle($('newProjectOverlay')).display !== 'none'")
    assert overlay.locator("#gateNewName").count() == 1
    assert overlay.locator("#gateNewCreate").count() == 1
    page.locator("#newProjectCancel").click()
    page.wait_for_function("() => getComputedStyle($('newProjectOverlay')).display === 'none'")
    print("    Start a new project: overlay opened with a real name box, Cancel closed it")


def _small_room_pts(path: str, seed: int, n: int = 20000) -> None:
    """A 4 x 3 x 2.5 m box room as a .pts file, for import tests."""
    import random
    rnd = random.Random(seed)
    lines = [str(n)]
    for _ in range(n):
        face = rnd.randrange(6)
        x, y, z = rnd.uniform(0, 4), rnd.uniform(0, 3), rnd.uniform(0, 2.5)
        x, y, z = [(x, y, 0.0), (x, y, 2.5), (0.0, y, z), (4.0, y, z), (x, 0.0, z), (x, 3.0, z)][face]
        lines.append(f"{x:.4f} {y:.4f} {z:.4f} -1 {120 + face * 20} 110 100")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def test_scans_from_anywhere_import_into_a_new_project_and_an_open_one(page):
    """Scans from outside any scans folder go in by the file picker and by drag and drop; a duplicate is refused and can
    be kept under a new name; a broken file is named; File > Import scans adds more to an open project. Makes its own
    project next to the QA one and removes it."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set - this check needs the projects folder to clean up after itself")
        return
    name = "qa import test"
    pdir = os.path.join(os.path.dirname(PROJECT_DIR), name)
    src = tempfile.mkdtemp(prefix="elsewhere-")
    try:
        for i, f in enumerate(("Lobby.pts", "Stage.pts", "Bar.pts")):
            _small_room_pts(os.path.join(src, f), seed=i)
        with open(os.path.join(src, "Broken.pts"), "w") as fh:
            fh.write("not a scan\n")
        before = {f: os.path.getmtime(os.path.join(src, f)) for f in os.listdir(src)}

        page.goto(BENCH_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
        page.wait_for_function("() => getComputedStyle($('projectGate')).display !== 'none'", timeout=LOAD_TIMEOUT_MS)
        page.locator("#startNewProject").click()
        page.locator("#gateNewName").fill(name)
        page.locator("#gateNewCreate").click()
        page.wait_for_function("() => getComputedStyle($('gateImportPane')).display !== 'none'")
        page.locator("#gateImportList input:checked").evaluate_all("els => els.forEach(e => e.checked = false)")

        with page.expect_file_chooser() as fc:
            page.locator("#importPick").click()
        fc.value.set_files([os.path.join(src, "Lobby.pts"), os.path.join(src, "Broken.pts")])
        with open(os.path.join(src, "Stage.pts")) as fh:
            dropped = fh.read()
        page.evaluate("""(txt) => { const dt = new DataTransfer(); dt.items.add(new File([txt], 'Stage.pts'));
            $('importDrop').dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true})); }""", dropped)
        with page.expect_file_chooser() as fc:
            page.locator("#importPick").click()
        fc.value.set_files([os.path.join(src, "Lobby.pts")])          # a second Lobby.pts: a duplicate

        row = lambda f: page.locator(f'#importUploads .importRow[data-file="{f}"]')
        page.wait_for_function("() => [...document.querySelectorAll('#importUploads .importRow .st')]"
                               ".every(s => !/waiting|sending|reading/.test(s.textContent))", timeout=60000)
        assert "20,000 points" in row("Lobby.pts").first.inner_text(), row("Lobby.pts").first.inner_text()
        assert "20,000 points" in row("Stage.pts").inner_text(), "the dropped file did not import"
        assert "could not be read" in row("Broken.pts").inner_text(), row("Broken.pts").inner_text()
        dup = row("Lobby.pts").nth(1)
        assert "already in this project" in dup.inner_text(), dup.inner_text()
        dup.locator("button", has_text="Keep both").click()
        page.wait_for_function(f"""() => /saved as Lobby \\(2\\)\\.pts/.test(
            document.querySelectorAll('#importUploads .importRow[data-file="Lobby.pts"]')[1].textContent)""",
                               timeout=30000)
        raw = sorted(os.listdir(os.path.join(pdir, "scans-raw")))
        assert raw == ["Lobby (2).pts", "Lobby.pts", "Stage.pts"], raw

        page.locator("#gateImportGo").click()
        page.wait_for_url(lambda u: "project=" in u, timeout=LOAD_TIMEOUT_MS)
        page.wait_for_function("() => document.querySelectorAll('#scanlist .scan').length === 3", timeout=LOAD_TIMEOUT_MS)

        page.locator('.menubtn[data-menu="menuFile"]').click()
        page.locator("#mImportScans").click()
        page.wait_for_function("() => getComputedStyle($('gateImportPane')).display !== 'none'")
        with page.expect_file_chooser() as fc:
            page.locator("#importPick").click()
        fc.value.set_files([os.path.join(src, "Bar.pts")])
        page.wait_for_function("() => /points/.test(document.querySelector('#importUploads .importRow .st').textContent)",
                               timeout=60000)
        with page.expect_navigation(timeout=LOAD_TIMEOUT_MS):
            page.locator("#gateImportGo").click()
        page.wait_for_function("() => document.querySelectorAll('#scanlist .scan').length === 4", timeout=LOAD_TIMEOUT_MS)

        after = {f: os.path.getmtime(os.path.join(src, f)) for f in os.listdir(src)}
        assert after == before, "an original file was changed, moved or added to"
        print(f"    new project from files elsewhere: {raw}, broken file named, then File > Import added Bar: 4 scans")
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(pdir, ignore_errors=True)


def test_open_existing_modal_opens_a_project(page):
    """Open a project from the Open existing modal, with a real click on a real row."""
    page.goto(BENCH_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        "() => getComputedStyle($('projectGate')).display !== 'none'", timeout=LOAD_TIMEOUT_MS)
    page.locator("#startOpenExisting").click()
    page.wait_for_function("() => getComputedStyle($('openExistingModal')).display !== 'none'")
    rows = page.locator("#openExistingList .modalRow")
    assert rows.count() >= 1, "the Open-existing modal has no project rows"
    target = rows.filter(has_text=BENCH_PROJECT)
    assert target.count() == 1, f"no row for {BENCH_PROJECT!r} in the Open-existing modal"
    opens = []
    page.on("request", lambda r: opens.append(r.url) if r.method == "POST" and r.url.endswith("/open") else None)
    target.first.click()
    # Impatient repeat clicks are ignored, and the row says it is opening.
    shown = page.evaluate("""(name) => {
        const row = [...document.querySelectorAll('#openExistingList [data-open]')].find(el => el.dataset.open === name);
        if (!row || !row.isConnected) return 'navigated';
        row.click(); row.click();
        return row.querySelector('.pmeta').textContent;
    }""", BENCH_PROJECT)
    assert shown in ("Opening...", "navigated"), f"the clicked row reads {shown!r}, not Opening..."
    page.wait_for_function(          # the click navigates: the new page's script may not have defined $ yet
        "() => typeof $ === 'function' && $('app') && $('app').classList.contains('open')", timeout=LOAD_TIMEOUT_MS)
    assert len(opens) == 2, f"expected one open from the picker plus the page's own, got {len(opens)}: {opens}"
    assert f"project={urllib.parse.quote(BENCH_PROJECT)}" in page.url or BENCH_PROJECT in page.url, (
        f"opening {BENCH_PROJECT!r} from the modal did not put it in the URL: {page.url}"
    )
    title = page.locator("#topbarTitle").inner_text()
    assert BENCH_PROJECT in title, f"topbar reads {title!r} after opening from the modal"
    print(f"    Open-existing modal: listed {rows.count()} project(s), opened {BENCH_PROJECT!r} from a row click")


def test_switch_project_returns_to_start_screen(page):
    """Switch project reaches the start screen and does not resume the project it just left."""
    _goto(page)
    assert page.evaluate("() => $('app').classList.contains('open')"), "setup: not inside a project"
    page.locator('.menubtn[data-menu="menuFile"]').click()
    page.locator("#mSwitchProject").click()
    page.wait_for_function(
        "() => getComputedStyle($('projectGate')).display !== 'none'", timeout=LOAD_TIMEOUT_MS)
    app_open = page.evaluate("() => $('app').classList.contains('open')")
    assert not app_open, "Switch project did not actually leave the project - #app is still open"
    assert "?project=" not in page.url, f"Switch project left a project in the URL: {page.url}"
    print("    Switch project: left the project and landed back on the start screen")


def test_escape_leaves_tool_and_shows_active_tool_label(page):
    """Escape leaves any tool (link and exclude included) and returns to Navigate; the active tool shows in its own
    always-visible label."""
    _goto(page)
    label = page.locator("#activeToolLabel")
    assert "Navigate" in label.inner_text(), f"label does not read Navigate by default: {label.inner_text()!r}"

    page.locator('[data-tool="exclude"]').click()
    assert page.evaluate("() => tool") == "exclude", "clicking Exclude region did not set the tool"
    assert "Exclude" in label.inner_text(), f"label did not update for the exclude tool: {label.inner_text()!r}"

    page.keyboard.press("Escape")
    page.wait_for_function("() => tool === 'navigate'", timeout=5000)
    assert "Navigate" in label.inner_text(), f"label did not return to Navigate after Escape: {label.inner_text()!r}"
    on_navigate = page.locator('[data-tool="navigate"]').evaluate("el => el.classList.contains('on')")
    assert on_navigate, "Navigate button is not highlighted after Escape"
    print("    active-tool label updates per tool; Escape from Exclude region returns to Navigate")


def test_sidebar_row_selects_on_the_name_text(page):
    """Clicking a scan's name in the sidebar selects it: the whole row is the click target."""
    _goto(page)
    other_i = _select_scan_row(page, 0)
    name_el = page.locator("#scanlist .scan").nth(1).locator("[data-rename]")
    target_i = int(page.locator("#scanlist .scan").nth(1).get_attribute("data-i"))
    assert target_i != other_i, "setup: need two distinct scans"
    name_el.click()
    sel_after = page.evaluate("() => sel")
    assert sel_after == target_i, (
        f"clicking the scan's name text did not select it (sel={sel_after}, expected {target_i})"
    )
    print(f"    clicking scan {target_i}'s name text in the sidebar selected it")


def test_seed_summary_shown_up_front_and_persists_on_reload(page):
    """The last run's seed summary shows up front and survives a reload. A real align takes minutes, so this writes
    a last_run into the placement file (the shape _stamp_last_run writes), reloads, and checks #seedSummary."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set - this check needs direct file access to the project")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path) as fh:
        data = json.load(fh)
    original_last_run = data.get("last_run")
    try:
        data["last_run"] = {
            "kind": "align_auto", "at": "2026-09-05T12:00:00Z",
            "n_pairs_attempted": 91, "n_accepted": 24, "connected": True, "n_unregistered": 0,
            "elapsed_s": 1380.0, "n_pairs_before_filter": 91, "n_placement_seeds": 91,
            "adjacency_padding_m": 3.0,
        }
        with open(placement_path, "w") as fh:
            json.dump(data, fh, indent=2)

        _goto(page)
        summary = page.locator("#seedSummary")
        page.wait_for_function("() => getComputedStyle($('seedSummary')).display !== 'none'", timeout=5000)
        text = summary.inner_text()
        assert "seeded from your placement" in text, f"#seedSummary does not name the seed source: {text!r}"
        assert "91" in text, f"#seedSummary does not carry the pair counts: {text!r}"
        assert "0 skipped" in text, f"#seedSummary does not say 0 skipped: {text!r}"
        assert "not a sign your placement did not save" in text, (
            f"#seedSummary does not reassure the misreading it exists to prevent: {text!r}"
        )
        print(f"    #seedSummary after reload: {text!r}")
    finally:
        with open(placement_path) as fh:
            data = json.load(fh)
        if original_last_run is not None:
            data["last_run"] = original_last_run
        else:
            data.pop("last_run", None)
        with open(placement_path, "w") as fh:
            json.dump(data, fh, indent=2)


def test_bent_scan_flags_show_in_the_links_panel_and_log_after_reload(page):
    """A run's bent-scan flags reach the links panel and the run log, by the scan's shown name. Writes last_run into the
    QA copy and restores it."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set - this check needs direct file access to the project")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path) as fh:
        data = json.load(fh)
    original_last_run = data.get("last_run")
    scan, other = data["scans"][0], data["scans"][1]
    original_other_alias = other.get("alias")
    shown = scan.get("alias") or scan["name"]
    shown_other = "back room (test name)"             # the compared scan must show by its shown name too
    text = (f"{scan['name']} reads 15 cm off on its east wall (compared with {other['name']}) while its other walls "
            f"agree within 3 cm. It may have drifted during the scan.")
    try:
        other["alias"] = shown_other
        data["last_run"] = {"kind": "align_auto", "at": "2026-10-04T12:00:00Z", "n_pairs_attempted": 3,
                            "bent_scans": [{"scan": scan["name"], "kind": "bent", "text": text}]}
        with open(placement_path, "w") as fh:
            json.dump(data, fh, indent=2)
        _goto(page)
        panel = page.locator("#bentScans")
        panel.wait_for(state="visible", timeout=10000)
        shown_text = panel.inner_text()
        assert f"{shown} reads 15 cm off on its east wall (compared with {shown_other})" in shown_text, shown_text
        assert "nothing was moved" in shown_text, shown_text
        log_text = page.locator("#out").input_value()
        assert "may have drifted during the scan" in log_text, log_text[-800:]
        assert "[object" not in log_text, "the last-run line printed a raw object"
        print(f"    links panel: {shown_text!r}")
    finally:
        with open(placement_path) as fh:
            data = json.load(fh)
        if original_last_run is not None:
            data["last_run"] = original_last_run
        else:
            data.pop("last_run", None)
        if original_other_alias:
            data["scans"][1]["alias"] = original_other_alias
        else:
            data["scans"][1].pop("alias", None)
        with open(placement_path, "w") as fh:
            json.dump(data, fh, indent=2)
    if not (original_last_run or {}).get("bent_scans"):
        _goto(page)
        assert page.locator("#bentScans").count() == 0, "flags from a restored last run are still showing"


def test_links_table_divider_drags_and_keys_and_is_remembered(page):
    """The divider above the links table resizes it by mouse and keyboard, and the height comes back after a reload."""
    _goto(page)
    page.evaluate("() => { try { localStorage.removeItem(bottomKey()); } catch (e) {} }")
    bottom = page.locator("#bottom")
    start = bottom.bounding_box()["height"]
    box = page.locator("#bottomResize").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y - 60, steps=6)
    page.mouse.move(x, y - 120, steps=6)
    page.mouse.up()
    dragged = bottom.bounding_box()["height"]
    assert abs(dragged - (start + 120)) <= 4, f"drag up 120 px: table {start} -> {dragged}"

    page.locator("#bottomResize").focus()
    page.keyboard.press("ArrowDown")
    keyed = bottom.bounding_box()["height"]
    assert abs(keyed - (dragged - 20)) <= 4, f"arrow down: table {dragged} -> {keyed}"
    assert page.locator("#bottomResize").get_attribute("aria-valuenow") == str(round(keyed)), "aria value not kept"

    _goto(page)
    again = page.locator("#bottom").bounding_box()["height"]
    assert abs(again - keyed) <= 4, f"after reload the table is {again} px, was {keyed}"
    page.evaluate("() => { try { localStorage.removeItem(bottomKey()); } catch (e) {} }")
    print(f"    links table {start:.0f} px, dragged to {dragged:.0f}, arrow down {keyed:.0f}, after reload {again:.0f}")


def test_run_cancel_and_save_are_on_the_top_bar_and_the_lists_are_bottom_tabs(page):
    """Run, Cancel and Save on the top bar, and the bottom tabs."""
    _goto(page)
    top = page.locator("#topbar")
    for sel in ("#bRunMenu", "#bCancelJob", "#bSave"):
        assert top.locator(sel).count() == 1 and page.locator(sel).is_visible(), f"{sel} is not on the top bar"
    right = page.locator("#right")
    for gone in ("#bRunMenu", "#bSave", "#exclusionsList", "#removalsList", "#bClearLinks"):
        assert right.locator(gone).count() == 0, f"{gone} is still in the right panel"
    assert right.locator("#out").count() == 1, "the run messages left the right panel"

    page.locator("#bRunMenu").click()
    kinds = page.locator("[data-run-kind]")
    assert kinds.count() == 10, f"the Run window lists {kinds.count()} kinds of run, expected 10"
    for k in range(kinds.count()):
        assert len(kinds.nth(k).inner_text().split("\n")) >= 2, "a kind of run has no description under its name"
    page.keyboard.press("Escape")
    assert not page.locator('[data-menu-modal="runModal"]').is_visible(), "Escape did not close the Run window"

    labels = page.locator("#bottomTabs [role='tab']").all_inner_texts()
    assert [l.split(" (")[0] for l in labels] == ["Links from runs", "Anchors", "Exclusions", "Removed in review",
                                                  "Landmarks", "Labels", "Measurements"], labels
    assert not page.locator("#bClearLinks").is_visible(), "Clear all anchors shows outside its tab"
    _show_bottom_tab(page, "anchors")
    assert page.locator("#bClearLinks").is_visible() and page.locator("#bClearLinks").inner_text() == "Clear all anchors"
    _show_bottom_tab(page, "links")
    print(f"    top bar: Run..., Cancel, Save; Run window: 10 kinds with descriptions; tabs: {labels}")


def test_each_panel_folds_from_a_tab_on_the_view_that_covers_nothing(page):
    """Left, right and bottom panels each fold from a tab on the view's side of their divider."""
    _goto(page)
    overlap = """([a, b]) => {
        const r = document.querySelector(a).getBoundingClientRect(), s = document.querySelector(b).getBoundingClientRect();
        return r.left < s.right && s.left < r.right && r.top < s.bottom && s.top < r.bottom;
    }"""
    stage = page.locator("#stage").bounding_box()
    for tab, panel, dim, slider in (("#leftCollapseTab", "#left", "width", "#rc-clip-y-root > div"),
                                    ("#rightCollapseTab", "#right", "width", "#rc-clip-z-root > div"),
                                    ("#bottomCollapseTab", "#bottom", "height", "#rc-clip-x-root > div")):
        box = page.locator(tab).bounding_box()
        assert stage["x"] <= box["x"] and box["x"] + box["width"] <= stage["x"] + stage["width"] + 0.5, f"{tab} is not on the view"
        assert not page.evaluate(overlap, [tab, slider]), f"{tab} covers its clip slider"
        before = page.locator(panel).bounding_box()[dim]
        page.locator(tab).click()
        folded = page.locator(panel).bounding_box()[dim]
        assert folded <= 1.5, f"{panel} is {folded} px after its tab was clicked (was {before})"
        assert page.locator(tab).is_visible(), f"{tab} vanished with its panel"
        page.locator(tab).click()
        back = page.locator(panel).bounding_box()[dim]
        assert abs(back - before) <= 2, f"{panel} came back at {back} px, was {before}"
        print(f"    {panel}: {before:.0f} px, folded to {folded:.0f}, back to {back:.0f}")
    assert not page.evaluate(overlap, ["#leftCollapseTab", "#scanlist"]), "the left tab covers the scan list"


def test_badges_never_cover_the_toolbar_at_any_width(page):
    """The state badges sit in their own row below the tool buttons, so no width or text length makes them cover a
    button. Checked at a narrow and a wide viewport with #seedSummary showing."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set - this check needs direct file access to the project")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path) as fh:
        data = json.load(fh)
    original_last_run = data.get("last_run")
    try:
        data["last_run"] = {
            "kind": "align_auto", "at": "2026-09-05T12:00:00Z",
            "n_pairs_attempted": 91, "n_accepted": 24, "connected": True, "n_unregistered": 0,
            "elapsed_s": 1380.0, "n_pairs_before_filter": 91, "n_placement_seeds": 91,
            "adjacency_padding_m": 3.0,
        }
        with open(placement_path, "w") as fh:
            json.dump(data, fh, indent=2)

        _goto(page)
        page.wait_for_function("() => getComputedStyle($('seedSummary')).display !== 'none'", timeout=5000)

        def overlaps(a, b):
            return a["x0"] < b["x1"] and b["x0"] < a["x1"] and a["y0"] < b["y1"] and b["y0"] < a["y1"]

        n_tools_checked = 0
        for width, height in ((900, 700), (1920, 1080)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(100)  # let layout settle (pure CSS)
            boxes = page.evaluate("""() => {
                const rect = el => { const r = el.getBoundingClientRect();
                    return {x0:r.x, y0:r.y, x1:r.x+r.width, y1:r.y+r.height}; };
                return {
                    stateLine: rect($('stateLine')),
                    seedSummary: rect($('seedSummary')),
                    tools: [...document.querySelectorAll('#tools button')].map(rect),
                };
            }""")
            n_tools_checked = len(boxes["tools"])
            for name, badge in (("#stateLine", boxes["stateLine"]), ("#seedSummary", boxes["seedSummary"])):
                for i, tool_box in enumerate(boxes["tools"]):
                    assert not overlaps(badge, tool_box), (
                        f"at {width}x{height}: {name} {badge} overlaps .tool[{i}] {tool_box}"
                    )
            assert not overlaps(boxes["stateLine"], boxes["seedSummary"]), (
                f"at {width}x{height}: #stateLine {boxes['stateLine']} overlaps "
                f"#seedSummary {boxes['seedSummary']}"
            )
        print(f"    checked {n_tools_checked} tool button(s) against both badges at 900x700 and 1920x1080")
    finally:
        with open(placement_path) as fh:
            data = json.load(fh)
        if original_last_run is not None:
            data["last_run"] = original_last_run
        else:
            data.pop("last_run", None)
        with open(placement_path, "w") as fh:
            json.dump(data, fh, indent=2)


OVERLAP_SIZES = ((1024, 768), (1280, 720), (1440, 900), (1920, 1080))


def _overlap_findings(page):
    """One line per problem the shared overlap detector finds on the page as it is now."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_overlap_check.js"), encoding="utf-8") as fh:
        res = page.evaluate(fh.read())
    lines = [f"overlap: {o['a']} <-> {o['b']} ({o['w']}x{o['h']}px)" for o in res["overlaps"]]
    lines += [f"covered: {c['control']} under {c['coveredBy']}" for c in res["covered"]]
    lines += [f"clipped: {c['el']} ({c['why']}, {c['cut']}px)" for c in res["clipped"]]
    lines += [f"cut off: {c['el']} ({c['hidden']}% hidden by {c['by']})" for c in res["cutOff"]]
    return res["items"], lines


def test_no_text_or_controls_overlap_at_common_window_sizes(page):
    """No text or controls on top of each other, in the states a person sees, at four window sizes: nothing drawn over
    anything else, no control covered, nothing cut off. Read only."""
    _goto(page)
    problems, checked = [], 0
    for width, height in OVERLAP_SIZES:
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(300)
        states = (
            ("default", lambda: None, lambda: None),
            ("status log full", lambda: page.evaluate(
                "() => { setStatus('align done:'); for (let i = 0; i < 14; i++) appendStatus('  a scan vs another scan: "
                "fitness=0.848 shift mean=1.295m max=1.864m discarded tilt=2.24deg [applied]'); }"),
             lambda: page.evaluate("() => { $('out').style.display = 'none'; $('out').value = ''; }")),
            ("anchor tool", lambda: page.locator('[data-tool="anchor"]').click(), lambda: page.keyboard.press("Escape")),
            ("panels scrolled", lambda: page.evaluate(
                "() => { for (const id of ['left', 'right', 'bottom']) { const e = document.getElementById(id); "
                "e.scrollTop = id === 'bottom' ? 300 : e.scrollHeight; } }"),
             lambda: page.evaluate("() => { for (const id of ['left', 'right', 'bottom']) document.getElementById(id).scrollTop = 0; }")),
            ("menu Settings", lambda: page.get_by_role("button", name="Settings", exact=True).first.click(),
             lambda: page.keyboard.press("Escape")),
        )
        for name, setup, undo in states:
            setup()
            page.wait_for_timeout(250)
            items, lines = _overlap_findings(page)
            checked += 1
            assert items > 0, f"the detector saw no text at all at {width}x{height} [{name}]"
            problems += [f"{width}x{height} [{name}] {line}" for line in lines]
            undo()
            page.wait_for_timeout(150)
    assert not problems, "text or controls overlap or are cut off:\n  " + "\n  ".join(problems[:12])
    print(f"    {checked} size and state combinations checked, nothing overlapping, covered or cut off")


def test_cancel_running_job_stops_at_once_keeps_poses_and_looks_live(page):
    """While a job runs Cancel looks live; pressed, the banner says Cancelling then Cancelled, the job stops within
    3 s whatever it was doing, nothing is kept, poses are unchanged, and a second job starts and cancels the same way."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set - this check needs direct file access to the project")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path) as fh:
        before_poses = {s["name"]: (s["x_m"], s["y_m"], s["yaw_deg"], s["z_m"])
                        for s in json.load(fh)["scans"]}

    def start_and_cancel():
        # Match on path too, so a cancel POST to .../cancel is not mistaken for the start response.
        with page.expect_response(
            lambda r: r.url.endswith("/api/jobs") and r.request.method == "POST"
        ) as resp_info:
            _pick_run(page, "align_auto")
            _press_run(page)
        assert resp_info.value.status == 202, f"job did not start: {resp_info.value.status}"
        job_id = resp_info.value.json()["id"]
        page.wait_for_function("() => !$('bCancelJob').disabled", timeout=10_000)
        look = page.evaluate("""() => { const b = $('bCancelJob'), cs = getComputedStyle(b);
            return {dataDisabled: b.getAttribute('data-disabled'), cursor: cs.cursor}; }""")
        assert look["dataDisabled"] is None and look["cursor"] != "not-allowed", f"Cancel looks dead while live: {look}"
        page.wait_for_timeout(8000)                  # a real mid-run stop, deep inside some stage
        page.locator("#bCancelJob").click()
        t0 = time.time()
        assert page.locator("#jobNotice").inner_text() == "Cancelling...", "no Cancelling banner at once"
        while time.time() - t0 < 30:                # the job API, not the button, says whether it stopped
            job = call(f"{BENCH_URL}/api/jobs/{job_id}", "GET", timeout=10)
            if job["state"] != "running":
                break
            time.sleep(0.1)
        job["stop_s"] = time.time() - t0
        page.wait_for_function("() => $('jobNotice').textContent.startsWith('Cancelled')", timeout=5_000)
        return job

    _goto(page)
    job = start_and_cancel()
    assert job["state"] == "cancelled", f"job state is {job['state']!r}, expected 'cancelled'"
    assert job["result"]["cancelled"] is True and job["result"]["stopped_during"], job["result"]
    assert job["stop_s"] < 3.0, f"took {job['stop_s']:.1f} s to stop during {job['result']['stopped_during']!r}"

    with open(placement_path) as fh:
        after_poses = {s["name"]: (s["x_m"], s["y_m"], s["yaw_deg"], s["z_m"])
                       for s in json.load(fh)["scans"]}
    assert after_poses == before_poses, (
        "cancel changed a pose - it must leave every scan exactly where it was"
    )

    # The job store must not be left phantom-"running" - a second job must
    # start and be cancellable the same way the first was.
    job2 = start_and_cancel()
    assert job2["state"] == "cancelled", f"second job state is {job2['state']!r}, expected 'cancelled'"
    print(f"    job {job['id']}: stopped {job['stop_s']:.1f} s after Cancel, during {job['result']['stopped_during']!r}; "
          f"poses unchanged; a second job stopped in {job2['stop_s']:.1f} s")


def test_job_running_confirm_modal_stops_and_opens(page):
    """Opening a project while a job runs shows a confirm modal naming the job; "Stop it and open" stops the job and
    opens the project."""
    _goto(page)
    _pick_run(page, "align_auto")
    with page.expect_response(
        lambda r: "/api/jobs" in r.url and r.request.method == "POST"
    ) as resp_info:
        _press_run(page)
    assert resp_info.value.status == 202
    page.wait_for_function("() => !$('bCancelJob').disabled", timeout=10_000)

    page.goto(BENCH_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        "() => getComputedStyle($('projectGate')).display !== 'none'", timeout=LOAD_TIMEOUT_MS)
    page.locator("#startContinue").click()
    page.wait_for_function(
        "() => getComputedStyle($('jobRunningModal')).display !== 'none'", timeout=10_000)
    body_text = page.locator("#jobRunningModalBody").inner_text()
    assert "Automatic align" in body_text, f"modal does not name the running job kind: {body_text!r}"
    assert "elapsed" in body_text, f"modal does not mention elapsed time: {body_text!r}"

    page.locator("#jobRunningStop").click()
    page.wait_for_function(
        "() => $('app').classList.contains('open')", timeout=120_000)
    print("    job-running confirm modal named the job; Stop-it-and-open stopped it and opened the project")


def test_console_clean(page):
    errors = _goto(page)
    # A /favicon.ico 404 is a browser network log, not the page's script; matched by the real response (_goto).
    generic_resource_errors = [e for e in errors if e.startswith("Failed to load resource")]
    real = [e for e in errors if not e.startswith("Failed to load resource")]
    if generic_resource_errors and not getattr(page, "expected_404", False):
        # A generic resource-load error happened but it wasn't one of the
        # two expected 404s - something else failed, which is real.
        real = real + generic_resource_errors
    assert not real, f"{len(real)} console error(s): {real}"


def test_react_scale_bar_renders_and_tracks_zoom(page):
    """The scale bar (React, mounted into #rc-scalebar-root) actually renders and updates live as cam.s changes. The
    expected label is derived independently with ScaleBar.jsx's round-length rule."""
    _goto(page)
    page.wait_for_function(
        "() => $('rc-scalebar-root') && $('rc-scalebar-root').textContent.trim().length > 0",
        timeout=LOAD_TIMEOUT_MS,
    )

    def expected_label(px_per_m):
        steps = [5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        best, best_dist = steps[0], None
        for m in steps:
            px = m * px_per_m
            if 40 <= px <= 90:
                best = m
                best_dist = 0
                break
            dist = (40 - px) if px < 40 else (px - 90)
            if best_dist is None or dist < best_dist:
                best, best_dist = m, dist
        return f"{best} m" if best >= 1 else f"{round(best * 100)} cm"

    cam_s = page.evaluate("() => cam.s")
    label = page.locator("#rc-scalebar-root").inner_text().strip()
    assert label == expected_label(cam_s), f"cam.s={cam_s}: expected {expected_label(cam_s)!r}, got {label!r}"

    # Zoom with the wheel and check the label follows, in Plan and in Iso. The camera is orthographic, so cam.s is the
    # true scale in either view.
    for view in ("plan", "iso"):
        page.locator(f'[data-view="{view}"]').click()
        rect = _canvas_rect(page)
        page.mouse.move(rect["x"] + 300, rect["y"] + 300)
        page.mouse.wheel(0, -1500)
        page.wait_for_timeout(250)  # one rAF tick's worth of margin for the poll to catch up
        cam_s_after = page.evaluate("() => cam.s")
        label_after = page.locator("#rc-scalebar-root").inner_text().strip()
        assert label_after == expected_label(cam_s_after), (
            f"{view}: cam.s={cam_s_after}: expected {expected_label(cam_s_after)!r}, got {label_after!r}"
        )
        print(f"    {view}: cam.s={cam_s_after:.2f}px/m, scale bar reads {label_after!r}")


def test_react_zoom_slider_drags_the_view_and_tracks_the_wheel(page):
    """The zoom slider works both ways: dragging its thumb changes cam.s, and a wheel zoom afterwards shows up in the
    slider's aria-valuenow."""
    _goto(page)
    root = page.locator("#rc-zoomslider-root")
    page.wait_for_function(
        "() => document.querySelector('#rc-zoomslider-root [role=slider]') !== null",
        timeout=LOAD_TIMEOUT_MS,
    )
    slider_handle = root.locator("[role=slider]").first
    track = root.locator('[class*="Slider-track"]').first

    cam_s_before = page.evaluate("() => cam.s")
    thumb_box = slider_handle.bounding_box()
    track_box = track.bounding_box()
    assert thumb_box and track_box, "could not find the zoom slider's own thumb/track to drag"

    page.mouse.move(thumb_box["x"] + thumb_box["width"] / 2, thumb_box["y"] + thumb_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(track_box["x"] + track_box["width"] * 0.85, thumb_box["y"] + thumb_box["height"] / 2, steps=8)
    page.mouse.up()
    page.wait_for_timeout(200)

    cam_s_after_drag = page.evaluate("() => cam.s")
    assert cam_s_after_drag != cam_s_before, (
        f"dragging the zoom slider did not change cam.s: stayed at {cam_s_before}"
    )
    slider_reports = float(slider_handle.get_attribute("aria-valuenow"))
    assert abs(slider_reports - cam_s_after_drag) < 0.01, (
        f"slider's own displayed value ({slider_reports}) disagrees with the cam.s it just set ({cam_s_after_drag})"
    )

    # The read direction: a wheel zoom must show on the slider too, proving the poll still runs after a drag.
    rect = _canvas_rect(page)  # {x, y} only - see its own definition
    page.mouse.move(rect["x"] + 300, rect["y"] + 300)
    page.mouse.wheel(0, -600)
    page.wait_for_timeout(300)
    cam_s_after_wheel = page.evaluate("() => cam.s")
    slider_after_wheel = float(slider_handle.get_attribute("aria-valuenow"))
    assert cam_s_after_wheel != cam_s_after_drag, "wheel zoom did not change cam.s at all"
    assert abs(slider_after_wheel - cam_s_after_wheel) < 0.01, (
        f"wheel changed cam.s to {cam_s_after_wheel} but the slider still reads {slider_after_wheel} - "
        f"the read-side poll looks stuck"
    )
    print(f"    drag: cam.s {cam_s_before:.1f} -> {cam_s_after_drag:.1f} (slider agreed); "
          f"wheel: -> {cam_s_after_wheel:.1f} (slider caught up to {slider_after_wheel:.1f})")


def test_react_clip_sliders_all_visible_and_independent(page):
    """All three clip edge sliders are visible at once, and dragging one axis's slider changes only that axis, in the JS
    model and with a real mouse drag. Persistence is covered by test_clip_state_survives_a_reload."""
    _goto(page)
    for axis in ("x", "y", "z"):
        page.wait_for_function(
            f"() => document.querySelectorAll('#rc-clip-{axis}-root [role=slider]').length === 2",
            timeout=LOAD_TIMEOUT_MS,
        )
    # All three visible at once, unlike the old #clipAxisSeg tab widget -
    # the actual point of "nothing to switch between."
    for axis in ("x", "y", "z"):
        box = page.locator(f"#rc-clip-{axis}-root").bounding_box()
        assert box and box["width"] > 0, f"#rc-clip-{axis}-root has no visible size"

    before = {ax: dict(page.evaluate(f"() => ({{on: clip.{ax}.on, lo: clip.{ax}.lo, hi: clip.{ax}.hi}})")) for ax in "xyz"}

    z_thumbs = page.locator("#rc-clip-z-root [role=slider]")
    thumb_box = z_thumbs.first.bounding_box()
    assert thumb_box, "could not find Z's own slider thumb to drag"
    # Z is mounted vertical (orientation="vertical") - drag along Y, not X.
    page.mouse.move(thumb_box["x"] + thumb_box["width"] / 2, thumb_box["y"] + thumb_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(thumb_box["x"] + thumb_box["width"] / 2, thumb_box["y"] + thumb_box["height"] / 2 - 60, steps=8)
    page.mouse.up()
    page.wait_for_timeout(200)

    after = {ax: dict(page.evaluate(f"() => ({{on: clip.{ax}.on, lo: clip.{ax}.lo, hi: clip.{ax}.hi}})")) for ax in "xyz"}
    assert after["z"] != before["z"], f"dragging Z's slider did not change clip.z: {before['z']} -> {after['z']}"
    assert after["x"] == before["x"], f"dragging Z's slider also changed clip.x: {before['x']} -> {after['x']}"
    assert after["y"] == before["y"], f"dragging Z's slider also changed clip.y: {before['y']} -> {after['y']}"

    # The checkbox writes the state and does not keep keyboard focus (which would swallow hotkeys).
    x_checkbox = page.locator("#rc-clip-x-root input[type=checkbox]")
    x_on_before = page.evaluate("() => clip.x.on")
    x_checkbox.click()
    x_on_after = page.evaluate("() => clip.x.on")
    assert x_on_after != x_on_before, f"clicking X's checkbox did not toggle clip.x.on: stayed {x_on_before}"
    active_tag = page.evaluate("() => document.activeElement.tagName")
    page.keyboard.press("h")  # an unrelated hotkey - must still fire
    heatmap_on = page.evaluate("() => $('cbHeatmap').checked")
    assert heatmap_on, f"'h' hotkey did not fire after clicking X's clip checkbox (focus stuck on {active_tag!r})"
    page.keyboard.press("h")  # restore
    x_checkbox.click()  # restore

    print(f"    Z drag changed clip.z {before['z']} -> {after['z']}, X/Y untouched; "
          f"X checkbox toggled clip.x.on {x_on_before} -> {x_on_after} without stranding hotkeys")


def test_scan_list_matches_api(page):
    _goto(page)
    api_scans = call(f"{BENCH_URL}/api/scans", "GET", timeout=10)["scans"]
    rows = page.locator("#scanlist .scan")
    n_rows = rows.count()
    assert n_rows == len(api_scans), (
        f"sidebar shows {n_rows} scan(s), /api/scans lists {len(api_scans)}"
    )
    counts = page.evaluate("() => S.map(s => s.points)")
    assert len(counts) == n_rows
    zero = [c for c in counts if not c or c <= 0]
    assert not zero, f"{len(zero)} of {n_rows} scan(s) have a zero/missing point count"
    print(f"    {n_rows} scan(s), point counts: {counts}")


def test_fresh_load_preserves_saved_pose(page):
    """A scan with a saved pose must reload exactly where it was saved, not be spread out like a new import.

    Plants the sample venue's saved poses into this run's QA project, reloads, and compares every pose. A project of
    other scans gets the venue's poses in turn, one per scan. Needs PROJECT_DIR; skips without it."""
    if not PROJECT_DIR:
        print("    PROJECT_DIR not set, skipping (need filesystem access to plant the fixture placement)")
        return
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "wedding_venue_1_placement.json")
    with open(fixture_path) as fh:
        fixture = json.load(fh)
    # The fixture names scans by room; match each to the project's scan file of that room, whatever its prefix.
    raw_dir = os.path.join(PROJECT_DIR, "scans-raw")
    on_disk = [f[:-4] for f in os.listdir(raw_dir) if f.endswith(".pts")] if os.path.isdir(raw_dir) else []
    for s in fixture["scans"]:
        s["name"] = next((n for n in on_disk if n == s["name"] or n.endswith(" " + s["name"])), s["name"])
    if not set(on_disk) & {s["name"] for s in fixture["scans"]}:   # not the venue's scans: lend its poses out
        fixture["scans"] = [dict(s, name=n) for s, n in zip(fixture["scans"], sorted(on_disk))]
    # Put back afterwards: on other scans the venue's poses would scatter them for every later test.
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    with open(placement_path, "w") as fh:
        json.dump(fixture, fh)
    try:
        _goto(page)
        n_expected = len(fixture["scans"])
        page.wait_for_function(
            "(n) => Array.isArray(S) && S.filter(s => s.hasSavedPose).length === n",
            arg=n_expected, timeout=LOAD_TIMEOUT_MS,
        )

        live = page.evaluate(
            "() => Object.fromEntries(S.map(s => "
            "[s.name, {x: s.x, y: s.y, yawDeg: s.yaw * 180 / Math.PI, z: s.dz}]))"
        )

        missing = [s["name"] for s in fixture["scans"] if s["name"] not in live]
        assert not missing, f"fresh load lost {len(missing)} scan(s) that had a saved pose: {missing}"

        worst = 0.0
        for s in fixture["scans"]:
            got = live[s["name"]]
            dx, dy, dz = abs(got["x"] - s["x_m"]), abs(got["y"] - s["y_m"]), abs(got["z"] - s["z_m"])
            dyaw = abs(((got["yawDeg"] - s["yaw_deg"]) + 180) % 360 - 180)
            worst = max(worst, dx, dy, dz, dyaw)
            assert dx < 1e-6 and dy < 1e-6 and dz < 1e-6, (
                f"{s['name']}: fresh load moved a scan that already had a saved pose - "
                f"x {got['x']!r} vs saved {s['x_m']!r}, y {got['y']!r} vs {s['y_m']!r}, "
                f"z {got['z']!r} vs {s['z_m']!r}"
            )
            assert dyaw < 1e-6, (
                f"{s['name']}: yaw moved on fresh load - {got['yawDeg']!r} vs saved {s['yaw_deg']!r} deg"
            )
        print(f"    all {n_expected} scans reloaded with their saved pose "
              f"unchanged (largest deviation {worst:.2e})")
    finally:
        with open(placement_path, "wb") as fh:
            fh.write(original)


def test_webgl_renderer(page):
    _goto(page)
    text = page.locator("#hudBuild").inner_text()
    assert "WebGL" in text, f"HUD build line does not report WebGL: {text!r}"


def test_build_info_matches_server(page):
    _goto(page)
    page_info = page.evaluate("() => window.BUILD_INFO")
    server_info = call(f"{BENCH_URL}/api/version", "GET", timeout=10)
    assert page_info["build_time"] == server_info["build_time"], (
        f"page build_time {page_info['build_time']!r} != "
        f"server build_time {server_info['build_time']!r}"
    )


def test_gpu_status_matches_reality(page):
    """The Settings GPU checkbox matches what /api/gpu-status reports, whichever way that goes, so the test passes with
    or without a GPU but fails on a disagreement."""
    _goto(page)
    status = call(f"{BENCH_URL}/api/gpu-status", "GET", timeout=10)
    page.wait_for_function(
        "() => $('gpuStatusNote').textContent !== 'checking GPU status…'",
        timeout=10_000,
    )
    checked = page.locator("#cbGpu").is_checked()
    disabled = page.locator("#cbGpu").is_disabled()
    note_text = page.locator("#gpuStatusNote").inner_text()
    if status["gpu_usable"]:
        assert not disabled, f"card is usable ({status}) but the checkbox is disabled"
        assert str(status["cuda_device_count"]) in note_text, (
            f"usable GPU status not reflected in the note text: {note_text!r}"
        )
    else:
        assert disabled, f"card is not usable ({status}) but the checkbox is not disabled"
        assert not checked, "an unusable GPU must not be left checked"
    print(f"    gpu_usable={status['gpu_usable']}, checkbox disabled={disabled}, note: {note_text[:70]!r}")


def test_canvas_draws_something(page):
    _goto(page)
    # Composite both canvases (cvgl: WebGL points, cv: everything else) offscreen and count pixels that differ from
    # the dark background (#0b0e12), in one evaluate call so the WebGL buffer cannot clear in between.
    n_diff = page.evaluate(
        """() => {
            draw();
            const off = document.createElement('canvas');
            off.width = cv.width; off.height = cv.height;
            const octx = off.getContext('2d');
            octx.drawImage(cvgl, 0, 0);
            octx.drawImage(cv, 0, 0);
            const data = octx.getImageData(0, 0, off.width, off.height).data;
            let n = 0;
            for (let i = 0; i < data.length; i += 4) {
                if (Math.abs(data[i]-0x0b) > 6 || Math.abs(data[i+1]-0x0e) > 6
                    || Math.abs(data[i+2]-0x12) > 6) n++;
            }
            return n;
        }"""
    )
    assert n_diff > 500, f"only {n_diff} non-background pixel(s) drawn - looks blank"
    print(f"    {n_diff} non-background pixels")


def test_drag_moves_scan(page):
    _goto(page)
    # Navigate is the default tool and never moves a scan, so switch to Move first.
    page.locator('[data-tool="move"]').click()
    scan_i = _select_scan_row(page, 0)
    before = _pose(page, scan_i)
    pt = _hit_testable_point(page, scan_i)
    assert pt, f"could not find any point of scan {scan_i} that the page's own hit() resolves to it"
    rect = _canvas_rect(page)
    vx, vy = _to_viewport(rect, (pt["x"], pt["y"]))
    cam_s = page.evaluate("() => cam.s")  # px per metre, for predicting the expected shift

    dx_px, dy_px = 46, -31
    page.mouse.move(vx, vy)
    page.mouse.down()
    diag = page.evaluate("() => (typeof drag !== 'undefined' && drag) ? {mode: drag.mode, primary: drag.primary} : null")
    page.mouse.move(vx + dx_px, vy + dy_px, steps=8)
    page.mouse.up()

    after = _pose(page, scan_i)
    expected_dx_m = dx_px / cam_s
    expected_dy_m = -dy_px / cam_s  # plan view: screen Y is inverted vs world Y
    got_dx_m = after["x"] - before["x"]
    got_dy_m = after["y"] - before["y"]
    assert abs(got_dx_m - expected_dx_m) < 0.02, (
        f"x moved {got_dx_m:.3f} m, expected {expected_dx_m:.3f} m from a {dx_px}px drag "
        f"at {cam_s:.1f} px/m (mousedown at ({vx:.1f},{vy:.1f}) started drag={diag!r})"
    )
    assert abs(got_dy_m - expected_dy_m) < 0.02, (
        f"y moved {got_dy_m:.3f} m, expected {expected_dy_m:.3f} m from a {dy_px}px drag "
        f"at {cam_s:.1f} px/m"
    )
    print(f"    dragged {dx_px},{dy_px}px at {cam_s:.1f}px/m -> "
          f"dx={got_dx_m:.3f}m dy={got_dy_m:.3f}m")


def test_navigate_never_moves_scan(page):
    """Navigate is the default tool and a drag in it never moves a scan, whatever is under the cursor: the pose is
    unchanged and the camera did move, so the drag was taken as a pan."""
    _goto(page)
    default_tool = page.evaluate("() => tool")
    assert default_tool == "navigate", f"default tool on load is {default_tool!r}, expected 'navigate'"

    scan_i = _select_scan_row(page, 0)
    before = _pose(page, scan_i)
    pt = _hit_testable_point(page, scan_i)
    assert pt, f"could not find any point of scan {scan_i} to drag directly over"
    rect = _canvas_rect(page)
    vx, vy = _to_viewport(rect, (pt["x"], pt["y"]))
    cam_before = page.evaluate("() => ({x: cam.x, y: cam.y})")

    page.mouse.move(vx, vy)
    page.mouse.down()
    page.mouse.move(vx + 60, vy + 40, steps=8)
    page.mouse.up()

    after = _pose(page, scan_i)
    cam_after = page.evaluate("() => ({x: cam.x, y: cam.y})")

    assert after["x"] == before["x"] and after["y"] == before["y"] and after["yaw"] == before["yaw"], (
        f"a drag starting directly over scan {scan_i}'s own points moved it while "
        f"in Navigate mode: before={before}, after={after}"
    )
    moved_cam = abs(cam_after["x"] - cam_before["x"]) + abs(cam_after["y"] - cam_before["y"])
    assert moved_cam > 1e-6, (
        f"the camera did not move either ({cam_before} -> {cam_after}) - the drag looks "
        f"dropped entirely rather than consumed as a pan"
    )
    print(f"    scan pose exactly unchanged (x={before['x']:.4f} y={before['y']:.4f} "
          f"yaw={before['yaw']:.4f}), camera panned by {moved_cam:.3f} instead")


def _cam_azi_ele(page):
    return page.evaluate("() => ({x: cam.x, y: cam.y, azi: azi, ele: ele})")


def test_left_drag_pans_outside_plan_view(page):
    """Left-drag in Navigate pans outside Plan too (whenever the elevation is not near the horizon): the camera
    translates and azi/ele stay exactly unchanged."""
    _goto(page)
    page.locator('[data-view="iso"]').click()
    before = _cam_azi_ele(page)

    rect = _canvas_rect(page)
    cx, cy = rect["x"] + 300, rect["y"] + 300
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + 50, cy - 30, steps=8)
    page.mouse.up()

    after = _cam_azi_ele(page)
    assert after["azi"] == before["azi"] and after["ele"] == before["ele"], (
        f"left-drag in Iso still rotated the view: {before} -> {after}"
    )
    moved = abs(after["x"] - before["x"]) + abs(after["y"] - before["y"])
    assert moved > 1e-6, f"left-drag in Iso did not pan either: {before} -> {after}"
    print(f"    Iso left-drag panned by {moved:.3f} (world units), azi/ele held at "
          f"{before['azi']:.1f}/{before['ele']:.1f}")


def test_middle_and_right_drag_always_orbit(page):
    """Middle and right drag orbit in every view, Plan included."""
    _goto(page)
    rect = _canvas_rect(page)
    cx, cy = rect["x"] + 300, rect["y"] + 300
    for view in ("plan", "iso"):
        page.locator(f'[data-view="{view}"]').click()
        for button in ("middle", "right"):
            before = _cam_azi_ele(page)
            page.mouse.move(cx, cy)
            page.mouse.down(button=button)
            page.mouse.move(cx + 40, cy + 25, steps=8)
            page.mouse.up(button=button)
            after = _cam_azi_ele(page)
            assert after["x"] == before["x"] and after["y"] == before["y"], (
                f"{button}-drag in {view} panned the camera instead of orbiting: {before} -> {after}"
            )
            rotated = abs(after["azi"] - before["azi"]) + abs(after["ele"] - before["ele"])
            assert rotated > 1e-6, f"{button}-drag in {view} did not orbit either: {before} -> {after}"
    print("    middle- and right-drag orbit (never pan) confirmed in both Plan and Iso")


def test_side_view_is_exactly_level_and_left_drag_pans_it(page):
    """Side is exactly level, and left-drag slides the picture with the mouse, sideways and up/down."""
    _goto(page)
    page.locator('[data-view="side"]').click()
    assert page.evaluate("() => ele") == 0, "Side view is not exactly level"
    probe = "() => proj(1.5, -2.0, 3.0)"  # any fixed world point; it must follow the mouse
    before = page.evaluate(probe)
    angles = page.evaluate("() => ({azi: azi, ele: ele})")

    rect = _canvas_rect(page)
    cx, cy = rect["x"] + 300, rect["y"] + 300
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + 50, cy + 30, steps=8)
    hint_shown = page.locator("#hint").is_visible()
    page.mouse.up()

    after = page.evaluate(probe)
    assert page.evaluate("() => ({azi: azi, ele: ele})") == angles, "left-drag orbited a level view instead of panning"
    assert abs((after[0] - before[0]) - 50) < 0.01 and abs((after[1] - before[1]) - 30) < 0.01, (
        f"the picture did not follow a 50,30 px drag: a fixed point moved {after[0]-before[0]:.3f},"
        f"{after[1]-before[1]:.3f} px"
    )
    assert not hint_shown, "an 'orbiting instead' style hint is still being shown in a level view"
    print(f"    Side is ele=0 exactly; a 50,30 px left-drag moved a fixed point by "
          f"{after[0]-before[0]:.2f},{after[1]-before[1]:.2f} px, no orbit, no hint")


def test_shift_drag_height_unaffected_by_the_pan_orbit_swap(page):
    """Shift+drag still changes height in every view. Needs the Move tool: in Navigate a drag always pans or orbits."""
    _goto(page)
    page.locator('[data-view="iso"]').click()
    page.locator('[data-tool="move"]').click()
    scan_i = _select_scan_row(page, 0)
    before = _pose(page, scan_i)
    pt = _hit_testable_point(page, scan_i)
    assert pt, f"could not find any point of scan {scan_i} to drag directly over"
    rect = _canvas_rect(page)
    vx, vy = _to_viewport(rect, (pt["x"], pt["y"]))

    page.mouse.move(vx, vy)
    page.keyboard.down("Shift")
    page.mouse.down()
    page.mouse.move(vx, vy - 40, steps=8)
    page.mouse.up()
    page.keyboard.up("Shift")

    after = _pose(page, scan_i)
    assert after["dz"] != before["dz"], f"shift+drag in Iso did not change height: {before} -> {after}"
    assert after["x"] == before["x"] and after["y"] == before["y"], (
        f"shift+drag in Iso also moved x/y, expected height only: {before} -> {after}"
    )
    print(f"    shift+drag height in Iso: dz {before['dz']:.3f} -> {after['dz']:.3f}, x/y unchanged")


def test_every_way_into_side_view_is_exactly_level(page):
    """Toolbar, view cube faces and level-view edges, and numpad 2/4/6/8 all land at elevation 0 with the height
    offset cleared."""
    _goto(page)

    def dirty():  # start from a non-Side view with a stale height offset, so a reset is visible
        page.locator('[data-view="iso"]').click()
        page.evaluate("() => { cam.z = 1.5; }")

    def check(how):
        got = page.evaluate("() => ({view: view, ele: ele, z: cam.z})")
        assert got == {"view": "side", "ele": 0, "z": 0}, f"{how}: expected exactly level Side, got {got}"

    dirty()
    page.locator('[data-view="side"]').click()
    check("toolbar Side")
    for face in ("side0", "side90", "side180", "side-90"):
        dirty()
        page.locator(f'.vc-face[data-vc="{face}"]').dispatch_event("click")
        check(f"view cube face {face}")
    edges = page.locator('.vc-hot[title="level view"]')
    assert edges.count() == 4, f"expected 4 level-view edge hotspots on the view cube, found {edges.count()}"
    for i in range(4):
        dirty()
        edges.nth(i).dispatch_event("click")
        check(f"view cube level-view edge #{i}")
    for key in ("2", "4", "6", "8"):
        dirty()
        page.keyboard.press(f"Numpad{key}")
        check(f"Numpad{key}")
    print("    toolbar, 4 cube faces, 4 cube edges and numpad 2/4/6/8 all land on ele=0, cam.z=0")


def test_side_view_boxes_and_walls_draw_flat(page):
    """At every azimuth in Side view a box's top and bottom edges are exactly level on screen, and every detected wall
    is a real vertical rectangle that rises with the scan's height offset."""
    _goto(page)
    page.locator('[data-view="side"]').click()
    res = page.evaluate(r"""() => {
        const out = {boxSpread: 0, walls: 0, wallBad: [], minHeight: 1e9, dzShift: null};
        // A scan the user has tilted legitimately tilts its walls; level them for the check, put back after.
        const saved = S.map(q => [q.pitch, q.roll, q.dz]);
        for (const q of S) { q.pitch = 0; q.roll = 0; }
        for (const a of [0, 90, 180, 270]) {
            jumpView('side', a, 0);
            const ys = {0: [], 3: []};
            for (const x of [0, 6]) for (const y of [0, 9]) for (const z of [0, 3]) ys[z].push(proj(x, y, z)[1]);
            for (const z of [0, 3]) out.boxSpread = Math.max(out.boxSpread, Math.max(...ys[z]) - Math.min(...ys[z]));
            for (const s of S) {
                if (!s.walls || !s.walls.length || !s.xy || !s.xy.length) continue;
                for (const w of s.walls) {
                    const c = wallCorners(s, w).map(p => proj(p[0], p[1], p[2]));
                    out.walls++;
                    const [z0, z1] = wallHeightRange(s, w);
                    out.minHeight = Math.min(out.minHeight, z1 - z0);
                    const level = Math.abs(c[0][1] - c[1][1]) < 1e-6 && Math.abs(c[2][1] - c[3][1]) < 1e-6;
                    const upright = Math.abs(c[0][0] - c[3][0]) < 1e-6 && Math.abs(c[1][0] - c[2][0]) < 1e-6;
                    if (!level || !upright || !(c[2][1] < c[0][1])) out.wallBad.push(`${s.name} W${w.i} azi ${a}`);
                }
            }
        }
        const s = S.find(q => q.walls && q.walls.length && q.xy && q.xy.length);
        if (s) {
            const z0 = wallCorners(s, s.walls[0])[0][2];
            s.dz += 2;
            out.dzShift = wallCorners(s, s.walls[0])[0][2] - z0;
        }
        S.forEach((q, i) => { q.pitch = saved[i][0]; q.roll = saved[i][1]; q.dz = saved[i][2]; });
        return out;
    }""")
    assert res["boxSpread"] < 1e-6, f"a level box edge is not level on screen: spread {res['boxSpread']} px"
    assert res["walls"] > 0, "no detected walls found on any scan, so the wall check proved nothing"
    assert not res["wallBad"], f"walls that are not level, upright rectangles: {res['wallBad'][:5]}"
    assert res["minHeight"] > 0.3, f"a wall came out only {res['minHeight']:.2f} m tall: no vertical extent was found"
    assert res["dzShift"] is not None and abs(res["dzShift"] - 2) < 1e-6, (
        f"raising a scan 2 m moved its walls by {res['dzShift']}, so they are not following the points")
    print(f"    {res['walls']} wall rectangles checked at 4 azimuths, all level and upright, shortest "
          f"{res['minHeight']:.2f} m; box edges level to {res['boxSpread']:.1e} px; walls follow height offset")


def test_projection_is_orthographic(page):
    """No perspective: the same world step is the same screen step wherever it starts, in every view, and the shader's
    w is always 1."""
    _goto(page)
    worst = page.evaluate(r"""() => {
        let worst = 0;
        for (const [a, e] of [[0, 90], [35, 32], [200, 20], [90, 0], [-45, 60]]) {
            jumpView('iso', a, e);
            for (const d of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, -3, 4]]) {
                let ref = null;
                for (const p of [[0, 0, 0], [7, -4, 2], [-30, 25, 9], [100, 80, -5]]) {
                    const p0 = proj(p[0], p[1], p[2]), p1 = proj(p[0] + d[0], p[1] + d[1], p[2] + d[2]);
                    const step = [p1[0] - p0[0], p1[1] - p0[1]];
                    if (!ref) ref = step;
                    worst = Math.max(worst, Math.abs(step[0] - ref[0]), Math.abs(step[1] - ref[1]));
                }
            }
        }
        return worst;
    }""")
    assert worst < 1e-6, f"a world step gave different screen steps at different places (worst {worst} px): not orthographic"
    ws = page.evaluate(r"""() => {
        const src = Array.from(document.scripts).map(s => s.textContent).join('\n');
        return Array.from(src.matchAll(/gl_Position\s*=\s*vec4\(([^;]*)\)\s*;/g)).map(m => m[1].replace(/\s+/g, ''));
    }""")
    assert len(ws) >= 2, f"could not find the vertex shader's gl_Position assignments: {ws}"
    assert all(w.endswith(",1.0") for w in ws), f"a gl_Position has a w other than 1.0 (a perspective divide): {ws}"
    hud = page.locator("#hudCam").inner_text()
    assert "orthographic" in hud, f"the HUD does not say the view is orthographic: {hud!r}"
    print(f"    linear in 5 views (worst error {worst:.1e} px), {len(ws)} shader gl_Position(s) all w=1.0, HUD says orthographic")


def test_wheel_zoom_in_a_level_view_holds_the_point_under_the_cursor(page):
    """Wheel zoom in Side view keeps the world point under the cursor fixed sideways and in height, even after the view
    has been panned up."""
    _goto(page)
    page.locator('[data-view="side"]').click()
    page.evaluate("() => { cam.z = 2.0; draw(); }")
    rect = _canvas_rect(page)
    mx, my = 420, 260
    world = page.evaluate(
        "([mx, my]) => { const w = s2w(mx, my); return [w[0], w[1], cam.z + (cv.height/2 - my)/cam.s]; }", [mx, my])
    s_before = page.evaluate("() => cam.s")
    page.mouse.move(rect["x"] + mx, rect["y"] + my)
    page.mouse.wheel(0, -120)
    page.mouse.wheel(0, -120)
    s_after = page.evaluate("() => cam.s")
    assert s_after > s_before, f"wheel did not zoom in ({s_before} -> {s_after})"
    got = page.evaluate("(p) => proj(p[0], p[1], p[2])", world)
    assert abs(got[0] - mx) < 0.01 and abs(got[1] - my) < 0.01, (
        f"the point under the cursor drifted to {got[0]:.2f},{got[1]:.2f} from {mx},{my} while zooming")
    print(f"    zoom {s_before:.1f} -> {s_after:.1f} px/m held the point under the cursor within 0.01 px")


def test_a_burst_of_wheel_events_zooms_by_distance_and_draws_once(page):
    """A touchpad sends many small wheel events per frame. The zoom follows the distance scrolled, in pixels or lines,
    so it is the same however a browser batches them, and the view redraws once per frame, not once per event."""
    _goto(page)
    result = page.evaluate("""async () => {
        const r = cv.getBoundingClientRect(), x = r.left + 300, y = r.top + 200;
        const wheel = (dy, mode) => cv.dispatchEvent(new WheelEvent('wheel',
            {clientX: x, clientY: y, deltaY: dy, deltaMode: mode, bubbles: true, cancelable: true}));
        await new Promise(res => requestAnimationFrame(res));
        const realGL = glDrawPoints; let draws = 0;
        glDrawPoints = function () { draws++; return realGL.apply(this, arguments); };
        try {
            const s0 = cam.s;
            for (let i = 0; i < 20; i++) wheel(-5, 0);           // 20 x 5 px = one notch, in one frame
            const pixels = cam.s / s0, drawsNow = draws;
            await new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
            const s1 = cam.s;
            for (let i = 0; i < 4; i++) wheel(1.5, 1);            // 4 x 1.5 lines = one notch out
            return {pixels, lines: cam.s / s1, drawsNow, drawsAfterFrame: draws};
        } finally { glDrawPoints = realGL; }
    }""")
    assert abs(result["pixels"] - 1.1) < 1e-6, f"one notch of small pixel steps zoomed x{result['pixels']}, not x1.1"
    assert abs(result["lines"] - 1 / 1.1) < 1e-6, f"one notch of small line steps zoomed x{result['lines']}, not x1/1.1"
    assert result["drawsNow"] == 0, f"{result['drawsNow']} redraws ran inside the burst instead of waiting for the frame"
    assert result["drawsAfterFrame"] == 1, f"20 wheel events in one frame drew {result['drawsAfterFrame']} times, not once"
    print(f"    20 pixel steps zoomed x{result['pixels']:.4f}, 4 line steps x{result['lines']:.4f}, one redraw")


def test_run_status_lines_say_what_actually_ran(page):
    """The post-run vote and coarse-search lines follow what the server reports, not the checkboxes."""
    _goto(page)
    got = page.evaluate(r"""() => ({
        off: voteStatusLine({detector_vote_enabled: false, adjacency_method: 'feature'}),
        unseeded: voteStatusLine({detector_vote_enabled: true, adjacency_method: 'bbox'}),
        on: voteStatusLine({detector_vote_enabled: true, adjacency_method: 'feature'}),
        allRan: coarseSearchStatusLine({n_placement_seeds: 0, n_pairs_after_filter: 8}),
        some: coarseSearchStatusLine({n_placement_seeds: 9, n_pairs_after_filter: 11}),
        none: coarseSearchStatusLine({n_placement_seeds: 8, n_pairs_after_filter: 8}),
    })""")
    assert got["off"].endswith("off"), got["off"]
    assert "did NOT run" in got["unseeded"] and not got["unseeded"].endswith("ON"), got["unseeded"]
    assert got["on"].endswith("ON"), got["on"]
    assert "ran for all 8" in got["allRan"], got["allRan"]
    assert "skipped for 9 of 11" in got["some"] and "ran normally for the other 2" in got["some"], got["some"]
    assert "skipped for 8 of 8" in got["none"] and "other" not in got["none"], got["none"]
    print("    vote line: off / did NOT run when unseeded / ON; coarse line: ran for all / skipped for 9 of 11 / skipped for 8 of 8")


def test_links_table_shows_a_refined_seed_pair_as_moved_not_kept(page):
    """A pair whose result is the refined seed (method 'refined_seed') reads 'kept, fine-tuned', not 'kept placement',
    with its plain-language explanation naming the scans as shown."""
    _goto(page)
    res = page.evaluate(r"""() => {
        const a = S[0].name, b = S[1].name;
        const poseOf = s => [s.x, s.y, s.yaw * 180 / Math.PI];
        applyPersistedLinksOnLoad([
            {source_name: a, target_name: b, accepted: true, method: 'refined_seed',
             fitness: 0.04, rmse_m: 0.02, tilt_deg: 0.5,
             why: 'Kept near where you put it, fine-tuned by 33 cm. The search found only 2.7% of ' + a + "'s points on a surface of " + b + '.',
             a_pose_at_align: poseOf(S[0]), b_pose_at_align: poseOf(S[1])},
        ]);
        renderLinks();
        const L = LINKS.find(l => (S[l.a].name === a && S[l.b].name === b) || (S[l.a].name === b && S[l.b].name === a));
        return {state: L.state, on: L.on, shownA: displayName(S[0])};
    }""")
    page.wait_for_timeout(300)
    html = page.evaluate("() => document.getElementById('links').innerHTML")
    why = page.locator("#links .linkWhy").first.inner_text()
    assert res["on"] is True and res["state"] == "refined", f"a refined_seed pair should read 'refined': {res}"
    assert "kept, fine-tuned" in html, "the links table does not show the 'kept, fine-tuned' label"
    assert "kept placement" not in html, "a moved pair is still labelled 'kept placement'"
    assert "fine-tuned by 33 cm" in why and res["shownA"] in why, f"the explanation is missing or uses file names: {why!r}"
    print(f"    a refined_seed pair reads 'kept, fine-tuned' and explains itself: {why[:90]}...")


def test_run_summary_counts_kept_placements_apart_from_real_matches(page):
    """The run summary counts each outcome separately, so kept placements are not reported as accepted moves."""
    _goto(page)
    line = page.evaluate("""() => describeLinkOutcomes([
        {accepted: true, method: 'fpfh_ransac'}, {accepted: true, method: 'kept_placement'},
        {accepted: true, method: 'kept_placement'}, {accepted: true, method: 'refined_seed'},
        {accepted: false, method: 'fpfh_ransac'}])""")
    assert line.startswith("pairs tried: 5:"), line
    assert "1 found a real match" in line and "2 kept your placement (not tested" in line, line
    assert "1 kept near your placement and fine-tuned" in line and "1 rejected" in line and "accepted" not in line, line
    src = page.evaluate("() => runAlignAuto.toString()")
    assert "describeLinkOutcomes(" in src and "accepted: ${" not in src, "the end of a run no longer prints this summary"
    print(f"    {line}")


def test_a_scan_that_did_not_connect_is_pointed_at_review(page):
    """After a run, a scan that did not connect gets a notice whose button opens review on it."""
    _goto(page)
    name, shown, i = page.evaluate("() => [S[2].name, displayName(S[2]), S[2].i]")
    page.evaluate("(n) => showConnectNudge([n])", name)
    note = page.locator("#connectNudge")
    assert note.is_visible() and f"{shown} did not connect" in note.inner_text(), note.inner_text()
    note.locator("button", has_text="Review it").click()
    assert page.evaluate("() => review && review.i") == i, "the button did not open review on that scan"
    assert not note.is_visible(), "the notice stayed up"
    page.locator("#rvDone").click()
    src = page.evaluate("() => runAlignAuto.toString()")
    assert "showConnectNudge(job.result.unregistered)" in src, "the end of a run no longer points at review"
    print(f"    {shown}: notice shown, Review it opened review on it")


def test_a_re_check_where_no_link_held_says_so_and_moves_nothing(page):
    """Re-check its links with no link holding reports that plainly and moves nothing. Server answer faked."""
    placement_path = os.path.join(PROJECT_DIR, "placement.json") if PROJECT_DIR else None
    original = open(placement_path, "rb").read() if placement_path else None
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        name, shown = page.evaluate("(i) => [S[i].name, displayName(S[i])]", scan_i)
        before = _pose(page, scan_i)
        result = {"no_link_held": name, "pairs": [{"source_name": name, "target_name": "other scan", "accepted": False,
                                                   "fitness": 0.01, "rmse_m": 0.05, "reason": "fitness below the gate"}]}
        page.route("**/api/jobs", lambda route: route.fulfill(json={"id": "fakejob"}) if route.request.method == "POST"
                   else route.continue_())
        page.route("**/api/jobs/fakejob", lambda route: route.fulfill(json={"id": "fakejob", "state": "done",
                                                                           "progress": None, "result": result}))
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        page.locator("#rvRecheck").click()
        page.wait_for_function("() => /held up, so nothing moved/.test($('out').value)", timeout=30_000)
        out = page.locator("#out").input_value()
        assert f"none of the 1 link(s) of {shown} held up" in out and "FAILED" not in out, out[-600:]
        assert _pose(page, scan_i) == before, "the scan moved"
        page.unroute("**/api/jobs"); page.unroute("**/api/jobs/fakejob")
        page.locator("#rvDone").click()
        print(f"    {shown}: 'none of the 1 link(s) ... held up, so nothing moved', the scan stayed put")
    finally:
        if placement_path:
            with open(placement_path, "wb") as fh:
                fh.write(original)


def test_detector_voting_is_ticked_by_default_and_can_be_unticked(page):
    """A fresh page has Detector voting ticked, and unticking it reaches the next job."""
    _goto(page)
    assert page.evaluate("() => $('cbDetectorVote').checked") is True, "Detector voting is not ticked on a fresh page"
    label = page.evaluate("() => document.querySelector('label[for=\"cbDetectorVote\"]').textContent")
    assert "experimental" not in label.lower(), f"the box still calls the vote experimental: {label!r}"
    page.evaluate("() => { $('cbDetectorVote').click(); }")
    assert page.evaluate("() => $('cbDetectorVote').checked") is False, "the box could not be unticked"
    print("    Detector voting is ticked on a fresh page, no longer labelled experimental, and can be unticked")


def test_undo_after_nudges(page):
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    # Move focus off the sidebar row onto something that is not an <input>,
    # matching the page's own guard (nudge keys are ignored while an <input>
    # has focus).
    _click_empty_props_area(page)
    before = _pose(page, scan_i)

    for _ in range(20):
        page.keyboard.press("ArrowRight")
    moved = _pose(page, scan_i)
    assert moved["x"] != before["x"], "20 presses of ArrowRight did not move the scan at all"

    # 20 nudges on the same scan with nothing in between coalesce into one undo entry.
    undo_btn = page.locator("#bUndo")
    assert not undo_btn.is_disabled(), "Undo is disabled right after 20 nudges"
    undo_btn.click()
    final = _pose(page, scan_i)
    assert abs(final["x"] - before["x"]) < 1e-9, (
        f"one Undo after 20 nudges left x={final['x']!r}, expected exactly the "
        f"starting {before['x']!r} (coalescing did not fold all 20 into one entry)"
    )
    assert undo_btn.is_disabled(), (
        "Undo is still enabled after the single coalesced entry was undone - "
        "something pushed more than one entry for this nudge session"
    )
    print(f"    20 nudges moved x by {moved['x']-before['x']:.4f}m; undo returned to exactly {before['x']:.4f}m")


def test_multiselect_move_same_delta(page):
    _goto(page)
    rows = page.locator("#scanlist .scan")
    assert rows.count() >= 2, "need at least 2 scans to test multi-select"
    i0 = int(rows.nth(0).get_attribute("data-i"))
    i1 = int(rows.nth(1).get_attribute("data-i"))
    rows.nth(0).click()
    rows.nth(1).click(modifiers=["Shift"])

    selected = page.evaluate("() => [...selection]")
    assert i0 in selected and i1 in selected, (
        f"shift-click did not select both rows: selection={selected}, expected {i0} and {i1}"
    )

    before0, before1 = _pose(page, i0), _pose(page, i1)
    _click_empty_props_area(page)
    page.keyboard.press("ArrowRight")
    after0, after1 = _pose(page, i0), _pose(page, i1)

    d0 = after0["x"] - before0["x"]
    d1 = after1["x"] - before1["x"]
    assert d0 != 0, "multi-selected scan 0 did not move"
    assert abs(d0 - d1) < 1e-9, f"the two selected scans moved by different amounts: {d0!r} vs {d1!r}"
    print(f"    both selected scans moved by exactly {d0:.4f}m")


# --------------------------------------------------------------------------
# One test per key binding, each asserting the specific change it makes.

def test_plain_arrows_move_xy(page):
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    _click_empty_props_area(page)
    step = _keyboard_step_m(page)
    before = _pose(page, scan_i)

    for _ in range(5):
        page.keyboard.press("ArrowRight")
    after_right = _pose(page, scan_i)
    assert abs((after_right["x"] - before["x"]) - 5 * step) < 1e-6, (
        f"5x ArrowRight moved x by {after_right['x']-before['x']!r}, expected exactly 5*{step!r}"
    )
    assert after_right["y"] == before["y"], "ArrowRight changed y too"

    for _ in range(3):
        page.keyboard.press("ArrowUp")
    after_up = _pose(page, scan_i)
    assert abs((after_up["y"] - after_right["y"]) - 3 * step) < 1e-6, (
        f"3x ArrowUp moved y by {after_up['y']-after_right['y']!r}, expected exactly 3*{step!r}"
    )
    print(f"    step={step:.4f}m: 5xRight -> dx={after_right['x']-before['x']:.4f}m, "
          f"3xUp -> dy={after_up['y']-after_right['y']:.4f}m")


def test_ctrl_arrows_yaw_and_height(page):
    """Yaw pivots on the cloud's 3D centre, so x/y change while the pivot's world position does not."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    _click_empty_props_area(page)
    step = _keyboard_step_m(page)
    rot_deg = _rot_step_deg(page)
    before = _pose(page, scan_i)
    centre_before = _world_centre(page, scan_i)

    for _ in range(3):
        page.keyboard.press("Control+ArrowRight")
    after_yaw = _pose(page, scan_i)
    got_deg = (after_yaw["yaw"] - before["yaw"]) * 180 / 3.141592653589793
    assert abs(got_deg - 3 * rot_deg) < 1e-6, (
        f"3x Ctrl+ArrowRight turned yaw by {got_deg!r} deg, expected exactly 3*{rot_deg!r}"
    )
    centre_after_yaw = _world_centre(page, scan_i)
    for a, b in zip(centre_before, centre_after_yaw):
        assert abs(a - b) < 1e-6, (
            f"Ctrl+ArrowRight (yaw) moved the pivot: {centre_before} -> {centre_after_yaw}"
        )

    for _ in range(2):
        page.keyboard.press("Control+ArrowUp")
    after_z = _pose(page, scan_i)
    assert abs((after_z["dz"] - after_yaw["dz"]) - 2 * step) < 1e-6, (
        f"2x Ctrl+ArrowUp moved height by {after_z['dz']-after_yaw['dz']!r}, expected exactly 2*{step!r}"
    )

    for _ in range(4):
        page.keyboard.press("Control+ArrowLeft")
    after_yaw2 = _pose(page, scan_i)
    got_deg2 = (after_yaw2["yaw"] - after_z["yaw"]) * 180 / 3.141592653589793
    assert abs(got_deg2 - (-4 * rot_deg)) < 1e-6, (
        f"4x Ctrl+ArrowLeft turned yaw by {got_deg2!r} deg, expected exactly -4*{rot_deg!r}"
    )
    print(f"    turn step={rot_deg}deg, keyboard step={step:.4f}m: "
          f"3xCtrl+Right -> {got_deg:.3f}deg, 2xCtrl+Up -> dz={after_z['dz']-after_yaw['dz']:.4f}m, "
          f"4xCtrl+Left -> {got_deg2:.3f}deg")


def test_alt_arrows_tilt_pitch_and_roll(page):
    """Pitch and roll by key press. They pivot on the cloud's 3D centre, so x/y/dz change while the pivot's world
    position does not."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    _click_empty_props_area(page)
    rot_deg = _rot_step_deg(page)
    before = _pose(page, scan_i)
    centre_before = _world_centre(page, scan_i)

    for _ in range(4):
        page.keyboard.press("Alt+ArrowUp")
    after_pitch = _pose(page, scan_i)
    got_pitch_deg = (after_pitch["pitch"] - before["pitch"]) * 180 / 3.141592653589793
    assert abs(got_pitch_deg - 4 * rot_deg) < 1e-6, (
        f"4x Alt+ArrowUp changed pitch by {got_pitch_deg!r} deg, expected exactly 4*{rot_deg!r}"
    )
    assert after_pitch["roll"] == before["roll"], "Alt+ArrowUp (pitch) changed roll too"
    assert after_pitch["yaw"] == before["yaw"], "Alt+ArrowUp (pitch) changed yaw too"
    centre_after_pitch = _world_centre(page, scan_i)
    for a, b in zip(centre_before, centre_after_pitch):
        assert abs(a - b) < 1e-6, (
            f"Alt+ArrowUp (pitch) moved the pivot: {centre_before} -> {centre_after_pitch}"
        )

    for _ in range(2):
        page.keyboard.press("Alt+ArrowRight")
    after_roll = _pose(page, scan_i)
    got_roll_deg = (after_roll["roll"] - after_pitch["roll"]) * 180 / 3.141592653589793
    assert abs(got_roll_deg - 2 * rot_deg) < 1e-6, (
        f"2x Alt+ArrowRight changed roll by {got_roll_deg!r} deg, expected exactly 2*{rot_deg!r}"
    )
    assert after_roll["pitch"] == after_pitch["pitch"], "Alt+ArrowRight (roll) changed pitch too"
    centre_after_roll = _world_centre(page, scan_i)
    for a, b in zip(centre_before, centre_after_roll):
        assert abs(a - b) < 1e-6, (
            f"Alt+ArrowRight (roll) moved the pivot: {centre_before} -> {centre_after_roll}"
        )
    print(f"    turn step={rot_deg}deg: 4xAlt+Up -> pitch={got_pitch_deg:.3f}deg, "
          f"2xAlt+Right -> roll={got_roll_deg:.3f}deg, pivot held at {centre_before}")


def test_a_and_s_cycle_scan_list_not_wasd_move(page):
    """A/S cycle the scan list, and W and D are unbound: checks both, so an old binding still live underneath would be
    caught."""
    _goto(page)
    _select_scan_row(page, 0)
    _click_empty_props_area(page)
    n = len(page.locator("#scanlist .scan").all())
    assert n >= 2, "need at least 2 scans to test list cycling"

    start_i = page.evaluate("() => sel")
    page.keyboard.press("s")
    after_s = page.evaluate("() => sel")
    assert after_s != start_i, "'s' did not change the selection at all"

    page.keyboard.press("a")
    after_a = page.evaluate("() => sel")
    assert after_a == start_i, (
        f"'a' should undo the 's' move (previous in the list); started at {start_i}, "
        f"after s+a landed at {after_a}"
    )

    # The old scheme: W/D moved x/y. They must do NOTHING now.
    scan_i = page.evaluate("() => sel")
    before = _pose(page, scan_i)
    page.keyboard.press("w")
    page.keyboard.press("d")
    after = _pose(page, scan_i)
    assert after == before, (
        f"W/D moved the scan ({before} -> {after}) - they must be fully unbound"
    )
    print(f"    s: {start_i}->{after_s}, a: {after_s}->{after_a} (back to start), w/d confirmed no-ops")


def test_numpad_5_jumps_to_plan_view(page):
    """Numpad 5 lands on plan, and a cardinal key still lands on side."""
    _goto(page)
    page.locator('[data-view="iso"]').click()
    assert page.evaluate("() => view") == "iso", "setup: could not start from a non-plan view"

    page.keyboard.press("Numpad5")
    page.wait_for_function("() => view === 'plan'", timeout=5000)
    assert page.evaluate("() => view") == "plan", "Numpad5 did not switch to the plan view"
    on_plan = page.locator('[data-view="plan"]').evaluate("el => el.classList.contains('on')")
    assert on_plan, "Plan button is not highlighted after Numpad5"

    page.locator('[data-view="iso"]').click()
    page.keyboard.press("Numpad8")
    page.wait_for_function("() => view === 'side'", timeout=5000)
    assert page.evaluate("() => view") == "side", "Numpad8 (a cardinal key) must still land on side"
    print("    Numpad5 -> plan, Numpad8 -> side, both confirmed via the live view state and button highlight")


def test_letter_hotkeys_match_their_buttons(page):
    """Tool and toggle hotkeys: N/M/R/L/E switch tools, W/H/G/C/T/K toggle their checkbox, X/Y/Z toggle that axis's
    clip and V all enabled axes. Each entry clicks the real control. A/S scan-list cycling still works."""
    _goto(page)
    # Hovering a control names its shortcut.
    missing = page.evaluate("""() => Object.entries(HOTKEYS).filter(([k, v]) => v.el).filter(([k, v]) => {
        const el = document.querySelector(v.el), host = el && (el.closest('[data-setting]') || el);
        return !host || !(host.title || '').includes(`(shortcut: ${k.toUpperCase()})`); }).map(([k]) => k)""")
    assert not missing, f"no shortcut in the hover text of the control for: {missing}"
    for key, expect_tool in (("m", "move"), ("r", "spin"), ("l", "anchor"), ("e", "exclude"), ("n", "navigate")):
        page.keyboard.press(key)
        got = page.evaluate("() => tool")
        assert got == expect_tool, f"{key!r} should select tool {expect_tool!r}, got {got!r}"

    view_before = page.evaluate("() => view")
    for key in ("x", "y", "z"):
        before = page.evaluate(f"() => clip.{key}.on")
        page.keyboard.press(key)
        after = page.evaluate(f"() => clip.{key}.on")
        assert after != before, f"{key!r} did not toggle clip.{key}.on"
        page.keyboard.press(key)
        assert page.evaluate(f"() => clip.{key}.on") == before, f"{key!r} did not toggle clip.{key}.on back"
    assert page.evaluate("() => view") == view_before, "a clip-axis hotkey must never change the camera view"

    page.evaluate("() => { clip.z.on = true; }")
    master_before = page.evaluate("() => clipMasterOn")
    page.keyboard.press("v")
    assert page.evaluate("() => clipMasterOn") != master_before, "'v' did not toggle clipMasterOn"
    page.keyboard.press("v")
    assert page.evaluate("() => clipMasterOn") == master_before, "'v' did not toggle clipMasterOn back"
    page.evaluate("() => { clip.z.on = false; }")

    before = page.evaluate("() => $('cbWall').checked")
    page.keyboard.press("w")
    after = page.evaluate("() => $('cbWall').checked")
    assert after != before, "'w' did not toggle the walls checkbox"
    page.keyboard.press("w")
    assert page.evaluate("() => $('cbWall').checked") == before, "'w' did not toggle back"

    # The reassignment must not have broken A/S list-cycling - the exact
    # feature that forced Z/L to be picked instead of S/A in the first place.
    _select_scan_row(page, 0)
    _click_empty_props_area(page)
    start_i = page.evaluate("() => sel")
    page.keyboard.press("s")
    after_s = page.evaluate("() => sel")
    assert after_s != start_i, "'s' must still cycle the scan list, unaffected by the Z reassignment"
    page.keyboard.press("a")
    assert page.evaluate("() => sel") == start_i, "'a' must still cycle back, unaffected by the L reassignment"

    help_text = page.locator("#hotkeyHelp").inner_text()
    for letter in "NMRLEXYZVHGCTKW":
        assert letter in help_text, f"generated Help panel does not mention {letter}: {help_text!r}"
    print("    n/m/r/l/e/x/y/z/v/w confirmed live; a/s list-cycling unaffected; Help panel lists all 15 keys")


# --------------------------------------------------------------------------


def _points_bbox_in_band(page, scan_i, target_h, band):
    """Screen box of the points of scan scan_i inside a height band, via xyAt(s, k). Fails if any corner is not finite,
    rather than passing a NaN box to the mouse API."""
    box = page.evaluate(
        """(args) => {
            const s = S.find(x => x.i === args.i);
            const idxs = [];
            for (let k = 0; k < s.h.length; k++) {
                if (Math.abs(s.h[k] - args.h) <= args.band) idxs.push(k);
            }
            if (!idxs.length) return null;
            const sample = idxs.length > 300
                ? idxs.filter((_, i) => i % Math.ceil(idxs.length / 300) === 0) : idxs;
            let x0=1e9, y0=1e9, x1=-1e9, y1=-1e9;
            for (const k of sample) {
                const g = place(s, xyAt(s, k));
                const p = proj(g[0], g[1], s.h[k] + s.dz);
                x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]);
                x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
            }
            return {x0, y0, x1, y1, n: sample.length};
        }""",
        {"i": scan_i, "h": target_h, "band": band},
    )
    # None (nothing in this band) is legitimate and checked by the caller; a non-finite box is not.
    if box is not None:
        _assert_finite(f"_points_bbox_in_band(scan_i={scan_i})",
                       x0=box["x0"], y0=box["y0"], x1=box["x1"], y1=box["y1"])
    return box


def _set_clip(page, axis, lo_cm, hi_cm, on=True):
    """Set one clip band on the shared `clip` state, then redraw."""
    page.evaluate(
        "(a) => { clip[a.ax].lo = a.lo; clip[a.ax].hi = a.hi; clip[a.ax].on = a.on; draw(); }",
        {"ax": axis, "lo": int(lo_cm), "hi": int(hi_cm), "on": on},
    )


def test_clip_range_reaches_below_zero_on_the_real_venue(page):
    """The Z clip range comes from the real scans and the Z slider uses the same bounds."""
    _goto(page)
    page.wait_for_function("() => clip.z.rangeMin != null && clip.z.rangeMax != null", timeout=LOAD_TIMEOUT_MS)
    lo, hi = page.evaluate("() => [clip.z.rangeMin, clip.z.rangeMax]")
    assert lo < 0 < hi, f"expected the real venue's Z range to straddle zero, got [{lo}, {hi}] (cm)"
    page.wait_for_function(
        "() => document.querySelectorAll('#rc-clip-z-root [role=slider]').length === 2", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_timeout(150)  # one frame for the slider to read the real bounds
    mins = page.evaluate(
        "() => [...document.querySelectorAll('#rc-clip-z-root [role=slider]')].map(e => +e.getAttribute('aria-valuemin'))")
    maxs = page.evaluate(
        "() => [...document.querySelectorAll('#rc-clip-z-root [role=slider]')].map(e => +e.getAttribute('aria-valuemax'))")
    assert mins == [lo, lo] and maxs == [hi, hi], f"Z slider bounds {mins}/{maxs} disagree with clip.z range [{lo}, {hi}]"
    print(f"    clip range [{lo/100:.2f}, {hi/100:.2f}] m, and the Z slider's own bounds match it")


def test_clip_vertical_thumbs_cannot_cross(page):
    """Dragging the lower thumb past the upper one clamps instead of crossing."""
    _goto(page)
    page.wait_for_function(
        "() => document.querySelectorAll('#rc-clip-y-root [role=slider]').length === 2", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function("() => clip.y.rangeMin != null", timeout=LOAD_TIMEOUT_MS)
    lo_min, hi_max = page.evaluate("() => [clip.y.rangeMin, clip.y.rangeMax]")
    mid = (lo_min + hi_max) // 2
    _set_clip(page, "y", mid - 600, mid + 600, on=True)
    page.wait_for_timeout(150)  # the slider re-reads the state on the next frame

    thumbs = page.locator("#rc-clip-y-root [role=slider]")
    track_box = thumbs.first.locator("xpath=..").bounding_box()
    lower = thumbs.first.bounding_box()
    assert lower and track_box, "could not find Y's own slider thumbs"
    cx = lower["x"] + lower["width"] / 2
    page.mouse.move(cx, lower["y"] + lower["height"] / 2)
    page.mouse.down()
    page.mouse.move(cx, track_box["y"] + 2, steps=10)  # all the way to the TOP, far past the upper thumb
    page.mouse.up()
    lo, hi = page.evaluate("() => [clip.y.lo, clip.y.hi]")
    assert lo <= hi, f"dragging the lower edge past the upper one crossed the band: lo={lo} hi={hi}"
    assert hi == mid + 600, f"the upper edge moved when only the lower one was dragged: hi={hi}"
    page.evaluate("() => { clip.y.on = false; }")
    print(f"    lower edge dragged to the top clamped at the upper edge: lo={lo} hi={hi}")


def test_clip_sliders_are_long_keep_a_thin_slice_and_step_like_a_ct_scan(page):
    """Clip slice controls."""
    _goto(page)
    page.wait_for_function("() => clip.y.rangeMin != null && clip.x.rangeMin != null", timeout=LOAD_TIMEOUT_MS)
    stage = page.locator("#stage").bounding_box()
    for axis, edge in (("x", stage["width"]), ("y", stage["height"]), ("z", stage["height"])):
        track = page.locator(f"#rc-clip-{axis}-root [role=slider]").first.locator("xpath=..").bounding_box()
        along = track["width"] if axis == "x" else track["height"]
        assert along >= 0.5 * edge, f"the {axis.upper()} slider is {along:.0f} px along a {edge:.0f} px edge"

    lo_min, hi_max = page.evaluate("() => [clip.y.rangeMin, clip.y.rangeMax]")
    mid = (lo_min + hi_max) // 2
    _set_clip(page, "y", mid - 300, mid + 300, on=True)
    page.wait_for_timeout(150)
    thumbs = page.locator("#rc-clip-y-root [role=slider]")
    upper = thumbs.nth(1).bounding_box()
    cx = upper["x"] + upper["width"] / 2
    page.mouse.move(cx, upper["y"] + upper["height"] / 2)
    page.mouse.down()
    page.mouse.move(cx, thumbs.first.bounding_box()["y"] + 40, steps=10)   # the upper end dragged far below the lower
    page.mouse.up()
    lo, hi = page.evaluate("() => [clip.y.lo, clip.y.hi]")
    assert hi - lo == 5, f"the ends came {hi - lo} cm apart, the thinnest slice is 5 cm"

    _set_clip(page, "y", mid - 100, mid + 100, on=True)
    # The slider reads the slice once a frame; on a busy test machine that took 600 ms, so wait for it, not a fixed time.
    page.wait_for_function("() => document.querySelector('#rc-clip-y-root .clipBand').getBoundingClientRect().height > 10",
                           timeout=10_000)
    band = page.locator("#rc-clip-y-root .clipBand").bounding_box()
    bx, by = band["x"] + band["width"] / 2, band["y"] + band["height"] / 2
    page.mouse.move(bx, by)
    page.mouse.down()
    page.mouse.move(bx, by - 30, steps=6)
    page.mouse.up()
    lo2, hi2 = page.evaluate("() => [clip.y.lo, clip.y.hi]")
    assert hi2 - lo2 == 200 and lo2 > mid - 100, f"dragging the band did not move the slice whole: {lo2}..{hi2}"

    scan_i = _select_scan_row(page, 0)
    before = _pose(page, scan_i)
    page.locator("#rc-clip-y-root [role=slider]").first.locator("xpath=..").focus()
    page.keyboard.press("ArrowUp")
    lo3, hi3 = page.evaluate("() => [clip.y.lo, clip.y.hi]")
    assert (lo3, hi3) == (lo2 + 200, hi2 + 200), f"one arrow press should step one slice: {lo2}..{hi2} -> {lo3}..{hi3}"
    assert _pose(page, scan_i) == before, "the arrow key also moved the selected scan"
    page.evaluate("() => { clip.y.on = false; draw(); }")
    print(f"    sliders >= half their edge; thinnest slice 5 cm; band drag {lo2}..{hi2}; one arrow step {lo3}..{hi3}")


def test_clip_bounded_region_works_outside_plan_view(page):
    """A Z clip band bounds a link region drawn in Iso view."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    page.locator('[data-view="iso"]').click()
    bands = page.evaluate("(i) => S.find(x => x.i === i).bands", scan_i)
    assert bands, "selected scan reports no floor-height bands to build a clip test around"
    target_h = max(bands, key=lambda b: b["n"])["h"]
    band = 0.15  # metres either side - matches the clip band set below

    # Must be computed after switching to Iso: proj() depends on the camera.
    box = _points_bbox_in_band(page, scan_i, target_h, band)
    assert box and box["n"] >= 4, (
        f"could not find enough of the selected scan's own points within "
        f"{band}m of height {target_h}m to draw a region ({box})"
    )

    page.locator('[data-tool="anchor"]').click()
    _set_clip(page, "z", round((target_h - band) * 100), round((target_h + band) * 100), on=True)
    assert page.evaluate("() => clipAxisActive('z')"), "Z clip did not turn on"
    assert page.evaluate("() => view") == "iso", "clipping must not have changed the camera view"

    rect = _canvas_rect(page)
    x0, y0 = _to_viewport(rect, (box["x0"] - 6, box["y0"] - 6))
    x1, y1 = _to_viewport(rect, (box["x1"] + 6, box["y1"] + 6))
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=6)
    page.mouse.up()

    status = page.locator("#out").input_value()
    m = re.search(
        r"clip-bounded: z ([\-\d.]+) to ([\-\d.]+) m of the hits; active band\(s\): "
        r"Z ([\-\d.]+) to ([\-\d.]+) m", status)
    assert m, f"no clip-bounded region reported in the status log (still in Iso view): {status!r}"
    z_min, z_max, band_lo, band_hi = (float(v) for v in m.groups())
    eps = 1e-6
    assert band_lo - eps <= z_min <= band_hi + eps, f"collected z_min {z_min} outside band [{band_lo}, {band_hi}]"
    assert band_lo - eps <= z_max <= band_hi + eps, f"collected z_max {z_max} outside band [{band_lo}, {band_hi}]"
    print(f"    Iso view, Z band [{band_lo:.3f}, {band_hi:.3f}] m, collected z in "
          f"[{z_min:.3f}, {z_max:.3f}] m - clipping held outside Plan")

    if page.locator("#bCancelAnchor").count():
        page.locator("#bCancelAnchor").click()
    page.evaluate("() => { clip.z.on = false; }")
    page.locator('[data-view="plan"]').click()


def test_clip_bounded_hits_report_world_height_even_when_the_scan_is_lifted(page):
    """The z range printed for a clip-bounded region is in the same world frame as the band beside it, even after the
    scan has been lifted."""
    _goto(page)
    res = page.evaluate(r"""() => {
        const s = S.find(q => !q.deleted && q.xy && q.xy.length);
        s.dz = 1.25;                       // lift the scan; its points keep their local heights
        clip.z.on = true; clip.z.lo = 175; clip.z.hi = 225; draw();   // world 1.75 to 2.25 m = local 0.5 to 1.0 m
        const h = collectBoxHits(s, {x0: 0, y0: 0, x1: cv.width, y1: cv.height});
        return h ? {zmin: h.hitZMin, zmax: h.hitZMax, n: h.n} : null;
    }""")
    assert res, "no points collected from a whole-canvas box inside the band, so the check proved nothing"
    assert 1.75 - 1e-6 <= res["zmin"] and res["zmax"] <= 2.25 + 1e-6, (
        f"reported hit heights [{res['zmin']:.3f}, {res['zmax']:.3f}] m are not inside the world band [1.75, 2.25] "
        f"they were collected under: a local height is being compared to a world band")
    print(f"    {res['n']} hits of a scan lifted 1.25 m, reported z [{res['zmin']:.3f}, {res['zmax']:.3f}] m, inside world band [1.75, 2.25]")


def _shader_world(lx, ly, h, pose):
    """The point shader's placement (turn, roll, pitch, shift, lift), redone in Python so the test does not trust the
    page's copy."""
    import math
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    rx, ry = lx * c - ly * s, lx * s + ly * c
    rc, rs = math.cos(pose["roll"]), math.sin(pose["roll"])
    ry2, h1 = ry * rc - h * rs, ry * rs + h * rc
    pc, ps = math.cos(pose["pitch"]), math.sin(pose["pitch"])
    return (rx * pc + h1 * ps + pose["x"], ry2 + pose["y"], -rx * ps + h1 * pc + pose["dz"])


def test_markers_and_region_boxes_follow_a_tilted_scan(page):
    """Link and exclusion markers sit on their points and a dragged box picks the points drawn under it, with the scan
    turned, tilted and lifted; each marker hides with its own point."""
    _goto(page)
    page.locator('[data-view="side"]').click()
    res = page.evaluate(r"""() => {
        const s = S.find(q => !q.deleted && q.ready && q.xy && nPoints(q) > 1000);
        window.__markerTestKeep = [s, {x: s.x, y: s.y, yaw: s.yaw, dz: s.dz, pitch: s.pitch, roll: s.roll}];
        s.x = 3.0; s.y = -2.0; s.yaw = 0.7; s.dz = 1.06; s.pitch = 2 * Math.PI / 180; s.roll = 3 * Math.PI / 180;
        draw();
        const n = nPoints(s), ks = [0.1, 0.3, 0.5, 0.7, 0.9].map(f => Math.floor(f * n));
        const pts = ks.map(k => { const raw = rawFromLeveled(s, xyAt(s, k), s.h[k]);
            const w = anchorWorldPt(s, raw); return {lx: s.xy[k*2], ly: s.xy[k*2+1], h: s.h[k], raw, w: [w.xy[0], w.xy[1], w.z]}; });
        return {name: s.name, pose: {x: s.x, y: s.y, yaw: s.yaw, dz: s.dz, pitch: s.pitch, roll: s.roll}, pts};
    }""")
    pose, worst_old = res["pose"], 0.0
    for p in res["pts"]:
        want = _shader_world(p["lx"], p["ly"], p["h"], pose)
        got = p["w"]
        assert max(abs(a - b) for a, b in zip(want, got)) < 1e-4, f"marker at {got}, its point is drawn at {want}"
        untilted_z = p["h"] + pose["dz"]   # what the old place() + height offset gave
        worst_old = max(worst_old, abs(untilted_z - want[2]))
    assert worst_old > 0.05, f"test proves nothing: the old untilted height was only {worst_old:.3f} m off"

    # A small box around one point's drawn position must collect only points drawn inside it.
    probe = res["pts"][2]
    box = page.evaluate(r"""(w) => { const p = proj(w[0], w[1], w[2]);
        return {x0: p[0] - 4, y0: p[1] - 4, x1: p[0] + 4, y1: p[1] + 4}; }""", probe["w"])
    hits = page.evaluate(r"""([name, box]) => { const s = S.find(q => q.name === name);
        const h = collectBoxHits(s, box); if (!h) return null;
        return h.rawHits.map(r => { const lv = leveledFromRaw(s, r); return [lv.xy[0], lv.xy[1], lv.h]; }); }""",
                         [res["name"], box])
    assert hits, "no points collected around a point that is drawn there"
    screen = page.evaluate(r"""(ws) => ws.map(w => proj(w[0], w[1], w[2]))""",
                           [list(_shader_world(x, y, h, pose)) for x, y, h in hits])
    outside = [sp for sp in screen if not (box["x0"] - 0.5 <= sp[0] <= box["x1"] + 0.5 and box["y0"] - 0.5 <= sp[1] <= box["y1"] + 0.5)]
    assert not outside, f"{len(outside)} of {len(hits)} collected points are drawn outside the box, e.g. {outside[:2]}"

    # Each marker circle hides with its own point: a Z band keeping only some of them.
    zs = sorted(p["w"][2] for p in res["pts"])
    lo_cm, hi_cm = round((zs[0] - 0.001) * 100, 3), round((zs[2] + 0.001) * 100, 3)
    counts = page.evaluate(r"""([name, pts, lo, hi]) => { const s = S.find(q => q.name === name);
        const keepEx = exclusions;
        exclusions = [{id: 'exclude-test', scan: name, points: pts, centroid: pts[0], extent: [0,0,0], n_points: pts.length}];
        let n = 0; const arc = cx.arc; cx.arc = function(...a) { n++; return arc.apply(this, a); };
        const count = () => { n = 0; drawExclusions(); return n; };
        const master = clipMasterOn; clipMasterOn = true;
        clip.z.on = false; const all = count();
        clip.z.on = true; clip.z.lo = lo; clip.z.hi = hi; const some = count();
        clip.z.lo = -99900; clip.z.hi = -99800; const none = count();
        cx.arc = arc; clip.z.on = false; clipMasterOn = master; exclusions = keepEx;
        const [ks, kp] = window.__markerTestKeep; Object.assign(ks, kp); draw(); return {all, some, none}; }""",
                           [res["name"], [p["raw"] for p in res["pts"]], lo_cm, hi_cm])
    # Each draw is one circle per visible point plus the centre dot.
    assert counts == {"all": 6, "some": 4, "none": 0}, counts
    print(f"    {res['name']} turned, tilted 2/3 deg, lifted 1.06 m: markers on their points (old height up to "
          f"{worst_old*100:.0f} cm off), {len(hits)} box hits all drawn in the box, circles {counts}")


def test_link_and_exclusion_markers_hide_with_x_y_slices_and_a_hidden_scan(page):
    """Every marker circle vanishes with its point under the X and Y slices too, and all vanish when their scan is
    hidden."""
    _goto(page)
    got = page.evaluate(r"""() => {
        const s = S.find(q => !q.deleted && q.ready && q.xy && nPoints(q) > 1000);
        const n = nPoints(s), ks = [0.1, 0.3, 0.5, 0.7, 0.9].map(f => Math.floor(f * n));
        const raws = ks.map(k => rawFromLeveled(s, xyAt(s, k), s.h[k]));
        const ws = raws.map(r => anchorWorldPt(s, r));
        const keep = [anchors, exclusions, pendingAnchor, clipMasterOn, JSON.stringify(clip), s.vis];
        const other = S.find(q => q !== s && !q.deleted && q.ready);
        const member = (scan, pts) => ({scan, points: pts, centroid: pts[0], extent: [0, 0, 0], n_points: pts.length});
        let circles = 0; const arc = cx.arc; cx.arc = function(...a) { circles++; return arc.apply(this, a); };
        const count = (fn) => { circles = 0; fn(); return circles; };
        const out = {};
        const sorted = (axis) => ws.map(w => axis === 'z' ? w.z : w.xy[axis === 'x' ? 0 : 1]).sort((a, b) => a - b);
        for (const which of ['links', 'exclusions']) {
            anchors = []; exclusions = []; pendingAnchor = null;
            if (which === 'links') anchors = [{id: 'a-test', label: '', members: [member(s.name, raws), member(other.name, [rawFromLeveled(other, xyAt(other, 0), other.h[0])])]}];
            else exclusions = [{id: 'x-test', ...member(s.name, raws)}];
            const draw1 = which === 'links' ? () => drawAnchors() : () => drawExclusions();
            clipMasterOn = true; for (const a of ['x', 'y', 'z']) clip[a].on = false;
            const res = {all: count(draw1)};
            for (const a of ['x', 'y']) {
                const v = sorted(a);
                clip[a].on = true; clip[a].lo = Math.round((v[0] - 0.001) * 100); clip[a].hi = Math.round((v[2] + 0.001) * 100);
                if (clip[a].hi <= clip[a].lo) clip[a].hi = clip[a].lo + 1;
                res[a + '_some'] = count(draw1);
                clip[a].lo = -99900; clip[a].hi = -99800; res[a + '_none'] = count(draw1);
                clip[a].on = false;
            }
            s.vis = false; res.hidden = count(draw1); s.vis = keep[5];
            out[which] = res;
        }
        cx.arc = arc; [anchors, exclusions, pendingAnchor, clipMasterOn] = keep.slice(0, 4);
        Object.assign(clip, JSON.parse(keep[4])); draw();
        return out;
    }""")
    for which, r in got.items():
        assert r["all"] >= 6, (which, r)
        for a in ("x", "y"):
            assert r[f"{a}_none"] < r["all"] and r[f"{a}_some"] < r["all"], (which, a, r)
        assert r["x_none"] <= (2 if which == "links" else 0) and r["y_none"] <= (2 if which == "links" else 0), (which, r)
        assert r["hidden"] <= (2 if which == "links" else 0), (which, r)
    print(f"    circles drawn per case: {got}")


def test_run_result_says_each_height_change_in_plain_words(page):
    """What a run changed, per scan, height first."""
    _goto(page)
    res = page.evaluate(r"""() => {
        const [a, b] = S.filter(q => !q.deleted && q.ready).slice(0, 2);
        const keep = S.map(s => [s, s.dz, s.yaw]);
        const none = describePoseChanges(snapshotPoses());
        const before = snapshotPoses();
        a.dz += 0.35; b.dz -= 0.08;
        const moved = describePoseChanges(before);
        keep.forEach(([s, dz, yaw]) => { s.dz = dz; s.yaw = yaw; }); draw();
        return {none, moved, a: a.name, b: b.name};
    }""")
    assert res["none"] == ["what this run changed: nothing, every scan is where it was."], res["none"]
    assert res["moved"][1] == f"  {res['a']}: moved up 35 cm", res["moved"]
    assert res["moved"][2] == f"  {res['b']}: moved down 8 cm", res["moved"]
    assert len(res["moved"]) == 3, res["moved"]
    print(f"    {res['moved']}")


def test_clip_state_survives_a_reload(page):
    """Save, reload: clip bands and on-states come back, and the edge checkboxes show them."""
    _goto(page)
    page.wait_for_function("() => clip.x.rangeMin != null && clip.z.rangeMin != null", timeout=LOAD_TIMEOUT_MS)
    xr, zr = page.evaluate("() => [clip.x.rangeMin, clip.z.rangeMin]")
    x_lo, x_hi, z_lo, z_hi = xr + 100, xr + 400, zr + 20, zr + 150
    _set_clip(page, "x", x_lo, x_hi, on=True)
    _set_clip(page, "z", z_lo, z_hi, on=True)
    before = page.evaluate("() => JSON.parse(JSON.stringify(clip))")

    page.locator("#bSave").click()
    page.wait_for_function("() => $('out').value.includes('saved to')", timeout=10_000)
    page.reload(wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        """() => {
            const rows = document.querySelectorAll('#scanlist .scan');
            if (!rows.length) return false;
            return [...rows].every(r => !r.textContent.includes('building preview'));
        }""",
        timeout=LOAD_TIMEOUT_MS,
    )

    after = page.evaluate("() => ({x: {on: clip.x.on, lo: clip.x.lo, hi: clip.x.hi}, "
                           "z: {on: clip.z.on, lo: clip.z.lo, hi: clip.z.hi}})")
    assert after["x"] == {"on": True, "lo": x_lo, "hi": x_hi}, f"X clip state did not survive reload: {after['x']}"
    assert after["z"] == {"on": True, "lo": z_lo, "hi": z_hi}, f"Z clip state did not survive reload: {after['z']}"
    page.wait_for_timeout(150)  # one frame for the sliders to re-read the restored state
    assert page.locator("#rc-clip-x-root input[type=checkbox]").is_checked(), "X's edge checkbox did not show the restored on-state"
    assert page.locator("#rc-clip-z-root input[type=checkbox]").is_checked(), "Z's edge checkbox did not show the restored on-state"
    print(f"    clip state before {before['x']}/{before['z']} matches after reload {after['x']}/{after['z']}")

    page.evaluate("() => { clip.x.on = false; clip.z.on = false; }")
    page.locator("#bSave").click()
    page.wait_for_function("() => $('out').value.includes('saved to')", timeout=10_000)


def test_clip_all_three_axes_together_form_a_box(page):
    """X, Y and Z bands together clip to a box, and a link region collects only inside it."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    page.wait_for_function("() => clip.x.rangeMin != null && clip.y.rangeMin != null && clip.z.rangeMin != null",
                           timeout=LOAD_TIMEOUT_MS)

    bounds = {}
    for ax in ("x", "y", "z"):
        lo, hi = page.evaluate(
            "(args) => { const s = S.find(x => x.i === args.i); return scanWorldExtent(s, args.ax); }",
            {"i": scan_i, "ax": ax},
        )
        mid, half = (lo + hi) / 2, (hi - lo) * 0.2  # middle 40% of the real extent
        bounds[ax] = (mid - half, mid + half)
        _set_clip(page, ax, round((mid - half) * 100), round((mid + half) * 100), on=True)
        assert page.evaluate(f"() => clipAxisActive('{ax}')"), f"{ax} clip did not turn on"

    # Ground truth uses the stored whole-cm bounds, not the unrounded floats above.
    check = page.evaluate(
        """(args) => {
            const s = S.find(x => x.i === args.i);
            const bx=[clip.x.lo/100, clip.x.hi/100], by=[clip.y.lo/100, clip.y.hi/100], bz=[clip.z.lo/100, clip.z.hi/100];
            let mismatches = 0, insideCount = 0;
            for (let k = 0; k < s.h.length; k++) {
                const g = place(s, xyAt(s, k));
                const worldZ = s.h[k] + s.dz;
                const shouldBeInside = g[0] >= bx[0] && g[0] <= bx[1]
                    && g[1] >= by[0] && g[1] <= by[1]
                    && worldZ >= bz[0] && worldZ <= bz[1];
                const clipped = isClipped(g, worldZ);
                if (shouldBeInside === clipped) mismatches++;
                if (!clipped) insideCount++;
            }
            return {mismatches, insideCount, total: s.h.length};
        }""",
        {"i": scan_i},
    )
    assert check["mismatches"] == 0, (
        f"isClipped disagreed with the independently-computed box on {check['mismatches']} point(s)"
    )
    assert 0 < check["insideCount"] < check["total"], (
        f"box is degenerate - {check['insideCount']} of {check['total']} points inside, expected a real subset"
    )

    page.locator('[data-tool="anchor"]').click()
    rect = page.evaluate("() => { const r = cv.getBoundingClientRect(); return {x:r.x, y:r.y, w:r.width, h:r.height}; }")
    x0, y0 = rect["x"] + 5, rect["y"] + 5
    x1, y1 = rect["x"] + rect["w"] - 5, rect["y"] + rect["h"] - 5
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=6)
    page.mouse.up()

    status = page.locator("#out").input_value()
    m = re.search(r"added .+? \((\d+) points,", status)
    assert m, f"no anchor region reported in the status log: {status!r}"
    n_collected = int(m.group(1))
    assert 0 < n_collected <= check["insideCount"], (
        f"anchor collected {n_collected} points, expected 1 to {check['insideCount']} (the box's own real subset)"
    )
    assert "X " in status and "Y " in status and "Z " in status, (
        f"status line did not name all three active axes: {status!r}"
    )
    print(f"    box [{bounds['x']}, {bounds['y']}, {bounds['z']}] m: {check['insideCount']} of {check['total']} "
          f"points inside per isClipped, anchor collected {n_collected} through the real UI path")

    if page.locator("#bCancelAnchor").count():
        page.locator("#bCancelAnchor").click()
    page.evaluate("() => { clip.x.on = false; clip.y.on = false; clip.z.on = false; }")


def _drag_finish_anchor_region(page, scan_i):
    """Real mouse drag over one of `scan_i`'s own rendered points, in plan
    view (no clip filtering), used to build up a 2-member anchor below."""
    pt = _scan_point_screen_pos(page, scan_i)
    rect = _canvas_rect(page)
    vx, vy = _to_viewport(rect, (pt["x"], pt["y"]))
    page.mouse.move(vx - 10, vy - 10)
    page.mouse.down()
    page.mouse.move(vx + 10, vy + 10, steps=4)
    page.mouse.up()


def test_clear_links_zero_anchors(page):
    _goto(page)
    rows = page.locator("#scanlist .scan")
    assert rows.count() >= 2, "need at least 2 scans to build an anchor"
    i0 = int(rows.nth(0).get_attribute("data-i"))
    i1 = int(rows.nth(1).get_attribute("data-i"))

    page.locator('[data-view="plan"]').click()
    page.locator('[data-tool="anchor"]').click()
    rows.nth(0).click()
    _drag_finish_anchor_region(page, i0)
    rows.nth(1).click()
    _drag_finish_anchor_region(page, i1)
    page.locator("#bFinishAnchor").click()

    n_anchors = page.evaluate("() => anchors.length")
    assert n_anchors >= 1, "building a 2-member anchor did not add it to anchors[]"

    # Save first: Clear all links can only back up what was already saved. Wait on the real POST, not log text.
    with page.expect_response(
        lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"
    ) as save_resp_info:
        page.locator("#bSave").click()
    assert save_resp_info.value.status == 200, (
        f"saving the anchor failed: HTTP {save_resp_info.value.status}"
    )

    # Clear all links backs up before it overwrites; existing backups are noted first so an older one is not taken for
    # the new one.
    backups_before = set(glob.glob(os.path.join(PROJECT_DIR, "backups", "placement.backup-*.json"))) if PROJECT_DIR else None

    page.on("dialog", lambda d: d.accept())
    _show_bottom_tab(page, "anchors")                 # Clear all links lives on the Anchors tab
    with page.expect_response(
        lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"
    ) as clear_resp_info:
        page.locator("#bClearLinks").click()
    assert clear_resp_info.value.status == 200, (
        f"saving the cleared placement failed: HTTP {clear_resp_info.value.status}"
    )
    assert page.evaluate("() => anchors.length") == 0, "anchors[] was not cleared client-side"

    def server_anchors():
        return call(f"{BENCH_URL}/api/placement", "GET", timeout=10).get("anchors", [])
    # An earlier save can still be landing under load: wait for the clear, then check nothing brings links back.
    deadline = time.time() + 15
    while server_anchors() and time.time() < deadline:
        page.wait_for_timeout(500)
    page.wait_for_timeout(1500)
    left = server_anchors()
    assert left == [], (f"server still holds {len(left)} anchor(s) after Clear all links: "
                        f"{[(a.get('id'), [m.get('scan') for m in a.get('members', [])]) for a in left]}")

    if backups_before is not None:
        backups_after = set(glob.glob(os.path.join(PROJECT_DIR, "backups", "placement.backup-*.json")))
        new_backups = backups_after - backups_before
        assert new_backups, (
            f"Clear all links wrote no new backup file in {PROJECT_DIR} - the "
            f"anchors it just cleared would have been unrecoverable"
        )
        # A late, debounced save can add a second new backup (of a state without the anchor); what matters is that
        # one of them holds what was cleared.
        holding = []
        for path in sorted(new_backups):
            with open(path) as fh:
                if json.load(fh).get("anchors"):
                    holding.append(path)
        assert holding, f"none of the new backups {sorted(new_backups)} holds the anchor that was about to be cleared"
        backup_path = holding[-1]
        print(f"    built {n_anchors} anchor(s), cleared, server confirms 0, "
              f"backup written to {os.path.basename(backup_path)} with the anchor still in it")
    else:
        print(f"    built {n_anchors} anchor(s), cleared, server confirms 0 "
              f"(PROJECT_DIR not set, skipped checking for a backup file)")


def _type_x(page, text):
    """Type into the X box and press Enter, the way a person does."""
    box = page.locator("#in_x")
    box.click()
    box.fill(text)
    box.press("Enter")


def test_save_reload_same_pose(page):
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    original_x = _pose(page, scan_i)["x"]
    try:
        _type_x(page, "3.14159")
        saved_x = _pose(page, scan_i)["x"]
        assert abs(saved_x - 3.14159) < 1e-4, f"typing into the x box did not set x (got {saved_x})"

        page.locator("#bSave").click()
        page.wait_for_function("() => $('out').value.includes('saved to')", timeout=10_000)

        page.reload(wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
        page.wait_for_function(
            """() => {
                const rows = document.querySelectorAll('#scanlist .scan');
                if (!rows.length) return false;
                return [...rows].every(r => !r.textContent.includes('building preview'));
            }""",
            timeout=LOAD_TIMEOUT_MS,
        )
        reloaded_x = _pose(page, scan_i)["x"]
        assert abs(reloaded_x - saved_x) < 1e-4, (
            f"after Save + reload, x={reloaded_x!r}, expected the saved {saved_x!r}"
        )
        print(f"    saved x={saved_x:.5f}, reload shows x={reloaded_x:.5f}")
    finally:
        # Put the scan back: a later test re-checks this scan's links.
        _select_scan_row(page, 0)
        _type_x(page, repr(original_x))
        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()


def test_links_table_shows_real_pair_results(page):
    """The links table has one row per pair (rebuildLinks() runs after scans load), and
    applyAlignAutoResultToLinks() fills in fit, RMSE, tilt and state from a job result. Uses a synthetic pairs array
    in the server's real shape: this tests the page's wiring, not registration."""
    _goto(page)
    n_scans, n_rows = page.evaluate("() => [S.length, LINKS.length]")
    expected = n_scans * (n_scans - 1) // 2
    assert n_rows == expected, f"LINKS has {n_rows} row(s) for {n_scans} scans, expected {expected} (n choose 2)"

    result = page.evaluate("""() => {
        const a = S[0].name, b = S[1].name, c = S[2].name;
        applyAlignAutoResultToLinks([
            {source_name: a, target_name: b, accepted: true,  fitness: 0.0843, rmse_m: 0.0151, tilt_deg: 1.74},
            {source_name: a, target_name: c, accepted: false, fitness: 0.0091, rmse_m: 0.0210, tilt_deg: 0.30},
        ]);
        renderLinks();
        const find=(x,y)=>LINKS.find(l=>(S[l.a].name===x&&S[l.b].name===y)||(S[l.a].name===y&&S[l.b].name===x));
        const ok=find(a,b), weak=find(a,c);
        return {
            okOn: ok.on, okState: ok.state, okFit: ok.fit,
            weakOn: weak.on, weakState: weak.state,
        };
    }""")
    page.wait_for_timeout(300)
    result["rowHtml"] = page.evaluate("() => document.getElementById('links').innerHTML")
    assert result["okOn"] is True, "an attempted pair must be marked on, not left dimmed as 'not linked'"
    assert result["okState"] == "ok", f"an accepted pair should read 'ok', got {result['okState']!r}"
    assert result["okFit"] == "0.0843", f"fitness was not carried through: {result['okFit']!r}"
    assert result["weakOn"] is True, "a REJECTED pair must still be visible - it is the one worth triaging"
    assert result["weakState"] == "weak", f"a rejected pair should read 'weak', got {result['weakState']!r}"
    assert "0.0843" in result["rowHtml"], "the rendered table does not show the real fitness number"
    print(f"    LINKS has {n_rows} real row(s) for {n_scans} scans; "
          f"an accepted pair shows on/ok with real numbers, a rejected one stays visible as 'weak'")


def test_persisted_links_survive_reload_and_show_declined_and_stale(page):
    """Links table rows persist across reloads (data.last_align_pairs): a 'kept_placement' pair renders as 'declined',
    and a pair recorded against a pose that no longer matches reads 'stale' with no numbers. Synthetic data in the
    real shape."""
    _goto(page)
    result = page.evaluate("""() => {
        const a = S[0].name, b = S[1].name, c = S[2].name;
        const poseOf = s => [s.x, s.y, s.yaw * 180 / Math.PI];
        applyPersistedLinksOnLoad([
            {source_name: a, target_name: b, accepted: true, method: 'kept_placement',
             fitness: 0.05, rmse_m: 0.02, tilt_deg: 0.5,
             a_pose_at_align: poseOf(S[0]), b_pose_at_align: poseOf(S[1])},
            {source_name: a, target_name: c, accepted: true, method: 'point_to_plane',
             fitness: 0.09, rmse_m: 0.01, tilt_deg: 0.3,
             a_pose_at_align: poseOf(S[0]),
             b_pose_at_align: [S[2].x + 1.0, S[2].y, S[2].yaw * 180 / Math.PI]},
        ]);
        renderLinks();
        const find=(x,y)=>LINKS.find(l=>(S[l.a].name===x&&S[l.b].name===y)||(S[l.a].name===y&&S[l.b].name===x));
        const declined=find(a,b), stale=find(a,c);
        return {
            declinedState: declined.state, declinedOn: declined.on, declinedFit: declined.fit,
            staleState: stale.state, staleFit: stale.fit,
        };
    }""")
    page.wait_for_timeout(300)
    result["rowHtml"] = page.evaluate("() => document.getElementById('links').innerHTML")
    assert result["declinedOn"] is True
    assert result["declinedState"] == "declined", (
        f"a kept_placement pair should read 'declined', got {result['declinedState']!r}")
    assert result["declinedFit"] == "0.0500", (
        f"a fresh declined row should still carry its numbers: {result['declinedFit']!r}")
    assert result["staleState"] == "stale", (
        f"a pair recorded against an old pose should read 'stale', got {result['staleState']!r}")
    assert result["staleFit"] is None, (
        "a stale row must not show a number that no longer describes the current scans")
    assert "kept placement" in result["rowHtml"], (
        "the table must say 'kept placement' in plain language, not a raw state code")
    assert "stale" in result["rowHtml"]
    print(f"    declined pair reads {result['declinedState']!r} with real numbers; "
          f"moved-scan pair reads {result['staleState']!r} with no numbers")


def test_pose_row_status_line_does_not_call_a_capped_row_moved(page):
    """A capped row (applied=true but the candidate rejected for moving too far, so the pose is the hand placement)
    never claims the scan moved or shows a tilt for it."""
    _goto(page)
    lines = page.evaluate("""() => [
        poseRowStatusLine({scan: 'A', applied: true, residual_tilt_deg: 1.74}),
        poseRowStatusLine({scan: 'B', applied: true, residual_tilt_deg: null, kept_placement_bound_exceeded: true}),
        poseRowStatusLine({scan: 'C', applied: false, reason: 'locked; automatic align does not move a locked scan'}),
    ]""")
    assert "moved" in lines[0] and "1.74" in lines[0], f"a real move should report its tilt: {lines[0]!r}"
    assert "moved" not in lines[1], f"a capped row must not be reported as moved: {lines[1]!r}"
    assert "null" not in lines[1] and "None" not in lines[1], f"a capped row must not leak the raw null value: {lines[1]!r}"
    assert "kept your placement" in lines[1], f"a capped row should say its placement was kept: {lines[1]!r}"
    assert lines[2] == "  C: kept - locked; automatic align does not move a locked scan"
    print("    a real move reports its tilt, a capped row says 'kept your placement' with no tilt number, "
          "a locked scan reports its own reason - all three distinct")


def test_note_toggles_hide_explanations_by_default_and_remember_state(page):
    """The notes under Move step, Floor height, Slice and Exclusions are hidden by default; each section's "?" reveals
    only its own note, and the choice survives a reload."""
    _goto(page)
    hidden_by_default = page.evaluate(
        "() => [...document.querySelectorAll('.note[data-notegroup]')].every(n => n.hidden)"
    )
    assert hidden_by_default, "at least one explanatory note is visible by default"

    page.locator('.noteToggle[data-notegroup="floorHeight"]').click()
    floor_hidden = page.evaluate(
        '() => document.querySelector(\'.note[data-notegroup="floorHeight"]\').hidden'
    )
    assert not floor_hidden, "clicking the Floor height '?' did not reveal its explanation"
    excl_hidden = page.evaluate(
        '() => document.querySelector(\'.note[data-notegroup="moveStep"]\').hidden'
    )
    assert excl_hidden, "revealing one section's note must not reveal another's"

    # Wait for scans to finish loading, as _goto does: the static .note element exists before initNoteToggles runs.
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function(
        """() => {
            const rows = document.querySelectorAll('#scanlist .scan');
            if (!rows.length) return false;
            return [...rows].every(r => !r.textContent.includes('building preview'));
        }""",
        timeout=LOAD_TIMEOUT_MS,
    )
    still_open = page.evaluate(
        '() => !document.querySelector(\'.note[data-notegroup="floorHeight"]\').hidden'
    )
    assert still_open, "the open/closed choice did not survive a reload"
    print("    notes hidden by default; Floor height's '?' reveals only its own note; state survives reload")


def test_settings_explanations_moved_to_hover_and_the_help_menu(page):
    """Settings explanations are hover titles on their controls, with one Help entry per setting under "Settings,
    explained". Mouse snapping lives in the View popover and has only its hover title."""
    _goto(page)
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    settings_text = page.locator('[data-menu-modal="menuSettings"]').text_content()
    assert "Off by default in the pipeline's own code" not in settings_text, (
        "the Search voxel explanation is still sitting permanently visible in Settings"
    )
    assert "Only applies to" not in settings_text, (
        "the Detector voting explanation is still sitting permanently visible in Settings"
    )

    voxel_title = page.locator("#searchVoxel").get_attribute("title")
    assert voxel_title and "find starting positions" in voxel_title, (
        f"Search voxel's explanation did not move to a hover title: {voxel_title!r}"
    )
    vote_title = page.locator("#cbDetectorVote").locator("xpath=ancestor::*[@title][1]").get_attribute("title")
    assert vote_title and 'Run from my alignment' in vote_title, (
        f"Detector voting's explanation did not move to a hover title: {vote_title!r}"
    )
    # Settings is a modal with a full-page overlay: close it first or the overlay takes the click below.
    page.keyboard.press("Escape")
    assert not page.locator('[data-menu-modal="menuSettings"]').is_visible(), "Escape did not close the Settings modal"
    # Mouse snapping lives in the View popover and keeps its own hover title.
    page.locator('.menubtn[data-menu="menuView"]').click()
    mouse_title = page.locator("#cbMouseAssist").locator("xpath=ancestor::*[@title][1]").get_attribute("title")
    assert mouse_title and "Reserved for later" in mouse_title, (
        f"Mouse snapping's explanation did not stay on a hover title: {mouse_title!r}"
    )

    page.locator('.menubtn[data-menu="menuHelp"]').click()
    help_text = page.locator('[data-menu-modal="menuHelp"]').text_content()
    # text_content() returns the raw heading text, not the CSS-uppercased version.
    assert "Settings, explained" in help_text, "Help does not have a Settings section"
    # Help entries are more detailed than the hover titles, so check what each entry states, not the title's wording.
    assert "Search voxel (automatic align)" in help_text and "candidate GENERATION" in help_text
    assert "Detector voting" in help_text and "Run from my alignment" in help_text
    assert "Interface text size" in help_text
    # One entry per real Settings-modal control, independently anchored.
    for anchor in ("resources", "levelTilt", "rangeLimit", "searchVoxel",
                   "cbDetectorVote", "strayRemoval", "themeSelect", "textSize"):
        assert page.locator(f"#help-{anchor}").count() == 1, f"Help has no entry for {anchor!r}"
    print("    Settings modal no longer shows permanent explanation paragraphs; the same text is on "
          "each control's hover title and Help's 'Settings, explained' tab has one entry per setting")


def test_settings_help_link_opens_help_at_that_settings_own_entry(page):
    """A setting's "?" opens Help on the "Settings, explained" tab, scrolled to that setting's entry. Checked on
    Detector voting, which is on a different Settings tab from the default."""
    _goto(page)
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    page.locator('button[role="tab"]:has-text("Aligning")').click()
    assert not page.locator('[data-menu-modal="menuHelp"]').is_visible(), "Help is already open before the '?' is clicked"

    page.locator('[data-setting="cbDetectorVote"] .mc-help-link').click()

    settings_panel = page.locator('[data-menu-modal="menuSettings"]')
    help_panel = page.locator('[data-menu-modal="menuHelp"]')
    assert not settings_panel.is_visible(), "clicking a setting's '?' should close Settings, not stack it under Help"
    assert help_panel.is_visible(), "Help did not open when a setting's '?' was clicked"

    entry = page.locator("#help-cbDetectorVote")
    assert entry.is_visible(), "Help opened but not on the Settings-explained tab (or not scrolled to it)"
    assert "Detector voting" in entry.inner_text() and "Run from my alignment" in entry.inner_text()
    box = entry.bounding_box()
    viewport = page.viewport_size
    assert box and 0 <= box["y"] <= viewport["height"], (
        f"Help opened at the right entry but did not actually scroll it into view: y={box}"
    )
    print("    Detector voting's '?' (Aligning tab) opened Help on Settings-explained, scrolled to its own entry")


def test_exclusions_and_removals_list_apart_and_the_export_setting_saves_with_its_help(page):
    """An exclude region and a review removal each show in their own list; the export setting has hover text, saves to
    the project, survives a reload, and its "?" opens its Help entry."""
    # The page autosaves settled changes, so the two made-up regions can reach the file: put it back after.
    placement_path = os.path.join(PROJECT_DIR, "placement.json") if PROJECT_DIR else ""
    original = open(placement_path, "rb").read() if placement_path and os.path.exists(placement_path) else None
    _goto(page)
    page.evaluate("""() => {
        const s = S[0], box = {scan: s.name, points: [], centroid: [0, 0, 0], extent: [0.5, 0.5, 0.5], n_points: 3};
        exclusions.push(Object.assign({id: 'test-region'}, box), Object.assign({id: 'test-removal', source: 'review'}, box));
        renderAll();
    }""")
    page.wait_for_function("() => document.querySelectorAll('#removalsList .projectRow').length === 1")
    assert page.locator("#exclusionsList .projectRow").count() == 1, "the exclusions list did not hold exactly the region"

    settings_url = f"{BENCH_URL}/api/settings"
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    page.locator('button[role="tab"]:has-text("Exporting")').click()
    setting = page.locator('[data-setting="cbExportKeepsExcluded"]')
    assert "still exported" in (setting.get_attribute("title") or ""), "the setting has no hover description"
    box = page.locator("#cbExportKeepsExcluded")
    assert box.is_checked(), "keeping exclude-region points in the export must be the default"
    try:
        with page.expect_response(lambda r: r.url == settings_url and r.request.method == "POST"):
            box.click()
        assert call(settings_url, "GET", timeout=10).get("export_keeps_excluded") is False, (
            "unticking did not reach the project")
        _goto(page)
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Exporting")').click()
        assert not page.locator("#cbExportKeepsExcluded").is_checked(), "the setting did not survive a reload"
        page.locator('[data-setting="cbExportKeepsExcluded"] .mc-help-link').click()
        entry = page.locator("#help-cbExportKeepsExcluded")
        assert entry.is_visible() and "thick band" in entry.inner_text(), "the '?' did not open this setting's Help entry"
    finally:
        if original is not None:
            page.goto("about:blank")                  # off the page first, so it cannot autosave over the restore
            with open(placement_path, "wb") as fh:
                fh.write(original)
        _goto(page)
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Exporting")').click()
        if not page.locator("#cbExportKeepsExcluded").is_checked():
            with page.expect_response(lambda r: r.url == settings_url and r.request.method == "POST"):
                page.locator("#cbExportKeepsExcluded").click()
    assert page.evaluate("() => !exclusions.some(e => e.id.startsWith('test-'))"), "the made-up regions stayed"
    print("    region and removal listed apart; export setting: hover text, saved, survived reload, Help entry")


def test_menu_bar_opens_and_closes(page):
    """File, Edit and View each open their own panel with real content and close on an outside click. Settings and Help
    are modals, found by their [data-menu-modal] marker."""
    _goto(page)
    popovers = ["menuFile", "menuEdit", "menuView"]
    for menu_id in popovers:
        page.locator(f'.menubtn[data-menu="{menu_id}"]').click()
        panel = page.locator(f"#{menu_id}")
        assert panel.is_visible(), f"{menu_id} did not open"
        text = panel.inner_text().strip()
        assert len(text) > 10, f"{menu_id} opened but looks empty: {text!r}"
        for other in popovers:
            if other != menu_id:
                assert not page.locator(f"#{other}").is_visible(), (
                    f"opening {menu_id} left {other} open too"
                )
        # Click outside on the right sidebar's heading, which no menu panel covers.
        page.locator('h2:has-text("Selected")').click()
        assert not panel.is_visible(), f"{menu_id} did not close on an outside click"
    print(f"    all {len(popovers)} popover menu(s) open individually, show real content, and close")

    for menu_id, marker in (("menuSettings", "menuSettings"), ("menuHelp", "menuHelp")):
        page.locator(f'.menubtn[data-menu="{menu_id}"]').click()
        panel = page.locator(f'[data-menu-modal="{marker}"]')
        assert panel.is_visible(), f"{menu_id} modal did not open"
        text = panel.inner_text().strip()
        assert len(text) > 10, f"{menu_id} modal opened but looks empty: {text!r}"
        other_marker = "menuHelp" if marker == "menuSettings" else "menuSettings"
        assert not page.locator(f'[data-menu-modal="{other_marker}"]').is_visible(), (
            f"opening {menu_id} left the other modal open too"
        )
        # Close the modal with a click on its overlay at a fixed corner.
        page.mouse.click(5, 5)
        assert not panel.is_visible(), f"{menu_id} modal did not close on an outside click"
    print("    Settings and Help modals open individually, show real content, and close")


def test_scan_alias_persists(page):
    """Double-click a scan's name to rename it: the alias shows in the sidebar and survives Save and reload, and the
    file name (not the alias) is still what is sent to the server."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    real_name = page.evaluate("(i) => S.find(x => x.i === i).name", scan_i)
    alias = "Green Room (my name)"

    name_el = page.locator(f'.scanname[data-rename="{scan_i}"]')
    name_el.dblclick()
    renamed_input = page.locator("#scanlist input.num")
    renamed_input.fill(alias)
    renamed_input.press("Enter")

    shown = page.locator(f'.scanname[data-rename="{scan_i}"]').inner_text()
    assert shown == alias, f"sidebar shows {shown!r}, expected the alias {alias!r}"

    page.locator("#bSave").click()
    page.wait_for_function("() => $('out').value.includes('saved to')", timeout=10_000)

    saved = call(f"{BENCH_URL}/api/placement", "GET", timeout=10)
    saved_scan = next(s for s in saved["scans"] if s["name"] == real_name)
    assert saved_scan.get("alias") == alias, (
        f"server saved alias={saved_scan.get('alias')!r}, expected {alias!r}"
    )
    assert saved_scan["name"] == real_name, (
        "the alias overwrote the real file name in the saved placement - "
        "exports and reports must keep naming the real file"
    )

    page.reload(wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        """() => {
            const rows = document.querySelectorAll('#scanlist .scan');
            if (!rows.length) return false;
            return [...rows].every(r => !r.textContent.includes('building preview'));
        }""",
        timeout=LOAD_TIMEOUT_MS,
    )
    reloaded_shown = page.locator(f'.scanname[data-rename="{scan_i}"]').inner_text()
    assert reloaded_shown == alias, (
        f"after reload, sidebar shows {reloaded_shown!r}, expected the alias {alias!r} to survive"
    )
    print(f"    real file name {real_name!r} kept; alias {alias!r} shown and survives reload")


def test_scan_display_colour_persists_and_stays_out_of_export(page):
    """A scan's false-colour swatch survives Save and reload, saved as display_color, separate from s.rgb (the real
    colour), so merge and export can never pick it up."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    real_name = page.evaluate("(i) => S.find(x => x.i === i).name", scan_i)
    chosen_hex = "#3399ff"

    color_input = page.locator("#in_col")
    color_input.fill(chosen_hex)
    color_input.dispatch_event("input")

    got_col = page.evaluate("(i) => S.find(x => x.i === i).col", scan_i)
    assert got_col == [0x33, 0x99, 0xFF], f"s.col={got_col!r} after picking {chosen_hex!r}"

    page.locator("#bSave").click()
    page.wait_for_function("() => $('out').value.includes('saved to')", timeout=10_000)

    saved = call(f"{BENCH_URL}/api/placement", "GET", timeout=10)
    saved_scan = next(s for s in saved["scans"] if s["name"] == real_name)
    assert saved_scan.get("display_color") == [0x33, 0x99, 0xFF], (
        f"server saved display_color={saved_scan.get('display_color')!r}, expected [51, 153, 255]"
    )
    assert "rgb" not in saved_scan, (
        "the real captured colour (rgb) has no business in placement.json at all - "
        "if this ever appears, display_color and export colour have been conflated"
    )

    page.reload(wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        """() => {
            const rows = document.querySelectorAll('#scanlist .scan');
            if (!rows.length) return false;
            return [...rows].every(r => !r.textContent.includes('building preview'));
        }""",
        timeout=LOAD_TIMEOUT_MS,
    )
    reloaded_col = page.evaluate("(i) => S.find(x => x.i === i).col", scan_i)
    assert reloaded_col == [0x33, 0x99, 0xFF], (
        f"after reload, s.col={reloaded_col!r}, expected the picked colour to survive"
    )
    print(f"    picked {chosen_hex!r} -> s.col={got_col!r}, survives Save+reload, "
          f"absent as a real-colour field in the saved file")


def test_exclude_region_roundtrip(page):
    """Draw an exclusion region with the Exclude tool, see it listed, Save writes it to the project file, it survives a
    reload, and restoring it removes it. Whether alignment and export skip the full-resolution points is covered by
    tests/test_exclusions.py."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    scan_name = page.evaluate("(i) => S.find(x => x.i === i).name", scan_i)
    pt = _scan_point_screen_pos(page, scan_i)
    rect = _canvas_rect(page)
    vx, vy = _to_viewport(rect, (pt["x"], pt["y"]))

    page.locator('[data-view="plan"]').click()
    page.locator('[data-tool="exclude"]').click()
    page.mouse.move(vx - 25, vy - 25)
    page.mouse.down()
    page.mouse.move(vx + 25, vy + 25, steps=4)
    page.mouse.up()

    n_exclusions = page.evaluate("() => exclusions.length")
    assert n_exclusions == 1, f"expected 1 exclusion after the drag, got {n_exclusions}"
    rows = page.locator("#exclusionsList .projectRow")
    assert rows.count() == 1, "the drawn exclusion did not appear in the sidebar list"

    page.locator("#bSave").click()
    page.wait_for_function("() => $('out').value.includes('saved to')", timeout=10_000)

    saved = call(f"{BENCH_URL}/api/placement", "GET", timeout=10)
    saved_exclusions = saved.get("exclusions", [])
    assert len(saved_exclusions) == 1, (
        f"server saved {len(saved_exclusions)} exclusion(s), expected 1: {saved_exclusions}"
    )
    assert saved_exclusions[0]["scan"] == scan_name, (
        f"saved exclusion names {saved_exclusions[0]['scan']!r}, expected {scan_name!r}"
    )

    page.reload(wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_function(
        """() => {
            const rows = document.querySelectorAll('#scanlist .scan');
            if (!rows.length) return false;
            return [...rows].every(r => !r.textContent.includes('building preview'));
        }""",
        timeout=LOAD_TIMEOUT_MS,
    )
    reloaded_n = page.evaluate("() => exclusions.length")
    assert reloaded_n == 1, f"after reload, {reloaded_n} exclusion(s) survived, expected 1"

    _show_bottom_tab(page, "exclusions")
    page.locator("#exclusionsList [data-restore]").click()
    after_restore = page.evaluate("() => exclusions.length")
    assert after_restore == 0, "restore did not remove the exclusion from the in-page list"

    # Save again so the restore reaches the server, and the exclusion does not leak into later tests.
    page.locator("#bSave").click()
    page.wait_for_function("() => $('out').value.includes('saved to')", timeout=10_000)
    saved_after = call(f"{BENCH_URL}/api/placement", "GET", timeout=10)
    assert saved_after.get("exclusions", []) == [], (
        "restored exclusion is still on the server after saving"
    )
    print(f"    excluded a region of {scan_name!r}, saved, survived reload, restored, re-saved clean")


def test_an_exclude_box_started_on_the_spin_handle_is_a_box_not_a_spin(page):
    """Under the exclude tool, a drag that begins on the selected scan's spin handle draws a box, not a turn."""
    _goto(page)
    _select_scan_row(page, 0)
    page.locator('[data-view="plan"]').click()
    page.locator('[data-tool="exclude"]').click()
    g = page.evaluate("() => { const g = gizmoAt(S[sel]); return {kx: g.kx, ky: g.ky, cx: g.c[0], cy: g.c[1], yaw: S[sel].yaw}; }")
    rect = _canvas_rect(page)
    sx, sy = _to_viewport(rect, (g["kx"], g["ky"]))
    # To beside the centre, not onto it: at yaw 0 the handle is straight above the centre, and a box with no width
    # is a click.
    ex, ey = _to_viewport(rect, (g["cx"] + 30, g["cy"]))
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(ex, ey, steps=6)
    page.mouse.up()
    yaw = page.evaluate("() => S[sel].yaw")
    assert yaw == g["yaw"], f"the scan turned from {g['yaw']} to {yaw}"
    assert page.evaluate("() => exclusions.length") == 1, "the drag did not make an exclusion box"
    print("    exclude box from the spin handle: scan did not turn, one exclusion made (not saved)")


def test_review_view_removes_points_by_box_lasso_and_brush_and_the_server_cuts_the_same(page):
    """Edit > Review shows one scan alone; box, lasso and brush select; Delete removes; Undo and Redo; Save and reload
    keep it; the server's full-resolution cut matches what the page hid. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set - this check reads the project's own scan file")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        page.locator('[data-view="plan"]').click()
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        page.wait_for_function("() => getComputedStyle($('reviewBar')).display !== 'none'")
        shown = page.evaluate("() => S.filter(s => s.vis).map(s => s.i)")
        assert shown == [scan_i], f"review should show the one scan alone, shows {shown}"
        full_buf = page.evaluate("() => glr ? glr.nPoints : -1")

        rect = page.locator("#cv").bounding_box()
        cx0, cy0 = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
        page.locator('#reviewBar [data-rvtool="box"]').click()
        page.mouse.move(cx0 - 40, cy0 - 40); page.mouse.down(); page.mouse.move(cx0 + 40, cy0 + 40, steps=5); page.mouse.up()
        n_box = page.evaluate("() => review.sel.n")
        assert n_box > 0, "the box selected nothing in the middle of the scan"
        page.locator('#reviewBar [data-rvtool="lasso"]').click()
        page.keyboard.down("Shift")
        page.mouse.move(cx0 + 60, cy0 - 30); page.mouse.down()
        for dx, dy in ((120, -30), (120, 30), (60, 30), (60, -30)):
            page.mouse.move(cx0 + dx, cy0 + dy, steps=4)
        page.mouse.up()
        page.locator('#reviewBar [data-rvtool="brush"]').click()
        page.mouse.move(cx0 - 120, cy0); page.mouse.down(); page.mouse.move(cx0 - 80, cy0 + 40, steps=8); page.mouse.up()
        page.keyboard.up("Shift")
        n_sel, n_shapes = page.evaluate("() => [review.sel.n, review.sel.shapes.length]")
        assert n_shapes == 3 and n_sel >= n_box, f"shift should add: {n_shapes} shapes, {n_sel} points (box {n_box})"
        assert "preview points selected" in page.locator("#rvInfo").inner_text()

        page.keyboard.press("Delete")
        gone = page.evaluate("() => glr.nPoints")
        assert gone == full_buf - n_sel, f"Delete should hide {n_sel} points: {full_buf} -> {gone}"
        assert page.evaluate("() => exclusions.filter(e => e.source === 'review').length") == 1
        page.locator("#rvUndo").click()
        assert page.evaluate("() => glr.nPoints") == full_buf, "Undo did not put the points back"
        page.locator("#rvRedo").click()
        assert page.evaluate("() => glr.nPoints") == gone, "Redo did not remove them again"
        page.locator("#rvReviewed").check()
        page.locator("#rvDone").click()
        assert page.evaluate("() => S.filter(s => s.vis).length") > 1, "Done did not bring the other scans back"

        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        _goto(page)
        page.wait_for_function("(i) => S[i]._removed && S[i]._removed.some(v => v)", arg=scan_i, timeout=20000)
        hidden = page.evaluate("(i) => S[i]._removed.reduce((a, v) => a + v, 0)", scan_i)
        n_preview = page.evaluate("(i) => S[i].xy.length / 2", scan_i)
        assert hidden == n_sel, f"after reload {hidden} preview points hidden, {n_sel} were removed"
        assert page.evaluate("(i) => S[i].reviewed", scan_i) is True, "the reviewed tick did not survive a reload"

        # The server's test on the very preview points the page tested must agree point for point (float32 on the
        # page, so a point sitting on an outline may differ), and it must cut full-resolution points too.
        import numpy as np
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from open_point_stitch.exclusions import excluded_mask
        from open_point_stitch.placement import load_placements
        from open_point_stitch.pts_file import read_pts
        name = page.evaluate("(i) => S[i].name", scan_i)
        regions = [e for e in load_placements(placement_path).exclusions if e.scan == name and e.shapes]
        raw = np.array(page.evaluate("(i) => Array.from(reviewRawPoints(S[i]))", scan_i), dtype=float).reshape(-1, 3)
        page_mask = np.array(page.evaluate("(i) => Array.from(S[i]._removed)", scan_i), dtype=bool)
        server_mask = excluded_mask(raw, regions)
        differ = int((server_mask != page_mask).sum())
        assert differ <= max(2, n_preview // 1000), f"server and page disagree on {differ} of {n_preview} preview points"
        pts = read_pts(os.path.join(PROJECT_DIR, "scans-raw", name + ".pts")).points
        cut = int(excluded_mask(pts, regions).sum())
        assert cut > hidden, f"the server cuts only {cut} full-resolution points for {hidden} preview points"
        print(f"    review: box, lasso and brush selected {n_sel} preview points; removed, undone, redone, saved; "
              f"server agrees on {n_preview - differ} of {n_preview} preview points and cuts {cut:,} of "
              f"{len(pts):,} full-resolution points")
    finally:
        with open(placement_path, "wb") as fh:
            fh.write(original)


def test_floor_and_ceiling_marks_drag_type_select_and_survive_a_reload(page):
    """In review the floor and ceiling show as planes; the floor label drags in side view, the ceiling can be typed,
    selected points can be marked floor, "as found" resets, and the Floor height panel's buttons set the same marks.
    Restores the project file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        page.locator('[data-view="side"]').click()
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        page.wait_for_function("() => getComputedStyle($('reviewBar')).display !== 'none'")
        planes = page.evaluate("(i) => markPlanes(S[i]).map(p => [p.key, p.h, p.tag])", scan_i)
        assert [p[0] for p in planes] == ["floor_h_m", "ceiling_h_m"], f"expected a floor and a ceiling plane: {planes}"
        assert float(page.locator("#rvFloor").input_value()) == 0.0

        rect = page.locator("#cv").bounding_box()
        x, y, w, h = planes[0][2]
        per_m = page.evaluate("() => cam.s * Math.max(Math.cos(ele * D2R), 0.05)")
        page.mouse.move(rect["x"] + x + w / 2, rect["y"] + y + h / 2); page.mouse.down()
        page.mouse.move(rect["x"] + x + w / 2, rect["y"] + y + h / 2 - 30, steps=5); page.mouse.up()
        dragged = page.evaluate("(i) => S[i].floor_h_m", scan_i)
        assert dragged is not None and abs(dragged - 30 / per_m) < 0.02, f"floor dragged to {dragged}, expected {30 / per_m:.3f}"

        page.locator("#rvCeil").fill("2.4"); page.locator("#rvCeil").press("Enter")
        assert page.evaluate("(i) => S[i].ceiling_h_m", scan_i) == 2.4, "typing the ceiling did not set it"
        page.locator("#rvUndo").click()
        assert page.evaluate("(i) => S[i].ceiling_h_m", scan_i) is None, "Undo did not take the typed ceiling back"

        tags = page.evaluate("(i) => markPlanes(S[i]).map(p => p.tag)", scan_i)
        for frac in [k / 20 for k in range(1, 20)]:                  # a real point the view itself can be clicked on,
            pt = _scan_point_screen_pos(page, scan_i, k_frac=frac)  # not under the review bar, clear of both labels
            on_view = page.evaluate("(p) => [[-25,-25],[25,25],[0,0]].every(([dx,dy]) => "
                                    "document.elementFromPoint(p.x + dx, p.y + dy) === cv)",
                                    {"x": rect["x"] + pt["x"], "y": rect["y"] + pt["y"]})
            if on_view and all(not (t[0] - 40 < pt["x"] < t[0] + t[2] + 40 and t[1] - 40 < pt["y"] < t[1] + t[3] + 40)
                               for t in tags):
                break
        bx, by = rect["x"] + pt["x"], rect["y"] + pt["y"]
        page.locator('#reviewBar [data-rvtool="box"]').click()
        page.mouse.move(bx - 25, by - 25); page.mouse.down(); page.mouse.move(bx + 25, by + 25, steps=4); page.mouse.up()
        assert page.evaluate("() => review.sel.n") > 0, "the box over a real point selected nothing"
        med = page.evaluate("""(i) => { const s = S[i], hs = [];
            review.sel.mask.forEach((v, k) => { if (v) hs.push(s.h[k]); }); hs.sort((a, b) => a - b); return hs[hs.length >> 1]; }""",
                            scan_i)
        page.locator("#rvFloorSel").click()
        assert abs(page.evaluate("(i) => S[i].floor_h_m", scan_i) - med) < 0.001, "selected points did not set the floor"
        page.locator("#rvMarksReset").click()
        assert page.evaluate("(i) => [S[i].floor_h_m, S[i].ceiling_h_m]", scan_i) == [None, None]
        page.locator("#rvDone").click()

        band_h = page.evaluate("(i) => [...S[i].bands].sort((a, b) => b.h - a.h)[0].h", scan_i)
        page.locator('#bands [data-set="ceiling"]').first.click()        # bands are listed highest first
        assert page.evaluate("(i) => S[i].ceiling_h_m", scan_i) == round(band_h, 3), "the panel's ceiling button did nothing"
        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        _goto(page)
        assert page.evaluate("(i) => S[i].ceiling_h_m", scan_i) == round(band_h, 3), "the ceiling mark did not survive a reload"
        print(f"    floor dragged to {dragged:.3f} m, typed ceiling undone, floor from selection {med:.3f} m, "
              f"panel ceiling {band_h:.2f} m saved and reloaded")
    finally:
        with open(placement_path, "wb") as fh:
            fh.write(original)


def test_walls_move_delete_and_add_in_review_and_the_server_applies_the_same(page):
    """In review, Plan view, the Walls tool drags a wall out, Delete removes one, a drag on empty space draws one; Undo
    works; saved and reloaded, the server's rule gives the walls the page shows."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        page.locator('[data-view="plan"]').click()
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        page.locator('#reviewBar [data-rvtool="walls"]').click()
        n0 = page.evaluate("(i) => effectiveWalls(S[i]).length", scan_i)
        assert n0 >= 2, f"need two detected walls, the scan has {n0}"
        rect = page.locator("#cv").bounding_box()

        def wall_mid(k):
            return page.evaluate("""([i, k]) => { const s = S[i], w = effectiveWalls(s)[k];
                const a = w2s(place(s, [w.x1, w.y1])), b = w2s(place(s, [w.x2, w.y2]));
                const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
                return {x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, nx: -(b[1] - a[1]) / len, ny: (b[0] - a[0]) / len}; }""",
                                 [scan_i, k])

        clickable = [k for k in range(n0) if page.evaluate(
            "(p) => document.elementFromPoint(p.x, p.y) === cv",
            {"x": rect["x"] + wall_mid(k)["x"], "y": rect["y"] + wall_mid(k)["y"]})]
        assert len(clickable) >= 2, f"only walls {clickable} of {n0} can be clicked in this view"
        first, second = clickable[0], clickable[1]
        m = wall_mid(first)
        page.mouse.move(rect["x"] + m["x"], rect["y"] + m["y"]); page.mouse.down()
        page.mouse.move(rect["x"] + m["x"] + 30 * m["nx"], rect["y"] + m["y"] + 30 * m["ny"], steps=5); page.mouse.up()
        edits = page.evaluate("(i) => S[i].wall_edits", scan_i)
        cam_s = page.evaluate("() => cam.s")
        assert len(edits) == 1 and edits[0]["op"] == "move" and abs(abs(edits[0]["by"]) - 30 / cam_s) < 0.02, (edits, 30 / cam_s)

        # The picked wall's end dot drags its end in X and Y.
        end = page.evaluate("""(i) => { const s = S[i], w = review.wallSel, b = w2s(place(s, [w[2], w[3]]));
            const a = w2s(place(s, [w[0], w[1]])), len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
            return {x: b[0], y: b[1], ux: (b[0] - a[0]) / len, uy: (b[1] - a[1]) / len}; }""", scan_i)
        page.mouse.move(rect["x"] + end["x"], rect["y"] + end["y"]); page.mouse.down()
        page.mouse.move(rect["x"] + end["x"] + 25 * end["ux"] + 10, rect["y"] + end["y"] + 25 * end["uy"], steps=5)
        page.mouse.up()
        edits = page.evaluate("(i) => S[i].wall_edits", scan_i)
        assert len(edits) == 2 and edits[1]["op"] == "ends", f"dragging the end dot did not set the wall's ends: {edits}"
        assert page.locator("#rvWallRemove").is_enabled(), "Remove wall is greyed out with a wall picked"

        m = wall_mid(second if second < first else second)    # the moved wall stays at its index
        page.mouse.click(rect["x"] + m["x"], rect["y"] + m["y"])
        page.locator("#rvWallRemove").click()
        assert page.evaluate("(i) => effectiveWalls(S[i]).length", scan_i) == n0 - 1, "Remove wall did not remove the picked wall"
        assert page.locator("#rvWallRemove").is_disabled(), "Remove wall stays live with nothing picked"
        assert not page.locator("#rvWallTags").is_visible(), "the contains boxes show with no wall picked"

        k = next(k for k in range(n0 - 1) if page.evaluate(    # with a wall picked, tick what it contains
            "(p) => document.elementFromPoint(p.x, p.y) === cv", {"x": rect["x"] + wall_mid(k)["x"], "y": rect["y"] + wall_mid(k)["y"]}))
        m = wall_mid(k)
        page.mouse.click(rect["x"] + m["x"], rect["y"] + m["y"])
        page.locator('#rvWallTags [data-walltag="door"]').check()
        page.locator('#rvWallTags [data-walltag="window"]').check()
        assert page.evaluate("() => pickedWall().contains") == ["door", "window"], "the ticks did not reach the wall"
        page.locator('#rvWallTags [data-walltag="window"]').uncheck()
        assert page.evaluate("() => pickedWall().contains") == ["door"], "unticking did not take window off"
        page.evaluate("() => { review.wallSel = null; reviewRefreshBar(); draw(); }")   # pick nothing again

        spot = page.evaluate("""([i, rx, ry]) => {           // a clickable stretch of the view, off every wall
            for (let y = cv.height * 0.35; y < cv.height * 0.9; y += 20) for (let x = 60; x < cv.width - 200; x += 40) {
                const ok = [0, 130].every(dx => document.elementFromPoint(rx + x + dx, ry + y) === cv)
                    && wallHitPlan(S[i], x, y) === null;
                if (ok) return [x, y];
            } return null; }""", [scan_i, rect["x"], rect["y"]])
        assert spot, "no empty, clickable stretch of the view to draw a wall on"
        page.mouse.move(rect["x"] + spot[0], rect["y"] + spot[1]); page.mouse.down()
        page.mouse.move(rect["x"] + spot[0] + 130, rect["y"] + spot[1], steps=5); page.mouse.up()
        assert page.evaluate("(i) => effectiveWalls(S[i]).filter(w => w.source === 'hand').length", scan_i) == 1, \
            "a drag on empty space did not draw a wall"
        page.locator("#rvUndo").click()
        assert page.evaluate("(i) => effectiveWalls(S[i]).filter(w => w.source === 'hand').length", scan_i) == 0, "Undo kept the drawn wall"
        page.locator("#rvRedo").click()
        page.locator("#rvDone").click()

        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        _goto(page)
        shown = page.evaluate("(i) => effectiveWalls(S[i]).map(w => [w.x1, w.y1, w.x2, w.y2, w.source || '', (w.contains || []).join()])", scan_i)
        detected, saved_edits = page.evaluate("(i) => [S[i].walls, S[i].wall_edits]", scan_i)
        assert len(saved_edits) == 7, f"seven edits (three of them ticks) should survive the reload: {saved_edits}"
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from open_point_stitch.wall_checks import apply_wall_edits
        server, missed = apply_wall_edits(detected, saved_edits)
        server = [[w["x1"], w["y1"], w["x2"], w["y2"], w.get("source", ""), ",".join(w.get("contains", []))] for w in server]
        assert missed == 0 and len(server) == len(shown) and all(
            a[4:] == b[4:] and max(abs(x - y) for x, y in zip(a[:4], b[:4])) < 1e-6 for a, b in zip(server, shown)), (server, shown)
        from open_point_stitch.wall_checks import frozen                 # 0.29.0: edits frozen when the wall finder changes
        kept = frozen(detected, saved_edits, "test")
        page_kept = page.evaluate("(e) => applyWallEdits([{i: 0, x1: 0, y1: 0, x2: 1, y2: 0}], e)"
                                  ".map(w => [w.x1, w.y1, w.x2, w.y2, w.source || '', (w.contains || []).join()])", kept)
        assert len(page_kept) == len(server) and all(
            a[4:] == b[4:] and max(abs(x - y) for x, y in zip(a[:4], b[:4])) < 1e-3 for a, b in zip(server, page_kept)), \
            (server, page_kept)
        print(f"    walls: moved one {abs(edits[0]['by']) * 100:.0f} cm, set its ends, removed one, drew one (undone, redone); "
              f"after reload the server's rule gives the same {len(shown)} walls as the page")
    finally:
        with open(placement_path, "wb") as fh:
            fh.write(original)


def _restore_and_rebuild_plain(placement_path: str, original: bytes, name: str) -> None:
    """Put the project file back and have the server rebuild the scan as it was: re-processing changed the preview it
    holds in memory, which later tests on this bench would otherwise inherit."""
    with open(placement_path, "wb") as fh:
        fh.write(original)
    try:
        call(f"{BENCH_URL}/api/scans/{urllib.parse.quote(name)}/reprocess", "POST", timeout=300)
    finally:
        with open(placement_path, "wb") as fh:          # the rebuild records reprocessed_with: [] in the file
            fh.write(original)


def test_reprocess_a_scan_says_what_changed_and_rechecks_only_its_links(page):
    """After removing points, Re-process rebuilds the scan without them and says what changed; a reload keeps the
    re-processed scan; Re-check its links runs only pairs with this scan. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    name = None
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        name = page.evaluate("(i) => S[i].name", scan_i)
        page.locator('[data-view="plan"]').click()
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        full_before = page.evaluate("(i) => S[i].points", scan_i)
        rect = page.locator("#cv").bounding_box()
        cx0, cy0 = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
        page.locator('#reviewBar [data-rvtool="box"]').click()
        page.mouse.move(cx0 - 40, cy0 - 40); page.mouse.down(); page.mouse.move(cx0 + 40, cy0 + 40, steps=4); page.mouse.up()
        page.keyboard.press("Delete")

        page.locator("#rvReprocess").click()
        page.wait_for_function("() => /^re-processed:|^could not/.test($('rvInfo').textContent)", timeout=240_000)
        info = page.locator("#rvInfo").inner_text()
        assert info.startswith("re-processed:") and "points kept" in info and "walls:" in info, info
        full_after, with_ids = page.evaluate("(i) => [S[i].points, S[i].reprocessed_with]", scan_i)
        assert full_after < full_before and len(with_ids) == 1, (full_before, full_after, with_ids)
        assert "re-processed" in page.locator("#out").input_value()

        _goto(page)
        assert page.evaluate("(i) => S[i].points", scan_i) == full_after, "a reload went back to the plain scan"

        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        with page.expect_response(lambda r: r.url.endswith("/api/jobs") and r.request.method == "POST") as started:
            page.locator("#rvRecheck").click()
        job_id = started.value.json()["id"]
        # Read the pairs it works on from its progress, then cancel: running them all proves nothing more.
        n_scans, seen, total = page.evaluate("() => S.length"), set(), None
        deadline = time.time() + 300
        while time.time() < deadline and len(seen) < 2:
            job = call(f"{BENCH_URL}/api/jobs/{job_id}", "GET", timeout=10)
            assert job["state"] == "running", f"the re-check ended {job['state']}: {job.get('error')}"
            m = re.match(r"(\d+)/(\d+): (.+ <-> .+?)(?:: |$)", (job.get("progress") or {}).get("detail") or "")
            if m:
                total = int(m.group(2))
                seen.add(m.group(3))
            time.sleep(0.5)
        assert seen, "the re-check never reached its pairs"
        assert all(name in d.split(" <-> ") for d in seen), seen
        assert 1 <= total <= n_scans - 1, f"{total} pairs for one scan's links among {n_scans} scans"
        assert "re-checking the links of" in page.locator("#out").input_value()
        page.locator("#bCancelJob").click()
        page.wait_for_function("() => /cancelled/.test($('out').value)", timeout=120_000)
        print(f"    re-processed: {full_before:,} -> {full_after:,} points; '{info[:90]}...'; "
              f"re-check queued {total} pair(s), the ones running all with this scan; cancelled")
    finally:
        if name:
            _restore_and_rebuild_plain(placement_path, original, name)
        else:
            with open(placement_path, "wb") as fh:
                fh.write(original)


def test_stairs_marked_in_review_survive_a_reload_and_re_processing_flags_the_same_points(page):
    """Points marked as stairs show purple, stay in the scan and survive Save and reload; re-processed, the server flags
    the same preview points the page shows purple. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    name = None
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        name = page.evaluate("(i) => S[i].name", scan_i)
        page.locator('[data-view="plan"]').click()
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        drawn = page.evaluate("() => glr.nPoints")
        rect = page.locator("#cv").bounding_box()
        cx0, cy0 = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
        page.locator('#reviewBar [data-rvtool="box"]').click()
        page.mouse.move(cx0 - 30, cy0 - 30); page.mouse.down(); page.mouse.move(cx0 + 30, cy0 + 30, steps=4); page.mouse.up()
        n_sel = page.evaluate("() => review.sel.n")
        page.locator("#rvStairs").click()
        marked = page.evaluate("(i) => S[i]._stairs.reduce((a, v) => a + v, 0)", scan_i)
        assert marked == n_sel > 0 and page.evaluate("() => glr.nPoints") == drawn, (marked, n_sel, "stairs must stay drawn")
        assert page.evaluate("() => stairs.length") == 1 and page.evaluate("() => exclusions.length") == 0

        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        _goto(page)
        assert page.evaluate("() => stairs.length") == 1, "the stairs mark did not survive a reload"
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        page.locator("#rvReprocess").click()
        page.wait_for_function("() => /^re-processed:|^could not/.test($('rvInfo').textContent)", timeout=240_000)
        assert page.locator("#rvInfo").inner_text().startswith("re-processed:")
        server_idx = page.evaluate("(i) => S[i].stairs || []", scan_i)
        page_idx = page.evaluate("(i) => S[i]._stairs.reduce((a, v, k) => (v && a.push(k), a), [])", scan_i)
        differ = len(set(server_idx) ^ set(page_idx))
        assert server_idx and differ <= max(2, len(page_idx) // 500), (len(server_idx), len(page_idx), differ)
        print(f"    stairs: {n_sel} preview points marked, kept and drawn; after re-processing the server flags "
              f"{len(server_idx)}, the page {len(page_idx)} ({differ} differ)")
    finally:
        if name:
            _restore_and_rebuild_plain(placement_path, original, name)
        else:
            with open(placement_path, "wb") as fh:
                fh.write(original)


def test_review_count_next_scan_and_the_prompt_before_a_first_run(page):
    """The count of scans reviewed and "Review the next one"; Next scan in review; before a project's first run one
    prompt, whose "Review scans first" starts no run and whose "Don't ask again" lets the run go and is not asked again.
    Runs are intercepted, never started. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    try:
        data = json.loads(original)
        for k in ("last_run", "align_stamp", "last_align_pairs"):
            data.pop(k, None)
        for s in data["scans"]:
            s.pop("reviewed", None)
        with open(placement_path, "w") as fh:
            json.dump(data, fh)
        _goto(page)
        page.evaluate("() => { try { localStorage.removeItem(nudgeOffKey()); } catch (e) {} }")
        n = page.evaluate("() => S.filter(s => !s.deleted).length")
        assert page.locator("#reviewNudgeText").inner_text() == f"Reviewed 0 of {n} scans."

        page.locator("#reviewNudgeGo").click()
        first = page.evaluate("() => review && review.i")
        assert first is not None, "Review the next one did not open a scan"
        page.locator("#rvReviewed").check()
        page.locator("#rvNext").click()
        second = page.evaluate("() => review.i")
        assert second != first, "Next scan stayed on the same scan"
        page.locator("#rvDone").click()
        assert page.locator("#reviewNudgeText").inner_text() == f"Reviewed 1 of {n} scans."

        starts = []
        page.route("**/api/jobs", lambda route: (starts.append(1), route.abort()) if route.request.method == "POST"
                   else route.continue_())
        _pick_run(page, "align_auto_seeded")
        _press_run(page)
        page.wait_for_function("() => getComputedStyle($('reviewNudgeModal')).display !== 'none'")
        page.locator("#reviewNudgeReview").click()
        page.wait_for_timeout(500)
        assert not starts and page.evaluate("() => !!review"), "Review scans first should open review and start nothing"
        page.locator("#rvDone").click()

        _press_run(page)
        page.wait_for_function("() => getComputedStyle($('reviewNudgeModal')).display !== 'none'")
        page.locator("#reviewNudgeNever").click()
        page.wait_for_timeout(1500)
        assert len(starts) == 1, f"Don't ask again should let the run start, {len(starts)} start(s) seen"
        _press_run(page)
        page.wait_for_timeout(1500)
        assert page.evaluate("() => getComputedStyle($('reviewNudgeModal')).display") == "none", "asked again"
        assert len(starts) == 2, "the second run was not let through"
        page.unroute("**/api/jobs")
        print(f"    review count 0 then 1 of {n}; Next scan moved on; the prompt held one run back and let two through")
    finally:
        with open(placement_path, "wb") as fh:
            fh.write(original)


def test_review_selected_walks_through_exactly_the_picked_scans(page):
    """Select all then Review selected reviews every scan in list order; a hand-picked pair (click, Ctrl+click) reviews
    just those two and stops after the second."""
    _goto(page)
    page.wait_for_function("() => S && S.filter(s => !s.deleted).length >= 3 && S.every(s => s.deleted || s.ready)",
                           timeout=120_000)
    order = page.evaluate("() => [...document.querySelectorAll('#scanlist .scan')].map(r => +r.dataset.i)"
                          ".filter(i => !S[i].deleted)")
    assert page.locator("#selbar #sbReviewSelected").is_visible(), "Review selected is not on the scan list bar"

    page.locator("#sbSelectAll").click()
    page.locator("#sbReviewSelected").click()
    assert page.evaluate("() => review && review.i") == order[0], "review did not start on the first listed scan"
    assert f"(1 of {len(order)})" in page.locator("#rvName").inner_text()
    seen = [order[0]]
    for _ in order[1:]:
        page.locator("#rvNext").click()
        seen.append(page.evaluate("() => review.i"))
    assert seen == order, f"Next scan walked {seen}, the list order is {order}"
    page.locator("#rvNext").click()
    assert page.evaluate("() => review.i") == order[-1], "Next scan went past the last picked scan"
    assert "last of the" in page.locator("#out").input_value()
    page.locator("#rvDone").click()

    rows = page.locator("#scanlist .scan")
    page.locator("#sbSelectNone").click()
    pick = [order[2], order[0]]
    page.locator(f"#scanlist .scan[data-i='{pick[0]}']").click()
    page.locator(f"#scanlist .scan[data-i='{pick[1]}']").click(modifiers=["Control"])
    assert page.evaluate("() => selection.size") == 2, "Ctrl+click did not make a two-scan selection"
    page.locator("#sbReviewSelected").click()
    assert page.evaluate("() => review.i") == order[0], "a picked pair should start at the one higher in the list"
    page.locator("#rvNext").click()
    assert page.evaluate("() => review.i") == order[2]
    assert "(2 of 2)" in page.locator("#rvName").inner_text()
    page.locator("#rvNext").click()
    assert page.evaluate("() => review.i") == order[2], "a picked pair went on to a third scan"
    page.locator("#rvDone").click()
    assert rows.count() >= 3
    print(f"    all {len(order)} scans in list order, then a picked pair of 2 and a stop after it")


def test_context_menu_right_click(page):
    """Right-clicking empty space shows the view-level menu; right-clicking a scan in the 3D view selects it and shows the
    scan menu; its items (hide/show, lock/unlock, reset rotation, revert) have their documented effect."""
    _goto(page)
    page.locator('[data-view="plan"]').click()
    rect = _canvas_rect(page)

    def open_menu_on(scan_i):
        pt = _scan_point_screen_pos(page, scan_i)
        vx, vy = _to_viewport(rect, (pt["x"], pt["y"]))
        page.mouse.click(vx, vy, button="right")
        page.wait_for_function("() => !$('ctxMenu').hidden")

    def click_item(label):
        btn = page.locator("#ctxMenu button", has_text=label)
        assert btn.count() == 1, f"{label!r} item missing from the context menu"
        btn.click()
        page.wait_for_function("() => $('ctxMenu').hidden")

    # Empty space: found with the page's own hit() function rather than a
    # guessed pixel, so this does not depend on how 14 real scans happen to
    # be laid out on screen.
    empty = page.evaluate(
        """() => {
            const r = cv.getBoundingClientRect();
            for (let x = 0; x < r.width; x += 20)
                for (let y = 0; y < r.height; y += 20)
                    if (hit(x, y) === null) return {x, y};
            return null;
        }"""
    )
    assert empty is not None, "could not find an empty spot on the canvas to right-click"
    page.mouse.click(rect["x"] + empty["x"], rect["y"] + empty["y"], button="right")
    page.wait_for_function("() => !$('ctxMenu').hidden")
    assert page.locator("#ctxMenu button").count() > 0, "empty-space context menu had no items"
    page.keyboard.press("Escape")
    page.wait_for_function("() => $('ctxMenu').hidden")

    # Right-click an unselected scan: it must be selected and get the menu. Every other scan is hidden first so hit()
    # cannot pick a closer point of theirs.
    target_i = _select_scan_row(page, 1)
    other_i = _select_scan_row(page, 0)
    assert other_i != target_i, "setup: need two distinct scans to prove the right-click re-selects"
    page.evaluate(
        "(i) => { S.forEach(s => { if (!s.deleted) s.vis = (s.i === i); }); glRebuildPoints(); }",
        target_i,
    )
    open_menu_on(target_i)
    sel_after = page.evaluate("() => sel")
    assert sel_after == target_i, f"right-clicking scan {target_i} did not select it (sel={sel_after})"

    click_item("Isolate")
    visible = page.evaluate("() => S.filter(s => !s.deleted && s.vis).map(s => s.i)")
    assert visible == [target_i], f"Isolate should leave only scan {target_i} visible, got {visible}"

    # Hide/Show (mBatchVis) and Lock/Unlock (mBatchLock), through the menu.
    was_vis = page.evaluate("(i) => S.find(s => s.i === i).vis", target_i)
    open_menu_on(target_i)
    click_item("Hide / Show")
    now_vis = page.evaluate("(i) => S.find(s => s.i === i).vis", target_i)
    assert now_vis != was_vis, "Hide / Show through the context menu did not toggle visibility"
    # Hide/Show just hid the scan, so show it again directly (setup, not under test) to right-click it below.
    page.evaluate(
        "(i) => { const s = S.find(x => x.i === i); s.vis = true; glRebuildPoints(); }",
        target_i,
    )

    was_locked = page.evaluate("(i) => S.find(s => s.i === i).lock", target_i)
    open_menu_on(target_i)
    click_item("Lock / Unlock")
    now_locked = page.evaluate("(i) => S.find(s => s.i === i).lock", target_i)
    assert now_locked != was_locked, "Lock / Unlock through the context menu did not toggle lock"
    if now_locked:
        # Undo the lock so the move/reset checks below are not refused by it.
        open_menu_on(target_i)
        click_item("Lock / Unlock")

    # Reset rotation: move and tilt the scan via setPose, then check the menu's Reset rotation zeros yaw/pitch/roll.
    page.evaluate(
        "(i) => { const s = S.find(x => x.i === i); setPose(s, {yaw: s.yaw + 0.3, x: s.x + 1.0}); }",
        target_i,
    )
    moved = _pose(page, target_i)
    centre_before = page.evaluate(
        "(i) => { const s = S.find(x => x.i === i);"
        "  return worldCentre3(localCentre3(s), s.yaw, s.pitch, s.roll, s.x, s.y, s.dz); }",
        target_i,
    )
    open_menu_on(target_i)
    click_item("Reset rotation")
    after_reset = _pose(page, target_i)
    assert abs(after_reset["yaw"]) < 1e-9, f"Reset rotation through the menu left yaw={after_reset['yaw']}"
    # A scan turns about its own centre, so the centre is preserved, not the stored x/y.
    centre_after = page.evaluate(
        "(i) => { const s = S.find(x => x.i === i);"
        "  return worldCentre3(localCentre3(s), s.yaw, s.pitch, s.roll, s.x, s.y, s.dz); }",
        target_i,
    )
    drift = max(abs(a - b) for a, b in zip(centre_before, centre_after))
    assert drift < 1e-6, (
        f"Reset rotation moved the scan's own centre by {drift:.6f}m - it should turn about it"
    )

    # Simulate an alignment landing by writing the pose fields directly: setPose would re-stamp the hand placement.
    placed = _pose(page, target_i)
    page.evaluate(
        "(i) => { const s = S.find(x => x.i === i); s.humanX = s.x; s.humanY = s.y; "
        "s.humanYaw = s.yaw; s.humanDz = s.dz; s.humanPitch = s.pitch; s.humanRoll = s.roll; }",
        target_i,
    )
    page.evaluate(
        "(i) => { const s = S.find(x => x.i === i); s.yaw += 0.4; s.x += 2.0; }",
        target_i,
    )
    algo_moved = _pose(page, target_i)
    assert abs(algo_moved["x"] - placed["x"] - 2.0) < 1e-6, "setup: the simulated align move did not take"

    open_menu_on(target_i)
    click_item("Revert to my placement")
    after_revert = _pose(page, target_i)
    assert after_revert == placed, (
        f"Revert to my placement through the menu did not restore the pre-align pose: "
        f"{after_revert} != {placed}"
    )
    print(f"    right-click menu: empty-space items, selects on click, Isolate/Hide/Lock/Reset/Revert all wired")


def test_as_imported_view_round_trip(page):
    """A read-only view draws every scan at its as-imported pose, then restores every scan exactly, bit for bit. Moves a
    scan by hand first so the round trip is tested against a real non-zero pose, and compares against the recorded
    asImported table."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    as_imported = page.evaluate(
        "() => S.map(s => { const r = asImported[s.name]; "
        "return r ? {x:r.x_m, y:r.y_m, yaw:r.yaw_deg*Math.PI/180, dz:r.z_m, "
        "pitch:r.pitch_deg*Math.PI/180, roll:r.roll_deg*Math.PI/180} : null; })"
    )
    assert all(r is not None for r in as_imported), (
        f"every scan should have a recorded as-imported pose by now: {as_imported}"
    )
    _type_x(page, "2.71828")
    before = page.evaluate(
        "() => S.map(s => ({x:s.x,y:s.y,yaw:s.yaw,dz:s.dz,pitch:s.pitch,roll:s.roll}))"
    )
    assert abs(before[0]["x"] - 2.71828) < 1e-4, "setup: typing into the x box did not move the scan"

    page.locator('.menubtn[data-menu="menuView"]').click()
    page.locator("#cbAsImported").click()
    page.wait_for_function("() => viewAsImported === true")

    during = page.evaluate(
        "() => S.map(s => ({x:s.x,y:s.y,yaw:s.yaw,dz:s.dz,pitch:s.pitch,roll:s.roll}))"
    )
    for p, rec in zip(during, as_imported):
        assert (abs(p["x"] - rec["x"]) < 1e-9 and abs(p["y"] - rec["y"]) < 1e-9
                and abs(p["yaw"] - rec["yaw"]) < 1e-9 and abs(p["dz"] - rec["dz"]) < 1e-9
                and abs(p["pitch"] - rec["pitch"]) < 1e-9 and abs(p["roll"] - rec["roll"]) < 1e-9), (
            f"the as-imported view did not draw this scan at its recorded grid slot: "
            f"drawn={p} recorded={rec}"
        )

    state_text = page.locator("#stateLine").inner_text()
    assert "As imported" in state_text, f"the state label did not say 'As imported' during the view: {state_text!r}"
    assert page.locator("#bSave").is_disabled(), "Save must be disabled while the read-only view is up"

    page.locator("#cbAsImported").click()
    page.wait_for_function("() => viewAsImported === false")

    after = page.evaluate(
        "() => S.map(s => ({x:s.x,y:s.y,yaw:s.yaw,dz:s.dz,pitch:s.pitch,roll:s.roll}))"
    )
    assert after == before, f"round trip was not bit-identical:\nbefore={before}\nafter={after}"
    assert not page.locator("#bSave").is_disabled(), "Save must be re-enabled once the view turns back off"
    print(f"    {len(before)} scan(s): as-imported view drew every one at its recorded grid slot, then restored exactly")


def test_revert_to_as_imported(page):
    """A per-scan revert all the way back to the as-imported pose (the scan's grid slot), distinct from "revert to my
    placement", which only undoes an automatic align."""
    _goto(page)
    scan_i = _select_scan_row(page, 0)
    expected = page.evaluate(
        "(i) => { const r = asImported[S[i].name]; "
        "return {x:r.x_m, y:r.y_m, yaw:r.yaw_deg*Math.PI/180, dz:r.z_m, "
        "pitch:r.pitch_deg*Math.PI/180, roll:r.roll_deg*Math.PI/180}; }",
        scan_i,
    )
    _type_x(page, "1.5")
    moved = _pose(page, scan_i)
    assert abs(moved["x"] - 1.5) < 1e-4, "setup: typing into the x box did not move the scan"

    btn = page.locator("#bRevertAsImported")
    assert btn.is_visible(), "the revert-to-as-imported button did not appear after moving the scan"
    btn.click()

    reverted = _pose(page, scan_i)
    assert all(abs(reverted[k] - expected[k]) < 1e-9 for k in expected), (
        f"revert to as imported did not return to the recorded grid slot: "
        f"reverted={reverted} expected={expected}"
    )
    print(f"    moved scan to x=1.5, 'revert to as imported' brought it back to its grid slot {reverted}")


def _tilt_state(page):
    return page.evaluate("() => S.map(s => ({i: s.i, x: s.x, y: s.y, dz: s.dz, pitch: s.pitch, roll: s.roll, hp: s.humanPitch, hr: s.humanRoll}))")


def _open_tilt_dialog(page):
    _pick_run(page, "tilt")
    _press_run(page)
    try:
        page.locator("#tiltModal").wait_for(state="visible", timeout=90_000)
    except Exception as exc:  # noqa: BLE001 - say what the page itself said
        raise AssertionError(f"the tilt dialog did not open; the page's status line says: {page.locator('#out').input_value()!r}") from exc


def test_tilt_suggestion_can_be_cancelled_applied_and_undone(page):
    """Run > Suggest tilt and height: nothing moves until Apply, Undo restores it, hand tilt stays part of the placement,
    and applying leaves nothing more to suggest."""
    _goto(page)
    before = _tilt_state(page)
    _open_tilt_dialog(page)
    text = page.locator("#tiltModalBody").inner_text()
    assert "Floors of overlapping scans" in text and "within 5 cm" in text, text
    assert page.locator("#tiltModalBody tbody tr").count() >= 2, "the dialog lists no scans"
    assert _tilt_state(page) == before, "the dialog opened but a scan had already moved"
    page.locator("#tiltCancel").click()
    assert not page.locator("#tiltModal").is_visible() and _tilt_state(page) == before, "Cancel moved something"

    _open_tilt_dialog(page)
    page.locator("#tiltApply").click()
    after = _tilt_state(page)
    tipped = [a for a, b in zip(after, before) if abs(a["pitch"] - b["pitch"]) > 0.0005 or abs(a["roll"] - b["roll"]) > 0.0005]
    assert len(tipped) >= 2, f"Apply tipped only {len(tipped)} scan(s)"
    assert all(a["hp"] == b["hp"] and a["hr"] == b["hr"] for a, b in zip(after, before)), "Apply overwrote the hand tilt"

    again = page.evaluate("""async () => { const r = await fetch('/api/tilt-suggestion', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({scans: buildPlacementPayload().scans})});
        return await r.json(); }""")
    worst = max(max(abs(s["d_pitch_deg"]), abs(s["d_roll_deg"])) for s in again["suggestions"])
    first = max(max(abs(a["pitch"] - b["pitch"]), abs(a["roll"] - b["roll"])) for a, b in zip(after, before)) * 180 / math.pi
    movers = sorted(again["suggestions"], key=lambda s: -max(abs(s["d_pitch_deg"]), abs(s["d_roll_deg"])))[:3]
    detail = [(m["scan"][-18:], m["d_pitch_deg"], m["d_roll_deg"], m["cells"], m["partners"]) for m in movers]
    assert worst < 0.5 and worst < 0.25 * first, (f"after Apply the solve still wants {worst:.2f} degrees more (first pass moved {first:.2f}): "
                                                  f"units or conversion are off. Second pass movers {detail}, left alone {list(again.get('skipped', {}))}")

    page.locator("#bUndo").click()
    assert _tilt_state(page) == before, "Undo did not restore the poses from before Apply"
    print(f"    dialog shown, Cancel moved nothing, Apply tipped {len(tipped)} scans by up to {first:.2f} degrees, re-suggesting wanted at most {worst:.2f}, Undo restored all")


def test_backup_list_and_restore_work_from_the_file_menu(page):
    """File > Restore a backup lists backups in backups/ and restores one."""
    _goto(page)
    page.on("dialog", lambda d: d.accept())
    name, x0 = page.evaluate("() => { const s = S.find(q => !q.lock && q.ready && !q.deleted); return [s.name, s.x]; }")

    def save_at(x):
        page.evaluate("([n, x]) => setPose(S.find(q => q.name === n), {x})", [name, x])
        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()

    save_at(x0 + 1.0)
    save_at(x0 + 2.0)     # this save backs up the x0 + 1 layout
    page.locator('.menubtn[data-menu="menuFile"]').click()
    page.locator("#mRestoreBackup").click()
    page.locator("#backupsList [data-restore]").first.wait_for(timeout=15_000)
    backup_id = page.locator("#backupsList [data-restore]").first.get_attribute("data-restore")
    page.locator("#backupsList [data-restore]").first.click()
    page.wait_for_function("() => /restored the backup|RESTORE FAILED/.test($('out').value)", timeout=30_000)
    assert "restored the backup" in page.locator("#out").input_value()
    x = page.evaluate("(n) => S.find(q => q.name === n).x", name)
    assert abs(x - (x0 + 1.0)) < 1e-6, f"restored x {x}, expected the backed-up {x0 + 1.0}"
    if PROJECT_DIR:
        assert os.path.isfile(os.path.join(PROJECT_DIR, "backups", backup_id)), f"{backup_id} is not in backups/"
    save_at(x0)           # leave the shared QA project as it was
    print(f"    restored {backup_id} from backups/: {name} back at x {x:.3f} m")


def test_saved_versions_can_be_pinned_and_tidied_from_the_restore_list(page):
    """Pin keeps a version through a tidy; Tidy up shows what goes before removing it."""
    if not PROJECT_DIR:
        print("    SKIP: needs the QA project folder to plant old saved versions")
        return
    _goto(page)                          # opening tidies automatically, so plant the old versions after
    folder = os.path.join(PROJECT_DIR, "backups")
    os.makedirs(folder, exist_ok=True)
    old = [f"placement.backup-20260101T10{m}00.000Z.json" for m in ("00", "10", "20")]   # one old day
    for name in old:
        with open(os.path.join(folder, name), "w") as fh:
            fh.write('{"schema": "x"}')
    page.locator('.menubtn[data-menu="menuFile"]').click()
    page.locator("#mRestoreBackup").click()
    pin = page.locator(f'#backupsList [data-pin="{old[0]}"]')
    pin.wait_for(timeout=15_000)
    pin.click()
    page.locator(f'#backupsList [data-pin="{old[0]}"][data-pinned="1"]').wait_for(timeout=10_000)
    page.locator("#backupsTidy").click()
    page.locator("#backupsTidyPreview").wait_for(state="visible", timeout=30_000)
    preview = page.locator("#backupsTidyPreview").inner_text()
    assert "Tidying would remove" in preview and "1 pinned" in preview, preview
    assert all(os.path.exists(os.path.join(folder, n)) for n in old), "the preview removed something"
    page.locator("#backupsTidyConfirm").click()
    page.wait_for_function("() => /removed \\d+ older saved versions|TIDY FAILED/.test($('out').value)", timeout=60_000)
    assert "removed" in page.locator("#out").input_value(), page.locator("#out").input_value()
    left = [n for n in old if os.path.exists(os.path.join(folder, n))]
    assert left == [old[0], old[2]], f"expected the pinned one and that day's newest to stay, got {left}"
    page.locator(f'#backupsList [data-pin="{old[0]}"]').click()     # unpin, then leave the project as it was
    page.locator(f'#backupsList [data-pin="{old[0]}"][data-pinned="0"]').wait_for(timeout=10_000)
    page.locator("#backupsCancel").click()
    for n in left:
        os.remove(os.path.join(folder, n))
    print(f"    pinned {old[0]}, tidy removed {old[1]}, kept the pinned one and {old[2]}")


def test_saved_versions_settings_survive_a_save_and_a_reload(page):
    """The Saved versions tab reaches the server, and a Save keeps the project's settings."""
    def saved_settings(**want) -> dict:
        return _saved_settings(page, **want)

    def open_tab():
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Saved versions")').click()

    _goto(page)
    open_tab()
    try:
        page.locator("#cbAutoTidy").uncheck()
        page.select_option("#backupKeepAll", "3")
        s = saved_settings(backup_auto_tidy=False, backup_keep_all_hours=3)
        assert s.get("backup_auto_tidy") is False and s.get("backup_keep_all_hours") == 3, s
        page.keyboard.press("Escape")
        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        assert saved_settings(backup_auto_tidy=False).get("backup_auto_tidy") is False, "Save dropped the project's settings"
        _goto(page)
        open_tab()
        assert not page.locator("#cbAutoTidy").is_checked() and page.locator("#backupKeepAll").input_value() == "3", \
            "the saved-version settings did not come back after a Save and a reload"
    finally:
        page.locator("#cbAutoTidy").check()
        page.select_option("#backupKeepAll", "1")
        s = saved_settings(backup_auto_tidy=True, backup_keep_all_hours=1)
        assert s.get("backup_auto_tidy") is True and s.get("backup_keep_all_hours") == 1, s
    print("    auto tidy off and 3 hours reached the server, survived a Save and a reload, then put back")


def test_plan_view_export_from_the_file_menu_writes_a_dxf(page):
    """File > Export plan view (DXF) cuts the placed scans and writes a DXF and a picture into the project. The band is
    the Z clip when it is on."""
    _goto(page)
    assert "not built yet" not in page.locator("#menuFile").inner_text(), "the File menu still says the export is not built"
    page.locator('.menubtn[data-menu="menuFile"]').click()
    page.locator("#mExportPlan").click()
    page.wait_for_function("() => /PLAN VIEW EXPORTED|PLAN EXPORT FAILED/.test($('out').value)", timeout=280_000)
    text = page.locator("#out").input_value()
    assert "PLAN VIEW EXPORTED" in text, text
    assert "1.0 to 1.6 m above the typical floor" in text, "with no Z clip the default band should be said"
    if PROJECT_DIR:
        # Exports go in exports/ with a time in the name; latest.json names the newest.
        exports = os.path.join(PROJECT_DIR, "exports")
        with open(os.path.join(exports, "latest.json")) as fh:
            latest = json.load(fh)
        dxf = os.path.join(exports, latest["plan_view_dxf"])
        assert latest["plan_view_dxf"].startswith("plan_view-"), latest
        with open(dxf, encoding="ascii") as fh:
            body = fh.read()
        assert "AC1009" in body and "PLAN_CUT" in body and body.rstrip().endswith("EOF"), "the DXF is not a whole R12 file"
        assert body.count("POLYLINE") >= 4, f"only {body.count('POLYLINE')} outline markers in the DXF"
        assert os.path.getsize(os.path.join(exports, latest["plan_view_png"])) > 2000
        assert not os.path.exists(os.path.join(PROJECT_DIR, "plan_view.dxf")), "an export landed at the top level"
    print("    " + [l for l in text.splitlines() if "PLAN VIEW EXPORTED" in l][0].strip())


def test_floors_ceilings_and_walls_export_in_3d_from_the_run_window(page):
    """Run > Export floors, ceilings and walls in 3D writes a DXF whose surfaces are closed 3D polylines on layers by
    kind, and a picture, into exports/, and the log says what it holds. Read from the file written."""
    _goto(page)
    _pick_run(page, "surfaces_3d")
    _press_run(page)
    page.wait_for_function("() => /3D EXPORTED|3D EXPORT FAILED/.test($('out').value)", timeout=280_000)
    text = page.locator("#out").input_value()
    assert "3D EXPORTED" in text, text[-600:]
    line = [l for l in text.splitlines() if "3D EXPORTED" in l][-1]
    if PROJECT_DIR:
        exports = os.path.join(PROJECT_DIR, "exports")
        with open(os.path.join(exports, "latest.json")) as fh:
            latest = json.load(fh)
        assert latest["surfaces_3d_dxf"].startswith("surfaces_3d-"), latest
        with open(os.path.join(exports, latest["surfaces_3d_dxf"]), encoding="ascii") as fh:
            body = fh.read().splitlines()
        assert body[-1].strip() == "EOF" and "AC1009" in body, "the DXF is not a whole R12 file"
        layers = {body[k + 1].strip() for k in range(len(body) - 1) if body[k].strip() == "8"}
        assert {"FLOOR", "CEILING", "WALLS", "LABELS"} <= layers, layers
        flags = [body[k + 1].strip() for k in range(len(body) - 1) if body[k].strip() == "70"]
        assert flags.count("9") >= 10 and flags.count("32") >= 40, "too few closed 3D outlines in the DXF"
        heights = {round(float(body[k + 1]), 1) for k in range(len(body) - 1) if body[k].strip() == "30"}
        assert len(heights) > 5, f"the outlines are not 3D: heights {sorted(heights)[:10]}"
        assert os.path.getsize(os.path.join(exports, latest["surfaces_3d_png"])) > 2000
    print("    " + line.strip())


def test_settings_by_stage_save_reload_and_reach_the_server(page):
    """Settings are grouped by pipeline stage, and the levelling limit, range limit and stray-point removal persist per
    project and reach the server. Defaults are put back afterwards.

    The stages are separate tab panels; text_content() confirms all three exist at once, since inactive panels are
    display:none. Controls on a tab are only used after that tab is clicked active."""
    _goto(page)
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    text = page.locator('[data-menu-modal="menuSettings"]').text_content().lower()
    for heading in ("preparing scans", "aligning", "exporting"):
        assert heading in text, f"the Settings modal has no {heading!r} tab"
    def saved_settings(**want) -> dict:
        return _saved_settings(page, **want)

    page.locator('button[role="tab"]:has-text("Preparing scans")').click()
    # Never leave the levelling limit changed across a project reopen: that rebuilds every preview.
    try:
        page.select_option("#levelTilt", "3")
        assert saved_settings(level_tilt_deg=3).get("level_tilt_deg") == 3, "the levelling limit did not reach the server"
        assert page.evaluate("async () => { await loadSettingsAndWireSave(); return $('levelTilt').value; }") == "3", \
            "the saved levelling limit was not read back into the menu"
    finally:
        page.select_option("#levelTilt", "15")
        assert saved_settings(level_tilt_deg=15).get("level_tilt_deg") == 15, "the levelling limit was not put back"
    # Wall search: the same rule.
    try:
        page.select_option("#wallSearch", "thorough")
        assert saved_settings(wall_search="thorough").get("wall_search") == "thorough", "Wall search did not reach the server"
        assert page.evaluate("async () => { await loadSettingsAndWireSave(); return $('wallSearch').value; }") == \
            "thorough", "the saved Wall search was not read back into the menu"
    finally:
        page.select_option("#wallSearch", "standard")
        assert saved_settings(wall_search="standard").get("wall_search") == "standard", "Wall search was not put back"

    page.select_option("#rangeLimit", "6")
    page.locator('button[role="tab"]:has-text("Exporting")').click()
    page.select_option("#strayRemoval", "2.0")
    saved = saved_settings(range_limit_m="6", stray_std_ratio="2.0")
    assert str(saved.get("range_limit_m")) == "6" and str(saved.get("stray_std_ratio")) == "2.0", saved
    try:
        _goto(page)
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Preparing scans")').click()
        assert page.locator("#rangeLimit").input_value() == "6" and page.locator("#levelTilt").input_value() == "15", (
            "the settings did not come back after a reload"
        )
        page.locator('button[role="tab"]:has-text("Exporting")').click()
        assert page.locator("#strayRemoval").input_value() == "2.0", "the settings did not come back after a reload"
    finally:
        page.select_option("#strayRemoval", "")   # already on the Exporting tab
        page.locator('button[role="tab"]:has-text("Preparing scans")').click()
        page.select_option("#rangeLimit", "")
        saved_settings(range_limit_m="", stray_std_ratio="")
    print("    three settings saved and reached the server, two survived a reload, the levelling limit was read back, defaults restored")


def test_one_end_of_an_anchor_can_be_redrawn_without_dropping_it(page):
    """A saved link's end is redrawn with a new box; the other end and the link stay, Undo brings the old end back, and
    a box on the wrong scan is refused. Puts the project file back."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    kept = _keep_project_file()
    try:
        _goto(page)
        rows = page.locator("#scanlist .scan")
        i0, i1 = int(rows.nth(0).get_attribute("data-i")), int(rows.nth(1).get_attribute("data-i"))
        page.locator('[data-view="plan"]').click()
        page.locator('[data-tool="anchor"]').click()
        rows.nth(0).click()
        _drag_finish_anchor_region(page, i0)
        rows.nth(1).click()
        _drag_finish_anchor_region(page, i1)
        page.locator("#bFinishAnchor").click()
        n = page.evaluate("() => anchors.length")
        before = page.evaluate("() => JSON.stringify(anchors[anchors.length - 1].members.map(m => [m.scan, m.centroid]))")
        last = n - 1

        page.locator(f'[data-redraw="{last}:1"]').click()
        assert page.evaluate("() => redrawTarget !== null"), "clicking redraw did not arm a redraw"
        rows.nth(0).click()                                    # the wrong scan: refused, still waiting
        _drag_finish_anchor_region(page, i0)
        assert page.evaluate("() => redrawTarget !== null"), "a box on the wrong scan ended the redraw"
        rows.nth(1).click()
        pt = _scan_point_screen_pos(page, i1, 0.2)
        vx, vy = _to_viewport(_canvas_rect(page), (pt["x"], pt["y"]))
        page.mouse.move(vx - 12, vy - 12)
        page.mouse.down()
        page.mouse.move(vx + 12, vy + 12, steps=4)
        page.mouse.up()
        after = page.evaluate("() => JSON.stringify(anchors[anchors.length - 1].members.map(m => [m.scan, m.centroid]))")
        assert page.evaluate("() => anchors.length") == n and page.evaluate("() => redrawTarget === null"), "the redraw did not finish in place"
        old, new = json.loads(before), json.loads(after)
        assert old[0] == new[0], "redrawing one end changed the other"
        assert old[1][0] == new[1][0] and old[1][1] != new[1][1], "the redrawn end kept its old region"

        page.locator("#bUndo").click()
        assert page.evaluate("() => JSON.stringify(anchors[anchors.length - 1].members.map(m => [m.scan, m.centroid]))") == before, "Undo did not bring the old end back"
        print("    redrew one end in place, the other end stayed, a wrong-scan box was refused, Undo restored the old end")
    finally:
        _put_back(page, kept)


def test_wall_thickness_report_from_the_file_menu(page):
    """File > Check wall thickness (inches): a plain summary and a line per wall naming the scans and how far each is
    off, in inches."""
    _goto(page)
    page.locator('.menubtn[data-menu="menuFile"]').click()
    page.locator("#mWallReport").click()
    page.wait_for_function("() => /WALL THICKNESS|no wall was found|WALL CHECK FAILED/.test($('out').value)", timeout=280_000)
    text = page.locator("#out").input_value()
    assert "WALL CHECK FAILED" not in text, text
    assert "WALL THICKNESS:" in text and " in thick" in text and "scans:" in text, text
    if PROJECT_DIR:
        with open(os.path.join(PROJECT_DIR, "reports", "wall_thickness.json")) as fh:  # reports/
            saved = json.load(fh)
        assert saved["summary"]["walls"] >= 3 and saved["walls"][0]["thickness_in"] >= saved["walls"][-1]["thickness_in"], saved["summary"]
        # The bent-scan check ran on the same placed scans without error (the QA layout may share no wall).
        bent = saved.get("bent_scans")
        assert bent is not None and "error" not in bent and isinstance(bent.get("flags"), list), bent
        print(f"    bent-scan check: {bent['checked']} scans checked, {len(bent['flags'])} flagged")
    print("    " + [l for l in text.splitlines() if "WALL THICKNESS:" in l][0].strip()[:200])


def _contrast_ratio(fg, bg):
    """WCAG contrast ratio between two [r, g, b] colours (1 = identical, 21 = black on white)."""
    def lum(c):
        r, g, b = [v / 255 for v in c[:3]]
        f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


_TEXT_AND_BACKGROUND_JS = """(sel) => {
    const e = document.querySelector(sel); if (!e) return null;
    const rgb = s => (s.match(/[\\d.]+/g) || []).map(Number);
    let bg = null;
    for (let n = e; n; n = n.parentElement) {
        const c = rgb(getComputedStyle(n).backgroundColor);
        if (c.length >= 3 && (c.length < 4 || c[3] > 0.5)) { bg = c.slice(0, 3); break; }
    }
    return {fg: rgb(getComputedStyle(e).color).slice(0, 3), bg, disabled: !!e.disabled};
}"""


def test_a_theme_can_be_chosen_stays_readable_persists_and_dark_comes_back(page):
    """Settings > Appearance > Colours recolours the panels and menus, main text stays readable in every theme, the
    choice is remembered, and the viewport background follows the theme."""
    _goto(page)
    readable = ('.menubtn[data-menu="menuFile"]', "#scanlist .scan", "#scanlist .scan button", "#links button", "#themeSelect")
    try:
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Appearance")').click()
        worst = (99.0, "")
        for name, canvas in (("light", "#0b0e12"), ("solarized", "#001f27"), ("synthwave", "#0e0720")):
            page.select_option("#themeSelect", name)
            assert page.evaluate("() => document.documentElement.dataset.theme") == name, f"{name} was not applied"
            assert page.evaluate("() => themeCanvas.hex") == canvas, f"{name}: the viewport did not take the theme's canvas"
            for sel in readable:
                got = page.evaluate(_TEXT_AND_BACKGROUND_JS, sel)
                if not got or not got["bg"] or got["disabled"]:       # a disabled control is meant to look faint
                    continue
                ratio = _contrast_ratio(got["fg"], got["bg"])
                worst = min(worst, (ratio, f"{name} {sel}"))
                assert ratio >= 4.5, f"{name}: {sel} text {got['fg']} on {got['bg']} is only {ratio:.1f}:1, hard to read"
        page.select_option("#themeSelect", "light")
        _goto(page)
        assert page.evaluate("() => document.documentElement.dataset.theme") == "light", "the theme did not survive a reload"
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Appearance")').click()
        assert page.locator("#themeSelect").input_value() == "light", "the selector does not show the remembered theme"
        page.select_option("#themeSelect", "dark")
        assert page.evaluate("() => document.documentElement.dataset.theme") is None, "dark should be the page's own colours"
        assert page.evaluate("() => themeCanvas.hex") == "#0b0e12"
    finally:
        page.evaluate("() => applyTheme('dark')")
    print(f"    three themes readable (lowest contrast {worst[0]:.1f}:1 at {worst[1]}), remembered across a reload, dark restored")


def test_depth_cue_dims_far_points_in_the_angled_view_without_moving_anything(page):
    """View > Depth cue fades points farther from the viewer; it is display only, so no scan moves and turning it off
    restores the picture."""
    _goto(page)
    page.locator('[data-view="iso"]').click()
    brightness = """() => {
        draw();
        const off = document.createElement('canvas'); off.width = cvgl.width; off.height = cvgl.height;
        const o = off.getContext('2d'); o.drawImage(cvgl, 0, 0);
        const d = o.getImageData(0, 0, off.width, off.height).data;
        const [br, bg, bb] = themeCanvas.rgb.map(v => v * 255);
        let sum = 0;                                             // brightness of the points only, not of the background
        for (let i = 0; i < d.length; i += 4) sum += Math.max(0, d[i] - br) + Math.max(0, d[i + 1] - bg) + Math.max(0, d[i + 2] - bb);
        return sum;
    }"""
    assert not page.evaluate("() => $('cbDepthCue').checked"), "depth cue should start off"
    # With the glow on, overlapped points add up to white and hide the fade; the cue is judged with it off.
    glow_was_on = page.evaluate("() => { const on = $('cbAdd').checked; $('cbAdd').checked = false; return on; }")
    pose_before = _pose(page, 0)
    plain = page.evaluate(brightness)
    page.locator('.menubtn[data-menu="menuView"]').click()
    page.locator("#cbDepthCue").click(force=True)
    try:
        assert page.evaluate("() => $('cbDepthCue').checked"), "the switch did not turn on"
        cued = page.evaluate(brightness)
        assert plain > 100_000, f"no points drawn to judge (brightness {plain})"
        assert cued < plain * 0.95, f"depth cue changed nothing visible (point brightness {plain} -> {cued})"
        assert _pose(page, 0) == pose_before, "a display setting moved a scan"
    finally:
        if page.evaluate("() => $('cbDepthCue').checked"):
            page.locator("#cbDepthCue").click(force=True)
    assert abs(page.evaluate(brightness) - plain) <= plain * 0.01, "turning depth cue off did not restore the picture"
    page.evaluate("(on) => { $('cbAdd').checked = on; draw(); }", glow_was_on)
    print(f"    brightness {plain} -> {cued} with depth cue on ({(1 - cued / plain) * 100:.0f}% dimmer), back to normal when off")


def test_floors_and_ceilings_show_as_pieces_from_the_view_menu(page):
    """View > Floors and ceilings asks the server for the selected scan's pieces, draws each on its own plane with its
    height, lists them in the log, and a second ask is served from the cache."""
    _goto(page)
    page.wait_for_function("() => S.length && S[0].ready", timeout=60_000)
    page.evaluate("() => { selectScan(S[0].i, false, false); draw(); }")
    ink = """() => { draw(); const d = cx.getImageData(0, 0, cv.width, cv.height).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n; }"""
    page.locator('[data-view="plan"]').click()
    before = page.evaluate(ink)
    page.locator('.menubtn[data-menu="menuView"]').click()
    page.locator("#cbFloorsCeil").click(force=True)
    try:
        page.wait_for_function("() => S[0].fcState === 'ready' || S[0].fcState === 'error'", timeout=50_000)
        pieces = page.evaluate("() => S[0].fc")
        assert page.evaluate("() => S[0].fcState") == "ready", page.locator("#out").input_value()[-400:]
        assert pieces and all(p["label"] and p["outlines"] and p["kind"] in ("floor", "ceiling", "surface")
                              for p in pieces), pieces
        assert "floor and ceiling piece" in page.locator("#out").input_value()
        after = page.evaluate(ink)
        assert after > before + 500, f"no pieces drawn ({before} -> {after} inked pixels)"
        t0 = time.time()
        name = page.evaluate("() => S[0].name")
        again = call(f"{BENCH_URL}/api/scans/{urllib.parse.quote(name)}/floors-ceilings", "GET", timeout=30)["pieces"]
        assert again == pieces and time.time() - t0 < 3, "the second ask was not the cached answer"
    finally:
        if page.evaluate("() => $('cbFloorsCeil').checked"):
            page.locator("#cbFloorsCeil").click(force=True)
    print(f"    {len(pieces)} pieces on {name}: " + "; ".join(p["label"] for p in pieces[:6]))


def test_align_from_links_says_what_each_scan_was_matched_on(page):
    """Align from links matches the points inside the boxes, and the status log says so per scan (or why it fell back
    to the whole scan). Builds a real two-scan link by mouse, presses Run, and reads the log. Align writes poses."""
    _goto(page)
    rows = page.locator("#scanlist .scan")
    assert rows.count() >= 2, "need at least 2 scans to build an anchor"
    i0, i1 = (int(rows.nth(k).get_attribute("data-i")) for k in (0, 1))
    page.locator('[data-view="plan"]').click()
    page.locator('[data-tool="anchor"]').click()
    rows.nth(0).click()
    _drag_finish_anchor_region(page, i0)
    rows.nth(1).click()
    _drag_finish_anchor_region(page, i1)
    page.locator("#bFinishAnchor").click()
    assert page.evaluate("() => anchors.length") >= 1, "building a 2-member anchor did not add it to anchors[]"

    _pick_run(page, "align")
    _press_run(page)
    page.wait_for_function(     # #out is a textarea: its text is `.value`, not innerText
        "() => { const t = document.getElementById('out').value; return t.includes('align done:') || t.includes('ALIGN FAILED'); }",
        timeout=240_000)
    log = page.locator("#out").input_value()
    assert "ALIGN FAILED" not in log, log[-600:]
    after = log[log.rindex("align done:"):]
    found = re.search(r"matched on: (the boxes at full resolution|the whole scan at preview resolution, because)[^\n]*", after)
    assert found, f"the align log does not say what was matched on: {after[:800]!r}"
    print(f"    {found.group(0)[:160]}")


def test_a_landmark_named_in_two_scans_links_them_in_a_run(page):
    """In review, name selected points; the same name (any case) in a second scan shows as "links 2 scans" on the
    Landmarks tab, survives Save and reload, and "Align from my anchors only" runs on it with no drawn link: the
    server's report has one link joining exactly those two scans. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()

    def name_selection(name: str) -> None:
        rect = page.locator("#cv").bounding_box()
        cx0, cy0 = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
        page.locator('#reviewBar [data-rvtool="box"]').click()
        page.mouse.move(cx0 - 30, cy0 - 30); page.mouse.down(); page.mouse.move(cx0 + 30, cy0 + 30, steps=5)
        page.mouse.up()
        assert page.evaluate("() => review.sel.n") > 0, "the box selected nothing"
        page.locator("#rvLandmark").click()
        assert page.locator("#rvLandmarkForm").is_visible(), "the name form did not open"
        page.locator("#rvLmName").fill(name)
        page.locator("#rvLmKind").select_option("column")
        page.locator("#rvLmName").press("Enter")

    try:
        _goto(page)
        assert page.evaluate("() => anchors.length") == 0, "this test needs a project with no drawn anchors"
        page.locator('[data-view="plan"]').click()
        a = _select_scan_row(page, 0)
        b = int(page.locator("#scanlist .scan").nth(1).get_attribute("data-i"))
        page.evaluate("(ab) => { selection = new Set(ab); renderAll(); }", [a, b])
        page.locator("#sbReviewSelected").click()
        name_selection("QA Column")
        assert "same name in another scan" in page.locator("#out").input_value()
        page.locator("#rvNext").click()
        assert page.evaluate("() => review.i") == b
        name_selection("qa  column")
        assert "Also marked in" in page.locator("#out").input_value(), page.locator("#out").input_value()[-300:]
        assert 'value="QA Column"' in page.evaluate("() => $('lmNames').innerHTML"), "the used name is not offered"
        page.locator("#rvDone").click()
        _show_bottom_tab(page, "landmarks")
        row = page.locator("#landmarksList").inner_text()
        assert "QA Column (column)" in row and "links 2 scans" in row, row

        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        _goto(page)
        assert page.evaluate("() => landmarks.length") == 2, "the landmarks did not survive a reload"

        reports = os.path.join(PROJECT_DIR, "reports")
        before = set(os.listdir(reports)) if os.path.isdir(reports) else set()
        _pick_run(page, "align")
        _press_run(page)
        page.wait_for_function(
            "() => { const t = $('out').value; return t.includes('align done:') || t.includes('ALIGN FAILED'); }",
            timeout=240_000)
        log = page.locator("#out").input_value()
        assert "ALIGN FAILED" not in log, log[-600:]
        new = sorted(f for f in os.listdir(reports) if f.startswith("align_anchors_pairs") and f not in before)
        assert new, "the run wrote no anchors report"
        with open(os.path.join(reports, new[-1])) as fh:
            report = json.load(fh)
        names = page.evaluate("(ab) => ab.map(i => S[i].name)", [a, b])
        pairs = {frozenset((r["scan"], r["against"])) for r in report["rows"]}
        assert report["n_anchors"] == 1 and pairs == {frozenset(names)}, (report["n_anchors"], pairs)
        print(f"    named in 2 scans, kept through a reload, and the run matched {names[0]} with {names[1]} on it")
    finally:
        with open(placement_path, "wb") as fh:
            fh.write(original)


def _point_colour_sameness(page) -> tuple[float, int]:
    """(share of point pixels in the most common colour, how many point pixels) in the middle of the view below the
    review bar, with the grid off: points are every pixel that is not the background. Coarse colour bins of 32."""
    png = page.locator("#cv").screenshot()
    box = page.evaluate("""() => { const r = cv.getBoundingClientRect(), bar = $('reviewBar').getBoundingClientRect();
        return [0.15, Math.max(bar.bottom - r.top + 4, 0) / r.height, 0.85, 0.85]; }""")
    bg = page.evaluate("() => themeCanvas.rgb.map(v => Math.round(v * 255))")
    return tuple(page.evaluate("""async ([b64, box, bg]) => {
        const img = await createImageBitmap(await (await fetch('data:image/png;base64,' + b64)).blob());
        const c = new OffscreenCanvas(img.width, img.height), g = c.getContext('2d'); g.drawImage(img, 0, 0);
        const x0 = Math.round(box[0] * img.width), y0 = Math.round(box[1] * img.height);
        const d = g.getImageData(x0, y0, Math.round(box[2] * img.width) - x0, Math.round(box[3] * img.height) - y0).data;
        const bins = new Map(); let n = 0;
        for (let k = 0; k < d.length; k += 4) {
          if (Math.abs(d[k] - bg[0]) + Math.abs(d[k + 1] - bg[1]) + Math.abs(d[k + 2] - bg[2]) < 30) continue;
          const key = (d[k] >> 5) * 64 + (d[k + 1] >> 5) * 8 + (d[k + 2] >> 5);
          bins.set(key, (bins.get(key) || 0) + 1); n++;
        }
        return [n ? Math.max(...bins.values()) / n : 0, n]; }""", [base64.b64encode(png).decode(), box, bg]))


def test_review_opens_in_true_colour_and_turning_it_off_there_makes_the_scan_one_colour(page):
    """Review opens in True colour whatever the main view had, and restores the main view's choice on leaving. Next scan
    keeps the tool, the colour choice and open labelling tools."""
    _goto(page)
    page.evaluate("() => { $('cbCol').checked = false; draw(); }")           # the main view in false colour
    rows = page.locator("#scanlist .scan")
    rows.nth(0).click(); rows.nth(1).click(modifiers=["Control"])
    page.locator('[data-view="plan"]').click()
    page.locator("#sbReviewSelected").click()
    # The ceiling sliced off (the review bar's own button) with the slice's bottom moved back down to the floor, as
    # its message says: a white ceiling seen from above is one colour in true colour too.
    page.evaluate("""() => { $('cbGrid').checked = false; sliceOffCeiling(); clip.z.lo = clip.z.rangeMin;
                             renderAll(); draw(); }""")
    assert page.evaluate("() => $('cbCol').checked"), "review did not open in True colour"
    first = page.evaluate("() => review.i")
    mixed, n_true = _point_colour_sameness(page)
    page.locator('.menubtn[data-menu="menuView"]').click()
    page.locator("#cbCol").uncheck()
    page.keyboard.press("Escape")
    same, n_flat = _point_colour_sameness(page)
    assert n_true > 2000 and n_flat > 2000, f"too few point pixels to judge: {n_true}, {n_flat}"
    # Overlapping see-through points spread the one colour over a few brightness steps, so it holds less than all.
    assert same >= 0.4 and same > mixed + 0.3, f"one colour should hold most point pixels: {mixed:.2f} true, {same:.2f} off"
    page.locator('#reviewBar [data-rvtool="walls"]').click()
    page.locator("#rvLabelOpen").click()
    page.locator("#rvNext").click()
    assert page.evaluate("() => review.i") != first, "Next scan did not move on"
    assert page.evaluate("() => review.tool") == "walls", "Next scan dropped the Walls tool"
    assert not page.evaluate("() => $('cbCol').checked"), "Next scan forgot the colour choice"
    assert page.locator("#rvLabelTools").is_visible(), "Next scan closed the labelling tools"
    assert not page.locator("#rvWallAdd").count(), "the Add wall button is back"
    page.evaluate("() => $('cbCol').checked = true")
    page.locator("#rvDone").click()
    assert not page.evaluate("() => $('cbCol').checked"), "leaving review did not put the main view's False colour back"
    print(f"    point pixels in their commonest colour: {mixed:.0%} in true colour, {same:.0%} with it off; "
          f"next scan kept walls, colour and labelling")


def test_labels_given_in_review_colour_the_points_list_on_their_tab_and_survive_a_reload(page):
    """In review, "slice off the ceiling" cuts just under the ceiling and above the floor; labelling selected points as
    wall colours them, the Labels tab lists it, and Save and reload keep it. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        page.locator('[data-view="plan"]').click()
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        page.wait_for_function("() => getComputedStyle($('reviewBar')).display !== 'none'")
        page.locator("#rvLabelOpen").click()
        page.locator("#rvHideCeiling").click()
        assert page.evaluate("() => clip.z.on && clip.z.lo < clip.z.hi"), "slicing off the ceiling set no Z slice"
        kinds = page.locator("#rvLabelPick option").evaluate_all("os => os.map(o => o.value)")
        assert len(kinds) == 14 and {"doorway", "cabinet", "sink", "tub", "toilet"} <= set(kinds), kinds
        rect = page.locator("#cv").bounding_box()
        cx0, cy0 = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
        page.evaluate("() => { clip.z.on = false; draw(); }")          # the middle of the view may be empty once sliced
        page.locator('#reviewBar [data-rvtool="box"]').click()
        page.mouse.move(cx0 - 40, cy0 - 40); page.mouse.down(); page.mouse.move(cx0 + 40, cy0 + 40, steps=5); page.mouse.up()
        n_sel = page.evaluate("() => review.sel.n")
        assert n_sel > 0, "the box selected nothing"
        page.locator("#rvLabelPick").select_option("wall")
        page.locator("#rvLabelApply").click()
        coloured = page.evaluate("(i) => S[i]._labelIdx ? S[i]._labelIdx.filter(v => v === 0).length : 0", scan_i)
        assert coloured == n_sel, f"{coloured} points carry the wall label, {n_sel} were selected"
        assert "labelled" in page.locator("#out").input_value()
        page.locator("#rvDone").click()
        _show_bottom_tab(page, "labels")
        assert "wall" in page.locator("#labelsList").inner_text()
        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        _goto(page)
        assert page.evaluate("() => labels.length") == 1 and page.evaluate("() => labels[0].cls") == "wall"
        with open(placement_path) as fh:
            assert json.load(fh)["labels"][0]["cls"] == "wall", "the label is not in the project file"
        print(f"    {n_sel} points labelled wall, listed, kept through a reload and in the project file")
    finally:
        with open(placement_path, "wb") as fh:
            fh.write(original)


def test_a_site_measurement_is_clicked_on_two_walls_listed_and_saved(page):
    """Measure tool, Plan view: click a wall, click the wall across, type a distance. The Measurements tab lists it
    against the layout's reading; undo, Save and reload behave. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    placement_path = os.path.join(PROJECT_DIR, "placement.json")
    with open(placement_path, "rb") as fh:
        original = fh.read()
    try:
        _goto(page)
        page.locator('[data-view="plan"]').click()
        page.evaluate("() => fitAll()")
        page.locator('[data-tool="measure"]').click()
        # The longest joined wall on screen and the farthest joined wall parallel to it, clicked a third of the way
        # along the scan wall nearest each (screen coordinates, read from the page).
        ends = page.evaluate("""() => {
          const r = cv.getBoundingClientRect(), on = p => p[0] > 5 && p[1] > 5 && p[0] < cv.width - 5 && p[1] < cv.height - 5;
          const ws = venueWalls().walls.filter(w => on(w2s(w.a)) && on(w2s(w.b)));
          const len = w => Math.hypot(w.b[0] - w.a[0], w.b[1] - w.a[1]);
          const A = ws.reduce((x, w) => len(w) > len(x) ? w : x);
          const par = ws.filter(w => w !== A && Math.abs(w.n[0] * A.n[0] + w.n[1] * A.n[1]) > 0.999 && len(w) > 0.8);
          const B = par.reduce((x, w) => Math.abs(w.off - A.off) > Math.abs(x.off - A.off) ? w : x);
          const pt = w => { const m = w.members.reduce((x, y) => Math.hypot(y.b[0]-y.a[0], y.b[1]-y.a[1]) > Math.hypot(x.b[0]-x.a[0], x.b[1]-x.a[1]) ? y : x);
                            const a = w2s(m.a), b = w2s(m.b); return [r.left + a[0] + (b[0] - a[0]) / 3, r.top + a[1] + (b[1] - a[1]) / 3]; };
          return [pt(A), pt(B)];
        }""")
        for x, y in ends:
            page.mouse.move(x, y)
            page.mouse.click(x, y)
        page.wait_for_selector("#measureForm", state="visible")
        page.fill("#msValue", "30' 1 1/4\"")
        page.press("#msValue", "Enter")
        assert page.evaluate("() => measurements.length") == 1, "Enter in the distance box added no measurement"
        site, layout, diff = page.evaluate("() => { const m = measurements[0], r = measureReading(m); return [m.site_m, r.layout_m, r.diff_m]; }")
        assert abs(site - 9.17575) < 1e-4, f"30' 1 1/4\" read as {site} m"
        assert layout > 0.5 and abs(layout - site - diff) < 1e-9, (layout, site, diff)
        assert page.locator("#bottomTabs-tab-measurements").get_attribute("aria-selected") == "true"
        row = page.locator("[data-measure-row='0']").inner_text()
        assert "9.176 m" in row and f"{layout:.3f} m" in row, row
        assert page.evaluate("""() => ["361.25\\"", "30ft 1.25in", "9.17575 m", "9175.75 mm"]
                                 .every(t => Math.abs(parseSiteDistance(t, 'm') - 9.17575) < 1e-6)"""), "a distance format misread"
        assert page.evaluate("() => parseSiteDistance('abc', 'm')") is None
        page.keyboard.press("Escape")                                     # leave the tool, so Ctrl+Z reaches undo
        page.keyboard.press("Control+z")
        assert page.evaluate("() => measurements.length") == 0, "undo did not take the measurement back"
        page.keyboard.press("Control+Shift+z")
        assert page.evaluate("() => measurements.length") == 1, "redo did not bring it back"
        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        _goto(page)
        again = page.evaluate("() => measurements.length && measureReading(measurements[0]).layout_m")
        assert again and abs(again - layout) < 1e-6, f"after a reload the layout reads {again}, before {layout}"
        with open(placement_path) as fh:
            assert abs(json.load(fh)["measurements"][0]["site_m"] - 9.17575) < 1e-4, "the measurement is not in the project file"
        print(f"    on site 9.176 m, layout {layout:.3f} m ({diff * 1000:+.0f} mm), kept through undo, Save and a reload")
    finally:
        _put_back(page, (placement_path, original))


def _keep_project_file():
    """(path, bytes) of the QA project's file, to put back after a test that changes state the page autosaves."""
    path = os.path.join(PROJECT_DIR, "placement.json")
    with open(path, "rb") as fh:
        return path, fh.read()


def _put_back(page, kept):
    page.goto("about:blank")                       # off the page first, so it cannot autosave over the restore
    with open(kept[0], "wb") as fh:
        fh.write(kept[1])


def test_export_from_the_file_menu_saves_then_starts_the_export_and_reports_it(page):
    """File > Export merged .pts saves, requests an export with the page's settings, and reports the finished file. The
    server's answer is faked; tests/test_export_job.py runs the real job."""
    _goto(page)
    sent = []
    page.route("**/api/jobs", lambda route: (sent.append(route.request.post_data_json), route.fulfill(json={"id": "fakeexport"}))
               if route.request.method == "POST" else route.continue_())
    page.route("**/api/jobs/fakeexport", lambda route: route.fulfill(json={
        "id": "fakeexport", "state": "done", "progress": None,
        "result": {"points": 1234567, "output_path": "/projects/qa/exports/merged-x.pts", "skipped_no_placement": []}}))
    page.locator('.menubtn[data-menu="menuFile"]').click()
    with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
        page.locator("#mExport").click()
    page.wait_for_function("() => /EXPORTED|EXPORT FAILED/.test($('out').value)", timeout=30_000)
    out = page.locator("#out").input_value()
    assert "EXPORTED: 1,234,567 points -> /projects/qa/exports/merged-x.pts" in out, out[-400:]
    assert sent and sent[0]["kind"] == "export" and {"threads", "gpu", "remove_strays"} <= set(sent[0]), sent
    page.unroute("**/api/jobs"); page.unroute("**/api/jobs/fakeexport")
    print(f"    saved, asked for {sent[0]}, reported the file")


def test_enter_commits_an_anchor_drawn_with_the_tool_and_its_drop_button_removes_it(page):
    """Anchor tool: a box on each of two scans, then Enter (not the Finish button) commits the anchor; the Anchors
    tab's drop removes it and undo brings it back. Puts the project file back."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    kept = _keep_project_file()
    try:
        _goto(page)
        rows = page.locator("#scanlist .scan")
        i0, i1 = int(rows.nth(0).get_attribute("data-i")), int(rows.nth(1).get_attribute("data-i"))
        before = page.evaluate("() => anchors.length")
        page.locator('[data-view="plan"]').click()
        page.locator('[data-tool="anchor"]').click()
        rows.nth(0).click()
        _drag_finish_anchor_region(page, i0)
        rows.nth(1).click()
        _drag_finish_anchor_region(page, i1)
        page.locator("#cv").focus()
        page.keyboard.press("Enter")
        assert page.evaluate("() => anchors.length") == before + 1, "Enter did not commit the anchor"
        assert page.evaluate("() => anchors[anchors.length - 1].members.length") == 2
        _show_bottom_tab(page, "anchors")
        page.locator(f'[data-delanchor="{before}"]').click()
        assert page.evaluate("() => anchors.length") == before, "drop did not remove the anchor"
        page.keyboard.press("Control+z")
        assert page.evaluate("() => anchors.length") == before + 1, "undo did not bring the dropped anchor back"
        print("    two boxes, Enter: one anchor of two scans; dropped, and undo brought it back")
    finally:
        _put_back(page, kept)


def test_unlink_and_link_buttons_in_the_links_table(page):
    """A link from a run can be turned off and on again from its row."""
    _goto(page)
    n = page.evaluate("() => LINKS.length")
    assert n, "the QA project has no links from a run to show"
    k = 0
    _show_bottom_tab(page, "links")
    button = page.locator(f'[data-link="{k}"]')
    was = page.evaluate("(k) => LINKS[k].on", k)
    word = {True: "unlink", False: "link"}
    assert button.inner_text().strip() == word[was]
    for on in (not was, was):
        button.click()
        assert page.evaluate("(k) => LINKS[k].on", k) is on and button.inner_text().strip() == word[on], on
    print(f"    link {k} of {n}: switched {'off' if was else 'on'} and back from its row")


def test_turn_buttons_and_revert_to_my_placement_in_the_selected_panel(page):
    """"Revert this scan to my placement" (shown for a scan a run moved) puts it back on the hand placement; the yaw
    buttons -15 and +1 then turn it by exactly that. In that order: a hand turn makes the scan's place the hand
    placement, so the revert button goes. Puts the project file back."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    kept = _keep_project_file()
    try:
        _goto(page)
        pos = page.evaluate("() => [...document.querySelectorAll('#scanlist .scan')].findIndex("
                            "r => !S.find(x => x.i === +r.dataset.i).lock)")       # a locked scan's buttons are off
        assert pos >= 0, "every scan in the QA project is locked"
        scan_i = _select_scan_row(page, pos)
        page.evaluate("""(i) => { const s = S.find(x => x.i === i);       // as a run leaves it: moved, and saying so
          s.x += 0.3; s.y -= 0.2; s.alignMovedM = 0.36; s.alignMovedDeg = 0; renderAll(); }""", scan_i)
        page.locator("#bRevertOne").click()
        hand = page.evaluate("(i) => { const s = S.find(x => x.i === i); return [s.humanX, s.humanY, s.humanYaw]; }", scan_i)
        now = _pose(page, scan_i)
        assert max(abs(now["x"] - hand[0]), abs(now["y"] - hand[1]), abs(now["yaw"] - hand[2])) < 1e-9, (now, hand)
        yaw0 = now["yaw"]
        page.locator('[data-nudge-axis="yaw"][data-nudge="-15"]').click()
        page.locator('[data-nudge-axis="yaw"][data-nudge="1"]').click()
        turned = math.degrees(_pose(page, scan_i)["yaw"] - yaw0)
        assert abs(turned + 14) < 1e-6, f"-15 then +1 turned it {turned} degrees"
        print(f"    revert put it back on the hand placement; -15 then +1 turned it {turned:.1f} deg")
    finally:
        _put_back(page, kept)


def test_drop_buttons_on_the_landmarks_labels_and_measurements_tabs(page):
    """Each list's own drop button takes its entry off, and undo puts it back. Entries are put in by the page's
    own lists, not drawn; drawing them is tested above. Puts the project file back."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    kept = _keep_project_file()
    try:
        _goto(page)
        page.evaluate("""() => {
          const s = S[0].name, end = {scan: s, wall: [0, 0, 1, 0], at: [0.5, 0]};
          landmarks.push({id: 'lm-t', scan: s, name: 'test column', kind: 'column', shapes: [], points: [],
                          centroid: [0, 0, 0], extent: [0, 0, 0], n_points: 0});
          labels.push({id: 'lb-t', scan: s, cls: 'wall', source: 'label', shapes: [], n_points: 0});
          measurements.push({id: 'ms-t', a: end, b: end, site_m: 3, entered: '3 m', unit: 'm', note: ''});
          renderAll();
        }""")
        for tab, button, arr in (("landmarks", '[data-droplandmark="lm-t"]', "landmarks"),
                                 ("labels", '[data-droplabel="lb-t"]', "labels"),
                                 ("measurements", '[data-dropmeasure="ms-t"]', "measurements")):
            _show_bottom_tab(page, tab)
            page.locator(button).click()
            assert page.evaluate(f"() => {arr}.length") == 0, f"{tab}: drop left the entry"
            page.keyboard.press("Control+z")
            assert page.evaluate(f"() => {arr}.length") == 1, f"{tab}: undo did not bring it back"
        print("    landmarks, labels, measurements: each drop button removed its entry, undo brought it back")
    finally:
        _put_back(page, kept)


def test_run_from_alignment_using_links_too_starts_with_the_right_params(page):
    """The fourth Run option seeds from the placement and applies drawn links as a second pass. On a large project this
    is a long job, so the test checks the option exists, the request carries seed_from_placement and use_anchors, and
    a job really started (resource controls lock), then cancels it."""
    _goto(page)
    rows = page.locator("#scanlist .scan")
    assert rows.count() >= 2, "need at least 2 scans to build an anchor"
    i0, i1 = (int(rows.nth(k).get_attribute("data-i")) for k in (0, 1))
    page.locator('[data-view="plan"]').click()
    page.locator('[data-tool="anchor"]').click()
    rows.nth(0).click()
    _drag_finish_anchor_region(page, i0)
    rows.nth(1).click()
    _drag_finish_anchor_region(page, i1)
    page.locator("#bFinishAnchor").click()
    assert page.evaluate("() => anchors.length") >= 1, "building a 2-member anchor did not add it to anchors[]"

    assert page.locator('[data-run-kind="align_auto_seeded_anchors"]').count() == 1, \
        "Run from my alignment, using my anchors too, is missing from the Run window"
    _pick_run(page, "align_auto_seeded_anchors")
    with page.expect_request(lambda r: "/api/jobs" in r.url and r.method == "POST") as req_info:
        _press_run(page)
    body = req_info.value.post_data_json
    assert body.get("seed_from_placement") is True, f"seed_from_placement missing/false in the request: {body}"
    assert body.get("use_anchors") is True, f"use_anchors missing/false in the request: {body}"
    assert page.locator("#bRunMenu").is_disabled(), "resource controls did not lock while the job runs"

    page.wait_for_timeout(1000)  # let it get past "starting" before asking it to stop
    page.locator("#bCancelJob").click()
    page.wait_for_function(
        "() => { const t = document.getElementById('out').value; "
        "return t.includes('cancelled') || t.includes('AUTOMATIC ALIGN FAILED'); }",
        timeout=120_000)
    log = page.locator("#out").input_value()
    assert "AUTOMATIC ALIGN FAILED" not in log, log[-600:]
    print(f"    request carried seed_from_placement={body.get('seed_from_placement')}, "
          f"use_anchors={body.get('use_anchors')}; job started, locked controls, cancelled cleanly")


def test_height_settle_setting_is_on_by_default_and_reaches_the_run(page):
    """Settings > Aligning > Settle heights: on by default, saved, and sent with a run. Cancelled at once, like the test
    above."""
    def saved_settings(**want) -> dict:
        return _saved_settings(page, **want)

    _goto(page)
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    page.locator('button[role="tab"]:has-text("Aligning")').click()
    assert page.locator("#cbHeightSettle").is_checked(), "height settle should be on by default"
    assert page.locator("#cbThicknessTrust").is_checked(), "thickness trust should be on by default"
    page.locator("#cbHeightSettle").uncheck()
    try:
        assert saved_settings(height_settle=False).get("height_settle") is False, "unticking did not reach the server"
        page.keyboard.press("Escape")
        _pick_run(page, "align_auto_seeded")
        with page.expect_request(lambda r: "/api/jobs" in r.url and r.method == "POST") as req_info:
            _press_run(page)
        body = req_info.value.post_data_json
        assert body.get("height_settle") is False, f"the run was not told the setting: {body}"
        assert body.get("thickness_trust") is True, f"the run was not told thickness trust is on: {body}"
        page.wait_for_timeout(1000)
        page.locator("#bCancelJob").click()
        page.wait_for_function("() => { const t = document.getElementById('out').value; "
                               "return t.includes('cancelled') || t.includes('AUTOMATIC ALIGN FAILED'); }", timeout=120_000)
    finally:
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Aligning")').click()
        page.locator("#cbHeightSettle").check()
        assert saved_settings(height_settle=True).get("height_settle") is True, "the setting was not put back"
        page.keyboard.press("Escape")
    print("    on by default, unticking reached the server and the run's request, put back on")


def test_relative_movement_limits_are_off_by_default_and_reach_the_run(page):
    """Settings > Aligning > Movement limits ignore the whole layout moving together: off by default, saved, and sent
    with a run; with no scan dragged to the top, no reference scan is sent. Cancelled at once."""
    _goto(page)
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    page.locator('button[role="tab"]:has-text("Aligning")').click()
    assert not page.locator("#cbLimitsRelative").is_checked(), "relative movement limits should be off by default"
    page.locator("#cbLimitsRelative").check()
    try:
        assert _saved_settings(page, limits_relative=True).get("limits_relative") is True, "ticking did not reach the server"
        page.keyboard.press("Escape")
        _pick_run(page, "align_auto_seeded")
        with page.expect_request(lambda r: "/api/jobs" in r.url and r.method == "POST") as req_info:
            _press_run(page)
        body = req_info.value.post_data_json
        assert body.get("limits_relative") is True, f"the run was not told the setting: {body}"
        assert "reference_scan" not in body, f"no scan was dragged to the top, so no reference is chosen: {body}"
        page.wait_for_timeout(1000)
        page.locator("#bCancelJob").click()
        page.wait_for_function("() => { const t = document.getElementById('out').value; "
                               "return t.includes('cancelled') || t.includes('AUTOMATIC ALIGN FAILED'); }", timeout=120_000)
    finally:
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Aligning")').click()
        page.locator("#cbLimitsRelative").uncheck()
        assert _saved_settings(page, limits_relative=False).get("limits_relative") is False, "the setting was not put back"
        page.keyboard.press("Escape")
    print("    off by default, ticking reached the server and the run (no reference chosen, none sent), put back off")


def test_the_processor_double_check_is_off_by_default_has_help_and_reaches_the_run(page):
    """Settings > Resources > double-check on the processor: off by default, hover text, its "?" opens its Help entry,
    saved, and sent with a run. Cancelled at once."""
    def open_resources():
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Resources")').click()

    _goto(page)
    open_resources()
    box = page.locator("#cbGpuCpuCheck")
    assert not box.is_checked(), "the processor double-check should be off by default"
    assert "Off by default" in (page.locator('[data-setting="cbGpuCpuCheck"]').get_attribute("title") or ""), \
        "the setting has no hover description"
    box.check()
    try:
        assert _saved_settings(page, gpu_cpu_check=True).get("gpu_cpu_check") is True, "ticking did not reach the server"
        page.keyboard.press("Escape")
        _pick_run(page, "align_auto_seeded")
        with page.expect_request(lambda r: "/api/jobs" in r.url and r.method == "POST") as req_info:
            _press_run(page)
        body = req_info.value.post_data_json
        assert body.get("gpu_cpu_check") is True, f"the run was not told the setting: {body}"
        page.wait_for_timeout(1000)
        page.locator("#bCancelJob").click()
        page.wait_for_function("() => { const t = document.getElementById('out').value; "
                               "return t.includes('cancelled') || t.includes('AUTOMATIC ALIGN FAILED'); }", timeout=120_000)
    finally:
        open_resources()
        page.locator("#cbGpuCpuCheck").uncheck()
        assert _saved_settings(page, gpu_cpu_check=False).get("gpu_cpu_check") is False, "the setting was not put back"
    page.locator('[data-setting="cbGpuCpuCheck"] .mc-help-link').click()
    entry = page.locator("#help-cbGpuCpuCheck")
    assert entry.is_visible() and "two best answers" in entry.inner_text(), "the '?' did not open this setting's Help entry"
    page.keyboard.press("Escape")
    print("    off by default, hover text, ticking reached the server and the run, put back off, '?' opened its Help")


def test_a_run_that_moved_off_a_full_graphics_card_says_so_in_the_log(page):
    """A run that used the processor because the graphics card was full says why in the run log. Server answer faked."""
    _goto(page)
    note = ("The graphics card is nearly full (9.9 of 10.7 GB in use, most likely by another program), so this run "
            "used the processor instead, which is slower.")
    result = {"pairs": [], "poses": [], "n_loop_edges": 0, "unregistered": [], "moves": [], "gpu_note": note}
    page.route("**/api/jobs", lambda route: route.fulfill(json={"id": "fakejob"}) if route.request.method == "POST"
               else route.continue_())
    page.route("**/api/jobs/fakejob", lambda route: route.fulfill(json={"id": "fakejob", "state": "done",
                                                                       "progress": None, "result": result}))
    try:
        page.evaluate("() => { document.getElementById('out').value = ''; }")
        _pick_run(page, "align_auto_seeded")
        _press_run(page)
        page.wait_for_function("() => document.getElementById('out').value.includes('automatic align done')",
                               timeout=60_000)
        out = page.locator("#out").input_value()
        assert note in out and "FAILED" not in out, out[-800:]
    finally:
        page.unroute("**/api/jobs"); page.unroute("**/api/jobs/fakejob")
    print("    the full-card note is in the run log")


def test_movement_limits_guard_and_link_distance_have_their_defaults_save_and_reach_both_runs(page):
    """Settings > Aligning: Movement limits (1.5 m / 15 deg / 1 m), the guard (on) and the largest link move (1.5 m)
    are the defaults; changed, they are saved and sent with Run from my alignment and with the Links button. The
    server's answer to both runs is faked."""
    _goto(page)
    sent = []
    page.route("**/api/jobs", lambda route: (sent.append(route.request.post_data_json),
                                             route.fulfill(status=500, json={"error": "faked for the test"}))
               if route.request.method == "POST" else route.continue_())
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    page.locator('button[role="tab"]:has-text("Aligning")').click()
    assert page.locator("#moveLimits").input_value() == "1.5|15|1", page.locator("#moveLimits").input_value()
    assert page.locator("#cbCollateralGuard").is_checked(), "the guard should be on by default"
    assert page.locator("#linksMaxShift").input_value() == "1.5", page.locator("#linksMaxShift").input_value()
    page.locator("#moveLimits").select_option("0.5|5|0.2")
    page.locator("#cbCollateralGuard").uncheck()
    page.locator("#linksMaxShift").select_option("1")
    try:
        got = _saved_settings(page, move_limits="0.5|5|0.2", collateral_guard=False, links_max_shift_m=1)
        assert got.get("move_limits") == "0.5|5|0.2" and got.get("collateral_guard") is False, got
        page.keyboard.press("Escape")
        _pick_run(page, "align_auto_seeded")
        _press_run(page)
        page.wait_for_function("() => document.getElementById('out').value.includes('AUTOMATIC ALIGN FAILED')",
                               timeout=60_000)
        run = sent[-1]
        assert (run.get("scan_bound_m"), run.get("scan_bound_deg"), run.get("layout_bound_mean_m"),
                run.get("collateral_guard"), run.get("max_shift_m")) == (0.5, 5, 0.2, False, 1), \
            f"the run was not told the settings: {run}"
        page.evaluate("() => { document.getElementById('out').value = ''; }")
        _pick_run(page, "align")
        _press_run(page)
        page.wait_for_function("() => document.getElementById('out').value.includes('ALIGN FAILED')", timeout=60_000)
        assert sent[-1].get("kind") == "align" and sent[-1].get("max_shift_m") == 1, sent[-1]
    finally:
        page.unroute("**/api/jobs")
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Aligning")').click()
        page.locator("#moveLimits").select_option("1.5|15|1")
        page.locator("#cbCollateralGuard").check()
        page.locator("#linksMaxShift").select_option("1.5")
        _saved_settings(page, move_limits="1.5|15|1", collateral_guard=True, links_max_shift_m=1.5)
        page.keyboard.press("Escape")
    print("    defaults 1.5 m / 15 deg / 1 m, guard on, links 1.5 m; changed values saved and sent with both runs, put back")


def test_export_tightened_with_links_is_off_by_default_and_reaches_the_export(page):
    """Settings > Exporting > Tighten the export with my links: off by default, saved, and sent as refine with File >
    Export; the export's per-link lines are reported. The server's answer is faked."""
    _goto(page)
    sent = []
    page.route("**/api/jobs", lambda route: (sent.append(route.request.post_data_json), route.fulfill(json={"id": "fakeexport"}))
               if route.request.method == "POST" else route.continue_())
    page.route("**/api/jobs/fakeexport", lambda route: route.fulfill(json={
        "id": "fakeexport", "state": "done", "progress": None,
        "result": {"points": 1000, "output_path": "/projects/qa/exports/merged-y.pts", "skipped_no_placement": [],
                   "refine_log": [{"scan": "nowhere", "against": "x", "icp_fitness": 0.42, "shift_mean_m": 0.03,
                                   "applied": True}]}}))
    page.locator('.menubtn[data-menu="menuSettings"]').click()
    page.locator('button[role="tab"]:has-text("Export")').click()
    assert not page.locator("#cbExportRefine").is_checked(), "tightening with links should be off by default"
    page.locator("#cbExportRefine").check()
    try:
        assert _saved_settings(page, export_refine=True).get("export_refine") is True, "ticking did not reach the server"
        page.keyboard.press("Escape")
        page.locator('.menubtn[data-menu="menuFile"]').click()
        page.locator("#mExport").click()
        page.wait_for_function("() => /EXPORTED|EXPORT FAILED/.test($('out').value)", timeout=30_000)
        assert sent and sent[-1].get("refine") is True, f"the export was not told to tighten with links: {sent}"
        assert "tightened with links: nowhere moved (fit 0.42" in page.locator("#out").input_value()
    finally:
        page.unroute("**/api/jobs"); page.unroute("**/api/jobs/fakeexport")
        page.locator('.menubtn[data-menu="menuSettings"]').click()
        page.locator('button[role="tab"]:has-text("Export")').click()
        page.locator("#cbExportRefine").uncheck()
        assert _saved_settings(page, export_refine=False).get("export_refine") is False, "the setting was not put back"
        page.keyboard.press("Escape")
    print(f"    off by default, ticking saved and sent refine={sent[-1].get('refine')}, link line reported, put back")


def test_a_scan_dragged_to_the_top_is_the_reference_after_a_reload(page):
    """Dragging a scan to the top of the list makes it the reference: the star moves, a save records it, a run is told
    it, and a reload puts it back on top. The run's answer is faked. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    kept = _keep_project_file()
    try:
        _goto(page)
        rows = page.locator("#scanlist .scan")
        second = page.evaluate("() => S[drawOrder[1]].name")
        rows.nth(1).drag_to(rows.nth(0))
        assert page.evaluate("() => S[drawOrder[0]].name") == second, "the drag did not move the scan to the top"
        assert "★" in rows.nth(0).inner_text(), "the reference star is not on the new top scan"
        with page.expect_response(lambda r: r.url.endswith("/api/placement") and r.request.method == "POST"):
            page.locator("#bSave").click()
        with open(kept[0]) as fh:
            assert json.load(fh).get("reference_scan") == second, "the save did not record the reference scan"
        sent = []
        page.route("**/api/jobs", lambda route: (sent.append(route.request.post_data_json),
                                                 route.fulfill(status=500, json={"error": "not run in this test"}))
                   if route.request.method == "POST" else route.continue_())
        _pick_run(page, "align_auto_seeded")
        _press_run(page)
        page.wait_for_function("() => /not run in this test/.test($('out').value)", timeout=30_000)
        page.unroute("**/api/jobs")
        assert sent and sent[-1].get("reference_scan") == second, f"the run was not told the reference scan: {sent}"
        _goto(page)
        page.wait_for_function(f"() => S.length && S[drawOrder[0]].name === {json.dumps(second)}", timeout=30_000)
    finally:
        _put_back(page, kept)
    print(f"    dragged {second!r} to the top, star moved, saved and sent as reference_scan, back on top after a reload")


def test_keep_the_iphones_level_rebuilds_that_scan_saves_it_and_unticking_levels_it_again(page):
    """Review bar > keep the iPhone's level: off by default; ticking saves the choice and rebuilds that scan's preview
    with no levelling turn; unticking levels it to its floor again. Restores the file."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set")
        return
    kept = _keep_project_file()
    try:
        _goto(page)
        scan_i = _select_scan_row(page, 0)
        page.locator('.menubtn[data-menu="menuEdit"]').click()
        page.locator("#mReviewScan").click()
        assert not page.locator("#rvKeepLevel").is_checked(), "a scan should be levelled to its floor by default"
        page.locator("#rvKeepLevel").check()
        page.wait_for_function("() => /keeps the iPhone's level|could not/.test($('rvInfo').textContent)", timeout=240_000)
        info = page.locator("#rvInfo").inner_text()
        assert "keeps the iPhone's level" in info, info
        rot = page.evaluate("(i) => S[i].level_rotation", scan_i)
        assert all(abs(rot[r][c] - (1.0 if r == c else 0.0)) < 1e-9 for r in range(3) for c in range(3)), rot
        with open(kept[0]) as fh:
            name = page.evaluate("(i) => S[i].name", scan_i)
            entry = next(s for s in json.load(fh)["scans"] if s["name"] == name)
        assert entry.get("keep_scanner_level") is True, entry
        page.locator("#rvKeepLevel").uncheck()
        page.wait_for_function("() => /levelled to its floor again|could not/.test($('rvInfo').textContent)", timeout=240_000)
        rot = page.evaluate("(i) => S[i].level_rotation", scan_i)
        assert any(abs(rot[r][c] - (1.0 if r == c else 0.0)) > 1e-9 for r in range(3) for c in range(3)), \
            "unticking did not level the scan to its floor again"
        page.locator("#rvDone").click()
    finally:
        _put_back(page, kept)
    print("    off by default; ticking rebuilt the scan with no levelling turn and saved it; unticking levelled it again")


def test_automatic_align_starts(page):
    """Automatic align can be started from the page: choosing it and pressing Run starts a real server job and locks the
    resource controls. Kept last: the job keeps running after this test returns, so nothing after it may rely on
    placement.json being unchanged."""
    _goto(page)
    _pick_run(page, "align_auto")
    with page.expect_response(
        lambda r: "/api/jobs" in r.url and r.request.method == "POST"
    ) as resp_info:
        _press_run(page)
    resp = resp_info.value
    assert resp.status == 202, f"starting automatic align returned HTTP {resp.status}"
    job_id = resp.json().get("id")
    assert job_id, f"no job id in the start response: {resp.json()}"
    assert page.locator("#bRunMenu").is_disabled(), "resource controls did not lock while the job runs"

    job = call(f"{BENCH_URL}/api/jobs/{job_id}", "GET", timeout=10)
    assert job["state"] == "running", f"job {job_id} was already {job['state']!r}, expected still running"
    print(f"    job {job_id} started and is running (left running in the background, not awaited)")


# --------------------------------------------------------------------------

def test_sidebar_selection_controls_icons_and_majority_rule(page):
    """Select all / Deselect all are visible sidebar controls, lock/hide apply to a selection by majority, and the
    per-scan chips are icons."""
    _goto(page)
    page.wait_for_function("() => S && S.filter(s => !s.deleted).length >= 3", timeout=60_000)

    # 1. The controls are in the sidebar itself, not only in a menu.
    assert page.locator("#selbar #sbSelectAll").is_visible(), "Select all is not a visible sidebar control"
    assert page.locator("#selbar #sbSelectNone").is_visible(), "Deselect all is not a visible sidebar control"

    page.locator("#sbSelectAll").click()
    n_all = page.evaluate("() => selection.size")
    n_scans = page.evaluate("() => S.filter(s => !s.deleted).length")
    assert n_all == n_scans, f"Select all selected {n_all} of {n_scans}"

    page.locator("#sbSelectNone").click()
    assert page.evaluate("() => selection.size") == 0, "Deselect all left a selection behind"

    # Clicking empty space in the list clears the selection too.
    page.locator("#sbSelectAll").click()
    assert page.evaluate("() => selection.size") > 0
    page.evaluate("() => $('scanlist').dispatchEvent(new MouseEvent('click', {bubbles: true}))")
    assert page.evaluate("() => selection.size") == 0, (
        "clicking empty space in the scan list did not clear the selection"
    )

    # 2. Icons, not text, and each one still carries a readable name.
    lock_chip = page.locator("#scanlist [data-lock]").first  # a Mantine ActionIcon, no .chip class
    assert lock_chip.locator("svg").count() == 1, "the lock chip is not an inline SVG icon"
    assert lock_chip.inner_text().strip() == "", f"the lock chip still renders text: {lock_chip.inner_text()!r}"
    assert lock_chip.get_attribute("aria-label"), "an icon with no accessible name is a guessing game"
    assert lock_chip.get_attribute("title"), "the icon has no tooltip"

    # The icon must not swallow the click - the row handler reads ev.target.
    first_i = page.evaluate("() => +$('scanlist').querySelector('.scan').dataset.i")
    before = page.evaluate("(i) => S.find(s => s.i === i).lock", first_i)
    lock_chip.click()
    after = page.evaluate("(i) => S.find(s => s.i === i).lock", first_i)
    assert after != before, "clicking the lock ICON did not toggle lock (the svg ate the click)"
    lock_chip.click()

    # 3. Majority rule: 3 scans, 2 already locked -> the action unlocks all
    # three. The old "any unlocked -> lock all" rule would have locked them.
    ids = page.evaluate("() => S.filter(s => !s.deleted).slice(0, 3).map(s => s.i)")
    page.evaluate(
        """(ids) => {
             selection = new Set(ids);
             ids.forEach((i, n) => { S.find(s => s.i === i).lock = n < 2; });
             renderAll();
           }""",
        ids,
    )
    page.locator('.menubtn[data-menu="menuEdit"]').click()
    page.locator("#mBatchLock").click()
    locks = page.evaluate("(ids) => ids.map(i => S.find(s => s.i === i).lock)", ids)
    assert locks == [False, False, False], (
        f"majority of the selection was locked, so the action should unlock all three, got {locks}"
    )
    print("    select all/none visible in the sidebar, empty-space deselect, SVG icons "
          "with names, and lock by majority rule")


def test_hand_moves_are_logged_to_the_project_folder(page):
    """Arrow-key nudges write one line to hand-moves.jsonl in the project folder, an undo writes a line marked as an
    undo, and with the setting off nothing is written."""
    if not PROJECT_DIR:
        print("SKIP: PROJECT_DIR not set - this check needs direct file access to the project")
        return
    log = os.path.join(PROJECT_DIR, "hand-moves.jsonl")

    def lines():
        if not os.path.exists(log):
            return []
        with open(log) as fh:
            return [json.loads(x) for x in fh if x.strip()]

    if os.path.exists(log):
        os.remove(log)
    try:
        _goto(page)
        assert page.evaluate("() => $('cbHandLog').checked"), "the hand-move log should be on by default"
        title = page.evaluate("() => $('cbHandLog').closest('[title]') ? $('cbHandLog').closest('[title]').title : ''")
        scan_i = _select_scan_row(page, 0)
        name = page.evaluate("(i) => S.find(x => x.i === i).name", scan_i)
        _click_empty_props_area(page)
        for _ in range(5):
            page.keyboard.press("ArrowRight")
        page.wait_for_timeout(1800)                       # the 0.8 s quiet gap, then the write
        got = lines()
        assert len(got) == 1, f"5 arrow presses should be ONE logged move, got {len(got)}: {got}"
        assert got[0]["scan"] == name and got[0]["kind"] == "nudge" and got[0]["how"] == "arrow keys", got[0]
        assert got[0]["before"]["x"] != got[0]["after"]["x"], got[0]

        page.locator("#bUndo").click()
        page.wait_for_timeout(800)
        got = lines()
        assert len(got) == 2 and got[1]["kind"] == "undo", f"the undo was not logged as an undo: {got}"

        page.evaluate("() => { $('cbHandLog').click(); }")
        assert page.evaluate("() => $('cbHandLog').checked") is False
        _click_empty_props_area(page)
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(1800)
        assert len(lines()) == 2, "with the setting off a move was still written"
        page.evaluate("() => { $('cbHandLog').click(); }")  # back on; it is saved per project
        print(f"    5 arrow presses -> 1 line ({got[0]['how']}), undo -> 1 line marked undo, setting off -> nothing; "
              f"setting text: {title[:60]!r}")
    finally:
        if os.path.exists(log):
            os.remove(log)


TESTS = [
    test_bookmarked_project_url_opens_that_project,
    test_bookmarked_url_for_missing_project_shows_readable_message,
    test_start_screen_no_silent_resume_and_actions_present,
    test_start_new_project_overlay_opens_and_cancels,
    test_scans_from_anywhere_import_into_a_new_project_and_an_open_one,
    test_open_existing_modal_opens_a_project,
    test_switch_project_returns_to_start_screen,
    # Run the cancel test early so a later failure cannot stop it from running.
    test_cancel_running_job_stops_at_once_keeps_poses_and_looks_live,
    test_job_running_confirm_modal_stops_and_opens,
    test_escape_leaves_tool_and_shows_active_tool_label,
    test_sidebar_row_selects_on_the_name_text,
    test_sidebar_selection_controls_icons_and_majority_rule,
    test_seed_summary_shown_up_front_and_persists_on_reload,
    test_bent_scan_flags_show_in_the_links_panel_and_log_after_reload,
    test_links_table_divider_drags_and_keys_and_is_remembered,
    test_each_panel_folds_from_a_tab_on_the_view_that_covers_nothing,
    test_run_cancel_and_save_are_on_the_top_bar_and_the_lists_are_bottom_tabs,
    test_badges_never_cover_the_toolbar_at_any_width,
    test_no_text_or_controls_overlap_at_common_window_sizes,
    test_console_clean,
    test_react_scale_bar_renders_and_tracks_zoom,
    test_react_zoom_slider_drags_the_view_and_tracks_the_wheel,
    test_react_clip_sliders_all_visible_and_independent,
    test_scan_list_matches_api,
    test_fresh_load_preserves_saved_pose,
    test_webgl_renderer,
    test_build_info_matches_server,
    test_gpu_status_matches_reality,
    test_canvas_draws_something,
    test_drag_moves_scan,
    test_navigate_never_moves_scan,
    test_left_drag_pans_outside_plan_view,
    test_middle_and_right_drag_always_orbit,
    test_side_view_is_exactly_level_and_left_drag_pans_it,
    test_every_way_into_side_view_is_exactly_level,
    test_side_view_boxes_and_walls_draw_flat,
    test_projection_is_orthographic,
    test_wheel_zoom_in_a_level_view_holds_the_point_under_the_cursor,
    test_a_burst_of_wheel_events_zooms_by_distance_and_draws_once,
    test_run_status_lines_say_what_actually_ran,
    test_links_table_shows_a_refined_seed_pair_as_moved_not_kept,
    test_run_summary_counts_kept_placements_apart_from_real_matches,
    test_a_scan_that_did_not_connect_is_pointed_at_review,
    test_a_re_check_where_no_link_held_says_so_and_moves_nothing,
    test_detector_voting_is_ticked_by_default_and_can_be_unticked,
    test_shift_drag_height_unaffected_by_the_pan_orbit_swap,
    test_undo_after_nudges,
    test_multiselect_move_same_delta,
    test_plain_arrows_move_xy,
    test_ctrl_arrows_yaw_and_height,
    test_alt_arrows_tilt_pitch_and_roll,
    test_a_and_s_cycle_scan_list_not_wasd_move,
    test_numpad_5_jumps_to_plan_view,
    test_letter_hotkeys_match_their_buttons,
    test_clip_range_reaches_below_zero_on_the_real_venue,
    test_clip_vertical_thumbs_cannot_cross,
    test_clip_sliders_are_long_keep_a_thin_slice_and_step_like_a_ct_scan,
    test_clip_bounded_region_works_outside_plan_view,
    test_clip_bounded_hits_report_world_height_even_when_the_scan_is_lifted,
    test_markers_and_region_boxes_follow_a_tilted_scan,
    test_link_and_exclusion_markers_hide_with_x_y_slices_and_a_hidden_scan,
    test_run_result_says_each_height_change_in_plain_words,
    test_clip_state_survives_a_reload,
    test_clip_all_three_axes_together_form_a_box,
    test_clear_links_zero_anchors,
    test_save_reload_same_pose,
    test_links_table_shows_real_pair_results,
    test_persisted_links_survive_reload_and_show_declined_and_stale,
    test_pose_row_status_line_does_not_call_a_capped_row_moved,
    test_note_toggles_hide_explanations_by_default_and_remember_state,
    test_settings_explanations_moved_to_hover_and_the_help_menu,
    test_settings_help_link_opens_help_at_that_settings_own_entry,
    test_exclusions_and_removals_list_apart_and_the_export_setting_saves_with_its_help,
    test_menu_bar_opens_and_closes,
    test_scan_alias_persists,
    test_scan_display_colour_persists_and_stays_out_of_export,
    test_exclude_region_roundtrip,
    test_an_exclude_box_started_on_the_spin_handle_is_a_box_not_a_spin,
    test_review_view_removes_points_by_box_lasso_and_brush_and_the_server_cuts_the_same,
    test_floor_and_ceiling_marks_drag_type_select_and_survive_a_reload,
    test_walls_move_delete_and_add_in_review_and_the_server_applies_the_same,
    test_reprocess_a_scan_says_what_changed_and_rechecks_only_its_links,
    test_stairs_marked_in_review_survive_a_reload_and_re_processing_flags_the_same_points,
    test_review_count_next_scan_and_the_prompt_before_a_first_run,
    test_review_selected_walks_through_exactly_the_picked_scans,
    test_context_menu_right_click,
    test_as_imported_view_round_trip,
    test_revert_to_as_imported,
    test_tilt_suggestion_can_be_cancelled_applied_and_undone,
    test_backup_list_and_restore_work_from_the_file_menu,
    test_saved_versions_can_be_pinned_and_tidied_from_the_restore_list,
    test_saved_versions_settings_survive_a_save_and_a_reload,
    test_plan_view_export_from_the_file_menu_writes_a_dxf,
    test_floors_ceilings_and_walls_export_in_3d_from_the_run_window,
    test_settings_by_stage_save_reload_and_reach_the_server,
    test_one_end_of_an_anchor_can_be_redrawn_without_dropping_it,
    test_wall_thickness_report_from_the_file_menu,
    test_a_theme_can_be_chosen_stays_readable_persists_and_dark_comes_back,
    test_depth_cue_dims_far_points_in_the_angled_view_without_moving_anything,
    test_floors_and_ceilings_show_as_pieces_from_the_view_menu,
    test_align_from_links_says_what_each_scan_was_matched_on,  # writes poses: keep next to the LAST test
    test_a_landmark_named_in_two_scans_links_them_in_a_run,    # runs and writes poses too; puts the file back
    test_review_opens_in_true_colour_and_turning_it_off_there_makes_the_scan_one_colour,
    test_labels_given_in_review_colour_the_points_list_on_their_tab_and_survive_a_reload,
    test_a_site_measurement_is_clicked_on_two_walls_listed_and_saved,
    test_export_from_the_file_menu_saves_then_starts_the_export_and_reports_it,
    test_export_tightened_with_links_is_off_by_default_and_reaches_the_export,
    test_a_scan_dragged_to_the_top_is_the_reference_after_a_reload,     # puts the file back
    test_keep_the_iphones_level_rebuilds_that_scan_saves_it_and_unticking_levels_it_again,  # puts the file back
    test_enter_commits_an_anchor_drawn_with_the_tool_and_its_drop_button_removes_it,
    test_unlink_and_link_buttons_in_the_links_table,
    test_turn_buttons_and_revert_to_my_placement_in_the_selected_panel,
    test_drop_buttons_on_the_landmarks_labels_and_measurements_tabs,
    test_hand_moves_are_logged_to_the_project_folder,  # before the job tests: a running job refuses every move
    test_run_from_alignment_using_links_too_starts_with_the_right_params,  # cancels its own job; must not leave one running before the next test
    test_height_settle_setting_is_on_by_default_and_reaches_the_run,  # cancels its own job too
    test_relative_movement_limits_are_off_by_default_and_reach_the_run,  # cancels its own job
    test_the_processor_double_check_is_off_by_default_has_help_and_reaches_the_run,  # cancels its own job
    test_movement_limits_guard_and_link_distance_have_their_defaults_save_and_reach_both_runs,
    test_a_run_that_moved_off_a_full_graphics_card_says_so_in_the_log,
    test_automatic_align_starts,  # LAST: leaves a real job running server-side, see its own docstring
]


# Every test runs under a hard wall-clock cap: SIGALRM on the main thread, since Playwright's sync API cannot be
# touched from a watchdog thread.
TEST_TIMEOUT_S = 60  # slowest observed real PASS so far is ~8s; generous, still 20x tighter than the hang this replaces
# Longer caps only for tests slow by design (real registrations), so the default stays tight.
TEST_TIMEOUT_OVERRIDES_S = {
    "test_cancel_running_job_stops_at_once_keeps_poses_and_looks_live": 480,
    "test_job_running_confirm_modal_stops_and_opens": 240,
    "test_tilt_suggestion_can_be_cancelled_applied_and_undone": 400,  # three real solves at one thread
    "test_plan_view_export_from_the_file_menu_writes_a_dxf": 320,     # reads every raw scan at one thread
    "test_floors_ceilings_and_walls_export_in_3d_from_the_run_window": 320,  # every raw scan, twice when not cached
    "test_floors_and_ceilings_show_as_pieces_from_the_view_menu": 120,       # one raw scan, first time
    "test_wall_thickness_report_from_the_file_menu": 320,             # the same, then measures every wall
    "test_align_from_links_says_what_each_scan_was_matched_on": 320,  # reads the linked scans' full-resolution files at one thread
    "test_a_landmark_named_in_two_scans_links_them_in_a_run": 320,    # a real anchors-only run on two scans
    "test_run_from_alignment_using_links_too_starts_with_the_right_params": 180,  # loads 14 real scans' full-res data before the first cancel check point
    "test_height_settle_setting_is_on_by_default_and_reaches_the_run": 180,  # same reason
    "test_relative_movement_limits_are_off_by_default_and_reach_the_run": 180,  # same reason
    "test_the_processor_double_check_is_off_by_default_has_help_and_reaches_the_run": 180,  # same reason
    "test_keep_the_iphones_level_rebuilds_that_scan_saves_it_and_unticking_levels_it_again": 300,  # two preview rebuilds
    "test_no_text_or_controls_overlap_at_common_window_sizes": 240,   # 20 full-page scans of every text box, at one thread
    "test_scans_from_anywhere_import_into_a_new_project_and_an_open_one": 240,  # opens a new project twice, previews built
    "test_review_view_removes_points_by_box_lasso_and_brush_and_the_server_cuts_the_same": 180,  # reads one full scan
    "test_reprocess_a_scan_says_what_changed_and_rechecks_only_its_links": 900,  # rebuilds a preview, then a real run
    "test_stairs_marked_in_review_survive_a_reload_and_re_processing_flags_the_same_points": 300,  # rebuilds a preview
}

# Runs in its own browser subprocess, so a wedged Chromium (its native right-click menu) cannot take the rest of the
# suite with it.
ISOLATED_TESTS = {"test_context_menu_right_click"}
ISOLATED_TIMEOUT_S = 90  # own browser launch adds a few seconds over the normal 60s cap


class _TestTimedOut(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TestTimedOut()


# Shared by main() and _run_one_in_new_browser so an isolated test gets the identical browser.
_CHROMIUM_LAUNCH_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--disable-dev-shm-usage",
]


def _run_one_in_new_browser(name: str) -> int:
    """Run one named test in its own fresh browser. Only entered via --run-isolated NAME."""
    # No "PASS "/"FAIL " lines printed here: the parent process (below) prints
    # the one canonical line qa.py's parser matches, so this can't double it.
    fn = globals().get(name)
    if fn is None or fn not in TESTS:
        print(f"error: {name!r} not a known test in TESTS", file=sys.stderr)
        return 1
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=_CHROMIUM_LAUNCH_ARGS)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        _js_coverage_start(page)
        try:
            fn(page)
            _js_coverage_save(page, name)
            return 0
        except Exception as exc:  # noqa: BLE001 - relayed to the parent's own FAIL line
            print(f"error: {exc}", file=sys.stderr)
            return 1
        # No browser.close() on a hang: this process is disposable, the
        # parent kills it on timeout rather than waiting for a clean exit.


def _run_isolated_test(name: str) -> tuple[bool, str]:
    """Run one ISOLATED_TESTS test via --run-isolated in its own subprocess. Returns (passed, output)."""
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--run-isolated", name],
            timeout=ISOLATED_TIMEOUT_S,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        return False, (
            out + f"\ntimed out after {ISOLATED_TIMEOUT_S}s in its own browser process"
        )


def _selected_tests():
    """TESTS narrowed by ONLY_TESTS / SKIP_TESTS (comma-separated names). A narrowed run prints a banner and is never a
    full pass."""
    only = {n for n in os.environ.get("ONLY_TESTS", "").split(",") if n}
    skip = {n for n in os.environ.get("SKIP_TESTS", "").split(",") if n}
    unknown = (only | skip) - {fn.__name__ for fn in TESTS}
    if unknown:
        raise SystemExit(f"ONLY_TESTS/SKIP_TESTS name test(s) that do not exist: {sorted(unknown)}")
    picked = [fn for fn in TESTS if (not only or fn.__name__ in only) and fn.__name__ not in skip]
    shard = os.environ.get("BROWSER_SHARD", "")   # "k/n": this bench runs its share; tools/qa.py checks the shares add up
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        heavy = [fn for fn in picked if fn.__name__ in TEST_TIMEOUT_OVERRIDES_S]
        light = [fn for fn in picked if fn.__name__ not in TEST_TIMEOUT_OVERRIDES_S]
        mine = set(heavy[k::n]) | set(light[(k + len(heavy)) % n::n])
        picked = [fn for fn in picked if fn in mine]
        print(f"BROWSER SHARD {k + 1} of {n}: {len(picked)} test(s)", flush=True)
    if only or skip:
        left_out = [fn.__name__ for fn in TESTS if fn not in picked]
        print(f"FILTERED RUN: {len(picked)} of {len(TESTS)} browser tests selected, NOT a full pass. "
              f"Not run: {', '.join(left_out)}", flush=True)
    return picked


def main() -> int:
    failures = 0
    timed_out = False
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=_CHROMIUM_LAUNCH_ARGS)
            for fn in _selected_tests():
                if fn.__name__ in ISOLATED_TESTS:
                    t0 = time.time()
                    passed, output = _run_isolated_test(fn.__name__)
                    if output.strip():
                        print(output, end="" if output.endswith("\n") else "\n")
                    if passed:
                        print(f"PASS {fn.__name__} (isolated): {time.time()-t0:.1f}s")
                    else:
                        failures += 1
                        print(f"FAIL {fn.__name__} (isolated)")
                    continue
                page = browser.new_page(viewport={"width": 1400, "height": 900})
                _js_coverage_start(page)
                t0 = time.time()
                this_timeout_s = TEST_TIMEOUT_OVERRIDES_S.get(fn.__name__, TEST_TIMEOUT_S)
                signal.alarm(this_timeout_s)
                try:
                    fn(page)
                    print(f"PASS {fn.__name__}: {time.time()-t0:.1f}s")
                    _js_coverage_save(page, fn.__name__)
                except _TestTimedOut:
                    failures += 1
                    timed_out = True
                    print(f"FAIL {fn.__name__}: timed out after {this_timeout_s}s - the browser "
                          f"driver is presumably dead; stopping the suite here "
                          f"rather than letting every remaining test hang the same way")
                except Exception as exc:  # noqa: BLE001 - a test failure must not stop the rest
                    failures += 1
                    print(f"FAIL {fn.__name__}: {exc}")
                finally:
                    signal.alarm(0)
                if timed_out:
                    break
                try:
                    page.close()
                except Exception:  # noqa: BLE001 - closing a page must not itself hang the suite
                    pass
            if not timed_out:
                browser.close()
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    if PAGE_LOADS:
        page_s, scans_s = sum(p for p, _ in PAGE_LOADS), sum(s for _, s in PAGE_LOADS)
        print(f"TIMING page loads: {len(PAGE_LOADS)}, {page_s:.1f}s for the page, {scans_s:.1f}s for the scan "
              f"previews, median {sorted(p + s for p, s in PAGE_LOADS)[len(PAGE_LOADS) // 2]:.1f}s each", flush=True)
    if timed_out:
        # Exit directly: a driver that died once cannot be trusted to close cleanly.
        print(f"{failures} failure(s), suite stopped early by the timeout")
        os._exit(1)
    return 1 if failures else 0


if __name__ == "__main__":
    # A re-invocation of this file for one test in a disposable browser process.
    if len(sys.argv) == 3 and sys.argv[1] == "--run-isolated":
        raise SystemExit(_run_one_in_new_browser(sys.argv[2]))
    raise SystemExit(main())
