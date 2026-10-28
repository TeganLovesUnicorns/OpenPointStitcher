# Changelog

All notable changes, newest first. Versions follow `major.minor.patch`: a new minor version for new features, a
patch for fixes. One line per change.

## 0.55.3 (2026-10-28)

- A new regression test aligns two generated phone-style scans of a two-room venue (drift, noise, a door closed in one
  scan and open in the other) against their known answer and checks that two separate runs agree to the bit, so the
  slow test layer needs no real scans.
- A project you just created shows in the project list straight away if you cancel its import.
- Choosing your own files to import unticks the folder list, so its scans no longer come along unasked.
- The real-scan regression test skips when no scans are mounted, instead of failing on every machine without them.
- A bookmarked link to a project with no scans yet opens its import window, as clicking it does.
- A project whose scans are present but whose loading hits another missing file reports that file, instead of
  saying it has no scans.
- Saved links read the same way everywhere: one with no id no longer stops the merge export from loading the project.

## 0.55.2 (2026-10-26)

- Scans that start with a ";" or "#" comment line, such as the GeoData location line newer iPhone scanner exports
  write before the point count, now import and show their point count instead of failing to read.
- Clicking a project that has no scans yet opens its import window, instead of an error with no way to add any.

## 0.55.1 (2026-10-23)

- Excluded regions and review removals are now left out of every alignment step (height settle, links, tilt
  suggestion), not only pair matching.
- The processor and the graphics card build each alignment step's rotation the same way, so with mid-air or learned
  trust they give the same answer.
- Shared code for writing DXF files, reading trained models and a few smaller helpers, instead of copies.

## 0.55.0 (2026-10-23)

- Settling heights now pairs only flat points on surfaces that face the same way and sit within 30 cm, and measures
  the height above the other scan's surface, so a step, stage edge or seat beside the floor no longer pulls a scan
  up or down. Floors and ceilings of overlapping scans agree better on both test venues.

## 0.54.0 (2026-10-23)

- The graphics card now weights each pair the way the processor does: "Trust thin, crisp surfaces" (and the mid-air
  and learned trust options) apply on the card too, and its full-detail alignment gives the processor's answer.
- The careful double-check runs on the card when it is free, so runs with it on take about a third less time.

## 0.53.0 (2026-10-21)

- "Run from my alignment" with the GPU on now takes the graphics card's answer for each pair instead of repeating
  the best two on the processor, so runs take about half as long with the same accuracy on both test venues. A new
  setting, off by default, brings the processor double-check back.

## 0.52.1 (2026-10-20)

- A run no longer slows to a crawl when another program fills the graphics card: while the card is full, pairs are
  matched on the processor, and the log says how many moved.
- A job's saved record now names the stage it reached, so a run cut off by a restart says where it stopped.

## 0.52.0 (2026-10-20)

- "Run from my alignment" may move a scan up to 1.5 m and 15 degrees (was 0.5 m and 5), and the whole layout by
  1 m on average (was 20 cm). A new setting, Movement limits, picks strict, normal or wide.
- A move that makes a good match worse is now kept back by default, and listed so you can apply it by hand (new
  setting, on by default).
- How far a link may move a scan (Links button, and links after a run) is a setting, 1.5 m by default (was a
  fixed 1 m).

## 0.51.0 (2026-10-19)

- New setting, off by default: movement limits can ignore the whole layout moving together, so only scans moving
  against each other count. Every correction a limit refuses is now listed in the run report.
- A scan can keep the level the iPhone recorded instead of being levelled to its floor (review bar), and the
  Levelling setting has an "Off" choice for a whole project, for raked and sloped floors.
- The scan dragged to the top of the list is the reference that runs keep still; until one is chosen, runs behave
  as before.
- Scan pairs never to be matched are set per project, not in the shipped settings.
- The optional colour and full-resolution refine steps keep a change only if surface fit agrees.
- Export can tighten the merge with your links (Settings, Exporting; off by default).

## 0.50.4 (2026-10-18)

- The run summary, log messages, command help and test messages no longer refer to internal development notes.

## 0.50.3 (2026-10-17)

- The package is 38 modules instead of 60: each group of small modules that did one job together is now one file
  (walls, wall checks, floors and ceilings, projects, the graphics card, the pose graph, jobs, explanations,
  merging). Results are unchanged.
- The server's jobs share one frame for the saved-placement check, errors, thread caps and Cancel reports.

## 0.50.2 (2026-10-17)

- Shared code moved into `open_point_stitch/lib/` (poses, geometry, file saving, grouping, point clouds) and the
  copies of it removed; alignment results are unchanged.
- Every saved file (placements, reports, job records, caches, DXF and PNG drawings) is now written whole or not at
  all, so a crash or a stopped job can no longer leave a half-written file.
- The page's job buttons and the test and QA tools share one request and job-running path.

## 0.50.1 (2026-10-16)

- Licensed under the GNU AGPL v3. `LICENSE` and the third-party license list ship inside the Docker image.

## 0.50.0 (2026-10-16)

- Renamed the project to **OpenPointStitcher**: package `open_point_stitch`, image `open-point-stitcher`, bench
  container `open-point-bench`, settings `OPENPOINT_*`. Versions 0.30 to 0.49 were skipped to mark the rename.
- Projects saved by earlier versions still open; they are saved in the new format from then on.
- Viewer settings saved in the browser (theme, text size, measuring unit, open panels) carry over.
- `bench-update` also removes the bench container left by earlier versions.
- New README, changelog and third-party license list.
- Code comments rewritten as short, conventional comments.

## 0.29.8 (2026-10-16)

- A GPU too old for this build (GTX 10-series and earlier) is detected and the tool stays on the CPU instead of
  failing a run.
- Every Python package in the image is pinned to an exact version, so a rebuild on another machine is identical.
- `NOMAD_NO_GPU=1` (now `OPENPOINT_NO_GPU`) runs the launchers without requesting a GPU.
- `stitch.sh`: Linux launcher with the same commands as `stitch.ps1`.

## 0.29.7 (2026-10-16)

- Hand edits to walls are kept when previews are rebuilt by a newer wall finder.
- Wall runs that overshoot a corner are cut back correctly across both wall finders, including runs needing
  several cuts and walls hidden behind curtains.
- Lower memory use when finding walls from above.

## 0.29.6 (2026-10-16)

- Wall finder gives the same walls for a scan however it is moved or turned, and finds walls more accurately.

## 0.29.5 (2026-10-15)

- Review tools: label doorways, cabinets, sinks, tubs, showers and toilets; walls are tagged with what they
  contain; keyboard shortcuts shown on hover.
- Labelling moved into the main bench; the separate labelling bench was removed.

## 0.29.4 (2026-10-15)

- Wall finder retuned against a second reference venue.

## 0.29.3 (2026-10-15)

- Test suite audit: coverage measured, a broken GPU path fixed, untested controls covered.

## 0.29.2 (2026-10-15)

- Site measurements: a Measure tool on the venue's walls and a Measurements tab comparing tape measurements with
  the model.

## 0.29.1 (2026-10-15)

- Codebase audit: dead code removed, faster page, lower memory use.

## 0.29.0 (2026-10-15)

- New wall finder, scored against hand-corrected walls; hand corrections are kept through it.

## 0.28.1 (2026-10-09)

- Bulkheads found as ceiling pieces; walls keep their full height; the levelling setting applies to previews and
  runs.

## 0.28.0 (2026-10-09)

- Floors and ceilings found as pieces (flat, dropped and sloped), shown in the bench and optionally used for
  alignment.
- 3D DXF export of floors, ceilings and walls as planes.
- Named landmarks that link scans.
- Excluded regions stay in the export by default; removed points are listed separately.
- Links explain why they hold; Run button in the top bar; bottom tabs; CT-style slicing; wall end points.
- Review several selected scans one after another.
- An exclusion box no longer spins its scan; panel fold tabs on the view.

## 0.27.2 (2026-10-08)

- Opening a project is fast and takes one click.

## 0.27.1 (2026-10-06)

- Scans that fail to connect point to the review tool; a re-check where no link holds says so.

## 0.27.0 (2026-10-06)

- Cancel stops any job at any point.

## 0.26.0 (2026-10-05)

- Import scans from anywhere, into a new or open project.
- Review one scan alone: remove points by box, lasso or brush; mark floor, ceiling and stairs; correct its walls;
  re-process it and re-check only its links.
- Links table can be resized; scans with a bent wall are flagged, with which face of the wall each scan saw.

## 0.25.4 (2026-10-05)

- QA runs at three speeds; a library of past bugs is re-checked on every run; bent-scan flag.

## 0.25.3 (2026-10-03)

- Walls behind curtains are found.

## 0.25.2 (2026-10-03)

- Log of hand nudges after an alignment.

## 0.25.1 (2026-10-02)

- Noise and surface thickness measured per scan.

## 0.25.0 (2026-10-01)

- Confidence reported for every pair; markers checked on slices and hidden scans; layout file five times smaller.

## 0.24.1 (2026-09-30)

- Height settle: scans settle onto shared floors and ceilings as the last stage of every alignment.

## 0.24.0 (2026-09-30)

- Links take height from floors and hand tilt; saved-version cleaner; settings kept on Save; dropping a tilt
  keeps the scan's centre.

## 0.23.2 (2026-09-29)

- Real progress reporting inside each pair's alignment; GPU problems in the container fixed.

## 0.23.1 (2026-09-29)

- Landmarks vote in alignment; optional evidence-weighted pose graph; organic shapes rejected as landmarks.

## 0.23.0 (2026-09-29)

- Fixed scans jumping in height after alignment; whole-layout bound on how far a run may move scans; Settings and
  Help as tabbed dialogs.

## 0.22.0 (2026-09-28)

- Combined run mode, wall tightness weighting, zoom and brightness controls, rebuilt link tool.

## 0.21.1 (2026-09-28)

- Docs checked against the code; overlapping text and buttons fixed.

## 0.21.0 (2026-09-27)

- Align from links matches the points inside the boxes at full resolution.

## 0.20.3 (2026-09-27)

- Cancel and progress for every job kind; restore from backup.

## 0.20.2 (2026-09-27)

- Warning before leaving with unsaved changes; autosave; unsaved-changes indicator.

## 0.20.1 (2026-09-27)

- Server no longer hangs when opening a project: previews build in worker processes.
- "Run from my alignment" screens candidates on the GPU; the CPU confirms the answer.

## 0.20.0 (2026-09-26)

- Tilt solve from floors and ceilings; plan-view DXF export; doors and windows; GPU on by default; settings
  grouped by stage; themes.

## 0.14.0 (2026-09-26)

- Detector voting on by default for "Run from my alignment".

## 0.13.2 (2026-09-24)

- Exactly level side view, vertical wall lines, clearer run status; QA deadlock fixed.
- React-based menus, toolbar, scan list, sidebars and bottom panel.

## 0.12.1 (2026-09-21)

- Links table persists across reloads; clip cube; left-drag pans outside plan view; zoom slider; edge clip sliders.

## 0.12.0 (2026-09-14)

- Corners, colour families and edges vote on candidate alignments.
- Plan-cut tools with a two-handle slice bar; settings explanations moved to hover and Help.
- Alignment declines rather than moving scans without evidence.

## 0.10.0 (2026-09-07)

- Pairs chosen by shared surface; coarse-to-fine refinement for seeded runs; per-scan false colour; rotation pivot
  fixed; scans arranged around the origin at import.

## 0.9.0 (2026-09-05)

- Projects: folders, a project file and a start screen.
- Exclusion regions.
- As-imported pose (record, label, view, revert) and a right-click menu.
- Acceptance test on synthetic scans.
- Cancel stops the worker pool; fail loudly on non-finite coordinates.

## 0.8.0 (2026-08-31)

- First tracked version: `.pts` loading, gravity levelling, wall-based and feature-based pairwise alignment, pose
  graph, lossless merge, the browser placement bench, and "Run from my alignment".
