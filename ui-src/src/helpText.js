// Help text: one entry per setting, each with an anchor the Settings modal's "?" opens. Static, trusted, injected as
// HTML.
export const HELP = {
 "Keyboard": {
  "attrs": "",
  "html": "Arrow keys move the selected scan(s) in the world X/Y plane, by the Keyboard-step amount on the right. Ctrl+Up/Down moves height (Z), the same Keyboard-step amount. Ctrl+Left/Right turns it (yaw), and Alt+arrows tilt it by hand (pitch on Up/Down, roll on Left/Right) - a correction for the small residual tilt every real scan has, on top of its own automatic levelling - both by the Turn-step amount below. A and S move the selection to the next/previous scan in the sidebar list. Tab cycles the four position boxes (x, y, height, yaw) when one has focus. Shift+drag on a scan always means height, in every view. Ctrl+Z undoes, Ctrl+Shift+Z redoes."
 },
 "Tool and view shortcuts": {
  "attrs": "id=\"hotkeyHelp\"",
  "html": ""
 },
 "How this tool works": {
  "attrs": "",
  "html": "Nomad scans are already levelled by the phone, so placing one is really just three numbers: slide it in X, slide it in Y, spin it (yaw). You place every scan roughly where it belongs by eye in Plan, Iso and Side views, then either group matching regions between scans as a link and align from those, or let Automatic align solve the whole arrangement from geometry alone. \"Run from my alignment\" does the same geometry search but only lets it refine near your placement, never overrule it outright. \"Run from my alignment, using my links too\" does that, then runs a second pass that applies your drawn links on top, so a scan a link names can be corrected or rescued even if the geometry search alone left it alone. Align only ever moves poses; nothing is merged into one file until you Export, which is the only step that touches the full-resolution points."
 },
 "Measuring the venue": {
  "attrs": "",
  "html": "Distances you measured on site with a tape or laser, checked against the layout. Pick the <b>Measure</b> tool and go to the Plan view: the walls of the whole venue are drawn in blue, with several scans' copies of the same wall joined into one, the same way the 3D drawing joins them. Click a wall, then the wall at the other end of the run, and type what you measured: <code>30' 1 1/4\"</code>, <code>30ft 1.25in</code>, <code>361.25\"</code>, <code>9.176 m</code> or <code>9176 mm</code>. A plain number uses the unit picked beside the box. The <b>Measurements</b> tab under the view lists each one: what you measured, what the layout reads now, the difference, and whether one scan sees both walls (a check on that scan) or the walls come from different scans (a check on how the scans were put together). The reading is square off the first wall, from the spot you clicked, to the second wall, so measure the same way on site. Each end stays pinned to the scan wall you clicked, so the numbers follow the scans as they move or are aligned again. Fix walls in review first: the measurement uses the walls as corrected. Undo takes a measurement back; Save keeps them in the project."
 }
};

// In the order the settings appear (Resources, Preparing scans, Aligning, Exporting, Saved versions, Appearance).
// A setting's "?" opens the modal scrolled to its anchor (id="help-" + anchor in HelpModal.jsx).
export const SETTINGS_HELP = [
 {
  anchor: "resources",
  heading: "CPU threads, pairs at once, and the GPU switch",
  html: "<b>CPU threads</b> caps how many CPU threads any one job (Align, Export, a wall report) is allowed to use at once - lower it if the machine needs to stay usable for something else while a job runs. <b>Pairs at once</b> is how many scan pairs an automatic align works on in parallel; higher finishes sooner but uses more memory and CPU at the same time. <b>Use GPU where possible</b> turns on the CUDA-based fast paths (the page's own repeatable screening pass for \"Run from my alignment\", and the tensor pipeline for an unseeded automatic align) when the card is usable - off, everything runs on the CPU alone. The status line under the checkbox says plainly whether a usable GPU was actually found, separately from whether this build of the tool CAN use one at all."
 },
 {
  anchor: "cbGpuCpuCheck",
  heading: "... and double-check its answers on the processor",
  html: "Only applies to \"Run from my alignment\" with the GPU on. For each pair of scans, the graphics card tries every starting position at once and keeps the one whose walls, floors and corners agree best. <b>Off (the default):</b> that is the answer. <b>On:</b> the processor repeats the card's two best answers more carefully, from the same starting positions, and its answer is used instead. Measured on both test venues against hand-measured positions, the card's answer was just as accurate (every check passed, worst scan 24 cm against 26 cm on the larger venue), and runs took about half as long: 11 minutes against 24, and 3.5 against 5.5. Turn it on to compare, or if a venue comes out worse without it. Pairs the card cannot take (no card, or another program has filled it) run on the processor either way."
 },
 {
  anchor: "levelTilt",
  heading: "Levelling: largest floor tilt to straighten",
  html: "When a scan is levelled to its own floor (a one-time step when its preview is built), this is the steepest floor the tool will accept as 'the floor' at all. Raked seating and sloped floors in real venues measured between about 2 and 8 degrees. Lower this (3 is the strict end) if a ramp, a raked section, or a stage lip is being mistaken for the floor and levelled flat when it should stay sloped; raise it only if a genuinely steep real floor is being rejected outright. Takes effect the next time previews are built (File &gt; Rebuild previews) - changing it does not touch a scan already loaded.<br><br><b>Off</b> does not level any scan to its floor: every scan keeps the level the iPhone recorded when it was scanned. Use it for a venue whose floors are raked or sloped everywhere, where straightening each scan to its own patch of floor would tip correctly levelled scans by a different amount each. The floor height is still found, so heights still read from the floor. To do this for one scan only, tick \"keep the iPhone's level\" in that scan's review bar."
 },
 {
  anchor: "rangeLimit",
  heading: "Ignore far points when aligning",
  html: "Alignment only, never the export: leave out points farther than the chosen distance from where each scan started, measured across the floor. A phone's depth sensor is measurably less exact far away, so a distant point pulls an alignment's answer around more than its accuracy deserves. Off by default because a scan captured while walking has plenty of GOOD points far from where it started - this is a blunt tool, worth trying only if a specific scan's far reach is visibly hurting its own alignment."
 },
 {
  anchor: "searchVoxel",
  heading: "Search voxel (automatic align)",
  html: "A coarser copy of the point clouds used only for candidate GENERATION (RANSAC, wall, and gravity search) - fine alignment afterward still works at full detail regardless of this setting, so this never limits the final answer's precision, only how the tool searches for a starting point. The command-line pipeline leaves it off; Automatic align defaults to 0.15 m, which makes large scans far faster to search."
 },
 {
  anchor: "cbDetectorVote",
  heading: "Detector voting",
  html: "Only applies to \"Run from my alignment\" (and its \"using my links too\" variant). Instead of ranking candidate alignments by raw point closeness alone, this checks whether the matched walls, floors and ceilings - and corners, where three of them meet, and the line where two meet at an edge - actually agree with each other once a candidate is applied, and, only as a second opinion, whether their colour does too. A candidate whose surfaces, corners or edges disagree geometrically is rejected outright; colour alone can never reject one. Corners are the strongest single check: in a symmetric room two walls can look like a match for the wrong wall, but a corner's three angles rarely can. <b>On by default.</b> Untick it to compare against plain point-closeness ranking."
 },
 {
  anchor: "cbHeightSettle",
  heading: "Settle heights on shared floors and ceilings",
  html: "The last step of every automatic run. Two scans of the same floor should put it at the same height, so for every pair of overlapping scans the tool measures how far apart their floors and ceilings sit, then nudges each scan the run was allowed to move up or down until those gaps close as well as they can all at once. Heights only: turn and position are never touched, and a locked scan never moves, so the others settle around it. A pair that disagrees with all the rest (for example raked seating seen from slightly different places) counts for less. Untick it to compare."
 },
 {
  anchor: "wallSearch",
  heading: "Wall search",
  html: "How hard the tool looks for walls when it builds each scan's preview: the walls you see on the page, correct in review, and get in the 3D drawing. <b>Standard</b> (the default) looks for up to 8 walls in a scan. In a busy room some of those 8 are the same wall found twice, and a real wall can be missed. <b>Thorough</b> looks for up to 24 and keeps a wall only where there is solid wall behind it, standing at least 40 cm tall, wherever that is, so half walls and walls hanging from the ceiling (a beam, a bulkhead face) stay, and a wall behind a curtain is kept whole. It finds walls Standard misses, and it can also take tall cabinets or stacked boxes for walls; you can remove those in review. Alignment does its own wall search and is not changed by this. Takes effect the next time previews are built (File &gt; Rebuild previews)."
 },
 {
  anchor: "cbLandmarksInDrawings",
  heading: "Named landmarks in the drawings",
  html: "Landmarks are features you name yourself in the review view: select the points of a window, doorway, half wall, drop ceiling or column (slice the scan with the X, Y and Z sliders to get a clean outline), press \"selected points are a landmark...\", and give it a name. Give the same feature the same name in another scan and the two scans are linked on it in every run that uses anchors. With this setting on (the default), every named landmark is also drawn in the plan-view drawing and in the 3D drawing, as the outline of its box with its name, on a layer of its own called LANDMARKS, so you can switch that layer off in Vectorworks. Untick it to leave landmarks out of both drawings. It changes nothing about alignment."
 },
 {
  anchor: "cbSettlePieces",
  heading: "... using only the floors and ceilings it found",
  html: "Changes what the height settle above matches on. Normally it uses every flat, level surface two overlapping scans share, so a table top or a row of seats seen by both can count as if it were floor. With this on, each scan you have not marked by hand uses only the floor and ceiling pieces the tool found in it (View &gt; Floors and ceilings shows them): every floor level, and flat, lowered and sloped ceiling sections, but not counters, tables or steps. Floors and ceilings you marked by hand in review still come first. <b>Off by default.</b> Try it on a venue full of furniture at floor-like heights."
 },
 {
  anchor: "cbThicknessTrust",
  heading: "Trust thin, crisp surfaces more than thick, fuzzy ones",
  html: "A wall or ceiling that reads as one thin, crisp sheet is good evidence of where it is; the same surface reading thick and spread out is weak evidence, whatever the reason (distance, a glancing view, curtains, raked seating, or scans disagreeing). With this on, \"Run from my alignment\" measures how thick each surface reads in each scan. When fine-tuning a pair, points sitting further from their match than that pair's typical surface thickness count for less, and a pair that could only be matched on fuzzy surfaces holds its two scans less firmly when everything is solved together. Weights only: no point is ever dropped, and the export is untouched. Runs take about a third longer. Untick it to compare."
 },
 {
  anchor: "cbLimitsRelative",
  heading: "Movement limits ignore the whole layout moving together",
  html: "\"Run from my alignment\" has two brakes (set under Movement limits): no scan may move more than its limit from where the run started, and the scans together may not move more than the layout limit on average, or the whole run is declined. With this off (the default) every move counts from where the scan started, so a run whose answer is the whole layout sliding over by 30 cm as one piece trips both brakes, even though no wall, floor or doorway moved against another. With this on, the run first takes out the slide and turn the scans made all together, and the brakes only count what is left: scans moving against each other. A scan the brake refuses keeps its place in the layout instead of jumping back to its old spot. If the reference scan (top of the list) or a locked scan ends up out of step with the rest, the whole run is declined, since those cannot move. Every refused correction is listed in the run report with which brake refused it, how big it was and how well it fit. The numbers themselves do not change."
 },
 {
  anchor: "moveLimits",
  heading: "Movement limits (Run from my alignment)",
  html: "How far one run may move things. Per scan: how far a scan may move and turn from where the run started; a scan over it stays where you put it. Layout: how far all the scans may move on average; over it, the whole run is declined. The default, 1.5 m and 15 degrees per scan and 1 m for the layout, measured best on both test venues: the strict 50 cm setting threw away most good corrections. With \"Keep back a move that makes a good match worse\" on, the wider limits do not let a scan wreck its neighbours."
 },
 {
  anchor: "cbCollateralGuard",
  heading: "Keep back a move that makes a good match worse",
  html: "On by default. After a run, the tool checks every scan it moved against all its neighbours. If the move made a match that was good before clearly worse, that scan is put back where it was, the check is repeated until no moved scan hurts a good match, and each move kept back is listed as a suggestion with what it would have done. This is what stops one scan lining up with another at the cost of a third."
 },
 {
  anchor: "linksMaxShift",
  heading: "Largest move a link may make",
  html: "Applies to the Links button and to links used by a run. A link is judged on three things together: how well the two boxes fit, whether the answer stays level, and how far it moves the scan. A link that fits poorly or tips the scan is refused whatever the distance; this setting is the distance part. The default is 1.5 m. Strict (1 m) refuses more links when scans start well out; wide (3 m or more) lets the linked spots meet but can pull the rest of the layout out of place."
 },
 {
  anchor: "strayRemoval",
  heading: "Remove stray points from the export",
  html: "Off by default: the export keeps every point and every colour exactly as scanned, on the standing rule that reduction is your call and best done in Vectorworks, where you can see what is actually being thrown away before it goes. Turning this on drops isolated points (people caught mid-scan, glints, reflections) that sit far from any neighbour, at export time only - it never touches the saved placement or any preview. Gentle drops only the most isolated points; Strong drops more, more aggressively."
 },
 {
  anchor: "cbExportRefine",
  heading: "Tighten the export with my links",
  html: "Off by default, and with it off the export puts every scan exactly where your layout has it. With it on, the export first fine-tunes the scans you linked: in each link, every other scan is matched against the first scan listed in that link (whole scans, at the same working resolution alignment uses), and the small correction is kept only if it moves that scan's points less than 50 cm; otherwise the scan stays where you placed it. A scan is adjusted at most once, by the first link that lists it. The merged file is still every point at full resolution with its colour, and your saved layout is never changed by this. The export's report lists each link it tried, how well it fit, and whether it was used."
 },
 {
  anchor: "cbExportKeepsExcluded",
  heading: "Keep exclude-region points in the export",
  html: "An exclude region marks points that are mostly right but show a problem, for example a wall that scanned as a thick band of points instead of one thin, dense line. Those points would mislead alignment, so alignment never uses them, whatever this setting says. They are still your real scan, so by default they stay in the exported file, and you decide in Vectorworks what to do with them. Untick this to leave them out of the export too, both the merged point file and the plan-view drawing. Points you removed in the review view (people, obvious drift) are a different thing: they are left out of alignment and the export always. Neither ever touches the original scan files."
 },
 {
  anchor: "clipMinGap",
  heading: "Thinnest slice",
  html: "The X, Y and Z clip sliders along the edges of the view each keep the points between their two ends. This is how close those two ends may come, so you can cut a thin slice and look through the building one slice at a time, like a CT scan. Drag an end to change the slice, drag the coloured band between the ends to move the whole slice, or click the slider and use the arrow keys to step through. 5 cm by default. It only changes what you see; nothing is saved into the scans."
 },
 {
  anchor: "clipStep",
  heading: "Arrow keys move the slice by",
  html: "With a clip slider clicked, each press of an arrow key moves the whole slice, keeping its thickness. By default it moves by its own thickness, so a 5 cm slice steps 5 cm at a time and every step shows fresh points with none missed and none repeated. Pick a fixed distance instead if you prefer. Shift with an arrow always moves 1 cm, for fine positioning."
 },
 {
  anchor: "cbAutoTidy",
  heading: "Tidy saved versions automatically",
  html: "Every save keeps a copy of the layout it replaced, so any earlier state can be brought back from File &gt; Restore a backup. Over a long session that becomes hundreds of copies. With this on, the older ones are thinned out whenever you save or open the project, by the two rules below. Never removed, whatever the rules say: the newest copy, any copy you pinned with its pin button in the restore list, and any copy a report, a run or the golden folder names. Removing is permanent. To see what a tidy would do before anything goes, use Tidy up in the restore list, which shows it first."
 },
 {
  anchor: "backupKeepAll",
  heading: "Keep every saved version from the last",
  html: "Every copy newer than this is kept, however many there are, so the last stretch of work can always be stepped back through save by save. Default 1 hour."
 },
 {
  anchor: "backupHourlyDays",
  heading: "Then keep one per hour for",
  html: "Past the window above, only the newest copy in each hour is kept, for this many days. Older than that, only the newest copy in each day. A day you worked on the layout always keeps at least one copy."
 },
 {
  anchor: "cbHandLog",
  heading: "Keep a log of my hand moves",
  html: "Each time you finish moving a scan by hand (a drag, the arrow keys, a typed number, a nudge button, or putting a scan back), one line is added to <b>hand-moves.jsonl</b> in this project's folder. It records which scan, where it was before and where you left it, how you moved it, which view you were in, and whether the place it started from was your own or the result of an automatic run. Undo and redo are recorded too, marked as such, so an undone move is never read as a correct one. It stays on this computer; nothing is sent anywhere and nothing reads it yet. Moves that correct an automatic run are the ones that can later teach the tool where it goes wrong. The file is plain text and capped in size (when full, it is renamed hand-moves.old.jsonl and a new one starts); the tidy-up of saved versions never touches it, and you can delete it whenever you like."
 },
 {
  anchor: "themeSelect",
  heading: "Colours (theme)",
  html: "Changes the colours of the panels and menus around the point cloud - the point view itself stays dark in every theme so points read the same regardless. Remembered in this browser (not saved into the project), so switching projects or machines does not carry it along."
 },
 {
  anchor: "textSize",
  heading: "Interface text size",
  html: "Small / Normal / Large / Extra large. Applies to sidebars, menus, the HUD and the links table - not the point cloud canvas itself, which is unaffected regardless of this setting. Persists across reloads on this computer."
 },
];
