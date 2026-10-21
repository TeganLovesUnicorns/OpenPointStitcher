# Known limitations

Situations that make alignment harder, how to notice them, and what to do. Your rough placement is the main
defence against most of them: alignment refines it and is never free to overrule it.

## Venue hazards

### Repeating structure: seat rows, columns, bays, truss

A scan of one bay can look exactly like a scan of the next, so a scan aligned one bay off still scores well
locally. This is worse than an obviously wrong answer, because it looks right.

- **Notice it:** the merged cloud looks clean but a known dimension is out by exactly one seat pitch or column
  spacing.
- **Avoid it:** place the scan roughly where it belongs (alignment cannot move it a whole bay away), and get
  something unique in every scan: a door, a distinct piece of equipment, the end of the row.

### Black drape and matte black surfaces

Depth sensors may return fewer points from heavy drape than from hard surfaces.

- **Notice it:** whole wall areas thin or missing from a scan.
- **Avoid it:** include hard, solid surfaces (floor, structure, doors) in every scan.

### Mirrors and glass

Mirrors reflect the scan into a phantom room behind them; glass returns nothing or whatever is beyond it.

- **Notice it:** geometry outside the building, or a room that does not exist.
- **Fix it:** remove those points in review before aligning or exporting.

### Raked seating and sloped floors

Levelling finds the floor, and a raked floor is not level. The levelling tilt limit (Settings, Preparing scans)
stops a steep rake from tipping a whole scan; keep it small for raked houses. A gentle rake sits under any limit, so
a scan standing mostly on it can be told to keep the level the device recorded ("keep the iPhone's level" in its
review bar), or Levelling can be set to Off for the whole project.

- **Notice it:** a scan's suggested tilt is several degrees, its walls lean, and rooms refuse to join.

### Balconies, mezzanines and split levels

A scan with two floor levels has two large horizontal planes, and the wrong one taken as "the floor" puts the scan
a storey out.

- **Notice it:** one scan floating at the wrong height.
- **Fix it:** mark the floor and ceiling in review; height settle then uses the marked heights.

### Long corridors and narrow rooms

In a featureless corridor, sliding a scan along the corridor barely changes how well surfaces agree, so alignment
has little to pull it into place along that axis.

- **Notice it:** corridors come out too long or too short.
- **Avoid it:** capture the corridor's ends and any doors along it; draw links at distinctive features.

## Capture hazards

### No floor in the scan

The tool cannot level the scan from its own points and falls back on the device's levelling, which is usually fine
but loses a check.

### A single flat surface

Wall matching needs two non-parallel planes. A scan of one sign or one wall gives it one. Landmarks and links can
still join it.

### One scan of a very large space

Far walls come back with sparse points, and sparse points give poorer surface fits. Several smaller scans work
better.

### Things that moved between scans

Doors open and close, cases get pushed, chairs move. Alignment tolerates a modest share of the scene changing, but a
pair will score lower than expected. Remove moved objects in review if they cause trouble.

## The computer

### Another program filling the graphics card

A graphics card that runs out of memory does not refuse more work: the programs sharing it keep swapping each
other's data in and out, and all of them slow to a crawl. While another program is using most of the card, a run
matches its pairs on the processor instead, goes back to the card once it frees up, and says in the log how many
pairs moved. It still finishes, more slowly.

## Output

### File size

Export is lossless by default. A dozen full-resolution scans can produce a file of several hundred megabytes, which
is correct but worth knowing before opening it in a CAD program.

### Coordinates far from the origin

Scans are arranged around the origin on import, but chains of scans can still grow large coordinates. Check your
CAD program is happy with the range.
