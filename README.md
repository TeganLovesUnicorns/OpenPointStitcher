# OpenPointStitcher

Stitch a set of indoor LiDAR scans into one accurately placed model of a building, and turn it into drawings.

OpenPointStitcher takes many separate scans of a space, the kind an iPhone or iPad LiDAR app produces room by
room, and lines them up into one coordinate system. You place each scan roughly by hand in a browser; the tool
then finds the exact fit from the geometry the scans share (walls, floors, ceilings, corners) without ever
overruling where you put things. The result exports as a merged point cloud, a plan-view drawing and a 3D model
of the building's real surfaces.

It was built for lighting designers who survey venues, but works for any room-scale indoor scan set.

- **Runs entirely offline.** No accounts, no cloud services, no network access at runtime. Your scans never
  leave your machine.
- **Lossless.** Scans are always processed at full resolution, and exports keep every point, colour and
  intensity value. Any reduction is yours to do downstream.
- **You stay in charge.** Your rough placement is the backbone of every alignment. Automatic alignment refines
  it; it cannot move a scan into the wrong room.

## Features

- Browser-based placement bench with plan, side and isometric views: drag, spin and lift scans into place
- Automatic alignment from your placement, using walls, floors, ceilings, corners and shared surfaces
- Links: mark regions two scans share and align from them
- Wall finder and review tools: correct, label and measure walls; tag doorways and fixtures
- Tape-measure check: enter distances measured on site and compare them with the model
- Exports
  - merged point cloud (`.pts`, lossless; `.ply`, `.pcd`, `.xyz` also supported)
  - plan view as DXF, in metres, ready to draw a plot over
  - 3D surfaces as DXF: floors, ceilings (flat, dropped and sloped) and walls as real planes
  - a full report of every alignment, with quality numbers for each pair of scans
- Optional NVIDIA GPU acceleration, with the CPU always confirming the final answer

## Supported scans

Today: `.pts` text point clouds (`X Y Z [I] [R G B]`, one point per line). Most LiDAR apps and scanner software
can export this, including Vectorworks Nomad. Support for PLY, E57 and LAS/LAZ input is planned.

Scans are expected to be gravity-levelled, which every phone and tablet LiDAR app does. That lets alignment work
with 4 degrees of freedom (position plus turn) instead of 6, which is a big part of why it is reliable.

## Requirements

- Docker: Docker Desktop on Windows 11, or Docker Engine on Linux (macOS is untested)
- 16 GB of RAM or more; 32 GB is comfortable for a full venue
- Optional: an NVIDIA RTX 20-series or newer GPU with a current driver. Older cards (GTX 10-series and earlier)
  are not supported by the GPU build and the tool runs on the CPU instead.

## Quick start

### Windows

```powershell
git clone https://github.com/<you>/OpenPointStitcher.git
cd OpenPointStitcher
.\stitch.ps1 build
.\stitch.ps1 bench
```

### Linux

```bash
git clone https://github.com/<you>/OpenPointStitcher.git
cd OpenPointStitcher
./stitch.sh build
./stitch.sh bench
```

Then open <http://localhost:8080>, create a project and import your scans. The bench only listens on your own
machine, never your network.

No NVIDIA card, or no NVIDIA container support? Set `OPENPOINT_NO_GPU=1` before running the script and
everything runs on the CPU.

On Windows, Docker Desktop gives its Linux VM half your RAM by default. For large venues raise it in
`%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=24GB
```

## Folders

| folder | what it holds |
|---|---|
| `scans/` | where you drop scans to import (read only to the tool) |
| `projects/` | one folder per project: its copy of the scans, placement, backups and exports |
| `out/` | output of the command-line tools |

None of these are tracked by git. Set `SCANS`, `PROJECTS` or `OUT` to use other locations.

To take a scan out of a project, right-click it in the bench and choose **Remove from project**. Nothing is
deleted: its file moves from the project's `scans-raw/` folder to `removed-scans/`, next to a small `.json` note of
everything the placement said about it (its position, links, exclusions, landmarks, measurements). The placement is
saved as a version first, as with any save. File > Import scans lists removed scans with a **Put back** button that
brings the file and all of that back. Removing the last scan closes the project and opens the import pane. Removing
and putting back both wait until a running job has finished.

## Sample data

A sample project (a wedding venue, 14 scans) will be published as a separate download on the
[Releases](../../releases) page, with its SHA-256 checksum. Unzip it into `scans/` and import it from the bench.
The tool itself never downloads anything.

## Command line

Everything the bench does is also available without the browser:

```powershell
.\stitch.ps1 inspect                    # load every scan in scans/ and print its stats
.\stitch.ps1 stitch                     # align and merge scans/ into out/merged.pts with a report
.\stitch.ps1 place-merge                # merge using a saved placement, no alignment
.\stitch.ps1 gpu-info                   # what this machine's GPU can do for the tool
```

`stitch.sh` takes the same commands on Linux.

## Testing

```powershell
.\stitch.ps1 qa-quick     # unit and synthetic tests, about a minute
.\stitch.ps1 qa-medium    # the browser test suite
.\stitch.ps1 qa-slow      # whole-registration regression tests, a few minutes
.\stitch.ps1 qa           # all of the above on a freshly built image
```

None of the layers needs scans of your own. The browser layer runs on a generated family flat
(`tests/synthetic_apartment.py`): eight phone-style scans of its rooms, terrace and garden, each with its own paint,
floors and furniture, built into `out-qa/` the first time (about 70 MB, 20 seconds) and reused after. Every bench in
the run starts from the same saved layout, with each scan where it really is in the flat.

The slow layer always runs `tests/test_regression_synth.py`, which aligns two generated scans of a two-room venue
against their exactly known answer and checks that two separate processes agree to the bit. The real-scan regression
test (`tests/test_regression_real.py`) needs a pair of real scans kept in their own repository, and skips cleanly
without them. Clone it as `scans-regression` inside this checkout and the launchers mount it:

```powershell
git clone https://github.com/TeganLovesUnicorns/OpenPointStitcher-regression-scans.git scans-regression
```

The real pair catches alignment mistakes that the generated pair does not, so run with it before changing anything
in registration.

## Project layout

| path | contents |
|---|---|
| `open_point_stitch/` | the Python package: loading, alignment, wall finding, export, and the bench server |
| `placement_bench.html` | the browser bench |
| `ui-src/` | source for the bench's React panels (built into `placement_bench.html`) |
| `tests/` | the test suite |
| `tools/` | QA runner and development tools |
| `stitch.ps1`, `stitch.sh` | launchers for Windows and Linux |

## Contributing

Issues and pull requests are welcome. Please run `qa-quick` before sending a change, and `qa` for anything that
touches alignment.

By submitting a contribution you agree that it may be distributed under the AGPL and under any other license the
project maintainer offers OpenPointStitcher under.

## License

Copyright (C) 2026 the OpenPointStitcher authors.

OpenPointStitcher is free software, licensed under the GNU Affero General Public License v3.0; see
[LICENSE](LICENSE). It comes with no warranty. If you run a modified version as a service for others, the AGPL
requires you to offer them its source.

Commercial licenses for use outside the AGPL's terms may be available from the maintainer.

Third-party components and their licenses are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md); all of
them are compatible with the AGPL.
