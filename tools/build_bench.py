"""Build scan previews for the placement bench from .pts scans. Previews are reduced for the browser only (real
points, random, with a density floor); nothing here feeds the merged output, which re-reads full resolution.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
from typing import Optional

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from open_point_stitch import __version__
from open_point_stitch.lib.files import write_json_atomic
from open_point_stitch.pts_file import read_pts, find_scans
from open_point_stitch.exclusions import excluded_mask, filter_scan
from open_point_stitch import align_gravity
from open_point_stitch.align_gravity import estimate_level
from open_point_stitch.footprint import footprint
from open_point_stitch.find_walls import drawn_walls, plane_rows, segment_row
from open_point_stitch.find_walls import wall_runs
from open_point_stitch.prepare import density_preserving_thin, thin_rng

log = logging.getLogger(__name__)

PREVIEW_VOXEL = 0.03        # metres; fine grid for level and wall finding, not the display budget
PREVIEW_MAX_POINTS = 8000   # browser point cap
BAND_BIN_M = 0.05           # bin width for the floor-height histogram


PREVIEW_BIN_M = 0.05   # counting-grid bin (metres) that measures local crowding; never an output point
PREVIEW_FLOOR_N = 3    # a bin with this many points or fewer keeps all of them


# Thinning lives in open_point_stitch.prepare, shared with align_pair.py; aliased to the old names here.
_density_preserving_preview = density_preserving_thin
_preview_rng = thin_rng


AGREEMENT_K = 20  # neighbours per point for the local-flatness measure below


def _agreement_scores(pts: np.ndarray, k: int = AGREEMENT_K) -> np.ndarray:
    """Per-point local flatness (local-PCA surface variation over k neighbours): near 1 on a flat wall, near 0 in
    scattered patches. Preview points only."""
    n = len(pts)
    if n <= k:
        return np.ones(n)  # too few points to judge locally; do not penalise
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    tree = o3d.geometry.KDTreeFlann(pcd)
    scores = np.empty(n)
    for i in range(n):
        cnt, idx, _ = tree.search_knn_vector_3d(pts[i], k)
        if cnt < 4:
            scores[i] = 0.0  # isolated point, nothing local to agree with
            continue
        nbrs = pts[np.asarray(idx)]
        cov = np.cov(nbrs.T)
        eigvals = np.linalg.eigvalsh(cov)  # ascending order: smallest first
        total = eigvals.sum()
        scores[i] = 1.0 if total <= 1e-12 else 1.0 - (eigvals[0] / total)
    return scores


def quick_bbox_payload(path: str) -> dict:
    """Near-instant raw min/max box (no Open3D, no lock, not levelled), replaced when the real payload lands."""
    scan = read_pts(path)
    lo, hi = scan.bounds()
    return {
        "name": scan.name,
        "points": scan.n_points,
        "bbox_raw": [round(float(lo[0]), 3), round(float(lo[1]), 3),
                     round(float(hi[0]), 3), round(float(hi[1]), 3)],
        "z_raw": [round(float(lo[2]), 3), round(float(hi[2]), 3)],
    }


TEMPLATE = os.path.join(ROOT, "placement_bench.html")

# Bump whenever build_scan_payload's OUTPUT SHAPE changes, so old caches are not reused.
BUILDER_VERSION = 11  # bump whenever the wall finder's output changes, so cached previews rebuild
WALL_FINDER_SINCE = 11  # previews older than this had another wall finder: hand wall edits on them are frozen first.
                        # Raise it with BUILDER_VERSION whenever the walls change (test_wall_finder_fingerprint)
CACHE_DIRNAME = ".bench_cache"


PREVIEW_MAX_WALLS = 8        # "standard" or "thorough" (looks further)
THOROUGH_MAX_WALLS = 24


def preview_options(settings: Optional[dict] = None) -> dict:
    """The project settings a preview is built with. Passed to every build explicitly: builds run in worker
    processes, which never see a setting changed in the server process."""
    settings = settings or {}
    try:
        tilt = float(settings.get("level_tilt_deg", align_gravity.DEFAULT_MAX_TILT_DEG))
    except (TypeError, ValueError):
        tilt = align_gravity.DEFAULT_MAX_TILT_DEG
    return {"level_tilt_deg": tilt,
            "max_walls": THOROUGH_MAX_WALLS if settings.get("wall_search") == "thorough" else PREVIEW_MAX_WALLS}


def _cache_key(path: str, options: Optional[dict] = None) -> dict:
    """Cache validity key: file name, size, mtime, BUILDER_VERSION and the settings it was built with."""
    options = options or preview_options()
    st = os.stat(path)
    key = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "size": st.st_size,
        "mtime": st.st_mtime,
        "version": BUILDER_VERSION,
        "level_tilt_deg": options["level_tilt_deg"],
    }
    if options["max_walls"] != PREVIEW_MAX_WALLS:     # only then, so standard previews saved earlier stay valid
        key["max_walls"] = options["max_walls"]
    return key


def _cache_path(cache_dir: str, path: str) -> str:
    return os.path.join(cache_dir, _slug(os.path.splitext(os.path.basename(path))[0]) + ".json")


def _cache_key_and_path(path: str, cache_dir: str, exclusions=None, stairs=None,
                        options: Optional[dict] = None) -> tuple[dict, str]:
    """A re-processed scan (`exclusions`, `stairs`) is cached under `edited/`, keyed on those too, so the plain
    cache stays as it was."""
    key = _cache_key(path, options)
    cpath = _cache_path(cache_dir, path)
    if exclusions or stairs:
        key["exclusions"] = json.loads(json.dumps([vars(e) for e in exclusions or []], default=list))
        key["stairs"] = json.loads(json.dumps([vars(e) for e in stairs or []], default=list))
        cpath = _cache_path(os.path.join(cache_dir, "edited"), path)
    return key, cpath


def load_cached_payload(path: str, cache_dir: str, exclusions=None, stairs=None,
                        options: Optional[dict] = None) -> Optional[dict]:
    """The saved preview if it is still valid, else None. Plain file reading, no Open3D."""
    key, cpath = _cache_key_and_path(path, cache_dir, exclusions, stairs, options)
    if not os.path.exists(cpath):
        return None
    try:
        with open(cpath) as fh:
            cached = json.load(fh)
        if cached.get("_cache_key") == key:
            return cached["payload"]
    except Exception:
        log.warning("%s: cache at %s unreadable, rebuilding", key["name"], cpath)
    return None


def build_scan_payload_cached(path: str, cache_dir: str, exclusions=None, stairs=None,
                              options: Optional[dict] = None) -> tuple[dict, bool]:
    """`build_scan_payload`, skipping the work if a valid cache exists. Returns (payload, was_cached)."""
    options = options or preview_options()
    cached = load_cached_payload(path, cache_dir, exclusions, stairs, options)
    if cached is not None:
        return cached, True
    key, cpath = _cache_key_and_path(path, cache_dir, exclusions, stairs, options)
    payload = build_scan_payload(path, exclusions=exclusions, stairs=stairs, options=options)
    write_json_atomic(cpath, {"_cache_key": key, "payload": payload})
    return payload, False


def _slug(name: str) -> str:
    """A short, readable id. Not used to find the file again; `name` is."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "scan"


_wall_segment = segment_row      # the measurement scripts call it by this name


def _floor_bands(height_above_floor: np.ndarray) -> list[dict]:
    """Histogram of heights above the floor, merged into a handful of bands (the "Floor height"
    panel): flat surfaces such as floor, seat backs, stage lip and ceiling show as peaks."""
    if height_above_floor.size == 0:
        return []
    lo, hi = float(height_above_floor.min()), float(height_above_floor.max())
    n_bins = max(1, int(np.ceil((hi - lo) / BAND_BIN_M)) + 1)
    counts, edges = np.histogram(height_above_floor, bins=n_bins, range=(lo, hi))

    bands = []
    i = 0
    threshold = max(30, int(0.002 * height_above_floor.size))
    while i < len(counts):
        if counts[i] < threshold:
            i += 1
            continue
        j = i
        total = 0
        weighted = 0.0
        while j < len(counts) and counts[j] >= threshold:
            total += int(counts[j])
            weighted += float(counts[j]) * (edges[j] + edges[j + 1]) / 2.0
            j += 1
        bands.append({"h": round(weighted / total, 3), "n": total})
        i = j
    bands.sort(key=lambda b: -b["n"])
    return bands[:12]


def build_scan_payload(path: str, preview_max_points: int = PREVIEW_MAX_POINTS, exclusions=None, stairs=None,
                       options: Optional[dict] = None) -> dict:
    """exclusions (this scan's, from the project file) are cut first when the scan was re-processed, so levelling,
    walls and bands are found without them. stairs (marked areas) stay in the preview but are left out of levelling,
    walls and bands, and are flagged in stairs. options comes from preview_options()."""
    options = options or preview_options()
    scan = read_pts(path)
    if exclusions:
        scan = filter_scan(scan, exclusions)
    pcd = scan.to_o3d()
    # Stage 1: fine fixed grid for level and walls; it must stay fine.
    down = pcd.voxel_down_sample(PREVIEW_VOXEL)
    if len(down.points) < 50:
        down = pcd  # a very sparse scan: use everything rather than nothing

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=PREVIEW_VOXEL * 4, max_nn=30)
    )

    on_stairs = excluded_mask(np.asarray(down.points), stairs) if stairs else np.zeros(len(down.points), dtype=bool)
    structure = down.select_by_index(np.flatnonzero(~on_stairs).tolist()) if on_stairs.any() else down
    level = estimate_level(structure, max_tilt_deg=options["level_tilt_deg"])
    down.rotate(level.rotation, center=(0, 0, 0))
    if structure is not down:
        structure.rotate(level.rotation, center=(0, 0, 0))

    xy_fine = np.asarray(down.points)[:, :2]
    height_fine = np.asarray(structure.points)[:, 2] - level.floor_z

    planes = plane_rows(structure, options["max_walls"])              # already rotated above. #drawn-walls
    wall_rows = drawn_walls(structure, level.floor_z, options["max_walls"], planes)
    if options["max_walls"] != PREVIEW_MAX_WALLS:     # thorough: the reports measure only planes with a solid run
        lifted = np.asarray(structure.points).copy()
        lifted[:, 2] -= level.floor_z
        solid = {r["i"] for r in wall_runs(lifted, planes)}
        planes = [w for w in planes if w["i"] in solid]

    polys, area = footprint(xy_fine)
    outline = [p.tolist() for p in polys]
    bbox = [
        round(float(xy_fine[:, 0].min()), 3), round(float(xy_fine[:, 1].min()), 3),
        round(float(xy_fine[:, 0].max()), 3), round(float(xy_fine[:, 1].max()), 3),
    ]
    # Height range above the floor, for the 3D bounding cube.
    height_all = np.asarray(down.points)[:, 2] - level.floor_z            # stairs included: the box holds them
    z_range = [round(float(height_all.min()), 3), round(float(height_all.max()), 3)]
    bands = _floor_bands(height_fine)

    # Stage 2: browser point budget, drawn points only.
    keep_idx, thin_frac = _density_preserving_preview(
        down, preview_max_points, _preview_rng(scan.name),
        bin_m=PREVIEW_BIN_M, floor_n=PREVIEW_FLOOR_N)
    down_display = down.select_by_index(keep_idx.tolist())
    display_pts = np.asarray(down_display.points)
    xy_all = display_pts[:, :2]
    height_above_floor = display_pts[:, 2] - level.floor_z

    n = xy_all.shape[0]
    if down_display.has_colors():
        rgb = np.clip(np.rint(np.asarray(down_display.colors) * 255.0), 0, 255).astype(int)
    else:
        rgb = np.full((n, 3), 160, dtype=int)

    # Agreement on the points sent to the browser; arrays sorted highest first.
    agreement = _agreement_scores(display_pts)
    order = np.argsort(-agreement)  # descending: highest agreement first
    xy_all = xy_all[order]
    height_above_floor = height_above_floor[order]
    rgb = rgb[order]
    agreement = agreement[order]
    # Preview points on marked stairs. select_by_index keeps the cloud's own order, not keep_idx's, hence the sort.
    stairs_idx = np.flatnonzero(on_stairs[np.sort(keep_idx)][order])

    return {
        "name": scan.name,                      # matches the on-disk filename,
                                                  # so place-merge can find it
        "points": scan.n_points,                 # FULL resolution count
        "tilt": round(float(level.tilt_deg), 3),
        "floor_z": round(float(level.floor_z), 4),
        # Saved back into the placement; without it every scan merges tilted.
        "level_rotation": [[round(float(v), 8) for v in row] for row in level.rotation.tolist()],
        "xy": [[round(float(x), 3), round(float(y), 3)] for x, y in xy_all],
        "h": [round(float(v), 3) for v in height_above_floor],
        "rgb": rgb.tolist(),
        # Local-flatness score, same length and order as xy/h/rgb; align.py weights ICP by it.
        "agreement": [round(float(v), 4) for v in agreement],
        "walls": wall_rows,
        "planes": planes,       # what the wall reports measure with (find_walls.plane_rows)
        "bands": bands,
        "outline": outline,
        "bbox": bbox,
        "zRange": z_range,
        "area": round(float(area), 2),
        # Preview points kept, and the fraction of each crowded bin's excess kept (0.0 = no thinning).
        "preview_points": int(n),
        "preview_thin_frac": round(float(thin_frac), 4),
        # indices into xy/h of points on marked stairs; height settle leaves them out
        **({"stairs": stairs_idx.tolist()} if len(stairs_idx) else {}),
    }


# Wire format only: flattens "xy" to two numbers per point for the browser. Used by the server and build().
def flatten_xy_for_wire(payload: dict) -> dict:
    if "xy" not in payload:
        return payload
    out = dict(payload)
    out["xy"] = [v for pair in payload["xy"] for v in pair]
    return out


def build(scans_folder: str, output_path: str, pattern: str = "*.pts",
         cache_dir: Optional[str] = None) -> None:
    """Bake a self-contained offline HTML file with every preview inlined as window.SCANDATA. Not used by the live
    server."""
    paths = find_scans(scans_folder, pattern)
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {scans_folder}")

    scans = []
    cached_count = 0
    for p in paths:
        log.info("processing %s", os.path.basename(p))
        if cache_dir:
            payload, was_cached = build_scan_payload_cached(p, cache_dir)
            cached_count += int(was_cached)
        else:
            payload = build_scan_payload(p)
        payload["ready"] = True
        scans.append(flatten_xy_for_wire(payload))

    data = {"scans": scans}
    payload_block = "<script>window.SCANDATA=" + json.dumps(data, separators=(",", ":")) + ";</script>\n"

    with open(TEMPLATE, "r", encoding="utf-8") as fh:
        html = fh.read()

    # Same build-identity badge as the live server; mode says standalone.
    build_info = {
        "version": __version__,
        "build_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "standalone (offline export, no server)",
    }
    html = html.replace(
        "<!--BUILD_INFO-->",
        "<script>window.BUILD_INFO=" + json.dumps(build_info) + ";</script>",
        1,  # count=1: an unqualified replace() would also hit the marker
            # text quoted inside a JS comment later in the same file
    )

    # Insert the SCANDATA block before the template's first <script> so it is global by then.
    marker = "<script>"
    idx = html.find(marker)
    if idx < 0:
        raise RuntimeError(f"{TEMPLATE}: no <script> tag found to insert data before")
    html = html[:idx] + payload_block + html[idx:]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    total_full = sum(s["points"] for s in scans)
    total_preview = sum(s["preview_points"] for s in scans)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"wrote {output_path}")
    print(f"{len(scans)} scan(s), {total_full:,} full-resolution points -> "
          f"{total_preview:,} preview points (max {PREVIEW_MAX_POINTS:,} per "
          f"scan, real points kept at random with a density floor - see "
          f"per-scan log lines above)")
    for s in scans:
        frac = s["preview_thin_frac"]
        detail = "no thinning, already under budget" if frac == 0 else f"kept {frac*100:.0f}% of each crowded bin's excess"
        print(f"  {s['name']}: {s['preview_points']:,} pts ({detail})")
    if cache_dir:
        print(f"{cached_count} of {len(scans)} scan(s) loaded from cache at {cache_dir}")
    print(f"file size: {size_mb:.1f} MB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scans_folder")
    ap.add_argument("-o", "--output", default="placement_bench.html")
    ap.add_argument("--pattern", default="*.pts")
    ap.add_argument("--cache-dir", default=None,
                    help="reuse cached previews from a prior server run or build")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    build(args.scans_folder, args.output, args.pattern, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
