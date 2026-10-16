"""The plan-view drawing: a horizontal cut through the placed scans at full resolution, traced into outlines and
written as a DXF to draw over. Units are metres."""
from __future__ import annotations

import logging
import os
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .exclusions import filter_scan
from .landmark_boxes import label_points
from .placement import Exclusion, Placement, tilt_matrix3
from .pts_file import Scan

log = logging.getLogger(__name__)

CELL_M = 0.02              # raster cell: the finest detail the outlines can show
MIN_POINTS_PER_CELL = 2    # a cell needs this many points in the cut before it counts as something there
CLOSE_CELLS = 2            # gaps this small (cells) are bridged so a wall is one shape
SIMPLIFY_M = 0.02          # outlines are straightened to within this
MIN_SHAPE_M = 0.30         # shapes smaller than this in both directions are specks, not architecture
MAX_CELLS = 60_000_000     # past this the cell grows rather than the raster
DEFAULT_ABOVE_FLOOR_M = (1.0, 1.6)


@dataclass
class PlanCut:
    polylines: list[np.ndarray] = field(default_factory=list)   # closed outlines, (n, 2) world metres
    footprints: list[np.ndarray] = field(default_factory=list)  # each scan's rough floor outline, world metres
    labels: list[tuple[str, float, float]] = field(default_factory=list)
    preview: Optional[np.ndarray] = None                         # uint8 image of the cut
    z_lo: float = 0.0
    z_hi: float = 0.0
    cell_m: float = CELL_M
    bounds: tuple = (0.0, 0.0, 0.0, 0.0)
    points_in_cut: int = 0
    scans_used: int = 0
    landmarks: list = field(default_factory=list)                # landmark_boxes.LandmarkBox, named in review


def default_band(placements: dict[str, Placement]) -> tuple[float, float]:
    """1.0 to 1.6 m above the typical scan's floor, in the bench's world height."""
    floors = [p.tz for p in placements.values()]
    base = float(np.median(floors)) if floors else 0.0
    return base + DEFAULT_ABOVE_FLOOR_M[0], base + DEFAULT_ABOVE_FLOOR_M[1]


def cut_points(scan: Scan, placement: Placement, z_lo: float, z_hi: float,
               crop: Optional[tuple] = None) -> np.ndarray:
    """The scan's points inside the band, as world x, y. Every point is considered, none are thinned."""
    m = placement.matrix()
    world = scan.points @ m[:3, :3].T + m[:3, 3]
    keep = (world[:, 2] >= z_lo) & (world[:, 2] <= z_hi)
    if crop is not None:
        x_lo, x_hi, y_lo, y_hi = crop
        keep &= (world[:, 0] >= x_lo) & (world[:, 0] <= x_hi) & (world[:, 1] >= y_lo) & (world[:, 1] <= y_hi)
    return world[keep, :2].astype(np.float32)


def trace_outlines(xy: np.ndarray, cell_m: float = CELL_M, min_points: int = MIN_POINTS_PER_CELL,
                   close_cells: int = CLOSE_CELLS, simplify_m: float = SIMPLIFY_M,
                   min_shape_m: float = MIN_SHAPE_M):
    """Raster the cut points, bridge tiny gaps, trace every shape's outline and straighten it.
    Returns (outlines in world metres, preview image, bounds, the cell size actually used)."""
    if len(xy) == 0:
        return [], np.zeros((1, 1), np.uint8), (0.0, 0.0, 0.0, 0.0), cell_m
    lo = xy.min(axis=0) - 0.2
    hi = xy.max(axis=0) + 0.2
    while ((hi - lo) / cell_m).prod() > MAX_CELLS:
        cell_m *= 1.5
    w, h = (np.ceil((hi - lo) / cell_m)).astype(int) + 1
    ix = np.clip(((xy[:, 0] - lo[0]) / cell_m).astype(np.int64), 0, w - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / cell_m).astype(np.int64), 0, h - 1)
    counts = np.bincount(iy * w + ix, minlength=w * h).reshape(h, w)
    occupied = (counts >= min_points).astype(np.uint8)
    if close_cells > 0:
        occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((close_cells * 2 + 1,) * 2, np.uint8))
    found, _ = cv2.findContours(occupied, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for c in found:
        c = cv2.approxPolyDP(c, max(simplify_m / cell_m, 0.75), True)[:, 0, :].astype(float)
        if len(c) < 3:
            continue
        world = np.column_stack([lo[0] + (c[:, 0] + 0.5) * cell_m, lo[1] + (c[:, 1] + 0.5) * cell_m])
        if max(np.ptp(world[:, 0]), np.ptp(world[:, 1])) >= min_shape_m:
            outlines.append(world)
    return outlines, occupied * 255, (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), cell_m


def plan_cut(scans: list[Scan], placements: dict[str, Placement], exclusions: list[Exclusion],
             z_lo: float, z_hi: float, crop: Optional[tuple] = None, progress_cb=None,
             should_cancel: Optional[Callable[[], bool]] = None) -> PlanCut:
    """The whole cut: every placed scan, at full resolution, through the placement the bench draws.
    `should_cancel`, if given, is checked once per scan - the loop below reads every one at full
    resolution, which is where a large plan export actually spends its time."""
    parts = []
    result = PlanCut(z_lo=z_lo, z_hi=z_hi)
    for i, scan in enumerate(scans):
        if scan.name not in placements:
            continue
        if should_cancel and should_cancel():
            raise CancelledError(f"cancelled after cutting {i}/{len(scans)} scan(s)")
        if progress_cb:
            progress_cb("cutting", f"{i + 1}/{len(scans)}: {scan.name}")
        s = filter_scan(scan, exclusions) if exclusions else scan
        xy = cut_points(s, placements[scan.name], z_lo, z_hi, crop)
        if len(xy):
            parts.append(xy)
            result.scans_used += 1
        p = placements[scan.name]
        result.labels.append((scan.name, p.tx, p.ty))
    xy_all = np.concatenate(parts) if parts else np.zeros((0, 2), np.float32)
    result.points_in_cut = int(len(xy_all))
    if progress_cb:
        progress_cb("tracing", f"{len(xy_all):,} points in the cut")
    result.polylines, result.preview, result.bounds, result.cell_m = trace_outlines(xy_all)
    return result


def footprints_from_payloads(payloads: dict[str, dict], placements: dict[str, Placement]) -> list[np.ndarray]:
    """Each scan's rough floor outline (already found for the bench), placed like the scan itself."""
    out = []
    for name, payload in payloads.items():
        p = placements.get(name)
        if p is None:
            continue
        rot = tilt_matrix3(p.pitch_deg, p.roll_deg, p.yaw_deg)
        for poly in payload.get("outline", []):
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            world = np.column_stack([pts, np.zeros(len(pts))]) @ rot.T + np.array([p.tx, p.ty, p.tz])
            out.append(world[:, :2])
    return out


# --- the DXF file itself -------------------------------------------------------------------------------
# R12 ASCII: the oldest form every CAD program reads, no handles or object tables to get wrong.

LAYERS = (("PLAN_CUT", 7), ("SCAN_FOOTPRINTS", 8), ("SCAN_NAMES", 3), ("LANDMARKS", 3), ("INFO", 2))


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def dxf_text(cut: PlanCut, note: str = "") -> str:
    """The DXF as text. Layers: PLAN_CUT (the traced outlines), SCAN_FOOTPRINTS, SCAN_NAMES, INFO."""
    out: list[str] = []

    def add(code: int, value) -> None:
        out.append(f"{code:3d}")
        out.append(str(value))

    add(0, "SECTION"); add(2, "HEADER"); add(9, "$ACADVER"); add(1, "AC1009")
    add(9, "$INSBASE"); add(10, 0.0); add(20, 0.0); add(30, 0.0)
    add(0, "ENDSEC")
    add(0, "SECTION"); add(2, "TABLES")
    add(0, "TABLE"); add(2, "LTYPE"); add(70, 1)
    add(0, "LTYPE"); add(2, "CONTINUOUS"); add(70, 0); add(3, "Solid line"); add(72, 65); add(73, 0); add(40, 0.0)
    add(0, "ENDTAB")
    add(0, "TABLE"); add(2, "LAYER"); add(70, len(LAYERS))
    for name, colour in LAYERS:
        add(0, "LAYER"); add(2, name); add(70, 0); add(62, colour); add(6, "CONTINUOUS")
    add(0, "ENDTAB")
    add(0, "ENDSEC")
    add(0, "SECTION"); add(2, "ENTITIES")

    def polyline(points: np.ndarray, layer: str, closed: bool) -> None:
        add(0, "POLYLINE"); add(8, layer); add(66, 1); add(70, 1 if closed else 0)
        for x, y in points:
            add(0, "VERTEX"); add(8, layer); add(10, f"{x:.4f}"); add(20, f"{y:.4f}"); add(30, "0.0")
        add(0, "SEQEND"); add(8, layer)

    def text(x: float, y: float, height: float, value: str, layer: str) -> None:
        add(0, "TEXT"); add(8, layer); add(10, f"{x:.4f}"); add(20, f"{y:.4f}"); add(30, "0.0")
        add(40, height); add(1, _ascii(value))

    for poly in cut.polylines:
        polyline(poly, "PLAN_CUT", True)
    for poly in cut.footprints:
        polyline(poly, "SCAN_FOOTPRINTS", True)
    for name, x, y in cut.labels:
        text(x, y, 0.35, name, "SCAN_NAMES")
    for box in cut.landmarks:                            # each named landmark's box from above. #named-landmarks
        polyline(cv2.convexHull(box.corners[:, :2].astype(np.float32))[:, 0, :], "LANDMARKS", True)
    for label, c in label_points(cut.landmarks):
        text(float(c[0]), float(c[1]), 0.2, label, "LANDMARKS")
    x0, y0, _, _ = cut.bounds
    text(x0, y0 - 1.0, 0.4, f"Plan cut {cut.z_lo:.2f} to {cut.z_hi:.2f} m, units metres, {cut.scans_used} scans"
         + (f", {note}" if note else ""), "INFO")
    add(0, "ENDSEC")
    add(0, "EOF")
    return "\n".join(out) + "\n"


def preview_png(cut: PlanCut, max_side: int = 2400) -> bytes:
    """A picture of the cut with the traced outlines, to check by eye. North (+y) is up."""
    img = cut.preview
    if img is None or img.size < 4:
        return b""
    h, w = img.shape
    scale = min(1.0, max_side / max(h, w))
    x0, y0 = cut.bounds[0], cut.bounds[1]
    canvas = cv2.cvtColor(np.flipud(img), cv2.COLOR_GRAY2BGR)
    canvas[canvas[:, :, 0] > 0] = (70, 70, 70)
    for poly in cut.polylines:
        px = np.column_stack([(poly[:, 0] - x0) / cut.cell_m, h - 1 - (poly[:, 1] - y0) / cut.cell_m])
        cv2.polylines(canvas, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 1)
    for name, x, y in cut.labels:
        cv2.putText(canvas, name.replace("Point Cloud ", "").replace("2026-", "", 1)[:28], (int((x - x0) / cut.cell_m), int(h - 1 - (y - y0) / cut.cell_m)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 120), 1, cv2.LINE_AA)
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", canvas)
    return png.tobytes() if ok else b""


def write_plan(cut: PlanCut, folder: str, note: str = "", stem: str = "plan_view") -> dict:
    """Write <stem>.dxf and <stem>.png into `folder`; returns what was written."""
    os.makedirs(folder, exist_ok=True)
    dxf_path = os.path.join(folder, stem + ".dxf")
    with open(dxf_path, "w", encoding="ascii", newline="\r\n") as fh:
        fh.write(dxf_text(cut, note))
    png_path = os.path.join(folder, stem + ".png")
    png = preview_png(cut)
    if png:
        with open(png_path, "wb") as fh:
            fh.write(png)
    return {"dxf_path": dxf_path, "png_path": png_path if png else None, "outlines": len(cut.polylines),
            "footprints": len(cut.footprints), "points_in_cut": cut.points_in_cut, "scans_used": cut.scans_used,
            "z_lo": round(cut.z_lo, 3), "z_hi": round(cut.z_hi, 3), "cell_m": round(cut.cell_m, 4)}
