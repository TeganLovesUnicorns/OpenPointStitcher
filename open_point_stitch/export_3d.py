"""Floors, ceilings (flat, dropped, sloped) and walls of the placed scans as 3D planes, in one DXF in metres.
Where several scans saw the same surface it is drawn once, and the file records how far apart the scans had it."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .lib import geometry
from .lib.files import write_drawing
from .lib.groups import connected_groups
from .placement import Placement, label_points

CELL_M = 0.05             # plan grid the pieces of one surface are joined on
SAME_PLANE_M = 0.05       # pieces from different scans this close in height where they overlap are one surface
SAME_TILT_DEG = 3.0       # ... and tilted within this of each other
WALL_JOIN_GAP_M = 0.30    # wall pieces on one line with gaps this small are one wall
WALL_WORDS = {"full": "wall", "half wall": "half wall", "hanging": "wall hanging from the ceiling",
              "behind curtain": "wall behind a curtain"}
WALL_SNAP_M = 0.15       # a wall ending this close to the floor or ceiling piece is drawn to it
SIMPLIFY_M = 0.05         # joined outlines are straightened to within this
LAYERS = (("FLOOR", 5), ("CEILING", 30), ("CEILING_SLOPED", 40), ("FLAT_SURFACES", 8), ("WALLS", 7),
          ("LANDMARKS", 3), ("LABELS", 3), ("INFO", 2))


@dataclass
class Surface:
    """One floor, ceiling, flat surface or wall, in the venue's frame (metres)."""

    kind: str                                  # "floor", "ceiling", "surface" or "wall"
    normal: np.ndarray                         # unit; up for floors and ceilings, level for walls
    offset: float                              # plane: normal . p = offset
    outlines: list = field(default_factory=list)   # closed 3D outlines, each (n, 3), on the plane
    scans: list = field(default_factory=list)
    spread_m: float = 0.0                      # how far apart the scans that saw it had it
    rms_m: float = 0.0                         # how far its points sit from the plane, worst scan
    label: str = ""
    sloped: bool = False
    wall_kind: str = "full"                    # walls: "full", "half wall", "hanging" or "behind curtain"

    @property
    def layer(self) -> str:
        if self.kind == "ceiling":
            return "CEILING_SLOPED" if self.sloped else "CEILING"
        return {"floor": "FLOOR", "surface": "FLAT_SURFACES", "wall": "WALLS"}[self.kind]


def _frame(p: Placement) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, shift) taking a point in the scan's levelled frame (height above its floor) to the venue."""
    m = p.matrix()
    tilt = m[:3, :3] @ np.asarray(p.level_rotation).T
    return tilt, m[:3, 3] + tilt @ np.array([0.0, 0.0, p.floor_z])


def _z_on(normal, offset, xy: np.ndarray) -> np.ndarray:
    return (offset - xy[:, 0] * normal[0] - xy[:, 1] * normal[1]) / normal[2]


def scan_surfaces(name: str, pieces: list[dict], walls: list[dict], placement: Placement) -> list[Surface]:
    """One scan's pieces and walls, moved into the venue. A wall runs from the floor piece under its middle to the
    ceiling piece over it; with neither found, from the scan's floor to the wall's own measured height."""
    rot, shift = _frame(placement)
    out = []
    for pc in pieces:
        n, off = np.asarray(pc["normal"], float), float(pc["offset"])
        polys = [np.asarray(o, float) for o in pc["outlines"]]
        local = [np.column_stack([q, _z_on(n, off, q)]) for q in polys]
        wn = rot @ n
        world = [q @ rot.T + shift for q in local]
        out.append(Surface(kind=pc["kind"], normal=wn, offset=float(wn @ world[0][0]), outlines=world, scans=[name],
                           rms_m=float(pc.get("rms_m", 0.0)), label=pc.get("label", ""),
                           sloped=float(pc.get("slope_deg", 0.0)) > 3.0))
    floors = [pc for pc in pieces if pc["kind"] == "floor"]
    ceils = [pc for pc in pieces if pc["kind"] == "ceiling"]
    for w in walls:
        a, b = np.array([w["x1"], w["y1"]], float), np.array([w["x2"], w["y2"]], float)
        if np.linalg.norm(b - a) < 0.05:
            continue
        mid = (a + b) / 2
        floor_z = _height_near(floors, mid, default=0.0)
        ceil_z = _height_near(ceils, mid, default=floor_z + float(w.get("h", 2.4)))
        if w.get("z_lo") is None:                     # behind a curtain, or not measured: floor to ceiling
            lo, hi = floor_z, ceil_z
        else:                                         # its own bottom and top, onto the floor or ceiling when that close
            lo = floor_z if abs(w["z_lo"] - floor_z) < WALL_SNAP_M else float(w["z_lo"])
            hi = ceil_z if abs(w["z_hi"] - ceil_z) < WALL_SNAP_M else float(w["z_hi"])
        local = np.array([[*a, lo], [*b, lo], [*b, hi], [*a, hi]])
        world = local @ rot.T + shift
        d = (b - a) / np.linalg.norm(b - a)
        wn = rot @ np.array([d[1], -d[0], 0.0])
        if wn[0] < -1e-9 or (abs(wn[0]) <= 1e-9 and wn[1] < 0):   # one way round, so scans' copies can be joined
            wn = -wn
        out.append(Surface(kind="wall", normal=wn, offset=float(wn @ world[0]), outlines=[world], scans=[name],
                           wall_kind=w.get("kind", "full")))
    return out


def _height_near(pieces: list[dict], xy: np.ndarray, default: float) -> float:
    """The plane height above `xy` of the piece whose outline comes nearest it (within 1.5 m), else `default`."""
    best, best_d = default, 1.5
    for pc in pieces:
        for o in pc["outlines"]:
            o = np.asarray(o, float)
            d = float(np.min(np.linalg.norm(o - xy, axis=1)))
            inside = cv2.pointPolygonTest(o.astype(np.float32).reshape(-1, 1, 2), (float(xy[0]), float(xy[1])), False)
            if inside >= 0:
                d = 0.0
            if d < best_d:
                best, best_d = float(_z_on(np.asarray(pc["normal"]), pc["offset"], xy[None])[0]), d
    return best


def _plan_grid(surfaces: list[Surface]):
    lo = np.min([o[:, :2].min(axis=0) for s in surfaces for o in s.outlines], axis=0) - 2 * CELL_M
    hi = np.max([o[:, :2].max(axis=0) for s in surfaces for o in s.outlines], axis=0) + 2 * CELL_M
    size = np.ceil((hi - lo) / CELL_M).astype(int) + 1
    return lo, size


def _raster(s: Surface, lo: np.ndarray, size: np.ndarray) -> np.ndarray:
    g = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(g, [np.round((o[:, :2] - lo) / CELL_M).astype(np.int32) for o in s.outlines], 1)
    return g


def _join_flat(surfaces: list[Surface]) -> list[Surface]:
    """Floors, ceilings and flat surfaces from different scans that overlap in plan and agree in height and tilt
    become one, outlined around all of them, on their area-weighted mean plane."""
    if not surfaces:
        return []
    lo, size = _plan_grid(surfaces)
    rasters = [_raster(s, lo, size) for s in surfaces]

    def same(i: int, j: int) -> bool:
        a, b = surfaces[i], surfaces[j]
        if a.kind != b.kind or geometry.angle_between(a.normal, b.normal) > SAME_TILT_DEG:
            return False
        both = rasters[i] & rasters[j]
        if not both.any():
            return False
        rc = np.argwhere(both)[:, ::-1] * CELL_M + lo
        return bool(np.median(np.abs(_z_on(a.normal, a.offset, rc) - _z_on(b.normal, b.offset, rc))) <= SAME_PLANE_M)

    out = []
    for members in connected_groups(len(surfaces), linked=same):
        ss = [surfaces[i] for i in members]
        cells = np.zeros_like(rasters[members[0]])
        for i in members:
            cells |= rasters[i]
        rows, cols = np.nonzero(cells)
        rc = np.column_stack([cols, rows]) * CELL_M + lo
        seen = np.array([rasters[i][rows, cols] > 0 for i in members])           # which scan saw which cell
        heights = np.array([_z_on(s.normal, s.offset, rc) for s in ss])
        # The scans' own planes averaged, each by how much it saw. A plane fitted through the patchwork instead
        # would tilt wherever one scan sits a little higher than the next.
        w = seen.sum(axis=1).astype(float)
        n = np.sum([s.normal * k for s, k in zip(ss, w)], axis=0)
        n /= np.linalg.norm(n)
        mid = rc.mean(axis=0)
        z_mid = float(np.average([_z_on(s.normal, s.offset, mid[None])[0] for s in ss], weights=w))
        off = float(n @ np.array([mid[0], mid[1], z_mid]))
        both = seen.sum(axis=0) >= 2
        hs = np.where(seen, heights, np.nan)[:, both]
        spread = float(np.median(np.nanmax(hs, axis=0) - np.nanmin(hs, axis=0))) if both.any() else 0.0
        found, _ = cv2.findContours(cells, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines = []
        for c in found:
            c = cv2.approxPolyDP(c, SIMPLIFY_M / CELL_M, True)[:, 0, :].astype(float)
            if len(c) < 3:
                continue
            xy = c * CELL_M + lo
            outlines.append(np.column_stack([xy, _z_on(n, off, xy)]))
        if not outlines:
            continue
        big = ss[int(np.argmax(seen.sum(axis=1)))]
        out.append(Surface(kind=big.kind, normal=n, offset=off, outlines=outlines,
                           scans=sorted({x for s in ss for x in s.scans}), spread_m=spread,
                           rms_m=max(s.rms_m for s in ss), sloped=big.sloped,
                           label=_label(big.kind, big.sloped, n, outlines)))
    return out


def _label(kind: str, sloped: bool, n: np.ndarray, outlines: list[np.ndarray]) -> str:
    z = np.concatenate([o[:, 2] for o in outlines])
    tilt = geometry.angle_between(n, np.array([0, 0, 1.0]))
    if kind == "surface":
        return f"flat surface {_m(z.min())} to {_m(z.max())} m, tilted {tilt:.0f} deg" if sloped \
            else f"flat surface {_m(float(np.median(z)))} m"
    if sloped:
        return f"sloped {kind}, {tilt:.0f} deg, {_m(z.min())} to {_m(z.max())} m"
    return f"{kind} {_m(float(np.median(z)))} m"


def _m(v: float) -> str:
    return f"{v + 0.0:.2f}".replace("-0.00", "0.00")


def _join_walls(walls: list[Surface]) -> list[Surface]:
    """Wall faces from different scans on the same line, facing the same way, that overlap or nearly touch, become
    one face from the lowest bottom to the highest top."""
    used = [False] * len(walls)
    out = []
    for i, a in enumerate(walls):
        if used[i]:
            continue
        used[i] = True
        group = [a]
        grew = True
        while grew:
            grew = False
            for j, b in enumerate(walls):
                if used[j] or b.wall_kind != a.wall_kind or geometry.angle_between(a.normal, b.normal) > SAME_TILT_DEG:
                    continue
                if abs(float(a.normal @ b.outlines[0][0]) - a.offset) > SAME_PLANE_M:
                    continue
                if _gap_along(group, b, a.normal) <= WALL_JOIN_GAP_M:
                    used[j] = True
                    group.append(b)
                    grew = True
        along = np.array([-a.normal[1], a.normal[0], 0.0])
        along /= np.linalg.norm(along)
        pts = np.concatenate([s.outlines[0] for s in group])
        t = pts @ along
        n = a.normal / np.linalg.norm(a.normal)
        off = float(np.mean([s.offset for s in group]))
        base = n * off
        z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
        ends = [base - along * float(base @ along) + along * v for v in (t.min(), t.max())]
        quad = np.array([[*ends[0][:2], z0], [*ends[1][:2], z0], [*ends[1][:2], z1], [*ends[0][:2], z1]])
        spread = float(np.ptp([s.offset for s in group])) if len(group) > 1 else 0.0
        out.append(Surface(kind="wall", normal=n, offset=off, outlines=[quad],
                           scans=sorted({x for s in group for x in s.scans}), spread_m=spread, wall_kind=a.wall_kind,
                           label=f"{WALL_WORDS[a.wall_kind]} {float(np.ptp(t)):.2f} m long, {_m(z0)} to {_m(z1)} m"))
    return out


def _gap_along(group: list[Surface], b: Surface, normal: np.ndarray) -> float:
    """How far b's extent along the wall line is from the group's (0 when they overlap)."""
    along = np.array([-normal[1], normal[0], 0.0])
    tg = np.concatenate([s.outlines[0] @ along for s in group])
    tb = b.outlines[0] @ along
    return max(0.0, max(tb.min() - tg.max(), tg.min() - tb.max()))


def venue_surfaces(per_scan: dict[str, list[Surface]]) -> list[Surface]:
    """Every scan's surfaces joined across scans: what the 3D file draws."""
    flat = [s for ss in per_scan.values() for s in ss if s.kind != "wall"]
    walls = [s for ss in per_scan.values() for s in ss if s.kind == "wall"]
    joined = _join_flat(flat) + _join_walls(walls)
    order = {"floor": 0, "surface": 1, "ceiling": 2, "wall": 3}
    return sorted(joined, key=lambda s: (order[s.kind], s.label))


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def dxf_text(surfaces: list[Surface], note: str = "", landmarks: list | None = None) -> str:
    """The DXF as text: each surface a closed 3D polyline on its plane, on a layer by kind, with its label; each named
    landmark (placement.LandmarkBox) the bottom and top of its box on LANDMARKS, named once."""
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
    def poly3d(o: np.ndarray, layer: str) -> None:
        add(0, "POLYLINE"); add(8, layer); add(66, 1); add(10, 0.0); add(20, 0.0); add(30, 0.0)
        add(70, 9)                                               # closed (1) and 3D (8)
        for x, y, z in o:
            add(0, "VERTEX"); add(8, layer); add(10, f"{x:.4f}"); add(20, f"{y:.4f}"); add(30, f"{z:.4f}")
            add(70, 32)                                          # a 3D polyline's vertex
        add(0, "SEQEND"); add(8, layer)

    def label(c, height: float, value: str, layer: str) -> None:
        add(0, "TEXT"); add(8, layer); add(10, f"{c[0]:.4f}"); add(20, f"{c[1]:.4f}"); add(30, f"{c[2]:.4f}")
        add(40, height); add(1, _ascii(value))

    for s in surfaces:
        for o in s.outlines:
            poly3d(o, s.layer)
        text = s.label + (f" (scans agree to {s.spread_m * 100:.0f} cm)" if len(s.scans) > 1 else "")
        label(np.concatenate(s.outlines).mean(axis=0), 0.15, text, "LABELS")
    for box in landmarks or []:
        poly3d(box.corners[:4], "LANDMARKS")
        poly3d(box.corners[4:], "LANDMARKS")
    for text, c in label_points(landmarks or []):
        label(c, 0.15, text, "LANDMARKS")
    if surfaces:
        lo = np.min([o.min(axis=0) for s in surfaces for o in s.outlines], axis=0)
        add(0, "TEXT"); add(8, "INFO"); add(10, f"{lo[0]:.4f}"); add(20, f"{lo[1] - 1.0:.4f}"); add(30, f"{lo[2]:.4f}")
        add(40, 0.4); add(1, _ascii("Floors, ceilings and walls as 3D planes, units metres" + (f", {note}" if note else "")))
    add(0, "ENDSEC")
    add(0, "EOF")
    return "\n".join(out) + "\n"


def preview_png(surfaces: list[Surface], px_per_m: float = 40.0) -> bytes:
    """Seen from above, to check by eye: floors blue, ceilings orange (sloped darker), flat surfaces grey, walls
    white lines. North (+y) is up."""
    flat = [s for s in surfaces if s.kind != "wall"]
    if not surfaces:
        return b""
    pts = np.concatenate([o[:, :2] for s in surfaces for o in s.outlines])
    lo = pts.min(axis=0) - 0.5
    size = np.ceil((pts.max(axis=0) + 0.5 - lo) * px_per_m).astype(int)
    img = np.full((size[1], size[0], 3), 20, np.uint8)
    colour = {"FLOOR": (230, 140, 70), "CEILING": (60, 150, 255), "CEILING_SLOPED": (40, 100, 200),
              "FLAT_SURFACES": (150, 150, 150)}
    to_px = lambda o: np.column_stack([(o[:, 0] - lo[0]) * px_per_m,
                                       size[1] - 1 - (o[:, 1] - lo[1]) * px_per_m]).round().astype(np.int32)
    for s in sorted(flat, key=lambda s: s.kind != "floor"):
        layer = np.zeros_like(img)
        cv2.fillPoly(layer, [to_px(o) for o in s.outlines], colour[s.layer])
        img = cv2.addWeighted(img, 1.0, layer, 0.45, 0)
    for s in surfaces:
        if s.kind == "wall":
            cv2.polylines(img, [to_px(s.outlines[0][:2])], False, (255, 255, 255), 2)
    ok, png = cv2.imencode(".png", img)
    return png.tobytes() if ok else b""


def write_3d(surfaces: list[Surface], folder: str, note: str = "", stem: str = "surfaces_3d",
             landmarks: list | None = None) -> dict:
    """Write <stem>.dxf and a top-down <stem>.png into `folder`; returns what was written."""
    dxf_path, png_path = write_drawing(folder, stem, dxf_text(surfaces, note, landmarks), preview_png(surfaces))
    count = {k: sum(1 for s in surfaces if s.kind == k) for k in ("floor", "ceiling", "surface", "wall")}
    shared = [s.spread_m for s in surfaces if len(s.scans) > 1]
    return {"dxf_path": dxf_path, "png_path": png_path, **{f"n_{k}": v for k, v in count.items()},
            "n_sloped": sum(1 for s in surfaces if s.kind == "ceiling" and s.sloped),
            "spread_median_cm": round(float(np.median(shared)) * 100, 1) if shared else None,
            "labels": [s.label for s in surfaces if s.kind != "wall"]}
