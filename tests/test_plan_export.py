"""Tests for export_plan.py: a room with a known shape and placement comes out as outlines of that shape, in the
right place, in a DXF that reads back."""
from __future__ import annotations

import os
import sys
import tempfile
from concurrent.futures import CancelledError

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.export_plan import default_band, dxf_text, plan_cut, trace_outlines, write_plan  # noqa: E402
from open_point_stitch.placement import Placement  # noqa: E402
from open_point_stitch.lib.poses import yaw_matrix2  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402

ROOM = (6.0, 4.0)        # metres, x by y, in the scan's own frame
FLOOR_Z = -1.4           # where this scan's floor sits in its own levelled frame
CEILING_GAP = 2.5        # floor to ceiling


def _room_scan(seed: int = 0, name: str = "room") -> Scan:
    """Four walls, a floor and a ceiling, walls about 2 cm thick, phone origin 1.4 m above the floor."""
    rng = np.random.default_rng(seed)
    w, d = ROOM
    pts = []
    for (x0, y0, x1, y1) in ((-w / 2, -d / 2, w / 2, -d / 2), (w / 2, -d / 2, w / 2, d / 2),
                             (w / 2, d / 2, -w / 2, d / 2), (-w / 2, d / 2, -w / 2, -d / 2)):
        n = int(np.hypot(x1 - x0, y1 - y0) * 4000)
        t = rng.uniform(0, 1, n)
        z = rng.uniform(FLOOR_Z, FLOOR_Z + CEILING_GAP, n)
        pts.append(np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0), z]) + rng.normal(0, 0.008, (n, 3)) * [1, 1, 0])
    n = 60000
    pts.append(np.column_stack([rng.uniform(-w / 2, w / 2, n), rng.uniform(-d / 2, d / 2, n), np.full(n, FLOOR_Z)]))
    pts.append(np.column_stack([rng.uniform(-w / 2, w / 2, n), rng.uniform(-d / 2, d / 2, n), np.full(n, FLOOR_Z + CEILING_GAP)]))
    return Scan(name=name, path=f"{name}.pts", points=np.vstack(pts))


def _placement(name: str = "room") -> Placement:
    return Placement(name=name, floor_z=FLOOR_Z, yaw_deg=30.0, tx=12.0, ty=-5.0, tz=1.0)


def _read_dxf(text: str):
    """Group-code pairs to a list of (entity type, layer, [vertex or text data])."""
    lines = [l.strip() for l in text.splitlines()]
    pairs = list(zip(lines[0::2], lines[1::2]))
    ents, cur = [], None
    in_entities = False
    for code, val in pairs:
        if code == "2" and val == "ENTITIES":
            in_entities = True
            continue
        if not in_entities:
            continue
        if code == "0":
            if val == "ENDSEC":
                break
            cur = {"type": val, "layer": None, "xy": []}
            if val != "SEQEND":
                ents.append(cur)
            continue
        if cur is None:
            continue
        if code == "8":
            cur["layer"] = val
        elif code == "10":
            cur.setdefault("_x", float(val))
        elif code == "20" and "_x" in cur:
            cur["xy"].append((cur.pop("_x"), float(val)))
    return ents


def _expected_outer_corners():
    w, d = ROOM
    corners = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    return corners @ yaw_matrix2(30.0).T + [12.0, -5.0]


def test_should_cancel_stops_between_scans_and_raises_cancelled_error():
    # Checked once per scan in the full-resolution cut loop, where a many-scan plan export spends its time.
    band = (2.0, 2.6)
    scans = [_room_scan(name="room_a"), _room_scan(name="room_b")]
    placements = {"room_a": _placement("room_a"), "room_b": _placement("room_b")}
    seen = []
    try:
        plan_cut(scans, placements, [], *band, should_cancel=lambda: len(seen) >= 1,
                progress_cb=lambda phase, detail: seen.append(detail))
        raise AssertionError("expected CancelledError, plan_cut returned normally")
    except CancelledError as exc:
        assert "1/2" in str(exc), f"should say how far it got: {exc}"
    assert len(seen) == 1, f"cut a second scan after should_cancel already said yes: {seen}"


def test_a_room_comes_out_as_the_outline_of_its_walls_in_the_right_place():
    band = (2.0, 2.6)   # 1.0 to 1.6 m above this scan's floor, which sits at world height 1.0
    cut = plan_cut([_room_scan()], {"room": _placement()}, [], *band)
    assert cut.points_in_cut > 10000 and cut.scans_used == 1
    assert len(cut.polylines) == 2, f"a walled room cuts to an outer and an inner outline, got {len(cut.polylines)}"
    outer = max(cut.polylines, key=lambda p: np.ptp(p[:, 0]) + np.ptp(p[:, 1]))
    want = _expected_outer_corners()
    assert abs(outer[:, 0].min() - want[:, 0].min()) < 0.06 and abs(outer[:, 0].max() - want[:, 0].max()) < 0.06
    assert abs(outer[:, 1].min() - want[:, 1].min()) < 0.06 and abs(outer[:, 1].max() - want[:, 1].max()) < 0.06
    inner = min(cut.polylines, key=lambda p: np.ptp(p[:, 0]) + np.ptp(p[:, 1]))
    assert np.ptp(inner[:, 0]) < np.ptp(outer[:, 0]) and np.ptp(inner[:, 1]) < np.ptp(outer[:, 1])


def test_the_floor_level_decides_which_slice_of_the_room_is_cut():
    """The scan's floor sits at world height 1.0, so a thin slice at 1.0 cuts the floor (one filled shape) and 1.0
    to 1.6 above it cuts the walls (a ring). If the floor level were ignored the floor would land 1.4 m lower."""
    ring = plan_cut([_room_scan()], {"room": _placement()}, [], 2.0, 2.6)
    floor = plan_cut([_room_scan()], {"room": _placement()}, [], 0.95, 1.05)
    assert len(ring.polylines) == 2, len(ring.polylines)
    assert len(floor.polylines) == 1 and floor.points_in_cut > 50000, (len(floor.polylines), floor.points_in_cut)


def test_the_dxf_reads_back_with_the_layers_and_shapes_it_was_given():
    cut = plan_cut([_room_scan()], {"room": _placement()}, [], 2.0, 2.6)
    cut.footprints = [_expected_outer_corners()]
    text = dxf_text(cut, note="test")
    assert text.rstrip().endswith("EOF") and "AC1009" in text
    ents = _read_dxf(text)
    plan = [e for e in ents if e["type"] == "POLYLINE" and e["layer"] == "PLAN_CUT"]
    assert len(plan) == 2
    verts = [e for e in ents if e["type"] == "VERTEX" and e["layer"] == "PLAN_CUT"]
    assert verts and all(np.isfinite(v["xy"]).all() for v in verts if v["xy"])
    assert any(e["type"] == "POLYLINE" and e["layer"] == "SCAN_FOOTPRINTS" for e in ents)
    assert any(e["type"] == "TEXT" and e["layer"] == "SCAN_NAMES" for e in ents)
    assert "Plan cut 2.00 to 2.60 m, units metres" in text


def test_write_plan_makes_the_dxf_and_a_preview():
    cut = plan_cut([_room_scan()], {"room": _placement()}, [], 2.0, 2.6)
    with tempfile.TemporaryDirectory() as folder:
        info = write_plan(cut, folder)
        assert os.path.getsize(info["dxf_path"]) > 500 and os.path.getsize(info["png_path"]) > 500
        assert info["outlines"] == 2 and info["scans_used"] == 1


def test_default_band_sits_above_the_typical_floor():
    lo, hi = default_band({"a": Placement(name="a", tz=1.0), "b": Placement(name="b", tz=1.2), "c": Placement(name="c", tz=1.1)})
    assert abs(lo - 2.1) < 1e-9 and abs(hi - 2.7) < 1e-9


def test_nothing_in_the_band_gives_an_empty_drawing_not_a_crash():
    cut = plan_cut([_room_scan()], {"room": _placement()}, [], 50.0, 51.0)
    assert cut.polylines == [] and cut.points_in_cut == 0
    ents = _read_dxf(dxf_text(cut))
    assert any(e["type"] == "TEXT" for e in ents)


def test_specks_are_dropped_and_a_huge_area_coarsens_the_cell_instead_of_running_out_of_memory():
    rng = np.random.default_rng(1)
    xy = np.vstack([rng.normal(0, 0.02, (50, 2)), [[900.0, 900.0]] * 30 + rng.normal(0, 0.02, (30, 2))]).astype(np.float32)
    outlines, _, _, cell = trace_outlines(xy)
    assert cell > 0.02 and len(outlines) == 0


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
