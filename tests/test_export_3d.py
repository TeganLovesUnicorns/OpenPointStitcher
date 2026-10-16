"""Floors, ceilings and walls as 3D planes: two scans' pieces join where they overlap, walls stop where the points
stop and break at a doorway, and the DXF holds closed 3D outlines on the planes."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.export_3d import dxf_text, scan_surfaces, venue_surfaces  # noqa: E402
from open_point_stitch.find_wall_runs import drop_crossing, wall_runs  # noqa: E402
from open_point_stitch.placement import Placement  # noqa: E402


def _square(x0, x1, y0, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]


def _piece(kind, h, outlines, slope=0.0):
    n = np.array([np.sin(np.radians(slope)), 0.0, np.cos(np.radians(slope))])
    return {"kind": kind, "normal": n.tolist(), "offset": float(h * n[2]), "outlines": outlines,
            "slope_deg": slope, "rms_m": 0.01, "label": kind}


def test_two_scans_of_one_room_give_one_floor_and_one_ceiling_on_the_right_planes():
    # Scan A sees x 0-6 of a room, B sees x 4-10; B is placed 4 m along with its floor 2 cm high.
    a = [_piece("floor", 0.0, _square(0, 6, 0, 5)), _piece("ceiling", 3.0, _square(0, 6, 0, 5))]
    b = [_piece("floor", 0.0, _square(0, 6, 0, 5)), _piece("ceiling", 3.0, _square(0, 6, 0, 5)),
         _piece("ceiling", 3.2, _square(2, 6, 0, 5), slope=20.0)]
    pa, pb = Placement(name="A"), Placement(name="B", tx=4.0, tz=0.02)
    joined = venue_surfaces({"A": scan_surfaces("A", a, [], pa), "B": scan_surfaces("B", b, [], pb)})
    floors = [s for s in joined if s.kind == "floor"]
    ceils = [s for s in joined if s.kind == "ceiling" and not s.sloped]
    slopes = [s for s in joined if s.kind == "ceiling" and s.sloped]
    assert len(floors) == 1 and len(ceils) == 1 and len(slopes) == 1, [s.label for s in joined]
    f = floors[0]
    xs = np.concatenate([o[:, 0] for o in f.outlines])
    assert xs.min() < 0.1 and xs.max() > 9.9, "the joined floor does not cover both scans"
    assert abs(f.spread_m - 0.02) < 0.005 and f.scans == ["A", "B"], (f.spread_m, f.scans)
    assert all(abs(o[:, 2] - 0.01).max() < 0.012 for o in f.outlines), "the floor outline is not on its plane"
    assert abs(np.ptp(slopes[0].outlines[0][:, 2]) - 4 * np.tan(np.radians(20))) < 0.1, "the slope lost its tilt"


def test_a_wall_stops_where_its_points_stop_and_breaks_at_a_doorway():
    rng = np.random.default_rng(5)
    # A wall along y = 0 from x 0 to 6 with a 1 m doorway at x 2-3; the detected wall claims x -1 to 9.
    xs = np.concatenate([rng.uniform(0, 2, 20000), rng.uniform(3, 6, 30000)])
    pts = np.column_stack([xs, rng.normal(0, 0.005, len(xs)), rng.uniform(0, 2.5, len(xs))])
    runs = wall_runs(pts, [{"x1": -1.0, "y1": 0.0, "x2": 9.0, "y2": 0.0, "h": 2.5}])
    ends = sorted((round(r["x1"], 1), round(r["x2"], 1)) for r in runs)
    assert len(runs) == 2, ends
    assert abs(ends[0][0] - 0.0) <= 0.1 and abs(ends[0][1] - 2.0) <= 0.1, ends
    assert abs(ends[1][0] - 3.0) <= 0.1 and abs(ends[1][1] - 6.0) <= 0.1, ends


def test_a_wall_running_past_a_corner_is_cut_back_to_it_and_one_crossing_mid_wall_is_dropped():
    # A wall poking 0.6 m through the wall it meets is cut back, not dropped.
    long = {"x1": 0.0, "y1": 0.0, "x2": 8.0, "y2": 0.0, "len": 8.0}
    over = {"x1": 3.0, "y1": -0.6, "x2": 3.0, "y2": 2.4, "len": 3.0,
            "_past": (0.6, 0.0)}                                         # stretched 0.6 m past y = 0
    mid = {"x1": 6.0, "y1": -1.5, "x2": 6.0, "y2": 1.5, "len": 3.0}        # crosses half way along: not a wall
    out = drop_crossing([long, over, mid])
    assert len(out) == 2 and out[0] == long, out
    assert abs(out[1]["y1"]) < 1e-6 and out[1]["y2"] == 2.4 and out[1]["len"] == 2.4, out[1]


def test_a_wall_running_past_two_walls_at_each_end_is_cut_to_the_inner_ones_and_kept():
    # Four cuts used to run out the pass count and drop it; the cut now goes to the crossing furthest in.
    across = lambda x: {"x1": x, "y1": -5.0, "x2": x, "y2": 5.0, "len": 10.0}
    w = {"x1": -0.9, "y1": 0.0, "x2": 5.9, "y2": 0.0, "len": 6.8, "_past": (0.9, 0.9)}
    out = drop_crossing([across(-0.6), across(-0.3), across(5.6), across(5.3), w])
    assert len(out) == 5, out
    assert (out[4]["x1"], out[4]["x2"], out[4]["len"]) == (-0.3, 5.3, 5.6), out[4]


def test_a_wall_behind_a_curtain_is_never_cut_short_by_a_crossing():
    # It has no stretch past its ends to cut (no `_past`), so a crossing near its end drops it, as before 0.29.6.
    rng = np.random.default_rng(7)
    pts = np.column_stack([rng.uniform(0, 8, 40000), rng.normal(0, 0.005, 40000), rng.uniform(0, 2.5, 40000)])
    behind = {"x1": 3.0, "y1": -0.5, "x2": 3.0, "y2": 3.0, "h": 2.5, "behind_m": 0.3}
    runs = wall_runs(pts, [{"x1": 0.0, "y1": 0.0, "x2": 8.0, "y2": 0.0, "h": 2.5}, behind])
    assert not [r for r in runs if r["kind"] == "behind curtain" and r["y1"] != -0.5], runs
    assert [r for r in runs if r["kind"] == "full"], runs


def test_a_half_wall_and_a_beam_from_the_ceiling_keep_their_own_heights_and_a_curtain_wall_is_kept_whole():
    # A 1 m half wall and a beam hanging at 2.1-2.5 m are walls; a wall behind a curtain has no points of its own and
    # must not be trimmed away.
    rng = np.random.default_rng(6)
    sheet = lambda n, x0, x1, y0, y1, z0, z1: np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n),
                                                               rng.uniform(z0, z1, n)])
    pts = np.vstack([sheet(80000, 0, 5, 0, 5, 0, 0.005), sheet(80000, 0, 5, 0, 5, 2.495, 2.5),
                     sheet(30000, 0.5, 4.5, 0, 0.005, 0, 2.5),             # full wall, y = 0
                     sheet(12000, 0.5, 3.5, 2, 2.005, 0, 1.0),             # half wall, y = 2
                     sheet(8000, 0.5, 4.5, 4, 4.005, 2.1, 2.5)])           # beam, y = 4
    line = lambda y, **k: {"x1": 0.0, "y1": y, "x2": 5.0, "y2": y, "h": 2.5, **k}
    runs = wall_runs(pts, [line(0.0), line(2.0), line(4.0), line(3.0, behind_m=0.3)])
    kinds = {round(r["y1"], 1): (r["kind"], r["z_lo"], r["z_hi"]) for r in runs}
    assert kinds[0.0][0] == "full" and kinds[2.0][0] == "half wall" and kinds[4.0][0] == "hanging", kinds
    assert kinds[2.0][2] < 1.1 and kinds[4.0][1] > 2.0, kinds
    assert kinds[3.0][0] == "behind curtain", kinds
    pieces = [_piece("floor", 0.0, _square(0, 5, 0, 5)), _piece("ceiling", 2.5, _square(0, 5, 0, 5))]
    walls = [s for s in venue_surfaces({"A": scan_surfaces("A", pieces, runs, Placement(name="A"))}) if s.kind == "wall"]
    by_y = {round(float(s.outlines[0][0][1]), 1): s for s in walls}
    assert abs(by_y[2.0].outlines[0][:, 2].max() - kinds[2.0][2]) < 1e-6 and "half wall" in by_y[2.0].label
    assert abs(by_y[4.0].outlines[0][:, 2].min() - kinds[4.0][1]) < 1e-6 and "hanging" in by_y[4.0].label
    assert by_y[0.0].outlines[0][:, 2].min() == 0.0 and by_y[0.0].outlines[0][:, 2].max() == 2.5


def test_walls_seen_by_two_scans_join_and_run_floor_to_ceiling():
    pieces = [_piece("floor", 0.0, _square(0, 6, 0, 5)), _piece("ceiling", 3.0, _square(0, 6, 0, 5))]
    wa = [{"x1": 0.0, "y1": 5.0, "x2": 4.0, "y2": 5.0, "h": 2.0}]
    wb = [{"x1": 6.0, "y1": 5.0, "x2": 3.0, "y2": 5.0, "h": 2.0}]       # drawn the other way round
    joined = venue_surfaces({"A": scan_surfaces("A", pieces, wa, Placement(name="A")),
                             "B": scan_surfaces("B", pieces, wb, Placement(name="B"))})
    walls = [s for s in joined if s.kind == "wall"]
    assert len(walls) == 1, [w.label for w in walls]
    q = walls[0].outlines[0]
    assert abs(q[:, 0].min()) < 1e-6 and abs(q[:, 0].max() - 6.0) < 1e-6, q
    assert abs(q[:, 2].min()) < 1e-6 and abs(q[:, 2].max() - 3.0) < 1e-6, "the wall does not run floor to ceiling"


def test_the_dxf_has_closed_3d_outlines_on_layers_by_kind():
    pieces = [_piece("floor", 0.0, _square(0, 6, 0, 5)), _piece("ceiling", 3.0, _square(0, 6, 0, 5)),
              _piece("ceiling", 3.2, _square(2, 6, 0, 5), slope=20.0)]
    walls = [{"x1": 0.0, "y1": 5.0, "x2": 6.0, "y2": 5.0, "h": 2.0}]
    body = dxf_text(venue_surfaces({"A": scan_surfaces("A", pieces, walls, Placement(name="A"))})).splitlines()
    assert body[-1] == "EOF" and "AC1009" in body
    layers = [body[k + 1] for k in range(len(body) - 1) if body[k].strip() == "8"]
    for name in ("FLOOR", "CEILING", "CEILING_SLOPED", "WALLS", "LABELS"):
        assert name in layers, (name, set(layers))
    flags = [body[k + 1] for k in range(len(body) - 1) if body[k].strip() == "70"]
    assert flags.count("9") == 4 and flags.count("32") >= 16, flags


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
