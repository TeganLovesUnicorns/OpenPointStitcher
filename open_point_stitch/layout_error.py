"""How far one saved layout is from another, per scan.

A layout turned, shifted or lifted as one piece is not a worse layout, so the best whole-layout move is removed
first."""
from __future__ import annotations

import json
import math


def load_scan_poses(path: str) -> dict[str, dict]:
    """Every scan's entry from a placement file (or backup), keyed by name."""
    with open(path) as fh:
        data = json.load(fh)
    return {s["name"]: s for s in data.get("scans", [])}


def _pose(s: dict) -> tuple[float, float, float, float]:
    return (float(s.get("x_m", 0.0)), float(s.get("y_m", 0.0)),
            float(s.get("z_m", 0.0)), float(s.get("yaw_deg", 0.0)))


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _weighted_fit(src, dst, w):
    k = sum(w)
    sx, sy = sum(wi * p[0] for wi, p in zip(w, src)) / k, sum(wi * p[1] for wi, p in zip(w, src)) / k
    tx, ty = sum(wi * p[0] for wi, p in zip(w, dst)) / k, sum(wi * p[1] for wi, p in zip(w, dst)) / k
    a = b = 0.0
    for wi, (x1, y1, _, _), (x2, y2, _, _) in zip(w, src, dst):
        x1, y1, x2, y2 = x1 - sx, y1 - sy, x2 - tx, y2 - ty
        a += wi * (x1 * x2 + y1 * y2)
        b += wi * (x1 * y2 - y1 * x2)
    return math.atan2(b, a), (sx, sy), (tx, ty)


def whole_layout_fit(layout: dict[str, dict], target: dict[str, dict], names: list[str],
                     iterations: int = 30, floor_m: float = 0.02) -> dict:
    """Best turn (about the vertical), sideways shift and lift moving layout's scan origins onto target's. Robust:
    scans that disagree most get less say each round, so one badly placed scan does not drag the fit."""
    src = [_pose(layout[n]) for n in names]
    dst = [_pose(target[n]) for n in names]
    w = [1.0] * len(names)
    for _ in range(iterations):
        th, sc, dc = _weighted_fit(src, dst, w)
        c, s = math.cos(th), math.sin(th)
        w = [1.0 / max(floor_m, math.hypot(c * (p[0] - sc[0]) - s * (p[1] - sc[1]) + dc[0] - q[0],
                                          s * (p[0] - sc[0]) + c * (p[1] - sc[1]) + dc[1] - q[1]))
             for p, q in zip(src, dst)]
    lifts = sorted(q[2] - p[2] for p, q in zip(src, dst))
    mid = len(lifts) // 2
    lift = lifts[mid] if len(lifts) % 2 else (lifts[mid - 1] + lifts[mid]) / 2
    return {"yaw_deg": math.degrees(th), "src_centre": sc, "dst_centre": dc, "lift_m": lift}


def per_scan_error(layout: dict[str, dict], target: dict[str, dict],
                   remove_layout_move: bool = True) -> list[dict]:
    """One row per scan in both layouts: sideways (m), height (m) and turn (deg) error."""
    names = sorted(n for n in target if n in layout)
    if not names:
        return []
    fit = whole_layout_fit(layout, target, names) if remove_layout_move and len(names) >= 2 else None
    rows = []
    for n in names:
        x, y, z, yaw = _pose(layout[n])
        gx, gy, gz, gyaw = _pose(target[n])
        if fit is not None:
            th = math.radians(fit["yaw_deg"])
            dx, dy = x - fit["src_centre"][0], y - fit["src_centre"][1]
            x = math.cos(th) * dx - math.sin(th) * dy + fit["dst_centre"][0]
            y = math.sin(th) * dx + math.cos(th) * dy + fit["dst_centre"][1]
            z += fit["lift_m"]
            yaw += fit["yaw_deg"]
        rows.append({"scan": n, "xy_m": math.hypot(x - gx, y - gy), "z_m": abs(z - gz),
                     "yaw_deg": abs(_wrap_deg(yaw - gyaw))})
    return rows


def summarise(rows: list[dict]) -> dict:
    """Mean and worst of each error, plus which scan is worst sideways."""
    if not rows:
        return {"n_scans": 0}
    worst = max(rows, key=lambda r: r["xy_m"])
    k = len(rows)
    return {"n_scans": k,
            "mean_xy_m": sum(r["xy_m"] for r in rows) / k, "max_xy_m": worst["xy_m"],
            "worst_xy_scan": worst["scan"],
            "mean_z_m": sum(r["z_m"] for r in rows) / k, "max_z_m": max(r["z_m"] for r in rows),
            "max_yaw_deg": max(r["yaw_deg"] for r in rows)}
