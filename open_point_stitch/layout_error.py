"""How far one saved layout is from another, per scan.

A layout turned, shifted or lifted as one piece is not a worse layout, so the best whole-layout move is removed
first."""
from __future__ import annotations

import json
import math

import numpy as np

from .lib.poses import fit_yaw_shift_2d, rot2, yaw_gap_deg, yaw_matrix2


def load_scan_poses(path: str) -> dict[str, dict]:
    """Every scan's entry from a placement file (or backup), keyed by name."""
    with open(path) as fh:
        data = json.load(fh)
    return {s["name"]: s for s in data.get("scans", [])}


def _pose(s: dict) -> tuple[float, float, float, float]:
    return (float(s.get("x_m", 0.0)), float(s.get("y_m", 0.0)),
            float(s.get("z_m", 0.0)), float(s.get("yaw_deg", 0.0)))


def whole_layout_fit(layout: dict[str, dict], target: dict[str, dict], names: list[str],
                     iterations: int = 30, floor_m: float = 0.02) -> dict:
    """Best turn (about the vertical), sideways shift and lift moving layout's scan origins onto target's. Robust:
    scans that disagree most get less say each round, so one badly placed scan does not drag the fit."""
    src = np.array([_pose(layout[n]) for n in names])
    dst = np.array([_pose(target[n]) for n in names])
    w = np.ones(len(names))
    for _ in range(iterations):
        th, shift = fit_yaw_shift_2d(src, dst, w)
        w = 1.0 / np.maximum(floor_m, np.linalg.norm(src[:, :2] @ rot2(th).T + shift - dst[:, :2], axis=1))
    lifts = sorted(q[2] - p[2] for p, q in zip(src, dst))
    mid = len(lifts) // 2
    lift = lifts[mid] if len(lifts) % 2 else (lifts[mid - 1] + lifts[mid]) / 2
    return {"yaw_deg": math.degrees(th), "shift": shift, "lift_m": lift}


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
            x, y = yaw_matrix2(fit["yaw_deg"]) @ [x, y] + fit["shift"]
            z += fit["lift_m"]
            yaw += fit["yaw_deg"]
        rows.append({"scan": n, "xy_m": math.hypot(x - gx, y - gy), "z_m": abs(z - gz),
                     "yaw_deg": yaw_gap_deg(yaw, gyaw)})
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
