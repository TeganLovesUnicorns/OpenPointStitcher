"""Where along each detected wall line a scan really has wall, and how tall it is there, found from every point.
A drawn wall stops where the wall stops, breaks at a doorway and keeps its real bottom and top, so a half wall or a
bulkhead face is drawn as what it is. Used by the 3D export and the bench's walls."""
from __future__ import annotations

import numpy as np

NEAR_M = 0.04          # a point this close to the wall line, seen from above, is on the wall
BIN_M = 0.10           # the wall is looked at in 10 cm steps along its length
LEVEL_M = 0.05         # ... and in 5 cm levels up it
MIN_LEVEL_PTS = 2      # a level counts in a step when it holds this many points
MIN_TALL_M = 0.4       # a step is wall when its points stand at least this tall, wherever that is; a floor or
                       # ceiling edge alone is one or two levels, so it never counts
REACH_M = 1.0          # a run may reach this far past the detected wall's ends
CLOSE_M = 0.30         # gaps this short (a pipe, a dark patch) do not break a wall
MIN_RUN_M = 0.5        # shorter runs are left out
STACK_SHARE = 0.2      # a level is part of the run's wall when this share of its steps have it
STACK_GAP = 2          # levels missing this many in a row still join one wall body (a shelf, a dark band)
AROUND_M = 0.6         # floor and ceiling beside a run are read from points this close to it
KIND_M = 0.4           # this close to the floor or ceiling counts as reaching it
HANG_ABOVE_M = 1.5     # a wall starting this far above the floor, and reaching the ceiling, hangs from it
HALF_BELOW_M = 1.6     # a wall standing on the floor and stopping below this is a half wall
CROSS_CLEAR_M = 0.2    # two runs crossing further than this inside both are not two walls


def wall_runs(points_levelled: np.ndarray, walls: list[dict], min_level_pts: int = MIN_LEVEL_PTS,
              keep_past: bool = False) -> list[dict]:
    """The walls (payload rows: x1, y1, x2, y2, h, ...) cut to their solid runs, each with z_lo and z_hi (where the
    wall body starts and stops, scan frame) and kind: "full", "half wall", "hanging" or "behind curtain" (kept
    whole, since its own points are hidden). min_level_pts is 1 on the 3 cm preview grid, where one point per level
    is already a real surface. keep_past leaves _past on each row for a caller that runs drop_crossing again."""
    xy, z = points_levelled[:, :2], points_levelled[:, 2]
    out = []
    for w in walls:
        if float(w.get("behind_m", 0.0)) > 0:
            out.append({**w, "kind": "behind curtain", "z_lo": None, "z_hi": None})
            continue
        a, b = np.array([w["x1"], w["y1"]], float), np.array([w["x2"], w["y2"]], float)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        d = (b - a) / length
        nrm = np.array([-d[1], d[0]])
        rel = xy - a
        t, off = rel @ d, rel @ nrm
        span = (t > -REACH_M) & (t < length + REACH_M)
        on = span & (np.abs(off) < NEAR_M)
        if on.sum() < 10:
            continue
        tb = ((t[on] + REACH_M) / BIN_M).astype(np.int64)
        zmin = float(z[on].min())
        zl = ((z[on] - zmin) / LEVEL_M).astype(np.int64)
        nl = int(zl.max()) + 1
        nb = int((length + 2 * REACH_M) / BIN_M) + 1
        uniq, counts = np.unique(tb * nl + zl, return_counts=True)
        filled = uniq[counts >= min_level_pts]
        occ = np.zeros((nb, nl), bool)
        occ[filled // nl, filled % nl] = True
        solid = occ.sum(axis=1) * LEVEL_M >= MIN_TALL_M
        for start, stop in _runs(_close(solid, int(round(CLOSE_M / BIN_M)))):
            if (stop - start) * BIN_M < MIN_RUN_M:
                continue
            lo_l, hi_l = _body(occ[start:stop].mean(axis=0) >= STACK_SHARE)
            z_lo, z_hi = zmin + lo_l * LEVEL_M, zmin + (hi_l + 1) * LEVEL_M
            p, q = a + d * (start * BIN_M - REACH_M), a + d * (stop * BIN_M - REACH_M)
            near = span & (np.abs(off) < AROUND_M) & (t >= start * BIN_M - REACH_M) & (t < stop * BIN_M - REACH_M)
            floor_here, ceil_here = (np.percentile(z[near], [2, 98]) if near.sum() > 20 else (z_lo, z_hi))
            # Hanging: starts above head height and reaches the ceiling. Half wall: stands on the floor and stops low.
            # A wall whose foot is hidden behind furniture is still a full wall.
            kind = ("hanging" if z_lo - floor_here > HANG_ABOVE_M and ceil_here - z_hi < KIND_M else
                    "half wall" if z_lo - floor_here < KIND_M and z_hi - floor_here < HALF_BELOW_M else "full")
            out.append({**w, "x1": round(float(p[0]), 3), "y1": round(float(p[1]), 3),
                        "x2": round(float(q[0]), 3), "y2": round(float(q[1]), 3),
                        "len": round(float((stop - start) * BIN_M), 2), "kind": kind,
                        "z_lo": round(float(z_lo), 3), "z_hi": round(float(z_hi), 3),
                        "_past": (max(0.0, REACH_M - start * BIN_M), max(0.0, stop * BIN_M - REACH_M - length))})
    return drop_crossing(out) if keep_past else without_past(drop_crossing(out))


def without_past(runs: list[dict]) -> list[dict]:
    """The rows with drop_crossing's working field taken off."""
    return [{k: v for k, v in r.items() if k != "_past"} for r in runs]


def _body(levels: np.ndarray) -> tuple[int, int]:
    """(first, last) level of the tallest stack of levels, gaps of up to STACK_GAP allowed: the wall itself, apart
    from a ceiling or floor just above or below it."""
    best, best_len, start, last, gap = (0, 0), -1, None, None, 0
    for k, on in enumerate(levels):
        if on:
            if start is None:
                start = k
            last, gap = k, 0
        elif start is not None:
            gap += 1
            if gap > STACK_GAP:
                if last - start > best_len:
                    best, best_len = (start, last), last - start
                start, gap = None, 0
    if start is not None and last - start > best_len:
        best = (start, last)
    return best


def _inner_cross(a: dict, b: dict) -> float | None:
    """Where along a (0 to 1) it crosses b, when the crossing is further than CROSS_CLEAR_M inside both; else None."""
    p, r = np.array([a["x1"], a["y1"]]), np.array([a["x2"] - a["x1"], a["y2"] - a["y1"]])
    q, s = np.array([b["x1"], b["y1"]]), np.array([b["x2"] - b["x1"], b["y2"] - b["y1"]])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
    la, lb = np.linalg.norm(r), np.linalg.norm(s)
    inside = CROSS_CLEAR_M / la < t < 1 - CROSS_CLEAR_M / la and CROSS_CLEAR_M / lb < u < 1 - CROSS_CLEAR_M / lb
    return float(t) if inside else None


def _shallow(w: dict, t: float) -> bool:
    """Whether t (0 to 1 along w) lies in the stretch w was given past its found end, plus one step. A row without
    `_past` (a wall behind a curtain, a plain row) was given none, so it is never cut."""
    length = float(np.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]))
    return min(t, 1 - t) * length <= w.get("_past", (0.0, 0.0))[0 if t <= 0.5 else 1] + BIN_M


def _cut(w: dict, t: float) -> dict:
    """w cut back to t (0 to 1 along it) at its nearer end."""
    length = float(np.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"]))
    cx, cy = w["x1"] + t * (w["x2"] - w["x1"]), w["y1"] + t * (w["y2"] - w["y1"])
    end = ("x1", "y1") if t <= 0.5 else ("x2", "y2")
    out = {**w, end[0]: round(float(cx), 3), end[1]: round(float(cy), 3),
           "len": round((1 - t if t <= 0.5 else t) * length, 2)}
    if "_past" in w:                                     # what is cut no longer counts as stretch
        lo, hi = w["_past"]
        out["_past"] = (max(0.0, lo - t * length), hi) if t <= 0.5 else (lo, max(0.0, hi - (1 - t) * length))
    return out


def drop_crossing(runs: list[dict]) -> list[dict]:
    """Walls meet at their ends (an L or a T) and never pass through each other. A run poking through a longer one
    within the stretch it was given past its found end ran past a corner and is cut back; where two runs cross
    further inside both, the shorter is something else (an open door, a shelf end) and is dropped. Rows carry _past
    from wall_runs."""
    order = sorted(range(len(runs)), key=lambda i: -float(runs[i].get("len", 0)))
    kept: dict[int, dict] = {}
    for i in order:
        w = runs[i]
        while True:                                      # every pass removes a crossing, so this ends
            hits = [(j, t) for j, k in kept.items() if (t := _inner_cross(w, k)) is not None]
            if not hits:
                kept[i] = w
                break
            if not all(_shallow(w, t) for _, t in hits):   # crossed well inside: not a wall. #corner-overshoot
                break
            w = _cut(w, max((t for _, t in hits), key=lambda t: min(t, 1 - t)))   # the corner furthest in
            if w["len"] < MIN_RUN_M:
                break
    return [kept[i] for i in sorted(kept)]


def _close(solid: np.ndarray, gap: int) -> np.ndarray:
    """Fill runs of False no longer than `gap` that sit between two True steps."""
    out = solid.copy()
    for start, stop in _runs(~solid):
        if stop - start <= gap and start > 0 and stop < len(solid):
            out[start:stop] = True
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each run of True, stop exclusive."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))
