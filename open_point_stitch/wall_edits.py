"""Hand corrections to walls from the review view: delete a false wall, move one in or out, set its end points,
add one the finder missed, say what it contains. Stored per scan as wall_edits in placement.json, in the scan's
plan frame, and applied on top of whatever the finder found, so a re-detected scan keeps them. applyWallEdits in
placement_bench.html applies the same rules."""
from __future__ import annotations

import math

MATCH_M = 0.15      # a stored wall and a detected one this close (across the wall) ...
MATCH_DEG = 5.0     # ... and this close in direction, overlapping along it, are the same wall


def _line(w: dict):
    dx, dy = w["x2"] - w["x1"], w["y2"] - w["y1"]
    length = math.hypot(dx, dy) or 1e-9
    return (w["x1"], w["y1"]), (dx / length, dy / length), length


def find_wall(walls: list[dict], at: list[float]) -> int | None:
    """Index of the wall in `walls` that is the wall stored as `at` = [x1, y1, x2, y2], or None."""
    ref = {"x1": at[0], "y1": at[1], "x2": at[2], "y2": at[3]}
    (rx, ry), (ux, uy), rlen = _line(ref)
    mid = (rx + ux * rlen / 2, ry + uy * rlen / 2)
    best, best_off = None, MATCH_M
    for k, w in enumerate(walls):
        (wx, wy), (vx, vy), wlen = _line(w)
        if math.degrees(math.acos(min(1.0, abs(ux * vx + uy * vy)))) > MATCH_DEG:
            continue
        off = abs(-vy * (mid[0] - wx) + vx * (mid[1] - wy))
        t = sorted([(rx - wx) * vx + (ry - wy) * vy, (rx + ux * rlen - wx) * vx + (ry + uy * rlen - wy) * vy])
        if min(t[1], wlen) - max(t[0], 0.0) <= 0.1 or off > best_off:
            continue
        best, best_off = k, off
    return best


def apply_wall_edits(walls: list[dict], edits: list[dict]) -> tuple[list[dict], int]:
    """(the walls with the edits applied, how many delete/move edits found no wall to apply to). Edits apply in order."""
    out = [dict(w) for w in walls]
    missed = 0
    for e in edits or []:
        if e.get("op") == "set":                        # the walls as they stood when the wall finder changed. #wall-edits
            out = [{"i": k, "x1": w[0], "y1": w[1], "x2": w[2], "y2": w[3], "len": round(math.hypot(w[2] - w[0], w[3] - w[1]), 2),
                    "h": w[5] if len(w) > 5 else 2.5, "pts": 0, "behind_m": w[6] if len(w) > 6 else 0.0,
                    **({"source": w[4]} if len(w) > 4 and w[4] else {}),       # a wall as detected has no source
                    **({"contains": list(w[7])} if len(w) > 7 and w[7] else {})}
                   for k, w in enumerate(e["walls"])]
            continue
        if e.get("op") == "add":
            x1, y1, x2, y2 = e["wall"]
            out.append({"i": -1, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "len": round(math.hypot(x2 - x1, y2 - y1), 2),
                        "h": e.get("h", 2.5), "pts": 0, "behind_m": 0.0, "source": "hand"})
            continue
        k = find_wall(out, e["at"])
        if k is None:
            missed += 1
            continue
        if e["op"] == "delete":
            out.pop(k)
        elif e["op"] == "move":
            w = out[k]
            (_, _), (ux, uy), _ = _line(w)
            nx, ny = -uy * e["by"], ux * e["by"]       # along the wall's left-hand normal
            out[k] = {**w, "x1": w["x1"] + nx, "y1": w["y1"] + ny, "x2": w["x2"] + nx, "y2": w["y2"] + ny,
                      "source": "moved"}
        elif e["op"] == "ends":                         # both end points set by hand
            x1, y1, x2, y2 = e["wall"]
            out[k] = {**out[k], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "len": round(math.hypot(x2 - x1, y2 - y1), 2), "source": "moved"}
        elif e["op"] == "tag":                          # what the wall has in it (door, window, ...), the whole list
            w = {k2: v for k2, v in out[k].items() if k2 != "contains"}
            out[k] = {**w, "contains": list(e["contains"])} if e["contains"] else w
    return out, missed


def frozen(walls: list[dict], edits: list[dict], why: str) -> list[dict]:
    """edits replaced by one "set" edit holding the walls they produce on walls, so the corrections survive a change
    to the wall finder."""
    final, _ = apply_wall_edits(walls, edits)
    return [{"op": "set", "why": why,
             "walls": [[round(float(w["x1"]), 4), round(float(w["y1"]), 4), round(float(w["x2"]), 4),
                        round(float(w["y2"]), 4), w.get("source", ""), w.get("h", 2.5), w.get("behind_m", 0.0)]
                       + ([list(w["contains"])] if w.get("contains") else [])
                       for w in final]}]


def edits_by_scan(placement_data: dict) -> dict[str, list[dict]]:
    """{scan name: its wall edits} from a placement file's contents."""
    return {s["name"]: s["wall_edits"] for s in placement_data.get("scans", []) if s.get("wall_edits")}


def with_wall_edits(payloads: dict[str, dict], edits: dict[str, list[dict]]) -> dict[str, dict]:
    """The preview payloads with each edited scan's walls replaced by the corrected ones (copies; others unchanged).
    The wall reports measure these corrected walls too (planes)."""
    out = dict(payloads)
    for name, ed in edits.items():
        if name in out:
            fixed = apply_wall_edits(out[name].get("walls", []), ed)[0]
            out[name] = {**out[name], "walls": fixed, "planes": fixed}
    return out
