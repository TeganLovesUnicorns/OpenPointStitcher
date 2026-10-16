"""Bent scans: a scan whose tracking drifted during capture, so one wall reads well off while its other walls agree
with the scans around it. Reports in plain words; moves nothing and changes no result. Also reports a scan that is
off on most of its walls together, which is a placement problem, not a bend."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .measure_walls import _segments
from .placement import Placement

SAME_M, SAME_DEG = 0.25, 4.0      # detections this close in place and angle, seen from the same side, are one wall
OFF_M = 0.10                      # a wall this far from where the other scans put it is "off"
AGREE_M = 0.03                    # ... and this close is "agrees"
SAME_HEIGHT_M = 0.05              # an off wall whose height bands differ by more than this ...
SAME_HEIGHT_FRAC = 0.5            # ... and by more than this share of its offset is a curtain or recess, not a bend
PARALLEL_DEG = 20.0               # an agreeing wall this near parallel to the off wall rules out a sideways shift
BANDS = ((0.2, 0.9), (0.9, 1.6), (1.6, 2.2), (2.2, 9.0))   # height above the scan's own floor, metres
MIN_BAND_POINTS = 80
BAND_WINDOW_M = 0.25              # height bands look this far either side of the scan's own detected wall
FACE_NEAR_M, FACE_FAR_M = 0.10, 0.80   # a scan's points this far in front of its wall show which face it saw ...
FACE_RATIO, FACE_MIN_POINTS = 3.0, 100  # ... when one side has this many times the other, and at least this many
FACE_FLOOR_M = 0.12                     # ... counting only points this close to the scan's own floor
THICK_MIN_M, THICK_MAX_M = 0.05, 0.30   # two opposite faces this far apart, overlapping, are one wall's two sides
THICK_DEFAULT_M = (0.08, 0.14)          # used when the project has no measured wall


def _face_from_points(s: dict, pts: np.ndarray, floor_z: float) -> int:
    """+1 or -1: the side of the detection (along its own n) its scan's FLOOR points sit on; 0 when they do not say.
    Floor only: through a window a scan sees things behind a wall, but not the floor right behind it."""
    off = pts[:, :2] @ s["n"] - s["n"] @ s["a"]
    t = pts[:, :2] @ s["u"]
    lo, hi = sorted([s["a"] @ s["u"], s["b"] @ s["u"]])
    sel = (t > lo) & (t < hi) & (np.abs(pts[:, 2] - floor_z) < FACE_FLOOR_M)
    plus = int((sel & (off > FACE_NEAR_M) & (off < FACE_FAR_M)).sum())
    minus = int((sel & (off < -FACE_NEAR_M) & (off > -FACE_FAR_M)).sum())
    if max(plus, minus) >= FACE_MIN_POINTS and max(plus, minus) >= FACE_RATIO * max(min(plus, minus), 1):
        return 1 if plus > minus else -1
    return 0


def group_walls(segments: list[dict], placements: dict[str, Placement],
                world: dict[str, np.ndarray] | None = None) -> list[list[dict]]:
    """Real walls: each group grows around its longest detection only (no chaining through a third wall), and a wall
    seen from both faces is two groups. The face is read from the scan's points when world is given and they are
    clear (s["face_by"] = "points"), otherwise from where the scan started ("origin")."""
    for s in segments:
        p = placements[s["scan"]]
        s["side"] = 1 if s["n"] @ (np.array([p.tx, p.ty]) - s["a"]) >= 0 else -1
        s["face_by"] = "origin"
        if world is not None and s["scan"] in world:
            f = _face_from_points(s, world[s["scan"]], p.tz)
            if f:
                s["side"], s["face_by"] = f, "points"
    left = sorted(segments, key=lambda s: -s["length"])
    groups = []
    while left:
        seed = left.pop(0)
        mid = (seed["a"] + seed["b"]) / 2
        g, rest = [seed], []
        for s in left:
            same_face = s["side"] * (s["n"] @ seed["n"]) * seed["side"] > 0
            ang = np.degrees(np.arccos(np.clip(abs(s["u"] @ seed["u"]), 0, 1)))
            off = abs(seed["n"] @ ((s["a"] + s["b"]) / 2 - mid))
            t = sorted([(s["a"] - mid) @ seed["u"], (s["b"] - mid) @ seed["u"]])
            half = seed["length"] / 2
            overlap = min(t[1], half) - max(t[0], -half) > 0.3
            (g if same_face and ang <= SAME_DEG and off <= SAME_M and overlap else rest).append(s)
        groups.append(g)
        left = rest
    return groups


def wall_thicknesses(groups: list[list[dict]]) -> list[float]:
    """Metres between the two faces of each wall seen from both sides: two parallel groups that overlap, face away
    from each other and sit THICK_MIN_M to THICK_MAX_M apart."""
    seeds = [max(g, key=lambda s: s["length"]) for g in groups]
    out = []
    for i, a in enumerate(seeds):
        fa = a["n"] * a["side"]                                  # points from the wall towards whoever saw this face
        ma = (a["a"] + a["b"]) / 2
        for b in seeds[i + 1:]:
            fb = b["n"] * b["side"]
            if fa @ fb > -np.cos(np.radians(SAME_DEG)):
                continue
            mb = (b["a"] + b["b"]) / 2
            gap = float((ma - mb) @ fa)                          # positive when b sits behind a's face
            t = sorted([(b["a"] - ma) @ a["u"], (b["b"] - ma) @ a["u"]])
            half = a["length"] / 2
            if THICK_MIN_M <= gap <= THICK_MAX_M and min(t[1], half) - max(t[0], -half) > 0.3:
                out.append(gap)
    return out


def thickness_range(measured: list[float]) -> tuple[float, float]:
    """The band an offset must fall in to read as "maybe the other face": THICK_DEFAULT_M widened to cover the walls
    measured in this layout, 2 cm either side."""
    lo, hi = THICK_DEFAULT_M
    if measured:
        lo, hi = min(lo, min(measured) - 0.02), max(hi, max(measured) + 0.02)
    return max(THICK_MIN_M, lo), hi


@dataclass
class WallOffset:
    wall: int                     # index of the real wall in this run's grouping
    at: list                      # plan position of the wall's middle, metres
    angle_deg: float              # plan direction of the wall, 0 to 180
    off_cm: float                 # this scan against the middle scan (three or more saw it) or the other scan (two)
    bands_cm: list                # this scan's offset in each height band that had enough points
    other_scans: int
    compared: list = field(default_factory=list)    # the other scans' names
    faces_sure: bool = True       # every detection here had its face read from its points, not from its start


@dataclass
class ScanVerdict:
    scan: str
    kind: str                     # "bent", "bent or shifted", "placed off", "other face" or "" (nothing to say)
    text: str
    off_walls: list = field(default_factory=list)
    walls: list = field(default_factory=list)


def _band_offsets(world: np.ndarray, floor_z: float, n, c, u, span) -> list[float]:
    """Where this scan's points are densest across its own detected wall, per height band, metres from that line."""
    off = world[:, :2] @ n - c
    t = world[:, :2] @ u
    h = world[:, 2] - floor_z
    sel = (np.abs(off) < BAND_WINDOW_M) & (t > span[0]) & (t < span[1])
    out = []
    for lo, hi in BANDS:
        b = sel & (h > lo) & (h < hi)
        if b.sum() < MIN_BAND_POINTS:
            continue
        hist, edges = np.histogram(off[b], bins=np.arange(-BAND_WINDOW_M, BAND_WINDOW_M + 0.01, 0.01))
        k = int(np.argmax(hist))
        out.append(float(edges[k] + 0.005))
    return out


def _compass(angle_deg: float, at, centre) -> str:
    """Which side of the scan a wall is on, as a compass word (+x east, +y north)."""
    d = np.asarray(at, float) - np.asarray(centre, float)
    a = np.radians(angle_deg)
    n = np.array([-np.sin(a), np.cos(a)])
    side = n if n @ d >= 0 else -n
    return ["east", "north", "west", "south"][int(((np.degrees(np.arctan2(side[1], side[0])) + 45) % 360) // 90)]


def find_bent_scans(world: dict[str, np.ndarray], placements: dict[str, Placement], payloads: dict[str, dict],
                    label=lambda name: name, info: dict | None = None) -> list[ScanVerdict]:
    """One verdict per scan that shares at least one wall with another scan. `world` holds each scan's full-resolution
    points already placed in the layout. Read only. `info`, if given, receives the measured wall thicknesses."""
    groups = group_walls(_segments(payloads, placements), placements, world)
    measured = wall_thicknesses(groups)
    thick = thickness_range(measured)
    if info is not None:
        info["wall_thickness_cm"] = sorted(round(t * 100, 1) for t in measured)
        info["other_face_band_cm"] = [round(thick[0] * 100, 1), round(thick[1] * 100, 1)]
    per_scan: dict[str, list[WallOffset]] = {}
    for gi, g in enumerate(groups):
        names = sorted({s["scan"] for s in g} & set(world))
        if len(names) < 2:
            continue
        ref = max(g, key=lambda s: s["length"])
        n = np.array(ref["n"], float)
        u = np.array([-n[1], n[0]])
        own = {k: max((s for s in g if s["scan"] == k), key=lambda s: s["length"]) for k in names}
        level = {k: float(((s["a"] + s["b"]) / 2) @ n) for k, s in own.items()}   # where each scan detected the wall
        bands = {k: _band_offsets(world[k], placements[k].tz, n, level[k], u,
                                  sorted([s["a"] @ u, s["b"] @ u])) for k, s in own.items()}
        mid = np.mean([(s["a"] + s["b"]) / 2 for s in g], axis=0)
        angle = float(np.degrees(np.arctan2(u[1], u[0])) % 180)
        # Three or more scans: the middle one is the reference, so one bent scan cannot drag it. Two: the other scan.
        middle = float(np.median(list(level.values()))) if len(level) >= 3 else None
        sure = all(s["face_by"] == "points" for s in own.values())
        for k, v in level.items():
            others = [x for j, x in level.items() if j != k]
            per_scan.setdefault(k, []).append(WallOffset(
                wall=gi, at=[round(float(x), 2) for x in mid], angle_deg=round(angle, 1),
                off_cm=round((v - (others[0] if middle is None else middle)) * 100, 1),
                bands_cm=[round(b * 100, 1) for b in bands[k]], other_scans=len(others),
                compared=[j for j in names if j != k], faces_sure=sure))
    return [_verdict(name, walls, placements[name], label, thick) for name, walls in sorted(per_scan.items())]


def _verdict(name: str, walls: list[WallOffset], p: Placement, label,
             thick: tuple[float, float] = THICK_DEFAULT_M) -> ScanVerdict:
    """Bent: one wall off by the same amount at every height while at least two other walls agree, one of them near
    parallel (so sliding the scan cannot explain it). Bent or shifted: the same, but no agreeing wall runs the same
    way. Placed off: most walls off together. Other face: the only off walls are about one wall thick and their faces
    could not be read, so they may be two sides of one wall."""
    def even(w):
        spread = max(w.bands_cm) - min(w.bands_cm) if len(w.bands_cm) >= 2 else float("inf")
        return spread <= max(SAME_HEIGHT_M * 100, SAME_HEIGHT_FRAC * abs(w.off_cm))

    def turn(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)

    def maybe_other_face(w):
        return not w.faces_sure and thick[0] * 100 <= abs(w.off_cm) <= thick[1] * 100

    def against(ws):
        names = sorted({label(c) for w in ws for c in w.compared})
        return f" (compared with {', '.join(names)})" if names else ""

    off_all = [w for w in walls if abs(w.off_cm) > OFF_M * 100 and even(w)]
    two_faces = [w for w in off_all if maybe_other_face(w)]
    off = [w for w in off_all if not maybe_other_face(w)]
    agree = [w for w in walls if abs(w.off_cm) <= AGREE_M * 100]
    v = ScanVerdict(scan=name, kind="", text="", walls=walls)
    if not off and two_faces:
        w = max(two_faces, key=lambda w: abs(w.off_cm))
        v.kind, v.off_walls = "other face", two_faces
        v.text = (f"{label(name)} reads {abs(w.off_cm):.0f} cm off on its "
                  f"{_compass(w.angle_deg, w.at, (p.tx, p.ty))} wall{against([w])}. That is about one wall's "
                  f"thickness ({thick[0] * 100:.0f} to {thick[1] * 100:.0f} cm here), so it may be the other side "
                  f"of the same wall rather than a drift.")
        return v
    if not off:
        return v
    if len(off) >= 2 and len(off) >= len(agree):
        v.kind, v.off_walls = "placed off", off
        v.text = (f"{label(name)} reads off on {len(off)} of its {len(walls)} shared walls together "
                  f"({', '.join(f'{abs(w.off_cm):.0f} cm' for w in off)}). That looks like the scan is placed wrong, "
                  f"not bent.")
        return v
    if len(agree) < 2:
        return v
    bent = [w for w in off if any(turn(a.angle_deg, w.angle_deg) <= PARALLEL_DEG for a in agree)]
    worst = max(bent or off, key=lambda w: abs(w.off_cm))
    within = max(abs(a.off_cm) for a in agree)
    side = _compass(worst.angle_deg, worst.at, (p.tx, p.ty))
    v.kind, v.off_walls = ("bent", bent) if bent else ("bent or shifted", off)
    v.text = (f"{label(name)} reads {abs(worst.off_cm):.0f} cm off on its {side} wall{against([worst])} while its "
              f"other walls agree within {max(within, 1):.0f} cm. ")
    if bent:
        v.text += "It may have drifted during the scan."
    else:
        across = "north-south" if side in ("north", "south") else "east-west"
        v.text += (f"Those other walls run the other way, so this could also be the whole scan sitting "
                   f"{abs(worst.off_cm):.0f} cm out of place {across}: either it drifted during the scan, or it needs moving.")
    return v


def as_report(verdicts: list[ScanVerdict], info: dict | None = None) -> dict:
    """The run report's and wall report's section: only scans with something to say, plus how many were checked
    and, given `info` from find_bent_scans, the wall thicknesses measured in this layout."""
    flagged = [v for v in verdicts if v.kind]
    return {"checked": len(verdicts), **(info or {}),
            "flags": [{"scan": v.scan, "kind": v.kind, "text": v.text,
                       "walls": [vars(w) for w in v.off_walls]} for v in flagged]}
