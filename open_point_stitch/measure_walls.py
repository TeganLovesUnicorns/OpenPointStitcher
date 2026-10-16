"""Wall thickness in inches, per wall, on the placed scans. A wall several scans saw should be one sheet; where the
scans disagree it reads as a band, and this reports how thick, and which scan is off by how much."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .placement import Placement, tilt_matrix3
from .pts_file import Scan

INCH_M = 0.0254
MAX_ANGLE_DEG = 4.0        # two detections this far from parallel are different walls
MAX_OFFSET_M = 0.6         # ... or this far apart across the wall (scans are only roughly placed)
MAX_GAP_M = 1.0            # ... or this far apart along it
SLAB_M = 0.5               # points this near the wall's plane are looked at
KEEP_M = 0.3               # the wall's own points are within this of the typical distance
BAND_ABOVE_FLOOR_M = (0.4, 2.0)
END_MARGIN_M = 0.2
MIN_POINTS = 200           # a scan needs this many points on the wall before it is quoted


@dataclass
class ScanOnWall:
    scan: str
    offset_in: float           # how far this scan's face of the wall sits from the typical one (+ is toward +normal)
    spread_in: float           # scatter of this scan's own points about its face
    points: int
    side: int                  # which side of the wall the scan was standing on


@dataclass
class WallReport:
    x_m: float
    y_m: float
    length_m: float
    angle_deg: float
    thickness_in: float        # how wide a band the scans' points make (5th to 95th percentile), scans on one side
    body_in: Optional[float]   # scans on both sides: the distance between their faces, which includes the wall itself
    points: int
    scans: list[ScanOnWall] = field(default_factory=list)


def _segments(payloads: dict[str, dict], placements: dict[str, Placement]) -> list[dict]:
    """Every wall the bench found in every scan as a plan line in the world frame: the plane lines (planes), or the
    hand-corrected walls where there are any."""
    out = []
    for name, payload in payloads.items():
        p = placements.get(name)
        if p is None:
            continue
        rot = tilt_matrix3(p.pitch_deg, p.roll_deg, p.yaw_deg)
        for w in payload.get("planes", payload.get("walls", [])):
            ends = np.array([[w["x1"], w["y1"], 1.0], [w["x2"], w["y2"], 1.0]]) @ rot.T + [p.tx, p.ty, p.tz]
            a, b = ends[0, :2], ends[1, :2]
            length = float(np.linalg.norm(b - a))
            if length < 0.5:
                continue
            u = (b - a) / length
            out.append({"scan": name, "a": a, "b": b, "u": u, "n": np.array([-u[1], u[0]]), "length": length,
                        "behind_m": float(w.get("behind_m", 0.0))})
    return out


def _same_wall(s: dict, t: dict) -> bool:
    if np.degrees(np.arccos(np.clip(abs(s["u"] @ t["u"]), 0.0, 1.0))) > MAX_ANGLE_DEG:
        return False
    mid_t, mid_s = (t["a"] + t["b"]) / 2, (s["a"] + s["b"]) / 2
    if abs(s["n"] @ (mid_t - s["a"])) > MAX_OFFSET_M or abs(t["n"] @ (mid_s - t["a"])) > MAX_OFFSET_M:
        return False
    s0, s1 = sorted([(s["a"] - s["a"]) @ s["u"], (s["b"] - s["a"]) @ s["u"]])
    t0, t1 = sorted([(t["a"] - s["a"]) @ s["u"], (t["b"] - s["a"]) @ s["u"]])
    return max(s0, t0) - min(s1, t1) <= MAX_GAP_M


def cluster_walls(segments: list[dict]) -> list[list[dict]]:
    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if find(i) != find(j) and _same_wall(segments[i], segments[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, seg in enumerate(segments):
        groups.setdefault(find(i), []).append(seg)
    return list(groups.values())


def _consensus(cluster: list[dict]):
    """(point on the wall, direction, normal, along-span) of the length-weighted line through a cluster."""
    ref = cluster[0]["u"]
    dirs = np.array([s["u"] if s["u"] @ ref >= 0 else -s["u"] for s in cluster])
    w = np.array([s["length"] for s in cluster])
    u = (dirs * w[:, None]).sum(axis=0)
    u /= np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    mids = np.array([(s["a"] + s["b"]) / 2 for s in cluster])
    offset = float(((mids @ n) * w).sum() / w.sum())
    origin = mids[0] - n * (mids[0] @ n - offset)
    ends = np.array([e for s in cluster for e in (s["a"], s["b"])])
    along = (ends - origin) @ u
    return origin, u, n, (float(along.min()) + END_MARGIN_M, float(along.max()) - END_MARGIN_M)


def measure_walls(scans: list[Scan], placements: dict[str, Placement], payloads: dict[str, dict],
                  progress_cb=None, should_cancel: Optional[Callable[[], bool]] = None) -> list[WallReport]:
    """One report per wall that at least one scan detected, worst first. `should_cancel`, if given, is
    checked once per scan (placing every point at full resolution) and once per wall cluster."""
    clusters = cluster_walls(_segments(payloads, placements))
    world: dict[str, np.ndarray] = {}
    for i, scan in enumerate(scans):
        if should_cancel and should_cancel():
            raise CancelledError(f"cancelled after placing {i}/{len(scans)} scan(s)")
        p = placements.get(scan.name)
        if p is not None:
            m = p.matrix()
            world[scan.name] = (scan.points @ m[:3, :3].T + m[:3, 3]).astype(np.float32)
    reports: list[WallReport] = []
    for k, cluster in enumerate(clusters):
        if should_cancel and should_cancel():
            raise CancelledError(f"cancelled after measuring {k}/{len(clusters)} wall(s)")
        if progress_cb:
            progress_cb("measuring walls", f"{k + 1}/{len(clusters)}")
        origin, u, n, (s0, s1) = _consensus(cluster)
        if s1 - s0 < 0.5:
            continue
        per_scan: dict[str, np.ndarray] = {}
        for name in sorted({s["scan"] for s in cluster}):
            pts = world.get(name)
            if pts is None:
                continue
            floor = placements[name].tz
            rel = pts[:, :2] - origin
            dist, along = rel @ n, rel @ u
            keep = ((np.abs(dist) <= SLAB_M) & (along >= s0) & (along <= s1)
                    & (pts[:, 2] >= floor + BAND_ABOVE_FLOOR_M[0]) & (pts[:, 2] <= floor + BAND_ABOVE_FLOOR_M[1]))
            if keep.sum() >= MIN_POINTS:
                per_scan[name] = dist[keep].astype(float)
        if not per_scan:
            continue
        # The reference is the middle of where the scans put the wall, not of the pooled points: with two scans of equal
        # weight the pooled median sits at the edge of one of them.
        first = float(np.median([np.median(d) for d in per_scan.values()]))
        kept = {name: dist[np.abs(dist - first) <= KEEP_M] for name, dist in per_scan.items()}
        kept = {name: d for name, d in kept.items() if len(d) >= MIN_POINTS}
        if not kept:
            continue
        typical = {name: float(np.median(d)) for name, d in kept.items()}
        centre = float(np.median(list(typical.values())))
        rows, by_side = [], {}
        for name, near in kept.items():
            side = 1 if n @ (np.array([placements[name].tx, placements[name].ty]) - origin) >= 0 else -1
            rows.append(ScanOnWall(name, round((typical[name] - centre) / INCH_M, 2),
                                   round(1.4826 * float(np.median(np.abs(near - typical[name]))) / INCH_M, 2),
                                   int(len(near)), side))
            by_side.setdefault(side, []).append(near)
        biggest = max(by_side.values(), key=lambda parts: sum(len(x) for x in parts))
        merged = np.concatenate(biggest)
        thickness = float(np.percentile(merged, 95) - np.percentile(merged, 5)) / INCH_M
        body = None
        if len(by_side) == 2:
            a, b = (float(np.median(np.concatenate(v))) for v in by_side.values())
            body = abs(a - b) / INCH_M
        mid = origin + u * (s0 + s1) / 2
        reports.append(WallReport(x_m=round(float(mid[0]), 2), y_m=round(float(mid[1]), 2), length_m=round(s1 - s0, 2),
                                  angle_deg=round(float(np.degrees(np.arctan2(u[1], u[0])) % 180), 1),
                                  thickness_in=round(thickness, 2), body_in=None if body is None else round(body, 2),
                                  points=int(sum(r.points for r in rows)), scans=rows))
    return sorted(reports, key=lambda r: -r.thickness_in)


def summarise(reports: list[WallReport]) -> dict:
    """The headline numbers: how many walls, how thick a typical one reads, and how many are too thick to draw from."""
    if not reports:
        return {"walls": 0}
    t = np.array([r.thickness_in for r in reports])
    return {"walls": len(reports), "median_in": round(float(np.median(t)), 1), "p90_in": round(float(np.percentile(t, 90)), 1),
            "over_2_in": int((t > 2).sum()), "over_4_in": int((t > 4).sum()), "over_6_in": int((t > 6).sum())}
