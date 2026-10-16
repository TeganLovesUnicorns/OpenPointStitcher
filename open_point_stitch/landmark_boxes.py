"""Named landmarks (marked in review) as boxes in the venue, for the plan-view and 3D drawings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .placement import Placement, landmark_key


@dataclass
class LandmarkBox:
    name: str
    kind: str
    scan: str
    corners: np.ndarray        # (8, 3) in the venue: the four bottom corners, then the four top ones


def world_boxes(landmarks: list[dict], placements: dict[str, Placement]) -> list[LandmarkBox]:
    """Each landmark's box (in its scan's own frame) moved into the venue with that scan's placement."""
    out = []
    for e in landmarks:
        p = placements.get(e.get("scan"))
        if p is None or e.get("centroid") is None or not e.get("name"):
            continue
        c, half = np.asarray(e["centroid"], float), np.asarray(e.get("extent", (0, 0, 0)), float) / 2
        signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
        m = p.matrix()
        out.append(LandmarkBox(name=e["name"], kind=e.get("kind", "other"), scan=e["scan"],
                               corners=(c + signs * half) @ m[:3, :3].T + m[:3, 3]))
    return out


def label_points(boxes: list[LandmarkBox]) -> list[tuple[str, np.ndarray]]:
    """One label per landmark name, at the middle of every box that carries it."""
    groups: dict[str, list[LandmarkBox]] = {}
    for b in boxes:
        groups.setdefault(landmark_key(b.name), []).append(b)
    return [(f"{g[0].name} ({g[0].kind})", np.mean([b.corners.mean(axis=0) for b in g], axis=0))
            for g in groups.values()]
