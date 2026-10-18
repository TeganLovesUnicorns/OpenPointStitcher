"""Per-scan placement: the data model behind the placement bench.

Each scan has x, y, z, yaw, hand-set pitch/roll and its levelling rotation, and converts to a 4x4 transform.
Absolute placements are per scan; relative placements are per pair."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .lib.files import write_json_atomic
from .lib.poses import tilt_matrix3

log = logging.getLogger(__name__)


@dataclass
class Placement:
    """Where one scan sits in the shared world frame."""

    name: str
    level_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    floor_z: float = 0.0
    yaw_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    # hand tilt correction (Alt+arrow in the bench); the solver never sets these
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    locked: bool = False

    @classmethod
    def from_dict(cls, s: dict, strict: bool = False, prefix: str = "") -> "Placement":
        """One scan's entry from a placement file. Missing tilt reads as zero (older files have none); missing yaw or
        position reads as zero too unless strict, which raises KeyError instead. prefix="human_" reads the hand
        placement saved beside the pose."""
        pose = (lambda k: float(s[prefix + k])) if strict else (lambda k: float(s.get(prefix + k, 0.0)))
        return cls(
            name=s["name"],
            level_rotation=np.array(s.get("level_rotation", np.eye(3).tolist()), dtype=float),
            floor_z=float(s.get("floor_z_local_m", 0.0)),
            yaw_deg=pose("yaw_deg"), tx=pose("x_m"), ty=pose("y_m"), tz=pose("z_m"),
            pitch_deg=float(s.get(prefix + "pitch_deg", 0.0)), roll_deg=float(s.get(prefix + "roll_deg", 0.0)),
            locked=bool(s.get("locked", False)),
        )

    def matrix(self) -> np.ndarray:
        """4x4 taking this scan's RAW points into the shared world frame."""
        m = np.eye(4)
        # Level first, then yaw, then the hand-set pitch/roll last, so pitch is always about world X and roll about
        # world Y whatever the scan's yaw.
        tilt = tilt_matrix3(self.pitch_deg, self.roll_deg, self.yaw_deg)
        m[:3, :3] = tilt @ self.level_rotation
        # The bench draws a scan's height above its own floor and tips it about that floor, so the floor level comes
        # off before rotating.
        m[:3, 3] = np.array([self.tx, self.ty, self.tz]) - tilt @ np.array([0.0, 0.0, self.floor_z])
        return m


def relative_transform(source: Placement, target: Placement) -> np.ndarray:
    """Transform mapping source's RAW points into target's RAW frame."""
    return np.linalg.inv(target.matrix()) @ source.matrix()


SCHEMA = "open_point_stitch.placement.v1"
# Files saved before the project was renamed (0.50.0) carry this one; same layout.
LEGACY_SCHEMAS = ("nomad_stitch.placement.v1",)


def is_placement_schema(value) -> bool:
    """True for the current placement schema or one saved under the project's old name."""
    return value == SCHEMA or value in LEGACY_SCHEMAS


def save_placements(path: str, placements: Iterable[Placement],
                    note: str = "") -> None:
    payload = {
        "schema": SCHEMA,
        "note": note or ("per-scan yaw + xyz in a shared world frame; scans are "
                         "gravity levelled, so roll and pitch are not stored"),
        "scans": [
            {
                "name": p.name,
                "yaw_deg": round(float(p.yaw_deg), 4),
                "pitch_deg": round(float(p.pitch_deg), 4),
                "roll_deg": round(float(p.roll_deg), 4),
                "x_m": round(float(p.tx), 5),
                "y_m": round(float(p.ty), 5),
                "z_m": round(float(p.tz), 5),
                "floor_z_local_m": round(float(p.floor_z), 5),
                "level_rotation": p.level_rotation.tolist(),
                "locked": bool(p.locked),
            }
            for p in placements
        ],
    }
    write_json_atomic(path, payload, indent=2)


@dataclass
class AnchorMember:
    """One scan's region of a link, in that scan's raw local frame."""

    scan: str
    points: list[tuple[float, float, float]]   # raw local frame; may be a
                                                 # subsample of a larger pick
    centroid: tuple[float, float, float]
    extent: tuple[float, float, float]          # dx, dy, dz bounding size
    n_points: int                               # TRUE count, even if `points` is thinned for storage


@dataclass
class Anchor:
    """One physical feature seen by two or more scans: a group, not a pair."""

    id: str
    members: list[AnchorMember] = field(default_factory=list)
    label: str = ""


@dataclass
class Exclusion:
    """A region of one scan left out of alignment (and of the export, for review removals or when the setting says
    so), in the raw local frame. exclusions.filter_scan reads centroid/extent; points is only for redrawing."""

    id: str
    scan: str
    points: list[tuple[float, float, float]]
    centroid: tuple[float, float, float]
    extent: tuple[float, float, float]
    n_points: int
    # Hand-drawn outlines from review. When present they decide, and the box is only for drawing.
    shapes: list = field(default_factory=list)
    source: str = ""              # "review" for the review view's removals, "" for an Exclude region box

    @property
    def is_removal(self) -> bool:
        """Removed in review (people, drift): out of everything. Otherwise an exclude region: out of alignment only,
        and in the export by default."""
        return self.source == "review"

    @classmethod
    def from_dict(cls, e: dict) -> "Exclusion":
        """One `exclusions` entry of a placement file, the same way for every reader."""
        return cls(id=e["id"], scan=e["scan"], points=[tuple(pt) for pt in e.get("points", [])],
                   centroid=tuple(e["centroid"]), extent=tuple(e.get("extent", (0.0, 0.0, 0.0))),
                   n_points=int(e.get("n_points", len(e.get("points", [])))),
                   shapes=list(e.get("shapes") or []), source=str(e.get("source", "")))


@dataclass
class LoadedPlacements:
    """Everything a saved placement file holds."""

    placements: dict[str, Placement]
    anchors: list[Anchor]
    exclusions: list[Exclusion] = field(default_factory=list)


def _migrate_old_pin(p: dict, placements: dict[str, Placement], path: str) -> Optional[Anchor]:
    """Convert an old {a, a_xy, b, b_xy} pin to a two-member link (height assumed at the floor)."""
    end_a, end_b = (p.get("a"), p.get("a_xy")), (p.get("b"), p.get("b_xy"))
    members = []
    for name, xy in (end_a, end_b):
        if name not in placements or xy is None:
            log.warning("%s: could not migrate a pin naming %r, skipping it", path, name)
            return None
        placement = placements[name]
        leveled = np.array([xy[0], xy[1], placement.floor_z])
        raw = leveled @ placement.level_rotation
        members.append(AnchorMember(
            scan=name, points=[tuple(float(v) for v in raw)],
            centroid=tuple(float(v) for v in raw), extent=(0.0, 0.0, 0.0), n_points=1,
        ))
    return Anchor(id=f"migrated-{end_a[0]}-{end_b[0]}", members=members,
                  label="migrated from a point pin, height approximate")


MAX_LANDMARK_POINTS = 400      # points kept per scan to seed the match; the box decides what is matched


def landmark_key(name: str) -> str:
    """Two landmarks are the same feature when their names match, ignoring case and spacing."""
    return " ".join(str(name).lower().split())


def anchors_from_landmarks(landmarks: list[dict]) -> list[Anchor]:
    """Landmarks named in review: the same name in two or more scans is one link between them, each member the box
    around its marked points. A name in one scan only links nothing yet; two marks of one name in the same scan are
    one member (their boxes joined)."""
    groups: dict[str, dict[str, list[dict]]] = {}
    for e in landmarks:
        if e.get("name") and e.get("scan") and e.get("centroid") is not None:
            groups.setdefault(landmark_key(e["name"]), {}).setdefault(e["scan"], []).append(e)
    out = []
    for key, by_scan in sorted(groups.items()):
        if len(by_scan) < 2:
            continue
        members = []
        for scan, marks in sorted(by_scan.items()):
            lo = np.min([np.asarray(m["centroid"]) - np.asarray(m.get("extent", (0, 0, 0))) / 2 for m in marks], axis=0)
            hi = np.max([np.asarray(m["centroid"]) + np.asarray(m.get("extent", (0, 0, 0))) / 2 for m in marks], axis=0)
            pts = [tuple(p) for m in marks for p in m.get("points", [])][:MAX_LANDMARK_POINTS]
            members.append(AnchorMember(scan=scan, points=pts, centroid=tuple((lo + hi) / 2), extent=tuple(hi - lo),
                                        n_points=int(sum(m.get("n_points", 0) for m in marks))))
        name = next(iter(by_scan.values()))[0]["name"]
        out.append(Anchor(id=f"landmark:{key}", members=members, label=f"landmark: {name}"))
    return out


def load_placements(path: str) -> LoadedPlacements:
    with open(path) as fh:
        raw = json.load(fh)
    if not is_placement_schema(raw.get("schema")):
        raise ValueError(
            f"{path}: expected schema {SCHEMA!r}, found {raw.get('schema')!r}"
        )
    out = {}
    all_identity = True
    for s in raw.get("scans", []):
        out[s["name"]] = Placement.from_dict(s)
        if not np.allclose(out[s["name"]].level_rotation, np.eye(3)):
            all_identity = False

    if out and all_identity:
        log.warning(
            "%s: every scan has an identity level_rotation. That is what a "
            "placement saved BEFORE the bench started exporting level_rotation "
            "looks like, not necessarily "
            "correct data. If this file came from the bench, re-save it; if "
            "every one of these scans is genuinely unrotated, ignore this.",
            path,
        )

    anchors: list[Anchor] = []
    for a in raw.get("anchors", []):
        members = [
            AnchorMember(
                scan=m["scan"],
                points=[tuple(pt) for pt in m.get("points", [])],
                centroid=tuple(m["centroid"]),
                extent=tuple(m.get("extent", (0.0, 0.0, 0.0))),
                n_points=int(m.get("n_points", len(m.get("points", [])))),
            )
            for m in a.get("members", [])
        ]
        anchors.append(Anchor(id=a["id"], members=members, label=a.get("label", "")))

    old_pins = raw.get("pins", [])
    if old_pins:
        log.warning(
            "%s: %d anchor(s) saved in the old point-pin format. Converting "
            "each to a 2-member region anchor now; re-save from the bench to "
            "make this permanent.",
            path, len(old_pins),
        )
        for p in old_pins:
            migrated = _migrate_old_pin(p, out, path)
            if migrated:
                anchors.append(migrated)

    anchors += anchors_from_landmarks(raw.get("landmarks", []))      # named in review
    exclusions = [Exclusion.from_dict(e) for e in raw.get("exclusions", [])]

    return LoadedPlacements(placements=out, anchors=anchors, exclusions=exclusions)


# -------------------------------------------------------------------------------------------------------------------
# Named landmarks (marked in review) as boxes in the venue, for the plan-view and 3D drawings.
# -------------------------------------------------------------------------------------------------------------------

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
