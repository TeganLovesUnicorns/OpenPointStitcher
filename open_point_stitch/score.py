"""Scoring candidate alignments. Candidates are ranked against each other with hard physical gates, not tested one
by one against a fixed threshold. Also colour as a tolerant second opinion, and the trained judge for "Run from my
alignment" from a rough start."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .lib.files import TrainedModel
from .lib.geometry import fold_between
from .lib.poses import tilt_of

log = logging.getLogger(__name__)

@dataclass
class Verdict:
    """Everything known about one candidate, kept together so a rejected candidate can say why in the report."""
    transform: np.ndarray        # the 4x4 being judged
    origin: str = ""             # which strategy proposed it: walls, gravity, ransac...
    surface_fit: float = 0.0     # near and facing the same way
    fitness: float = 0.0         # the old number, kept for comparison and reports
    rmse: float = 0.0            # average distance of the points that did match
    normal_agree: float = 0.0    # how parallel the matched surfaces are, 0 to 1
    n_matched: int = 0           # how many points actually agreed
    tilt_deg: float = 0.0        # how far this tips the world off level
    rejected_by: str = ""        # empty means it survived every gate
    # Set by apply_detector_vote(). Until then vote_score equals surface_fit, so callers can always rank on
    # vote_score.
    plane_agreement: float = 0.0
    colour_agreement: float = 0.0
    n_planes_considered: int = 0
    corner_agreement: float = 0.0
    n_corners_considered: int = 0
    edge_agreement: float = 0.0
    n_edges_considered: int = 0
    opening_agreement: float = 0.0
    n_openings_considered: int = 0
    # landmarks: bounded bonus, never a veto
    landmark_agreement: float = 0.0
    n_landmarks_considered: int = 0
    vote_score: float = 0.0
    # The detectors alone, without surface_fit. vote_score says whether the points are near each other; this says
    # whether the building agrees, which decides whether a move is justified.
    detector_score: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.rejected_by


# Batched tensor nearest-neighbour search for score_candidate_fast.

def build_target_index(target: o3d.geometry.PointCloud,
                       device: Optional["o3d.core.Device"] = None) -> "o3d.core.nns.NearestNeighborSearch":
    """Build the target's search index ONCE per pair, outside any
    per-candidate loop, and reuse it for every candidate scored against
    that target."""
    device = device or o3d.core.Device("CPU:0")
    pts = np.asarray(target.points, dtype=np.float64)
    t = o3d.core.Tensor(pts, dtype=o3d.core.float64, device=device)
    nns = o3d.core.nns.NearestNeighborSearch(t)
    nns.knn_index()
    return nns


def score_candidate_fast(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    target_nns: "o3d.core.nns.NearestNeighborSearch",
    transform: np.ndarray,
    cfg: PipelineConfig,
    origin: str = "",
    device: Optional["o3d.core.Device"] = None,
    src_w: Optional[np.ndarray] = None,
    tgt_w: Optional[np.ndarray] = None,
) -> Verdict:
    """Same judgement as score_candidate, with a batched tensor nearest-neighbour search instead of a KD-tree loop.
    With per-point trust (src_w, tgt_w), surface fit counts each agreeing point by both points' trust; the gates
    still count points."""
    device = device or o3d.core.Device("CPU:0")
    T = np.asarray(transform, dtype=np.float64)
    v = Verdict(transform=T, origin=origin)

    src_pts = np.asarray(source.points)
    src_nrm = np.asarray(source.normals)
    moved = (T[:3, :3] @ src_pts.T).T + T[:3, 3]
    moved_nrm = (T[:3, :3] @ src_nrm.T).T

    v.tilt_deg = tilt_of(T)

    q = o3d.core.Tensor(moved, dtype=o3d.core.float64, device=device)
    idx_t, sq_t = target_nns.knn_search(q, 1)
    idx = idx_t.cpu().numpy().reshape(-1)
    dist = np.sqrt(sq_t.cpu().numpy().reshape(-1))

    window = cfg.voxel_size * cfg.icp_distance_mult
    near = dist <= window
    v.fitness = float(near.mean())
    v.rmse = float(dist[near].mean()) if near.any() else float("inf")

    if near.any():
        tgt_nrm = np.asarray(target.normals)
        dots = np.einsum('ij,ij->i', moved_nrm[near], tgt_nrm[idx[near]])
        dots = np.abs(dots)
        v.normal_agree = float(dots.mean())
        agreeing = dots > 0.8
        v.n_matched = int(agreeing.sum())
        v.surface_fit = float(v.n_matched / len(moved))
        if src_w is not None and tgt_w is not None:
            near_i = np.flatnonzero(near)[agreeing]
            v.surface_fit = float((src_w[near_i] * tgt_w[idx[near_i]]).sum() / src_w.sum())
    else:
        v.rmse = float("inf")

    if cfg.assume_level and v.tilt_deg > cfg.max_tilt_deg:
        v.rejected_by = f"tips the world {v.tilt_deg:.1f} deg off level"
    elif v.n_matched < cfg.min_correspondences:
        v.rejected_by = (f"only {v.n_matched} points agree on a surface, "
                         f"need {cfg.min_correspondences}")
    v.vote_score = v.surface_fit  # overwritten by apply_detector_vote when used
    return v


def _closeness(distance_m: float, scale_m: float) -> float:
    """1.0 when on top of each other, falling smoothly with distance and never reaching 0, so detectors can see
    translation."""
    x = distance_m / max(scale_m, 1e-9)
    return 1.0 / (1.0 + x * x)


def plane_colour_agreement(
    transform: np.ndarray, src_features, tgt_features, cfg: PipelineConfig,
) -> tuple[float, float, int]:
    """Do this candidate's surfaces line up with the target's, by normal and colour, weighted by support?
    Surfaces with no counterpart in bounds are skipped, not scored as disagreement. Features are
    find_surfaces.ScanFeatures (typed loosely to avoid a circular import).
    Returns (plane_agreement, colour_agreement, n_considered), both 0-1."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    plane_num = plane_den = 0.0
    colour_num = colour_den = 0.0
    n_considered = 0
    for s in src_features.surfaces:
        c = R @ s.centroid + t
        n = R @ s.normal
        best = None
        best_ang = cfg.vote_plane_angle_bound_deg
        for u in tgt_features.surfaces:
            dist = float(np.linalg.norm(c - u.centroid))
            if dist > cfg.vote_plane_position_bound_m:
                continue
            ang = fold_between(n, u.normal)
            if ang < best_ang:
                best, best_ang = u, ang
        if best is None:
            continue
        n_considered += 1
        w = float(s.support)
        angle_score = max(0.0, 1.0 - best_ang / cfg.vote_plane_angle_bound_deg)
        # Perpendicular distance, not centroid to centroid: a plane cannot constrain position along its own face.
        offset = abs(float((c - best.centroid) @ best.normal))
        plane_num += w * 0.5 * (angle_score + _closeness(offset, cfg.vote_position_scale_m))
        plane_den += w
        colour_num += w * hue_agreement(s.hue, s.saturation, best.hue, best.saturation, cfg)
        colour_den += w
    plane_agreement = (plane_num / plane_den) if plane_den > 0 else 0.0
    colour_agreement = (colour_num / colour_den) if colour_den > 0 else 0.0
    return plane_agreement, colour_agreement, n_considered


def corner_agreement(
    transform: np.ndarray, src_features, tgt_features, cfg: PipelineConfig,
) -> tuple[float, float, int]:
    """Do transformed corners land on the target's corners with a matching angle signature? Corners below
    vote_corner_min_cohesion on either side are ignored; angles compare as sorted triples.
    Returns (corner_agreement, corner_colour_agreement, n_considered)."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    angle_num = angle_den = 0.0
    colour_num = colour_den = 0.0
    n_considered = 0
    for c in src_features.corners:
        if c.cohesion < cfg.vote_corner_min_cohesion:
            continue
        p = R @ c.position + t
        best = None
        best_d = cfg.vote_corner_position_bound_m
        for u in tgt_features.corners:
            if u.cohesion < cfg.vote_corner_min_cohesion:
                continue
            d = float(np.linalg.norm(p - u.position))
            if d < best_d:
                best, best_d = u, d
        if best is None:
            continue
        # A corner's angles do not change when the cloud moves, so they only identify the corner. Gate on them.
        angle_diff = sum(abs(a - b) for a, b in zip(sorted(c.angles_deg), sorted(best.angles_deg)))
        if angle_diff > 3.0 * cfg.vote_corner_angle_tolerance_deg:
            continue
        n_considered += 1
        w = c.cohesion * best.cohesion
        # What IS alignment quality is where the corner landed. A corner is a
        # POINT, the one feature here that pins all three axes, so distance is
        # the whole score. Without this the vote was blind to translation.
        angle_num += w * _closeness(best_d, cfg.vote_position_scale_m)
        angle_den += w
        colour_num += w * hue_agreement(c.hue, c.saturation, best.hue, best.saturation, cfg)
        colour_den += w
    agreement = (angle_num / angle_den) if angle_den > 0 else 0.0
    colour_score = (colour_num / colour_den) if colour_den > 0 else 0.0
    return agreement, colour_score, n_considered


def edge_agreement(
    transform: np.ndarray, src_features, tgt_features, cfg: PipelineConfig,
) -> tuple[float, float, int]:
    """Do transformed junction lines lie along the target's? Direction, perpendicular gap, overlap and fold angle
    are compared, never midpoints. Returns (edge_agreement, edge_colour_agreement, n_considered)."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    cos_tol = float(np.cos(np.radians(cfg.vote_edge_angle_tolerance_deg)))
    num = den = 0.0
    colour_num = colour_den = 0.0
    n_considered = 0
    for e in src_features.edges:
        if e.cohesion < cfg.vote_edge_min_cohesion:
            continue
        mid = R @ e.midpoint + t
        direction = R @ e.direction
        best = None
        best_score = 0.0
        for u in tgt_features.edges:
            if u.cohesion < cfg.vote_edge_min_cohesion:
                continue
            parallel = abs(float(direction @ u.direction))
            if parallel < cos_tol:
                continue
            delta = mid - u.midpoint
            along = float(delta @ u.direction)
            gap = float(np.linalg.norm(delta - along * u.direction))
            if gap > cfg.vote_edge_position_bound_m:
                continue
            lo = max(along - e.length_m / 2, -u.length_m / 2)
            hi = min(along + e.length_m / 2, u.length_m / 2)
            if (hi - lo) < cfg.vote_edge_min_overlap_m:
                continue
            if abs(e.interior_angle_deg - u.interior_angle_deg) > cfg.vote_edge_fold_tolerance_deg:
                continue   # a different KIND of junction, not a worse look at this one
            # Averaged, not multiplied: multiplying three penalties scores even correct alignments too low.
            score = 0.5 * (_closeness(gap, cfg.vote_position_scale_m)
                           + max(0.0, 1.0 - np.arccos(min(parallel, 1.0))
                                 / np.radians(cfg.vote_edge_angle_tolerance_deg)))
            if score > best_score:
                best, best_score = u, float(score)
        if best is None:
            continue
        n_considered += 1
        w = e.cohesion * best.cohesion * min(e.length_m, best.length_m)
        num += w * best_score
        den += w
        colour_num += w * hue_agreement(e.hue, e.saturation,
                                               best.hue, best.saturation, cfg)
        colour_den += w
    agreement = (num / den) if den > 0 else 0.0
    colour_score = (colour_num / colour_den) if colour_den > 0 else 0.0
    return agreement, colour_score, n_considered


def opening_agreement(
    transform: np.ndarray, src_features, tgt_features, cfg: PipelineConfig,
) -> tuple[float, int]:
    """Do this candidate's doors and windows land on the other scan's, on a wall, the same way round and of the
    same size? An opening with no counterpart is skipped: the other scan often never saw it.
    Returns (agreement, n_considered)."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    cos_tol = float(np.cos(np.radians(cfg.vote_opening_angle_tolerance_deg)))
    num = den = 0.0
    n_considered = 0
    for o in src_features.openings:
        if o.cohesion < cfg.vote_opening_min_cohesion:
            continue
        p = R @ o.position + t
        direction = R @ o.direction
        best_closeness = None
        best_cohesion = 0.0
        for u in tgt_features.openings:
            if u.kind != o.kind or u.cohesion < cfg.vote_opening_min_cohesion:
                continue
            if abs(float(direction @ u.direction)) < cos_tol:
                continue
            if (abs(o.width - u.width) > cfg.vote_opening_size_tolerance_m
                    or abs(o.height - u.height) > cfg.vote_opening_size_tolerance_m):
                continue
            delta = p - u.position
            # Across the wall the two readings differ by the wall's thickness, so that much is not a disagreement.
            across = max(0.0, abs(float(delta @ u.wall_normal)) - cfg.vote_opening_wall_thickness_m)
            gap = float(np.sqrt(float(delta @ u.direction) ** 2 + float(delta[2]) ** 2 + across ** 2))
            if gap > cfg.vote_opening_position_bound_m:
                continue
            closeness = _closeness(gap, cfg.vote_position_scale_m)
            if best_closeness is None or closeness > best_closeness:
                best_closeness, best_cohesion = closeness, u.cohesion
        if best_closeness is None:
            continue
        n_considered += 1
        w = o.cohesion * best_cohesion
        num += w * best_closeness
        den += w
    return ((num / den) if den > 0 else 0.0), n_considered


def landmark_agreement(
    transform: np.ndarray, src_features, tgt_features, cfg: PipelineConfig,
) -> tuple[float, int]:
    """Do this candidate's landmarks (furniture and fixtures) land on the other scan's, at a compatible height and
    footprint? A landmark with no counterpart is skipped: the other scan often never saw it. Bonus only, never a
    veto. Returns (agreement, n_considered)."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    num = den = 0.0
    n_considered = 0
    for lm in src_features.landmarks:
        if lm.raw_xyz is None:
            continue
        p = R @ lm.raw_xyz + t
        best_closeness = None
        best_weight = 0.0
        for u in tgt_features.landmarks:
            if u.raw_xyz is None:
                continue
            if abs(lm.h - u.h) > cfg.vote_landmark_height_tolerance_m:
                continue
            a, b = max(lm.length, 0.05), max(u.length, 0.05)
            if max(a / b, b / a) > cfg.vote_landmark_size_ratio:
                continue
            gap = float(np.linalg.norm(p - u.raw_xyz))
            if gap > cfg.vote_landmark_position_bound_m:
                continue
            closeness = _closeness(gap, cfg.vote_position_scale_m)
            if best_closeness is None or closeness > best_closeness:
                best_closeness, best_weight = closeness, u.weight
        if best_closeness is None:
            continue
        n_considered += 1
        w = lm.weight * best_weight
        num += w * best_closeness
        den += w
    return ((num / den) if den > 0 else 0.0), n_considered


def apply_detector_vote(
    v: Verdict, plane_agreement: float, colour_agreement: float,
    n_considered: int, cfg: PipelineConfig,
    corner_agreement: float = 0.0, corner_colour_agreement: float = 0.0, n_corners: int = 0,
    edge_agreement: float = 0.0, edge_colour_agreement: float = 0.0, n_edges: int = 0,
    opening_agreement: float = 0.0, n_openings: int = 0,
    landmark_agreement: float = 0.0, n_landmarks: int = 0,
) -> None:
    """Fold the detectors into v in place: weight of evidence across the detectors that had something to say.
    surface_fit stays the base ranking signal."""
    v.plane_agreement = plane_agreement
    v.colour_agreement = colour_agreement
    v.n_planes_considered = n_considered
    v.corner_agreement = corner_agreement
    v.n_corners_considered = n_corners
    v.edge_agreement = edge_agreement
    v.n_edges_considered = n_edges
    v.opening_agreement = opening_agreement
    v.n_openings_considered = n_openings
    v.landmark_agreement = landmark_agreement
    v.n_landmarks_considered = n_landmarks

    # Only detectors that saw something get a say, and they are not rescaled against each other.
    voices = []
    if n_considered > 0:
        voices.append((cfg.vote_weight_plane, plane_agreement, n_considered))
    if n_corners > 0:
        voices.append((cfg.vote_weight_corner, corner_agreement, n_corners))
    if n_edges > 0:
        voices.append((cfg.vote_weight_edge, edge_agreement, n_edges))

    # Openings and landmarks move the ranking but stay out of the veto test, so they cannot reject a candidate.
    gate_voices = list(voices)
    gate_weight = sum(w for w, _a, _n in gate_voices)
    gate_score = (sum(w * a for w, a, _n in gate_voices) / gate_weight) if gate_weight else 0.0
    if n_openings > 0:
        voices.append((cfg.vote_weight_opening, opening_agreement, n_openings))
    if n_landmarks > 0:
        voices.append((cfg.vote_weight_landmark, landmark_agreement, n_landmarks))

    weight_total = sum(w for w, _a, _n in voices)
    v.detector_score = (sum(w * a for w, a, _n in voices) / weight_total) if weight_total else 0.0
    if v.rejected_by:
        return
    if not voices:
        return  # nothing to compare anywhere - vote_score stays == surface_fit

    # One combined verdict, not a chain of per-detector vetoes. It needs
    # enough evidence behind it to be worth trusting: a single match is a
    # coincidence, several agreeing is a finding.
    evidence = sum(n for _w, _a, n in gate_voices)
    if (len(gate_voices) >= cfg.vote_min_detectors
            and evidence >= cfg.vote_min_evidence
            and gate_score < cfg.vote_min_agreement):
        v.rejected_by = (
            f"the building does not agree: {gate_score:.2f} combined agreement "
            f"across {len(gate_voices)} detector(s) over {evidence} match(es)"
        )
        # A caller that only compares vote_score (not .ok) - register_pair's
        # own candidate loop does - must still never pick a vetoed candidate.
        v.vote_score = -1.0
        return

    colour_weight_total = n_considered + n_corners + n_edges
    combined_colour = (
        (colour_agreement * n_considered + corner_colour_agreement * n_corners
         + edge_colour_agreement * n_edges) / colour_weight_total
        if colour_weight_total > 0 else 0.0
    )
    bonus = v.detector_score + cfg.vote_weight_colour * combined_colour
    v.vote_score = v.surface_fit * (1.0 + bonus)


def rank(verdicts: list[Verdict]) -> list[Verdict]:
    """Survivors only, best first by vote_score."""
    return sorted([v for v in verdicts if v.ok], key=lambda v: -v.vote_score)


def confidence(ranked: list[Verdict]) -> tuple[str, str]:
    """How far to trust the winner, as (level, why). Reported, never acted on."""
    if not ranked:
        return "none", "every candidate failed a physical check"
    if len(ranked) == 1:
        return "unchecked", "only one candidate survived, nothing to compare it against"

    best, second = ranked[0], ranked[1]
    if best.surface_fit <= 0:
        return "none", "the best candidate matched no surfaces at all"

    margin = (best.surface_fit - second.surface_fit) / best.surface_fit
    if margin >= 0.25:
        return "clear", f"the best fit is {margin*100:.0f}% ahead of the next one"
    if margin >= 0.10:
        return "workable", (f"the best fit is only {margin*100:.0f}% ahead of the "
                            f"next one, worth eyeballing")
    return "ambiguous", (f"the top two are within {margin*100:.0f}% of each other, "
                         f"so this pair is a guess. Use your rough placement.")


# -------------------------------------------------------------------------------------------------------------------
# Colour as a tolerant second opinion: hue families and local colour composition.
# -------------------------------------------------------------------------------------------------------------------

def hue_family(hue_deg: float, saturation: float, cfg) -> str:
    """Nearest named hue family, or "neutral" below `cfg.colour_saturation_floor` (C2).
    Nearest-centre over `cfg.colour_family_edges`, so no special case at the 0/360 wrap."""
    if np.isnan(hue_deg) or saturation < cfg.colour_saturation_floor:
        return "neutral"
    h = hue_deg % 360.0
    best_name, best_d = "", 361.0
    for name, centre in cfg.colour_family_edges:
        d = abs(h - centre) % 360.0
        d = min(d, 360.0 - d)
        if d < best_d:
            best_name, best_d = name, d
    return best_name


def composition(
    families: Iterable[str], weights: Iterable[float],
) -> dict[str, float]:
    """Colour signature (C4): weighted proportion of each family; {} for no input."""
    totals: dict[str, float] = {}
    total_w = 0.0
    for fam, w in zip(families, weights):
        w = float(w)
        if w <= 0:
            continue
        totals[fam] = totals.get(fam, 0.0) + w
        total_w += w
    if total_w <= 0:
        return {}
    return {fam: w / total_w for fam, w in totals.items()}


def composition_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """0 to 1 histogram intersection of two compositions. Two empty ones agree (1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    return float(sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys))


def hue_agreement(hue_a: float, sat_a: float, hue_b: float, sat_b: float, cfg) -> float:
    """0-1 agreement of two hues: neutral+neutral 1, neutral+coloured 0, else a circular
    tolerance (`colour_hue_tolerance_deg`), never a family-boundary cliff."""
    fam_a = hue_family(hue_a, sat_a, cfg)
    fam_b = hue_family(hue_b, sat_b, cfg)
    if fam_a == "neutral" and fam_b == "neutral":
        return 1.0
    if fam_a == "neutral" or fam_b == "neutral":
        return 0.0
    d = abs(hue_a - hue_b) % 360.0
    d = min(d, 360.0 - d)
    return float(max(0.0, 1.0 - d / cfg.colour_hue_tolerance_deg))


# -------------------------------------------------------------------------------------------------------------------
# The trained judge for "Run from my alignment" from a rough start: one list of clues per candidate, shared by
# training and the run, and the model's probability that a candidate is right.
# -------------------------------------------------------------------------------------------------------------------

FEATURES = ["surface_fit", "fitness", "rmse", "normal_agree", "n_matched", "tilt_deg", "plane_agreement",
            "colour_agreement", "n_planes_considered", "corner_agreement", "n_corners_considered", "edge_agreement",
            "n_edges_considered", "opening_agreement", "n_openings_considered", "landmark_agreement",
            "n_landmarks_considered", "vote_score", "detector_score",
            "move_m", "move_deg", "is_proposer", "proposer_agree", "fit_gain"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_judge.json")
_MODEL = TrainedModel(MODEL_PATH, FEATURES, "the seed judge")


def clues(values: dict) -> np.ndarray:
    """FEATURES in order from a dict; a missing, empty or non-finite clue reads as 0, as the answer key writes it."""
    out = []
    for f in FEATURES:
        v = values.get(f, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        out.append(v if np.isfinite(v) else 0.0)
    return np.array(out)


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    """Probability each row is right: boosted stumps or a logistic fit, whichever `model` holds."""
    x = np.atleast_2d(x)
    if "stumps" in model:
        f = sum(a * np.where(x[:, j] > t, s, -s) for j, t, s, a in model["stumps"])
        return 1 / (1 + np.exp(-2 * f))
    z = (x - np.asarray(model["mean"])) / np.asarray(model["std"])
    return 1 / (1 + np.exp(-(z @ np.asarray(model["w"]) + model["b"])))


def load_model(path: str = MODEL_PATH) -> dict:
    return _MODEL.load(path)
