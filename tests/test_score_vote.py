"""Tests for score.py's detector vote: plane_colour_agreement and apply_detector_vote, which let planes, corners
and colour vote on a candidate instead of raw point proximity deciding alone. Synthetic surfaces only.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.find_surfaces import Surface, ScanFeatures  # noqa: E402
from open_point_stitch.find_corners import Corner, Edge  # noqa: E402
from open_point_stitch.score import (  # noqa: E402
    Verdict, plane_colour_agreement, corner_agreement, edge_agreement,
    apply_detector_vote,
)

CFG = PipelineConfig()


def _corner(index, position, angles_deg=(90.0, 90.0, 90.0), cohesion=1.0,
            hue=float("nan"), saturation=0.0):
    return Corner(
        position=np.array(position, dtype=float), surfaces=(0, 1, 2),
        angles_deg=tuple(angles_deg),
        interior_angles_deg=tuple(180.0 - a for a in angles_deg), cohesion=cohesion,
        hue=hue, saturation=saturation, value=0.5, index=index,
    )


def _surface(index, normal, centroid, support=1000, hue=float("nan"), saturation=0.0):
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)
    c = np.array(centroid, dtype=float)
    return Surface(
        normal=n, offset=float(n @ c), centroid=c,
        support=support, area=4.0, extent_u=2.0, extent_v=2.0, corners=4,
        hue=hue, saturation=saturation, value=0.5, index=index,
    )


def _features(surfaces, corners=None, edges=None):
    return ScanFeatures(surfaces=surfaces, residual_idx=np.array([], dtype=np.int64),
                        corners=corners or [], edges=edges or [])


def _edge(index, midpoint, direction, length_m=4.0, interior_angle_deg=90.0,
          cohesion=1.0, hue=float("nan"), saturation=0.0):
    d = np.array(direction, dtype=float)
    return Edge(
        midpoint=np.array(midpoint, dtype=float), direction=d / np.linalg.norm(d),
        length_m=length_m, surfaces=(0, 1), interior_angle_deg=interior_angle_deg,
        cohesion=cohesion, hue=hue, saturation=saturation, value=0.5, index=index,
    )


def test_plane_colour_agreement_matches_an_identical_wall():
    src = _features([_surface(0, [1, 0, 0], [0, 0, 1], hue=20.0, saturation=0.6)])
    tgt = _features([_surface(0, [1, 0, 0], [0, 0, 1], hue=20.0, saturation=0.6)])
    plane, colour_score, n = plane_colour_agreement(np.eye(4), src, tgt, CFG)
    assert n == 1
    assert plane > 0.99
    assert colour_score > 0.99


def test_plane_colour_agreement_nothing_in_range_is_zero_not_a_guess():
    src = _features([_surface(0, [1, 0, 0], [0, 0, 0])])
    tgt = _features([_surface(0, [1, 0, 0], [50, 50, 50])])  # far away
    plane, colour_score, n = plane_colour_agreement(np.eye(4), src, tgt, CFG)
    assert n == 0 and plane == 0.0 and colour_score == 0.0


def test_plane_colour_agreement_perpendicular_walls_are_outside_the_angle_bound():
    src = _features([_surface(0, [1, 0, 0], [0, 0, 0])])
    tgt = _features([_surface(0, [0, 1, 0], [0.1, 0.1, 0])])  # 90deg off, still close enough in position
    plane, colour_score, n = plane_colour_agreement(np.eye(4), src, tgt, CFG)
    # Outside vote_plane_angle_bound_deg entirely - never even shortlisted,
    # so this is "nothing to compare", not "compared and disagreed".
    assert n == 0


def test_plane_colour_agreement_a_real_within_bound_tilt_scores_partial():
    tilt_deg = CFG.vote_plane_angle_bound_deg / 2.0  # inside the bound, not a perfect match either
    tilted_normal = [np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg)), 0.0]
    src = _features([_surface(0, [1, 0, 0], [0, 0, 0])])
    tgt = _features([_surface(0, tilted_normal, [0.1, 0.1, 0])])
    plane, _colour_score, n = plane_colour_agreement(np.eye(4), src, tgt, CFG)
    assert n == 1
    assert 0.3 < plane < 0.7, f"expected a partial score near the bound's midpoint, got {plane}"


def test_plane_colour_agreement_weights_by_support():
    # A tiny surface agrees perfectly, a big one disagrees on colour only -
    # the big one should dominate the colour score.
    src = _features([
        _surface(0, [1, 0, 0], [0, 0, 0], support=10, hue=20.0, saturation=0.6),
        _surface(1, [1, 0, 0], [0, 0, 5], support=10000, hue=200.0, saturation=0.6),
    ])
    tgt = _features([
        _surface(0, [1, 0, 0], [0, 0, 0], hue=20.0, saturation=0.6),
        _surface(1, [1, 0, 0], [0, 0, 5], hue=20.0, saturation=0.6),
    ])
    _plane, colour_score, n = plane_colour_agreement(np.eye(4), src, tgt, CFG)
    assert n == 2
    assert colour_score < 0.1, f"the big disagreeing surface should dominate, got {colour_score}"


def test_apply_detector_vote_noop_when_nothing_was_compared():
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.0, colour_agreement=0.0, n_considered=0, cfg=CFG)
    assert v.vote_score == 0.08
    assert v.rejected_by == ""


def test_apply_detector_vote_adds_a_bounded_bonus_on_agreement():
    """Planes are the only detector with anything to look at, so the
    combined verdict IS the plane verdict - a weighted mean over the
    detectors that spoke, not a sum over all of them."""
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=1.0, colour_agreement=1.0,
                        n_considered=CFG.vote_min_planes, cfg=CFG)
    expected = 0.08 * (1.0 + 1.0 + CFG.vote_weight_colour)
    assert abs(v.vote_score - expected) < 1e-9
    assert abs(v.detector_score - 1.0) < 1e-9
    assert v.vote_score > v.surface_fit
    assert v.rejected_by == ""


def test_a_silent_detector_does_not_drag_the_verdict_down():
    """A detector with nothing in view (no corners on a blank wall) is left out, not scored zero."""
    speaking = Verdict(transform=np.eye(4), surface_fit=0.08)
    speaking.vote_score = speaking.surface_fit
    apply_detector_vote(speaking, plane_agreement=0.9, colour_agreement=0.0,
                        n_considered=4, cfg=CFG)

    silent_others = Verdict(transform=np.eye(4), surface_fit=0.08)
    silent_others.vote_score = silent_others.surface_fit
    apply_detector_vote(silent_others, plane_agreement=0.9, colour_agreement=0.0,
                        n_considered=4, cfg=CFG,
                        corner_agreement=0.0, n_corners=0,
                        edge_agreement=0.0, n_edges=0)
    assert abs(speaking.detector_score - silent_others.detector_score) < 1e-12, (
        "detectors with nothing to look at changed the verdict"
    )
    assert abs(silent_others.detector_score - 0.9) < 1e-9


def test_two_detectors_disagreeing_together_do_reject():
    """Weight of evidence: corroborated disagreement still rejects."""
    v = Verdict(transform=np.eye(4), surface_fit=0.12)  # scores WELL geometrically
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.05, colour_agreement=0.9,
                        n_considered=4, cfg=CFG,
                        edge_agreement=0.05, n_edges=3)
    assert v.rejected_by, "two detectors disagreeing over real evidence must reject"
    assert v.vote_score < 0, "a rejected candidate must never win a max() comparison"


def test_one_detector_cannot_sink_a_candidate_on_its_own():
    """One detector disagreeing alone cannot sink a candidate two others agree on."""
    alone = Verdict(transform=np.eye(4), surface_fit=0.12)
    alone.vote_score = alone.surface_fit
    apply_detector_vote(alone, plane_agreement=0.05, colour_agreement=0.0,
                        n_considered=8, cfg=CFG)
    assert alone.rejected_by == "", "a lone detector rejected a candidate by itself"

    outvoted = Verdict(transform=np.eye(4), surface_fit=0.12)
    outvoted.vote_score = outvoted.surface_fit
    apply_detector_vote(outvoted, plane_agreement=0.05, colour_agreement=0.0,
                        n_considered=2, cfg=CFG,
                        corner_agreement=0.95, n_corners=3,
                        edge_agreement=0.95, n_edges=4)
    assert outvoted.rejected_by == "", (
        "one unhappy detector outvoted by two happy ones must not reject"
    )
    assert outvoted.detector_score > CFG.vote_min_agreement


def test_apply_detector_vote_colour_alone_never_vetoes():
    """Colour is a second opinion, never the deciding vote: bad colour agreement with good plane agreement must not
    veto."""
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.9, colour_agreement=0.0,
                        n_considered=CFG.vote_min_planes, cfg=CFG)
    assert v.rejected_by == ""
    assert v.vote_score >= v.surface_fit


def test_apply_detector_vote_needs_enough_surfaces_to_veto():
    """One spurious plane match disagreeing should not, by itself, veto a
    candidate - vote_min_planes exists specifically to require more than
    one piece of evidence first."""
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.0, colour_agreement=0.0,
                        n_considered=1, cfg=CFG)
    assert CFG.vote_min_planes > 1, "test assumes the real default requires more than one"
    assert v.rejected_by == ""


def test_apply_detector_vote_does_not_override_an_existing_rejection():
    v = Verdict(transform=np.eye(4), surface_fit=0.08, rejected_by="tips the world 20.0 deg off level")
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=1.0, colour_agreement=1.0,
                        n_considered=CFG.vote_min_planes, cfg=CFG)
    assert v.rejected_by == "tips the world 20.0 deg off level"
    assert v.vote_score == 0.08


# --- corner_agreement --------------------------------------------------------

def test_corner_agreement_matches_an_identical_corner():
    src = _features([], [_corner(0, [1, 2, 3], (90, 90, 90))])
    tgt = _features([], [_corner(0, [1, 2, 3], (90, 90, 90))])
    agreement, colour_score, n = corner_agreement(np.eye(4), src, tgt, CFG)
    assert n == 1 and agreement > 0.99


def test_corner_agreement_ignores_a_low_cohesion_corner_on_either_side():
    src = _features([], [_corner(0, [0, 0, 0], cohesion=0.1)])  # below vote_corner_min_cohesion
    tgt = _features([], [_corner(0, [0, 0, 0], cohesion=1.0)])
    agreement, _colour_score, n = corner_agreement(np.eye(4), src, tgt, CFG)
    assert n == 0 and agreement == 0.0


def test_corner_agreement_different_angle_signatures_are_not_the_same_corner():
    """Two corners in the same place with different shapes are two different corners. The angle signature identifies
    a corner; it gates rather than scores."""
    src = _features([], [_corner(0, [0, 0, 0], (90, 90, 90))])
    tgt = _features([], [_corner(0, [0, 0, 0], (60, 60, 150))])  # same position, different shape
    agreement, _colour_score, n = corner_agreement(np.eye(4), src, tgt, CFG)
    assert n == 0, "a shape mismatch means it is not the same corner at all"
    assert agreement == 0.0


def test_corner_agreement_scores_how_far_off_the_alignment_is():
    """The point of a corner: it pins all three axes, so the score has to
    move when the candidate is translated. Before this it did not, and the
    whole vote was blind to translation."""
    src = _features([], [_corner(0, [0, 0, 0])])
    tgt = _features([], [_corner(0, [0, 0, 0])])
    exact, _c, n_exact = corner_agreement(np.eye(4), src, tgt, CFG)
    assert n_exact == 1 and exact > 0.999

    previous = exact
    for offset_m in (0.01, 0.05, 0.10, 0.30):
        T = np.eye(4)
        T[0, 3] = offset_m
        shifted, _c2, n = corner_agreement(T, src, tgt, CFG)
        assert n == 1, f"still the same corner at {offset_m} m, just misaligned"
        assert shifted < previous, (
            f"moving {offset_m} m off must score worse than the step before, "
            f"got {shifted} against {previous}"
        )
        previous = shifted


def test_corner_agreement_nothing_within_the_position_bound_is_zero():
    src = _features([], [_corner(0, [0, 0, 0])])
    tgt = _features([], [_corner(0, [50, 50, 50])])
    agreement, colour_score, n = corner_agreement(np.eye(4), src, tgt, CFG)
    assert n == 0 and agreement == 0.0 and colour_score == 0.0


def test_corners_disagreeing_with_a_second_detector_reject():
    """Corners still carry a rejection - they just need corroboration now,
    like every other detector."""
    v = Verdict(transform=np.eye(4), surface_fit=0.12)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.1, colour_agreement=0.9, n_considered=3, cfg=CFG,
                        corner_agreement=0.05, corner_colour_agreement=0.9,
                        n_corners=2)
    assert v.rejected_by, "corners plus planes disagreeing over real evidence must reject"
    assert v.vote_score < 0


def test_apply_detector_vote_corner_colour_alone_never_vetoes():
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.0, colour_agreement=0.0, n_considered=0, cfg=CFG,
                        corner_agreement=0.9, corner_colour_agreement=0.0,
                        n_corners=CFG.vote_min_corners)
    assert v.rejected_by == ""
    assert v.vote_score >= v.surface_fit


def test_two_detectors_that_both_agree_combine_to_full_agreement():
    """A weighted MEAN, not a sum: two detectors that both agree
    completely give complete agreement, not double it. A sum let the
    bonus grow just because more detectors happened to be in view."""
    v = Verdict(transform=np.eye(4), surface_fit=0.10)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=1.0, colour_agreement=1.0,
                        n_considered=4, cfg=CFG,
                        corner_agreement=1.0, corner_colour_agreement=1.0,
                        n_corners=2)
    assert abs(v.detector_score - 1.0) < 1e-9
    expected = 0.10 * (1.0 + 1.0 + CFG.vote_weight_colour)
    assert abs(v.vote_score - expected) < 1e-9


def test_a_weak_detector_pulls_the_verdict_down_without_sinking_it():
    """Between the two extremes: one detector unhappy, one happy, and the
    verdict lands between them weighted by how much each counts."""
    v = Verdict(transform=np.eye(4), surface_fit=0.10)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.2, colour_agreement=0.0,
                        n_considered=4, cfg=CFG,
                        corner_agreement=1.0, n_corners=2)
    blended = ((CFG.vote_weight_plane * 0.2 + CFG.vote_weight_corner * 1.0)
               / (CFG.vote_weight_plane + CFG.vote_weight_corner))
    assert abs(v.detector_score - blended) < 1e-9
    assert 0.2 < v.detector_score < 1.0
    assert v.rejected_by == "", "a blended verdict above the floor must not reject"


def test_edge_agreement_matches_the_same_junction_seen_at_different_lengths():
    """Two scans see different stretches of one wall-to-floor line, so
    their midpoints differ by metres. Same junction, must still match."""
    src = _features([], edges=[_edge(0, [0.0, 0.0, 0.0], [1, 0, 0], length_m=4.0)])
    tgt = _features([], edges=[_edge(0, [5.0, 0.0, 0.0], [1, 0, 0], length_m=12.0)])
    agree, _colour, n = edge_agreement(np.eye(4), src, tgt, CFG)
    assert n == 1, "the same line seen over different stretches must be one match"
    assert agree > 0.99


def test_edge_agreement_ignores_a_parallel_junction_across_the_room():
    """Every wall-to-floor line in a rectangular room is parallel to
    another one. Being parallel is not being the same junction."""
    src = _features([], edges=[_edge(0, [0.0, 0.0, 0.0], [1, 0, 0])])
    tgt = _features([], edges=[_edge(0, [0.0, 6.0, 0.0], [1, 0, 0])])
    agree, _colour, n = edge_agreement(np.eye(4), src, tgt, CFG)
    assert n == 0 and agree == 0.0


def test_edge_agreement_ignores_two_junctions_end_to_end_on_one_line():
    """Collinear but not overlapping - two different junctions that happen
    to share a line, not evidence of the same one."""
    src = _features([], edges=[_edge(0, [0.0, 0.0, 0.0], [1, 0, 0], length_m=2.0)])
    tgt = _features([], edges=[_edge(0, [20.0, 0.0, 0.0], [1, 0, 0], length_m=2.0)])
    _agree, _colour, n = edge_agreement(np.eye(4), src, tgt, CFG)
    assert n == 0


def test_edge_agreement_separates_a_wall_floor_from_a_vault_ridge():
    """Same line, same direction, different fold (90 degrees is a wall meeting the floor, 114 two roof slopes) is not
    the same feature. A near miss inside the gate must still match."""
    src = _features([], edges=[_edge(0, [0, 0, 0], [1, 0, 0], interior_angle_deg=90.0)])
    same = _features([], edges=[_edge(0, [0, 0, 0], [1, 0, 0], interior_angle_deg=90.0)])
    noisy = _features([], edges=[_edge(0, [0, 0, 0], [1, 0, 0], interior_angle_deg=97.0)])
    ridge = _features([], edges=[_edge(0, [0, 0, 0], [1, 0, 0], interior_angle_deg=114.0)])

    matched, _c, n_same = edge_agreement(np.eye(4), src, same, CFG)
    assert matched > 0.99 and n_same == 1

    # 7 degrees is the 90th percentile of real cross-scan fold noise.
    near, _c2, n_near = edge_agreement(np.eye(4), src, noisy, CFG)
    assert n_near == 1, "ordinary measurement noise must not lose a real match"
    assert near > 0.99, f"the gate is pass/fail, it must not also penalise: {near}"

    mismatched, _c3, n_ridge = edge_agreement(np.eye(4), src, ridge, CFG)
    assert n_ridge == 0, "a 24 degree fold difference is a different junction"
    assert mismatched == 0.0


def test_edge_agreement_weights_by_the_shorter_shared_length():
    """A 14 m junction is stronger evidence than a 0.6 m one, so a long
    agreeing edge must outweigh a short disagreeing one."""
    src = _features([], edges=[
        _edge(0, [0, 0, 0], [1, 0, 0], length_m=14.0),
        _edge(1, [0, 0, 9.0], [1, 0, 0], length_m=0.6),
    ])
    tgt = _features([], edges=[
        _edge(0, [0, 0, 0], [1, 0, 0], length_m=14.0),
        _edge(1, [0, 0.55, 9.0], [1, 0, 0], length_m=0.6),
    ])
    agree, _colour, n = edge_agreement(np.eye(4), src, tgt, CFG)
    assert n == 2
    assert agree > 0.9, f"the long clean match should dominate, got {agree}"


def test_apply_detector_vote_edges_add_a_bonus_and_never_veto():
    """Edges alone are one voice, and one voice never rejects - see
    test_one_detector_cannot_sink_a_candidate_on_its_own."""
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    apply_detector_vote(v, plane_agreement=0.0, colour_agreement=0.0, n_considered=0,
                        cfg=CFG, edge_agreement=1.0, edge_colour_agreement=0.0, n_edges=3)
    assert v.rejected_by == "", "an edge signal must not reject a candidate"
    assert v.vote_score > v.surface_fit
    assert v.n_edges_considered == 3

    worst = Verdict(transform=np.eye(4), surface_fit=0.08)
    worst.vote_score = worst.surface_fit
    apply_detector_vote(worst, plane_agreement=0.0, colour_agreement=0.0, n_considered=0,
                        cfg=CFG, edge_agreement=0.0, edge_colour_agreement=0.0, n_edges=3)
    assert worst.rejected_by == ""
    assert worst.vote_score < v.vote_score, "total edge disagreement must still rank lower"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
