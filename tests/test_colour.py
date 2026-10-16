"""Tests for colour.py: hue families, local colour composition, and the tolerant agreement scores.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.colour import (  # noqa: E402
    hue_family, composition, composition_similarity, hue_agreement,
)

CFG = PipelineConfig()


def test_hue_family_classifies_the_named_examples():
    # red ~0deg, blue ~230deg, a "tan" is a desaturated orange (~30-40deg)
    assert hue_family(5.0, 0.6, CFG) == "red"
    assert hue_family(230.0, 0.6, CFG) == "blue"
    assert hue_family(35.0, 0.6, CFG) == "orange"


def test_hue_family_wraps_across_the_0_360_seam():
    # red spans the wrap (350deg is still red, not "falls off the end")
    assert hue_family(350.0, 0.6, CFG) == "red"
    assert hue_family(0.5, 0.6, CFG) == "red"


def test_hue_family_below_saturation_floor_is_neutral():
    # C2: a grey/tan-adjacent surface's hue is noise below the floor,
    # regardless of what that noisy hue number happens to be.
    assert hue_family(200.0, 0.02, CFG) == "neutral"
    assert hue_family(float("nan"), 0.0, CFG) == "neutral"


def test_composition_weights_by_area_not_by_count():
    fams = ["red", "red", "blue"]
    weights = [1.0, 1.0, 8.0]  # two small red surfaces, one big blue one
    comp = composition(fams, weights)
    assert abs(comp["blue"] - 0.8) < 1e-9
    assert abs(comp["red"] - 0.2) < 1e-9


def test_composition_empty_input_is_empty_not_an_error():
    assert composition([], []) == {}


def test_composition_similarity_identical_is_1():
    a = {"red": 0.5, "tan": 0.5}
    assert composition_similarity(a, dict(a)) == 1.0


def test_composition_similarity_disjoint_is_0():
    assert composition_similarity({"red": 1.0}, {"blue": 1.0}) == 0.0


def test_composition_similarity_partial_overlap():
    a = {"red": 0.6, "blue": 0.4}
    b = {"red": 0.3, "blue": 0.3, "green": 0.4}
    # histogram intersection: min(0.6,0.3) + min(0.4,0.3) + min(0,0.4)
    assert abs(composition_similarity(a, b) - 0.6) < 1e-9


def test_composition_similarity_both_empty_agrees_not_fails():
    # Nothing to disagree about - two entirely neutral neighbourhoods.
    assert composition_similarity({}, {}) == 1.0


def test_composition_similarity_one_empty_disagrees():
    assert composition_similarity({}, {"red": 1.0}) == 0.0


def test_hue_agreement_same_hue_is_1():
    assert hue_agreement(20.0, 0.5, 20.0, 0.5, CFG) == 1.0


def test_hue_agreement_two_neutrals_agree():
    assert hue_agreement(np.nan, 0.02, 123.0, 0.01, CFG) == 1.0


def test_hue_agreement_neutral_against_coloured_disagrees():
    assert hue_agreement(np.nan, 0.02, 20.0, 0.6, CFG) == 0.0


def test_hue_agreement_far_apart_is_0():
    assert hue_agreement(10.0, 0.6, 190.0, 0.6, CFG) == 0.0


def test_hue_agreement_near_the_tolerance_edge_is_partial_not_a_cliff():
    half_tol = CFG.colour_hue_tolerance_deg / 2.0
    score = hue_agreement(10.0, 0.6, 10.0 + half_tol, 0.6, CFG)
    assert 0.3 < score < 0.7, f"expected a partial score near the tolerance midpoint, got {score}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
