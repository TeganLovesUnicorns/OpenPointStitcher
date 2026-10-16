"""Tests for align_graph._detector_weighted_information: scaling an accepted edge's trust in the global solve by
how strongly the detectors agreed on it. Off by default, and never touching an edge the vote did not score."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.align_graph import _detector_weighted_information  # noqa: E402
from open_point_stitch.align_pair import PairResult  # noqa: E402
from open_point_stitch.config import PipelineConfig  # noqa: E402

INFO = np.eye(6) * 4.0


def _result(vote_detector_score=None, n_planes=0, n_corners=0, n_edges=0, n_openings=0, n_landmarks=0):
    return PairResult(
        source=0, target=1, source_name="a", target_name="b",
        transform=np.eye(4), information=INFO.copy(),
        vote_detector_score=vote_detector_score,
        vote_n_planes_considered=n_planes, vote_n_corners_considered=n_corners,
        vote_n_edges_considered=n_edges, vote_n_openings_considered=n_openings,
        vote_n_landmarks_considered=n_landmarks,
    )


def test_off_by_default_leaves_every_edge_untouched():
    cfg = PipelineConfig()
    assert cfg.posegraph_detector_weighting_enabled is False
    r = _result(vote_detector_score=0.1, n_planes=5)
    out = _detector_weighted_information(r.information, r, cfg)
    assert np.array_equal(out, INFO), "off by default must be a true no-op, not a 1.0 multiplier applied"


def test_no_vote_ran_on_this_pair_leaves_it_untouched_even_when_enabled():
    cfg = PipelineConfig(posegraph_detector_weighting_enabled=True)
    r = _result(vote_detector_score=None)   # the vote never ran (e.g. detector_vote_enabled was off)
    out = _detector_weighted_information(r.information, r, cfg)
    assert np.array_equal(out, INFO)


def test_too_little_evidence_leaves_it_untouched_even_when_enabled():
    cfg = PipelineConfig(posegraph_detector_weighting_enabled=True, posegraph_detector_weight_min_evidence=3)
    r = _result(vote_detector_score=0.0, n_planes=1, n_corners=1)   # 2 total, under the 3 required
    out = _detector_weighted_information(r.information, r, cfg)
    assert np.array_equal(out, INFO)


def test_strong_agreement_leaves_the_information_close_to_unscaled():
    cfg = PipelineConfig(posegraph_detector_weighting_enabled=True, posegraph_detector_weight_floor=0.2)
    r = _result(vote_detector_score=1.0, n_planes=5)
    out = _detector_weighted_information(r.information, r, cfg)
    assert np.allclose(out, INFO), out


def test_weak_agreement_scales_toward_the_floor_not_to_zero():
    cfg = PipelineConfig(posegraph_detector_weighting_enabled=True, posegraph_detector_weight_floor=0.2)
    r = _result(vote_detector_score=0.0, n_planes=5)
    out = _detector_weighted_information(r.information, r, cfg)
    assert np.allclose(out, INFO * 0.2), out
    assert not np.allclose(out, 0.0), "a strongly-disagreeing edge must be downweighted, never zeroed by this alone"


def test_partial_agreement_scales_linearly_between_the_floor_and_one():
    cfg = PipelineConfig(posegraph_detector_weighting_enabled=True, posegraph_detector_weight_floor=0.2)
    r = _result(vote_detector_score=0.5, n_planes=5)
    out = _detector_weighted_information(r.information, r, cfg)
    assert np.allclose(out, INFO * 0.6), out   # 0.2 + 0.8*0.5 = 0.6


def test_evidence_is_summed_across_every_detector_kind():
    cfg = PipelineConfig(posegraph_detector_weighting_enabled=True, posegraph_detector_weight_min_evidence=3)
    r = _result(vote_detector_score=0.0, n_planes=1, n_edges=1, n_landmarks=1)   # 1 each, 3 total - just clears it
    out = _detector_weighted_information(r.information, r, cfg)
    assert not np.array_equal(out, INFO), "3 total across different detectors should count as enough evidence"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc!r}")
    raise SystemExit(1 if failures else 0)
