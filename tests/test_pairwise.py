"""Tests for register_pair: its wire format (strict JSON, no bare NaN), seeded behaviour, the detector vote and
progress reporting."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.align_pair import achievable_overlap_frac, register_pair  # noqa: E402
from open_point_stitch.find_surfaces import extract_scan_features  # noqa: E402
from open_point_stitch.prepare import preprocess_scan  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402


def _room_points(rng, n=2000):
    floor = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), np.zeros(n)])
    wall1 = np.column_stack([np.full(n // 2, -3.0), rng.uniform(-3, 3, n // 2),
                             rng.uniform(0, 2.5, n // 2)])
    wall2 = np.column_stack([rng.uniform(-3, 3, n // 2), np.full(n // 2, -3.0),
                             rng.uniform(0, 2.5, n // 2)])
    return np.vstack([floor, wall1, wall2])


def _rejects_non_strict_json(text: str) -> bool:
    """True if text only parses because of Python's NaN/Infinity extension to json.loads, i.e. what a browser's
    JSON.parse would refuse. Raising from parse_constant makes the parser strict."""
    def _reject(tok):
        raise ValueError(f"non-finite JSON constant: {tok}")
    try:
        json.loads(text, parse_constant=_reject)
        return False
    except ValueError:
        return True


def test_local_only_result_has_no_nan_in_its_wire_format():
    rng = np.random.default_rng(7)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0]))

    cfg = PipelineConfig()
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    # A seeded (local_only) run never runs a coarse stage.
    result = register_pair(prep_a, prep_b, cfg, init=np.eye(4),
                           seeded_from="manual", local_only=True)
    row = result.as_row()

    assert row["coarse_fitness"] is None, "no coarse stage ran - NaN must serialise as null, not NaN"
    assert row["coarse_rmse_m"] is None

    # The server's transport (json.dumps(..., allow_nan=False)) must not raise, and its output must be strict JSON.
    text = json.dumps(row, allow_nan=False)
    assert "NaN" not in text
    assert not _rejects_non_strict_json(text), "response is not strict JSON - a real browser would refuse it"


def test_unseeded_result_still_carries_a_real_coarse_fitness():
    """When a coarse stage did run (unseeded), coarse_fitness/coarse_rmse are real numbers, not nulled by the NaN
    guard."""
    rng = np.random.default_rng(11)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.3, -0.2, 0.0]))

    cfg = PipelineConfig(coarse_method="ransac")
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    result = register_pair(prep_a, prep_b, cfg)  # no init: coarse RANSAC runs
    row = result.as_row()
    assert row["coarse_fitness"] is not None
    assert isinstance(row["coarse_fitness"], float)


def test_achievable_overlap_frac_high_for_coincident_low_for_far_apart():
    """The fitness ceiling is near 1.0 when the scans coincide at transform and near 0.0 when they sit far apart."""
    rng = np.random.default_rng(3)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b_here = Scan(name="B", path="B.pts", points=local.copy())
    b_far = Scan(name="B", path="B.pts", points=local + np.array([500.0, 500.0, 0.0]))

    cfg = PipelineConfig()
    radius = cfg.voxel_size * cfg.icp_distance_mult
    assert achievable_overlap_frac(a, b_here, np.eye(4), radius, cfg) > 0.9
    assert achievable_overlap_frac(a, b_far, np.eye(4), radius, cfg) < 0.01


def test_relative_fitness_gate_off_by_default_leaves_achievable_none():
    rng = np.random.default_rng(13)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0]))

    cfg = PipelineConfig()  # relative_fitness_gate_enabled defaults False
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    result = register_pair(prep_a, prep_b, cfg, init=np.eye(4), seeded_from="manual",
                           local_only=True, src_scan=a, tgt_scan=b)
    row = result.as_row()
    assert row["achievable_frac"] is None
    assert row["fitness_over_achievable"] is None


def test_relative_fitness_gate_when_enabled_measures_achievable():
    rng = np.random.default_rng(17)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0]))

    cfg = PipelineConfig(relative_fitness_gate_enabled=True, relative_fitness_min_ratio=0.0)
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    result = register_pair(prep_a, prep_b, cfg, init=np.eye(4), seeded_from="manual",
                           local_only=True, src_scan=a, tgt_scan=b)
    row = result.as_row()
    assert row["achievable_frac"] is not None
    assert 0.0 <= row["achievable_frac"] <= 1.0
    assert row["fitness_over_achievable"] is not None


def test_vote_diagnostics_stay_none_when_the_vote_never_ran():
    """PairResult.vote_* default to None, not 0, when the vote never ran: None means "did not run", 0 would mean
    "ran and found nothing"."""
    rng = np.random.default_rng(19)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0]))
    cfg = PipelineConfig(detector_vote_enabled=False)
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    result = register_pair(prep_a, prep_b, cfg, init=np.eye(4), seeded_from="manual", local_only=True)
    row = result.as_row()
    assert row["vote_n_planes_considered"] is None
    assert row["vote_n_corners_considered"] is None
    assert row["vote_n_openings_considered"] is None and row["vote_opening_agreement"] is None
    assert row["vote_score"] is None
    assert row["vote_rejected_count"] == 0


def test_vote_diagnostics_stay_none_without_scan_features_even_if_enabled():
    """detector_vote_enabled=True but no ScanFeatures supplied: the vote must not run on empty input."""
    rng = np.random.default_rng(23)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0]))
    cfg = PipelineConfig(detector_vote_enabled=True)
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    result = register_pair(prep_a, prep_b, cfg, init=np.eye(4), seeded_from="manual", local_only=True)
    row = result.as_row()
    assert row["vote_n_planes_considered"] is None, (
        "the vote ran with no features supplied - use_vote's own guard did not hold"
    )


def test_vote_diagnostics_populate_when_the_vote_actually_runs():
    """The vote fields prove it ran and had real surfaces to compare, on a room built to give it three."""
    rng = np.random.default_rng(29)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0]))
    cfg = PipelineConfig(detector_vote_enabled=True)
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    a_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a.points))
    b_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b.points))
    feat_a = extract_scan_features(a_pcd)
    feat_b = extract_scan_features(b_pcd)
    assert len(feat_a.surfaces) >= 3 and len(feat_b.surfaces) >= 3, (
        "setup: expected the floor+2-wall room to yield real surfaces"
    )

    result = register_pair(prep_a, prep_b, cfg, init=np.eye(4), seeded_from="manual",
                           local_only=True, src_features=feat_a, tgt_features=feat_b)
    row = result.as_row()
    assert row["vote_n_planes_considered"] is not None, "the vote ran but left no diagnostics behind"
    assert row["vote_n_planes_considered"] > 0, (
        f"real overlapping surfaces existed but n_considered came back "
        f"{row['vote_n_planes_considered']}"
    )
    assert row["vote_score"] is not None


def test_the_opening_voice_runs_through_register_pair_and_leaves_its_numbers():
    """A door present in both scans' features is compared, counted and reported."""
    from open_point_stitch.find_openings import Opening
    rng = np.random.default_rng(29)
    local = _room_points(rng)
    shift = np.array([0.2, 0.1, 0.0])
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + shift)
    cfg = PipelineConfig(detector_vote_enabled=True)
    prep_a, prep_b = preprocess_scan(a, 0, cfg), preprocess_scan(b, 1, cfg)
    feat_a = extract_scan_features(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a.points)))
    feat_b = extract_scan_features(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b.points)))

    def door(position):
        return Opening(kind="door", wall_index=0, along=0.0, width=0.9, height=2.1, sill=0.0, centre=np.array(position),
                       normal=np.array([1.0, 0.0]), clear_fraction=1.0, bounded=1.0, position=np.array(position),
                       direction=np.array([0.0, 1.0, 0.0]), wall_normal=np.array([1.0, 0.0, 0.0]), cohesion=1.0)
    feat_a.openings = [door([0.0, 1.0, 1.0])]
    feat_b.openings = [door(list(np.array([0.0, 1.0, 1.0]) + shift))]

    row = register_pair(prep_a, prep_b, cfg, init=np.eye(4), seeded_from="manual", local_only=True,
                        src_features=feat_a, tgt_features=feat_b).as_row()
    assert row["vote_n_openings_considered"] >= 1, row["vote_n_openings_considered"]
    assert row["vote_opening_agreement"] is not None and row["vote_opening_agreement"] > 0.0


def test_a_good_placement_is_returned_untouched_not_fiddled_with():
    """A pair that is already right comes back exactly as handed in, not refined by a centimetre."""
    rng = np.random.default_rng(11)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local.copy())   # already perfectly aligned

    cfg = PipelineConfig()
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)
    init = np.eye(4)

    result = register_pair(prep_a, prep_b, cfg, init=init,
                           seeded_from="manual", local_only=True)

    assert result.accepted, "keeping a good placement is a success, not a rejection"
    assert result.declined_no_improvement, "nothing can beat an already-correct placement"
    assert result.method == "kept_placement"
    assert np.allclose(result.transform, init, atol=0.0), (
        "declining must hand back the placement byte for byte, not a refined "
        f"version of it - got {result.transform}"
    )
    row = result.as_row()
    assert row["declined_no_improvement"] is True
    assert row["seed_score"] is not None and row["best_candidate_score"] is not None


def test_a_placement_that_is_really_wrong_still_gets_corrected():
    """The other half: declining must not become never moving. A seed
    genuinely off by a quarter of a metre has to be improved on."""
    rng = np.random.default_rng(13)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local.copy())

    cfg = PipelineConfig()
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)
    wrong = np.eye(4)
    wrong[0, 3] = 0.25   # a quarter metre out, well beyond the detectors' noise

    result = register_pair(prep_a, prep_b, cfg, init=wrong,
                           seeded_from="manual", local_only=True)

    assert result.accepted
    assert not result.declined_no_improvement, "a 25cm error is worth correcting"
    moved = float(np.linalg.norm(result.transform[:3, 3] - wrong[:3, 3]))
    assert moved > 0.05, f"expected a real correction, moved only {moved:.3f}m"


def test_detector_voting_is_on_by_default():
    """A job that does not say gets the vote. The server reads PipelineConfig's default, so this is where it lives."""
    assert PipelineConfig().detector_vote_enabled is True
    assert PipelineConfig.detector_vote_enabled is True   # what server.py reads when a job says nothing
    assert PipelineConfig(detector_vote_enabled=False).detector_vote_enabled is False   # still switchable off


def test_a_refined_seed_that_fails_the_gate_is_labelled_as_refined_not_kept():
    """When the winning local candidate fails the fitness gate, the result is the ICP-refined seed, not the
    placement, and it is labelled as refined, with by how much."""
    rng = np.random.default_rng(13)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local.copy())

    cfg = PipelineConfig(min_fitness=1.01)   # fitness cannot exceed 1.0, so the gate always rejects
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)
    wrong = np.eye(4)
    wrong[0, 3] = 0.25   # off enough that a candidate beats the placement, so it is the GATE that rejects it

    result = register_pair(prep_a, prep_b, cfg, init=wrong, seeded_from="manual", local_only=True)

    moved = float(np.linalg.norm(result.transform[:3, 3] - wrong[:3, 3]))
    assert result.accepted and not result.declined_no_improvement
    assert moved > 0.05, f"expected the refined seed to have moved, moved only {moved:.3f} m"
    assert result.method == "refined_seed", f"a moved pair must not be labelled kept: {result.method!r}"
    assert "refined by" in result.reason and "failed a quality check" in result.reason, result.reason
    assert f"{moved:.2f} m" in result.reason, f"the reason should state the distance {moved:.2f} m: {result.reason!r}"
    # The same in plain words for the links table, naming the failed check.
    assert result.why.startswith("Kept near where you put it, fine-tuned by"), result.why
    assert "of A's points landed on a surface of B" in result.why and "fitness" not in result.why, result.why


def test_register_pair_reports_ransac_and_refining_substages():
    """progress_cb is told the real stages as they happen, not just called once at the end."""
    rng = np.random.default_rng(5)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.3, -0.2, 0.0]))

    cfg = PipelineConfig(coarse_method="ransac")
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    stages = []
    register_pair(prep_a, prep_b, cfg, progress_cb=stages.append)  # no init: coarse RANSAC runs

    assert "RANSAC" in stages, stages
    assert any(s.startswith("refining") for s in stages), stages
    # RANSAC has to run before any candidate can be refined.
    assert stages.index("RANSAC") < next(i for i, s in enumerate(stages) if s.startswith("refining")), stages


def test_register_pair_reports_detector_substages_when_vote_runs():
    """Progress names the detector stages: corners, edges, planes."""
    rng = np.random.default_rng(29)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0]))
    cfg = PipelineConfig(detector_vote_enabled=True)
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    feat_a = extract_scan_features(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a.points)))
    feat_b = extract_scan_features(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b.points)))
    assert len(feat_a.surfaces) >= 3, "setup: expected the floor+2-wall room to yield real surfaces"

    stages = []
    register_pair(prep_a, prep_b, cfg, init=np.eye(4), seeded_from="manual", local_only=True,
                  src_features=feat_a, tgt_features=feat_b, progress_cb=stages.append)

    for expected in ("matching planes", "matching corners", "matching edges",
                     "matching doors and windows", "matching landmarks"):
        assert expected in stages, f"{expected!r} never reported: {stages}"


def test_run_pairwise_folds_the_substage_into_the_same_progress_cb():
    """pipeline.run_pairwise uses the server's (done, total, desc) progress shape: desc gains a ": {stage}" suffix
    while a pair is in flight, and done does not count the pair until it finishes."""
    from open_point_stitch.pipeline import run_pairwise

    rng = np.random.default_rng(5)
    local = _room_points(rng)
    a = Scan(name="A", path="A.pts", points=local)
    b = Scan(name="B", path="B.pts", points=local + np.array([0.3, -0.2, 0.0]))
    cfg = PipelineConfig(coarse_method="ransac", concurrency=1)
    prep_a = preprocess_scan(a, 0, cfg)
    prep_b = preprocess_scan(b, 1, cfg)

    calls = []
    run_pairwise([prep_a, prep_b], [(0, 1)], cfg, progress_cb=lambda d, t, desc: calls.append((d, t, desc)))

    substage_calls = [c for c in calls if ": RANSAC" in c[2]]
    assert substage_calls, f"no RANSAC sub-stage call seen: {calls}"
    d, t, desc = substage_calls[0]
    assert d == 0 and t == 1, "the in-flight pair must not count as done yet"
    assert desc.startswith("A <-> B: RANSAC")

    completion_calls = [c for c in calls if c[2] == "A <-> B"]
    assert completion_calls and completion_calls[-1][0] == 1, "the final call must be the plain, un-suffixed desc"


def test_a_pair_that_kept_the_placement_with_nothing_to_measure_still_serialises():
    """A kept placement with no nearby points has rmse = infinity, which must not reach the wire as Infinity."""
    from open_point_stitch.align_pair import PairResult
    res = PairResult(source=0, target=4, source_name="a", target_name="b", transform=np.eye(4), information=np.eye(6),
                     fitness=0.0, rmse=float("inf"), method="kept_placement", accepted=True)
    row = res.as_row()
    assert row["rmse_m"] is None and row["fitness"] == 0.0, (row["rmse_m"], row["fitness"])
    text = json.dumps(row, allow_nan=False)
    assert not _rejects_non_strict_json(text)


def test_thick_surfaces_read_thick_and_are_trusted_less_in_the_pose_graph():
    """The same room, once crisp and once with 3 cm of scatter."""
    from open_point_stitch.align_pair import _information_matrix, surface_thickness
    rng = np.random.default_rng(11)
    crisp = _room_points(rng, n=6000)
    fuzzy = crisp + rng.normal(0, 0.03, crisp.shape)
    cfg_on = PipelineConfig(thickness_trust=True)
    preps = [preprocess_scan(Scan(name=n, path=f"{n}.pts", points=p), i, cfg_on)
             for i, (n, p) in enumerate([("crisp", crisp), ("crisp2", crisp + 0.001), ("fuzzy", fuzzy)])]
    thin, thick = (float(np.median(surface_thickness(p, cfg_on))) for p in (preps[0], preps[2]))
    assert thick > 2 * thin, (thin, thick)
    off = _information_matrix(preps[0], preps[2], np.eye(4), PipelineConfig())
    on = _information_matrix(preps[0], preps[2], np.eye(4), cfg_on)
    assert np.trace(on) < 0.5 * np.trace(off), (np.trace(on), np.trace(off))
    assert np.allclose(_information_matrix(preps[0], preps[1], np.eye(4), PipelineConfig()),
                       _information_matrix(preps[0], preps[1], np.eye(4), PipelineConfig())), "off must not change"


def test_seeded_refinement_with_thickness_trust_still_recovers_a_known_shift():
    rng = np.random.default_rng(12)
    local = _room_points(rng, n=6000)
    cfg = PipelineConfig(thickness_trust=True)
    a = preprocess_scan(Scan(name="A", path="A.pts", points=local), 0, cfg)
    b = preprocess_scan(Scan(name="B", path="B.pts", points=local + np.array([0.15, -0.1, 0.0])), 1, cfg)
    from open_point_stitch.align_pair import fine_register_ladder
    got = np.asarray(fine_register_ladder(b, a, np.eye(4), cfg).transformation)
    assert np.allclose(got[:3, 3], [-0.15, 0.1, 0.0], atol=0.02), got[:3, 3]


def test_every_pair_result_says_how_sure_it_is_in_plain_words():
    """Every pair reports confidence and confidence_why."""
    rng = np.random.default_rng(14)
    local = _room_points(rng)
    cfg = PipelineConfig()
    a = preprocess_scan(Scan(name="A", path="A.pts", points=local), 0, cfg)
    b = preprocess_scan(Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0])), 1, cfg)
    row = register_pair(a, b, cfg, init=np.eye(4), seeded_from="manual", local_only=True).as_row()
    assert row["confidence"] in ("clear", "workable", "ambiguous", "unchecked", "none"), row["confidence"]
    assert row["confidence_why"] and "—" not in row["confidence_why"], row["confidence_why"]


def test_candidate_recording_is_off_unless_asked_and_writes_every_clue_when_on():
    """OPENPOINT_DUMP_CANDIDATES writes one JSON line per candidate."""
    import tempfile
    rng = np.random.default_rng(13)
    local = _room_points(rng)
    cfg = PipelineConfig()
    a = preprocess_scan(Scan(name="A", path="A.pts", points=local), 0, cfg)
    b = preprocess_scan(Scan(name="B", path="B.pts", points=local + np.array([0.2, 0.1, 0.0])), 1, cfg)
    os.environ.pop("OPENPOINT_DUMP_CANDIDATES", None)
    register_pair(a, b, cfg, init=np.eye(4), seeded_from="manual", local_only=True)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.jsonl")
        os.environ["OPENPOINT_DUMP_CANDIDATES"] = path
        try:
            register_pair(a, b, cfg, init=np.eye(4), seeded_from="manual", local_only=True)
        finally:
            os.environ.pop("OPENPOINT_DUMP_CANDIDATES", None)
        rows = [json.loads(line) for line in open(path)]
    assert rows and all({"T", "src_centre", "surface_fit", "detector_score", "origin"} <= set(r) for r in rows), rows[:1]
    assert rows[0]["src"] == "A" and rows[0]["tgt"] == "B"


def test_an_outside_proposal_near_the_placement_can_pull_a_rough_seed_in_and_a_far_one_is_ignored():
    """Outside proposals join a seeded pair's candidates."""
    rng = np.random.default_rng(15)
    local = _room_points(rng, n=4000)
    shift = np.array([0.9, -0.6, 0.0])                    # 1.08 m: beyond the 0.5 m local bound
    cfg = PipelineConfig()
    a = preprocess_scan(Scan(name="A", path="A.pts", points=local), 0, cfg)
    b = preprocess_scan(Scan(name="B", path="B.pts", points=local + shift), 1, cfg)
    near = np.eye(4)
    near[:3, 3] = shift + [0.04, -0.03, 0.0]
    without = register_pair(a, b, cfg, init=np.eye(4), seeded_from="manual", local_only=True)
    assert np.linalg.norm(without.transform[:3, 3] - shift) > 0.3, without.transform[:3, 3]
    with_prop = register_pair(a, b, PipelineConfig(proposals={"B|A": np.linalg.inv(near).tolist()}),
                              init=np.eye(4), seeded_from="manual", local_only=True)
    assert np.allclose(with_prop.transform[:3, 3], shift, atol=0.02), with_prop.transform[:3, 3]
    assert with_prop.seeded_from == "proposer", with_prop.seeded_from
    far = np.eye(4)
    far[:3, 3] = [3.0, 0.0, 0.0]
    gated = register_pair(a, b, PipelineConfig(proposals={"A|B": far.tolist()}),
                          init=np.eye(4), seeded_from="manual", local_only=True)
    assert gated.seeded_from != "proposer" and np.linalg.norm(gated.transform[:3, 3]) < 0.6, gated.transform[:3, 3]


def _two_rooms(seed: int, shift):
    rng = np.random.default_rng(seed)
    local = _room_points(rng, n=4000)
    cfg = PipelineConfig(thickness_trust=True)
    a = preprocess_scan(Scan(name=f"A{seed}", path="A.pts", points=local), 0, cfg)
    b = preprocess_scan(Scan(name=f"B{seed}", path="B.pts", points=local + np.asarray(shift)), 1, cfg)
    return a, b, cfg


def test_midair_weight_is_full_on_a_surface_and_low_for_a_lone_point():
    """Mid-air trust weights."""
    rng = np.random.default_rng(16)
    pts = np.vstack([_room_points(rng, n=4000), [[0.0, 0.0, 1.5]]])          # one point floating mid-room
    cfg = PipelineConfig()
    p = preprocess_scan(Scan(name="M16", path="M.pts", points=pts), 0, cfg)
    from open_point_stitch.align_pair import MIDAIR_MIN_W, midair_weight
    w = midair_weight(p, cfg)
    lone = int(np.argmin(np.linalg.norm(np.asarray(p.down.points) - [0.0, 0.0, 1.5], axis=1)))
    assert w[lone] == MIDAIR_MIN_W, w[lone]
    assert np.median(w) > 0.8, np.median(w)              # this synthetic room is sparser than a real scan


def test_weighted_steps_match_open3d_when_every_weight_is_one():
    """The mid-air path differs from the plain one only by its weights."""
    import dataclasses
    from open_point_stitch import align_pair as ap
    a, b, cfg = _two_rooms(17, [0.12, -0.08, 0.0])
    for p in (a, b):
        ap._MIDAIR[(p.name, len(p.down.points))] = np.ones(len(p.down.points))
    dist = cfg.voxel_size * 1.5
    ours = ap._weighted_information(a, b, dist, np.eye(4), cfg)
    theirs = np.asarray(o3d.pipelines.registration.get_information_matrix_from_point_clouds(a.down, b.down, dist, np.eye(4)))
    assert np.allclose(ours, theirs), (ours[0], theirs[0])
    plain = np.asarray(ap.fine_register_ladder(a, b, np.eye(4), cfg).transformation)
    weighted = np.asarray(ap.fine_register_ladder(a, b, np.eye(4), dataclasses.replace(cfg, midair_trust=True)).transformation)
    assert np.linalg.norm(plain[:3, 3] - weighted[:3, 3]) < 0.002, (plain[:3, 3], weighted[:3, 3])


def test_learned_trust_is_per_patch_and_scales_the_judge():
    """Learned trust weights."""
    import dataclasses
    from open_point_stitch import point_trust as pt
    from open_point_stitch import score
    a, b, cfg = _two_rooms(19, [0.0, 0.0, 0.0])
    pid, table = pt.patch_table(pt.point_features(a, cfg), np.asarray(a.down.points))
    assert table.shape[1] == len(pt.FEATURES) and pid.max() == len(table) - 1
    keys = np.floor(np.asarray(a.down.points) / pt.PATCH_M).astype(int)
    same = pid[:50, None] == pid[None, :50]
    assert np.array_equal(same, (keys[:50, None] == keys[None, :50]).all(-1) & (pid[:50, None] >= 0)), "one patch per cube"
    pt._MODEL = {"features": list(pt.FEATURES), "coef": [np.log(0.02)] + [0.0] * len(pt.FEATURES), "floor_m": 0.01}
    pt._LEARNED.clear()
    try:
        w = pt.learned_weight(a, dataclasses.replace(cfg, learned_trust=True))
        assert np.allclose(w[pid >= 0], 0.25) and np.all(w[pid < 0] == pt.MIDAIR_MIN_W), np.unique(w)
        nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.asarray(b.down.points)))
        nns.knn_index()
        plain = score.score_candidate_fast(a.down, b.down, nns, np.eye(4), cfg)
        ones = np.ones(len(a.down.points)), np.ones(len(b.down.points))
        same_w = score.score_candidate_fast(a.down, b.down, nns, np.eye(4), cfg, src_w=ones[0], tgt_w=ones[1])
        half = score.score_candidate_fast(a.down, b.down, nns, np.eye(4), cfg, src_w=ones[0], tgt_w=ones[1] * 0.5)
        assert abs(plain.surface_fit - same_w.surface_fit) < 1e-12 and abs(half.surface_fit - 0.5 * plain.surface_fit) < 1e-12
    finally:
        pt._MODEL = None
        pt._LEARNED.clear()


def test_the_seed_judge_ranks_candidates_and_decides_whether_to_move():
    """A judge that likes proposals moves to one; a judge that dislikes them keeps the placement."""
    import dataclasses
    from open_point_stitch import seed_judge as sj
    a, b, cfg = _two_rooms(20, [0.9, -0.6, 0.0])
    near = np.eye(4)
    near[:3, 3] = [0.94, -0.63, 0.0]
    cfg = dataclasses.replace(cfg, proposals={"A20|B20": near.tolist()}, seed_judge=True)
    k = sj.FEATURES.index("is_proposer")
    try:
        for weight, expect_proposer in ((10.0, True), (-10.0, False)):
            w = np.zeros(len(sj.FEATURES))
            w[k] = weight
            sj._MODEL = {"features": sj.FEATURES, "mean": [0.0] * len(w), "std": [1.0] * len(w), "w": w.tolist(), "b": 0.0}
            res = register_pair(a, b, cfg, init=np.eye(4), seeded_from="manual", local_only=True)
            moved = np.allclose(res.transform[:3, 3], [0.9, -0.6, 0.0], atol=0.02)
            assert moved == expect_proposer and (res.seeded_from == "proposer") == expect_proposer, (weight, res.seeded_from)
    finally:
        sj._MODEL = None


def test_midair_trust_still_recovers_a_known_shift():
    import dataclasses
    from open_point_stitch.align_pair import fine_register_ladder
    a, b, cfg = _two_rooms(18, [0.15, -0.1, 0.0])
    got = np.asarray(fine_register_ladder(a, b, np.eye(4), dataclasses.replace(cfg, midair_trust=True)).transformation)
    assert np.allclose(got[:3, 3], [0.15, -0.1, 0.0], atol=0.02), got[:3, 3]


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
