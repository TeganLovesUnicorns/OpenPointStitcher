"""Tests for find_landmarks.py: clustering the points no plane explains into things, matching things between scans
by their separations, the end-to-end candidate from two synthetic scans, and that the option is off by default."""
from __future__ import annotations

import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import align_pair  # noqa: E402
from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.find_landmarks import (  # noqa: E402
    Landmark, find_landmarks, landmark_candidates, match_landmarks,
    build_scan_landmarks, _organic_fraction,
)
from open_point_stitch.find_surfaces import ScanFeatures  # noqa: E402
from open_point_stitch.prepare import preprocess_scan  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402
from open_point_stitch.score import Verdict, apply_detector_vote, landmark_agreement  # noqa: E402

CFG = PipelineConfig()


def _blob(rng, centre, spread, n, h0, h1):
    xy = rng.normal(centre, spread, size=(n, 2))
    return xy, rng.uniform(h0, h1, n)


def test_separate_things_become_separate_landmarks_and_noise_is_ignored():
    rng = np.random.default_rng(1)
    parts = [_blob(rng, (2.0, 3.0), 0.12, 300, 0.2, 1.4), _blob(rng, (6.0, 1.0), 0.25, 500, 0.3, 1.0),
             _blob(rng, (9.0, 7.0), 0.10, 200, 0.2, 2.0)]
    lone = (rng.uniform(0, 10, (30, 2)), rng.uniform(0.3, 1.5, 30))                    # scattered single points
    xy = np.vstack([p[0] for p in parts] + [lone[0]])
    h = np.concatenate([p[1] for p in parts] + [lone[1]])
    found = find_landmarks(xy, h)
    assert len(found) == 3, [lm.xy.round(1).tolist() for lm in found]
    for centre in ((2.0, 3.0), (6.0, 1.0), (9.0, 7.0)):
        assert min(np.linalg.norm(lm.xy - centre) for lm in found) < 0.15, centre
    tallest = max(found, key=lambda lm: lm.top)
    assert abs(tallest.top - 2.0) < 0.15 and tallest.base < 0.4


def test_things_off_the_floor_range_or_far_too_big_are_not_landmarks():
    rng = np.random.default_rng(2)
    floor_bits = _blob(rng, (1.0, 1.0), 0.2, 400, 0.0, 0.05)                           # on the floor: not a thing
    ceiling_bits = _blob(rng, (4.0, 4.0), 0.2, 400, 2.9, 3.4)                          # up at the ceiling
    long_run = (np.column_stack([np.linspace(0, 12, 900), rng.normal(0, 0.05, 900)]), rng.uniform(0.3, 1.0, 900))
    xy = np.vstack([floor_bits[0], ceiling_bits[0], long_run[0]])
    h = np.concatenate([floor_bits[1], ceiling_bits[1], long_run[1]])
    assert find_landmarks(xy, h) == []


def _lm(x, y, top=1.2, base=0.1, length=0.8, width=0.5):
    return Landmark(xy=np.array([x, y], dtype=float), h=(top + base) / 2, length=length, width=width, top=top, base=base,
                    n_points=100, brightness=0.0, weight=float(np.log1p(100)))


def _arrangement():
    """Eight things with a lopsided layout, so it looks different turned round and in a mirror."""
    pos = [(1.0, 1.0), (3.0, 2.6), (6.6, 0.4), (1.8, 5.7), (5.0, 4.1), (9.2, 3.4), (6.1, 8.3), (0.6, 9.2)]
    sizes = [(0.8, 0.5, 1.2), (1.5, 0.6, 0.9), (0.6, 0.6, 1.6), (1.0, 0.7, 1.0), (2.0, 0.8, 1.1), (0.7, 0.5, 1.4), (1.2, 0.9, 0.8), (0.9, 0.6, 1.3)]
    return [_lm(x, y, top=t, length=l, width=w) for (x, y), (l, w, t) in zip(pos, sizes)]


def _moved(lms, yaw_deg, shift, noise=0.05, seed=3):
    rng = np.random.default_rng(seed)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    rot = np.array([[c, -s], [s, c]])
    return [Landmark(**{**lm.__dict__, "xy": rot @ lm.xy + shift + rng.normal(0, noise, 2)}) for lm in lms]


def test_a_known_pose_is_recovered_with_noise_extra_things_and_missing_things():
    src = _arrangement()
    tgt = _moved(src[:6], 37.0, np.array([4.0, -3.0])) + [_lm(11.0, 2.0, top=1.5), _lm(-2.0, 6.5, top=0.7, length=1.6), _lm(4.0, 12.0, top=1.0)]
    matches = match_landmarks(src, tgt)
    assert matches, "nothing matched"
    best = matches[0]
    assert abs((best.yaw_deg - 37.0 + 180.0) % 360.0 - 180.0) < 1.5, best.yaw_deg
    assert np.linalg.norm(best.translation - np.array([4.0, -3.0])) < 0.15, best.translation
    assert len(best.pairs) >= 5, len(best.pairs)


def test_a_mirror_image_is_not_a_match():
    src = _arrangement()
    mirrored = [Landmark(**{**lm.__dict__, "xy": np.array([-lm.xy[0], lm.xy[1]])}) for lm in src]
    true_best = match_landmarks(src, _moved(src, 37.0, np.array([4.0, -3.0])))[0]
    mirror_best = max(len(m.pairs) for m in match_landmarks(src, mirrored))
    assert len(true_best.pairs) == len(src), len(true_best.pairs)
    assert mirror_best <= len(src) - 3, f"a mirror image lined up {mirror_best} of {len(src)}"


def test_a_different_arrangement_does_not_match_and_things_of_the_wrong_size_do_not_pair():
    src = _arrangement()
    rng = np.random.default_rng(5)
    unrelated = [_lm(*rng.uniform(0, 15, 2)) for _ in range(8)]
    assert all(len(m.pairs) <= 4 for m in match_landmarks(src, unrelated))
    huge = _moved(src, 20.0, np.array([1.0, 1.0]))
    for lm in huge:
        lm.length, lm.width = lm.length * 5.0, lm.width * 5.0
    assert match_landmarks(src, huge) == []


def test_too_few_things_says_nothing():
    assert match_landmarks(_arrangement()[:2], _arrangement()) == []


def _scene(rng):
    """A floor, two walls and spheres (curved, so no plane strips them) at lopsided places. Surfaces are jittered a few mm
    like a real scan: a perfect lattice at exactly the voxel size stalls Open3D's normal orientation."""
    def jitter(p):
        return p + rng.normal(0, 0.004, p.shape)
    fx, fy = np.meshgrid(np.arange(0, 14, 0.07), np.arange(0, 10, 0.07))
    floor = jitter(np.column_stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)]))
    wy, wz = np.meshgrid(np.arange(0, 10, 0.07), np.arange(0, 3, 0.07))
    wall_x = jitter(np.column_stack([np.zeros(wy.size), wy.ravel(), wz.ravel()]))
    wx, wz2 = np.meshgrid(np.arange(0, 14, 0.07), np.arange(0, 3, 0.07))
    wall_y = jitter(np.column_stack([wx.ravel(), np.zeros(wx.size), wz2.ravel()]))
    spheres = []
    for (cx, cy, cz, r) in [(3.0, 2.5, 0.8, 0.45), (7.5, 2.0, 1.2, 0.6), (11.0, 4.0, 0.7, 0.35), (4.5, 6.5, 1.0, 0.5),
                            (9.0, 7.5, 1.5, 0.7), (12.5, 8.5, 0.9, 0.4), (6.0, 4.5, 0.6, 0.3)]:
        v = rng.normal(size=(1500, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        spheres.append(np.array([cx, cy, cz]) + r * v)
    return np.vstack([floor, wall_x, wall_y] + spheres)


def test_two_synthetic_scans_of_a_room_with_things_in_it_give_the_true_pose_as_a_candidate():
    rng = np.random.default_rng(7)
    a = _scene(rng)
    yaw, t = np.radians(33.0), np.array([3.5, -2.0, 0.0])
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    b = a @ rot.T + t                                                                    # the same room seen from elsewhere
    cfg = PipelineConfig()
    pa, pb = preprocess_scan(Scan(name="A", path="A.pts", points=a), 0, cfg), preprocess_scan(Scan(name="B", path="B.pts", points=b), 1, cfg)
    cands, matches, src_lms, tgt_lms = landmark_candidates(pa.down, pb.down, top_k=5)
    assert len(src_lms) >= 5 and len(tgt_lms) >= 5, (len(src_lms), len(tgt_lms))
    truth = np.eye(4)
    truth[:3, :3], truth[:3, 3] = rot, t
    errors = []
    for full in cands:
        delta = np.linalg.inv(truth) @ full
        errors.append((float(np.linalg.norm(delta[:3, 3])), float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))))
    assert any(e[0] < 0.15 and e[1] < 1.5 for e in errors[:3]), errors


def test_landmarks_are_only_used_when_asked_for():
    rng = np.random.default_rng(9)
    pts = _scene(rng)[::8]
    a, b = Scan(name="A", path="A.pts", points=pts), Scan(name="B", path="B.pts", points=pts + np.array([0.3, 0.1, 0.0]))
    assert PipelineConfig().use_landmarks is False
    calls = []
    real = align_pair.landmark_candidates
    align_pair.landmark_candidates = lambda *a_, **k_: calls.append(1) or ([], [], [], [])
    try:
        for flag in (False, True):
            cfg = PipelineConfig(use_landmarks=flag, coarse_method="gravity")
            pa, pb = preprocess_scan(a, 0, cfg), preprocess_scan(b, 1, cfg)
            align_pair.register_pair(pa, pb, cfg, init=None)
            assert len(calls) == (1 if flag else 0), (flag, calls)
    finally:
        align_pair.landmark_candidates = real


def _rigid_panel(rng, centre, w=0.7, h=1.1, h0=0.3, n=600, noise=0.006):
    """A flat panel's worth of points (a chair back, a case face) - locally planar everywhere,
    even though `find_landmarks` itself only ever sees its footprint, not this local structure."""
    u, v = rng.uniform(-w / 2, w / 2, n), rng.uniform(-h / 2, h / 2, n)
    xy = np.column_stack([centre[0] + u, centre[1] + rng.normal(0, noise, n)])
    return xy, h0 + h / 2 + v


def _organic_blob(rng, centre, radius=0.35, h0=0.4, n=600, roughness=0.3):
    """A rough, irregular surface (a person, a pet): a sphere with per-point radial roughness large enough to break
    local flatness. A smooth sphere would still read as locally planar."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    r = radius * (1.0 + rng.normal(0, roughness, n))
    p = np.array([centre[0], centre[1], h0 + radius]) + r[:, None] * v
    return p[:, :2], p[:, 2]


def test_organic_fraction_is_low_for_a_flat_panel_and_high_for_a_rough_blob():
    rng = np.random.default_rng(11)
    xy, h = _rigid_panel(rng, (0.0, 0.0))
    p3 = np.column_stack([xy, h])
    panel_frac = _organic_fraction(p3)
    xy2, h2 = _organic_blob(rng, (0.0, 0.0))
    blob_frac = _organic_fraction(np.column_stack([xy2, h2]))
    assert panel_frac < 0.15, panel_frac
    assert blob_frac > 0.6, blob_frac


def test_organic_fraction_says_nothing_on_too_few_points():
    assert _organic_fraction(np.zeros((5, 3))) == 0.0


def test_reject_organic_drops_the_blob_and_keeps_the_panel():
    rng = np.random.default_rng(12)
    panel = _rigid_panel(rng, (2.0, 2.0))
    blob = _organic_blob(rng, (6.0, 6.0))
    xy = np.vstack([panel[0], blob[0]])
    h = np.concatenate([panel[1], blob[1]])
    without = find_landmarks(xy, h)
    assert len(without) == 2, "both should cluster as things before any organic check runs"
    found = find_landmarks(xy, h, reject_organic=True)
    assert len(found) == 1, [lm.xy.round(1).tolist() for lm in found]
    assert np.linalg.norm(found[0].xy - np.array([2.0, 2.0])) < 0.2, found[0].xy


def test_reject_organic_leaves_a_small_cluster_alone_not_enough_evidence():
    rng = np.random.default_rng(13)
    # small and tight (radius 0.15m) so all 30 points land in one connected cluster - under
    # organic_min_points (40), so the organic check must be skipped, not applied on thin evidence.
    xy, h = _organic_blob(rng, (1.0, 1.0), radius=0.15, n=30)
    found = find_landmarks(xy, h, reject_organic=True, min_points=10)
    assert len(found) == 1, [lm.xy.round(2).tolist() for lm in found]


def test_build_scan_landmarks_ignores_a_stale_residual_idx_and_still_finds_real_things():
    """build_scan_landmarks must ignore a caller's residual_idx and recompute its own: one computed from a slightly
    different down cloud is off by a few points. A deliberately wrong residual_idx is passed in here."""
    rng = np.random.default_rng(15)
    scene = _scene(rng)                                          # floor + 2 walls + 7 spheres, see its own docstring
    scan = preprocess_scan(Scan(name="S", path="S.pts", points=scene), 0, PipelineConfig())
    features = ScanFeatures(surfaces=[], residual_idx=np.array([999999, 1000000]))  # deliberately wrong/stale
    build_scan_landmarks(features, scan.down, np.eye(3), 0.0, min_points=10)
    assert len(features.landmarks) >= 3, "the real spheres should still be found as landmarks"
    for lm in features.landmarks:
        assert lm.raw_xyz is not None


def test_build_scan_landmarks_raw_xyz_lands_near_a_real_spheres_own_centre():
    rng = np.random.default_rng(16)
    scene = _scene(rng)
    yaw = np.radians(21.0)
    r = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    floor_z = -1.3
    # levelled = raw @ r.T and h = levelled_z - floor_z, so for h to equal the scene's z (floor at z=0),
    # raw = (scene + [0, 0, floor_z]) @ r.
    raw = (scene + [0.0, 0.0, floor_z]) @ r
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw))
    features = ScanFeatures(surfaces=[], residual_idx=np.array([]))   # ignored - see the test above
    build_scan_landmarks(features, pcd, r, floor_z, min_points=10)
    assert len(features.landmarks) >= 3, features.landmarks
    known_centres_raw = [(np.array([cx, cy, cz]) + [0.0, 0.0, floor_z]) @ r
                         for (cx, cy, cz, _r) in
                         [(3.0, 2.5, 0.8, 0.45), (7.5, 2.0, 1.2, 0.6), (11.0, 4.0, 0.7, 0.35),
                          (4.5, 6.5, 1.0, 0.5), (9.0, 7.5, 1.5, 0.7), (12.5, 8.5, 0.9, 0.4), (6.0, 4.5, 0.6, 0.3)]]
    for lm in features.landmarks:
        assert lm.raw_xyz is not None
        nearest = min(np.linalg.norm(lm.raw_xyz - c) for c in known_centres_raw)
        assert nearest < 0.3, (lm.raw_xyz, nearest)


def _lm_raw(x, y, z, length=0.8, h=0.85, weight=5.0):
    return Landmark(xy=np.array([x, y]), h=h, length=length, width=length * 0.7, top=h + length / 2, base=h - length / 2,
                    n_points=100, brightness=0.0, weight=weight, raw_xyz=np.array([x, y, z]))


def _lfeatures(*landmarks):
    return ScanFeatures(surfaces=[], residual_idx=np.array([], dtype=np.int64), landmarks=list(landmarks))


def test_landmark_agreement_matches_an_identical_landmark():
    a = _lm_raw(1.0, 2.0, 0.5)
    agree, n = landmark_agreement(np.eye(4), _lfeatures(a), _lfeatures(_lm_raw(1.0, 2.0, 0.5)), CFG)
    assert n == 1 and agree > 0.99, (agree, n)


def test_landmark_agreement_falls_off_with_distance_and_far_is_not_counted():
    a = _lm_raw(0.0, 0.0, 0.5)
    near, n_near = landmark_agreement(np.eye(4), _lfeatures(a), _lfeatures(_lm_raw(0.15, 0.0, 0.5)), CFG)
    far, n_far = landmark_agreement(np.eye(4), _lfeatures(a), _lfeatures(_lm_raw(2.0, 0.0, 0.5)), CFG)
    assert n_near == 1 and near < 0.9, near
    assert n_far == 0 and far == 0.0


def test_landmark_agreement_height_or_size_mismatch_does_not_pair():
    a = _lm_raw(0.0, 0.0, 0.5)
    assert landmark_agreement(np.eye(4), _lfeatures(a), _lfeatures(_lm_raw(0.0, 0.0, 1.5)), CFG)[1] == 0
    assert landmark_agreement(np.eye(4), _lfeatures(a), _lfeatures(_lm_raw(0.0, 0.0, 0.5, length=3.0)), CFG)[1] == 0


def test_landmark_agreement_with_no_raw_xyz_says_nothing():
    a = Landmark(xy=np.array([0.0, 0.0]), h=0.5, length=0.5, width=0.4, top=0.7, base=0.3,
                n_points=100, brightness=0.0, weight=5.0)   # raw_xyz left at its default (None)
    assert landmark_agreement(np.eye(4), _lfeatures(a), _lfeatures(_lm_raw(0.0, 0.0, 0.5)), CFG) == (0.0, 0)


def _verdict():
    v = Verdict(transform=np.eye(4), surface_fit=0.08)
    v.vote_score = v.surface_fit
    return v


def test_landmarks_alone_raise_the_ranking_and_can_never_reject():
    good, bad = _verdict(), _verdict()
    apply_detector_vote(good, 0.0, 0.0, 0, CFG, landmark_agreement=1.0, n_landmarks=3)
    apply_detector_vote(bad, 0.0, 0.0, 0, CFG, landmark_agreement=0.0, n_landmarks=3)
    assert good.rejected_by == "" and bad.rejected_by == ""
    assert good.vote_score > good.surface_fit and bad.vote_score < good.vote_score
    assert good.n_landmarks_considered == 3 and good.landmark_agreement == 1.0


def test_landmarks_cannot_rescue_a_candidate_the_building_rejects_or_sink_one_it_accepts():
    rejected = _verdict()
    apply_detector_vote(rejected, 0.05, 0.0, 3, CFG, corner_agreement=0.05, n_corners=2,
                        landmark_agreement=1.0, n_landmarks=4)
    assert rejected.rejected_by != "", "a strong landmark match must not overrule planes and corners that disagree"
    accepted = _verdict()
    apply_detector_vote(accepted, 0.9, 0.0, 3, CFG, corner_agreement=0.9, n_corners=2,
                        landmark_agreement=0.0, n_landmarks=4)
    assert accepted.rejected_by == "", "landmarks that disagree must not reject what planes and corners accept"


def test_with_no_landmarks_the_vote_is_exactly_what_it_was():
    plain, with_zero = _verdict(), _verdict()
    apply_detector_vote(plain, 0.6, 0.2, 3, CFG, corner_agreement=0.7, n_corners=2, edge_agreement=0.5, n_edges=3)
    apply_detector_vote(with_zero, 0.6, 0.2, 3, CFG, corner_agreement=0.7, n_corners=2, edge_agreement=0.5, n_edges=3,
                        landmark_agreement=0.0, n_landmarks=0)
    assert plain.vote_score == with_zero.vote_score and plain.detector_score == with_zero.detector_score


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
