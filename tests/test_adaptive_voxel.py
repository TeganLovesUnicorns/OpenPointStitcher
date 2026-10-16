"""Tests for the per-pair search voxel: off changes nothing, on is never finer than configured, dense pairs are
untouched, and sparse pairs get a coarser search that is actually used."""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import align_pair  # noqa: E402
from open_point_stitch.align_pair import pair_search_voxel, register_pair  # noqa: E402
from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.prepare import Prepared, preprocess_scan  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402


def _prep(spacing_m: float) -> Prepared:
    return Prepared(name="s", index=0, down=o3d.geometry.PointCloud(), fpfh=None, n_full=1, spacing_m=spacing_m)


def test_off_by_default_and_then_the_configured_search_voxel_is_used_as_it_always_was():
    assert PipelineConfig().adaptive_search_voxel is False
    for configured in (None, 0.15):
        cfg = PipelineConfig(search_voxel_size=configured)
        assert pair_search_voxel(_prep(0.09), _prep(0.01), cfg) == configured


def test_a_dense_pair_is_untouched_even_when_on():
    cfg = PipelineConfig(adaptive_search_voxel=True)
    assert pair_search_voxel(_prep(0.012), _prep(0.0177), cfg) is None       # the real venue's densest scans
    assert pair_search_voxel(_prep(0.012), _prep(0.0177), replace(cfg, search_voxel_size=0.15)) == 0.15


def test_a_sparse_scan_makes_its_pairs_search_coarser_rounded_up_to_a_centimetre():
    cfg = PipelineConfig(adaptive_search_voxel=True)
    assert pair_search_voxel(_prep(0.0254), _prep(0.012), cfg) == 0.07       # the real venue's sparsest scan: 6.35 cm wanted
    assert pair_search_voxel(_prep(0.04), _prep(0.04), cfg) == 0.10


def test_it_is_never_finer_than_the_configured_search_voxel():
    cfg = PipelineConfig(adaptive_search_voxel=True, search_voxel_size=0.15)
    assert pair_search_voxel(_prep(0.0254), _prep(0.012), cfg) == 0.15       # 0.07 wanted, 0.15 configured: stays 0.15
    assert pair_search_voxel(_prep(0.08), _prep(0.02), cfg) == 0.20          # a very sparse scan: coarser than configured


def _room(rng, n=6000):
    floor = np.column_stack([rng.uniform(0, 6, n), rng.uniform(0, 4, n), np.zeros(n)])
    wall = np.column_stack([rng.uniform(0, 6, n), np.zeros(n), rng.uniform(0, 2.5, n)])
    wall2 = np.column_stack([np.zeros(n), rng.uniform(0, 4, n), rng.uniform(0, 2.5, n)])
    return np.vstack([floor, wall, wall2])


def test_a_sparse_pair_really_searches_at_the_coarser_voxel_and_a_dense_one_does_not():
    rng = np.random.default_rng(3)
    pts = _room(rng)
    cfg = PipelineConfig(adaptive_search_voxel=True, coarse_method="gravity")
    a, b = Scan(name="A", path="A.pts", points=pts), Scan(name="B", path="B.pts", points=pts + np.array([0.3, 0.1, 0.0]))
    prep_a, prep_b = preprocess_scan(a, 0, cfg), preprocess_scan(b, 1, cfg)
    seen = []
    real = align_pair._search_prepared

    def spy(base, c):
        seen.append(c.search_voxel_size)
        return real(base, c)
    align_pair._search_prepared = spy
    try:
        prep_a.spacing_m = prep_b.spacing_m = 0.01
        register_pair(prep_a, prep_b, cfg, init=None)
        dense = list(seen)
        seen.clear()
        prep_a.spacing_m = 0.04
        register_pair(prep_a, prep_b, cfg, init=None)
        sparse = list(seen)
    finally:
        align_pair._search_prepared = real
    assert dense == [], f"a dense pair should not have needed a coarser copy, saw {dense}"
    assert sparse == [0.10, 0.10], f"a 4 cm spacing should search at 10 cm, saw {sparse}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc!r}")
    raise SystemExit(1 if failures else 0)
