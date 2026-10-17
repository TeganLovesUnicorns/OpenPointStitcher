"""Tests for prepare.preprocess_all_offloaded: same result as the in-process path, stoppable, and errors reach
the caller."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.prepare import preprocess_all, preprocess_all_offloaded  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402


def _room(seed: int, n: int = 6000, colour: bool = True) -> Scan:
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.normal(0, 0.003, n)])
    wall = np.column_stack([np.full(n // 2, -3.0) + rng.normal(0, 0.003, n // 2), rng.uniform(-3, 3, n // 2),
                            rng.uniform(0, 2.5, n // 2)])
    pts = np.vstack([floor, wall])
    return Scan(name=f"room{seed}", path=f"room{seed}.pts", points=pts,
                colors=rng.uniform(0, 1, pts.shape) if colour else None)


def _same(a, b) -> bool:
    return (a.name == b.name and a.index == b.index and a.n_full == b.n_full and a.spacing_m == b.spacing_m
            and np.array_equal(np.asarray(a.down.points), np.asarray(b.down.points))
            and np.array_equal(np.asarray(a.down.normals), np.asarray(b.down.normals))
            and np.array_equal(np.asarray(a.down.colors), np.asarray(b.down.colors))
            and np.array_equal(np.asarray(a.fpfh.data), np.asarray(b.fpfh.data)))


def test_offloaded_result_is_bit_identical_to_in_process():
    cfg = PipelineConfig()
    scans = [_room(1), _room(2)]
    inproc = preprocess_all(scans, cfg)
    off = preprocess_all_offloaded(scans, cfg, workers=2)
    assert off is not None and len(off) == 2
    for a, b in zip(inproc, off):
        assert _same(a, b), f"{a.name}: the worker's answer differs from the in-process one"


def test_a_scan_without_colour_still_round_trips():
    cfg = PipelineConfig()
    scans = [_room(3, colour=False)]
    got = preprocess_all_offloaded(scans, cfg, workers=1)[0]
    assert not got.down.has_colors() and got.down.has_normals() and np.asarray(got.fpfh.data).shape[0] == 33


def test_order_and_index_follow_the_scans_not_the_finish_order():
    cfg = PipelineConfig()
    scans = [_room(10, n=9000), _room(11, n=1500), _room(12, n=4000)]
    got = preprocess_all_offloaded(scans, cfg, workers=3)
    assert [p.name for p in got] == ["room10", "room11", "room12"] and [p.index for p in got] == [0, 1, 2]


def test_cancel_returns_none_without_waiting_for_the_scans():
    cfg = PipelineConfig()
    assert preprocess_all_offloaded([_room(20), _room(21)], cfg, workers=1, should_cancel=lambda: True) is None


def test_a_scan_too_sparse_to_use_raises_in_the_caller():
    cfg = PipelineConfig()
    tiny = Scan(name="tiny", path="tiny.pts", points=np.random.default_rng(0).uniform(0, 100, (5, 3)))
    try:
        preprocess_all_offloaded([tiny], cfg, workers=1)
    except ValueError as exc:
        assert "tiny" in str(exc), exc
        return
    raise AssertionError("a scan with nothing left after the voxel downsample should raise ValueError")


def test_progress_is_reported():
    cfg = PipelineConfig()
    seen = []
    preprocess_all_offloaded([_room(30), _room(31)], cfg, workers=2, progress_cb=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][1] == 2, seen


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
