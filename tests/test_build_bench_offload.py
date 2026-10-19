"""server.py builds previews (build_scan_payload_cached) in a worker process, not the request thread. The payload
a scan's preview gets must be the same either way."""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import build_bench  # noqa: E402


def _write_room(path: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(0, 6, 5000), rng.uniform(0, 4, 5000), rng.normal(0, 0.004, 5000)])
    wall_a = np.column_stack([rng.uniform(0, 6, 3000), rng.normal(0, 0.004, 3000), rng.uniform(0, 2.6, 3000)])
    wall_b = np.column_stack([rng.normal(0, 0.004, 3000), rng.uniform(0, 4, 3000), rng.uniform(0, 2.6, 3000)])
    pts = np.vstack([floor, wall_a, wall_b])
    rgb = rng.integers(0, 256, size=(len(pts), 3))
    with open(path, "w") as fh:
        fh.write(f"{len(pts)}\n")
        for p, c in zip(pts, rgb):
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def test_offloaded_payload_is_identical_to_in_process():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "room.pts")
        _write_room(path, seed=1)
        cache_a = os.path.join(tmp, "cache_a")
        cache_b = os.path.join(tmp, "cache_b")

        inproc, was_cached_inproc = build_bench.build_scan_payload_cached(path, cache_a)

        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            offloaded, was_cached_off = pool.submit(
                build_bench.build_scan_payload_cached, path, cache_b).result()

        assert not was_cached_inproc and not was_cached_off  # both a fresh build, not a cache hit
        assert inproc.keys() == offloaded.keys()
        for key in inproc:
            a, b = inproc[key], offloaded[key]
            if isinstance(a, float):
                assert a == b, f"{key}: {a} != {b}"
            else:
                assert a == b, f"{key}: differs between in-process and offloaded"


def test_a_second_call_through_the_pool_hits_the_cache():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "room.pts")
        _write_room(path, seed=2)
        cache_dir = os.path.join(tmp, "cache")

        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            first, was_cached_1 = pool.submit(
                build_bench.build_scan_payload_cached, path, cache_dir).result()
            second, was_cached_2 = pool.submit(
                build_bench.build_scan_payload_cached, path, cache_dir).result()

        assert not was_cached_1 and was_cached_2
        assert first == second


def _write_tipped_room(path: str, deg: float) -> None:
    """The room with its floor 1.2 m below the scanner, then tipped `deg` about x, as a raked floor would look."""
    _write_room(path, seed=4)
    with open(path) as fh:
        rows = [ln.split() for ln in fh.read().splitlines()[1:]]
    pts = np.array([[float(v) for v in r[:3]] for r in rows]) - [0.0, 0.0, 1.2]
    a = np.radians(deg)
    pts = pts @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]).T
    with open(path, "w") as fh:
        fh.write(f"{len(pts)}\n")
        for p, r in zip(pts, rows):
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r[3]} {r[4]} {r[5]}\n")


def test_keeping_the_iphones_level_leaves_the_scan_unturned_and_still_finds_its_floor():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "room.pts")
        _write_tipped_room(path, 2.0)
        plain = build_bench.preview_options({})
        kept = build_bench.preview_options({}, keep_level_scans=["room"])
        p_plain, _ = build_bench.build_scan_payload_cached(path, os.path.join(tmp, "a"), None, None, plain)
        p_kept, _ = build_bench.build_scan_payload_cached(path, os.path.join(tmp, "a"), None, None, kept)
        assert not np.allclose(p_plain["level_rotation"], np.eye(3)), "the plain build should level to the floor"
        assert np.allclose(p_kept["level_rotation"], np.eye(3)), p_kept["level_rotation"]
        assert abs(p_kept["tilt"] - 2.0) < 0.5, f"the floor's own slope is still reported: {p_kept['tilt']}"
        assert abs(p_kept["floor_z"] - (-1.2 / np.cos(np.radians(2.0)))) < 0.02, p_kept["floor_z"]
        assert build_bench.load_cached_payload(path, os.path.join(tmp, "a"), options=kept) == p_kept
        assert build_bench.load_cached_payload(path, os.path.join(tmp, "a"), options=plain) == p_plain, \
            "the plain saved preview must survive next to the kept-level one"
        other = build_bench.preview_options({}, keep_level_scans=["some other scan"])
        assert build_bench._cache_key(path, other) == build_bench._cache_key(path, plain), \
            "another scan's switch must not invalidate this scan's saved preview"


def test_levelling_off_for_the_project_keeps_every_scan_at_the_iphones_level():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "room.pts")
        _write_tipped_room(path, 2.0)
        off = build_bench.preview_options({"level_tilt_deg": 0})
        p_off, _ = build_bench.build_scan_payload_cached(path, os.path.join(tmp, "a"), None, None, off)
        assert np.allclose(p_off["level_rotation"], np.eye(3)), p_off["level_rotation"]
        assert abs(p_off["tilt"] - 2.0) < 0.5, p_off["tilt"]


def test_the_levelling_setting_reaches_the_worker_and_its_saved_preview_is_found_again():
    # Builds run in fresh worker processes, which never see a setting changed in the server process, so settings must
    # be passed in and be part of the cache key.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "room.pts")
        _write_room(path, seed=3)
        with open(path) as fh:
            rows = [ln.split() for ln in fh.read().splitlines()[1:]]
        pts = np.array([[float(v) for v in r[:3]] for r in rows])
        a = np.radians(5.0)                                   # the whole room tipped 5 degrees
        pts = pts @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]).T
        with open(path, "w") as fh:
            fh.write(f"{len(pts)}\n")
            for p, r in zip(pts, rows):
                fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r[3]} {r[4]} {r[5]}\n")
        strict = build_bench.preview_options({"level_tilt_deg": 3})
        loose = build_bench.preview_options({})
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            p_strict, _ = pool.submit(build_bench.build_scan_payload_cached, path, os.path.join(tmp, "a"), None, None,
                                      strict).result()
            p_loose, _ = pool.submit(build_bench.build_scan_payload_cached, path, os.path.join(tmp, "b"), None, None,
                                     loose).result()
        assert abs(p_loose["tilt"] - 5.0) < 0.5, p_loose["tilt"]          # 15 degrees allowed: straightened
        assert p_strict["tilt"] < 1.0, p_strict["tilt"]                    # 3 allowed: left as it is
        assert build_bench.load_cached_payload(path, os.path.join(tmp, "a"), options=strict) == p_strict
        assert build_bench.load_cached_payload(path, os.path.join(tmp, "a"), options=loose) is None


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
