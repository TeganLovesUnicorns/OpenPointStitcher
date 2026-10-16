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
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
