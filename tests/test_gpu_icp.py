"""Tests for the GPU ICP (align_gpu.py) and the screen-then-confirm path in register_pair: it agrees with the
CPU ICP, gives the same answer every run, hands the final answer to the CPU, and falls back to the CPU when there
is no GPU or it fails. Tests that need a GPU skip without one."""
from __future__ import annotations

import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import align_pair  # noqa: E402
from open_point_stitch.align_pair import register_pair  # noqa: E402
from open_point_stitch.config import PipelineConfig  # noqa: E402
from open_point_stitch.prepare import preprocess_scan  # noqa: E402
from open_point_stitch.pts_file import Scan  # noqa: E402


def _card() -> bool:
    try:
        from open_point_stitch.limits import gpu_status
        return bool(gpu_status()["gpu_usable"])
    except Exception:  # noqa: BLE001
        return False


def _room(rng, n=7000):
    """A floor and two walls with a little furniture, jittered a few mm like a real scan."""
    floor = np.column_stack([rng.uniform(0, 8, n), rng.uniform(0, 6, n), rng.normal(0, 0.004, n)])
    wall_a = np.column_stack([rng.uniform(0, 8, n), rng.normal(0, 0.004, n), rng.uniform(0, 2.6, n)])
    wall_b = np.column_stack([rng.normal(0, 0.004, n), rng.uniform(0, 6, n), rng.uniform(0, 2.6, n)])
    box = np.column_stack([rng.uniform(4, 5, 1500), rng.uniform(3, 4, 1500), rng.uniform(0, 1.0, 1500)])
    return np.vstack([floor, wall_a, wall_b, box])


def _pair(shift=(0.25, -0.15, 0.0), yaw_deg=2.0):
    rng = np.random.default_rng(5)
    pts = _room(rng)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    moved = pts @ rot.T + np.array(shift)
    cfg = PipelineConfig()
    return (preprocess_scan(Scan(name="A", path="A.pts", points=pts), 0, cfg),
            preprocess_scan(Scan(name="B", path="B.pts", points=moved), 1, cfg), cfg)


def _cpu_icp(a, b, init, dist, cfg):
    reg = o3d.pipelines.registration
    r = reg.registration_icp(a.down, b.down, dist, init, reg.TransformationEstimationPointToPlane(),
                             reg.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iteration))
    return np.asarray(r.transformation), float(r.fitness), float(r.inlier_rmse)


def test_the_gpu_icp_agrees_with_the_cpu_icp():
    if not _card():
        print("    SKIP: no usable GPU here")
        return
    from open_point_stitch.align_gpu import cloud_for, icp_point_to_plane
    a, b, cfg = _pair()
    init = np.eye(4)
    for dist in (0.3, 0.1, 0.04):
        Tc, fc, rc = _cpu_icp(a, b, init, dist, cfg)
        Tg, fg, rg, _n = icp_point_to_plane(cloud_for(a), cloud_for(b), init, dist, cfg.icp_max_iteration)
        delta = np.linalg.inv(Tc) @ Tg
        assert np.linalg.norm(delta[:3, 3]) < 1e-3, (dist, np.linalg.norm(delta[:3, 3]))
        assert np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))) < 0.02, dist
        assert abs(fg - fc) < 2e-3 and abs(rg - rc) < 2e-4, (dist, fg, fc, rg, rc)
        init = Tc


def test_a_batch_gives_each_start_the_answer_it_would_get_alone_and_repeats_exactly():
    if not _card():
        print("    SKIP: no usable GPU here")
        return
    from open_point_stitch.align_gpu import cloud_for, icp_point_to_plane, icp_point_to_plane_batch
    a, b, cfg = _pair()
    cs, ct = cloud_for(a), cloud_for(b)
    starts = [np.eye(4)]
    for dx, dy, yaw in ((0.1, 0.05, 1.0), (-0.1, 0.1, -2.0), (0.05, -0.1, 3.0)):
        m = np.eye(4)
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        m[:2, :2] = [[c, -s], [s, c]]
        m[:2, 3] = (dx, dy)
        starts.append(m)
    once = icp_point_to_plane_batch(cs, ct, starts, 0.1, cfg.icp_max_iteration)
    twice = icp_point_to_plane_batch(cs, ct, starts, 0.1, cfg.icp_max_iteration)
    for x, y in zip(once, twice):
        assert np.array_equal(x[0], y[0]) and x[1] == y[1], "the GPU ICP did not repeat exactly"
    for start, got in zip(starts, once):
        alone = icp_point_to_plane(cs, ct, start, 0.1, cfg.icp_max_iteration)
        assert np.abs(alone[0] - got[0]).max() < 1e-5 and abs(alone[1] - got[1]) < 1e-6, "a batch changed a start's answer"


def _run(a, b, cfg, init):
    return register_pair(a, b, cfg, init=init, seeded_from="manual", local_only=True)


def test_the_screen_and_confirm_path_returns_the_cpus_answer_for_its_winner():
    if not _card():
        print("    SKIP: no usable GPU here")
        return
    a, b, cfg = _pair()
    off = np.eye(4)
    off[:2, 3] = (0.12, -0.05)                                    # a placement that is a little off
    cpu = _run(a, b, cfg, off)
    gpu = _run(a, b, PipelineConfig(use_gpu_ladder=True), off)
    assert gpu.gpu_screened is True and gpu.gpu_verified >= 1, (gpu.gpu_screened, gpu.gpu_verified)
    assert gpu.accepted == cpu.accepted and gpu.method == cpu.method, (gpu.method, cpu.method)
    assert np.linalg.norm(gpu.transform[:3, 3] - cpu.transform[:3, 3]) < 0.01, "the confirmed winner is not the CPU's answer"
    assert gpu.gpu_vs_cpu_cm is not None and gpu.gpu_vs_cpu_cm < 5.0, gpu.gpu_vs_cpu_cm
    row = gpu.as_row()
    assert row["gpu_screened"] is True and row["gpu_verified"] == gpu.gpu_verified


def test_the_from_scratch_gpu_pipeline_finds_a_synthetic_pair():
    """align_gpu's CUDA pair pipeline, the path "Run automatic" takes with no placement and a card: no other test finishes it."""
    if not _card():
        print("    SKIP: no usable GPU here")
        return
    from open_point_stitch.align_gpu import prepare_all_gpu, register_pair_cuda_gated
    a, b, cfg = _pair(shift=(0.6, -0.4, 0.0), yaw_deg=20.0)
    ga, gb = prepare_all_gpu([a, b], cfg)
    r = register_pair_cuda_gated(a, b, ga, gb, cfg)
    c, s = np.cos(np.radians(20.0)), np.sin(np.radians(20.0))
    truth = np.eye(4)
    truth[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    truth[:3, 3] = (0.6, -0.4, 0.0)
    delta = np.linalg.inv(truth) @ r.transform
    off_cm = np.linalg.norm(delta[:3, 3]) * 100
    off_deg = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
    print(f"    accepted {r.accepted}, {off_cm:.2f} cm and {off_deg:.3f} deg from the truth")
    assert r.accepted and off_cm < 2.0 and off_deg < 0.5, (r.accepted, r.reason, off_cm, off_deg)


def test_with_no_card_the_pair_runs_wholly_on_the_cpu():
    a, b, cfg = _pair()
    off = np.eye(4)
    off[:2, 3] = (0.12, -0.05)
    cpu = _run(a, b, cfg, off)
    align_pair._gpu_ready.cache_clear()
    real = align_pair._gpu_ready
    align_pair._gpu_ready = lambda: False
    try:
        got = _run(a, b, PipelineConfig(use_gpu_ladder=True), off)
    finally:
        align_pair._gpu_ready = real
    assert got.gpu_screened is None and got.gpu_verified is None
    assert np.abs(got.transform - cpu.transform).max() < 1e-9 and got.fitness == cpu.fitness, "no card must mean the CPU answer"


def test_a_gpu_failure_falls_back_to_the_cpu_instead_of_failing_the_pair():
    a, b, cfg = _pair()
    off = np.eye(4)
    off[:2, 3] = (0.12, -0.05)
    cpu = _run(a, b, cfg, off)
    import open_point_stitch.align_gpu as icp_gpu
    real_ready, real_ladder = align_pair._gpu_ready, icp_gpu.ladder_batch
    align_pair._gpu_ready = lambda: True

    def boom(*args, **kwargs):
        raise RuntimeError("the card went away")
    icp_gpu.ladder_batch = boom
    try:
        got = _run(a, b, PipelineConfig(use_gpu_ladder=True), off)
    finally:
        align_pair._gpu_ready, icp_gpu.ladder_batch = real_ready, real_ladder
    assert got.gpu_screened is None and np.abs(got.transform - cpu.transform).max() < 1e-9, "a GPU error must fall back to the CPU answer"


def test_a_full_card_sends_each_pair_to_the_cpu_until_it_frees_up():
    a, b, cfg = _pair()
    off = np.eye(4)
    off[:2, 3] = (0.12, -0.05)
    cpu = _run(a, b, cfg, off)
    from open_point_stitch import limits
    real_ready, real_full = align_pair._gpu_ready, limits.card_too_full
    align_pair._gpu_ready = lambda: True
    asked, card = [], {"full": True}

    def full(needs_gb):
        asked.append(needs_gb)
        return "the graphics card is nearly full (9.9 of 10.2 GB in use, most likely by another program)" if card["full"] else ""
    limits.card_too_full = full
    try:
        align_pair.reset_gpu_stop()
        first = _run(a, b, PipelineConfig(use_gpu_ladder=True), off)
        second = _run(a, b, PipelineConfig(use_gpu_ladder=True), off)
        reason, moved = align_pair.gpu_stop_reason(), align_pair.gpu_moved_pairs()
        card["full"] = False
        freed = align_pair.card_full_now(pair=True)
    finally:
        align_pair._gpu_ready, limits.card_too_full = real_ready, real_full
    assert first.gpu_screened is None and second.gpu_screened is None
    assert np.abs(first.transform - cpu.transform).max() < 1e-9, "a full card must give the CPU answer"
    assert asked[:2] == [limits.GPU_RUN_NEEDS_GB] * 2, "every pair asks again, needing room for the run"
    assert "nearly full" in reason and moved == 2
    assert not freed and not align_pair.gpu_busy_now() and align_pair.gpu_moved_pairs() == 2
    align_pair.reset_gpu_stop()
    assert align_pair.gpu_stop_reason() == "" and align_pair.gpu_moved_pairs() == 0


def test_the_card_check_says_nothing_when_the_card_cannot_be_read_or_has_room():
    from open_point_stitch import limits
    real = limits.card_memory
    try:
        limits.card_memory = lambda: None
        assert limits.card_too_full(2.0) == ""
        limits.card_memory = lambda: (2.5, 10.7)
        assert limits.card_too_full(2.0) == ""
        limits.card_memory = lambda: (9.9, 10.7)
        assert "9.9 of 10.7 GB" in limits.card_too_full(2.0)
    finally:
        limits.card_memory = real


def test_a_real_card_reports_its_memory_for_every_program_on_it():
    from open_point_stitch.limits import card_memory
    if not _card():
        print("    SKIP: no CUDA device visible here")
        return
    mem = card_memory()
    assert mem is not None and 0 < mem[0] < mem[1], mem


def test_the_gpu_ladder_is_off_unless_asked_for():
    assert PipelineConfig().use_gpu_ladder is False


def test_a_visible_card_that_cannot_run_the_gpu_code_is_not_usable():
    # e.g. a GTX 1060 under this build: visible, but no kernel for it
    from open_point_stitch import limits
    cuda = o3d.core.cuda
    real = (cuda.is_available, cuda.device_count, limits._cuda_kernel_runs)
    cuda.is_available, cuda.device_count = (lambda: True), (lambda: 1)
    limits._cuda_kernel_runs = lambda: (False, "RuntimeError: no kernel image is available for execution on the device")
    try:
        st = limits.gpu_status()
        resolved = limits.resolve_use_gpu(True)
    finally:
        cuda.is_available, cuda.device_count, limits._cuda_kernel_runs = real
    assert st["cuda_device_available"] is True and st["cuda_kernel_runs"] is False, st
    assert st["gpu_usable"] is False and resolved is False, st
    assert "older than this build supports" in st["notes"], st["notes"]


def test_a_real_visible_card_passes_the_kernel_check():
    from open_point_stitch.limits import gpu_status
    st = gpu_status()
    if not st["cuda_device_available"]:
        print("    SKIP: no CUDA device visible here")
        return
    assert st["cuda_kernel_runs"] is True and st["gpu_usable"] is True, st


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
