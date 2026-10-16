"""Run-scoped resource controls.

Apply before the numeric libraries start their thread pools: the CLI calls apply_thread_cap() first, since changing
the environment afterwards has no reliable effect."""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import os
import threading
from typing import Optional

log = logging.getLogger(__name__)

# Serialises single_threaded()/limited_to(): omp_set_num_threads() is process-wide, so overlapping calls could
# restore the wrong previous count. RLock so a re-entering caller cannot deadlock.
_THREAD_LOCK = threading.RLock()

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def apply_thread_cap(threads: Optional[int]) -> None:
    """Cap CPU thread pools for this process. No-op when threads is None."""
    if threads is None:
        return
    if threads < 1:
        raise ValueError("threads must be >= 1")
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(threads)
    log.info("CPU thread cap set to %d", threads)


def configure_open3d_threads(threads: Optional[int]) -> None:
    """Ask Open3D itself to limit CPU threads, where the build exposes it."""
    if threads is None:
        return
    try:
        import open3d as o3d
        # Available in recent builds. Absence is not an error worth failing on.
        setter = getattr(o3d.utility, "set_cpu_thread_num", None)
        if callable(setter):
            setter(threads)
    except Exception as exc:  # pragma: no cover - depends on build
        log.debug("Could not set Open3D thread count: %s", exc)


def gpu_status() -> dict:
    """Report what the installed Open3D can do on the GPU. Whether it was compiled with CUDA and whether a device is
    visible are separate questions; keep them separate."""
    status = {
        "open3d_version": None,
        "cuda_build": False,          # compiled with CUDA
        "cuda_version": None,         # which CUDA it was built against
        "cuda_device_available": False,  # a GPU is visible right now
        "cuda_device_count": 0,
        "gpu_usable": False,          # both of the above
        "accelerated_stages": [],
        "notes": "",
    }
    try:
        import open3d as o3d
        status["open3d_version"] = o3d.__version__
    except Exception as exc:  # pragma: no cover
        status["notes"] = f"Open3D not importable: {exc}"
        return status

    # 1. what the wheel was built with
    try:
        from open3d._build_config import _build_config
        status["cuda_build"] = bool(_build_config.get("BUILD_CUDA_MODULE", False))
        status["cuda_version"] = _build_config.get("CUDA_VERSION") or None
    except Exception:
        # Older or unusual builds may not expose _build_config. Fall back to
        # asking whether a CUDA device type is even constructible.
        try:
            o3d.core.Device("CUDA:0")
            status["cuda_build"] = True
        except Exception:
            status["cuda_build"] = False

    # 2. whether a card is actually here
    try:
        cuda_mod = o3d.core.cuda
        status["cuda_device_available"] = bool(cuda_mod.is_available())
        status["cuda_device_count"] = int(cuda_mod.device_count())
    except Exception:
        pass

    # 3. whether this build's GPU code runs on that card (a card older than the build supports is visible but
    #    cannot run it)
    status["cuda_kernel_runs"] = False
    if status["cuda_build"] and status["cuda_device_available"]:
        status["cuda_kernel_runs"], kernel_error = _cuda_kernel_runs()
        if kernel_error:
            status["cuda_kernel_error"] = kernel_error

    status["gpu_usable"] = bool(status["cuda_build"] and status["cuda_device_available"]
                                and status["cuda_kernel_runs"])

    # 3. what this version can actually run on the GPU, checked by looking for
    #    the functions rather than by assuming.
    treg = getattr(getattr(o3d, "t", None), "pipelines", None)
    treg = getattr(treg, "registration", None) if treg else None
    tpc = getattr(getattr(o3d, "t", None), "geometry", None)
    tpc = getattr(tpc, "PointCloud", None) if tpc else None
    stages = []
    if tpc is not None:
        if hasattr(tpc, "voxel_down_sample"):
            stages.append("voxel downsample")
        if hasattr(tpc, "estimate_normals"):
            stages.append("normals")
    if treg is not None:
        if hasattr(treg, "compute_fpfh_feature"):
            stages.append("FPFH features")
        if hasattr(treg, "correspondences_from_features"):
            stages.append("feature matching")
        if hasattr(treg, "multi_scale_icp") or hasattr(treg, "icp"):
            stages.append("ICP")
        if hasattr(treg, "get_information_matrix"):
            stages.append("information matrix")
    status["accelerated_stages"] = stages

    if status["gpu_usable"]:
        status["notes"] = (
            f"CUDA build (built against CUDA {status['cuda_version']}) and "
            f"{status['cuda_device_count']} device(s) visible. GPU is usable. "
            "RANSAC global registration has no tensor implementation in this "
            "version and stays on CPU; the pose graph optimiser is CPU only."
        )
    elif status["cuda_device_available"]:
        status["notes"] = (
            f"A CUDA device is visible but this Open3D's GPU code would not run on it "
            f"({status.get('cuda_kernel_error', 'unknown error')}). The card is probably older "
            "than this build supports. Falling back to CPU."
        )
    elif status["cuda_build"]:
        status["notes"] = (
            f"CUDA build (built against CUDA {status['cuda_version']}) but no "
            "CUDA device is visible. Running on GPU needs the container started "
            "with --gpus all, a current NVIDIA driver, and on Windows, Docker "
            "Desktop on the WSL2 backend. Falling back to CPU."
        )
    else:
        status["notes"] = (
            "This Open3D was built without CUDA. On Linux x86_64 the default "
            "`open3d` wheel is the CUDA build and `open3d-cpu` is the CPU one, "
            "so check which of those is installed."
        )
    return status


@functools.lru_cache(maxsize=1)
def _cuda_kernel_runs() -> tuple[bool, str]:
    """Run one tiny calculation on CUDA:0, once per process: (worked, error text)."""
    try:
        import open3d as o3d
        t = o3d.core.Tensor([1.0, 2.0, 3.0], device=o3d.core.Device("CUDA:0"))
        total = float((t * 2.0).sum().item())
        if total != 12.0:
            return False, f"test sum came back {total}, expected 12.0"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def resolve_use_gpu(requested: bool) -> bool:
    """Honour the run's GPU toggle, but never claim GPU we do not have."""
    if not requested:
        return False
    status = gpu_status()
    if not status["gpu_usable"]:
        log.warning("GPU requested but not usable. %s", status["notes"])
        return False
    log.info("GPU enabled: %d device(s), accelerating %s",
             status["cuda_device_count"], ", ".join(status["accelerated_stages"]))
    return True


@contextlib.contextmanager
def single_threaded():
    """Pin Open3D's plane-fitting RANSAC to one thread for this block; restores on exit. Uses omp_set_num_threads()
    via ctypes."""
    with limited_to(1, warn_label="pin plane fitting to one thread"):
        yield


@contextlib.contextmanager
def limited_to(n: int, warn_label: str = "cap CPU threads"):
    """Cap Open3D/OpenMP to n threads for this block (single_threaded() is n=1)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    with _THREAD_LOCK:
        try:
            lib = ctypes.CDLL("libgomp.so.1")
            lib.omp_get_max_threads.restype = ctypes.c_int
            previous = int(lib.omp_get_max_threads())
            # Inside the try so a failing set call falls back like every other failure here.
            lib.omp_set_num_threads(int(n))
        except (OSError, AttributeError) as exc:
            log.warning(
                "could not reach libgomp to %s (%s); proceeding at whatever "
                "thread count is already set", warn_label, exc,
            )
            yield
            return

        try:
            yield
        finally:
            # Restore whatever was there before, not a hardcoded core count,
            # so this behaves correctly whether the caller runs with 2
            # threads or 32.
            lib.omp_set_num_threads(previous)


def apply_seed(seed: int) -> None:
    """Make Open3D's RANSAC deterministic for this process."""
    try:
        import open3d as o3d
        o3d.utility.random.seed(int(seed))
    except Exception as exc:  # pragma: no cover - depends on build
        log.debug("Could not seed Open3D RNG: %s", exc)
