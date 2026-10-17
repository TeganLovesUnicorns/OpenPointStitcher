"""Tests for limits.single_threaded() using a fake ctypes.CDLL."""

from __future__ import annotations

import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import limits  # noqa: E402


class _FakeGetMaxThreads:
    """A plain callable, not a bound method: single_threaded() sets `.restype` on it."""
    def __init__(self, lib):
        self.lib = lib
        self.restype = None

    def __call__(self):
        return self.lib.current


class _FakeLib:
    def __init__(self):
        self.current = 16
        self.calls = []
        self.omp_get_max_threads = _FakeGetMaxThreads(self)

    def omp_set_num_threads(self, n):
        self.calls.append(int(n))
        self.current = int(n)


def _patch_cdll(monkeypatch_holder, lib):
    original = ctypes.CDLL

    def fake_cdll(name):
        if name == "libgomp.so.1":
            return lib
        return original(name)

    ctypes.CDLL = fake_cdll
    monkeypatch_holder.append(original)


def _unpatch(monkeypatch_holder):
    ctypes.CDLL = monkeypatch_holder.pop()


def test_pins_to_one_and_restores():
    lib = _FakeLib()
    holder = []
    _patch_cdll(holder, lib)
    try:
        assert lib.current == 16
        with limits.single_threaded():
            assert lib.current == 1
        assert lib.current == 16
        assert lib.calls == [1, 16]
    finally:
        _unpatch(holder)


def test_restores_even_on_exception():
    lib = _FakeLib()
    lib.current = 8
    holder = []
    _patch_cdll(holder, lib)
    try:
        try:
            with limits.single_threaded():
                assert lib.current == 1
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert lib.current == 8, "must restore the ORIGINAL count, not a hardcoded one"
    finally:
        _unpatch(holder)


def test_no_op_when_libgomp_unreachable():
    holder = []
    original = ctypes.CDLL

    def fake_cdll(name):
        raise OSError("no such library")

    ctypes.CDLL = fake_cdll
    holder.append(original)
    try:
        # Must not raise. The whole point is that a missing libgomp degrades
        # to "plane fitting keeps whatever thread count it already had",
        # not to a crash.
        with limits.single_threaded():
            pass
    finally:
        _unpatch(holder)


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
