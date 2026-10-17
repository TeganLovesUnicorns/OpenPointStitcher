"""Shared by the test files: the runner every file ends with, so a test file is run the same way everywhere."""
from __future__ import annotations

from typing import Callable, Optional


def run_tests(namespace: dict, part: Optional[tuple[int, int]] = None, after: Optional[Callable] = None) -> int:
    """Run every test_* function in namespace (a module's globals()), in name order, one PASS or FAIL line each;
    one failing test never stops the rest. part=(k, n) runs every n-th test starting at k. after() runs at the end
    even if the run is interrupted. Returns the exit code: 1 if anything failed."""
    names = sorted(name for name, fn in namespace.items() if name.startswith("test_") and callable(fn))
    if part is not None:
        names = names[part[0]::part[1]]
    failures = 0
    try:
        for name in names:
            try:
                namespace[name]()
                print(f"PASS {name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {str(exc) or repr(exc)}", flush=True)
    finally:
        if after is not None:
            after()
    return 1 if failures else 0
