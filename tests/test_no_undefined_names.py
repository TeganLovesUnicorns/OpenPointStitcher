"""No Python file uses a name that is not defined, so a removed line that something still needed fails here in
seconds. Needs pyflakes (the QA image)."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def test_no_python_file_uses_an_undefined_name():
    try:
        import pyflakes  # noqa: F401
    except ImportError:
        print("    SKIP: pyflakes is not installed (it is in the QA image)")
        return
    out = subprocess.run([sys.executable, "-m", "pyflakes", "open_point_stitch", "tools", "tests"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    bad = [line for line in out.splitlines() if "undefined name" in line or "undefined local" in line]
    assert not bad, "names used but never defined:\n" + "\n".join(bad)
    print(f"    no undefined names ({len(out.splitlines())} other pyflakes notes, not checked here)")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
