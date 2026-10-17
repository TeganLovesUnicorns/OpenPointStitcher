"""Run every n-th test of one test file (part k of n), one PASS/FAIL line per test, so tools/qa.py can split a long
file across lanes and add the parts up against what the file declares.
Usage: python tools/qa_run_part.py tests/test_x.py k n"""
from __future__ import annotations

import importlib.util
import os
import sys


def main() -> int:
    path, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("part_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                       # the file's own runner sits under __main__ and stays idle
    from support import run_tests
    return run_tests(vars(module), part=(k, n))


if __name__ == "__main__":
    raise SystemExit(main())
