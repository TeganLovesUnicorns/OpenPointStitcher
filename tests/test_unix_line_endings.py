"""Every shell script the container runs has Unix line endings: a Windows ending on the first line stops the
container starting ("no such file or directory"). Checks the copies inside the image, which is what runs."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
SHEBANG_SHELLS = (b"#!/bin/sh", b"#!/bin/bash", b"#!/usr/bin/env sh", b"#!/usr/bin/env bash")


def shell_scripts(roots: list[str]) -> list[str]:
    """Files ending .sh, or starting with a shell #! line, under the given folders (or the files themselves)."""
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(root)
            continue
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", "fixtures")]
            for f in files:
                p = os.path.join(base, f)
                if f.endswith(".sh"):
                    found.append(p)
                    continue
                try:
                    with open(p, "rb") as fh:
                        if fh.read(24).startswith(SHEBANG_SHELLS):
                            found.append(p)
                except OSError:
                    pass
    return found


def windows_endings(paths: list[str]) -> list[str]:
    bad = []
    for p in paths:
        with open(p, "rb") as fh:
            if b"\r" in fh.read():
                bad.append(p)
    return bad


def _in_image_roots() -> list[str]:
    roots = [os.path.join(APP, d) for d in ("tools", "open_point_stitch", "tests")]
    return roots + [p for p in ("/usr/local/bin/docker-entrypoint.sh",) if os.path.exists(p)]


def test_every_shell_script_in_the_image_has_unix_endings():
    if not os.path.exists("/usr/local/bin/docker-entrypoint.sh"):
        print("SKIP: not inside the image (on Windows the checkout has Windows endings by design)")
        return
    scripts = shell_scripts(_in_image_roots())
    assert "/usr/local/bin/docker-entrypoint.sh" in scripts, scripts
    bad = windows_endings(scripts)
    assert not bad, f"Windows line endings in {bad}: the container cannot run these"


def test_the_check_finds_a_windows_ending():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good, bad = os.path.join(d, "good.sh"), os.path.join(d, "run")
        with open(good, "wb") as fh:
            fh.write(b"#!/bin/sh\necho ok\n")
        with open(bad, "wb") as fh:
            fh.write(b"#!/bin/sh\r\necho ok\r\n")
        assert sorted(shell_scripts([d])) == sorted([good, bad])
        assert windows_endings([good, bad]) == [bad]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
