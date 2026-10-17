"""The offline export (tools/build_bench.py build): a self-contained page with every preview baked in, in the flat
coordinate format the page reads."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import build_bench  # noqa: E402


def _write_room(path: str, seed: int) -> int:
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
    return len(pts)


def test_the_offline_file_carries_flat_finite_coordinates_the_page_reads():
    with tempfile.TemporaryDirectory() as folder:
        scans = os.path.join(folder, "scans")
        os.makedirs(scans)
        counts = [_write_room(os.path.join(scans, "room a.pts"), 1), _write_room(os.path.join(scans, "room b.pts"), 2)]
        out = os.path.join(folder, "offline.html")
        build_bench.build(scans, out)
        with open(out, encoding="utf-8") as fh:
            html = fh.read()
    start = html.index("window.SCANDATA=") + len("window.SCANDATA=")
    data = json.loads(html[start:html.index(";</script>", start)])
    assert [s["name"] for s in data["scans"]] == ["room a", "room b"], [s["name"] for s in data["scans"]]
    for s, n in zip(data["scans"], counts):
        xy, h = np.asarray(s["xy"], dtype=float), np.asarray(s["h"], dtype=float)
        assert xy.ndim == 1 and xy.size == 2 * h.size, f"xy must be one flat list of numbers, two per point ({xy.shape} vs {h.shape})"
        assert np.isfinite(xy).all() and np.isfinite(h).all()
        assert s["ready"] is True and s["points"] == n, (s["points"], n)
    assert "standalone (offline export" in html, "the build badge should say this is an offline export"


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
