"""Tests for find_elevations.py: a wall drawn as an image from two scans, and the shift between the two images.
Synthetic textured walls with a known shift."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.find_elevations import Elevation, match_elevations, render_elevation  # noqa: E402

NORMAL, OFFSET = np.array([1.0, 0.0]), 0.0          # the wall is the plane x = 0; along it runs y
A_RANGE, H_RANGE = (-1.0, 9.0), (-0.5, 4.0)


def _texture(seed: int):
    """A wall's colour as a function of (along, height): soft blobs and a few sharp-edged panels, like real paint and posters."""
    rng = np.random.default_rng(seed)
    blobs = [(rng.uniform(0, 8), rng.uniform(0, 3), rng.uniform(0.05, 0.3), rng.uniform(-0.4, 0.4)) for _ in range(90)]
    panels = [(rng.uniform(0, 7.5), rng.uniform(0, 2.5), rng.uniform(0.2, 0.7), rng.uniform(0.2, 0.7), rng.uniform(-0.3, 0.3)) for _ in range(25)]

    def gray(a, h):
        g = np.full(a.shape, 0.5)
        for ca, ch, s, amp in blobs:
            g += amp * np.exp(-((a - ca) ** 2 + (h - ch) ** 2) / (2 * s * s))
        for pa, ph, w, hh, amp in panels:
            g += amp * ((a >= pa) & (a < pa + w) & (h >= ph) & (h < ph + hh))
        return np.clip(g, 0, 1)
    return gray


def _wall(seed=1, along_shift=0.0, up_shift=0.0, keep=None, light=(1.0, 0.0), step=0.02, rng_seed=0):
    """Points and colours of the textured wall as one scan saw it, its whole picture displaced by the given shift."""
    a, h = np.meshgrid(np.arange(0.0, 8.0, step), np.arange(0.0, 3.0, step))
    a, h = a.ravel(), h.ravel()
    g = _texture(seed)(a, h) * light[0] + light[1]
    rng = np.random.default_rng(rng_seed)
    pts = np.column_stack([rng.normal(0, 0.002, a.size), a + along_shift, h + up_shift])
    if keep is not None:
        sel = keep(a, h)
        pts, g = pts[sel], g[sel]
    return pts, np.repeat(np.clip(g, 0, 1)[:, None], 3, axis=1)


def _elev(points, colors):
    return render_elevation(points, colors, NORMAL, OFFSET, A_RANGE, H_RANGE)


def test_a_known_shift_between_two_pictures_of_the_same_wall_is_recovered():
    a = _elev(*_wall())
    b = _elev(*_wall(along_shift=0.42, up_shift=0.27))
    m = match_elevations(a, b)
    assert m is not None, "no shift found"
    assert abs(m.along_m + 0.42) < 0.04 and abs(m.up_m + 0.27) < 0.04, (m.along_m, m.up_m)
    assert m.n_inliers >= 20, m.n_inliers


def test_the_shift_is_found_when_the_second_scan_is_lit_differently():
    a = _elev(*_wall())
    b = _elev(*_wall(along_shift=-0.6, up_shift=0.15, light=(0.55, 0.18)))
    m = match_elevations(a, b)
    assert m is not None and abs(m.along_m - 0.6) < 0.04 and abs(m.up_m + 0.15) < 0.04, m


def test_the_shift_is_found_when_the_second_scan_saw_only_part_of_the_wall_and_sparsely():
    a = _elev(*_wall())
    pts, col = _wall(along_shift=0.3, up_shift=-0.2, keep=lambda a_, h_: a_ > 3.0, step=0.035, rng_seed=4)
    m = match_elevations(a, _elev(pts, col))
    assert m is not None and abs(m.along_m + 0.3) < 0.05 and abs(m.up_m - 0.2) < 0.05, m


def test_a_blank_wall_and_two_different_walls_give_no_shift():
    blank = _elev(*_wall(seed=3, light=(0.0, 0.5)))
    assert blank is None or match_elevations(blank, blank) is None
    assert match_elevations(_elev(*_wall(seed=1)), _elev(*_wall(seed=2, along_shift=0.3))) is None


def test_too_little_wall_seen_is_not_drawn_and_mixed_pixel_sizes_are_refused():
    pts, col = _wall()
    assert render_elevation(pts[:100], col[:100], NORMAL, OFFSET, A_RANGE, H_RANGE) is None
    far = pts + np.array([5.0, 0.0, 0.0])                                    # nothing near the plane
    assert render_elevation(far, col, NORMAL, OFFSET, A_RANGE, H_RANGE) is None
    a = _elev(pts, col)
    b = Elevation(image=a.image, covered=a.covered, a0=a.a0, h1=a.h1, res=a.res * 2)
    try:
        match_elevations(a, b)
    except ValueError:
        return
    raise AssertionError("two pixel sizes must be refused")


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
