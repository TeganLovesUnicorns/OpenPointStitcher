"""Round-trip and tolerance tests for the .pts reader and writer. Run with: python tests/test_ptsio.py"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.pts_file import read_pts, write_pts  # noqa: E402


def _tmp(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(), name)


def test_roundtrip_with_color():
    pts = np.array([[0.0, 1.0, 2.0], [-3.5, 4.25, 0.125]])
    cols = np.array([[1.0, 0.0, 0.5], [0.25, 0.75, 0.0]])
    p = _tmp("rt.pts")
    write_pts(p, pts, cols)
    scan = read_pts(p)
    assert scan.n_points == 2
    assert scan.has_color
    assert np.allclose(scan.points, pts, atol=1e-4)
    assert np.allclose(scan.colors, np.rint(cols * 255) / 255.0, atol=1e-3)
    # Intensity was written as -1 throughout, so it should be dropped on read.
    assert scan.intensity is None
    assert scan.header_count == 2


def test_xyz_only():
    p = _tmp("xyz.pts")
    with open(p, "w") as fh:
        fh.write("2\n0 0 0\n1 2 3\n")
    scan = read_pts(p)
    assert scan.n_points == 2
    assert not scan.has_color


def test_no_header_line():
    p = _tmp("nohdr.pts")
    with open(p, "w") as fh:
        fh.write("0 0 0 -1 10 20 30\n1 2 3 -1 40 50 60\n")
    scan = read_pts(p)
    assert scan.n_points == 2
    assert scan.header_count is None
    assert scan.has_color


def test_leading_geodata_comment_is_skipped():
    p = _tmp("geo.pts")
    with open(p, "w") as fh:
        fh.write(';{"GeoData":{"Lat":45.0265,"Lon":-93.1478}}\n2\n0 0 0 -1 10 20 30\n1 2 3 -1 40 50 60\n')
    scan = read_pts(p)
    assert scan.n_points == 2
    assert scan.header_count == 2
    assert scan.has_color


def test_wrong_header_count_is_tolerated():
    p = _tmp("badhdr.pts")
    with open(p, "w") as fh:
        fh.write("99\n0 0 0\n1 2 3\n")
    scan = read_pts(p)
    assert scan.n_points == 2
    assert scan.header_count == 99


def test_real_intensity_is_kept():
    p = _tmp("inten.pts")
    with open(p, "w") as fh:
        fh.write("2\n0 0 0 0.5 10 20 30\n1 2 3 0.9 40 50 60\n")
    scan = read_pts(p)
    assert scan.intensity is not None
    assert np.allclose(scan.intensity, [0.5, 0.9])


def test_non_finite_rows_dropped():
    p = _tmp("nan.pts")
    with open(p, "w") as fh:
        fh.write("3\n0 0 0\nnan 1 2\n1 2 3\n")
    scan = read_pts(p)
    assert scan.n_points == 2


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
