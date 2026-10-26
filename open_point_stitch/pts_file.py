"""Reader and writer for .pts point clouds.

Line 1 is a point count (a hint only), then "X Y Z I R G B" lines. Tolerates 3, 4, 6 and 7 columns,
and leading ";" or "#" comment lines."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .lib.clouds import as_cloud

log = logging.getLogger(__name__)

# Intensity values written when the field is unpopulated.
_UNSET_INTENSITY = -1.0

# Lines starting with these are skipped wherever they appear.
_COMMENT_PREFIXES = (";", "#")


@dataclass
class Scan:
    """One loaded scan, plus the metadata we want to carry through the pipeline."""

    name: str
    path: str
    points: np.ndarray                      # (N, 3) float64, metres
    colors: Optional[np.ndarray] = None     # (N, 3) float64 in 0..1, or None
    intensity: Optional[np.ndarray] = None  # (N,) float64, or None
    header_count: Optional[int] = None      # what line 1 claimed

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def has_color(self) -> bool:
        return self.colors is not None

    def to_o3d(self) -> o3d.geometry.PointCloud:
        return as_cloud(self.points, self.colors)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def describe(self) -> str:
        lo, hi = self.bounds()
        span = hi - lo
        return (
            f"{self.name}: {self.n_points:,} pts, "
            f"colour={'yes' if self.has_color else 'no'}, "
            f"extent {span[0]:.2f} x {span[1]:.2f} x {span[2]:.2f} m"
        )


def _looks_like_count_header(first_line: str) -> Optional[int]:
    """Return the integer point count if line 1 is a bare count, else None."""
    tokens = first_line.split()
    if len(tokens) != 1:
        return None
    try:
        value = int(tokens[0])
    except ValueError:
        return None
    return value if value >= 0 else None


def _first_content_line(fh) -> tuple[int, str]:
    """(offset, text) of the first line that is not a comment; text is "" at end of file."""
    while True:
        start = fh.tell()
        line = fh.readline()
        if not line.lstrip().startswith(_COMMENT_PREFIXES):
            return start, line


def quick_point_count(path: str) -> int:
    """The header's point count, untrusted and without reading the points; 0 if there is none or it cannot be read."""
    try:
        with open(path, "r", errors="replace") as fh:
            return _looks_like_count_header(_first_content_line(fh)[1]) or 0
    except OSError:
        return 0


def read_pts(path: str, name: Optional[str] = None) -> Scan:
    """Load a single .pts file into a Scan.

    Raises ValueError if the file has no parseable point rows.
    """
    with open(path, "r", errors="replace") as fh:
        # Some exporters open with comment lines, e.g. ';{"GeoData":{...}}' ahead of the count.
        start, first_line = _first_content_line(fh)
        header_count = _looks_like_count_header(first_line)
        # If that line was not a count it is a real data row, so rewind to it.
        if header_count is None:
            fh.seek(start)
        try:
            data = np.loadtxt(fh, dtype=np.float64, ndmin=2, comments=_COMMENT_PREFIXES)
        except ValueError as exc:
            raise ValueError(f"{path}: could not parse point rows ({exc})") from exc

    if data.size == 0:
        raise ValueError(f"{path}: file contains no point data")

    ncols = data.shape[1]
    if ncols < 3:
        raise ValueError(f"{path}: expected at least 3 columns, found {ncols}")

    points = np.ascontiguousarray(data[:, 0:3])
    intensity = None
    colors = None

    if ncols == 3:
        pass
    elif ncols == 4:
        intensity = np.ascontiguousarray(data[:, 3])
    elif ncols == 6:
        colors = data[:, 3:6]
    elif ncols >= 7:
        intensity = np.ascontiguousarray(data[:, 3])
        colors = data[:, 4:7]
        if ncols > 7:
            log.warning("%s: ignoring %d extra column(s)", path, ncols - 7)
    else:
        raise ValueError(f"{path}: unsupported column count {ncols}")

    if colors is not None:
        # 0-255 integers in the file; Open3D wants 0-1 floats.
        colors = np.clip(np.ascontiguousarray(colors) / 255.0, 0.0, 1.0)

    if intensity is not None and np.all(intensity == _UNSET_INTENSITY):
        # Unpopulated field: drop it rather than carry a column of -1 around.
        intensity = None

    if header_count is not None and header_count != points.shape[0]:
        log.warning(
            "%s: header claims %d points, parsed %d. Using the parsed count.",
            path, header_count, points.shape[0],
        )

    # Drop NaN/inf rows in XYZ, colour or intensity; they break normal estimation downstream.
    finite = np.isfinite(points).all(axis=1)
    if colors is not None:
        finite &= np.isfinite(colors).all(axis=1)
    if intensity is not None:
        finite &= np.isfinite(intensity)
    if not finite.all():
        dropped = int((~finite).sum())
        log.warning("%s: dropping %d non-finite point(s)", path, dropped)
        points = points[finite]
        if colors is not None:
            colors = colors[finite]
        if intensity is not None:
            intensity = intensity[finite]

    if points.shape[0] == 0:
        raise ValueError(f"{path}: no finite points remained after cleaning")

    return Scan(
        name=name or os.path.splitext(os.path.basename(path))[0],
        path=os.path.abspath(path),
        points=points,
        colors=colors,
        intensity=intensity,
        header_count=header_count,
    )


def write_pts(
    path: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    precision: int = 4,
) -> None:
    """Write points to .pts: count header, 7 columns when colour is present (intensity -1 if none)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = points.shape[0]

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if colors.shape[0] != n:
            raise ValueError("colors length does not match points length")
        rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.int32)
    else:
        rgb = None

    if intensity is None:
        inten = np.full(n, _UNSET_INTENSITY)
    else:
        inten = np.asarray(intensity, dtype=np.float64).reshape(-1)
        if inten.shape[0] != n:
            raise ValueError("intensity length does not match points length")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    xyz_fmt = f"%.{precision}f"
    with open(path, "w", newline="\n") as fh:
        fh.write(f"{n}\n")
        if rgb is None:
            block = np.column_stack([points, inten])
            np.savetxt(fh, block, fmt=[xyz_fmt, xyz_fmt, xyz_fmt, "%g"], delimiter=" ")
        else:
            block = np.column_stack([points, inten, rgb.astype(np.float64)])
            np.savetxt(
                fh, block,
                fmt=[xyz_fmt, xyz_fmt, xyz_fmt, "%g", "%d", "%d", "%d"],
                delimiter=" ",
            )


def find_scans(folder: str, pattern: str = "*.pts") -> list[str]:
    """Return sorted .pts paths in a folder. Sorting keeps runs reproducible."""
    import glob
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    return paths


def load_folder(folder: str, pattern: str = "*.pts") -> list[Scan]:
    paths = find_scans(folder, pattern)
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    scans: list[Scan] = []
    for p in paths:
        scans.append(read_pts(p))
    return scans

