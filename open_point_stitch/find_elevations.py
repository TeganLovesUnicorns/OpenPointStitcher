"""Wall elevations: a wall viewed straight on is an image with colour and texture, unlike a room viewed from above.
Two scans' images of the same wall on one shared grid differ only by a plain shift, and matching features between
them gives that shift along the wall and in height."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Elevation:
    image: np.ndarray         # uint8, rows run from the top height downwards, columns along the wall
    covered: np.ndarray       # bool, pixels that have at least one point
    a0: float                 # along-wall coordinate of the left edge of column 0
    h1: float                 # height of the top edge of row 0
    res: float                # metres per pixel
    supported: Optional[np.ndarray] = None   # bool, pixels close enough to points to trust (gaps between points are filled in)


@dataclass
class ElevationShift:
    along_m: float            # how far to move the SECOND scan along the wall to line up with the first
    up_m: float               # and upward
    n_inliers: int
    n_matches: int


def render_elevation(
    points: np.ndarray, colors: np.ndarray, normal: np.ndarray, offset: float,
    a_range: tuple[float, float], h_range: tuple[float, float], res: float = 0.03, slab: float = 0.10,
) -> Optional[Elevation]:
    """A grey image of one wall: the points within `slab` of the plane `normal . xy = offset` (`points` are x, y, z with z
    up, in a frame both scans share), drawn by position along the wall and height. Lighting gradients are taken out so two
    scans lit differently still agree. None if too little of the wall was seen."""
    normal = np.asarray(normal, dtype=float)
    across = points[:, :2] @ normal - offset
    near = np.abs(across) < slab
    if near.sum() < 200:
        return None
    along = points[near, :2] @ np.array([-normal[1], normal[0]])
    height = points[near, 2]
    gray = np.asarray(colors)[near] @ np.array([0.299, 0.587, 0.114])
    a0, a1 = a_range
    h0, h1 = h_range
    cols, rows = int(np.ceil((a1 - a0) / res)), int(np.ceil((h1 - h0) / res))
    if cols < 20 or rows < 20:
        return None
    ci = np.floor((along - a0) / res).astype(int)
    ri = np.floor((h1 - height) / res).astype(int)
    inside = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows)
    flat = ri[inside] * cols + ci[inside]
    count = np.bincount(flat, minlength=rows * cols).reshape(rows, cols)
    total = np.bincount(flat, weights=gray[inside], minlength=rows * cols).reshape(rows, cols)
    covered = count > 0
    if covered.mean() < 0.15:
        return None
    img = np.where(covered, total / np.maximum(count, 1), 0.0).astype(np.float32)
    # Fill the gaps between points from their neighbours (a flat fill would read as texture to the feature finder).
    seen = covered.astype(np.float32)
    weight = cv2.GaussianBlur(seen, (0, 0), 2.0)
    smooth = cv2.GaussianBlur(img, (0, 0), 2.0) / np.maximum(weight, 1e-3)
    img = np.where(covered, img, np.where(weight > 0.02, smooth, img[covered].mean())).astype(np.float32)
    # Take out slow lighting changes: subtract a heavy blur, so what is left is the wall's own texture.
    detail = img - cv2.GaussianBlur(img, (0, 0), 15)
    out = np.clip(128.0 + detail * 255.0 * 2.5, 0, 255).astype(np.uint8)
    return Elevation(image=out, covered=covered, a0=float(a0), h1=float(h1), res=float(res), supported=weight > 0.25)


def match_elevations(a: Elevation, b: Elevation, ratio: float = 0.8, min_inliers: int = 8, tol_px: float = 2.5) -> Optional[ElevationShift]:
    """The shift that lines b's image of the wall up with a's: SIFT features in both, matched, and the one shift most
    matches agree on. None if too few agree (a blank wall, or two different walls)."""
    if abs(a.res - b.res) > 1e-9:
        raise ValueError("both elevations must use the same pixel size")
    detector = cv2.SIFT_create()

    def features(e: Elevation):
        trusted = e.supported if e.supported is not None else e.covered
        mask = cv2.erode(trusted.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        return detector.detectAndCompute(e.image, mask)

    ka, da = features(a)
    kb, db = features(b)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < min_inliers:
        return None
    # Each match says "move b this far to sit on a"; the answer is the shift most of them agree on.
    pa = np.array([ka[m.trainIdx].pt for m in good])
    pb = np.array([kb[m.queryIdx].pt for m in good])
    along = (a.a0 + pa[:, 0] * a.res) - (b.a0 + pb[:, 0] * b.res)
    up = (a.h1 - pa[:, 1] * a.res) - (b.h1 - pb[:, 1] * b.res)
    votes = np.column_stack([along, up])
    tol = tol_px * a.res
    best = None
    for v in votes:
        agree = np.linalg.norm(votes - v, axis=1) <= tol
        if best is None or agree.sum() > best.sum():
            best = agree
    if best.sum() < min_inliers:
        return None
    shift = votes[best].mean(axis=0)
    return ElevationShift(along_m=float(shift[0]), up_m=float(shift[1]), n_inliers=int(best.sum()), n_matches=len(good))
