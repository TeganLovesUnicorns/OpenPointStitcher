"""Footprint outline from an occupancy grid, optionally snapped to the room's dominant wall directions (off by
default)."""
import numpy as np, cv2


def _dominant_directions(edges, tol_deg=12.0, max_dirs=4):
    """Angles (radians, mod pi) that the room's edges actually run along."""
    angs, wts = [], []
    for (a, b) in edges:
        v = b - a
        L = float(np.hypot(*v))
        if L < 0.25:
            continue
        angs.append(np.arctan2(v[1], v[0]) % np.pi)
        wts.append(L)
    if not angs:
        return []
    angs = np.array(angs); wts = np.array(wts)
    dirs = []
    remaining = np.ones(len(angs), bool)
    for _ in range(max_dirs):
        if not remaining.any():
            break
        # weight each candidate by the total edge length agreeing with it
        best, bestw = None, -1
        for a in angs[remaining]:
            d = np.abs((angs - a + np.pi / 2) % np.pi - np.pi / 2)
            w = wts[(d < np.radians(tol_deg)) & remaining].sum()
            if w > bestw:
                bestw, best = w, a
        d = np.abs((angs - best + np.pi / 2) % np.pi - np.pi / 2)
        grp = (d < np.radians(tol_deg)) & remaining
        if not grp.any():
            break
        # circular-safe mean of the group
        m = np.arctan2((np.sin(2 * angs[grp]) * wts[grp]).sum(),
                       (np.cos(2 * angs[grp]) * wts[grp]).sum()) / 2 % np.pi
        dirs.append((m, float(wts[grp].sum())))
        remaining &= ~grp
    total = sum(w for _, w in dirs)
    return [a for a, w in dirs if w > total * 0.06]


def _snap_polygon(poly, dirs, tol_deg=18.0, keep=None):
    """Force each edge onto a dominant direction, then re-intersect."""
    n = len(poly)
    if n < 3 or not dirs:
        return poly
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        v = b - a
        L = float(np.hypot(*v))
        if L < 1e-9:
            lines.append(None); continue
        # never straighten an edge that is part of a genuine curve
        if keep is not None and (keep[i] or keep[(i + 1) % n]):
            lines.append(None); continue
        if L < 0.8:
            lines.append(None); continue
        ang = np.arctan2(v[1], v[0]) % np.pi
        diffs = [abs((ang - d + np.pi / 2) % np.pi - np.pi / 2) for d in dirs]
        k = int(np.argmin(diffs))
        if np.degrees(diffs[k]) > tol_deg:
            lines.append(None)            # genuinely angled, leave it alone
            continue
        u = np.array([np.cos(dirs[k]), np.sin(dirs[k])])
        mid = (a + b) / 2
        lines.append((u, mid))

    out = []
    for i in range(n):
        p, q = lines[i - 1], lines[i]
        if p is None or q is None:
            out.append(poly[i]); continue
        (u1, c1), (u2, c2) = p, q
        A = np.array([[u1[0], -u2[0]], [u1[1], -u2[1]]])
        if abs(np.linalg.det(A)) < 1e-6:
            out.append(poly[i]); continue
        t = np.linalg.solve(A, c2 - c1)
        pt = c1 + u1 * t[0]
        # refuse silly extrapolations
        out.append(pt if np.hypot(*(pt - poly[i])) < 1.5 else poly[i])
    return np.array(out)


def _resample_smooth(c, res, step_m=0.10, smooth_m=0.45):
    """Even out and gently smooth a traced contour: noise goes, curvature stays."""
    p = c.reshape(-1, 2).astype(float)
    if len(p) < 8:
        return p
    seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    per = seg.sum()
    if per < 1e-9:
        return p
    n = max(16, int(per * res / step_m))
    t = np.concatenate([[0.0], np.cumsum(seg)]) / per
    tt = np.linspace(0, 1, n, endpoint=False)
    q = np.column_stack([np.interp(tt, t, np.vstack([p, p[:1]])[:, k]) for k in (0, 1)])
    w = max(3, int(round(smooth_m / step_m)) | 1)
    k = np.ones(w) / w
    pad = np.vstack([q[-w:], q, q[:w]])
    sm = np.column_stack([np.convolve(pad[:, i], k, mode="same") for i in (0, 1)])
    return sm[w:-w]


def _turn_angles(p):
    """Signed turn at each vertex, radians."""
    a = np.roll(p, 1, axis=0); b = p; c = np.roll(p, -1, axis=0)
    v1 = b - a; v2 = c - b
    return np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0],
                      v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1])


def _curved_mask(simple, dense, window_m=1.2, total_turn_deg=22.0):
    """True for simplified vertices inside a smooth arc, measured on the dense contour (never the simplified polygon):
    plenty of total turn, no single step dominating."""
    if dense is None or len(dense) < 8 or len(simple) < 4:
        return np.zeros(len(simple), bool)
    seg = np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1)
    step = float(np.median(seg)) or 0.1
    half = max(2, int(window_m / step / 2))
    t = _turn_angles(dense)
    n = len(dense)

    out = np.zeros(len(simple), bool)
    for k, v in enumerate(simple):
        j = int(np.argmin(np.sum((dense - v) ** 2, axis=1)))
        idx = [(j + m) % n for m in range(-half, half + 1)]
        turns = t[idx]
        total = float(np.abs(np.degrees(turns).sum()))
        biggest = float(np.max(np.abs(np.degrees(turns))))
        same = np.abs(np.sign(turns).sum()) > len(turns) * 0.6
        if total > total_turn_deg and biggest < total * 0.5 and same:
            out[k] = True
    return out


def footprint(xy, res=0.06, close_m=0.30, simplify_m=0.14,
              min_area_frac=0.04, snap=False, max_corners=None):
    """Return (list of polygons in metres, total area in m2). snap squares edges to dominant wall directions; keep it
    off for anything that will be drawn from, since it straightens real curved walls."""
    lo = xy.min(axis=0) - res * 6
    ij = np.floor((xy - lo) / res).astype(int)
    W, H = ij[:, 0].max() + 7, ij[:, 1].max() + 7
    g = np.zeros((H, W), np.uint8)
    g[ij[:, 1], ij[:, 0]] = 255

    k = max(3, int(round(close_m / res)) | 1)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], 0.0
    total = sum(cv2.contourArea(c) for c in cnts)
    polys = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < total * min_area_frac:
            continue
        sm = _resample_smooth(c, res)
        a = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2),
                             simplify_m / res, True).reshape(-1, 2).astype(float)
        if max_corners:
            eps = simplify_m / res
            while len(a) > max_corners and eps < 40:
                eps *= 1.3
                a = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2),
                                     eps, True).reshape(-1, 2).astype(float)
        p = a * res + lo
        if snap and len(p) >= 5:
            dense = sm * res + lo
            edges = [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
            p = _snap_polygon(p, _dominant_directions(edges),
                              keep=_curved_mask(p, dense))
        polys.append(p)
    return polys, float(total * res * res)
