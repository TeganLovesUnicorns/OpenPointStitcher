"""A generated family flat scanned room by room the way a phone scans it: the browser QA's scans, the same on any
machine, so the bench needs no real scans to be tested.

An L-shaped flat after an architect's cut-away drawing: a living room with a brown accent wall, a green sofa and a rug,
opening onto a decked terrace with a garden 1.2 m below it down a short stair; a hall and an entry with a black and
white diamond rug; a main bedroom with an olive wall, a guest bedroom, a tiled bathroom, a laundry, a yellow kids'
room, and a kitchen-diner. Each room has its own paint, floor (planks, tiles, decking, grass) and furniture. Walls hide
what is behind them except through doorways and windows, so each scan sees its own room and a little through each
opening, as a real one does, and neighbouring scans overlap where a walk steps through a door. How a phone scans
(density, noise, drift, strays) is synthetic_venue's. Same seed, same files."""
from __future__ import annotations

import os

import numpy as np

from synthetic_venue import _Surface, _Walk, _scan, _transform

H = 2.6                 # ceiling height
DOOR_H = 2.1
SILL, HEAD = 0.9, 2.1   # windows
WALL_T = 0.1
GARDEN_Z = -1.2         # the garden lies this far below the flat's floor and terrace

WHITE, CREAM, BEIGE = (0.93, 0.92, 0.89), (0.90, 0.86, 0.78), (0.86, 0.80, 0.70)
OUTSIDE = (0.95, 0.94, 0.92)
TRIM = (0.88, 0.87, 0.84)


# ---- patterns: venue points (n x 3) -> colours (n x 3) ----

def _hash(i, j):
    return np.modf(np.abs(np.sin(i * 12.9898 + j * 78.233) * 43758.5453))[0]


def _planks(colour, width=0.16, length=1.4, along_x=True, spread=0.15):
    c = np.asarray(colour, float)

    def f(p):
        u, v = (p[:, 0], p[:, 1]) if along_x else (p[:, 1], p[:, 0])
        row = np.floor(v / width)
        board = np.floor(u / length + row * 0.37)                # staggered butt joints
        shade = 1 + spread * (2 * _hash(row, board) - 1) + 0.04 * np.sin(u * 35 + row * 2.1)
        seam = np.abs(v / width - np.round(v / width)) * width < 0.004
        return c[None, :] * (shade * np.where(seam, 0.55, 1.0))[:, None]
    return f


def _tiles(colour, grout, size=0.3, vertical=False):
    c, g = np.asarray(colour, float), np.asarray(grout, float)

    def f(p):
        u, v = (p[:, 0] + p[:, 1], p[:, 2]) if vertical else (p[:, 0], p[:, 1])
        line = (np.abs(u / size - np.round(u / size)) * size < 0.005) | (np.abs(v / size - np.round(v / size)) * size < 0.005)
        shade = 1 + 0.05 * (2 * _hash(np.floor(u / size), np.floor(v / size)) - 1)
        return np.where(line[:, None], g[None, :], c[None, :] * shade[:, None])
    return f


def _diamonds(a, b, size=0.3):
    a, b = np.asarray(a, float), np.asarray(b, float)

    def f(p):
        odd = (np.floor((p[:, 0] + p[:, 1]) / size) + np.floor((p[:, 0] - p[:, 1]) / size)) % 2 == 1
        return np.where(odd[:, None], a[None, :], b[None, :])
    return f


def _checks(a, b, size=0.25):
    a, b = np.asarray(a, float), np.asarray(b, float)

    def f(p):
        odd = (np.floor(p[:, 0] / size) + np.floor(p[:, 1] / size)) % 2 == 1
        return np.where(odd[:, None], a[None, :], b[None, :])
    return f


def _rug(inner, border, x0, x1, y0, y1, w=0.15):
    a, b = np.asarray(inner, float), np.asarray(border, float)

    def f(p):
        edge = (p[:, 0] < x0 + w) | (p[:, 0] > x1 - w) | (p[:, 1] < y0 + w) | (p[:, 1] > y1 - w)
        return np.where(edge[:, None], b[None, :], a[None, :])
    return f


def _speckle(colour, spread=0.25, cell=0.06):
    """Grass and foliage: every few centimetres a different shade, greener or browner."""
    c = np.asarray(colour, float)

    def f(p):
        i, j = np.floor(p[:, 0] / cell) + np.floor(p[:, 2] / cell) * 7, np.floor(p[:, 1] / cell)
        h1, h2 = _hash(i, j), _hash(j, i)
        out = c[None, :] * (1 + spread * (2 * h1 - 1))[:, None]
        out[:, 0] += 0.12 * (h2 - 0.5)
        return out
    return f


def _flowers(base, cell=0.22, share=0.45):
    colours = np.array([(0.92, 0.35, 0.55), (0.98, 0.62, 0.15), (0.45, 0.45, 0.95), (0.95, 0.90, 0.30)])
    leaves = _speckle(base)

    def f(p):
        i, j = np.floor(p[:, 0] / cell), np.floor(p[:, 1] / cell)
        bloom = _hash(i, j) < share
        out = leaves(p)
        out[bloom] = colours[(_hash(j, i)[bloom] * len(colours)).astype(int) % len(colours)]
        return out
    return f


# ---- shapes ----

def _rect(o, e1, e2, colour, holes=(), edgy=False) -> _Surface:
    pattern = colour if callable(colour) else None
    return _Surface(np.asarray(o, float), np.asarray(e1, float), np.asarray(e2, float),
                    (0.5, 0.5, 0.5) if pattern else colour, holes, 0.0, edgy, pattern)


def _floor(x0, x1, y0, y1, z, colour) -> _Surface:
    return _rect((x0, y0, z), (x1 - x0, 0, 0), (0, y1 - y0, 0), colour)


def _box(x0, x1, y0, y1, z0, z1, colour, top=None, faces="tnsew") -> list[_Surface]:
    """A piece of furniture: the listed faces (t(op), n/s the y1/y0 sides, e/w the x1/x0 sides); top may differ."""
    out = []
    if "t" in faces:
        out.append(_rect((x0, y0, z1), (x1 - x0, 0, 0), (0, y1 - y0, 0), top or colour, edgy=True))
    if "s" in faces:
        out.append(_rect((x0, y0, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0), colour, edgy=True))
    if "n" in faces:
        out.append(_rect((x0, y1, z0), (x1 - x0, 0, 0), (0, 0, z1 - z0), colour, edgy=True))
    if "w" in faces:
        out.append(_rect((x0, y0, z0), (0, y1 - y0, 0), (0, 0, z1 - z0), colour, edgy=True))
    if "e" in faces:
        out.append(_rect((x1, y0, z0), (0, y1 - y0, 0), (0, 0, z1 - z0), colour, edgy=True))
    return out


# ---- the flat ----

# Rooms: name, x0, x1, y0, y1, wall paint (a colour or a pattern). The first a point falls in is its room.
ROOMS = [
    ("living", 0.0, 5.0, 4.0, 10.0, CREAM),
    ("hall", 5.0, 13.5, 5.0, 6.5, WHITE),
    ("entry", 5.0, 7.0, 0.0, 5.0, WHITE),
    ("main bedroom", 5.0, 8.0, 6.5, 10.0, WHITE),
    ("guest bedroom", 8.0, 11.0, 6.5, 10.0, BEIGE),
    ("bathroom", 11.0, 13.5, 6.5, 10.0, _tiles((0.60, 0.62, 0.64), (0.82, 0.82, 0.80), 0.25, vertical=True)),
    ("laundry", 7.0, 9.0, 1.5, 5.0, WHITE),
    ("kids room", 9.0, 13.5, 1.5, 5.0, (0.96, 0.76, 0.16)),
    ("kitchen", 7.0, 13.5, -2.5, 1.5, WHITE),
]


def _room_at(x: float, y: float):
    return next((r for r in ROOMS if r[1] <= x <= r[2] and r[3] <= y <= r[4]), None)


# Walls: (a, b) plan ends of the centre line, openings as (from, to) along the wall's own axis (absolute x or y) with
# (z0, z1), and accent paints by room. Doors are open (no leaf) except the front door.
DOOR, WIN = (0.0, DOOR_H), (SILL, HEAD)
WALLS = [
    # outside walls
    ((0, 4), (0, 10), [(6.0, 8.5, 0.0, 2.25)], {}),                         # living: sliding door to the terrace
    ((0, 10), (5, 10), [(0.8, 2.0) + WIN], {"living": (0.55, 0.38, 0.27)}),  # the brown accent wall
    ((5, 10), (8, 10), [(5.6, 7.2) + WIN], {"main bedroom": (0.46, 0.49, 0.26)}),   # olive
    ((8, 10), (11, 10), [(8.6, 10.4) + WIN], {}),
    ((11, 10), (13.5, 10), [(11.6, 12.4) + WIN], {}),
    ((13.5, 10), (13.5, 6.5), [], {}),
    ((13.5, 6.5), (13.5, 5), [], {}),
    ((13.5, 5), (13.5, 1.5), [(2.4, 4.2) + WIN], {}),
    ((13.5, 1.5), (13.5, -2.5), [(-2.2, -1.2) + WIN], {}),
    ((13.5, -2.5), (7, -2.5), [(9.5, 12.0) + WIN], {}),
    ((7, -2.5), (7, 0), [], {}),
    ((7, 0), (5, 0), [], {}),                                                # the front door, closed
    ((5, 0), (5, 4), [(1.5, 3.0) + WIN], {}),
    ((5, 4), (0, 4), [(3.6, 4.7) + WIN], {}),
    # inside walls
    ((5, 4), (5, 6.5), [(4.8, 6.4) + DOOR], {}),                             # living open to the entry and hall
    ((5, 6.5), (5, 10), [], {}),
    ((5, 6.5), (8, 6.5), [(5.8, 6.6) + DOOR], {}),
    ((8, 6.5), (11, 6.5), [(8.4, 9.2) + DOOR], {}),
    ((11, 6.5), (13.5, 6.5), [(11.6, 12.3) + DOOR], {}),
    ((8, 6.5), (8, 10), [], {}),
    ((11, 6.5), (11, 10), [], {}),
    ((7, 5), (9, 5), [], {}),
    ((9, 5), (13.5, 5), [(9.4, 10.2) + DOOR], {}),
    ((7, 0), (7, 1.5), [(0.2, 1.2) + DOOR], {}),                             # entry to kitchen
    ((7, 1.5), (7, 5), [(3.6, 4.4) + DOOR], {}),                             # entry to laundry
    ((7, 1.5), (9, 1.5), [], {}),
    ((9, 1.5), (13.5, 1.5), [], {}),
    ((9, 1.5), (9, 5), [], {}),
]


def _paint(x, y, accents):
    r = _room_at(x, y)
    if r is None:
        return OUTSIDE
    return accents.get(r[0], r[5])


def _walls() -> list[_Surface]:
    out = []
    for a, b, openings, accents in WALLS:
        a, b = np.array(a, float), np.array(b, float)
        length = float(np.linalg.norm(b - a))
        d = (b - a) / length
        n = np.array([-d[1], d[0]])
        axis = 0 if abs(d[0]) > 0.5 else 1
        spans = [(abs(u0 - a[axis]), abs(u1 - a[axis]), z0, z1) for u0, u1, z0, z1 in openings]
        spans = [(min(s0, s1), max(s0, s1), z0, z1) for s0, s1, z0, z1 in spans]
        holes = tuple((s0 / length, s1 / length, z0 / H, z1 / H) for s0, s1, z0, z1 in spans)
        mid = (a + b) / 2
        for side in (1, -1):
            o = a + n * side * WALL_T / 2
            probe = mid + n * side * 0.3
            out.append(_rect((o[0], o[1], 0), (b - a)[0:2].tolist() + [0], (0, 0, H),
                             _paint(probe[0], probe[1], accents), holes))
        for s0, s1, z0, z1 in spans:                  # reveals: the wall's thickness round each opening
            for s in (s0, s1):
                c = a + d * s - n * WALL_T / 2
                out.append(_rect((c[0], c[1], z0), (n[0] * WALL_T, n[1] * WALL_T, 0), (0, 0, z1 - z0), TRIM, edgy=True))
            c = a + d * s0 - n * WALL_T / 2
            if z1 < H:
                out.append(_rect((c[0], c[1], z1), (d[0] * (s1 - s0), d[1] * (s1 - s0), 0),
                                 (n[0] * WALL_T, n[1] * WALL_T, 0), TRIM, edgy=True))
            if z0 > 0:
                out.append(_rect((c[0], c[1], z0), (d[0] * (s1 - s0), d[1] * (s1 - s0), 0),
                                 (n[0] * WALL_T, n[1] * WALL_T, 0), WHITE, edgy=True))
    # the front door's leaf, closed, and the plinth below the outside walls where the garden drops away
    out.append(_rect((5.5, 0.06, 0), (0.9, 0, 0), (0, 0, 2.05), (0.42, 0.28, 0.18), edgy=True))
    out.append(_rect((0, 3.95, GARDEN_Z), (5, 0, 0), (0, 0, -GARDEN_Z), (0.75, 0.74, 0.72)))
    out.append(_rect((5.05, 0, GARDEN_Z), (0, 4, 0), (0, 0, -GARDEN_Z), (0.75, 0.74, 0.72)))
    return out


def _floors_and_ceilings() -> list[_Surface]:
    oak, tiles = (0.74, 0.58, 0.40), _tiles((0.86, 0.84, 0.80), (0.62, 0.60, 0.58))
    return [
        _floor(0, 5, 4, 10, 0, _planks(oak)),
        _floor(5, 13.5, 5, 6.5, 0, _planks(oak)),
        _floor(5, 7, 0, 5, 0, _planks(oak, along_x=False)),
        _floor(5, 8, 6.5, 10, 0, _planks((0.60, 0.46, 0.33))),
        _floor(8, 11, 6.5, 10, 0, _planks((0.70, 0.60, 0.47))),
        _floor(11, 13.5, 6.5, 10, 0, _tiles((0.88, 0.88, 0.87), (0.55, 0.56, 0.58), 0.2)),
        _floor(7, 9, 1.5, 5, 0, tiles),
        _floor(9, 13.5, 1.5, 5, 0, _planks((0.82, 0.72, 0.56), along_x=False)),
        _floor(7, 13.5, -2.5, 1.5, 0, _tiles((0.90, 0.85, 0.76), (0.66, 0.62, 0.56), 0.45)),
        _floor(0, 13.5, 4, 10, H, (0.96, 0.96, 0.95)),
        _floor(5, 13.5, -2.5, 4, H, (0.96, 0.96, 0.95)),
        # outside: the decked terrace level with the flat, and the garden below it
        _floor(-4, 0, 4, 10, 0, _planks((0.56, 0.48, 0.40), width=0.14, along_x=False, spread=0.2)),
        _floor(-4, 5, -0.9, 4, GARDEN_Z, _speckle((0.30, 0.62, 0.20))),
        _floor(-4, 5, -1.5, -0.9, GARDEN_Z, _flowers((0.25, 0.45, 0.18))),
        _rect((-4, 4, GARDEN_Z), (4, 0, 0), (0, 0, -GARDEN_Z), (0.45, 0.40, 0.34)),   # the terrace's face to the garden
    ]


def _furniture() -> list[_Surface]:
    green, white, dark = (0.55, 0.76, 0.22), (0.92, 0.92, 0.90), (0.22, 0.18, 0.17)
    wood, deck_wood = (0.62, 0.45, 0.30), (0.66, 0.52, 0.36)
    leaves = _speckle((0.22, 0.50, 0.18), 0.35)
    s = []
    # living room
    s.append(_floor(1.2, 3.8, 5.6, 8.4, 0.012, _rug((0.40, 0.31, 0.23), (0.28, 0.22, 0.17), 1.2, 3.8, 5.6, 8.4)))
    s += _box(1.5, 3.7, 8.9, 9.6, 0, 0.42, green)
    s += _box(1.5, 3.7, 9.6, 9.85, 0, 0.85, green)
    s += _box(1.25, 1.5, 8.9, 9.85, 0, 0.6, green) + _box(3.7, 3.95, 8.9, 9.85, 0, 0.6, green)
    s += _box(2.0, 2.45, 9.35, 9.6, 0.42, 0.78, white) + _box(2.9, 3.35, 9.35, 9.6, 0.42, 0.78, (0.40, 0.70, 0.35))
    s += _box(0.6, 1.1, 7.3, 7.8, 0, 0.4, green)                                            # pouf
    s += _box(2.2, 2.9, 6.6, 7.3, 0, 0.4, (0.58, 0.58, 0.60))                               # coffee table
    s += _box(1.2, 3.4, 4.1, 4.5, 0, 0.5, (0.32, 0.23, 0.16))                               # TV unit
    s += _box(1.6, 3.0, 4.2, 4.26, 0.8, 1.6, (0.05, 0.05, 0.06), faces="tnew")              # the TV
    s += _box(0.3, 1.1, 4.5, 5.3, 0, 0.75, (0.45, 0.32, 0.22))                              # armchair
    s += _box(4.4, 4.46, 9.4, 9.46, 0, 1.5, dark) + _box(4.2, 4.66, 9.2, 9.66, 1.5, 1.85, (0.96, 0.90, 0.72))
    s += _box(4.4, 4.8, 4.4, 4.8, 0, 0.4, (0.70, 0.40, 0.25)) + _box(4.3, 4.9, 4.3, 4.9, 0.4, 1.3, leaves)
    s += _box(0.8, 1.6, 9.9, 9.95, 1.3, 2.0, (0.06, 0.06, 0.06), faces="s")                 # pictures
    s += _box(3.0, 3.5, 9.9, 9.95, 1.4, 1.9, (0.80, 0.78, 0.70), faces="s")
    # main bedroom
    s += _box(5.8, 7.4, 7.9, 9.9, 0, 0.3, (0.30, 0.22, 0.16))
    s += _box(5.85, 7.35, 7.95, 9.85, 0.3, 0.55, (0.42, 0.30, 0.24), top=(0.40, 0.28, 0.22))
    s += _box(5.95, 6.55, 9.4, 9.8, 0.55, 0.7, white) + _box(6.65, 7.25, 9.4, 9.8, 0.55, 0.7, white)
    s += _box(5.7, 7.5, 9.88, 9.95, 0, 1.1, (0.30, 0.26, 0.22), faces="ts")                  # headboard
    s += _box(5.3, 5.7, 9.5, 9.95, 0, 0.5, white) + _box(7.5, 7.9, 9.5, 9.95, 0, 0.5, white)
    s += _box(7.2, 7.9, 6.6, 7.8, 0, 2.2, (0.95, 0.94, 0.92))                               # wardrobe
    # guest bedroom
    s += _box(8.8, 10.4, 8.0, 9.9, 0, 0.55, (0.25, 0.25, 0.27), top=_checks((0.20, 0.20, 0.22), (0.70, 0.70, 0.68)))
    s += _box(8.7, 10.5, 9.88, 9.95, 0, 1.2, (0.78, 0.70, 0.58), faces="ts")
    s += _box(8.9, 9.5, 9.4, 9.8, 0.55, 0.7, (0.95, 0.95, 0.95)) + _box(9.7, 10.3, 9.4, 9.8, 0.55, 0.7, (0.6, 0.6, 0.6))
    s += _box(10.3, 10.95, 6.6, 7.6, 0, 2.1, (0.75, 0.62, 0.48))
    s += _box(10.55, 10.95, 9.5, 9.95, 0, 0.5, wood) + _box(10.65, 10.85, 9.6, 9.8, 0.5, 0.8, (0.98, 0.85, 0.55))
    # bathroom
    s += _box(12.9, 13.4, 7.6, 8.2, 0, 0.42, white) + _box(13.2, 13.42, 7.6, 8.2, 0.42, 0.8, white)
    s += _box(11.1, 11.6, 8.2, 9.0, 0, 0.85, white)
    s += _box(12.4, 13.45, 9.0, 9.95, 0, 0.06, white)
    # laundry
    s += _box(8.3, 8.95, 4.3, 4.95, 0, 0.85, white) + _box(7.1, 7.5, 1.6, 2.6, 0, 1.8, white)
    # kids' room
    s.append(_floor(10.0, 11.6, 2.2, 3.6, 0.012, (0.45, 0.72, 0.30)))
    s += _box(11.5, 13.4, 1.6, 2.6, 0, 0.35, white)
    s += _box(11.55, 13.35, 1.65, 2.55, 0.35, 0.55, (0.50, 0.78, 0.28), top=(0.92, 0.38, 0.62))
    s += _box(9.1, 9.45, 2.0, 3.4, 0, 1.5, (0.50, 0.76, 0.26))                              # shelves
    s += _box(10.5, 10.9, 3.0, 3.4, 0, 0.4, (0.85, 0.20, 0.18)) + _box(11.0, 11.3, 3.6, 3.9, 0, 0.3, (0.98, 0.60, 0.10))
    s += _box(13.0, 13.45, 3.4, 4.6, 0, 0.75, white)
    # kitchen-diner
    s += _box(9.2, 13.4, 0.85, 1.45, 0, 0.86, (0.70, 0.80, 0.62), faces="sew")
    s += _box(9.2, 13.4, 0.82, 1.45, 0.86, 0.9, (0.96, 0.96, 0.95), faces="tsew")           # worktop
    s += _box(12.75, 13.45, -0.6, 0.2, 0, 1.9, (0.90, 0.90, 0.92))                           # fridge
    s += _box(9.5, 11.0, -1.7, -0.8, 0.72, 0.76, (0.76, 0.62, 0.45))
    for x, y in ((9.55, -1.65), (10.9, -1.65), (9.55, -0.9), (10.9, -0.9)):
        s += _box(x, x + 0.05, y, y + 0.05, 0, 0.72, (0.76, 0.62, 0.45), faces="nsew")
    for x, y in ((9.0, -1.25), (11.5, -1.25), (10.25, -2.2), (10.25, -0.3)):
        s += _box(x - 0.21, x + 0.21, y - 0.21, y + 0.21, 0, 0.46, dark)
    # entry
    s.append(_floor(5.4, 6.6, 1.6, 3.4, 0.012, _diamonds((0.06, 0.06, 0.07), (0.93, 0.93, 0.92))))
    s += _box(5.1, 5.45, 0.3, 1.3, 0, 1.0, white)
    s += _box(6.5, 6.9, 4.4, 4.8, 0, 0.35, (0.70, 0.40, 0.25)) + _box(6.45, 6.95, 4.35, 4.85, 0.35, 1.1, leaves)
    # terrace: a table and four chairs, planters along the edge, rail posts
    s += _box(-2.6, -1.8, 7.2, 8.0, 0.7, 0.74, deck_wood) + _box(-2.25, -2.15, 7.55, 7.65, 0, 0.7, dark, faces="nsew")
    for x, y in ((-3.0, 7.6), (-1.4, 7.6), (-2.2, 8.4), (-2.2, 6.8)):
        s += _box(x - 0.2, x + 0.2, y - 0.2, y + 0.2, 0, 0.45, deck_wood)
    for y in (4.6, 6.0, 9.2):
        s += _box(-3.9, -3.4, y, y + 0.5, 0, 0.45, (0.30, 0.30, 0.32)) + _box(-3.95, -3.35, y - 0.05, y + 0.55, 0.45, 1.3, leaves)
    s += _box(-0.9, -0.4, 9.3, 9.8, 0, 0.5, (0.70, 0.40, 0.25)) + _box(-1.2, -0.1, 9.0, 10.0, 0.5, 2.2, leaves)  # palm
    for x in (-4.0, -3.0, -2.0, -0.5):
        s += _box(x, x + 0.05, 3.98, 4.03, 0, 1.0, (0.40, 0.40, 0.42), faces="nsew")
    # the stair down to the garden, and a hedge along its far side
    for k in (1, 2, 3):
        s += _box(-1.5, -0.6, 4.0 - 0.3 * k, 4.3 - 0.3 * k, GARDEN_Z, -0.3 * k, deck_wood, faces="tsew")
    s += _box(-4.0, 5.0, -1.8, -1.5, GARDEN_Z, -0.4, _speckle((0.15, 0.35, 0.12), 0.35), top=_speckle((0.18, 0.40, 0.15)))
    return s


def _surfaces() -> list[_Surface]:
    return _walls() + _floors_and_ceilings() + _furniture()


def _blocker(range_m: float):
    """(scanner position, points) -> which points a wall hides: the straight line to them crosses a wall's centre
    line anywhere but through an opening. Furniture hides nothing."""
    segs = []
    for a, b, openings, _ in WALLS:
        a, b = np.array(a, float), np.array(b, float)
        axis = 0 if abs(b[0] - a[0]) > abs(b[1] - a[1]) else 1
        segs.append((a, b - a, axis, openings))

    def blocked(eye: np.ndarray, p: np.ndarray) -> np.ndarray:
        out = np.zeros(len(p), bool)
        near = np.flatnonzero(np.linalg.norm(p - eye, axis=1) < range_m)
        d = p[near] - eye
        hidden = np.zeros(len(near), bool)
        for a, e, axis, openings in segs:
            den = d[:, 0] * e[1] - d[:, 1] * e[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((a[0] - eye[0]) * e[1] - (a[1] - eye[1]) * e[0]) / den
                s = ((a[0] - eye[0]) * d[:, 1] - (a[1] - eye[1]) * d[:, 0]) / den
            hit = (t > 0) & (t < 1) & (s >= 0) & (s <= 1)
            if not hit.any():
                continue
            u = eye[axis] + t * d[:, axis]
            z = eye[2] + t * d[:, 2]
            through = np.zeros(len(near), bool)
            for u0, u1, z0, z1 in openings:
                through |= (u > min(u0, u1)) & (u < max(u0, u1)) & (z > z0) & (z < z1)
            hidden |= hit & ~through
        out[near] = hidden
        return out
    return blocked


def _walk(path, frame, range_m=5.5, drift_yaw=0.3, exposure=1.0, z=0.0) -> tuple:
    path = np.array([(*pt, z + 1.4) if len(pt) == 2 else pt for pt in path], float)   # phone at chest height
    walk = _Walk(path=path, door_open=True, drift_yaw_deg=drift_yaw, drift_tilt_deg=0.3,
                 drift_m=(0.01, -0.008, 0.006), exposure=exposure, stage_light=False, stray_share=0.008,
                 range_m=range_m, full_density_m=3.0, texture=0.04, blocked=_blocker(range_m))
    return walk, frame


# One scan per room, each starting or ending a step through its door so that neighbours overlap, in its own raw frame
# (a phone starts every scan at its own origin and heading, a little off level).
QA_WALKS = {
    "living room": _walk([(1.0, 9.0), (3.5, 8.4), (4.2, 6.0), (1.6, 5.0), (0.9, 7.2), (5.7, 5.6)],
                         _transform(23.0, 0.3, -0.2, (-2.0, -7.0, -1.4))),
    "terrace": _walk([(0.8, 7.2, 1.4), (-1.0, 9.0, 1.4), (-3.0, 7.5, 1.4), (-2.5, 5.0, 1.4), (-1.0, 4.4, 1.4),
                      (-1.0, 3.2, 0.5), (0.5, 2.0, 0.2), (3.5, 1.0, 0.2), (-2.5, 0.5, 0.2)],
                     _transform(-71.0, -0.4, 0.3, (4.0, 1.0, -1.4)), range_m=8.0, exposure=1.15, drift_yaw=0.5),
    "hall": _walk([(5.8, 1.0), (6.0, 3.5), (6.0, 5.7), (8.5, 5.8), (11.0, 5.8), (13.0, 5.8), (10.0, 5.7), (6.2, 5.6)],
                  _transform(140.0, 0.2, 0.4, (3.0, -2.0, -1.4)), exposure=0.95),
    "main bedroom": _walk([(6.0, 9.2), (7.0, 8.2), (6.5, 7.2), (6.2, 6.9), (6.2, 5.8)],
                          _transform(-160.0, -0.3, 0.2, (8.0, 9.0, -1.4)), exposure=0.9),
    "guest bedroom": _walk([(9.0, 9.2), (10.4, 8.0), (9.5, 7.2), (8.8, 6.9), (8.8, 5.8)],
                           _transform(95.0, 0.4, -0.3, (-1.0, 12.0, -1.4)), exposure=1.05),
    "bathroom": _walk([(12.8, 9.3), (12.0, 8.0), (12.0, 7.0), (11.95, 5.8)],
                      _transform(12.0, -0.2, -0.4, (-12.0, -8.0, -1.4))),
    "kids room": _walk([(12.8, 3.6), (10.5, 2.5), (10.0, 4.3), (9.8, 5.6)],
                       _transform(-35.0, 0.3, 0.3, (-6.0, 4.0, -1.4)), exposure=1.1),
    "kitchen": _walk([(12.8, -1.8), (10.0, -0.3), (8.0, 0.5), (7.6, 0.7), (6.2, 0.8)],
                     _transform(-150.0, -0.3, -0.2, (9.0, 2.0, -1.4))),
}


def write_qa_set(folder: str, seed: int = 20261029, n_target: int = 140_000) -> list[str]:
    """Writes the QA scans (one .pts per QA_WALKS entry) into folder and returns their paths. Same seed, same files."""
    from open_point_stitch.pts_file import write_pts

    rng = np.random.default_rng(seed)
    surfaces = _surfaces()
    os.makedirs(folder, exist_ok=True)
    paths = []
    for name, (walk, frame) in QA_WALKS.items():
        pts, cols = _scan(surfaces, walk, frame, n_target, rng)
        path = os.path.join(folder, f"{name}.pts")
        write_pts(path, pts, cols, precision=4)
        paths.append(path)
    return paths
