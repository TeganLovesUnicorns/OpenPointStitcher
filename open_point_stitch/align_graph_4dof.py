"""Solve several scans' yaw and position together from pairwise constraints, four degrees of freedom each. Each
constraint carries how tightly its matched points pin things down, so a wall that fixes only one direction leaves
the others to other constraints."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGMA_POINT_M = 0.03        # how far a matched point may honestly sit from its surface
PRIOR_POSITION_M = 1.0      # the weak pull toward where a scan is now: this far is "no surprise"
PRIOR_YAW_RAD = np.radians(5.0)
_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Constraint:
    """One pair `b` against `a`. `target_b` is where matching put b, as (yaw radians, dx, dy, dz) change from
    its current pose; `info` is 8 x 8 over the changes of [b, a], from `edge_information`."""
    b: str
    a: str
    target_b: np.ndarray
    info: np.ndarray


def edge_information(b_points: np.ndarray, a_points: np.ndarray, a_normals: np.ndarray, weights: np.ndarray,
                     t_b: np.ndarray, t_a: np.ndarray) -> np.ndarray:
    """How tightly these matched points pin down the change in [yaw, x, y, z] of scan b and of scan a, from
    distances measured along the surface normal. `b_points` are b's matched points at their matched position, `a_points`
    the points they matched, `t_b`/`t_a` the scans' positions (a yaw turns a scan about its own origin)."""
    turn_b = np.cross(_UP, b_points - t_b)
    turn_a = np.cross(_UP, a_points - t_a)
    j = np.hstack([(a_normals * turn_b).sum(axis=1, keepdims=True), a_normals,
                   -(a_normals * turn_a).sum(axis=1, keepdims=True), -a_normals])
    return (j * weights[:, None]).T @ j / SIGMA_POINT_M ** 2


def robust_edge(b_points: np.ndarray, a_points: np.ndarray, a_normals: np.ndarray, weights: np.ndarray,
                t_b: np.ndarray, t_a: np.ndarray, rounds: int = 6, cutoff: float = 3.0,
                floor_ratio: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """(step, info) for one pair. `step` is the change in [yaw radians, dx, dy, dz] of scan b that best closes the
    distances between its matched points and the surfaces they matched, measured along the surface normal; `info` is
    `edge_information` with the same weights. Nearest-point matching pairs a wall the other scan never saw with its floor,
    which is wrong by a lot and always the same way, so pairs far from the typical distance count for little. Directions
    the surfaces cannot pin down (sliding along a wall) are left alone."""
    turn_b = np.cross(_UP, b_points - t_b)
    j = np.hstack([(a_normals * turn_b).sum(axis=1, keepdims=True), a_normals])
    distance = ((b_points - a_points) * a_normals).sum(axis=1)
    step = np.zeros(4)
    w = weights
    for _ in range(rounds):
        r = distance + j @ step
        away = r - np.median(r)     # judged against the typical pair, not against zero: the typical pair is what is off
        scale = max(1.4826 * np.median(np.abs(away)), 0.01)
        w = weights * (1.0 - np.clip(np.abs(away) / (cutoff * scale), 0.0, 1.0) ** 2) ** 2
        h = (j * w[:, None]).T @ j
        g = (j * w[:, None]).T @ distance
        values, vectors = np.linalg.eigh(h)
        keep = values > floor_ratio * values.max()
        step = -(vectors[:, keep] / values[keep]) @ (vectors[:, keep].T @ g)
    return step, edge_information(b_points, a_points, a_normals, w, t_b, t_a)


def solve(constraints: list[Constraint], free: list[str]) -> dict[str, np.ndarray]:
    """The change in [yaw radians, x, y, z] for each free scan that best satisfies every constraint at once,
    with a weak pull toward not moving. Scans not in `free` do not move."""
    at = {name: 4 * i for i, name in enumerate(free)}
    n = 4 * len(free)
    a_mat = np.zeros((n, n))
    rhs = np.zeros(n)
    for c in constraints:
        names = [c.b] * 4 + [c.a] * 4
        local = [k for k in range(8) if names[k] in at]
        if not local:
            continue
        glob = [at[names[k]] + (k % 4) for k in local]
        target = np.concatenate([c.target_b, np.zeros(4)])
        a_mat[np.ix_(glob, glob)] += c.info[np.ix_(local, local)]
        rhs[glob] += (c.info @ target)[local]
    prior = np.tile([1.0 / PRIOR_YAW_RAD ** 2] + [1.0 / PRIOR_POSITION_M ** 2] * 3, len(free))
    a_mat += np.diag(prior)
    delta = np.linalg.solve(a_mat, rhs) if n else np.zeros(0)
    return {name: delta[at[name]:at[name] + 4] for name in free}
