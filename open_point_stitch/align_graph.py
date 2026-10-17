"""Solving every scan's pose together. The pose graph: each accepted pair is an edge; trusted (placement-adjacent)
edges anchor the optimiser, and the rest may be down-weighted or pruned (cfg.posegraph_edge_prune_threshold). And the
four-degree-of-freedom solve behind "Align from links": yaw and position per scan from constraints that know which
directions they pin down."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .align_pair import PairResult
from .config import PipelineConfig
from .lib.groups import connected_groups

log = logging.getLogger(__name__)

reg = o3d.pipelines.registration


@dataclass
class GlobalResult:
    poses: list[np.ndarray]           # world-from-scan 4x4 per scan index
    n_odometry_edges: int
    n_loop_edges: int
    connected: bool
    components: list[list[int]]


def _detector_weighted_information(info: np.ndarray, r: PairResult, cfg: PipelineConfig) -> np.ndarray:
    """Scale one edge's information matrix (how much the global solve trusts it) by how strongly the detectors agreed
    at that edge's transform. Left at 1.0 when the vote did not run or had too little evidence; this only ranks
    edges that already passed every gate, it is not a second veto."""
    if not cfg.posegraph_detector_weighting_enabled or r.vote_detector_score is None:
        return info
    evidence = ((r.vote_n_planes_considered or 0) + (r.vote_n_corners_considered or 0)
                + (r.vote_n_edges_considered or 0) + (r.vote_n_openings_considered or 0)
                + (r.vote_n_landmarks_considered or 0))
    if evidence < cfg.posegraph_detector_weight_min_evidence:
        return info
    floor = cfg.posegraph_detector_weight_floor
    multiplier = floor + (1.0 - floor) * float(np.clip(r.vote_detector_score, 0.0, 1.0))
    return info * multiplier


def build_and_optimize(
    n_scans: int,
    results: list[PairResult],
    cfg: PipelineConfig,
    trusted_pairs: Optional[set[tuple[int, int]]] = None,
) -> GlobalResult:
    """trusted_pairs: set of (i, j), i < j, placement-adjacent scan-index pairs; an accepted edge in it is trusted.
    None falls back to consecutive indices."""
    accepted = [r for r in results if r.accepted]
    if not accepted:
        raise RuntimeError(
            "No pairwise registration was accepted, so there is nothing to "
            "optimize. Loosen --min-fitness/--max-rmse, lower --voxel-size, or "
            "check that the scans actually overlap."
        )

    edge_pairs = [(r.source, r.target) for r in accepted]
    components = sorted(connected_groups(n_scans, edge_pairs), key=lambda g: (-len(g), g[0]))
    connected = len(components) == 1
    if not connected:
        log.warning(
            "Scans form %d disconnected groups: %s. Only the largest group can "
            "be aligned into one coherent cloud.",
            len(components), components,
        )

    pose_graph = reg.PoseGraph()
    # Node poses are initialised by walking a spanning tree of accepted edges,
    # so the optimizer starts from something plausible rather than all identity.
    init_poses: list[np.ndarray | None] = [None] * n_scans
    adj: dict[int, list[PairResult]] = {}
    for r in accepted:
        adj.setdefault(r.source, []).append(r)
        adj.setdefault(r.target, []).append(r)

    for comp in components:
        root = comp[0]
        init_poses[root] = np.eye(4)
        stack = [root]
        seen = {root}
        while stack:
            cur = stack.pop()
            for r in adj.get(cur, []):
                other = r.target if r.source == cur else r.source
                if other in seen:
                    continue
                # r.transform maps source points into the target frame.
                if r.source == cur:
                    # other is target: world_T_other = world_T_cur @ inv(T)
                    init_poses[other] = init_poses[cur] @ np.linalg.inv(r.transform)
                else:
                    init_poses[other] = init_poses[cur] @ r.transform
                seen.add(other)
                stack.append(other)

    for i in range(n_scans):
        if init_poses[i] is None:
            init_poses[i] = np.eye(4)
        pose_graph.nodes.append(reg.PoseGraphNode(init_poses[i]))

    n_odo = 0
    n_loop = 0
    for r in accepted:
        if trusted_pairs is not None:
            uncertain = (min(r.source, r.target), max(r.source, r.target)) not in trusted_pairs
        else:
            uncertain = abs(r.target - r.source) != 1
        pose_graph.edges.append(
            reg.PoseGraphEdge(
                r.source, r.target,
                r.transform, _detector_weighted_information(r.information, r, cfg),
                uncertain=uncertain,
            )
        )
        if uncertain:
            n_loop += 1
        else:
            n_odo += 1

    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=cfg.voxel_size
        * cfg.posegraph_max_correspondence_distance_mult,
        edge_prune_threshold=cfg.posegraph_edge_prune_threshold,
        preference_loop_closure=cfg.posegraph_preference_loop_closure,
        reference_node=min(cfg.posegraph_reference_node, n_scans - 1),
    )

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        reg.global_optimization(
            pose_graph,
            reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(),
            option,
        )

    poses = [np.array(node.pose, copy=True) for node in pose_graph.nodes]
    log.info("pose graph optimized: %d nodes, %d odometry edges, %d loop edges",
             n_scans, n_odo, n_loop)

    return GlobalResult(
        poses=poses,
        n_odometry_edges=n_odo,
        n_loop_edges=n_loop,
        connected=connected,
        components=components,
    )


# -------------------------------------------------------------------------------------------------------------------
# Solve several scans' yaw and position together from pairwise constraints, four degrees of freedom each. Each
# constraint carries how tightly its matched points pin things down, so a wall that fixes only one direction leaves
# the others to other constraints.
# -------------------------------------------------------------------------------------------------------------------

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


def solve_4dof(constraints: list[Constraint], free: list[str]) -> dict[str, np.ndarray]:
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
