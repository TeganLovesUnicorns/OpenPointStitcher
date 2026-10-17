"""Global multiway registration by pose graph optimisation. Each accepted pair is an edge; trusted
(placement-adjacent) edges anchor the optimiser, and the rest may be down-weighted or pruned
(cfg.posegraph_edge_prune_threshold)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .config import PipelineConfig
from .align_pair import PairResult
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
