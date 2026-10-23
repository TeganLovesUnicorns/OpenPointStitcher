"""Tunable parameters for the stitching pipeline, in one place.

Distances are in metres. Defaults suit room-scale indoor scans; the voxel_size multipliers follow Open3D's
multiway registration example."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class PipelineConfig:
    # --- Preprocessing -------------------------------------------------
    voxel_size: float = 0.05           # 5 cm downsample for registration work
    normal_radius_mult: float = 2.0    # normal search radius = mult * voxel
    normal_max_nn: int = 30
    fpfh_radius_mult: float = 5.0      # FPFH search radius = mult * voxel
    fpfh_max_nn: int = 100

    # Coarse strategy:
    #   ransac  FPFH feature matching over all 6 degrees of freedom
    #   gravity level both clouds, then exhaustive yaw + FFT translation search
    #   walls   match dominant wall planes (adjacent rather than overlapping rooms)
    #   both    try all of them and keep whichever scores best after ICP
    coarse_method: str = "both"

    # Scans are gravity levelled, so a relative transform tipping past max_tilt_deg is rejected regardless of
    # score. Set False for input that is not levelled.
    assume_level: bool = True
    max_tilt_deg: float = 6.0

    gravity_grid_res: float = 0.10     # metres per cell in the top down image
    gravity_yaw_step_deg: float = 1.0
    gravity_top_k: int = 8

    # --- Coarse registration (FPFH + RANSAC) ---------------------------
    ransac_distance_mult: float = 1.5  # correspondence distance = mult * voxel
    ransac_n: int = 3                  # points sampled per RANSAC iteration
    ransac_max_iteration: int = 4_000_000
    ransac_confidence: float = 0.999
    ransac_edge_length_ratio: float = 0.9

    # --- Fine registration (ICP) ---------------------------------------
    icp_distance_mult: float = 0.4     # tighter than RANSAC, by design
    icp_max_iteration: int = 60
    icp_method: str = "point_to_plane"  # point_to_plane | point_to_point | colored

    # Optional colour ICP pass on top of the geometric result. Off by default: no measurable gain.
    use_colored_icp: bool = False
    colored_icp_scales: tuple[float, ...] = (4.0, 2.0, 1.0)  # * voxel_size
    colored_icp_iters: tuple[int, ...] = (50, 30, 14)

    # Optional ICP pass on both scans cropped to their overlap at full resolution. Off by default.
    full_res_refine: bool = False

    # Pair acceptance
    min_fitness: float = 0.05          # below this, a pair is not trusted
    max_rmse: float = 0.10             # 10 cm, above this a pair is not trusted
    min_correspondences: int = 500

    # Direct-overlap check on top of fitness, so a pose cannot pass by pushing one cloud inside another.
    # Needs full-resolution scans; skipped (not failed) when a caller does not supply them.
    overlap_gate_enabled: bool = True
    overlap_gate_threshold_m: float = 0.15   # 15 cm
    overlap_gate_min_frac: float = 0.05      # smaller of the two directions, at or above 5%

    # Pairs always rejected regardless of score, for pairs scoring cannot tell right from wrong. Each entry is a
    # frozenset of two scan names; a project sets its own in its file ("excluded_pairs").
    excluded_pairs: frozenset = field(default_factory=frozenset)

    # Local search around a hand-placed seed: the seed is one candidate plus this many sampled nearby, all ranked
    # with the unseeded path's own candidates.
    local_search_n: int = 12
    local_search_radius_m: float = 1.5   # xy jitter, uniform in [-r, r]
    local_search_height_m: float = 0.3   # z jitter - height is usually the most trustworthy axis
    local_search_yaw_deg: float = 5.0    # yaw jitter, uniform in [-y, y] degrees

    # "Run from my alignment": the radius pairs are chosen within (shared surface), and the per-scan limit's fallback.
    local_only_bound_m: float = 0.5
    local_only_bound_deg: float = 5.0
    # A candidate further than this from the placement it was seeded from cannot win. Wider lets wrong answers win.
    candidate_bound_m: float = 0.5
    candidate_bound_deg: float = 5.0
    # Mean shift over every scan searched; above it the whole run's geometry search is declined.
    local_only_layout_bound_mean_m: float = 1.0
    # Outside proposals (seed_key -> 4x4 raw source to raw target) that join a seeded pair's candidates when
    # within the placement gate, and a wider per-scan bound for them. Not set by the bench.
    proposals: dict = field(default_factory=dict)
    proposal_agree: dict = field(default_factory=dict)   # seed_key -> how many of the proposer's matches agreed
    proposal_gate_m: float = 1.5
    proposal_gate_deg: float = 15.0
    scan_bound_m: Optional[float] = 1.5
    scan_bound_deg: Optional[float] = 15.0

    # Coarse-to-fine ICP windows for seeded runs, as multiples of voxel_size.
    local_only_icp_distance_mults: tuple = (6.0, 2.0, 0.8, 0.4)

    # Relative fitness gate, off; min_ratio is unused while it is.
    relative_fitness_gate_enabled: bool = False
    relative_fitness_min_ratio: float = 0.20

    # Coarse search resolution. None uses voxel_size. Only candidate generation (RANSAC, walls, gravity) uses
    # it; fine ICP always uses voxel_size.
    search_voxel_size: Optional[float] = None
    # Off: a pair's search voxel is search_voxel_size. On: never finer than this many point spacings of the
    # sparser scan.
    adaptive_search_voxel: bool = False
    adaptive_voxel_spacings: float = 2.5
    # "Run from my alignment" refines every candidate on the GPU at once, then the CPU re-runs the best few from
    # their starting poses, so the final answer is always the CPU's.
    use_gpu_ladder: bool = False
    gpu_verify_top: int = 2
    gpu_screen_iterations: int = 30
    # With the GPU screen on, the full ladder repeats the best gpu_verify_top candidates and its answer is final.
    gpu_cpu_check: bool = False
    # Starting poses from matched landmarks (find_landmarks).
    use_landmarks: bool = False
    landmark_top_k: int = 8
    # Weight a wall's fitted position by how tightly its points scatter locally, instead of a plain average.
    wall_tightness_weighting: bool = False
    # Thick surfaces count for less in refinement and the pose graph. Weights only.
    thickness_trust: bool = False
    # Mid-air points (few neighbours) count for less in refinement and the pose graph. Weights only.
    midair_trust: bool = False
    # Points weighted by the trained trust map (trust_model.json) in refinement, the pose graph and the judge.
    learned_trust: bool = False
    # Run the weighted refinement's nearest-neighbour search on the GPU. Varies by about 0.7 mm between runs;
    # the CPU version does not vary.
    trust_search_on_gpu: bool = False
    # In "Run from my alignment", a trained judge (seed_judge.json) ranks candidates and decides whether to move.
    seed_judge: bool = False
    # Bins with at most this many points are never thinned. Matches the browser preview's PREVIEW_FLOOR_N.
    search_thin_floor_n: int = 3

    # --- Pose graph ------------------------------------------------------
    # Edges between consecutive scans are treated as odometry (uncertain=False),
    # everything else as loop closure candidates that the optimizer may reject.
    posegraph_max_correspondence_distance_mult: float = 1.5
    posegraph_edge_prune_threshold: float = 0.25
    posegraph_preference_loop_closure: float = 0.1
    posegraph_reference_node: int = 0
    # Weight each accepted pose-graph edge by how strongly the detectors agreed on it.
    posegraph_detector_weighting_enabled: bool = False
    posegraph_detector_weight_floor: float = 0.2   # a strongly-disagreeing edge is downweighted, never zeroed by this alone
    posegraph_detector_weight_min_evidence: int = 2   # fewer matches than this: leave the edge untouched, not enough to judge by

    # --- Adjacency -------------------------------------------------------
    adjacency_mode: str = "all"        # all | sequential | bbox
    bbox_padding: float = 1.0          # metres of slack when testing bbox overlap

    # Seeded runs: attempt pairs that share surface, within this radius (the expected placement error),
    # rather than pairs whose bounding boxes touch.
    adjacency_surface_min_frac: float = 0.02
    adjacency_surface_thin_points: int = 20_000

    # Pair selection by matched static surfaces, falling back per scan to raw density. position_bound_m is
    # deliberately separate from local_only_bound_m.
    adjacency_feature_position_bound_m: float = 2.0
    adjacency_feature_min_frac: float = 0.05

    # Detector voting. Only acts where both scans have ScanFeatures.
    detector_vote_enabled: bool = True
    # score.py's colour family classifier; separate from find_surfaces._rgb_to_hsv's floor.
    colour_saturation_floor: float = 0.12
    colour_hue_tolerance_deg: float = 35.0   # how close two hues must be to "agree"
    # Six colour families with centres 60 degrees apart around the hue wheel. score.hue_family picks the
    # nearest centre, so the wrap at 0/360 needs no special case.
    colour_family_edges: tuple = (
        ("red", 0.0), ("orange", 60.0), ("yellow", 120.0),
        ("green", 180.0), ("blue", 240.0), ("purple", 300.0),
    )
    # How close a candidate's transformed surfaces must land to the target's to count as the same surface.
    # Deliberately not local_only_bound_m, which would hide the placement error this needs to see.
    vote_plane_position_bound_m: float = 1.0
    vote_plane_angle_bound_deg: float = 30.0
    # Read only by tests.
    vote_min_planes: int = 2

    # The one verdict that can reject a candidate: weight of evidence across every detector that had a say.
    vote_min_agreement: float = 0.15
    vote_min_evidence: int = 3
    vote_min_detectors: int = 2

    # Geometry (surface_fit) is the base ranking; plane and colour agreement only add a bounded bonus.
    vote_weight_plane: float = 0.20
    vote_weight_colour: float = 0.10

    # Corners are a stronger signal than single planes: a corner with three specific angles is nearly unique.
    vote_corner_position_bound_m: float = 0.6
    vote_corner_angle_tolerance_deg: float = 20.0
    vote_corner_min_cohesion: float = 0.5   # find_corners.Corner.cohesion floor to trust one at all
    # Read only by tests.
    vote_min_corners: int = 1
    vote_weight_corner: float = 0.30

    # Edges: two surfaces meeting along a line. Far more common than three-plane corners in real rooms.
    vote_edge_position_bound_m: float = 0.6   # perpendicular gap between the two lines
    vote_edge_angle_tolerance_deg: float = 20.0
    vote_edge_min_overlap_m: float = 0.25     # shared run before it counts as the same line
    vote_edge_min_cohesion: float = 0.5
    # A gate, not a graded penalty. Genuine matches differ by a few degrees; 20 is about three times that spread.
    vote_edge_fold_tolerance_deg: float = 20.0

    # How sharply every detector's position term falls off; separate from the bounds.
    vote_position_scale_m: float = 0.10

    # Fraction by which a candidate must beat the user's own placement before a scan is moved; otherwise the
    # placement is kept.
    min_improvement_frac: float = 0.05
    vote_weight_edge: float = 0.30

    # Doors and windows (find_openings): can raise a candidate, never veto one.
    vote_opening_position_bound_m: float = 0.6
    vote_opening_angle_tolerance_deg: float = 10.0
    vote_opening_size_tolerance_m: float = 0.35
    vote_opening_min_cohesion: float = 0.5
    vote_opening_wall_thickness_m: float = 0.30   # a door seen from two rooms sits a wall's thickness apart
    vote_weight_opening: float = 0.15

    # Landmarks (find_landmarks): can raise a candidate, never veto one. Looser than openings because a
    # landmark's footprint estimate is noisier.
    vote_landmark_position_bound_m: float = 0.6
    vote_landmark_height_tolerance_m: float = 0.3
    vote_landmark_size_ratio: float = 2.0
    vote_weight_landmark: float = 0.15

    # Input limits
    # Alignment only, never the export: ignore points further than this from the scan's origin, measured across
    # the floor. None keeps everything. Off by default, since a walked scan has good points far from its origin.
    max_range_m: Optional[float] = None

    # Output. Both of these discard points, so both are off by default: reduction belongs downstream, where a
    # person can see what is being removed.
    merge_voxel_size: Optional[float] = None   # metres; None keeps every point
    outlier_nb_neighbors: Optional[int] = None  # None keeps every point
    outlier_std_ratio: float = 2.0
    # Six decimals is well under a micron and keeps the file a sensible size.
    output_precision: int = 6

    # Every RANSAC in the pipeline is seeded from this, so the same input gives
    # the same wall labels and the same candidates on every run. That matters
    # for the manual workflows, where a person refers to walls by label.
    random_seed: int = 42

    # --- Resources ---------------------------------------------------------
    threads: Optional[int] = None      # None means "let the libraries decide"

    # Run the expensive stages through Open3D's tensor API (the CUDA-capable path). Off by default: its results
    # differ slightly from the legacy path.
    use_gpu: bool = False
    # The CUDA pipeline in align_gpu.py, for unseeded runs only.
    use_gpu_pipeline: bool = False
    # Scan pairs processed at once. Plane fitting always runs single-threaded (limits.single_threaded) because
    # Open3D's RANSAC races across threads; RANSAC and ICP parallelise across workers.
    concurrency: int = 1

    def scaled(self) -> dict:
        """Absolute distances derived from voxel_size, for logging/reports."""
        v = self.voxel_size
        return {
            "voxel_size": v,
            "normal_radius": v * self.normal_radius_mult,
            "fpfh_radius": v * self.fpfh_radius_mult,
            "ransac_distance": v * self.ransac_distance_mult,
            "icp_distance": v * self.icp_distance_mult,
        }

    def to_dict(self) -> dict:
        # asdict() cannot serialise a frozenset.
        d = asdict(self)
        d["excluded_pairs"] = sorted(sorted(pair) for pair in self.excluded_pairs)
        d["proposals"] = len(self.proposals)       # a count, not every 4x4
        d["proposal_agree"] = len(self.proposal_agree)
        return d
