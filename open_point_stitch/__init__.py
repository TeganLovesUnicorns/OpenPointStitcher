"""OpenPointStitcher.

Local, offline pipeline that aligns a folder of gravity-levelled indoor scans into one model, using the geometry
the scans share.

Stages:
    1. pts_file     load and save .pts point clouds
    2. prepare      voxel downsample, normal estimation, FPFH features
    3. align_pair   coarse then fine (ICP) registration per pair
    4. pair_filter  decide which pairs are worth attempting
    5. align_graph  global optimisation over all pairwise constraints
    6. merge        apply final poses, combine, export
"""

__version__ = "0.55.0"
