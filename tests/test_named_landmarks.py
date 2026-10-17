"""Named landmarks: the same name in two scans is one link between them, and both drawings show each landmark's box
with its name."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.export_3d import dxf_text as dxf_3d  # noqa: E402
from open_point_stitch.export_plan import PlanCut, dxf_text as dxf_plan  # noqa: E402
from open_point_stitch.placement import world_boxes  # noqa: E402
from open_point_stitch.placement import SCHEMA, Placement, anchors_from_landmarks, load_placements  # noqa: E402


def _mark(scan, name, centre, size=(0.4, 0.4, 2.0), kind="column"):
    return {"id": f"{scan}-{name}-{centre[0]}", "scan": scan, "name": name, "kind": kind,
            "points": [list(centre)], "centroid": list(centre), "extent": list(size), "n_points": 120}


def test_the_same_name_in_two_scans_is_one_anchor_and_a_name_in_one_scan_links_nothing():
    marks = [_mark("A", "Stage Left Column", (1, 2, 1)), _mark("B", "stage  left column", (5, 6, 1)),
             _mark("A", "Window", (0, 0, 1.5))]
    anchors = anchors_from_landmarks(marks)
    assert len(anchors) == 1, [a.label for a in anchors]
    a = anchors[0]
    assert sorted(m.scan for m in a.members) == ["A", "B"] and a.label == "landmark: Stage Left Column", a
    m = a.members[0]
    assert np.allclose(m.centroid, (1, 2, 1)) and np.allclose(m.extent, (0.4, 0.4, 2.0)), (m.centroid, m.extent)


def test_two_marks_of_one_name_in_one_scan_are_one_member_around_both():
    marks = [_mark("A", "doorway", (0, 0, 1), size=(0.2, 0.2, 2)), _mark("A", "doorway", (1, 0, 1), size=(0.2, 0.2, 2)),
             _mark("B", "doorway", (3, 3, 1))]
    a = anchors_from_landmarks(marks)[0]
    m = next(m for m in a.members if m.scan == "A")
    assert np.allclose(m.centroid, (0.5, 0, 1)) and np.allclose(m.extent, (1.2, 0.2, 2)), (m.centroid, m.extent)
    assert m.n_points == 240


def test_a_saved_project_brings_its_landmarks_to_every_run_as_anchors():
    data = {"schema": SCHEMA, "scans": [{"name": "A"}, {"name": "B"}], "anchors": [],
            "landmarks": [_mark("A", "column", (1, 1, 1)), _mark("B", "column", (2, 2, 1))]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "placement.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        loaded = load_placements(path)
    assert [a.id for a in loaded.anchors] == ["landmark:column"], loaded.anchors


def test_both_drawings_show_each_landmark_box_and_its_name_once():
    marks = [_mark("A", "column", (1, 0, 1)), _mark("B", "column", (0, 1, 1))]
    placements = {"A": Placement(name="A", tx=10.0), "B": Placement(name="B", yaw_deg=90.0, tx=10.0)}
    boxes = world_boxes(marks, placements)
    assert np.allclose(boxes[0].corners.mean(axis=0), (11, 0, 1)), boxes[0].corners.mean(axis=0)
    assert np.allclose(boxes[1].corners.mean(axis=0), (9, 0, 1), atol=1e-9), boxes[1].corners.mean(axis=0)
    plan = dxf_plan(PlanCut(landmarks=boxes)).splitlines()
    assert plan.count("LANDMARKS") >= 3 and sum("column (column)" in l for l in plan) == 1
    body = dxf_3d([], landmarks=boxes).splitlines()
    on_layer = [body[k + 1] for k in range(len(body) - 1) if body[k].strip() == "8" and body[k + 1] == "LANDMARKS"]
    assert len(on_layer) >= 4 * 2 and sum("column (column)" in l for l in body) == 1


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
