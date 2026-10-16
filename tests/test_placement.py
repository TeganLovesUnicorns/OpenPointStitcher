"""Tests for placement.py: the transform maths, pitch/roll storage and the save/load round trip."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch.placement import (  # noqa: E402
    Placement, SCHEMA, save_placements, load_placements,
    pitch_matrix3, roll_matrix3, yaw_matrix3,
)


def test_zero_pitch_roll_matches_old_behaviour():
    """With pitch and roll at zero, matrix() is yaw_matrix3(yaw) @ level_rotation, as before tilt was stored, so old
    placements keep their exact transforms."""
    rng = np.random.default_rng(0)
    level = np.linalg.qr(rng.normal(size=(3, 3)))[0]  # some orthonormal rotation
    p = Placement(name="x", level_rotation=level, yaw_deg=37.0, tx=1.0, ty=-2.0, tz=0.5)
    expected = np.eye(4)
    expected[:3, :3] = yaw_matrix3(37.0) @ level
    expected[:3, 3] = [1.0, -2.0, 0.5]
    assert np.allclose(p.matrix(), expected)


def test_pitch_and_roll_are_rotations_about_the_right_axis():
    """pitch tips X toward Z (rotation about Y), roll tips Y toward Z
    (rotation about X) - a unit vector on the axis being rotated ABOUT must
    be left unchanged."""
    y_axis = pitch_matrix3(15.0) @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(y_axis, [0.0, 1.0, 0.0])
    x_axis = roll_matrix3(15.0) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(x_axis, [1.0, 0.0, 0.0])


def test_pitch_roll_changes_the_matrix_when_nonzero():
    p_level = Placement(name="x")
    p_tilted = Placement(name="x", pitch_deg=3.0, roll_deg=-2.0)
    assert not np.allclose(p_level.matrix(), p_tilted.matrix())


def test_save_and_load_round_trips_pitch_and_roll():
    placements = {
        "a": Placement(name="a", yaw_deg=10.0, tx=1.0, ty=2.0, tz=0.1,
                       pitch_deg=1.74, roll_deg=-0.9),
        "b": Placement(name="b"),  # left at defaults, including zero tilt
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "placement.json")
        save_placements(path, placements.values())
        loaded = load_placements(path)

    assert loaded.placements["a"].pitch_deg == 1.74
    assert loaded.placements["a"].roll_deg == -0.9
    assert loaded.placements["b"].pitch_deg == 0.0
    assert loaded.placements["b"].roll_deg == 0.0


def test_old_placement_file_without_pitch_roll_still_loads():
    """A file without pitch_deg/roll_deg keys loads at zero tilt instead of raising KeyError."""
    old_style = {
        "schema": SCHEMA,
        "scans": [{
            "name": "old",
            "yaw_deg": 12.5,
            "x_m": 3.0, "y_m": 4.0, "z_m": 0.0,
            "floor_z_local_m": 0.0,
            "level_rotation": np.eye(3).tolist(),
            "locked": False,
        }],
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old_placement.json")
        with open(path, "w") as fh:
            json.dump(old_style, fh)
        loaded = load_placements(path)

    p = loaded.placements["old"]
    assert p.pitch_deg == 0.0
    assert p.roll_deg == 0.0
    assert p.yaw_deg == 12.5


def test_a_scans_floor_lands_at_its_height():
    """tz is the world height of the scan's own floor: a point on that floor lands at tz, whatever the floor level."""
    p = Placement(name="x", floor_z=-1.6, tz=0.75)
    on_the_floor = np.array([3.0, 2.0, -1.6, 1.0])
    assert np.isclose((p.matrix() @ on_the_floor)[2], 0.75)


def test_a_project_saved_under_the_old_name_still_loads():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "placement.json")
        with open(path, "w") as fh:
            json.dump({"schema": "nomad_stitch.placement.v1", "scans": [{"name": "a", "yaw_deg": 30.0}]}, fh)
        loaded = load_placements(path)
    assert loaded.placements["a"].yaw_deg == 30.0


def test_an_unknown_schema_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "placement.json")
        with open(path, "w") as fh:
            json.dump({"schema": "something_else.v1", "scans": []}, fh)
        try:
            load_placements(path)
        except ValueError:
            return
    raise AssertionError("an unknown schema should raise ValueError")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
