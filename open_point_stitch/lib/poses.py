"""Turning, tilting and placing scans: yaw rotations, the bench's tilt order, a scan's levelled frame, and the
closed-form yaw-and-shift fit between matched points. Every module that turns a scan uses these."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def rot2(theta_rad: float) -> np.ndarray:
    """2x2 plan rotation by an angle in radians, counter-clockwise."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def yaw_matrix2(deg: float) -> np.ndarray:
    return rot2(np.radians(deg))


def yaw_matrix3(deg: float) -> np.ndarray:
    m = np.eye(3)
    m[:2, :2] = yaw_matrix2(deg)
    return m


def pitch_matrix3(deg: float) -> np.ndarray:
    """Rotation about Y: tips the scan front-to-back (tilts the X axis)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def roll_matrix3(deg: float) -> np.ndarray:
    """Rotation about X: tips the scan side-to-side (tilts the Y axis)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def tilt_matrix3(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """A scan's rotation as the bench draws it: yaw, then the hand-set roll, then pitch (world axes)."""
    return pitch_matrix3(pitch_deg) @ roll_matrix3(roll_deg) @ yaw_matrix3(yaw_deg)


def small_motion(x: np.ndarray) -> np.ndarray:
    """The 4x4 for one ICP step's small motion (rx, ry, rz, tx, ty, tz), composed as Open3D's ICP composes it (Rz Ry Rx).
    Every ICP solver here builds its step with this, so the CPU and the card agree."""
    cx, sx, cy, sy, cz, sz = np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1]), np.cos(x[2]), np.sin(x[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = x[3:]
    return m


def yaw_transform(yaw_deg: float, t: np.ndarray) -> np.ndarray:
    """4x4 turning about the vertical by yaw_deg, then shifting by t."""
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    m = np.eye(4)
    m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m[:3, 3] = t
    return m


def yaw_of(m: np.ndarray) -> float:
    """Degrees a 3x3 rotation or 4x4 transform turns the plan, read from its first column."""
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def wrap_deg(a: float) -> float:
    """An angle in degrees brought into [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def yaw_gap_deg(a: float, b: float) -> float:
    """How far apart two headings are, 0 to 180 degrees, whichever way round."""
    return abs(wrap_deg(a - b))


def tilt_of(m: np.ndarray) -> float:
    """How far a 3x3 rotation or 4x4 transform tips the vertical axis, in degrees."""
    z = m[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))


def rotation_angle_deg(r: np.ndarray) -> float:
    """Total turn of a 3x3 rotation about any axis, in degrees, from its trace."""
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def levelled_points(points: np.ndarray, level_rotation, floor_z: Optional[float] = None) -> np.ndarray:
    """Raw scan points in the scan's levelled frame; with floor_z, its detected floor moved to 0."""
    out = np.asarray(points, dtype=float) @ np.asarray(level_rotation, dtype=float).T
    if floor_z is not None:
        out[:, 2] -= floor_z
    return out


def fit_yaw_shift_2d(src, tgt, weights=None, require_spread: bool = False) -> tuple[float, np.ndarray]:
    """Closed-form least-squares plan turn (radians) and shift taking src points onto tgt (N x 2 or more; only x
    and y are used). Optional per-pair weights. require_spread raises ValueError when the points are coincident
    and the turn is undetermined; otherwise that case returns a turn of 0."""
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    if weights is None:
        w = np.ones(src.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != src.shape[0]:
            raise ValueError("weights must have one entry per correspondence")
        if np.any(np.isnan(w)):
            raise ValueError("weights must not contain NaN")     # NaN passes the w < 0 check below
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    wsum = float(w.sum())
    if wsum <= 0:
        raise ValueError("weights sum to zero; nothing to solve from")

    sc = (w[:, None] * src[:, :2]).sum(axis=0) / wsum
    tc = (w[:, None] * tgt[:, :2]).sum(axis=0) / wsum
    a = src[:, :2] - sc
    b = tgt[:, :2] - tc
    num = float(np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))
    den = float(np.sum(w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    if require_spread and abs(num) < 1e-12 and abs(den) < 1e-12:
        raise ValueError("picked points are coincident, so yaw is undetermined. Pick two "
                         "features that are far apart.")
    theta = float(np.arctan2(num, den))
    return theta, tc - rot2(theta) @ sc


def solve_yaw_translation(src: np.ndarray, tgt: np.ndarray,
                          weights: Optional[np.ndarray] = None) -> tuple[float, np.ndarray]:
    """Least-squares yaw (degrees) and 3D translation mapping src onto tgt, both levelled (N x 3, N >= 2). Height
    is the weighted mean height difference."""
    src = np.asarray(src, dtype=float).reshape(-1, 3)
    tgt = np.asarray(tgt, dtype=float).reshape(-1, 3)
    if src.shape != tgt.shape or src.shape[0] < 2:
        raise ValueError("need at least 2 matching points in each cloud")
    theta, t_xy = fit_yaw_shift_2d(src, tgt, weights, require_spread=True)
    w = np.ones(src.shape[0], dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    t_z = float(np.sum(w * (tgt[:, 2] - src[:, 2])) / float(w.sum()))
    return float(np.degrees(theta)), np.array([t_xy[0], t_xy[1], t_z])


@dataclass
class ScanPose:
    """Where the bench draws a scan: heading, position of its floor-level origin, and the hand-set tilt."""
    yaw_deg: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    # hand-set tilt; the solve never changes it
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    @classmethod
    def from_dict(cls, s: dict, strict: bool = False, prefix: str = "") -> "ScanPose":
        """One scan's pose from a placement entry; strict and prefix as in Placement.from_dict."""
        pose = (lambda k: float(s[prefix + k])) if strict else (lambda k: float(s.get(prefix + k, 0.0)))
        return cls(yaw_deg=pose("yaw_deg"), x_m=pose("x_m"), y_m=pose("y_m"), z_m=pose("z_m"),
                   pitch_deg=float(s.get(prefix + "pitch_deg", 0.0)), roll_deg=float(s.get(prefix + "roll_deg", 0.0)))

    def place(self, local: np.ndarray) -> np.ndarray:
        """Levelled, floor-relative points (N x 3) to where the bench draws them: yaw, roll, pitch, then shift."""
        return local @ tilt_matrix3(self.pitch_deg, self.roll_deg, self.yaw_deg).T + np.array([self.x_m, self.y_m, self.z_m])


def world_cloud_points(payload: dict, pose: ScanPose) -> np.ndarray:
    """A scan's preview points (payload "xy" and "h") turned and shifted by pose, tilt left out."""
    xy = np.asarray(payload["xy"], dtype=float)
    h = np.asarray(payload["h"], dtype=float)
    world_xy = xy @ yaw_matrix2(pose.yaw_deg).T + [pose.x_m, pose.y_m]
    return np.column_stack([world_xy, h + pose.z_m])


def world_cloud_points_as_drawn(payload: dict, pose: ScanPose) -> np.ndarray:
    """Preview points where the bench draws them, hand tilt included."""
    return pose.place(np.column_stack([np.asarray(payload["xy"], dtype=float).reshape(-1, 2),
                                       np.asarray(payload["h"], dtype=float)]))


def raw_to_world_points(raw_xyz, level_rotation: np.ndarray, floor_z: float, pose: ScanPose) -> np.ndarray:
    """Where raw-frame points (N x 3) are drawn, hand tilt included, as in world_cloud_points_as_drawn."""
    return pose.place(levelled_points(np.asarray(raw_xyz, dtype=float).reshape(-1, 3), level_rotation, floor_z))


def raw_to_world_point(raw_xyz, level_rotation: np.ndarray, floor_z: float, pose: ScanPose) -> np.ndarray:
    """`raw_to_world_points` for one point."""
    return raw_to_world_points(raw_xyz, level_rotation, floor_z, pose)[0]


def pose_shift_m(payload: dict, pose1: ScanPose, pose2: ScanPose) -> float:
    """Mean distance this scan's preview points move between two poses; degrees and metres cannot just be
    subtracted."""
    return float(np.linalg.norm(world_cloud_points(payload, pose1) - world_cloud_points(payload, pose2), axis=1).mean())


def common_motion(before: list[np.ndarray], after: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Yaw (degrees, about the world origin) and 3D shift that best carry every scan's points from before to after
    together. Each scan counts equally however many points it has."""
    weights = np.concatenate([np.full(len(b), 1.0 / len(b)) for b in before])
    return solve_yaw_translation(np.vstack(before), np.vstack(after), weights)


def carry_pose(pose: ScanPose, yaw_deg: float, shift: np.ndarray) -> ScanPose:
    """pose moved along with a whole-layout turn about the world origin and shift, as common_motion returns."""
    xy = yaw_matrix2(yaw_deg) @ np.array([pose.x_m, pose.y_m]) + np.asarray(shift[:2], dtype=float)
    return ScanPose(yaw_deg=pose.yaw_deg + yaw_deg, x_m=float(xy[0]), y_m=float(xy[1]), z_m=pose.z_m + float(shift[2]),
                    pitch_deg=pose.pitch_deg, roll_deg=pose.roll_deg)
