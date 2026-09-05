"""UAV LiDAR coarse FOV from val-set percentiles (sensor frame).

Calibrated on V2U4Real val (every 5th multi_frame case, mid frame; ~1.65M pts):
  r_horiz / azimuth / elevation at p5–p95 in the UAV LiDAR frame.

Pool ATTACK sight:
  inside FOV  → max(LOS_lidar, C_uav)   (polar-depth occlusion + density)
  outside FOV → C_uav only              (blind zone: empty depth is meaningless)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Horizontal range (m): upper = p95; lower open (near field is in FOV)
UAV_FOV_R_P5 = 7.17   # recorded for reference; not used as hard min
UAV_FOV_R_P95 = 69.39
UAV_FOV_R_LO = 0.0
UAV_FOV_R_HI = UAV_FOV_R_P95
# Azimuth deg in UAV frame, atan2(y, x) — p5 / p95
UAV_FOV_AZ_P5 = -141.87
UAV_FOV_AZ_P95 = 142.18
# Elevation deg, atan2(z, r_horiz) — p5 / p95 (top-down: strongly negative)
UAV_FOV_EL_P5 = -81.20
UAV_FOV_EL_P95 = -28.94


def point_polar_uav(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Return (r_horiz_m, az_deg, el_deg) for a point in the UAV LiDAR frame."""
    r = float(np.hypot(x, y))
    az = float(np.degrees(np.arctan2(y, x)))
    el = float(np.degrees(np.arctan2(z, max(r, 1e-6))))
    return r, az, el


def in_uav_fov_xyz(
    x: float,
    y: float,
    z: float,
    r_lo: float = UAV_FOV_R_LO,
    r_hi: float = UAV_FOV_R_HI,
    az_lo: float = UAV_FOV_AZ_P5,
    az_hi: float = UAV_FOV_AZ_P95,
    el_lo: float = UAV_FOV_EL_P5,
    el_hi: float = UAV_FOV_EL_P95,
) -> bool:
    """True if (x,y,z) in UAV frame lies in the percentile FOV box."""
    r, az, el = point_polar_uav(x, y, z)
    if r < float(r_lo) or r > float(r_hi):
        return False
    if az < float(az_lo) or az > float(az_hi):
        return False
    if el < float(el_lo) or el > float(el_hi):
        return False
    return True


def _box_center_to_uav(box_ego, pose_ego, pose_uav) -> Optional[np.ndarray]:
    """Geometric center of ego-frame box → UAV LiDAR xyz."""
    from opencood.utils.transformation_utils import x1_to_x2

    from .confidence import boxes_geometric_center

    b = np.asarray(box_ego, dtype=np.float64).reshape(-1)
    if b.size < 7:
        return None
    bc = boxes_geometric_center(b[:7].reshape(1, 7))[0]
    T = x1_to_x2(
        np.asarray(pose_ego).reshape(-1).tolist(),
        np.asarray(pose_uav).reshape(-1).tolist(),
    )
    q = T.dot(np.array([float(bc[0]), float(bc[1]), float(bc[2]), 1.0]))
    return np.array([q[0], q[1], q[2]], dtype=np.float64)


def box_in_uav_fov(
    box_ego,
    frame,
    ego_id: str = "1",
    uav_id: str = "2",
    **kwargs,
) -> Optional[bool]:
    """Warp ego-frame box center into UAV LiDAR frame, then FOV-test.

    Returns None if poses/box missing (caller should fall back safely).
    """
    if frame is None or ego_id not in frame or uav_id not in frame:
        return None
    pose_e = frame[ego_id].get("lidar_pose")
    pose_u = frame[uav_id].get("lidar_pose")
    if pose_e is None or pose_u is None:
        return None
    try:
        xyz = _box_center_to_uav(box_ego, pose_e, pose_u)
        if xyz is None:
            return None
        return in_uav_fov_xyz(float(xyz[0]), float(xyz[1]), float(xyz[2]), **kwargs)
    except Exception:
        return None
