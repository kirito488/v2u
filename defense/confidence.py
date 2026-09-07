"""Perception confidence C = C_abs * C_n  (UAV_UGV_DEFENSE_SCHEME.html §2.1).

C answers: did this sensor see the object clearly?
  C_abs  — enough absolute points in the box?
  C_n    — enough points relative to the expected count at this range?

n is counted in the source LiDAR frame. r is range to that source origin.

Visible area A (scheme geometry, not a printed PDV/GACE formula):
  Ego (side):  A = h * (l |sin(θ−φ)| + w |cos(θ−φ)|),  φ = atan2(y, x)
  UAV (top):   A = l * w
Old A_ego = h * max(l, w) ignored heading and is not used.
"""
from __future__ import annotations

from typing import Optional

import math

import numpy as np

from .geometry import count_points_in_box

# §2.1.2  “typical vehicle, seen clearly”
# Calibrated 2026-09-07 on val clear-car pool (V≥0.9, r∈[8,40], median κ):
#   logs/calibrate_confidence.json  (ego n_pool=113, uav n_pool=462)
N_REF_EGO = 2029.72
N_REF_UAV = 330.02
# §2.1.3  near-field floor; reference range (per source after calib)
R_MIN = 4.0
R0 = 10.60
R0_EGO = 10.60
R0_UAV = 15.10
# Typical passenger car (m) for A / A_ref (fallback if A_REF_* unset)
L_REF, W_REF, H_REF = 4.5, 1.8, 1.5
# Direct A_ref overrides (m^2). Calibrated medians from clear-car pool.
A_REF_EGO: Optional[float] = 5.339
A_REF_UAV: Optional[float] = 10.854


def n_ref_for(source: str) -> float:
    return N_REF_UAV if source == "uav" else N_REF_EGO


def r0_for(source: str) -> float:
    return float(R0_UAV if source == "uav" else R0_EGO)


def boxes_geometric_center(boxes):
    """Detector boxes from OpencoodPerception are bottom-center (z -= h/2).

    Scheme n uses the geometric center, same as OpenCOOD GT. Lift z by h/2
    in the frame the box is currently in, before counting or warping.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 6:
        return boxes
    out = boxes.copy()
    out[:, 2] = out[:, 2] + 0.5 * out[:, 5]
    return out


def visible_area(bbox, source: str) -> float:
    """§2.1.3  A for n_exp. bbox must be in the source LiDAR frame."""
    l, w, h = float(bbox[3]), float(bbox[4]), float(bbox[5])
    if source == "uav":
        return max(l * w, 1e-6)
    x, y = float(bbox[0]), float(bbox[1])
    yaw = float(bbox[6]) if len(bbox) > 6 else 0.0
    bearing = math.atan2(y, x)
    rel = yaw - bearing
    face = l * abs(math.sin(rel)) + w * abs(math.cos(rel))
    return max(h * face, 1e-6)


def area_ref(source: str) -> float:
    """§2.1.3  A_ref: Ego side ≈ h_ref * l_ref; UAV top ≈ l_ref * w_ref.

    Prefer calibrated A_REF_EGO / A_REF_UAV when set.
    """
    if source == "uav":
        if A_REF_UAV is not None:
            return max(float(A_REF_UAV), 1e-6)
        return max(L_REF * W_REF, 1e-6)
    if A_REF_EGO is not None:
        return max(float(A_REF_EGO), 1e-6)
    return max(H_REF * L_REF, 1e-6)


def apply_calibration(
    *,
    n_ref_ego: Optional[float] = None,
    n_ref_uav: Optional[float] = None,
    r0_ego: Optional[float] = None,
    r0_uav: Optional[float] = None,
    a_ref_ego: Optional[float] = None,
    a_ref_uav: Optional[float] = None,
) -> dict:
    """Set module-level C calibration (used by perception_confidence)."""
    global N_REF_EGO, N_REF_UAV, R0, R0_EGO, R0_UAV, A_REF_EGO, A_REF_UAV
    if n_ref_ego is not None:
        N_REF_EGO = float(n_ref_ego)
    if n_ref_uav is not None:
        N_REF_UAV = float(n_ref_uav)
    if r0_ego is not None:
        R0_EGO = float(r0_ego)
        R0 = float(r0_ego)
    if r0_uav is not None:
        R0_UAV = float(r0_uav)
    if a_ref_ego is not None:
        A_REF_EGO = float(a_ref_ego)
    if a_ref_uav is not None:
        A_REF_UAV = float(a_ref_uav)
    return {
        "N_REF_EGO": N_REF_EGO,
        "N_REF_UAV": N_REF_UAV,
        "R0_EGO": R0_EGO,
        "R0_UAV": R0_UAV,
        "A_REF_EGO": A_REF_EGO,
        "A_REF_UAV": A_REF_UAV,
    }


def range_xy(bbox, r_min: float = R_MIN) -> float:
    """r = max(sqrt(x^2+y^2), r_min) from the sensor that owns this box."""
    return max(math.hypot(float(bbox[0]), float(bbox[1])), float(r_min))


def expected_points(r, area, n0, a_ref, r0: float = R0) -> float:
    """n_exp = n0 * (A / A_ref) * (r0 / r)^2."""
    r = max(float(r), 1e-6)
    a_ref = max(float(a_ref), 1e-6)
    return float(n0) * (float(area) / a_ref) * (float(r0) / r) ** 2


def c_abs(n_pts, n_ref) -> float:
    """§2.1.2  clip(n / n_ref, 0, 1)."""
    if n_ref <= 0:
        return 0.0
    return float(np.clip(float(n_pts) / float(n_ref), 0.0, 1.0))


def c_n(n_pts, n_exp) -> float:
    """§2.1.3  clip(n / n_exp, 0, 1)."""
    if n_exp <= 0:
        return 0.0
    return float(np.clip(float(n_pts) / float(n_exp), 0.0, 1.0))


def density_confidence(n_pts, n_ref=N_REF_EGO):
    """C_abs only (kept for callers / ablation)."""
    return c_abs(n_pts, n_ref)


def perception_confidence(
    n_pts,
    bbox_sensor,
    source: str,
    n_ref: Optional[float] = None,
    n0: Optional[float] = None,
    r0: Optional[float] = None,
    r_min: float = R_MIN,
    a_ref: Optional[float] = None,
) -> float:
    """§2.1.4  C = C_abs * C_n.

    bbox_sensor must be in the source LiDAR frame (geometric-center z).
    Default calibration: n0 = n_ref, r0 = r0_for(source).
    """
    n_ref = float(n_ref_for(source) if n_ref is None else n_ref)
    n0 = float(n_ref if n0 is None else n0)
    r0_use = float(r0_for(source) if r0 is None else r0)
    a_ref_use = float(area_ref(source) if a_ref is None else a_ref)
    n = float(n_pts)
    if n_ref <= 0 or n <= 0:
        return 0.0
    r = range_xy(bbox_sensor, r_min=r_min)
    n_exp = expected_points(
        r, visible_area(bbox_sensor, source), n0, a_ref_use, r0=r0_use
    )
    return float(c_abs(n, n_ref) * c_n(n, n_exp))


def confidences_for_boxes(
    boxes,
    lidar,
    source: str = "ego",
    n_ref: Optional[float] = None,
    pad: Optional[float] = None,
    r0: Optional[float] = None,
    r_min: float = R_MIN,
    bottom_center: bool = True,
    a_ref: Optional[float] = None,
):
    """Per-box (C, n). boxes and lidar in the same sensor frame.

    If bottom_center, z is lifted to the geometric center before counting.
    UAV pad is larger (scheme: top-view boxes may need extra pad).
    """
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    n_ref = n_ref_for(source) if n_ref is None else float(n_ref)
    if pad is None:
        pad = 0.5 if source == "uav" else 0.25
    boxes_use = boxes_geometric_center(boxes) if bottom_center else boxes
    cs = np.zeros((boxes_use.shape[0],), dtype=np.float64)
    ns = np.zeros((boxes_use.shape[0],), dtype=np.int32)
    for i, b in enumerate(boxes_use):
        n = count_points_in_box(lidar, b, pad=pad)
        ns[i] = n
        cs[i] = perception_confidence(
            n, b, source, n_ref=n_ref, r0=r0, r_min=r_min, a_ref=a_ref
        )
    return cs, ns
