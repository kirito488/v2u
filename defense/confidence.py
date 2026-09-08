"""Perception confidence C = C_abs * V  (occlusion replaces relative density C_n).

C answers: did this sensor see the object clearly?
  C_abs  — enough absolute points in the box?  clip(n / n_ref)
  V      — polar-depth ray visibility (fraction of free rays to the box)

Previously C = C_abs * C_n with C_n = n / n_exp(r, A). That path is kept only
as ``use_cn=True`` for ablation; default is V.

n is counted in the source LiDAR frame.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import math

import numpy as np

from .geometry import count_points_in_box

# §2.1.2  “typical vehicle, seen clearly”
# Probe default for C=C_abs*V: ~25 m ego p10–p25 scale (see logs n@25m).
# Old clear-pool median (~2029) crushed mid-range C_abs under V product.
N_REF_EGO = 200.0
N_REF_UAV = 200.0
# Legacy / optional C_n calibration (unused when use_cn=False)
R_MIN = 4.0
R0 = 10.60
R0_EGO = 10.60
R0_UAV = 15.10
L_REF, W_REF, H_REF = 4.5, 1.8, 1.5
A_REF_EGO: Optional[float] = 5.339
A_REF_UAV: Optional[float] = 10.854

# Default: replace C_n with polar visibility V
USE_CN = False


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
    """§2.1.3  A for n_exp (legacy C_n only). bbox in source LiDAR frame."""
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
    """§2.1.3  A_ref (legacy C_n only)."""
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
    """n_exp = n0 * (A / A_ref) * (r0 / r)^2.  Legacy C_n only."""
    r = max(float(r), 1e-6)
    a_ref = max(float(a_ref), 1e-6)
    return float(n0) * (float(area) / a_ref) * (float(r0) / r) ** 2


def c_abs(n_pts, n_ref) -> float:
    """§2.1.2  clip(n / n_ref, 0, 1)."""
    if n_ref <= 0:
        return 0.0
    return float(np.clip(float(n_pts) / float(n_ref), 0.0, 1.0))


def c_n(n_pts, n_exp) -> float:
    """§2.1.3  clip(n / n_exp, 0, 1). Legacy."""
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
    visibility: Optional[float] = None,
    use_cn: Optional[bool] = None,
) -> float:
    """C = C_abs * V  (default), or C_abs * C_n if use_cn.

    bbox_sensor must be in the source LiDAR frame (geometric-center z).
    ``visibility`` in [0,1] from polar-depth rays; if None and not use_cn, V=1
    (C degenerates to C_abs).
    """
    n_ref = float(n_ref_for(source) if n_ref is None else n_ref)
    n = float(n_pts)
    if n_ref <= 0 or n <= 0:
        return 0.0
    ca = c_abs(n, n_ref)
    use_cn_flag = bool(USE_CN if use_cn is None else use_cn)
    if use_cn_flag:
        n0 = float(n_ref if n0 is None else n0)
        r0_use = float(r0_for(source) if r0 is None else r0)
        a_ref_use = float(area_ref(source) if a_ref is None else a_ref)
        r = range_xy(bbox_sensor, r_min=r_min)
        n_exp = expected_points(
            r, visible_area(bbox_sensor, source), n0, a_ref_use, r0=r0_use
        )
        return float(ca * c_n(n, n_exp))
    v = 1.0 if visibility is None else float(np.clip(visibility, 0.0, 1.0))
    return float(ca * v)


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
    use_cn: Optional[bool] = None,
    return_visibility: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Per-box (C, n[, V]). boxes and lidar in the same sensor frame.

    Builds one polar depth map and sets C = C_abs * V (unless use_cn).
    If bottom_center, z is lifted to the geometric center before counting / V.
    UAV pad is larger (scheme: top-view boxes may need extra pad).
    """
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        empty_c = np.zeros((0,), dtype=np.float64)
        empty_n = np.zeros((0,), dtype=np.int32)
        empty_v = np.zeros((0,), dtype=np.float64)
        if return_visibility:
            return empty_c, empty_n, empty_v, empty_c.copy()
        return empty_c, empty_n, empty_c.copy()
    n_ref = n_ref_for(source) if n_ref is None else float(n_ref)
    if pad is None:
        pad = 0.5 if source == "uav" else 0.25
    boxes_use = boxes_geometric_center(boxes) if bottom_center else boxes
    use_cn_flag = bool(USE_CN if use_cn is None else use_cn)

    depth = None
    if not use_cn_flag:
        try:
            from .occlusion import build_polar_depth

            depth = build_polar_depth(lidar)
        except Exception:
            depth = None

    cs = np.zeros((boxes_use.shape[0],), dtype=np.float64)
    ns = np.zeros((boxes_use.shape[0],), dtype=np.int32)
    vs = np.ones((boxes_use.shape[0],), dtype=np.float64)
    cas = np.zeros((boxes_use.shape[0],), dtype=np.float64)
    for i, b in enumerate(boxes_use):
        n = count_points_in_box(lidar, b, pad=pad)
        ns[i] = n
        ca = c_abs(n, n_ref)
        cas[i] = ca
        v = 1.0
        if not use_cn_flag and depth is not None:
            try:
                from .occlusion import visibility_ratio_lidar

                v = float(
                    visibility_ratio_lidar(
                        b,
                        depth=depth,
                        origin=(0.0, 0.0, 0.0),
                        # ego boxes may be out of eval range; uav frame boxes are not
                        check_ego_range=(source == "ego"),
                        bottom_center=False,  # already geometric-center
                    )
                )
            except Exception:
                v = 0.0
        vs[i] = v
        cs[i] = perception_confidence(
            n,
            b,
            source,
            n_ref=n_ref,
            r0=r0,
            r_min=r_min,
            a_ref=a_ref,
            visibility=None if use_cn_flag else v,
            use_cn=use_cn_flag,
        )
    if return_visibility:
        return cs, ns, vs, cas
    return cs, ns, cas
