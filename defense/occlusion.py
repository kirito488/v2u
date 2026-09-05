"""BEV ray visibility from a sensor LiDAR origin (default: ego (0,0)).

v = free_rays / M. Out of lidar range → v = 0.
Occluders: other boxes (not the query itself).
`origin` may be any pose in the same (ego) frame, e.g. the UAV LiDAR origin.
"""
from __future__ import annotations

import numpy as np

from .geometry import _bev_corners
from .metrics import lidar_range_mask

V_MIN = 0.25
K_ATK = 10
_HIT_EPS = 0.3


def _as_boxes(boxes) -> np.ndarray:
    a = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if a.ndim == 1 and a.size >= 7:
        return np.asarray(a, dtype=np.float64).reshape(1, -1)[:, :7]
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)[:, :7]


def _sample_points(box) -> np.ndarray:
    """BEV center + four corners."""
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    corners = _bev_corners(b)
    center = np.array([[float(b[0]), float(b[1])]], dtype=np.float64)
    return np.vstack([center, corners])


def _ray_hits_obb(origin, direction, bbox, t_max: float) -> bool:
    """Unit-direction ray vs BEV OBB. Hit if first intersection is in [0, t_max)."""
    x, y = float(bbox[0]), float(bbox[1])
    l, w = float(bbox[3]), float(bbox[4])
    yaw = float(bbox[6]) if len(bbox) > 6 else 0.0
    c, s = np.cos(-yaw), np.sin(-yaw)
    ox, oy = float(origin[0]) - x, float(origin[1]) - y
    lx = c * ox - s * oy
    ly = s * ox + c * oy
    dx, dy = float(direction[0]), float(direction[1])
    ldx = c * dx - s * dy
    ldy = s * dx + c * dy
    tmin = 0.0
    tmax = float(t_max)
    for orig_i, dir_i, half in ((lx, ldx, l / 2.0), (ly, ldy, w / 2.0)):
        if abs(dir_i) < 1e-12:
            if abs(orig_i) > half:
                return False
            continue
        t1 = (-half - orig_i) / dir_i
        t2 = (half - orig_i) / dir_i
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        tmin = max(tmin, lo)
        tmax = min(tmax, hi)
        if tmin > tmax:
            return False
    return tmax >= 0.0 and tmin < float(t_max) - _HIT_EPS


def _is_query(box, other, dist=0.5) -> bool:
    """True if `other` is a duplicate of the query, not a separate occluder."""
    a = np.asarray(box, dtype=np.float64).reshape(-1)
    b = np.asarray(other, dtype=np.float64).reshape(-1)
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))) <= float(dist)


def visibility_ratio(
    box,
    occluders=None,
    origin=(0.0, 0.0),
) -> float:
    """Fraction of rays from ego origin to the box that are not blocked."""
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
    if not bool(lidar_range_mask(b.reshape(1, -1))[0]):
        return 0.0
    pts = _sample_points(b)
    ox, oy = float(origin[0]), float(origin[1])
    occ = _as_boxes(occluders)
    others = []
    for o in occ:
        if _is_query(b, o):
            continue
        others.append(o)

    n_free = 0
    n = int(pts.shape[0])
    for p in pts:
        dx, dy = float(p[0]) - ox, float(p[1]) - oy
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            n_free += 1
            continue
        direction = np.array([dx / dist, dy / dist], dtype=np.float64)
        blocked = False
        for o in others:
            if _ray_hits_obb((ox, oy), direction, o, dist):
                blocked = True
                break
        if not blocked:
            n_free += 1
    return float(n_free) / float(n) if n else 0.0


def is_visible(box, occluders=None, v_min: float = V_MIN, origin=(0.0, 0.0)) -> bool:
    return visibility_ratio(box, occluders=occluders, origin=origin) >= float(v_min)
