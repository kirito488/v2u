"""Geometry helpers for box / point-cloud confidence."""
from __future__ import annotations

import numpy as np


def blank_lidar(lidar):
    """Keep one far point so voxel preprocess does not crash on empty cloud."""
    x = np.asarray(lidar)
    if x.ndim != 2 or x.shape[1] < 3:
        return np.zeros((1, 4), dtype=np.float32)
    row = np.zeros((1, x.shape[1]), dtype=x.dtype)
    row[0, 0], row[0, 1], row[0, 2] = 200.0, 200.0, 0.0
    if x.shape[1] > 3:
        row[0, 3] = 1.0
    return row


def points_in_obb(pts, bbox, pad=0.25):
    """Axis-aligned check in box local frame. pts: (N,3), bbox: (7,) center-lwh-yaw.

    OpenCOOD / scheme: (x,y,z) is the geometric center, so z spans ±h/2.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    x, y, z, l, w, h, yaw = [float(v) for v in bbox[:7]]
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    lz = pts[:, 2] - z
    return (
        (np.abs(lx) <= l / 2.0 + pad)
        & (np.abs(ly) <= w / 2.0 + pad)
        & (np.abs(lz) <= h / 2.0 + pad)
    )


def count_points_in_box(lidar, bbox, pad=0.25):
    """Count lidar points inside OBB (center-based). No soft-pad fallback."""
    pts = np.asarray(lidar)
    if pts.ndim != 2 or pts.shape[0] == 0 or bbox is None:
        return 0
    return int(points_in_obb(pts[:, :3], bbox, pad=pad).sum())


def pack_dets(pb, ps):
    if pb is None or len(pb) == 0:
        return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(pb, dtype=np.float64), np.asarray(ps, dtype=np.float64).reshape(-1)


def _bev_corners(bbox):
    x, y, l, w, yaw = float(bbox[0]), float(bbox[1]), float(bbox[3]), float(bbox[4]), float(bbox[6])
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    r = np.array([[c, -s], [s, c]])
    return local.dot(r.T) + np.array([x, y])


def iou_bev(a, b):
    """Oriented bird's-eye IoU for [x,y,z,l,w,h,yaw]. Ignores z."""
    from shapely.geometry import Polygon
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pa, pb = Polygon(_bev_corners(a)), Polygon(_bev_corners(b))
        if (not pa.is_valid) or pa.area <= 0 or (not pb.is_valid) or pb.area <= 0:
            return 0.0
        inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0
