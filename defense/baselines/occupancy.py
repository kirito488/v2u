"""CAD occupancy from Ego lidar, matching official geometry.

Official CARLA path uses HD-map ground + SqueezeSeg. V2U4Real has neither.
Attack-repo fallbacks (same functions CAD already ships):
  get_ground_plane_ransac + lidar_segmentation_dbscan
then get_occupied_space / get_free_space / run_core thresholds.
All polygons stay in Ego lidar frame (boxes are also ego-lidar).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .common import box_polygon

CAD_MAX_RANGE = 50.0
CAD_HEIGHT_THRES = 0.0
CAD_HEIGHT_TOL = 0.2


def _empty() -> Polygon:
    return Polygon()


def _as_union(geoms) -> Polygon:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return _empty()
    u = unary_union(geoms)
    if u.is_empty:
        return _empty()
    if not u.is_valid:
        u = u.buffer(0)
    return u


def build_occupancy(ego_lidar, backend: str = "auto") -> Tuple[object, object, dict]:
    """Return (occupied_union, free_union, info) in ego lidar XY."""
    pts = np.asarray(ego_lidar) if ego_lidar is not None else np.zeros((0, 3))
    if pts.ndim != 2 or pts.shape[0] < 8 or pts.shape[1] < 3:
        return _empty(), _empty(), {"n_segments": 0, "backend": "empty"}

    want = str(backend or "auto")
    if want == "fallback":
        occupied, free, n_seg = _fallback_occupancy(pts)
        return occupied, free, {"n_segments": n_seg, "backend": "fallback"}

    try:
        occupied, free, n_seg = _official_occupancy(pts)
        return occupied, free, {"n_segments": n_seg, "backend": "mvp"}
    except Exception as exc:
        occupied, free, n_seg = _fallback_occupancy(pts)
        return occupied, free, {
            "n_segments": n_seg,
            "backend": "fallback",
            "mvp_error": str(exc),
        }


def _official_occupancy(lidar_xyz: np.ndarray):
    from mvp.tools.ground_detection import get_ground_plane_ransac
    from mvp.tools.lidar_seg import lidar_segmentation_dbscan
    from mvp.tools.polygon_space import get_free_space, get_occupied_space
    from mvp.defense.detection_util import filter_segmentation

    pcd = np.asarray(lidar_xyz, dtype=np.float64)
    if pcd.shape[1] < 4:
        pad = np.zeros((pcd.shape[0], 4 - pcd.shape[1]), dtype=np.float64)
        pcd = np.hstack([pcd, pad])

    _, ground_idx = get_ground_plane_ransac(pcd)
    ground_idx = np.asarray(ground_idx, dtype=np.int64).reshape(-1)
    point_height = np.zeros((pcd.shape[0],), dtype=np.float64)
    if ground_idx.size:
        z0 = float(np.median(pcd[ground_idx, 2]))
        point_height = pcd[:, 2] - z0
    in_lane = np.ones((pcd.shape[0],), dtype=bool)

    lidar_seg = lidar_segmentation_dbscan(
        pcd, ground_idx, cluster_thres=0.5, min_point_num=8
    )
    identity_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    object_segments = filter_segmentation(
        pcd,
        lidar_seg,
        identity_pose,
        in_lane_mask=None,
        point_height=point_height,
        max_range=CAD_MAX_RANGE,
    )
    object_mask = np.zeros((pcd.shape[0],), dtype=bool)
    if object_segments:
        object_mask[np.hstack(object_segments)] = True

    occupied_list, _ = get_occupied_space(
        pcd, object_segments, point_height=point_height, height_thres=CAD_HEIGHT_THRES
    )
    free_list = get_free_space(
        pcd,
        identity_pose,
        object_mask,
        in_lane_mask=in_lane,
        point_height=point_height,
        height_thres=CAD_HEIGHT_THRES,
        height_tolerance=CAD_HEIGHT_TOL,
        max_range=CAD_MAX_RANGE,
    )
    return _as_union(occupied_list), _as_union(free_list), len(object_segments)


def _fallback_occupancy(lidar_xyz: np.ndarray):
    """RANSAC-free polar fallback if mvp tools cannot import."""
    xyz = np.asarray(lidar_xyz[:, :3], dtype=np.float64)
    z0 = float(np.percentile(xyz[:, 2], 10))
    height = xyz[:, 2] - z0
    dist = np.hypot(xyz[:, 0], xyz[:, 1])
    obj = (height > 0.3) & (height < 3.0) & (dist < CAD_MAX_RANGE) & (dist > 0.5)
    hits = np.full((180,), CAD_MAX_RANGE, dtype=np.float64)
    if np.any(obj):
        ang = np.floor((np.arctan2(xyz[obj, 1], xyz[obj, 0]) + 2.0 * np.pi) / (2.0 * np.pi) * 180)
        ang = np.clip(ang.astype(np.int32) % 180, 0, 179)
        for a, d in zip(ang.tolist(), dist[obj].tolist()):
            if d < hits[a]:
                hits[a] = d
    ring = []
    step = 2.0 * np.pi / 180.0
    for a in range(180):
        th = a * step
        r = float(hits[a])
        ring.append((r * np.cos(th), r * np.sin(th)))
    free = Polygon(ring)
    if not free.is_valid:
        free = free.buffer(0)
    occupied = _empty()
    if np.any(obj):
        from shapely.geometry import MultiPoint
        hull = MultiPoint(xyz[obj, :2].tolist()).convex_hull
        if hull is not None and (not hull.is_empty) and float(getattr(hull, "area", 0.0) or 0.0) >= 0.5:
            occupied = hull
    return occupied, free if not free.is_empty else _empty(), int(np.any(obj))


def pred_polygons(boxes) -> List[Polygon]:
    return [box_polygon(b) for b in np.asarray(boxes)]
