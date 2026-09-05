"""CAD (USENIX Security 2024) occupancy consistency on V2U4Real.

Official run_core (mvp/defense/perception_defender.py):
  spoof: pred_box ∩ free_areas, drop if area >= 1.7 m^2
  remove: occupied_area \\ pred_union, recover if area >= 1.7 m^2

Occupancy is built from Ego lidar (RANSAC ground + DBSCAN), not Ego boxes.
See occupancy.py for the CARLA-map / SqueezeSeg substitution.
"""
from __future__ import annotations

import numpy as np

from .common import (
    BaselineResult,
    empty_result,
    nms_boxes,
    source_boxes_scores,
)
from .occupancy import _as_union, _empty, build_occupancy, pred_polygons

CAD_THRES = 1.7


def _area(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area)


def defend_cad(
    ego,
    uav,
    init,
    ego_lidar=None,
    thres: float = CAD_THRES,
    occupancy_backend: str = "auto",
    **_kwargs,
) -> BaselineResult:
    ego_b, ego_s = source_boxes_scores(ego)
    init_b, init_s = source_boxes_scores(init)
    occupied, free, occ_info = build_occupancy(ego_lidar, backend=occupancy_backend)
    try:
        free = free.difference(occupied)
    except Exception:
        pass
    if free is None or free.is_empty:
        free = _empty()

    keep_b, keep_s = [], []
    n_drop_spoof = 0
    for b, s in zip(init_b, init_s):
        poly = pred_polygons([b])[0]
        spoof_err = _area(poly.intersection(free))
        if spoof_err >= float(thres):
            n_drop_spoof += 1
            continue
        keep_b.append(b)
        keep_s.append(float(s))

    pred_union = _as_union(pred_polygons(keep_b)) if keep_b else _empty()
    n_recover = 0
    for b, s in zip(ego_b, ego_s):
        poly = pred_polygons([b])[0]
        # Official: each occupied region vs pred union. We have a union map,
        # so recover an Ego box if it covers leftover occupied area.
        leftover = occupied.intersection(poly).difference(pred_union)
        remove_err = _area(leftover)
        if remove_err >= float(thres):
            keep_b.append(b)
            keep_s.append(float(s))
            n_recover += 1

    info = {
        "name": "cad",
        "thres": float(thres),
        "n_drop_spoof": n_drop_spoof,
        "n_recover": n_recover,
        "occupancy": occ_info,
    }
    if not keep_b:
        out = empty_result()
        info["output"] = "empty"
        out.info = info
        return out
    stacked = np.stack(keep_b)
    scores = np.asarray(keep_s, dtype=np.float64)
    merged = nms_boxes(stacked, scores, iou_thres=0.5)
    info["output"] = "filtered"
    merged.info.update(info)
    return merged
