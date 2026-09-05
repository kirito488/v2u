"""Shared helpers for algorithm-level baseline defenses.

ROBOSAC official set IoU is Jaccard after Hungarian matching
(coperception/tools/det/box_matching.py::associate_2_detections).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..associate import hungarian_match
from ..geometry import _bev_corners, iou_bev

# ROBOSAC box_matching.py pair gate (not the CLI Jaccard threshold).
ROBOSAC_PAIR_IOU = 0.5
# ROBOSAC CLI --box_matching_thresh: consensus if Jaccard >= this.
ROBOSAC_JAC_THRES = 0.3


@dataclass
class BaselineResult:
    boxes: np.ndarray
    scores: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.boxes.shape[0]) if self.boxes.ndim == 2 else 0


def empty_result(info: Optional[Dict[str, Any]] = None) -> BaselineResult:
    return BaselineResult(
        boxes=np.zeros((0, 7), dtype=np.float64),
        scores=np.zeros((0,), dtype=np.float64),
        info=dict(info or {}),
    )


def as_boxes7(boxes) -> np.ndarray:
    a = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)[:, :7]


def as_scores(scores, n: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1) if scores is not None else np.zeros((0,))
    if s.shape[0] == n:
        return s
    if s.shape[0] == 0:
        return np.ones((n,), dtype=np.float64)
    out = np.ones((n,), dtype=np.float64)
    m = min(n, int(s.shape[0]))
    out[:m] = s[:m]
    return out


def source_boxes_scores(src):
    boxes = as_boxes7(getattr(src, "boxes", None))
    scores = as_scores(getattr(src, "scores", None), int(boxes.shape[0]))
    return boxes, scores


def set_jaccard(boxes_a, boxes_b, pair_iou: float = ROBOSAC_PAIR_IOU) -> float:
    """Jaccard of two box sets after Hungarian matching (ROBOSAC associate_2_detections).

    intersect = #pairs with IoU >= pair_iou
    union = n_a + n_b - intersect
    """
    a = as_boxes7(boxes_a)
    b = as_boxes7(boxes_b)
    n_a, n_b = int(a.shape[0]), int(b.shape[0])
    if n_a == 0 and n_b == 0:
        return 1.0
    if n_a == 0 or n_b == 0:
        return 0.0
    pairs = hungarian_match(a, b, iou_thres=float(pair_iou))
    inter = len(pairs)
    union = n_a + n_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def box_polygon(bbox):
    from shapely.geometry import Polygon

    poly = Polygon(_bev_corners(bbox))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def nms_boxes(boxes, scores, iou_thres: float = 0.5) -> BaselineResult:
    boxes = as_boxes7(boxes)
    scores = as_scores(scores, int(boxes.shape[0]))
    n = int(boxes.shape[0])
    if n == 0:
        return empty_result()
    order = np.argsort(-scores)
    keep = []
    for i in order:
        b = boxes[i]
        if any(float(iou_bev(b, boxes[j])) >= float(iou_thres) for j in keep):
            continue
        keep.append(int(i))
    keep = np.asarray(keep, dtype=np.int64)
    return BaselineResult(boxes=boxes[keep], scores=scores[keep], info={"n_nms": int(keep.size)})


def raster_occupancy(boxes, x_max: float = 100.8, y_max: float = 80.0, res: float = 0.8) -> np.ndarray:
    """Binary BEV occupancy from oriented boxes (CP-Guard CCLoss analog)."""
    nx = int(np.ceil(2.0 * x_max / res))
    ny = int(np.ceil(2.0 * y_max / res))
    grid = np.zeros((ny, nx), dtype=np.float64)
    boxes = as_boxes7(boxes)
    if boxes.shape[0] == 0:
        return grid
    xs = np.linspace(-x_max + res * 0.5, x_max - res * 0.5, nx)
    ys = np.linspace(-y_max + res * 0.5, y_max - res * 0.5, ny)
    xx, yy = np.meshgrid(xs, ys)
    for b in boxes:
        x, y, l, w, yaw = float(b[0]), float(b[1]), float(b[3]), float(b[4]), float(b[6])
        c, s = np.cos(-yaw), np.sin(-yaw)
        dx, dy = xx - x, yy - y
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        grid[(np.abs(lx) <= l / 2.0) & (np.abs(ly) <= w / 2.0)] = 1.0
    return grid


def occupancy_ccloss(occ_ego, occ_other) -> float:
    """CP-Guard CCLoss: sum(A*B) / sum(A+B) on occupancy maps."""
    a = np.asarray(occ_ego, dtype=np.float64)
    b = np.asarray(occ_other, dtype=np.float64)
    num = float(np.sum(a * b))
    den = float(np.sum(a + b))
    return num / den if den > 0 else 0.0
