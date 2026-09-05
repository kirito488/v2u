"""Hungarian matching for three-source association (scheme §4).

Pairs two box sets with cost 1 - BEV IoU. A pair is the same object only if
IoU >= theta_iou (ROBOSAC box_matching_thresh = 0.3).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .geometry import iou_bev

THETA_IOU = 0.3
# Pool-only: same object if IoU>=θ_iou OR BEV center distance ≤θ_d (fast lateral motion).
THETA_DIST = 4.0


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    a = np.asarray(boxes_a)
    b = np.asarray(boxes_b)
    n, m = (0 if a.ndim != 2 else a.shape[0], 0 if b.ndim != 2 else b.shape[0])
    out = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            out[i, j] = iou_bev(a[i], b[j])
    return out


def _assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Row/col indices of a min-cost bipartite matching."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:
        return _greedy_assignment(cost)


def _greedy_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback if scipy is missing: repeatedly take the globally cheapest pair."""
    n, m = cost.shape
    used_r = np.zeros(n, dtype=bool)
    used_c = np.zeros(m, dtype=bool)
    rows, cols = [], []
    work = cost.copy()
    for _ in range(min(n, m)):
        work[used_r, :] = np.inf
        work[:, used_c] = np.inf
        k = int(np.argmin(work))
        i, j = divmod(k, m)
        if not np.isfinite(work[i, j]):
            break
        used_r[i] = True
        used_c[j] = True
        rows.append(i)
        cols.append(j)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def hungarian_match(
    boxes_a,
    boxes_b,
    iou_thres: float = THETA_IOU,
    min_iou_rows=None,
) -> List[Tuple[int, int, float]]:
    """Return (i, j, iou) pairs with IoU >= iou_thres (scheme §4.1).

    min_iou_rows: optional per-row floor (e.g. trust-pool covariance gate).
    A pair is kept only if IoU >= max(iou_thres, min_iou_rows[i]).
    """
    ious = iou_matrix(boxes_a, boxes_b)
    if ious.size == 0:
        return []
    floors = None
    if min_iou_rows is not None:
        floors = np.asarray(min_iou_rows, dtype=np.float64).reshape(-1)
        gated = ious.copy()
        for i in range(min(gated.shape[0], len(floors))):
            gated[i, gated[i] < float(floors[i])] = 0.0
        ious_use = gated
    else:
        ious_use = ious
    ri, cj = _assignment(1.0 - ious_use)
    pairs = []
    for i, j in zip(ri.tolist(), cj.tolist()):
        v = float(ious[i, j])
        thr = float(iou_thres)
        if floors is not None and i < len(floors):
            thr = max(thr, float(floors[i]))
        if v >= thr:
            pairs.append((int(i), int(j), v))
    return pairs


def hungarian_match_pool(
    boxes_a,
    boxes_b,
    iou_thres: float = THETA_IOU,
    dist_thres: float = THETA_DIST,
) -> List[Tuple[int, int, float]]:
    """Pool association: pair if IoU>=θ_iou or BEV center distance ≤θ_d.

    Returns (i, j, iou). Assignment prefers higher IoU, then smaller distance.
    """
    a = np.asarray(boxes_a)
    b = np.asarray(boxes_b)
    n = 0 if a.ndim != 2 else int(a.shape[0])
    m = 0 if b.ndim != 2 else int(b.shape[0])
    if n == 0 or m == 0:
        return []
    ious = iou_matrix(a, b)
    dxy = np.hypot(
        a[:, 0:1] - b[:, 0].reshape(1, -1),
        a[:, 1:2] - b[:, 1].reshape(1, -1),
    )
    aff = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou = float(ious[i, j])
            d = float(dxy[i, j])
            if iou >= float(iou_thres):
                aff[i, j] = iou
            elif d <= float(dist_thres):
                aff[i, j] = 0.5 * (1.0 - d / max(float(dist_thres), 1e-6))
    ri, cj = _assignment(1.0 - aff)
    pairs = []
    for i, j in zip(ri.tolist(), cj.tolist()):
        if float(aff[i, j]) > 0.0:
            pairs.append((int(i), int(j), float(ious[i, j])))
    return pairs


def same_object(
    box_a,
    box_b,
    iou_thres: float = THETA_IOU,
    dist_thres: float = THETA_DIST,
) -> bool:
    a = np.asarray(box_a, dtype=np.float64).reshape(-1)
    b = np.asarray(box_b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return False
    if float(iou_bev(a, b)) >= float(iou_thres):
        return True
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) <= float(dist_thres)


def match_indices(
    boxes_a,
    boxes_b,
    iou_thres: float = THETA_IOU,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each box in A, matched index in B or -1; likewise for B."""
    a = np.asarray(boxes_a)
    b = np.asarray(boxes_b)
    na = 0 if a.ndim != 2 else int(a.shape[0])
    nb = 0 if b.ndim != 2 else int(b.shape[0])
    a2b = np.full((na,), -1, dtype=np.int32)
    b2a = np.full((nb,), -1, dtype=np.int32)
    for i, j, _ in hungarian_match(a, b, iou_thres=iou_thres):
        a2b[i] = j
        b2a[j] = i
    return a2b, b2a
