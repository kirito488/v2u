"""Label attack targets in GT dumps: spoof id, original_id(remove)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

SPOOF_ID = "spoof"
REMOVE_SUFFIX = "(remove)"


def nearest_index(boxes, target, max_dist: float = 4.0) -> int:
    """Index of GT nearest to target in BEV, or -1 if farther than max_dist."""
    if target is None:
        return -1
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return -1
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    d = np.hypot(boxes[:, 0] - t[0], boxes[:, 1] - t[1])
    j = int(np.argmin(d))
    return j if float(d[j]) <= float(max_dist) else -1


def original_oid(oid) -> str:
    s = str(oid)
    if s.endswith(REMOVE_SUFFIX):
        return s[: -len(REMOVE_SUFFIX)]
    return s


def is_car_oid(oid, type_map) -> bool:
    """Car filter that still matches spoof / id(remove)."""
    s = str(oid)
    if s == SPOOF_ID or s.startswith("spoof") or s == "remove" or s.endswith(REMOVE_SUFFIX):
        return True
    type_map = type_map or {}
    return type_map.get(original_oid(s)) == "Car"


def annotate_remove(
    boxes,
    ids: Optional[Sequence],
    target,
    max_dist: float = 15.0,
) -> Tuple[np.ndarray, List]:
    """Tag nearest GT as id(remove), or append the attack box as id=remove."""
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    ids = list(ids or [])
    if target is None:
        return boxes, ids
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    j = nearest_index(boxes, t, max_dist=max_dist)
    if j >= 0:
        orig = original_oid(ids[j] if j < len(ids) else j)
        while j >= len(ids):
            ids.append(str(len(ids)))
        ids[j] = "{}{}".format(orig, REMOVE_SUFFIX)
        return boxes, ids
    # No GT near the remove target: do not invent an empty dump/eval box.
    return boxes, ids


def annotate_spoof(boxes, ids: Optional[Sequence], ghost) -> Tuple[np.ndarray, List]:
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    ids = list(ids or [])
    if ghost is None:
        return boxes, ids
    g = np.asarray(ghost, dtype=np.float64).reshape(1, 7)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return g, [SPOOF_ID]
    return np.vstack([boxes, g]), ids + [SPOOF_ID]


def _target_list(target) -> List[np.ndarray]:
    if target is None:
        return []
    if isinstance(target, (list, tuple)):
        return [
            np.asarray(t, dtype=np.float64).reshape(-1)[:7]
            for t in target
            if t is not None
        ]
    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 7:
        return [arr[i, :7].copy() for i in range(arr.shape[0])]
    return [arr.reshape(-1)[:7]]


def annotate_gt(boxes, ids, mode, target, add_spoof: bool = True) -> Tuple[np.ndarray, List]:
    """Return dump GT with spoof appended or remove target relabeled.

    add_spoof=False: do not inject the ghost into this GT list (ego GT stays clean
    when only UAV lidar is attacked).
    mode: spoof / multi_spoof / remove / mass_remove
    target: one box or a list of boxes.
    """
    if target is None or mode in (None, "none", ""):
        boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
        return boxes, list(ids or [])
    mode = str(mode)
    tgts = _target_list(target)
    if not tgts:
        boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
        return boxes, list(ids or [])
    if mode in ("remove", "mass_remove"):
        for t in tgts:
            boxes, ids = annotate_remove(boxes, ids, t)
        return boxes, ids
    if mode in ("spoof", "multi_spoof"):
        if not add_spoof:
            boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
            return boxes, list(ids or [])
        for i, t in enumerate(tgts):
            boxes, ids = annotate_spoof(boxes, ids, t)
            if i > 0 and ids and ids[-1] == SPOOF_ID:
                ids[-1] = "spoof{}".format(i + 1)
        return boxes, ids
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    return boxes, list(ids or [])
