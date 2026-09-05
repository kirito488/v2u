"""Donor-object and blind-spot/danger-region selection for the V2U4Real attacks.

For the intermediate-level spoofing attack we do not inject point clouds; instead
we borrow the *geometry* (length/width/height + orientation) of real objects from
the dataset (the "donor") and place a fake feature block at a location chosen to be
a **danger region** (in front of the ego in its lane) or a **blind spot** (where the
ego's own LiDAR is sparse / occluded).

Positions are returned in the **ego coordinate frame**, which is what
``OpencoodPerception.attack_intermediate`` expects for its ``bboxes``.
"""

import copy
import numpy as np

from mvp.data.util import bbox_sensor_to_map


def select_donor_boxes(multi_frame_case, attacker_id, rng, K):
    """Pick up to ``K`` real objects from the scene to borrow geometry from.

    Scans every frame of the case, collects the attacker's GT boxes, converts them
    to the world/map frame (to recover a meaningful world yaw), and randomly samples
    ``K`` distinct donors. Returns a list of ``[l, w, h, world_yaw]``.

    Parameters
    ----------
    multi_frame_case : list
        The multi-frame case (each element is an OrderedDict of per-vehicle data).
    attacker_id : str
        The vehicle id of the attacker (donor objects are taken from its view).
    rng : numpy.random.RandomState
        Reproducible RNG.
    K : int
        Number of donors to select.

    Returns
    -------
    list of numpy.ndarray
        ``K`` entries of shape (4,) = [length, width, height, world_yaw(rad)].
    """
    candidates = []
    for frame in multi_frame_case:
        if attacker_id not in frame:
            continue
        att_case = frame[attacker_id]
        if att_case["gt_bboxes"].shape[0] == 0:
            continue
        for bbox in att_case["gt_bboxes"]:
            # Recover world-frame box to get a world-consistent yaw.
            bbox_map = bbox_sensor_to_map(bbox, att_case["lidar_pose"])
            candidates.append(bbox_map)
    if len(candidates) == 0:
        return []

    K = min(K, len(candidates))
    indices = rng.choice(len(candidates), size=K, replace=False)
    donors = []
    for i in indices:
        b = candidates[i]
        donors.append(np.array([b[3], b[4], b[5], b[6]], dtype=np.float64))
    return donors


def _candidate_density(ego_lidar, x, y, radius=2.0):
    """Count ego LiDAR points within a horizontal cylinder around (x, y)."""
    if ego_lidar is None or ego_lidar.shape[0] == 0:
        return 0
    dx = ego_lidar[:, 0] - x
    dy = ego_lidar[:, 1] - y
    dist = np.sqrt(dx * dx + dy * dy)
    return int((dist <= radius).sum())


def select_attack_positions(ego_lidar, ego_gt_bboxes, K, rng,
                            x_min=12.0, x_max=45.0, y_range=12.0,
                            lane_half=3.5, density_thresh=20.0,
                            w_danger=1.0, w_blind=1.0, min_sep=8.0,
                            avoid_gt_dist=6.0):
    """Pick up to ``K`` ego-frame positions that are dangerous or blind spots.

    A candidate grid spans the area ahead of the ego (``x`` forward, ``y`` lateral).
    Each candidate is scored by:

    - ``danger``: 1.0 when inside the ego's forward lane (``|y| <= lane_half``), else 0.
    - ``blind``: ``1 - clip(density / density_thresh, 0, 1)`` where ``density`` is the
      number of ego LiDAR points near the candidate (low density => blind/occluded).

    Final score = ``w_danger * danger + w_blind * blind``. Candidates too close to
    existing ground-truth objects (``avoid_gt_dist``) are skipped, and selected
    positions are spread out by ``min_sep``. Returns ego-frame positions
    ``[[x, y, 0], ...]`` plus the per-candidate scores for debugging.
    """
    if K <= 0:
        return [], []

    ys = np.arange(-y_range, y_range + 1e-9, 1.5)
    xs = np.arange(x_min, x_max + 1e-9, 3.0)

    scored = []
    for x in xs:
        for y in ys:
            # Skip candidates that collide with existing ground truth.
            if ego_gt_bboxes is not None and ego_gt_bboxes.shape[0] > 0:
                if np.any(np.sqrt((ego_gt_bboxes[:, 0] - x) ** 2 +
                                  (ego_gt_bboxes[:, 1] - y) ** 2) < avoid_gt_dist):
                    continue
            density = _candidate_density(ego_lidar, x, y)
            blind = 1.0 - min(float(density) / density_thresh, 1.0)
            danger = 1.0 if abs(y) <= lane_half else 0.0
            score = w_danger * danger + w_blind * blind
            scored.append((score, x, y, density))

    # Highest score first.
    scored.sort(key=lambda t: t[0], reverse=True)

    positions = []
    scores = []
    for score, x, y, density in scored:
        if len(positions) >= K:
            break
        # Enforce a minimum separation between injected objects.
        if len(positions) > 0:
            dist = min(np.sqrt((px - x) ** 2 + (py - y) ** 2) for px, py, _ in positions)
            if dist < min_sep:
                continue
        positions.append([float(x), float(y), 0.0])
        scores.append((score, density, abs(y) <= lane_half))

    # Fill the remainder from the scored list ignoring the separation constraint.
    if len(positions) < K:
        for score, x, y, density in scored:
            if len(positions) >= K:
                break
            if any(abs(px - x) < 1e-6 and abs(py - y) < 1e-6 for px, py, _ in positions):
                continue
            positions.append([float(x), float(y), 0.0])
            scores.append((score, density, abs(y) <= lane_half))

    return positions, scores
