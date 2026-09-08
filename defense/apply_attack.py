"""Apply AdvCollaborativePerception spoof/remove onto a V2U4Real case."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .paths import ATTACK_ROOT, UAV_ID, EGO_ID

# Per-frame target: one box, a list of boxes, or None.
FrameTarget = Optional[Union[np.ndarray, List[np.ndarray]]]


def _cuda_empty_cache(tag: str = "") -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if tag:
                print("[cuda] empty_cache ({})".format(tag), flush=True)
    except Exception:
        pass


def _import_attack_script():
    scripts = os.path.join(ATTACK_ROOT, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import attack_v2u4real_online as atk  # noqa: WPS433

    return atk


def normalize_targets(target: FrameTarget) -> List[np.ndarray]:
    """Flatten a per-frame target into a list of 7-d boxes."""
    if target is None:
        return []
    if isinstance(target, (list, tuple)):
        out = []
        for t in target:
            if t is None:
                continue
            out.append(np.asarray(t, dtype=np.float64).reshape(-1)[:7].copy())
        return out
    arr = np.asarray(target, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 7:
        return [arr[i, :7].copy() for i in range(arr.shape[0])]
    return [arr.reshape(-1)[:7].copy()]


def primary_target(target: FrameTarget) -> Optional[np.ndarray]:
    ts = normalize_targets(target)
    return ts[0] if ts else None


def _pack_dets(pb, ps):
    if pb is None or len(pb) == 0:
        return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(pb, dtype=np.float64), np.asarray(ps, dtype=np.float64).reshape(-1)


def _run_late_spoof(perception, multi_frame_case, frame_ids, ego_id, att_id, atk):
    """Naive late-fusion spoof: inject ghost into attacker CAV detections then fuse."""
    bbox_ego = atk._select_spoof_ghost(multi_frame_case, frame_ids, ego_id)
    print(
        "[attack] late spoof ghost ego ({:.1f}, {:.1f})".format(
            float(bbox_ego[0]), float(bbox_ego[1])
        ),
        flush=True,
    )
    attacked = []
    ghosts = []
    for f in frame_ids:
        frame = multi_frame_case[f]
        result = perception.attack_late(
            frame, ego_id, att_id, mode="spoof", bbox=np.asarray(bbox_ego, dtype=np.float64).copy()
        )
        attacked.append(_pack_dets(result["pred_bboxes"], result["pred_scores"]))
        ghosts.append(bbox_ego.copy())
    return attacked, ghosts


def _run_late_remove(perception, multi_frame_case, frame_ids, ego_id, att_id, atk, clean):
    """Naive late-fusion remove: drop attacker dets near the selected target."""
    target_att, target_ego, _ = atk._plan_remove_targets(
        multi_frame_case, frame_ids, ego_id, att_id, clean
    )
    if target_att is None:
        return None, None
    attacked = []
    for fi, f in enumerate(frame_ids):
        tgt = target_ego[fi]
        if tgt is None:
            pb, ps = perception.run(multi_frame_case[f], ego_id)
            attacked.append(_pack_dets(pb, ps))
            continue
        result = perception.attack_late(
            multi_frame_case[f],
            ego_id,
            att_id,
            mode="remove",
            bbox=np.asarray(tgt, dtype=np.float64).copy(),
        )
        attacked.append(_pack_dets(result["pred_bboxes"], result["pred_scores"]))
    t0 = next((t for t in target_ego if t is not None), None)
    if t0 is not None:
        print(
            "[attack] late remove target ego ({:.1f}, {:.1f})".format(
                float(t0[0]), float(t0[1])
            ),
            flush=True,
        )
    return attacked, target_ego


def _run_multi_spoof(
    perception, dataset, multi_frame_case, frame_ids, ego_id, att_id, iters, n_objects
):
    """Intermediate multi-ghost spoof (K fake objects per frame)."""
    from mvp.attack.lidar_spoof_intermediate_multi_attacker import (
        LidarSpoofIntermediateMultiAttacker,
    )

    K = max(1, int(n_objects))
    print("[attack] multi_spoof intermediate K={} iters={}".format(K, iters), flush=True)
    attacker = LidarSpoofIntermediateMultiAttacker(
        perception, dataset, num_objects=K, step=iters
    )
    _, info = attacker.run(
        multi_frame_case,
        {
            "frame_ids": frame_ids,
            "attacker_vehicle_id": att_id,
            "victim_vehicle_id": ego_id,
            "num_objects": K,
        },
    )
    attacked = []
    ghosts: List[FrameTarget] = []
    for f in frame_ids:
        row = (info[f] if f < len(info) else {}) or {}
        ego_row = row.get(ego_id) or {}
        pb = ego_row.get("pred_bboxes")
        ps = ego_row.get("pred_scores")
        attacked.append(_pack_dets(pb, ps))
        inj = ego_row.get("injected_bboxes") or []
        if inj:
            ghosts.append([np.asarray(b, dtype=np.float64)[:7].copy() for b in inj])
        else:
            ghosts.append(None)
    n_ok = sum(1 for g in ghosts if g)
    print("[attack] multi_spoof frames with ghosts={}/{}".format(n_ok, len(frame_ids)), flush=True)
    return attacked, ghosts


def _det_matches_gt(
    box,
    gt_boxes,
    iou_thres: float = 0.3,
    max_dist: float = 4.0,
) -> bool:
    """True if box overlaps any GT by IoU or is within max_dist in BEV."""
    from .geometry import iou_bev

    gt = np.asarray(gt_boxes) if gt_boxes is not None else np.zeros((0, 7))
    if gt.ndim != 2 or gt.shape[0] == 0:
        return False
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
    for g in gt:
        if iou_bev(b, g) >= float(iou_thres):
            return True
        if float(np.hypot(b[0] - g[0], b[1] - g[1])) <= float(max_dist):
            return True
    return False


def _pick_mass_targets(
    clean_dets,
    k: int,
    min_score: float = 0.3,
    gt_boxes=None,
    iou_thres: float = 0.3,
    max_dist: float = 4.0,
    min_range: float = 8.0,
) -> List[np.ndarray]:
    """Top-K clean fusion boxes that match GT (skip Init FPs / near-ego)."""
    if clean_dets is None:
        return []
    boxes, scores = clean_dets
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    scores = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((0,))
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return []
    keep = []
    for i in range(len(scores)):
        if float(scores[i]) < float(min_score):
            continue
        b = boxes[i][:7]
        if float(np.hypot(b[0], b[1])) < float(min_range):
            continue
        if not _det_matches_gt(b, gt_boxes, iou_thres=iou_thres, max_dist=max_dist):
            continue
        keep.append(i)
    keep = sorted(keep, key=lambda i: -float(scores[i]))[: max(1, int(k))]
    return [boxes[i][:7].copy() for i in keep]


def _run_mass_remove(
    perception, multi_frame_case, frame_ids, ego_id, att_id, iters, n_objects, clean
):
    """Suppress K clean detections jointly via intermediate remove PGD."""
    from .three_source import build_v2u4_eval_gt
    from .paths import UAV_ID as _UAV

    K = max(1, int(n_objects))
    print("[attack] mass_remove intermediate K={} iters={}".format(K, iters), flush=True)
    attacked = []
    targets: List[FrameTarget] = []
    n_skip_fp = 0
    for fi, f in enumerate(frame_ids):
        frame = multi_frame_case[f]
        gt_boxes, _ = build_v2u4_eval_gt(frame, ego_id, att_id or _UAV)
        raw = clean[fi] if fi < len(clean) else None
        if raw is not None:
            boxes, scores = raw
            boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
            n_raw = int(boxes.shape[0]) if boxes.ndim == 2 else 0
        else:
            n_raw = 0
        tgts = _pick_mass_targets(raw, K, gt_boxes=gt_boxes)
        if n_raw > 0 and not tgts:
            n_skip_fp += 1
        if not tgts:
            pb, ps = perception.run(frame, ego_id)
            attacked.append(_pack_dets(pb, ps))
            targets.append(None)
            continue
        result = perception.attack_intermediate(
            frame,
            ego_id,
            att_id,
            mode="remove",
            bboxes=tgts,
            max_iteration=iters,
            lr=0.05,
            feature_size=5,
        )
        attacked.append(_pack_dets(result["pred_bboxes"], result["pred_scores"]))
        targets.append(tgts)
        _cuda_empty_cache()
    n_ok = sum(1 for t in targets if t)
    print(
        "[attack] mass_remove frames with targets={}/{} (skipped no-GT/near-ego={})".format(
            n_ok, len(frame_ids), n_skip_fp
        ),
        flush=True,
    )
    return attacked, targets


def apply_attack(
    mode: str,
    level: str,
    perception,
    dataset,
    multi_frame_case: Dict[int, Any],
    frame_ids: List[int],
    ego_id: str = EGO_ID,
    att_id: str = UAV_ID,
    iters: int = 20,
    also_ego: bool = False,
    ghost: str = "near",
    dense: int = 3,
    n_objects: int = 3,
    verbose: bool = False,
    early_remove_mode: str = "box",
) -> Tuple[Dict[int, Any], List[FrameTarget], Optional[List]]:
    """Return (case_for_three_source, per-frame targets, init_override).

    init_override is fusion (boxes, scores) for intermediate/late, else None.
    targets[i] is one box, a list of boxes, or None for frame_ids[i].

    Modes: spoof | remove | multi_spoof | mass_remove
    Levels: early | intermediate | late
      - multi_spoof / mass_remove: intermediate only
      - late: spoof / remove only (naive late fusion)
    """
    if mode in (None, "none", ""):
        return multi_frame_case, [None] * len(frame_ids), None

    mode = str(mode)
    level = str(level or "early")
    if mode in ("multi_spoof", "mass_remove") and level != "intermediate":
        print(
            "[attack] {!r} forces level=intermediate (was {!r})".format(mode, level),
            flush=True,
        )
        level = "intermediate"
    if level == "late" and mode not in ("spoof", "remove"):
        raise ValueError("late level only supports spoof/remove, got {!r}".format(mode))

    atk = _import_attack_script()
    atk.VERBOSE = bool(verbose)

    print(
        "[attack] clean baseline for target selection (mode={} level={} n_objects={} early_remove={}) ...".format(
            mode, level, n_objects,
            early_remove_mode if (mode == "remove" and level == "early") else "-",
        ),
        flush=True,
    )
    clean, _ = atk.run_clean(perception, multi_frame_case, frame_ids, ego_id)

    case = multi_frame_case
    targets: List[FrameTarget] = [None] * len(frame_ids)
    init_override = None

    if mode == "spoof":
        if level == "early":
            _, ghosts, case = atk.run_spoof_early_points(
                perception,
                dataset,
                multi_frame_case,
                frame_ids,
                ego_id,
                att_id,
                dense=dense,
                also_ego=also_ego,
                clean=clean,
                ghost=ghost,
            )
            targets = list(ghosts)
        elif level == "late":
            attacked, ghosts = _run_late_spoof(
                perception, multi_frame_case, frame_ids, ego_id, att_id, atk
            )
            targets = list(ghosts)
            init_override = attacked
        else:
            attacked, ghosts = atk.run_spoof_online(
                perception, dataset, multi_frame_case, frame_ids, ego_id, att_id, iters
            )
            targets = list(ghosts)
            init_override = attacked
        g0 = primary_target(next((t for t in targets if t is not None), None))
        if g0 is not None:
            print(
                "[attack] spoof ghost ego ({:.1f}, {:.1f})".format(float(g0[0]), float(g0[1])),
                flush=True,
            )
    elif mode == "remove":
        if level == "early":
            attacked, target_ego, case = atk.run_remove_early_points(
                perception,
                dataset,
                multi_frame_case,
                frame_ids,
                ego_id,
                att_id,
                dense=dense,
                also_ego=also_ego,
                clean=clean,
                remove_mode=early_remove_mode,
            )
            if attacked is None:
                print("[attack] no remove target, keep clean case", flush=True)
                return multi_frame_case, targets, None
            targets = list(target_ego)
        elif level == "late":
            attacked, target_ego = _run_late_remove(
                perception, multi_frame_case, frame_ids, ego_id, att_id, atk, clean
            )
            if attacked is None:
                print("[attack] no remove target, keep clean case", flush=True)
                return multi_frame_case, targets, None
            targets = list(target_ego)
            init_override = attacked
        else:
            attacked, target_ego = atk.run_remove_online(
                perception,
                dataset,
                multi_frame_case,
                frame_ids,
                ego_id,
                att_id,
                iters,
                clean=clean,
            )
            if attacked is None:
                print("[attack] no remove target, keep clean case", flush=True)
                return multi_frame_case, targets, None
            targets = list(target_ego)
            init_override = attacked
        t0 = primary_target(next((t for t in targets if t is not None), None))
        if t0 is not None:
            print(
                "[attack] remove target ego ({:.1f}, {:.1f})".format(
                    float(t0[0]), float(t0[1])
                ),
                flush=True,
            )
    elif mode == "multi_spoof":
        attacked, ghosts = _run_multi_spoof(
            perception,
            dataset,
            multi_frame_case,
            frame_ids,
            ego_id,
            att_id,
            iters,
            n_objects,
        )
        targets = list(ghosts)
        init_override = attacked
    elif mode == "mass_remove":
        attacked, tgts = _run_mass_remove(
            perception,
            multi_frame_case,
            frame_ids,
            ego_id,
            att_id,
            iters,
            n_objects,
            clean,
        )
        targets = list(tgts)
        init_override = attacked
    else:
        raise ValueError("unknown attack mode {!r}".format(mode))

    return case, targets, init_override


def _case_as_list(multi_frame_case, ids: List[int]) -> list:
    """Frames for ids as a plain list (Adv attackers index by 0..n-1)."""
    return [multi_frame_case[int(f)] for f in ids]


def _write_case_frames(multi_frame_case, ids: List[int], sub) -> None:
    """Write remapped sub-case frames back onto absolute ids."""
    for i, f in enumerate(ids):
        multi_frame_case[int(f)] = sub[i]


def apply_attack_by_case_spans(
    mode: str,
    level: str,
    perception,
    dataset,
    multi_frame_case,
    frame_ids: List[int],
    case_frame_spans: List[tuple],
    ego_id: str = EGO_ID,
    att_id: str = UAV_ID,
    iters: int = 20,
    also_ego: bool = False,
    ghost: str = "near",
    dense: int = 3,
    n_objects: int = 3,
    verbose: bool = False,
    early_remove_mode: str = "box",
):
    """Apply attack per 10-frame case span, then keep one stitched case.

    Defense (pool/buffer) runs on the full stitched sequence. Adv attackers
    hardcode info/positions indexed by frame_id in 0..9, so each chunk is
    remapped to a 0-based sub-case before calling apply_attack.
    """
    if mode in (None, "none", "") or not case_frame_spans:
        return apply_attack(
            mode, level, perception, dataset, multi_frame_case, frame_ids,
            ego_id=ego_id, att_id=att_id, iters=iters, also_ego=also_ego,
            ghost=ghost, dense=dense, n_objects=n_objects, verbose=verbose,
            early_remove_mode=early_remove_mode,
        )

    keep = set(int(f) for f in frame_ids)
    work = multi_frame_case
    targets: List[FrameTarget] = [None] * len(frame_ids)
    init_override = None
    abs_to_pos = {int(f): i for i, f in enumerate(frame_ids)}

    for span_i, (s, e) in enumerate(case_frame_spans):
        chunk = [f for f in range(int(s), int(e)) if f in keep]
        if not chunk:
            continue
        print(
            "[attack] case-chunk {}/{} frames {}-{} (n={})".format(
                span_i + 1, len(case_frame_spans), chunk[0], chunk[-1], len(chunk)
            ),
            flush=True,
        )
        local_ids = list(range(len(chunk)))
        sub = _case_as_list(work, chunk)
        sub_out, chunk_tgts, chunk_init = apply_attack(
            mode,
            level,
            perception,
            dataset,
            sub,
            local_ids,
            ego_id=ego_id,
            att_id=att_id,
            iters=iters,
            also_ego=also_ego,
            ghost=ghost,
            dense=dense,
            n_objects=n_objects,
            verbose=verbose,
            early_remove_mode=early_remove_mode,
        )
        if sub_out is not None:
            _write_case_frames(work, chunk, sub_out)
        for j, f in enumerate(chunk):
            pos = abs_to_pos.get(int(f))
            if pos is None:
                continue
            if chunk_tgts is not None and j < len(chunk_tgts):
                targets[pos] = chunk_tgts[j]
        if chunk_init is not None:
            if init_override is None:
                init_override = [None] * len(frame_ids)
            for j, f in enumerate(chunk):
                pos = abs_to_pos.get(int(f))
                if pos is None or j >= len(chunk_init):
                    continue
                init_override[pos] = chunk_init[j]

    return work, targets, init_override


def replace_init_with_attack(frames, init_override, detector, multi_frame_case):
    """Swap Init detections for intermediate/late-attack fusion output and re-run defense."""
    if not init_override:
        return
    from .geometry import pack_dets
    from .three_source import SourceResult, _c_init

    for fr, ov in zip(frames, init_override):
        if ov is None:
            continue
        boxes, scores = ov
        b, p = pack_dets(boxes, scores)
        frame = multi_frame_case[fr.frame_id]
        c, n, v, ca = _c_init(
            b,
            frame,
            detector.ego_id,
            detector.uav_id,
            detector.n_ref,
            detector.n_ref_uav,
            detector.r0,
            detector.r_min,
        )
        init = SourceResult("init", b, p, c, n, v, ca)
        if detector.score_thres is not None:
            init = init.filter_by_score(detector.score_thres)
        fr.init = init
    detector.tent_buffer.reset()
    detector.trust_pool.reset()
    if getattr(detector, "far_tracker", None) is not None:
        detector.far_tracker.reset()
    for fr in frames:
        frame = multi_frame_case[fr.frame_id]
        gating, buf, pool = detector.apply_defense(
            fr.ego,
            fr.uav,
            fr.init,
            fr.gt_eval,
            fr.gt_eval_ids,
            gt_ego=fr.gt_ego,
            gt_ego_ids=fr.gt_ego_ids,
            ego_lidar=frame[detector.ego_id]["lidar"],
            frame_id=fr.frame_id,
            frame=frame,
            ego_raw=getattr(fr, "ego_raw", None),
        )
        fr.gating = gating
        fr.buffer = buf
        fr.pool = pool
    from .far_certain import T_FAR, backfill_tentative_to_certain

    if getattr(detector, "far_tracker", None) is not None:
        backfill_tentative_to_certain(
            frames, detector.far_tracker.promoted_hists(), t_far=T_FAR
        )
    backfill_tentative_to_certain(
        frames, getattr(detector.tent_buffer, "confirmed_hists", None) or [], t_far=None
    )
    backfill_tentative_to_certain(
        frames, getattr(detector.tent_buffer, "gate_certain_hists", None) or [], t_far=None
    )
