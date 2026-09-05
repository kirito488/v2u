"""Metrics aligned with V2U4Real OpenCOOD inference.py + eval_utils.py.

Official clean eval:
  - GT: obj_type == 'Car' only, unique id, inside cav_lidar_range
  - Pred: post-NMS boxes, sorted by score
  - Match: greedy BEV polygon IoU, one GT consumed per TP
  - Report: VOC AP @ IoU 0.25 / 0.5 / 0.7  (NOT 0.3)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .geometry import iou_bev

# V2U4Real inference.py / eval_utils.eval_final_results
IOU_LEVELS = (0.25, 0.5, 0.7)
# attfuse/where2comm/coalign config cav_lidar_range xy
LIDAR_X = 100.8
LIDAR_Y = 80.0


def lidar_range_mask(boxes) -> np.ndarray:
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (np.abs(boxes[:, 0]) <= LIDAR_X)
        & (np.abs(boxes[:, 1]) <= LIDAR_Y)
    )


def voc_ap(rec, prec):
    """Same as opencood.utils.eval_utils.voc_ap."""
    rec = [0.0] + list(rec) + [1.0]
    prec = [0.0] + list(prec) + [0.0]
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = [i for i in range(1, len(rec)) if rec[i] != rec[i - 1]]
    ap = 0.0
    for i in i_list:
        ap += (rec[i] - rec[i - 1]) * mpre[i]
    return ap


def match_score_sorted(preds, scores, gts, iou_thres: float):
    """V2U4 calculate_tp_fp: high score first, pop matched GT.

    Returns per-det tp/fp lists (0/1) and n_gt.
    """
    preds = np.asarray(preds) if preds is not None else np.zeros((0, 7))
    gts = np.asarray(gts) if gts is not None else np.zeros((0, 7))
    n_gt = 0 if gts.ndim != 2 else int(gts.shape[0])
    n = 0 if preds.ndim != 2 else int(preds.shape[0])
    tp, fp = [], []
    if n == 0:
        return tp, fp, n_gt
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores)
    taken = np.zeros((n_gt,), dtype=bool)
    for i in order:
        if n_gt == 0:
            fp.append(1)
            tp.append(0)
            continue
        ious = np.array(
            [0.0 if taken[j] else iou_bev(preds[i], gts[j]) for j in range(n_gt)]
        )
        j = int(np.argmax(ious)) if n_gt else -1
        if n_gt == 0 or float(ious[j]) < float(iou_thres):
            fp.append(1)
            tp.append(0)
        else:
            fp.append(0)
            tp.append(1)
            taken[j] = True
    return tp, fp, n_gt


def ap_from_lists(tp, fp, n_gt) -> float:
    if n_gt <= 0:
        return 0.0
    if not tp:
        return 0.0
    tp_c = np.cumsum(tp).astype(np.float64)
    fp_c = np.cumsum(fp).astype(np.float64)
    rec = (tp_c / float(n_gt)).tolist()
    prec = (tp_c / np.maximum(tp_c + fp_c, 1e-9)).tolist()
    return float(voc_ap(rec, prec))


def pr_counts(tp, fp, n_gt):
    tps = int(sum(tp))
    fps = int(sum(fp))
    fn = int(n_gt) - tps
    prec = tps / float(tps + fps) if (tps + fps) else 0.0
    rec = tps / float(n_gt) if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tps, fps, fn, prec, rec, f1


def eval_gt_boxes(fr) -> np.ndarray:
    """Prefer V2U4 Car+range GT stored on the frame."""
    g = getattr(fr, "gt_eval", None)
    if g is not None:
        g = np.asarray(g)
        if g.ndim == 2:
            return g
    return np.zeros((0, 7), dtype=np.float64)


def evaluate_source(frames, attr: str, iou_levels: Iterable[float] = IOU_LEVELS):
    rows = []
    for th in iou_levels:
        tp_all, fp_all = [], []
        n_gt = 0
        for fr in frames:
            src = getattr(fr, attr)
            gts = eval_gt_boxes(fr)
            tp, fp, ng = match_score_sorted(src.boxes, src.scores, gts, th)
            tp_all.extend(tp)
            fp_all.extend(fp)
            n_gt += ng
        ap = ap_from_lists(tp_all, fp_all, n_gt)
        tps, fps, fn, prec, rec, f1 = pr_counts(tp_all, fp_all, n_gt)
        rows.append({
            "source": attr, "iou": float(th), "ap": ap,
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tps, "fp": fps, "fn": fn, "gt": n_gt,
        })
    return rows


def greedy_bev_matches(boxes_a, boxes_b, iou_thres: float):
    """1-1 greedy BEV IoU matches. Returns (n_match, n_a, n_b)."""
    a = np.asarray(boxes_a) if boxes_a is not None else np.zeros((0, 7))
    b = np.asarray(boxes_b) if boxes_b is not None else np.zeros((0, 7))
    n_a = 0 if a.ndim != 2 else int(a.shape[0])
    n_b = 0 if b.ndim != 2 else int(b.shape[0])
    if n_a == 0 or n_b == 0:
        return 0, n_a, n_b
    ious = np.zeros((n_a, n_b), dtype=np.float64)
    for i in range(n_a):
        for j in range(n_b):
            ious[i, j] = iou_bev(a[i], b[j])
    used_a = np.zeros((n_a,), dtype=bool)
    used_b = np.zeros((n_b,), dtype=bool)
    n_match = 0
    while True:
        best = -1.0
        ii = jj = -1
        for i in range(n_a):
            if used_a[i]:
                continue
            for j in range(n_b):
                if used_b[j]:
                    continue
                v = float(ious[i, j])
                if v > best:
                    best, ii, jj = v, i, j
        if best < float(iou_thres) or ii < 0:
            break
        used_a[ii] = True
        used_b[jj] = True
        n_match += 1
    return n_match, n_a, n_b


def _ratio(num, den):
    return float(num) / float(den) if den else None


def _fmt_ratio(r):
    return "  n/a" if r is None else "{:6.1%}".format(r)


def pair_overlap(boxes_a, boxes_b, iou_thres: float) -> dict:
    n_match, n_a, n_b = greedy_bev_matches(boxes_a, boxes_b, iou_thres)
    union = n_a + n_b - n_match
    return {
        "n_match": n_match,
        "n_a": n_a,
        "n_b": n_b,
        "a_cov": _ratio(n_match, n_a),
        "b_cov": _ratio(n_match, n_b),
        "jaccard": _ratio(n_match, union),
    }


def hit_boxes(boxes, gts, iou_thres: float) -> np.ndarray:
    """Keep dets that hit any eval GT (BEV IoU >= thres). Same gate as dump hit."""
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    gts = np.asarray(gts) if gts is not None else np.zeros((0, 7))
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    if gts.ndim != 2 or gts.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    keep = []
    for b in boxes:
        if any(iou_bev(b, g) >= float(iou_thres) for g in gts):
            keep.append(b)
    if not keep:
        return np.zeros((0, 7), dtype=np.float64)
    return np.stack(keep)


def frame_source_overlap(fr, iou_thres: float):
    """Ego vs UAV / fuse, using only dets that hit gt_eval."""
    gts = eval_gt_boxes(fr)
    ego_h = hit_boxes(fr.ego.boxes, gts, iou_thres)
    uav_h = hit_boxes(fr.uav.boxes, gts, iou_thres)
    init_h = hit_boxes(fr.init.boxes, gts, iou_thres)
    return {
        "ego_uav": pair_overlap(ego_h, uav_h, iou_thres),
        "ego_init": pair_overlap(ego_h, init_h, iou_thres),
    }


def overlap_frame_line(fr, iou_thres: float) -> str:
    d = frame_source_overlap(fr, iou_thres)
    eu, ei = d["ego_uav"], d["ego_init"]
    return (
        "  same-obj(hit) IoU>={:.2f} | ego-uav match={}/{} ego_cov={} uav_cov={} | "
        "ego-fuse match={}/{} ego_cov={} fuse_cov={}".format(
            float(iou_thres),
            eu["n_match"], eu["n_a"], _fmt_ratio(eu["a_cov"]).strip(),
            _fmt_ratio(eu["b_cov"]).strip(),
            ei["n_match"], ei["n_a"], _fmt_ratio(ei["a_cov"]).strip(),
            _fmt_ratio(ei["b_cov"]).strip(),
        )
    )


def print_source_overlap(frames, iou_thres: float = 0.3):
    """Pooled same-object ratio on hit dets only (vs gt_eval).

    ego_cov = matched / n_ego_hit
    other_cov = matched / n_other_hit
    jaccard = matched / union
    """
    pairs = (
        ("ego-uav", "ego", "uav"),
        ("ego-fuse", "ego", "init"),
    )
    print("\n=== same-object overlap on HIT dets only (BEV IoU>={:.2f}, greedy 1-1) ===".format(
        float(iou_thres)))
    print("hit = det matches gt_eval; miss/FP boxes excluded before pairing")
    print("ego_cov = share of vehicle HIT boxes also HIT by the other source")
    print("{:<10} {:>6} {:>5} {:>6} {:>8} {:>9} {:>8}".format(
        "pair", "match", "ego", "other", "ego_cov", "other_cov", "jaccard"))
    for name, a_attr, b_attr in pairs:
        n_match = n_a = n_b = 0
        for fr in frames:
            gts = eval_gt_boxes(fr)
            a_h = hit_boxes(getattr(fr, a_attr).boxes, gts, iou_thres)
            b_h = hit_boxes(getattr(fr, b_attr).boxes, gts, iou_thres)
            st = pair_overlap(a_h, b_h, iou_thres)
            n_match += st["n_match"]
            n_a += st["n_a"]
            n_b += st["n_b"]
        union = n_a + n_b - n_match
        print("{:<10} {:>6d} {:>5d} {:>6d} {:>8} {:>9} {:>8}".format(
            name, n_match, n_a, n_b,
            _fmt_ratio(_ratio(n_match, n_a)),
            _fmt_ratio(_ratio(n_match, n_b)),
            _fmt_ratio(_ratio(n_match, union)),
        ))


def print_metrics_table(frames, sources=("ego", "uav", "init"), iou_levels=IOU_LEVELS):
    n_gt = sum(int(eval_gt_boxes(fr).shape[0]) for fr in frames)
    n_raw = sum(
        0 if fr.gt_ego is None else int(np.asarray(fr.gt_ego).shape[0])
        for fr in frames
    )
    print("\n=== V2U4Real-style eval (same as OpenCOOD inference.py) ===")
    print("GT: obj_type==Car, unique id, |x|<={:.1f} |y|<={:.1f}  (not yaml Truck/Cyclist/Ped)".format(
        LIDAR_X, LIDAR_Y))
    print("IoU: BEV, greedy high-score-first  |  official report is AP @ 0.25/0.5/0.7")
    print("GT boxes: yaml-all={}  eval-Car={}".format(n_raw, n_gt))
    print("{:<6} {:>5} {:>8} {:>10} {:>8} {:>8} {:>5} {:>5} {:>5} {:>5}".format(
        "src", "IoU", "AP", "precision", "recall", "F1", "TP", "FP", "FN", "GT"))
    for attr in sources:
        for row in evaluate_source(frames, attr, iou_levels):
            print("{:<6} {:>5.2f} {:>8.3f} {:>10.3f} {:>8.3f} {:>8.3f} {:>5d} {:>5d} {:>5d} {:>5d}".format(
                row["source"], row["iou"], row["ap"], row["precision"],
                row["recall"], row["f1"],
                row["tp"], row["fp"], row["fn"], row["gt"]))


# ---------------------------------------------------------------------------
# Defense output AP + attack miss counts
# ---------------------------------------------------------------------------

# Official accepted-state set (AP / ASR). Includes pool ATTACK (coast = object kept).
ACCEPTED_STATES = frozenset({
    "certain",
    "pool",
    "attack",
    "ambiguous -> certain",
    "tentative -> certain",
})


def _as_boxes7(boxes) -> np.ndarray:
    a = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)[:, :7]


def defense_output_boxes(fr) -> np.ndarray:
    """Defended Ŷ^t: baseline boxes if set, else pool + Certain + buffer CONFIRM."""
    bb = getattr(fr, "baseline_boxes", None)
    if bb is not None:
        return _as_boxes7(bb)
    buf = getattr(fr, "buffer", None)
    if buf is None:
        return np.zeros((0, 7), dtype=np.float64)
    return _as_boxes7(getattr(buf, "output_boxes", None))


def _overlay_pool_on_gate_maps(gate_maps: dict, pool) -> dict:
    """Same rule as run_three_source.overlay_pool_states."""
    out = {
        "ego": dict(gate_maps.get("ego") or {}),
        "uav": dict(gate_maps.get("uav") or {}),
        "init": dict(gate_maps.get("init") or {}),
    }
    if pool is None:
        return out
    for src, mp in pool.source_state_maps().items():
        dest = out.setdefault(src, {})
        for i, lab in mp.items():
            lab = str(lab)
            if lab == "certain":
                cur = str(dest.get(int(i), "") or "")
                if cur.startswith("ambiguous -> certain") or cur.startswith(
                    "tentative -> certain"
                ):
                    continue
                dest[int(i)] = "certain"
            elif lab == "pool":
                dest[int(i)] = "pool"
            elif lab == "attack":
                dest[int(i)] = "attack"
    return out


def frame_accepted_boxes(fr) -> np.ndarray:
    """Boxes whose dump state is certain / pool / attack / am→certain / tentative→certain.

    Pool ATTACK coast boxes (no matching source det) are included via pool records.
    Baselines have no gating labels: use their output boxes.
    """
    bb = getattr(fr, "baseline_boxes", None)
    if bb is not None:
        return _as_boxes7(bb)
    gate_maps = (
        fr.gating.source_state_maps()
        if getattr(fr, "gating", None) is not None
        else {"ego": {}, "uav": {}, "init": {}}
    )
    gate_maps = _overlay_pool_on_gate_maps(gate_maps, getattr(fr, "pool", None))
    rows = []
    for name in ("ego", "uav", "init"):
        src = getattr(fr, name, None)
        if src is None:
            continue
        boxes = _as_boxes7(src.boxes)
        states = gate_maps.get(name) or {}
        for i in range(int(boxes.shape[0])):
            st = str(states.get(i) or "")
            if not st and getattr(fr, "gating", None) is not None:
                st = str(fr.gating.label_for_box(boxes[i]) or "")
            if st in ACCEPTED_STATES:
                rows.append(boxes[i])
    # Gating objects may carry backfill labels not mirrored on every source index.
    if getattr(fr, "gating", None) is not None:
        for o in fr.gating.objects:
            if o.box is None:
                continue
            if str(getattr(o, "display_state", "") or "") in ACCEPTED_STATES:
                rows.append(np.asarray(o.box, dtype=np.float64).reshape(-1)[:7])
    # Pool ATTACK coast: KF predict box with no source det to stamp.
    pool = getattr(fr, "pool", None)
    if pool is not None:
        for rec in getattr(pool, "records", None) or []:
            if str(getattr(rec, "action", "") or "") != "attack":
                continue
            box = getattr(rec, "box", None)
            if box is None:
                continue
            rows.append(np.asarray(box, dtype=np.float64).reshape(-1)[:7])
    if not rows:
        return np.zeros((0, 7), dtype=np.float64)
    stacked = np.stack(rows)
    # Dedup near-duplicates (ego/uav/init of same object).
    keep = []
    for b in stacked:
        if any(float(iou_bev(b, k)) >= 0.5 for k in keep):
            continue
        keep.append(b)
    return np.stack(keep) if keep else np.zeros((0, 7), dtype=np.float64)


def target_matched(boxes, target, iou_thres: float = 0.3) -> bool:
    """True if any box BEV-IoU-matches the attack target."""
    if target is None:
        return False
    boxes = _as_boxes7(boxes)
    t = np.asarray(target, dtype=np.float64).reshape(-1)[:7]
    if boxes.shape[0] == 0:
        return False
    return any(float(iou_bev(b, t)) >= float(iou_thres) for b in boxes)


def evaluate_boxes_list(
    pred_list: List[np.ndarray],
    score_list: List[np.ndarray],
    frames,
    source_name: str = "defense",
    iou_levels: Iterable[float] = IOU_LEVELS,
):
    """AP / P / R / F1 for a list of per-frame prediction sets vs gt_eval."""
    rows = []
    for th in iou_levels:
        tp_all, fp_all = [], []
        n_gt = 0
        for fr, preds, scores in zip(frames, pred_list, score_list):
            gts = eval_gt_boxes(fr)
            tp, fp, ng = match_score_sorted(preds, scores, gts, th)
            tp_all.extend(tp)
            fp_all.extend(fp)
            n_gt += ng
        ap = ap_from_lists(tp_all, fp_all, n_gt)
        tps, fps, fn, prec, rec, f1 = pr_counts(tp_all, fp_all, n_gt)
        rows.append({
            "source": source_name, "iou": float(th), "ap": ap,
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tps, "fp": fps, "fn": fn, "gt": n_gt,
        })
    return rows


def evaluate_defense(frames, iou_levels: Iterable[float] = IOU_LEVELS):
    """AP of defended Ŷ (buffer.output_boxes) vs gt_eval."""
    pred_list, score_list = [], []
    for fr in frames:
        boxes = defense_output_boxes(fr)
        pred_list.append(boxes)
        score_list.append(np.ones((boxes.shape[0],), dtype=np.float64))
    return evaluate_boxes_list(pred_list, score_list, frames, "defense", iou_levels)


def evaluate_accepted(frames, iou_levels: Iterable[float] = IOU_LEVELS):
    """AP of accepted-state boxes (certain/pool/attack/am→certain/tentative→certain)."""
    pred_list, score_list = [], []
    for fr in frames:
        boxes = frame_accepted_boxes(fr)
        pred_list.append(boxes)
        score_list.append(np.ones((boxes.shape[0],), dtype=np.float64))
    return evaluate_boxes_list(pred_list, score_list, frames, "accept", iou_levels)


SPOOF_MODES = frozenset({"spoof", "multi_spoof"})
REMOVE_MODES = frozenset({"remove", "mass_remove"})
ATTACK_MODES = SPOOF_MODES | REMOVE_MODES


def frame_attack_targets(fr) -> List[np.ndarray]:
    """All ego-frame attack boxes on this frame (multi/mass aware)."""
    ts = getattr(fr, "attack_targets", None)
    if ts:
        return [
            np.asarray(t, dtype=np.float64).reshape(-1)[:7]
            for t in ts
            if t is not None
        ]
    tgt = getattr(fr, "attack_target", None)
    if tgt is None:
        return []
    return [np.asarray(tgt, dtype=np.float64).reshape(-1)[:7]]


def evaluate_attack_miss(
    frames,
    mode: str,
    iou_thres: float = 0.3,
) -> dict:
    """Count attack frames / instances where the attack was *not* identified.

    With-defense always uses accepted-state boxes
    (certain / pool / attack / am→certain / tentative→certain; includes pool coast).
    No-defense always uses Init.

    Spoof / multi_spoof:
      miss = any ghost still present in predictions (treated as real).
    Remove / mass_remove:
      miss = any target still absent from predictions.

    Also reports ASR (= miss rate) and ORR / ghost_hit at instance level.
    """
    mode = str(mode or "none")
    out = {
        "mode": mode,
        "iou_thres": float(iou_thres),
        "n_attack": 0,
        "n_targets": 0,
        "miss_no_defense": 0,
        "miss_defense": 0,
        "caught_no_defense": 0,
        "caught_defense": 0,
        "orr_no_defense": 0.0,
        "orr_defense": 0.0,
        "asr_no_defense": 0.0,
        "asr_defense": 0.0,
        "n_absent_init": 0,
        "n_absent_defense": 0,
    }
    if mode not in ATTACK_MODES:
        return out

    is_spoof = mode in SPOOF_MODES
    absent_init = absent_def = 0

    for fr in frames:
        tgts = frame_attack_targets(fr)
        if not tgts:
            continue
        out["n_attack"] += 1
        out["n_targets"] += len(tgts)
        init_boxes = _as_boxes7(getattr(fr.init, "boxes", None))
        def_boxes = frame_accepted_boxes(fr)
        hits_init = [target_matched(init_boxes, t, iou_thres) for t in tgts]
        hits_def = [target_matched(def_boxes, t, iou_thres) for t in tgts]
        if is_spoof:
            # Frame miss if any ghost still matched.
            miss_nd = any(hits_init)
            miss_d = any(hits_def)
            absent_init += sum(1 for h in hits_init if h)
            absent_def += sum(1 for h in hits_def if h)
        else:
            # Frame miss if any remove target still absent.
            miss_nd = any(not h for h in hits_init)
            miss_d = any(not h for h in hits_def)
            absent_init += sum(1 for h in hits_init if not h)
            absent_def += sum(1 for h in hits_def if not h)
        if miss_nd:
            out["miss_no_defense"] += 1
        else:
            out["caught_no_defense"] += 1
        if miss_d:
            out["miss_defense"] += 1
        else:
            out["caught_defense"] += 1

    n = out["n_attack"]
    nt = out["n_targets"]
    out["n_absent_init"] = absent_init
    out["n_absent_defense"] = absent_def
    if n > 0:
        out["asr_no_defense"] = out["miss_no_defense"] / float(n)
        out["asr_defense"] = out["miss_defense"] / float(n)
    if nt > 0:
        # ORR / ghost-hit rate at instance level.
        out["orr_no_defense"] = absent_init / float(nt)
        out["orr_defense"] = absent_def / float(nt)
    return out


def print_defense_metrics(frames, iou_levels=IOU_LEVELS):
    names = [getattr(fr, "baseline_name", None) for fr in frames]
    bl = next((n for n in names if n), None)
    print("\n=== Accepted-state eval (report metric) ===")
    if bl:
        print("baseline={!r}: output boxes (no gating states)".format(bl))
    else:
        print("certain / pool / attack / am→certain / tentative→certain  |  includes pool ATTACK coast")
    print("score=1.0 | GT = eval-Car")
    print("{:<8} {:>5} {:>8} {:>10} {:>8} {:>8} {:>5} {:>5} {:>5} {:>5}".format(
        "src", "IoU", "AP", "precision", "recall", "F1", "TP", "FP", "FN", "GT"))
    for row in evaluate_accepted(frames, iou_levels):
        print("{:<8} {:>5.2f} {:>8.3f} {:>10.3f} {:>8.3f} {:>8.3f} {:>5d} {:>5d} {:>5d} {:>5d}".format(
            row["source"], row["iou"], row["ap"], row["precision"],
            row["recall"], row["f1"],
            row["tp"], row["fp"], row["fn"], row["gt"]))
    print("\n=== Ŷ (pool + Certain + buffer CONFIRM; not the report metric) ===")
    print("{:<8} {:>5} {:>8} {:>10} {:>8} {:>8} {:>5} {:>5} {:>5} {:>5}".format(
        "src", "IoU", "AP", "precision", "recall", "F1", "TP", "FP", "FN", "GT"))
    for row in evaluate_defense(frames, iou_levels):
        print("{:<8} {:>5.2f} {:>8.3f} {:>10.3f} {:>8.3f} {:>8.3f} {:>5d} {:>5d} {:>5d} {:>5d}".format(
            row["source"], row["iou"], row["ap"], row["precision"],
            row["recall"], row["f1"],
            row["tp"], row["fp"], row["fn"], row["gt"]))


def print_attack_miss(frames, mode: str, iou_thres: float = 0.3):
    st = evaluate_attack_miss(frames, mode, iou_thres=iou_thres)
    if st["mode"] not in ATTACK_MODES or st["n_attack"] <= 0:
        print("\n=== Attack miss ===")
        print("  (no attack targets; mode={!r})".format(mode))
        return st
    n = st["n_attack"]
    nt = st["n_targets"]
    is_spoof = st["mode"] in SPOOF_MODES
    print("\n=== Attack miss (IoU>={:.2f}, per-frame targets) ===".format(float(iou_thres)))
    print(
        "mode={}  attack_frames={}  target_instances={}".format(
            st["mode"], n, nt
        )
    )
    if is_spoof:
        print("miss = any ghost still matched (treated as real object)")
        print("ASR = frame-level attack success (ghost present)")
    else:
        print("miss = any remove target still absent (object not recovered)")
        print("ASR = frame-level remove success (target absent)")
        print("ORR = instance-level absent rate among all remove targets")
    print("  with-defense = accepted (certain/pool/attack/am→certain/tentative→certain; + pool coast)")
    print("  no-defense (Init):  miss={:>4d}/{:<4d}  ASR={:.1%}  caught={:>4d}".format(
        st["miss_no_defense"], n, st["asr_no_defense"],
        st["caught_no_defense"]))
    print("  with-defense:       miss={:>4d}/{:<4d}  ASR={:.1%}  caught={:>4d}".format(
        st["miss_defense"], n, st["asr_defense"],
        st["caught_defense"]))
    if nt > 0:
        label = "ghost_hit" if is_spoof else "ORR"
        print(
            "  instance-level {}:  Init={:.1%} ({}/{})  Defense={:.1%} ({}/{})".format(
                label,
                st["orr_no_defense"], st["n_absent_init"], nt,
                st["orr_defense"], st["n_absent_defense"], nt,
            )
        )
    return st
