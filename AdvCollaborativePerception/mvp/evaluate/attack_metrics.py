"""Attack evaluation metrics for the V2U4Real intermediate attacks.

All bboxes are in the **ego frame**. A frame's detection is ``(bboxes, scores)``
where ``bboxes`` is (N, 7) ``[x,y,z,l,w,h,yaw]`` and ``scores`` is (N,).

Metrics reported (per attack mode):

- **ASR** (Attack Success Rate): fraction of frames where the attack succeeds.
  - spoof: a ghost object is detected at the injected location (IoU & score above
    thresholds).
  - remove: the targeted object is no longer detected (no matched detection).
- **Target confidence**: detection score of the target (ghost for spoof / victim for
  remove) — and its change vs the clean baseline.
- **FP growth / FN growth**: false-positive and false-negative count increase between
  the clean and the attacked frame, using standard IoU matching against GT.
- **Object disappearance rate** (remove only): fraction of frames where the target
  object vanished.

Helpers reuse ``mvp.tools.iou.iou3d``.
"""

import warnings

import numpy as np
from shapely.geometry import Polygon

from mvp.tools.iou import iou3d


def _bev_corners(bbox):
    """LiDAR 鸟瞰四角 (x,y)，yaw 绕 z。"""
    x, y, l, w, yaw = float(bbox[0]), float(bbox[1]), float(bbox[3]), float(bbox[4]), float(bbox[6])
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    r = np.array([[c, -s], [s, c]])
    return local.dot(r.T) + np.array([x, y])


def iou_bev(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pa, pb = Polygon(_bev_corners(a)), Polygon(_bev_corners(b))
        if (not pa.is_valid) or pa.area <= 0 or (not pb.is_valid) or pb.area <= 0:
            return 0.0
        inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def filter_eval_gt(gt_bboxes, min_range=3.0, max_range=80.0):
    """评估用 GT：范围内、车大小的框（与 AttFuse 只训 Car 对齐）。"""
    if gt_bboxes is None or len(gt_bboxes) == 0:
        return np.zeros((0, 7))
    gt = np.asarray(gt_bboxes, dtype=np.float64)
    dist = np.hypot(gt[:, 0], gt[:, 1])
    l, w = gt[:, 3], gt[:, 4]
    keep = (dist >= min_range) & (dist <= max_range) & (l >= 2.5) & (l <= 10.0) & (w >= 1.0) & (w <= 3.5)
    return gt[keep]


def _keep_high_score(bboxes, scores, score_thres):
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0, 7)), np.zeros((0,))
    bboxes = np.asarray(bboxes)
    scores = np.asarray(scores).reshape(-1)
    keep = scores >= score_thres
    return bboxes[keep], scores[keep]


def match_pred_sets(ref_boxes, ref_scores, hyp_boxes, hyp_scores,
                    score_thres=0.3, max_dist=4.0, iou_thres=0.5):
    """把干净检测当参考、攻击检测当假设，一对一匹配。

    Returns
    -------
    tp : 两边都有的框
    newborn : 攻击后新出现（相对干净检测的 FP）
    vanished : 干净时有、攻击后没了（相对干净检测的 FN）
    """
    ref, _ = _keep_high_score(ref_boxes, ref_scores, score_thres)
    hyp, _ = _keep_high_score(hyp_boxes, hyp_scores, score_thres)
    P, PP = len(ref), len(hyp)
    if P == 0:
        return 0, PP, 0
    if PP == 0:
        return 0, 0, P
    pairs = []
    for i in range(P):
        for j in range(PP):
            d = float(np.hypot(ref[i, 0] - hyp[j, 0], ref[i, 1] - hyp[j, 1]))
            iou = iou_bev(ref[i], hyp[j])
            if d <= max_dist or iou >= iou_thres:
                pairs.append((d, -iou, i, j))
    pairs.sort()
    used_r, used_h = set(), set()
    for _, _, i, j in pairs:
        if i in used_r or j in used_h:
            continue
        used_r.add(i)
        used_h.add(j)
    tp = len(used_r)
    newborn = PP - len(used_h)
    vanished = P - tp
    return tp, newborn, vanished


def match_gt(bboxes, scores, gt_bboxes, iou_thres, max_dist=4.0, score_thres=0.3):
    """一对一匹配：只计 score>=阈值 的检测；中心距 <= max_dist 或鸟瞰 IoU >= iou_thres。"""
    gt_bboxes = filter_eval_gt(gt_bboxes)
    P = gt_bboxes.shape[0]
    if bboxes is None or len(bboxes) == 0:
        return 0, 0, P
    bboxes = np.asarray(bboxes)
    scores = np.asarray(scores).reshape(-1)
    keep = scores >= score_thres
    bboxes = bboxes[keep]
    PP = bboxes.shape[0]
    if P == 0:
        return 0, PP, 0
    if PP == 0:
        return 0, 0, P

    pairs = []
    for i in range(P):
        for j in range(PP):
            d = float(np.hypot(gt_bboxes[i, 0] - bboxes[j, 0],
                               gt_bboxes[i, 1] - bboxes[j, 1]))
            iou = iou_bev(gt_bboxes[i], bboxes[j])
            if d <= max_dist or iou >= iou_thres:
                pairs.append((d, -iou, i, j))
    pairs.sort()
    used_gt, used_pred = set(), set()
    for _, _, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
    tp = len(used_gt)
    fp = PP - len(used_pred)
    fn = P - tp
    return tp, fp, fn


def nearest_det_score(bboxes, scores, target, max_dist=4.0):
    """用 BEV 中心距离关联目标，避免 KITTI 系 3D IoU 在雷达框上几乎恒为 0。"""
    if bboxes is None or len(bboxes) == 0 or target is None:
        return 0.0, -1, 1e9
    bboxes = np.asarray(bboxes)
    scores = np.asarray(scores).reshape(-1)
    d = np.hypot(bboxes[:, 0] - target[0], bboxes[:, 1] - target[1])
    i = int(np.argmin(d))
    if d[i] > max_dist:
        return 0.0, -1, float(d[i])
    return float(scores[i]), i, float(d[i])


# ---------------------------------------------------------------------------
# low-level matching
# ---------------------------------------------------------------------------
def _as_box_list(injected):
    """把评估传入的幽灵框规范成 list[(7,)]。单个 (7,) 不能 for 循环，否则会拆成 7 个标量。"""
    if injected is None:
        return []
    arr = np.asarray(injected, dtype=np.float64)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [arr.reshape(-1)[:7]] if arr.size >= 7 else []
    if arr.ndim == 2:
        return [row[:7] for row in arr]
    return []


def match_bbox(bbox, target, iou_thres):
    """True if any ``bbox`` overlaps ``target`` above ``iou_thres``."""
    if bbox is None or np.asarray(bbox).size == 0:
        return False
    target = np.asarray(target).reshape(-1)
    if target.size < 7:
        return False
    ious = np.array([iou3d(b, target) for b in np.asarray(bbox)])
    return bool((ious >= iou_thres).any())


def _max_score(bboxes, scores, target, iou_thres):
    """Highest score among detections overlapping ``target`` (0.0 if none)."""
    if bboxes.shape[0] == 0:
        return 0.0
    ious = np.array([iou3d(b, target) for b in bboxes])
    hit = ious >= iou_thres
    if not hit.any():
        return 0.0
    return float(scores[hit].max())


# ---------------------------------------------------------------------------
# spoof
# ---------------------------------------------------------------------------
def evaluate_spoof(clean, attacked, gt_list, injected_list,
                   iou_thres=0.5, score_thres=0.3, max_dist=4.0):
    """Evaluate a spoofing attack over a sequence of frames.

    ASR 相对干净检测：幽灵位置干净时分数 < score_thres，攻击后 >= score_thres。
    匹配用 BEV 中心距离，不用 KITTI camera 的 iou3d。
    """
    asr, ghost_conf_c, ghost_conf_a = [], [], []
    newborn_all, vanished_all = [], []
    gt_fp_growth, gt_fn_growth = [], []
    for i in range(len(clean)):
        c_box, c_score = clean[i]
        a_box, a_score = attacked[i]
        gt = gt_list[i]
        ghosts = _as_box_list(injected_list[i])
        if len(ghosts) == 0:
            continue
        ghost = ghosts[0]

        cc, _, _ = nearest_det_score(c_box, c_score, ghost, max_dist=max_dist)
        ca, _, _ = nearest_det_score(a_box, a_score, ghost, max_dist=max_dist)
        ghost_conf_c.append(cc)
        ghost_conf_a.append(ca)
        if cc < score_thres:
            asr.append(1.0 if ca >= score_thres else 0.0)

        _, newborn, vanished = match_pred_sets(
            c_box, c_score, a_box, a_score, score_thres=score_thres,
            max_dist=max_dist, iou_thres=iou_thres)
        newborn_all.append(newborn)
        vanished_all.append(vanished)

        _, fp_c, fn_c = match_gt(c_box, c_score, gt, iou_thres)
        _, fp_a, fn_a = match_gt(a_box, a_score, gt, iou_thres)
        gt_fp_growth.append(fp_a - fp_c)
        gt_fn_growth.append(fn_a - fn_c)

    return {
        "ASR": float(np.mean(asr)) if asr else 0.0,
        "ASR_frames": asr,
        "ghost_confidence_clean": ghost_conf_c,
        "ghost_confidence": ghost_conf_a,
        "ghost_confidence_mean": float(np.mean(ghost_conf_a)) if ghost_conf_a else 0.0,
        "newborn_mean": float(np.mean(newborn_all)) if newborn_all else 0.0,
        "vanished_mean": float(np.mean(vanished_all)) if vanished_all else 0.0,
        "FP_growth_mean": float(np.mean(gt_fp_growth)) if gt_fp_growth else 0.0,
        "FN_growth_mean": float(np.mean(gt_fn_growth)) if gt_fn_growth else 0.0,
        "FP_growth": gt_fp_growth,
        "FN_growth": gt_fn_growth,
    }


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def evaluate_remove(clean, attacked, gt_list, target_list,
                    iou_thres=0.5, score_thres=0.3, max_dist=4.0):
    """Evaluate a removal attack over a sequence of frames.

    ASR 只统计「干净帧里目标置信度 >= score_thres」的帧：
    成功 = 攻击后 4m 内不再有 >= score_thres 的检测。
    后续帧若跟踪框对不上干净检测，会标 invalid，不计入 ASR。
    """
    asr, conf_before, conf_after, fp_growth, fn_growth = [], [], [], [], []
    n_clean, n_atk = [], []
    valid = []
    collaterals = []
    gt_fp_growth, gt_fn_growth = [], []
    n_gt, tp_c_all, fn_c_all, tp_a_all, fn_a_all = [], [], [], [], []
    tgt_on_gt, tgt_became_fn = [], []
    for i in range(len(clean)):
        c_box, c_score = clean[i]
        a_box, a_score = attacked[i]
        gt = gt_list[i]
        target = target_list[i]
        if target is None:
            continue

        n_clean.append(0 if c_box is None else len(c_box))
        n_atk.append(0 if a_box is None else len(a_box))

        cb, _, _ = nearest_det_score(c_box, c_score, target, max_dist=max_dist)
        ca, _, _ = nearest_det_score(a_box, a_score, target, max_dist=max_dist)
        conf_before.append(cb)
        conf_after.append(ca)

        is_valid = cb >= score_thres
        valid.append(is_valid)
        if is_valid:
            asr.append(1.0 if ca < score_thres else 0.0)

        # 主指标：相对干净检测。newborn≈FP，vanished≈FN（含被打掉的目标）。
        _, newborn, vanished = match_pred_sets(
            c_box, c_score, a_box, a_score, score_thres=score_thres,
            max_dist=max_dist, iou_thres=iou_thres)
        # 误伤：干净时其他高分框丢了（不含攻击目标本身）
        collateral = vanished - (1 if is_valid and ca < score_thres else 0)
        if collateral < 0:
            collateral = 0
        fp_growth.append(newborn)
        fn_growth.append(vanished)
        collaterals.append(collateral)

        tp_c, fp_c, fn_c = match_gt(c_box, c_score, gt, iou_thres)
        tp_a, fp_a, fn_a = match_gt(a_box, a_score, gt, iou_thres)
        gt_fp_growth.append(fp_a - fp_c)
        gt_fn_growth.append(fn_a - fn_c)
        gt_f = filter_eval_gt(gt)
        n_gt.append(len(gt_f))
        tp_c_all.append(tp_c)
        fn_c_all.append(fn_c)
        tp_a_all.append(tp_a)
        fn_a_all.append(fn_a)

        d_gt = 1e9
        if len(gt_f) > 0:
            d_gt = float(np.min(np.hypot(gt_f[:, 0] - target[0], gt_f[:, 1] - target[1])))
        on_gt = d_gt <= 4.0
        tgt_on_gt.append(on_gt)
        # 目标对应的那辆 GT：干净时 4m 内有检测，攻击后没有
        gt_miss_clean = nearest_det_score(c_box, c_score, target, max_dist=4.0)[0] >= score_thres
        gt_miss_atk = nearest_det_score(a_box, a_score, target, max_dist=4.0)[0] < score_thres
        tgt_became_fn.append(on_gt and gt_miss_clean and gt_miss_atk)

    conf_drop = [b - a for b, a in zip(conf_before, conf_after)]
    return {
        "ASR": float(np.mean(asr)) if asr else float("nan"),
        "ASR_frames": asr,
        "n_valid_frames": int(np.sum(valid)) if valid else 0,
        "n_frames": len(conf_before),
        "object_disappearance_rate": float(np.mean(asr)) if asr else float("nan"),
        "target_conf_before_mean": float(np.mean(conf_before)) if conf_before else 0.0,
        "target_conf_after_mean": float(np.mean(conf_after)) if conf_after else 0.0,
        "target_conf_before": conf_before,
        "target_conf_after": conf_after,
        "target_conf_drop_mean": float(np.mean(conf_drop)) if conf_drop else 0.0,
        "target_conf_drop": conf_drop,
        "n_det_clean_mean": float(np.mean(n_clean)) if n_clean else 0.0,
        "n_det_attack_mean": float(np.mean(n_atk)) if n_atk else 0.0,
        "newborn_mean": float(np.mean(fp_growth)) if fp_growth else 0.0,
        "vanished_mean": float(np.mean(fn_growth)) if fn_growth else 0.0,
        "collateral_mean": float(np.mean(collaterals)) if collaterals else 0.0,
        "FP_growth_mean": float(np.mean(fp_growth)) if fp_growth else 0.0,
        "FN_growth_mean": float(np.mean(fn_growth)) if fn_growth else 0.0,
        "FP_growth": fp_growth,
        "FN_growth": fn_growth,
        "GT_FP_growth_mean": float(np.mean(gt_fp_growth)) if gt_fp_growth else 0.0,
        "GT_FN_growth_mean": float(np.mean(gt_fn_growth)) if gt_fn_growth else 0.0,
        "n_gt_mean": float(np.mean(n_gt)) if n_gt else 0.0,
        "clean_TP_mean": float(np.mean(tp_c_all)) if tp_c_all else 0.0,
        "clean_FN_mean": float(np.mean(fn_c_all)) if fn_c_all else 0.0,
        "attack_TP_mean": float(np.mean(tp_a_all)) if tp_a_all else 0.0,
        "attack_FN_mean": float(np.mean(fn_a_all)) if fn_a_all else 0.0,
        "target_on_gt_frames": int(np.sum(tgt_on_gt)) if tgt_on_gt else 0,
        "target_became_FN_rate": float(np.mean(tgt_became_fn)) if tgt_became_fn else 0.0,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def print_spoof_report(report, mode_name="spoof", verbose=False):
    n_asr = len(report.get("ASR_frames") or [])
    gc = report.get("ghost_confidence_clean") or []
    ga = report.get("ghost_confidence") or []
    print("")
    print("── {} 结果 ──".format(mode_name))
    print("ASR          {:.3f}  ({} 帧满足「干净未检出→攻击检出」)".format(
        report["ASR"], n_asr))
    print("幽灵置信度   {:.3f} → {:.3f}".format(
        float(np.mean(gc)) if gc else 0.0, report["ghost_confidence_mean"]))
    print("副作用       newborn {:.2f}  vanished {:.2f}".format(
        report.get("newborn_mean", 0.0), report.get("vanished_mean", 0.0)))
    if verbose:
        print("逐帧干净/攻击: {} / {}".format(
            [round(x, 3) for x in gc], [round(x, 3) for x in ga]))
        print("逐帧 ASR: {}".format([round(x, 2) for x in report["ASR_frames"]]))
        print("GT-FP/FN 增长: {:.2f} / {:.2f}".format(
            report["FP_growth_mean"], report["FN_growth_mean"]))


def print_remove_report(report, mode_name="remove", verbose=False):
    n_valid = report.get("n_valid_frames", 0)
    n_frames = report.get("n_frames", 0)
    asr = report["ASR"]
    asr_s = "n/a" if asr != asr else "{:.3f}".format(asr)
    print("")
    print("── {} 结果 ──".format(mode_name))
    print("ASR          {}  ({}/{} 有效帧, score≥0.3→<0.3)".format(
        asr_s, n_valid, n_frames))
    print("目标置信度   {:.3f} → {:.3f}  (降 {:.3f})".format(
        report["target_conf_before_mean"],
        report["target_conf_after_mean"],
        report["target_conf_drop_mean"]))
    print("检测数       {:.1f} → {:.1f}".format(
        report.get("n_det_clean_mean", 0.0), report.get("n_det_attack_mean", 0.0)))
    print("副作用       newborn {:.2f}  vanished {:.2f}  误伤 {:.2f}".format(
        report.get("newborn_mean", 0.0),
        report.get("vanished_mean", 0.0),
        report.get("collateral_mean", 0.0)))
    if verbose:
        print("逐帧干净/攻击: {} / {}".format(
            [round(x, 3) for x in report.get("target_conf_before", [])],
            [round(x, 3) for x in report.get("target_conf_after", [])]))
        print("逐帧 ASR: {}".format([round(x, 2) for x in report["ASR_frames"]]))
        print("GT-FP/FN 增长: {:.2f} / {:.2f}".format(
            report.get("GT_FP_growth_mean", 0.0), report.get("GT_FN_growth_mean", 0.0)))
        print("目标对齐 GT: {}/{}  真值 FN 率 {:.3f}".format(
            report.get("target_on_gt_frames", 0), n_frames,
            report.get("target_became_FN_rate", 0.0)))
