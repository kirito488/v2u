"""Recompute accepted-only AP/ASR from a saved --defense all JSON (CPU only)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.baselines import apply_baseline
from defense.gating import FrameGating, GatedObject
from defense.metrics import evaluate_accepted, evaluate_attack_miss
from defense.three_source import FrameThreeSource, SourceResult
from defense.trust_pool import ATTACK as POOL_ATTACK
from defense.trust_pool import BIRTH, FramePool, PoolRecord


def _src(name, d):
    boxes = np.asarray(d.get("boxes") or [], dtype=np.float64)
    if boxes.ndim != 2:
        boxes = np.zeros((0, 7), dtype=np.float64)
    n = int(boxes.shape[0])
    scores = np.asarray(d.get("scores") or [], dtype=np.float64).reshape(-1)
    conf = np.asarray(d.get("confidences") or [], dtype=np.float64).reshape(-1)
    pcs = np.asarray(d.get("point_counts") or [], dtype=np.int32).reshape(-1)
    if scores.shape[0] != n:
        scores = np.ones((n,), dtype=np.float64)
    if conf.shape[0] != n:
        conf = np.ones((n,), dtype=np.float64)
    if pcs.shape[0] != n:
        pcs = np.zeros((n,), dtype=np.int32)
    return SourceResult(name, boxes, scores, conf, pcs)


def _gating(d):
    if not d:
        return None
    objs = []
    for row in d.get("objects") or []:
        box = row.get("box")
        objs.append(
            GatedObject(
                state=str(row.get("state") or "tentative"),
                d_ego=int(row.get("d_ego") or 0),
                d_uav=int(row.get("d_uav") or 0),
                d_init=int(row.get("d_init") or 0),
                q_ego=float(row.get("q_ego") or 0.0),
                p_ego=float(row.get("p_ego") or 0.0),
                c_ego=float(row.get("c_ego") or 0.0),
                ego_i=int(row.get("ego_i") if row.get("ego_i") is not None else -1),
                uav_i=int(row.get("uav_i") if row.get("uav_i") is not None else -1),
                init_i=int(row.get("init_i") if row.get("init_i") is not None else -1),
                dual=bool(row.get("dual")),
                from_ambiguous=bool(row.get("from_ambiguous")),
                from_far=bool(row.get("from_far")),
                backfill_certain=bool(row.get("backfill_certain")),
                box=None if box is None else np.asarray(box, dtype=np.float64),
            )
        )
        # Prefer dumped display_state when present.
        if row.get("display_state"):
            ds = str(row["display_state"])
            if ds == "tentative -> certain":
                objs[-1].backfill_certain = True
            elif ds.startswith("ambiguous ->"):
                objs[-1].from_ambiguous = True
    return FrameGating(objects=objs)


def _pool(d):
    if not d:
        return None
    labels_ego, labels_uav, labels_init = {}, {}, {}
    recs = []
    for row in d.get("records") or []:
        box = row.get("box")
        action = str(row.get("action") or "")
        rec = PoolRecord(
            tid=int(row.get("tid") or -1),
            action=action,
            iou=float(row.get("iou") or 0.0),
            age=int(row.get("age") or 0),
            box=None if box is None else np.asarray(box, dtype=np.float64),
            ego_i=int(row.get("ego_i") if row.get("ego_i") is not None else -1),
            uav_i=int(row.get("uav_i") if row.get("uav_i") is not None else -1),
            init_i=int(row.get("init_i") if row.get("init_i") is not None else -1),
            gt_tag=str(row.get("gt_tag") or ""),
            v=float(row.get("v") if row.get("v") is not None else 1.0),
        )
        recs.append(rec)
        if action == POOL_ATTACK:
            tag = "attack"
        elif action == BIRTH:
            tag = "certain"
        else:
            tag = "pool"
        if rec.ego_i >= 0:
            labels_ego[rec.ego_i] = tag
        if rec.uav_i >= 0:
            labels_uav[rec.uav_i] = tag
        if rec.init_i >= 0:
            labels_init[rec.init_i] = tag
    out_b = d.get("output_boxes")
    return FramePool(
        records=recs,
        output_boxes=None if out_b is None else np.asarray(out_b, dtype=np.float64),
        labels_ego=labels_ego,
        labels_uav=labels_uav,
        labels_init=labels_init,
    )


def frame_from_dict(d) -> FrameThreeSource:
    ge = d.get("gt_ego")
    gt_eval = np.asarray(ge, dtype=np.float64) if ge else np.zeros((0, 7))
    if gt_eval.ndim != 2:
        gt_eval = np.zeros((0, 7), dtype=np.float64)
    fr = FrameThreeSource(
        frame_id=int(d.get("frame_id") or 0),
        ego=_src("ego", d.get("ego") or {}),
        uav=_src("uav", d.get("uav") or {}),
        init=_src("init", d.get("init") or {}),
        gt_eval=gt_eval,
        gt_eval_ids=list(d.get("gt_ego_ids") or []),
        gating=_gating(d.get("gating")),
        pool=_pool(d.get("pool")),
        buffer=None,
    )
    at = d.get("attack_target")
    fr.attack_target = None if at is None else np.asarray(at, dtype=np.float64)
    fr.attack_targets = [fr.attack_target] if fr.attack_target is not None else None
    return fr


def ap_at(rows, iou):
    for r in rows:
        if abs(float(r["iou"]) - float(iou)) < 1e-9:
            return float(r["ap"])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--iou_thres", type=float, default=0.3)
    args = ap.parse_args()
    data = json.load(open(args.json_path, encoding="utf-8"))
    mode = str(data.get("mode") or "none")
    frames = [frame_from_dict(d) for d in data.get("frames") or []]
    print("mode={} frames={}".format(mode, len(frames)))
    print("with-defense = accepted | AP for ours uses saved accepted_metrics (gt_eval not dumped)")
    print("{:<10} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
        "defense", "AP@.25", "AP@.5", "AP@.7", "ASR_def", "ASR_init"))

    ours_ap = data.get("accepted_metrics") or []
    ours_miss = evaluate_attack_miss(frames, mode, iou_thres=args.iou_thres)
    rows = [{
        "defense": "ours",
        "ap025": ap_at(ours_ap, 0.25),
        "ap05": ap_at(ours_ap, 0.5),
        "ap07": ap_at(ours_ap, 0.7),
        "asr_defense": ours_miss["asr_defense"],
        "asr_no_defense": ours_miss["asr_no_defense"],
        "n_attack": ours_miss["n_attack"],
        "miss_defense": ours_miss["miss_defense"],
    }]

    for name in ("none", "robosac", "cad", "cp_guard", "made"):
        frs = []
        for fr in frames:
            bl = apply_baseline(name, fr.ego, fr.uav, fr.init, ego_lidar=None)
            nf = FrameThreeSource(
                frame_id=fr.frame_id,
                ego=fr.ego,
                uav=fr.uav,
                init=fr.init,
                gt_eval=fr.gt_eval,
                gt_eval_ids=fr.gt_eval_ids,
                baseline_name=name,
                baseline_boxes=None if bl is None else bl.boxes,
                baseline_scores=None if bl is None else bl.scores,
                baseline_info=None if bl is None else dict(bl.info or {}),
            )
            nf.attack_target = fr.attack_target
            nf.attack_targets = fr.attack_targets
            frs.append(nf)
        # Prefer saved compare accept_ap for baselines when CAD had lidar at save time.
        saved = {r["defense"]: r for r in (data.get("compare") or [])}
        arows = evaluate_accepted(frs)
        miss = evaluate_attack_miss(frs, mode, iou_thres=args.iou_thres)
        if name == "cad" and name in saved and saved[name].get("accept_ap05") is not None:
            # CAD needs ego lidar; keep AP from original run, recompute ASR only.
            ap025 = float(saved[name]["ap025"])  # was Ŷ==accept for baseline
            ap05 = float(saved[name]["accept_ap05"])
            ap07 = float(saved[name]["ap07"])
        else:
            ap025, ap05, ap07 = ap_at(arows, 0.25), ap_at(arows, 0.5), ap_at(arows, 0.7)
            if name in saved and saved[name].get("accept_ap05") is not None:
                ap05 = float(saved[name]["accept_ap05"])
                ap025 = float(saved[name]["ap025"])
                ap07 = float(saved[name]["ap07"])
        rows.append({
            "defense": name,
            "ap025": ap025,
            "ap05": ap05,
            "ap07": ap07,
            "asr_defense": miss["asr_defense"],
            "asr_no_defense": miss["asr_no_defense"],
            "n_attack": miss["n_attack"],
            "miss_defense": miss["miss_defense"],
        })

    for r in rows:
        print("{:<10} {:>8.3f} {:>8.3f} {:>8.3f} {:>9.1%} {:>9.1%}".format(
            r["defense"], r["ap025"], r["ap05"], r["ap07"],
            r["asr_defense"], r["asr_no_defense"],
        ))
    out_path = args.json_path.replace(".json", "_accept.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "compare_accepted": rows, "ours_attack_miss": ours_miss}, f, indent=2)
    print("[save]", out_path)


if __name__ == "__main__":
    main()
