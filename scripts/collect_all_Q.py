#!/usr/bin/env python3
"""Collect Q=P·C for all detections with P >= p_min (default 0.05).

No clear-car / V filter — empirical Q distribution for later percentile ablation.

  python scripts/collect_all_Q.py --gpu 1 \\
      --calib_json logs/calibrate_confidence.json \\
      --save logs/all_Q_distribution.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _bind_gpu():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", type=int, default=None)
    args, _ = p.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))


_bind_gpu()

import numpy as np  # noqa: E402

from defense.confidence import (  # noqa: E402
    R_MIN,
    apply_calibration,
    boxes_geometric_center,
    perception_confidence,
    range_xy,
    visible_area,
)
from defense.gating import detection_quality  # noqa: E402
from defense.geometry import count_points_in_box, iou_bev, pack_dets  # noqa: E402
from defense.paths import (  # noqa: E402
    EGO_ID,
    UAV_ID,
    EGO_POINTPILLAR_DIR,
    UAV_POINTPILLAR_DIR,
    setup_import_paths,
    val_dir,
)
from defense.three_source import _boxes_to_pose  # noqa: E402

setup_import_paths()

from mvp.data.v2u4real_dataset import V2U4RealDataset  # noqa: E402

from scripts.run_three_source import (  # noqa: E402
    all_dataset_scenes,
    load_late_pointpillar,
)


def _pct_table(a, ps=(1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99)):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def _match_gt(box, gts, iou_thres=0.3) -> bool:
    gts = np.asarray(gts) if gts is not None else np.zeros((0, 7))
    if gts.ndim != 2 or gts.shape[0] == 0:
        return False
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
    for g in gts:
        if float(iou_bev(b, np.asarray(g, dtype=np.float64).reshape(-1)[:7])) >= float(
            iou_thres
        ):
            return True
    return False


def _collect(
    boxes_sensor,
    scores,
    lidar,
    gts_match_frame,
    boxes_for_match,
    source: str,
    p_min: float,
    pad: float,
    bottom_center: bool,
):
    """All dets with P>=p_min. Optional TP flag for side stats only."""
    boxes_sensor = np.asarray(boxes_sensor) if boxes_sensor is not None else np.zeros((0, 7))
    boxes_for_match = (
        np.asarray(boxes_for_match) if boxes_for_match is not None else boxes_sensor
    )
    scores = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((0,))
    if boxes_sensor.ndim != 2 or boxes_sensor.shape[0] == 0:
        return []
    if bottom_center:
        boxes_cnt = boxes_geometric_center(boxes_sensor)
    else:
        boxes_cnt = boxes_sensor
    rows = []
    for i in range(int(boxes_sensor.shape[0])):
        p = float(scores[i]) if i < int(scores.shape[0]) else 0.0
        if p < float(p_min):
            continue
        b_cnt = boxes_cnt[i, :7]
        n = int(count_points_in_box(lidar, b_cnt, pad=pad))
        # still record n=0 → C≈0, Q≈0 (valid low-quality dets)
        r = float(range_xy(b_cnt, r_min=R_MIN))
        A = float(visible_area(b_cnt, source))
        c = float(perception_confidence(n, b_cnt, source))
        q = float(detection_quality(p, c))
        b_match = (
            boxes_for_match[i, :7]
            if i < int(boxes_for_match.shape[0])
            else boxes_sensor[i, :7]
        )
        tp = bool(_match_gt(b_match, gts_match_frame))
        rows.append(
            {
                "P": p,
                "C": c,
                "Q": q,
                "n": n,
                "r": r,
                "A": A,
                "tp": tp,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--p_min", type=float, default=0.05, help="Keep dets with P>=this")
    ap.add_argument(
        "--calib_json",
        type=str,
        default="logs/calibrate_confidence.json",
    )
    ap.add_argument("--save", type=str, default="logs/all_Q_distribution.json")
    ap.add_argument(
        "--save_pool",
        type=str,
        default="logs/all_Q_pool.json",
        help="Raw per-det records (can be large)",
    )
    args = ap.parse_args()

    calib = json.load(open(args.calib_json))
    apply_calibration(
        n_ref_ego=float(calib["ego"]["n_ref"]),
        n_ref_uav=float(calib["uav"]["n_ref"]),
        r0_ego=float(calib["ego"]["r0"]),
        r0_uav=float(calib["uav"]["r0"]),
        a_ref_ego=float(calib["ego"]["A_ref"]),
        a_ref_uav=float(calib["uav"]["A_ref"]),
    )
    print(
        "[calib]",
        "n_ref ego/uav=",
        calib["ego"]["n_ref"],
        calib["uav"]["n_ref"],
        "r0=",
        calib["ego"]["r0"],
        calib["uav"]["r0"],
        flush=True,
    )

    print("[load] PointPillar ...", flush=True)
    ego_p = load_late_pointpillar(EGO_POINTPILLAR_DIR)
    uav_p = load_late_pointpillar(UAV_POINTPILLAR_DIR)
    dataset = V2U4RealDataset(root_path=val_dir(), mode="val")
    scenes = all_dataset_scenes(dataset)

    pool = {"ego": [], "uav": []}
    for si, (sname, rec) in enumerate(scenes.items()):
        print(f"[scan] {si + 1}/{len(scenes)} {sname}", flush=True)
        mfc = dataset.get_case_by_meta(
            {"scenario_name": sname, "frame_ids": rec["frame_ts"]}, tag="multi_frame"
        )
        for f in range(len(mfc)):
            frame = mfc[f]
            gt = frame[EGO_ID].get("gt_bboxes")
            pose_e = frame[EGO_ID]["lidar_pose"]
            pose_u = frame[UAV_ID]["lidar_pose"]
            try:
                eb, es = pack_dets(*ego_p.run(frame, EGO_ID))
            except Exception:
                eb, es = np.zeros((0, 7)), np.zeros((0,))
            rows_e = _collect(
                eb,
                es,
                frame[EGO_ID]["lidar"],
                gt,
                eb,
                "ego",
                args.p_min,
                pad=0.25,
                bottom_center=True,
            )
            for r in rows_e:
                r["scene"] = sname
                r["frame"] = int(f)
            pool["ego"].extend(rows_e)

            try:
                ub, us = pack_dets(*uav_p.run(frame, EGO_ID))
            except Exception:
                ub, us = np.zeros((0, 7)), np.zeros((0,))
            if len(ub):
                ub_u = _boxes_to_pose(boxes_geometric_center(ub), pose_e, pose_u)
                rows_u = _collect(
                    ub_u,
                    us,
                    frame[UAV_ID]["lidar"],
                    gt,
                    ub,  # match in ego frame
                    "uav",
                    args.p_min,
                    pad=0.5,
                    bottom_center=False,
                )
                for r in rows_u:
                    r["scene"] = sname
                    r["frame"] = int(f)
                pool["uav"].extend(rows_u)

    def summarize(tag, rows):
        Qs = np.array([x["Q"] for x in rows], dtype=np.float64)
        Ps = np.array([x["P"] for x in rows], dtype=np.float64)
        Cs = np.array([x["C"] for x in rows], dtype=np.float64)
        n_tp = int(sum(1 for x in rows if x["tp"]))
        out = {
            "n": int(len(rows)),
            "n_tp": n_tp,
            "n_fp": int(len(rows) - n_tp),
            "Q": _pct_table(Qs),
            "P": _pct_table(Ps),
            "C": _pct_table(Cs),
            "Q_mean": float(Qs.mean()) if Qs.size else float("nan"),
            "Q_max": float(Qs.max()) if Qs.size else float("nan"),
        }
        # TP-only side table (not for primary ablation, just reference)
        Q_tp = np.array([x["Q"] for x in rows if x["tp"]], dtype=np.float64)
        out["Q_tp"] = _pct_table(Q_tp)
        out["n_tp_check"] = int(Q_tp.size)
        print(f"\n=== {tag}  n={out['n']}  tp={n_tp}  fp={out['n_fp']} ===")
        print("Q all:", {k: round(v, 4) for k, v in out["Q"].items()})
        print("Q tp :", {k: round(v, 4) for k, v in out["Q_tp"].items()})
        return out

    summary = {
        "p_min": float(args.p_min),
        "calib_json": args.calib_json,
        "note": "All dets with P>=p_min; Q=P*C under calibrated C. Ablate Q_h/Q_l from Q percentiles.",
        "ego": summarize("ego", pool["ego"]),
        "uav": summarize("uav", pool["uav"]),
    }

    # Suggested ablation anchors from ego Q percentiles (informational)
    qe = summary["ego"]["Q"]
    suggestions = []
    for hi, lo in [(90, 70), (75, 50), (50, 25), (40, 20), (25, 10)]:
        qh, ql = qe[f"p{hi}"], qe[f"p{lo}"]
        suggestions.append(
            {
                "rule": f"ego Q p{hi}/p{lo}",
                "q_h": qh,
                "q_l": ql,
                "ok": bool(qh > ql),
            }
        )
    summary["ego_ablation_suggestions"] = suggestions
    print("\n=== suggested (ego Q percentiles) ===")
    for s in suggestions:
        print(
            f"  {s['rule']}: Q_h={s['q_h']:.4f}  Q_l={s['q_l']:.4f}  ok={s['ok']}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {args.save}", flush=True)

    # raw pool (keep fields compact)
    with open(args.save_pool, "w") as f:
        json.dump(pool, f)
    print(f"[save] {args.save_pool}  ego={len(pool['ego'])} uav={len(pool['uav'])}", flush=True)


if __name__ == "__main__":
    main()
