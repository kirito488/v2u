#!/usr/bin/env python3
"""Correlate ego polar V with detector P and GT recall (clean dump + LiDAR).

  python scripts/analyze_v_vs_p.py \\
      --dump logs/calib_clean_none_20260907_101704.json \\
      --save logs/v_vs_p_analysis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="logs/calib_clean_none_20260907_101704.json")
    ap.add_argument("--save", default="logs/v_vs_p_analysis.json")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth dump frame")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import numpy as np

    from defense.geometry import iou_bev
    from defense.metrics import lidar_range_mask
    from defense.occlusion import build_polar_depth, visibility_ratio_lidar
    from defense.paths import EGO_ID, setup_import_paths, val_dir

    setup_import_paths()
    from mvp.data.v2u4real_dataset import V2U4RealDataset

    dump = json.loads(open(os.path.join(_ROOT, args.dump), encoding="utf-8").read())
    dataset = V2U4RealDataset(root_path=val_dir(), mode="val")

    pairs = []
    for _sname, case_ids in dump["scenes"].items():
        for ci in case_ids:
            meta = dataset.cases["multi_frame"][int(ci)]
            for local in range(len(meta["frame_ids"])):
                pairs.append((int(ci), local))
    assert len(pairs) == len(dump["frames"]), (len(pairs), len(dump["frames"]))

    idxs = list(range(0, len(pairs), max(1, int(args.stride))))
    if args.max_frames and args.max_frames > 0:
        idxs = idxs[: int(args.max_frames)]
    print(f"[v_vs_p] frames={len(idxs)}/{len(pairs)} stride={args.stride}", flush=True)

    Ps, Vs, TPs = [], [], []
    gt_V, gt_hit = [], []
    IOU_TP = 0.3
    t0 = time.time()
    case_cache: dict = {}

    def get_case(ci: int):
        if ci not in case_cache:
            meta = dataset.cases["multi_frame"][ci]
            case_cache.clear()
            case_cache[ci] = dataset.get_case_by_meta(
                {
                    "scenario_name": meta["scenario_name"],
                    "frame_ids": list(meta["frame_ids"]),
                },
                tag="multi_frame",
            )
        return case_cache[ci]

    for n_done, fi in enumerate(idxs, 1):
        ci, local = pairs[fi]
        fr = dump["frames"][fi]
        frame = get_case(ci)[local]
        depth = build_polar_depth(np.asarray(frame[EGO_ID]["lidar"]))

        ego_boxes = (
            np.asarray(fr["ego"]["boxes"], float)
            if fr["ego"]["n"]
            else np.zeros((0, 7))
        )
        scores = (
            np.asarray(fr["ego"]["scores"], float)
            if fr["ego"]["n"]
            else np.zeros((0,))
        )
        gts_raw = np.asarray(fr.get("gt_ego") or [], float)
        if gts_raw.ndim == 2 and gts_raw.shape[0]:
            gts = gts_raw[lidar_range_mask(gts_raw)]
        else:
            gts = np.zeros((0, 7))

        for i in range(len(scores)):
            b = ego_boxes[i]
            v = float(
                visibility_ratio_lidar(
                    b,
                    depth=depth,
                    origin=(0.0, 0.0, 0.0),
                    check_ego_range=True,
                    bottom_center=True,
                )
            )
            p = float(scores[i])
            tp = 0.0
            if gts.shape[0]:
                tp = float(max(float(iou_bev(b, g)) for g in gts) >= IOU_TP)
            Ps.append(p)
            Vs.append(v)
            TPs.append(tp)

        for g in gts:
            v = float(
                visibility_ratio_lidar(
                    g,
                    depth=depth,
                    origin=(0.0, 0.0, 0.0),
                    check_ego_range=True,
                    bottom_center=True,
                )
            )
            hit = 0.0
            if len(scores):
                hit = float(max(float(iou_bev(b, g)) for b in ego_boxes) >= IOU_TP)
            gt_V.append(v)
            gt_hit.append(hit)

        if n_done % 100 == 0 or n_done == len(idxs):
            print(
                f"  [{n_done}/{len(idxs)}] {time.time()-t0:.1f}s dets={len(Ps)} gts={len(gt_V)}",
                flush=True,
            )

    Ps = np.asarray(Ps)
    Vs = np.asarray(Vs)
    TPs = np.asarray(TPs)
    gt_V = np.asarray(gt_V)
    gt_hit = np.asarray(gt_hit)

    def corr(a, b):
        if len(a) < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def by_level(V, Y):
        out = {}
        for k in range(9):
            thr = k / 8.0
            m = np.abs(V - thr) < 1e-6
            if not m.any():
                continue
            out[f"{thr:.3f}"] = {
                "N": int(m.sum()),
                "mean": float(Y[m].mean()),
                "p50": float(np.median(Y[m])),
            }
        return out

    def by_coarse(V, Y):
        out = {}
        for a, b, lab in [
            (0.0, 0.25, "[0,0.25)"),
            (0.25, 0.5, "[0.25,0.5)"),
            (0.5, 0.75, "[0.5,0.75)"),
            (0.75, 1.0001, "[0.75,1]"),
        ]:
            m = (V >= a) & (V < b)
            if not m.any():
                continue
            out[lab] = {
                "N": int(m.sum()),
                "mean": float(Y[m].mean()),
                "p50": float(np.median(Y[m])),
            }
        return out

    result = {
        "dump": args.dump,
        "n_frames": len(idxs),
        "stride": int(args.stride),
        "n_ego_dets": int(len(Ps)),
        "n_gt": int(len(gt_V)),
        "corr_V_P": corr(Vs, Ps),
        "corr_V_TP": corr(Vs, TPs),
        "corr_V_gt_recall": corr(gt_V, gt_hit),
        "ego_P_by_V": by_level(Vs, Ps),
        "ego_TP_by_V": by_level(Vs, TPs),
        "ego_P_coarse": by_coarse(Vs, Ps),
        "ego_TP_coarse": by_coarse(Vs, TPs),
        "gt_recall_by_V": by_level(gt_V, gt_hit),
        "gt_recall_coarse": by_coarse(gt_V, gt_hit),
        "contrast": {
            "ego_V0": {
                "N": int((np.abs(Vs) < 1e-6).sum()),
                "P_mean": float(Ps[np.abs(Vs) < 1e-6].mean())
                if (np.abs(Vs) < 1e-6).any()
                else None,
                "TP": float(TPs[np.abs(Vs) < 1e-6].mean())
                if (np.abs(Vs) < 1e-6).any()
                else None,
            },
            "ego_V1": {
                "N": int((np.abs(Vs - 1) < 1e-6).sum()),
                "P_mean": float(Ps[np.abs(Vs - 1) < 1e-6].mean())
                if (np.abs(Vs - 1) < 1e-6).any()
                else None,
                "TP": float(TPs[np.abs(Vs - 1) < 1e-6].mean())
                if (np.abs(Vs - 1) < 1e-6).any()
                else None,
            },
            "gt_V_lt_0.25_recall": float(gt_hit[gt_V < 0.25].mean())
            if (gt_V < 0.25).any()
            else None,
            "gt_V_ge_0.75_recall": float(gt_hit[gt_V >= 0.75].mean())
            if (gt_V >= 0.75).any()
            else None,
        },
    }

    save = os.path.join(_ROOT, args.save)
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    with open(save, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[v_vs_p] saved {save}", flush=True)


if __name__ == "__main__":
    main()
