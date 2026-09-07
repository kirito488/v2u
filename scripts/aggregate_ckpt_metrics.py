#!/usr/bin/env python3
"""Aggregate ours metrics from scene ckpts WITHOUT loading lidars / models.

Avoids OOM from loading ~20G+ of ego lidars needed only for MADE replay.

  python scripts/aggregate_ckpt_metrics.py \\
      --ckpt_dir logs/qablate_clean_q014_004_20260907_140228_ckpt \\
      --save logs/qablate_clean_q014_004_20260907_140228.json \\
      --mode none --q_h 0.14 --q_l 0.04
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.metrics import (  # noqa: E402
    evaluate_accepted,
    evaluate_attack_miss,
    evaluate_defense,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--mode", default="none")
    ap.add_argument("--level", default="early")
    ap.add_argument("--q_h", type=float, default=None)
    ap.add_argument("--q_l", type=float, default=None)
    ap.add_argument("--iou_thres", type=float, default=0.3)
    ap.add_argument("--asr_dist_thres", type=float, default=0.0)
    ap.add_argument("--calib_json", default="logs/calibrate_confidence.json")
    args = ap.parse_args()

    man_path = os.path.join(args.ckpt_dir, "manifest.json")
    man = json.load(open(man_path)) if os.path.isfile(man_path) else {}
    done = list(man.get("done") or [])
    fp = dict(man.get("fingerprint") or {})
    if not done:
        # fallback: all scene_*.pkl
        done = sorted(
            f[6:-4]
            for f in os.listdir(args.ckpt_dir)
            if f.startswith("scene_") and f.endswith(".pkl")
        )
        # undo safe replace? names are as-is in filename after scene_
        done = [x for x in done]

    all_frames = []
    for si, scene in enumerate(done):
        path = os.path.join(args.ckpt_dir, "scene_{}.pkl".format(scene))
        if not os.path.isfile(path):
            # try safe name
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in scene)
            path = os.path.join(args.ckpt_dir, "scene_{}.pkl".format(safe))
        print("[load] {}/{} {}".format(si + 1, len(done), path), flush=True)
        with open(path, "rb") as f:
            blob = pickle.load(f)
        frs = list(blob.get("frames") or [])
        # drop lidars immediately
        if "lidars" in blob:
            del blob["lidars"]
        del blob
        all_frames.extend(frs)
        print("  +{} frames  total={}".format(len(frs), len(all_frames)), flush=True)

    defense_metrics = evaluate_defense(all_frames)
    accepted_metrics = evaluate_accepted(all_frames)
    attack_miss = evaluate_attack_miss(
        all_frames,
        args.mode,
        args.iou_thres,
        dist_thres=args.asr_dist_thres,
    )

    def ap_at(rows, iou=0.5):
        for r in rows or []:
            if abs(float(r.get("iou", -1)) - iou) < 1e-6:
                return float(r.get("ap", 0))
        return None

    compare = [
        {
            "defense": "ours",
            "ap025": ap_at(accepted_metrics, 0.25),
            "ap05": ap_at(accepted_metrics, 0.5),
            "ap07": ap_at(accepted_metrics, 0.7),
            "asr_defense": attack_miss.get("asr_defense"),
            "asr_no_defense": attack_miss.get("asr_no_defense"),
            "miss_defense": attack_miss.get("miss_defense"),
            "n_attack": attack_miss.get("n_attack"),
            "note": "offline agg (frames only; no MADE)",
        }
    ]

    q_h = args.q_h if args.q_h is not None else fp.get("q_h")
    q_l = args.q_l if args.q_l is not None else fp.get("q_l")
    calib = {}
    if args.calib_json and os.path.isfile(args.calib_json):
        calib = json.load(open(args.calib_json))

    out = {
        "offline_agg": True,
        "ckpt_dir": args.ckpt_dir,
        "scenes": done,
        "n_frames": len(all_frames),
        "mode": args.mode,
        "level": args.level,
        "q_h": q_h,
        "q_l": q_l,
        "score_thres": fp.get("score_thres", 0.3),
        "theta_p": fp.get("theta_p", 0.3),
        "calib_json": args.calib_json,
        "n_ref": (calib.get("ego") or {}).get("n_ref"),
        "n_ref_uav": (calib.get("uav") or {}).get("n_ref"),
        "r0": (calib.get("ego") or {}).get("r0"),
        "r0_uav": (calib.get("uav") or {}).get("r0"),
        "a_ref_ego": (calib.get("ego") or {}).get("A_ref"),
        "a_ref_uav": (calib.get("uav") or {}).get("A_ref"),
        "defense_metrics": defense_metrics,
        "accepted_metrics": accepted_metrics,
        "attack_miss": attack_miss,
        "compare": compare,
        "fingerprint": fp,
        # frames omitted (large); metrics already aggregated above
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(out, f)
    print("[save]", args.save, "frames=", len(all_frames), flush=True)
    print(
        "ours AP@.5={} ASR={} initASR={}".format(
            compare[0]["ap05"], compare[0]["asr_defense"], compare[0]["asr_no_defense"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
