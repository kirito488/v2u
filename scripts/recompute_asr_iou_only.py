"""Recompute Init/ours/MADE ASR with IoU-only match (no dxy≤4m), compare batches.

Usage:
  python scripts/recompute_asr_iou_only.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.baselines import apply_baseline  # noqa: E402
from defense.metrics import evaluate_attack_miss  # noqa: E402
from defense.three_source import FrameThreeSource  # noqa: E402
from scripts.recompute_accept_compare import frame_from_dict  # noqa: E402

OLD, NEW = "20260904_210648", "20260905_182559"
IOU = 0.3
# dxy never matches when threshold is negative → pure IoU
DIST_OFF = -1.0

JOBS = [
    # label, old_stem (None if no json), new_stem
    ("remove_early", "compare_remove_early", "compare_remove_early"),
    ("remove_inter", "compare_remove_intermediate", "compare_remove_intermediate"),
    ("remove_late", "compare_remove_late", "compare_remove_late"),
    ("spoof_early", None, "compare_spoof_early"),  # old: log-only, already IoU
    ("spoof_inter", None, "compare_spoof_intermediate"),
    ("spoof_late", None, "compare_spoof_late"),
    ("mass_remove", "compare_mass_remove_intermediate", "compare_mass_remove_intermediate"),
    ("multi_spoof", "compare_multi_spoof_intermediate", "compare_multi_spoof_intermediate"),
]

# Old spoof ASR from logs/HTML (IoU-only era). AP from HTML §8.
OLD_SPOOF_FALLBACK = {
    "spoof_early": {
        "init_asr": 0.247, "ours_asr": 0.062, "made_asr": 0.057,
        "init_ap": 0.453, "ours_ap": 0.387, "made_ap": 0.386,
        "ours_p": 0.810, "ours_r": 0.447, "source": "log/HTML (no json)",
    },
    "spoof_inter": {
        "init_asr": 0.676, "ours_asr": 0.008, "made_asr": 0.058,
        "init_ap": 0.342, "ours_ap": 0.391, "made_ap": 0.371,
        "ours_p": 0.812, "ours_r": 0.448, "source": "log/HTML (no json)",
    },
    "spoof_late": {
        "init_asr": 0.001, "ours_asr": 0.001, "made_asr": None,
        "init_ap": 0.491, "ours_ap": 0.393, "made_ap": None,
        "ours_p": 0.814, "ours_r": 0.449, "source": "log/HTML (no json)",
    },
}


def _ap05(rows):
    for r in rows or []:
        if abs(float(r.get("iou", -1)) - 0.5) < 1e-6:
            return float(r["ap"]), float(r.get("precision", float("nan"))), float(
                r.get("recall", float("nan"))
            )
    return None, None, None


def _init_ap_from_log(stem: str, ts: str):
    import re

    text = ""
    for suf in (f"_{ts}.log", f"_{ts}_cont.log"):
        p = Path(_ROOT) / "logs" / f"{stem}{suf}"
        if p.exists():
            text += p.read_text(errors="ignore") + "\n"
    ms = list(re.finditer(r"^init\s+0\.50\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text, re.M))
    if not ms:
        return None
    return float(ms[-1].group(1))


def recompute(path: Path):
    data = json.load(open(path, encoding="utf-8"))
    mode = str(data.get("mode") or (data.get("attack_miss") or {}).get("mode") or "none")
    frames = [frame_from_dict(d) for d in data.get("frames") or []]
    ours = evaluate_attack_miss(frames, mode, iou_thres=IOU, dist_thres=DIST_OFF)

    # MADE replay (same ego/uav/init dets)
    made_frames = []
    for fr in frames:
        bl = apply_baseline("made", fr.ego, fr.uav, fr.init, ego_lidar=None)
        nf = FrameThreeSource(
            frame_id=fr.frame_id,
            ego=fr.ego,
            uav=fr.uav,
            init=fr.init,
            gt_eval=fr.gt_eval,
            gt_eval_ids=fr.gt_eval_ids,
            baseline_name="made",
            baseline_boxes=None if bl is None else bl.boxes,
            baseline_scores=None if bl is None else bl.scores,
        )
        nf.attack_target = fr.attack_target
        nf.attack_targets = fr.attack_targets
        made_frames.append(nf)
    made = evaluate_attack_miss(made_frames, mode, iou_thres=IOU, dist_thres=DIST_OFF)

    ours_ap, ours_p, ours_r = _ap05(data.get("accepted_metrics"))
    # MADE AP from saved compare (AP already IoU-based; unchanged by ASR match rule)
    made_ap = None
    for c in data.get("compare") or []:
        if c.get("defense") == "made" and c.get("ap05") is not None:
            made_ap = float(c["ap05"])
            break
    init_ap = _init_ap_from_log(path.stem.replace(f"_{OLD}", "").replace(f"_{NEW}", ""), path.stem.split("_")[-1] if False else "")
    # stem like compare_spoof_intermediate_20260905_182559
    stem = path.name.replace(f"_{OLD}.json", "").replace(f"_{NEW}.json", "")
    ts = OLD if OLD in path.name else NEW
    init_ap = _init_ap_from_log(stem, ts)

    return {
        "mode": mode,
        "n_attack": ours["n_attack"],
        "n_targets": ours["n_targets"],
        "init_asr": ours["asr_no_defense"],
        "ours_asr": ours["asr_defense"],
        "made_asr": made["asr_defense"],
        "miss_init": ours["miss_no_defense"],
        "miss_ours": ours["miss_defense"],
        "miss_made": made["miss_defense"],
        "init_ap": init_ap,
        "ours_ap": ours_ap,
        "made_ap": made_ap,
        "ours_p": ours_p,
        "ours_r": ours_r,
        "recorded_init_asr": (data.get("attack_miss") or {}).get("asr_no_defense"),
        "recorded_dist": (data.get("attack_miss") or {}).get("dist_thres"),
        "source": str(path.name),
    }


def pct(x):
    return "—" if x is None else f"{100 * float(x):.1f}%"


def f3(x):
    return "—" if x is None else f"{float(x):.3f}"


def dpp(a, b):
    if a is None or b is None:
        return "—"
    return f"{100 * (float(b) - float(a)):+.1f}pp"


def dap(a, b):
    if a is None or b is None:
        return "—"
    return f"{float(b) - float(a):+.3f}"


def main():
    logdir = Path(_ROOT) / "logs"
    rows = []
    for label, old_stem, new_stem in JOBS:
        old = None
        new = None
        if old_stem:
            p = logdir / f"{old_stem}_{OLD}.json"
            if p.exists():
                print(f"[recompute IoU-only] OLD {p.name} ...", flush=True)
                old = recompute(p)
            else:
                print(f"[skip] missing {p}", flush=True)
        elif label in OLD_SPOOF_FALLBACK:
            old = dict(OLD_SPOOF_FALLBACK[label])
            print(f"[fallback] OLD {label} from {old['source']}", flush=True)

        p = logdir / f"{new_stem}_{NEW}.json"
        print(f"[recompute IoU-only] NEW {p.name} ...", flush=True)
        new = recompute(p)
        rows.append((label, old, new))

    out = {
        "match": "IoU>=0.3 only (dist_thres disabled)",
        "old_ts": OLD,
        "new_ts": NEW,
        "jobs": [
            {"label": lab, "old": o, "new": n} for lab, o, n in rows
        ],
    }
    out_path = logdir / f"compare_asr_iou_only_{OLD}_vs_{NEW}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[save]", out_path)

    print()
    print("=" * 120)
    print("ASR 纯 IoU≥0.3（关闭 dxy≤4m）· 旧 {} → 新 {}".format(OLD, NEW))
    print("=" * 120)
    hdr = (
        f"{'attack':<14} {'Init旧→新':^18} {'ours旧→新':^18} {'Δours':>8} "
        f"{'made旧→新':^18} {'oursAP旧→新':^16} {'ΔAP':>7}"
    )
    print(hdr)
    print("-" * 120)
    for lab, o, n in rows:
        if o is None or n is None:
            print(f"{lab:<14} incomplete")
            continue
        print(
            f"{lab:<14} "
            f"{pct(o.get('init_asr')):>6}→{pct(n.get('init_asr')):<6} "
            f"{pct(o.get('ours_asr')):>6}→{pct(n.get('ours_asr')):<6} "
            f"{dpp(o.get('ours_asr'), n.get('ours_asr')):>8} "
            f"{pct(o.get('made_asr')):>6}→{pct(n.get('made_asr')):<6} "
            f"{f3(o.get('ours_ap')):>5}→{f3(n.get('ours_ap')):<5} "
            f"{dap(o.get('ours_ap'), n.get('ours_ap')):>7}"
        )

    print()
    print("相对 Init 降幅 (InitASR−oursASR)，越大越好：")
    print(f"{'attack':<14} {'旧Δ':>8} {'新Δ':>8} {'改善':>8}")
    print("-" * 50)
    for lab, o, n in rows:
        if not o or not n:
            continue
        if o.get("init_asr") is None or o.get("ours_asr") is None:
            continue
        do = float(o["init_asr"]) - float(o["ours_asr"])
        dn = float(n["init_asr"]) - float(n["ours_asr"])
        print(f"{lab:<14} {100*do:+6.1f}pp {100*dn:+6.1f}pp {100*(dn-do):+6.1f}pp")

    print()
    print("注：multi_spoof / mass_remove 的 JSON 只 dump 了单个 attack_target，")
    print("    实例级多目标 ASR 无法从 dump 完整还原；两边用同一限制，跨批仍可比。")
    print("    AP/P/R 仍用原 accepted_metrics（AP 匹配本就是 IoU，不受 dist_thres 影响）。")


if __name__ == "__main__":
    main()
