#!/usr/bin/env python3
"""Compare soft02 clean_none vs 20260905 (Init/Ego/UAV/Accept/Defense + attack ASR).

Memory-safe streaming:
- NEW: load one scene ckpt at a time, drop lidars, accumulate Init/Ego/UAV AP lists
- NEW Accept/Defense: reuse offline JSON
- OLD: frame_from_dict on 29MB dump for Init/Ego/UAV; Accept/Defense from JSON
"""
from __future__ import annotations

import gc
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.metrics import (  # noqa: E402
    IOU_LEVELS,
    ap_from_lists,
    eval_gt_boxes,
    match_score_sorted,
    pr_counts,
)
from scripts.recompute_accept_compare import frame_from_dict  # noqa: E402

CKPT = os.path.join(_ROOT, "logs/attack_suite_soft02/clean_none_20260909_112324_ckpt")
NEW_JSON = os.path.join(_ROOT, "logs/attack_suite_soft02/clean_none_20260909_112324.json")
OLD_JSON = os.path.join(_ROOT, "logs/clean_none_20260905_182559.json")
OUT_JSON = os.path.join(_ROOT, "logs/attack_suite_soft02/clean_none_compare_vs_20260905.json")


def ap_table(rows, name):
    print(f"\n### {name}", flush=True)
    print(
        f"{'IoU':>6} {'AP':>8} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'GT':>6}",
        flush=True,
    )
    for r in rows or []:
        print(
            f"{float(r['iou']):6.2f} {float(r['ap']):8.4f} {float(r['precision']):8.4f} "
            f"{float(r['recall']):8.4f} {float(r['f1']):8.4f} {int(r['tp']):6d} "
            f"{int(r['fp']):6d} {int(r['fn']):6d} {int(r['gt']):6d}",
            flush=True,
        )


def get(rows, iou=0.5):
    for r in rows or []:
        if abs(float(r.get("iou", -1)) - iou) < 1e-6:
            return r
    return {}


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, list):
        return [conv(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        info = {}
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.strip().split()[0])
    return info.get("MemAvailable", info.get("MemFree", 0)) / (1024.0 ** 2)


def wait_for_ram(need_gb: float, tag: str):
    while True:
        avail = mem_avail_gb()
        if avail >= need_gb:
            print(f"[ram] {tag}: avail={avail:.1f}G >= need={need_gb:.1f}G", flush=True)
            return
        print(f"[ram] wait {tag}: avail={avail:.1f}G < need={need_gb:.1f}G", flush=True)
        gc.collect()
        time.sleep(30)


def rows_from_acc(acc, source_name):
    """acc[iou] = {tp_all, fp_all, n_gt}"""
    rows = []
    for th in IOU_LEVELS:
        a = acc[float(th)]
        ap = ap_from_lists(a["tp"], a["fp"], a["n_gt"])
        tps, fps, fn, prec, rec, f1 = pr_counts(a["tp"], a["fp"], a["n_gt"])
        rows.append(
            {
                "source": source_name,
                "iou": float(th),
                "ap": ap,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tps,
                "fp": fps,
                "fn": fn,
                "gt": a["n_gt"],
            }
        )
    return rows


def new_acc():
    return {
        float(th): {"tp": [], "fp": [], "n_gt": 0} for th in IOU_LEVELS
    }


def accumulate_frames(frames, accs):
    """accs: dict attr -> acc"""
    for fr in frames:
        gts = eval_gt_boxes(fr)
        for attr, acc in accs.items():
            src = getattr(fr, attr)
            for th in IOU_LEVELS:
                tp, fp, ng = match_score_sorted(src.boxes, src.scores, gts, th)
                bucket = acc[float(th)]
                bucket["tp"].extend(tp)
                bucket["fp"].extend(fp)
                # n_gt counted once per iou level identically
                if th == IOU_LEVELS[0]:
                    pass
                bucket["n_gt"] += ng


def eval_new_sources():
    man = json.load(open(os.path.join(CKPT, "manifest.json")))
    done = list(man.get("done") or [])
    sized = []
    for s in done:
        p = os.path.join(CKPT, f"scene_{s}.pkl")
        sized.append((os.path.getsize(p) if os.path.isfile(p) else 0, s))
    sized.sort()  # small first

    accs = {"init": new_acc(), "ego": new_acc(), "uav": new_acc()}
    n_frames = 0
    for i, (sz, s) in enumerate(sized):
        path = os.path.join(CKPT, f"scene_{s}.pkl")
        # Peak ≈ pickle size + modest overhead; wait rather than OOM.
        need = max(5.5, (sz / (1024.0 ** 3)) * 2.0 + 1.5)
        wait_for_ram(need, s)
        print(
            f"[NEW] load {i+1}/{len(sized)} {s} ({sz/1e9:.2f}G) avail={mem_avail_gb():.1f}G",
            flush=True,
        )
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if "lidars" in blob:
            del blob["lidars"]
        frames = list(blob.get("frames") or [])
        del blob
        gc.collect()
        accumulate_frames(frames, accs)
        n_frames += len(frames)
        print(f"  +{len(frames)} frames total={n_frames} avail={mem_avail_gb():.1f}G", flush=True)
        del frames
        gc.collect()

    out = {k: rows_from_acc(accs[k], k) for k in ("init", "ego", "uav")}
    out["n_frames"] = n_frames
    return out, man


def eval_old_sources():
    wait_for_ram(6.0, "OLD_JSON")
    old_dump = json.load(open(OLD_JSON))
    frames = [frame_from_dict(d) for d in old_dump.get("frames") or []]
    print(f"OLD n_frames={len(frames)}", flush=True)
    accs = {"init": new_acc(), "ego": new_acc(), "uav": new_acc()}
    accumulate_frames(frames, accs)
    out = {k: rows_from_acc(accs[k], k) for k in ("init", "ego", "uav")}
    out["accepted"] = old_dump.get("accepted_metrics")
    out["defense"] = old_dump.get("defense_metrics")
    out["n_frames"] = len(frames)
    old_params = {
        k: old_dump.get(k)
        for k in ("score_thres", "theta_p", "q_h", "q_l", "mode", "defense", "n_objects")
    }
    del frames, old_dump
    gc.collect()
    return out, old_params


def main():
    print("=== NEW soft02 streaming Init/Ego/UAV ===", flush=True)
    new, man = eval_new_sources()
    if os.path.isfile(NEW_JSON):
        nj = json.load(open(NEW_JSON))
        new["accepted"] = nj.get("accepted_metrics")
        new["defense"] = nj.get("defense_metrics")
        fp = nj.get("fingerprint") or man.get("fingerprint") or {}
        print("[NEW] reused accepted/defense from offline JSON", flush=True)
    else:
        raise SystemExit("missing NEW JSON for accept/defense")
    for k in ("init", "ego", "uav", "accepted", "defense"):
        ap_table(new[k], f"NEW soft02 {k}")

    print("\n=== OLD 20260905 Init/Ego/UAV ===", flush=True)
    old, old_params = eval_old_sources()
    for k in ("init", "ego", "uav", "accepted", "defense"):
        ap_table(old[k], f"OLD 20260905 {k}")

    print("\n======== @IoU=0.5 并排（含无防御 Init） ========", flush=True)
    print(
        f"{'source':<14} {'NEW AP':>8} {'OLD AP':>8} {'ΔAP':>8} "
        f"{'NEW P':>8} {'OLD P':>8} {'NEW R':>8} {'OLD R':>8}",
        flush=True,
    )
    side = {}
    for name, key in [
        ("Init无防御", "init"),
        ("Ego", "ego"),
        ("UAV", "uav"),
        ("Accept ours", "accepted"),
        ("Defense buf", "defense"),
    ]:
        ra, rb = get(new[key]), get(old[key])
        da = float(ra.get("ap", 0)) - float(rb.get("ap", 0))
        side[name] = {"new": ra, "old": rb, "dap": da}
        print(
            f"{name:<14} {float(ra.get('ap', 0)):8.4f} {float(rb.get('ap', 0)):8.4f} {da:8.4f} "
            f"{float(ra.get('precision', 0)):8.4f} {float(rb.get('precision', 0)):8.4f} "
            f"{float(ra.get('recall', 0)):8.4f} {float(rb.get('recall', 0)):8.4f}",
            flush=True,
        )

    print(
        "\n======== 全场景全攻击 20260905：AcceptAP / ASR无防御(Init) / ASR有防御(Ours) ========",
        flush=True,
    )
    atk_rows = []
    paths = sorted(glob.glob(os.path.join(_ROOT, "logs/compare_*_20260905_182559.json")))
    paths = [p for p in paths if "asr_iou" not in p]
    paths.append(OLD_JSON)
    print(
        f"{'tag':<28} {'AccAP@.5':>9} {'P':>7} {'R':>7} "
        f"{'ASR_Init':>9} {'ASR_Ours':>9} {'ΔASR':>8} {'n_atk':>6}",
        flush=True,
    )
    for p in paths:
        d = json.load(open(p))
        tag = (
            os.path.basename(p)
            .replace("compare_", "")
            .replace("_20260905_182559.json", "")
            .replace(".json", "")
        )
        if tag == "clean_none_20260905_182559":
            tag = "clean_none"
        ours = next((c for c in d.get("compare") or [] if c.get("defense") == "ours"), {})
        am = d.get("attack_miss") or {}
        acc = get(d.get("accepted_metrics") or [])
        ai = ours.get("asr_no_defense", am.get("asr_no_defense"))
        ao = ours.get("asr_defense", am.get("asr_defense"))
        dasr = (float(ao) - float(ai)) if ai is not None and ao is not None else None
        row = {
            "tag": tag,
            "accept_ap05": acc.get("ap", ours.get("ap05")),
            "P": acc.get("precision"),
            "R": acc.get("recall"),
            "asr_init": ai,
            "asr_ours": ao,
            "dasr": dasr,
            "n_attack": ours.get("n_attack", am.get("n_attack")),
        }
        atk_rows.append(row)
        print(
            f"{tag:<28} {float(row['accept_ap05'] or 0):9.4f} {float(row['P'] or 0):7.4f} "
            f"{float(row['R'] or 0):7.4f} {float(ai or 0):9.4f} {float(ao or 0):9.4f} "
            f"{float(dasr or 0):8.4f} {int(row['n_attack'] or 0):6d}",
            flush=True,
        )

    summary = {
        "new_soft02_clean": new,
        "old_20260905_clean": old,
        "old_params": old_params,
        "new_fingerprint": {
            k: fp.get(k)
            for k in [
                "theta_soft",
                "score_thres",
                "theta_p",
                "q_h",
                "q_l",
                "theta_confirm",
                "theta_reject",
                "kappa_pos",
                "kappa_neg",
                "v_min",
                "bev_tau",
                "bev_c_min",
                "det_score_thres",
                "t_timeout",
                "k_atk",
                "pool_k",
                "beta_v",
                "gamma_r",
                "tau",
                "gate_q_mode",
                "buf_q_mode",
                "defense",
            ]
        },
        "side_by_side_iou05": side,
        "attack_suite_20260905": atk_rows,
    }
    json.dump(conv(summary), open(OUT_JSON, "w"), indent=2)
    print(f"\n[saved] {OUT_JSON}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
