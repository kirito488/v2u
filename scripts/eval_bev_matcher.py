"""Evaluate trained BevLocalMatcher on V2U4 val (frozen Ego/UAV PointPillars)."""
from __future__ import annotations

import argparse
import json
import os
import sys


def _bind_gpu():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", type=int, default=None)
    args, _ = p.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    return args.gpu


_BIND_GPU = _bind_gpu()

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from defense.paths import (
    EGO_ID,
    UAV_ID,
    EGO_POINTPILLAR_DIR,
    UAV_POINTPILLAR_DIR,
    V2U4REAL_ROOT,
    setup_import_paths,
    train_dir,
    val_dir,
)

setup_import_paths()

from mvp.data.v2u4real_dataset import V2U4RealDataset
from mvp.perception.opencood_perception import OpencoodPerception

from defense.associate import hungarian_match
from defense.bev_matcher import BevGrid, crop_patches, last_bev_tensor, load_matcher
from defense.confidence import boxes_geometric_center
from defense.geometry import count_points_in_box, iou_bev, pack_dets
from defense.three_source import build_v2u4_eval_gt
from scripts.train_bev_matcher import (
    _best_iou,
    _n_ego_pts,
    iter_frames,
    label_uav_boxes,
    load_late,
    run_pair,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=os.path.join(_ROOT, "logs/bev_matcher/matcher_last.pth"))
    ap.add_argument("--tau", type=float, default=0.50)
    ap.add_argument(
        "--taus",
        type=str,
        default="",
        help="comma-separated thresholds, e.g. 0.6,0.7,0.8 (overrides --tau)",
    )
    ap.add_argument("--c_min", type=float, default=0.0, help="unused here; promotion uses S only")
    ap.add_argument("--max_cases", type=int, default=30)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--det_score", type=float, default=0.1)
    ap.add_argument("--out", type=str, default=os.path.join(_ROOT, "logs/bev_matcher/eval_val.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher, meta = load_matcher(args.ckpt, device=device)
    k = int(meta.get("k") or matcher.k)
    grid = BevGrid()
    if str(args.taus or "").strip():
        taus = [float(x) for x in args.taus.split(",") if x.strip()]
    else:
        taus = [float(args.tau)]
    print("[load] matcher", args.ckpt, "meta", meta, "taus", taus, flush=True)

    ds = V2U4RealDataset(val_dir(), mode="val")
    ego_perc = load_late(EGO_POINTPILLAR_DIR, args.det_score)
    uav_perc = load_late(UAV_POINTPILLAR_DIR, args.det_score)

    buckets = {"pos": [], "hard": [], "fp": [], "other": []}
    recs = []  # s, hit, no_ego, kind
    n_uav = n_frame = 0

    matcher.eval()
    for ci, fi, frame in iter_frames(ds, args.max_cases, args.frame_stride):
        if EGO_ID not in frame or UAV_ID not in frame:
            continue
        try:
            b_e, b_u, fe, fu = run_pair(ego_perc, uav_perc, frame)
        except Exception as e:
            print("[warn]", ci, fi, e, flush=True)
            continue
        if fe is None or fu is None or b_u.shape[0] == 0:
            continue
        gt, _ = build_v2u4_eval_gt(frame, EGO_ID, UAV_ID)
        ego_lidar = frame[EGO_ID].get("lidar")
        fe_t = fe.float().to(device)
        fu_t = fu.float().to(device)
        with torch.no_grad():
            pe = crop_patches(fe_t, b_u, k=k, grid=grid)
            pu = crop_patches(fu_t, b_u, k=k, grid=grid)
            s = matcher(pe, pu).detach().cpu().numpy()

        used_e = set()
        if b_e.shape[0] and b_u.shape[0]:
            for a, b, _iou in hungarian_match(b_e, b_u, 0.3):
                used_e.add(int(b))

        n_frame += 1
        n_uav += int(b_u.shape[0])
        for i, box in enumerate(b_u):
            si = float(s[i])
            iou = _best_iou(box, gt)
            npts = _n_ego_pts(ego_lidar, box)
            kind = "pos" if iou >= 0.3 and npts >= 10 else (
                "hard" if iou >= 0.3 and npts <= 2 else (
                    "fp" if iou < 0.1 else "other"
                )
            )
            buckets[kind].append(si)
            recs.append((si, iou >= 0.3, i not in used_e, kind))

        if n_frame % 20 == 0:
            print("[prog] frames", n_frame, "uav", n_uav, flush=True)

    def _summ(xs, tau):
        xs = np.asarray(xs, dtype=np.float64)
        if xs.size == 0:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "ge_tau": 0}
        return {
            "n": int(xs.size),
            "mean": float(xs.mean()),
            "p50": float(np.median(xs)),
            "ge_tau": int((xs >= float(tau)).sum()),
        }

    def _at_tau(tau):
        n_p = n_gt = n_noe = n_noe_gt = 0
        kind_ge = {k: 0 for k in buckets}
        for si, hit, noe, kind in recs:
            if si < tau:
                continue
            n_p += 1
            kind_ge[kind] = kind_ge.get(kind, 0) + 1
            if hit:
                n_gt += 1
            if noe:
                n_noe += 1
                if hit:
                    n_noe_gt += 1
        return {
            "tau": tau,
            "n_promote": n_p,
            "n_promote_gt": n_gt,
            "prec_promote": (n_gt / n_p) if n_p else 0.0,
            "n_promote_no_ego": n_noe,
            "n_promote_no_ego_gt": n_noe_gt,
            "prec_promote_no_ego": (n_noe_gt / n_noe) if n_noe else 0.0,
            "ge_tau_by_kind": kind_ge,
        }

    by_tau = [_at_tau(t) for t in taus]
    out = {
        "ckpt": args.ckpt,
        "taus": taus,
        "n_frame": n_frame,
        "n_uav": n_uav,
        "by_kind": {k: _summ(v, taus[0]) for k, v in buckets.items()},
        "by_tau": by_tau,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print("\n=== tau sweep ===", flush=True)
    print("tau  promote  prec    noEgo  noEgo_GT  noEgo_prec", flush=True)
    for r in by_tau:
        print(
            "{:.2f}  {:>5d}   {:.3f}   {:>5d}   {:>5d}     {:.3f}".format(
                r["tau"], r["n_promote"], r["prec_promote"],
                r["n_promote_no_ego"], r["n_promote_no_ego_gt"], r["prec_promote_no_ego"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
