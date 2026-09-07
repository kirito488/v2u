#!/usr/bin/env python3
"""Calibrate n_ref / r0 / A_ref from a clear-car reference pool (median).

Pool (per source): clean TP dets with P>=score_thres, V>=v_min, r in [r_lo,r_hi].
Per car: κ_i = n_i * r_i^2 / A_i
Median → κ, r0, A_ref;  n_ref = n0 = κ * A_ref / r0^2

  python scripts/calibrate_confidence.py --gpu 3 --score_thres 0.3 \\
      --v_min 0.9 --save logs/calibrate_confidence.json
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
    boxes_geometric_center,
    range_xy,
    visible_area,
)
from defense.geometry import count_points_in_box, iou_bev, pack_dets  # noqa: E402
from defense.occlusion import build_polar_depth, visibility_ratio_lidar  # noqa: E402
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


def _median(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    return float(np.median(a)) if a.size else float("nan")


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


def _collect_sensor(
    boxes_sensor,
    scores,
    lidar,
    gts_match_frame,
    boxes_for_match,
    source: str,
    score_thres: float,
    v_min: float,
    r_lo: float,
    r_hi: float,
    pad: float,
    bottom_center: bool,
):
    """boxes_sensor: in lidar frame for n/r/A/V. boxes_for_match: same index, GT frame."""
    boxes_sensor = np.asarray(boxes_sensor) if boxes_sensor is not None else np.zeros((0, 7))
    boxes_for_match = (
        np.asarray(boxes_for_match) if boxes_for_match is not None else boxes_sensor
    )
    scores = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((0,))
    if boxes_sensor.ndim != 2 or boxes_sensor.shape[0] == 0:
        return []
    depth = build_polar_depth(lidar)
    if bottom_center:
        boxes_cnt = boxes_geometric_center(boxes_sensor)
        boxes_vis = boxes_sensor  # visibility_ratio_lidar lifts if bottom_center
    else:
        boxes_cnt = boxes_sensor
        boxes_vis = boxes_sensor
    rows = []
    n_box = int(boxes_sensor.shape[0])
    for i in range(n_box):
        p = float(scores[i]) if i < int(scores.shape[0]) else 0.0
        if p < float(score_thres):
            continue
        b_match = boxes_for_match[i, :7] if i < int(boxes_for_match.shape[0]) else boxes_sensor[i, :7]
        if not _match_gt(b_match, gts_match_frame):
            continue
        b_cnt = boxes_cnt[i, :7]
        r = float(range_xy(b_cnt, r_min=R_MIN))
        if r < float(r_lo) or r > float(r_hi):
            continue
        v = float(
            visibility_ratio_lidar(
                boxes_vis[i],
                depth=depth,
                bottom_center=bool(bottom_center),
            )
        )
        if v < float(v_min):
            continue
        n = int(count_points_in_box(lidar, b_cnt, pad=pad))
        if n <= 0:
            continue
        a = float(visible_area(b_cnt, source))
        if a <= 1e-6:
            continue
        rows.append(
            {
                "n": n,
                "r": r,
                "A": a,
                "V": v,
                "P": p,
                "kappa": float(n) * (r ** 2) / a,
            }
        )
    return rows


def _aggregate(rows, source: str):
    if not rows:
        return {"source": source, "n_pool": 0}
    kappas = [x["kappa"] for x in rows]
    rs = [x["r"] for x in rows]
    As = [x["A"] for x in rows]
    ns = [x["n"] for x in rows]
    kappa = _median(kappas)
    r0 = _median(rs)
    a_ref = _median(As)
    n0 = float(kappa * a_ref / max(r0 ** 2, 1e-6))
    return {
        "source": source,
        "n_pool": len(rows),
        "kappa": kappa,
        "r0": r0,
        "A_ref": a_ref,
        "n_ref": n0,
        "n0": n0,
        "n_median": _median(ns),
        "n_p25": float(np.percentile(ns, 25)),
        "n_p70": float(np.percentile(ns, 70)),
        "n_p75": float(np.percentile(ns, 75)),
        "V_median": _median([x["V"] for x in rows]),
        "r_p25": float(np.percentile(rs, 25)),
        "r_p75": float(np.percentile(rs, 75)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--max_scenes", type=int, default=0, help="0 = all val scenes")
    ap.add_argument("--score_thres", type=float, default=0.3)
    ap.add_argument("--v_min", type=float, default=0.9)
    ap.add_argument("--r_lo", type=float, default=8.0)
    ap.add_argument("--r_hi", type=float, default=40.0)
    ap.add_argument("--save", type=str, default="logs/calibrate_confidence.json")
    args = ap.parse_args()

    dataset = V2U4RealDataset(root_path=val_dir(), mode="val")
    scenes = all_dataset_scenes(dataset)
    items = list(scenes.items())
    if args.max_scenes and args.max_scenes > 0:
        items = items[: int(args.max_scenes)]

    print(
        "[calib] scenes={} V>={} P>={} r∈[{},{}]".format(
            len(items), args.v_min, args.score_thres, args.r_lo, args.r_hi
        ),
        flush=True,
    )
    for n, _ in items:
        print(" ", n, flush=True)

    print("[load] PointPillars ego/uav ...", flush=True)
    ego_p = load_late_pointpillar(EGO_POINTPILLAR_DIR)
    uav_p = load_late_pointpillar(UAV_POINTPILLAR_DIR)

    pool = {"ego": [], "uav": []}
    scene_counts = {}

    for si, (sname, rec) in enumerate(items):
        print(
            "\n[calib] scene {}/{} {} cases={}".format(
                si + 1, len(items), sname, rec["cases"]
            ),
            flush=True,
        )
        n_e0, n_u0 = len(pool["ego"]), len(pool["uav"])
        mfc = dataset.get_case_by_meta(
            {"scenario_name": sname, "frame_ids": rec["frame_ts"]},
            tag="multi_frame",
        )
        for f in range(len(mfc)):
            frame = mfc[f]
            gt_ego = frame[EGO_ID].get("gt_bboxes")
            pose_e = frame[EGO_ID]["lidar_pose"]
            pose_u = frame[UAV_ID]["lidar_pose"]
            lidar_e = frame[EGO_ID]["lidar"]
            lidar_u = frame[UAV_ID]["lidar"]

            # Ego late PP: boxes in ego frame
            try:
                eb, es = pack_dets(*ego_p.run(frame, EGO_ID))
            except Exception as e:
                print("  ego fail f={}: {}".format(f, e), flush=True)
                eb, es = np.zeros((0, 7)), np.zeros((0,))
            rows_e = _collect_sensor(
                eb,
                es,
                lidar_e,
                gt_ego,
                eb,
                "ego",
                args.score_thres,
                args.v_min,
                args.r_lo,
                args.r_hi,
                pad=0.25,
                bottom_center=True,
            )
            for row in rows_e:
                row.update(scene=sname, frame=int(f))
            pool["ego"].extend(rows_e)

            # UAV late PP: boxes projected to ego (same as three_source);
            # count/V/A in UAV frame after pose warp.
            try:
                ub_ego, us = pack_dets(*uav_p.run(frame, EGO_ID))
            except Exception as e:
                print("  uav fail f={}: {}".format(f, e), flush=True)
                ub_ego, us = np.zeros((0, 7)), np.zeros((0,))
            if ub_ego.ndim == 2 and ub_ego.shape[0] > 0:
                ub_u = _boxes_to_pose(
                    boxes_geometric_center(ub_ego), pose_e, pose_u
                )
                rows_u = _collect_sensor(
                    ub_u,
                    us,
                    lidar_u,
                    gt_ego,  # match in ego frame
                    ub_ego,
                    "uav",
                    args.score_thres,
                    args.v_min,
                    args.r_lo,
                    args.r_hi,
                    pad=0.5,
                    bottom_center=False,  # already geometric center
                )
            else:
                rows_u = []
            for row in rows_u:
                row.update(scene=sname, frame=int(f))
            pool["uav"].extend(rows_u)

        scene_counts[sname] = {
            "ego_added": len(pool["ego"]) - n_e0,
            "uav_added": len(pool["uav"]) - n_u0,
        }
        print(
            "  pool ego={} (+{}) uav={} (+{})".format(
                len(pool["ego"]),
                scene_counts[sname]["ego_added"],
                len(pool["uav"]),
                scene_counts[sname]["uav_added"],
            ),
            flush=True,
        )

    ego_agg = _aggregate(pool["ego"], "ego")
    uav_agg = _aggregate(pool["uav"], "uav")
    out = {
        "score_thres": args.score_thres,
        "v_min": args.v_min,
        "r_lo": args.r_lo,
        "r_hi": args.r_hi,
        "aggregate": "median",
        "scenes": [n for n, _ in items],
        "scene_counts": scene_counts,
        "ego": ego_agg,
        "uav": uav_agg,
        "old_defaults": {
            "N_REF_EGO": 40.0,
            "N_REF_UAV": 25.0,
            "R0": 20.0,
            "A_ref_ego": 1.5 * 4.5,
            "A_ref_uav": 4.5 * 1.8,
        },
        "recommended_cli": {
            "n_ref": ego_agg.get("n_ref"),
            "n_ref_uav": uav_agg.get("n_ref"),
            "r0": ego_agg.get("r0"),
            "r0_uav": uav_agg.get("r0"),
            "A_ref_ego": ego_agg.get("A_ref"),
            "A_ref_uav": uav_agg.get("A_ref"),
        },
    }
    save = args.save
    os.makedirs(os.path.dirname(os.path.abspath(save)) or ".", exist_ok=True)
    with open(save, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    pool_path = save.replace(".json", "_pool.json")
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump({"ego": pool["ego"], "uav": pool["uav"]}, f)

    print("\n=== Calibration (median over clear-car pool) ===", flush=True)
    for agg in (ego_agg, uav_agg):
        print(
            "[{}] n_pool={}  κ={:.4f}  r0={:.2f}m  A_ref={:.3f}  n_ref=n0={:.2f}  "
            "n_med={:.1f} n_p75={:.1f}".format(
                agg.get("source"),
                agg.get("n_pool", 0),
                float(agg.get("kappa") or float("nan")),
                float(agg.get("r0") or float("nan")),
                float(agg.get("A_ref") or float("nan")),
                float(agg.get("n_ref") or float("nan")),
                float(agg.get("n_median") or float("nan")),
                float(agg.get("n_p75") or float("nan")),
            ),
            flush=True,
        )
    print("[save]", save, flush=True)
    print("[save]", pool_path, flush=True)


if __name__ == "__main__":
    main()
