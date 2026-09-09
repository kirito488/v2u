"""Train BevLocalMatcher on V2U4Real train (frozen Ego/UAV PointPillars).

Positive: UAV det matches GT and Ego cloud has enough points at that box.
Hard-neg: UAV det matches GT but Ego points ≈ 0 (occlusion / out of view).
FP-neg:   UAV det matches no GT.
Shift-neg: positive UAV crop vs Ego crop shifted 8–20 m.

Usage (repo root):
    python scripts/train_bev_matcher.py --gpu 0
    python scripts/train_bev_matcher.py --gpu 0 --max_cases 20 --epochs 2   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time


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
import torch.nn.functional as F

from defense.paths import (  # noqa: E402
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

from mvp.data.v2u4real_dataset import V2U4RealDataset  # noqa: E402
from mvp.perception.opencood_perception import OpencoodPerception  # noqa: E402

from defense.bev_matcher import (  # noqa: E402
    DEFAULT_IN_CH,
    DEFAULT_K,
    BevGrid,
    BevLocalMatcher,
    crop_patches,
    last_bev_tensor,
)
from defense.confidence import boxes_geometric_center  # noqa: E402
from defense.geometry import count_points_in_box, iou_bev, pack_dets  # noqa: E402
from defense.three_source import build_v2u4_eval_gt  # noqa: E402


def load_late(model_dir: str, score_threshold: float) -> OpencoodPerception:
    perc = OpencoodPerception(
        fusion_method="late",
        model_name="pointpillar",
        model_dir=model_dir,
        opencood_root=V2U4REAL_ROOT,
        root_dir=train_dir(),
        validate_dir=val_dir(),
        score_threshold=score_threshold,
    )
    perc.model.eval()
    for p in perc.model.parameters():
        p.requires_grad_(False)
    return perc


def _best_iou(box, gts) -> float:
    gts = np.asarray(gts) if gts is not None else np.zeros((0, 7))
    if gts.ndim != 2 or gts.shape[0] == 0:
        return 0.0
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
    return max(float(iou_bev(b, g)) for g in gts)


def _n_ego_pts(lidar, box) -> int:
    b = np.asarray(box, dtype=np.float64).reshape(1, 7)
    b = boxes_geometric_center(b)[0]
    return int(count_points_in_box(lidar, b, pad=0.25))


def _shift_box(box, dist: float) -> np.ndarray:
    b = np.asarray(box, dtype=np.float64)[:7].copy()
    ang = random.uniform(0.0, 2.0 * np.pi)
    b[0] += float(dist) * np.cos(ang)
    b[1] += float(dist) * np.sin(ang)
    return b


def label_uav_boxes(uav_boxes, gt_boxes, ego_lidar, pos_pts=10, hard_pts=2, iou_pos=0.3, iou_fp=0.1):
    """Return lists of (box, kind) with kind in {pos, hard, fp}."""
    rows = []
    uav_boxes = np.asarray(uav_boxes) if uav_boxes is not None else np.zeros((0, 7))
    if uav_boxes.ndim != 2 or uav_boxes.shape[0] == 0:
        return rows
    for b in uav_boxes:
        iou = _best_iou(b, gt_boxes)
        n = _n_ego_pts(ego_lidar, b)
        if iou >= float(iou_pos) and n >= int(pos_pts):
            rows.append((b[:7].copy(), "pos"))
        elif iou >= float(iou_pos) and n <= int(hard_pts):
            rows.append((b[:7].copy(), "hard"))
        elif iou < float(iou_fp):
            rows.append((b[:7].copy(), "fp"))
    return rows


def balance_frame(rows, n_pos_cap=8):
    pos = [r for r in rows if r[1] == "pos"]
    hard = [r for r in rows if r[1] == "hard"]
    fp = [r for r in rows if r[1] == "fp"]
    if not pos:
        return []
    random.shuffle(pos)
    pos = pos[: int(n_pos_cap)]
    n = len(pos)
    random.shuffle(hard)
    random.shuffle(fp)
    hard = hard[:n]
    fp = fp[: max(n, 2 * n - len(hard))]
    out = list(pos) + list(hard) + list(fp)
    # one spatial-mismatch neg per positive
    for b, _ in pos:
        out.append((b, "shift"))
    random.shuffle(out)
    return out


def crops_for_kind(feat_e, feat_u, box, kind, k, grid):
    b = np.asarray(box, dtype=np.float64).reshape(1, 7)
    pu = crop_patches(feat_u, b, k=k, grid=grid)
    if kind == "shift":
        sb = _shift_box(b[0], dist=random.uniform(8.0, 20.0)).reshape(1, 7)
        pe = crop_patches(feat_e, sb, k=k, grid=grid)
        y = 0.0
    else:
        pe = crop_patches(feat_e, b, k=k, grid=grid)
        y = 1.0 if kind == "pos" else 0.0
    return pe, pu, y


@torch.no_grad()
def run_pair(ego_perc, uav_perc, frame):
    b_e, p_e = pack_dets(*ego_perc.run(frame, EGO_ID))
    b_u, p_u = pack_dets(*uav_perc.run(frame, EGO_ID))
    fe = last_bev_tensor(ego_perc)
    fu = last_bev_tensor(uav_perc)
    return b_e, b_u, fe, fu


def iter_frames(dataset, max_cases=None, frame_stride=2):
    n = int(dataset.case_number("multi_frame"))
    ids = list(range(n))
    if max_cases is not None:
        ids = ids[: int(max_cases)]
    for ci in ids:
        case = dataset.get_case(ci, tag="multi_frame", use_lidar=True)
        if not case:
            continue
        # multi_frame → list of per-frame {cav_id: data}; multi_vehicle → dict
        frames = list(case) if isinstance(case, (list, tuple)) else [case]
        step = max(1, int(frame_stride))
        for fi in range(0, len(frames), step):
            yield ci, int(fi), frames[fi]


def train_one_epoch(
    matcher,
    opt,
    ego_perc,
    uav_perc,
    dataset,
    device,
    args,
    grid,
):
    matcher.train()
    buf_e, buf_u, buf_y = [], [], []
    stats = {"n_pos": 0, "n_neg": 0, "loss_sum": 0.0, "n_step": 0, "n_frame": 0}

    def flush():
        if len(buf_y) < int(args.batch_size):
            return
        fe = torch.cat(buf_e[: args.batch_size], dim=0).to(device)
        fu = torch.cat(buf_u[: args.batch_size], dim=0).to(device)
        y = torch.tensor(buf_y[: args.batch_size], dtype=torch.float32, device=device)
        del buf_e[: args.batch_size]
        del buf_u[: args.batch_size]
        del buf_y[: args.batch_size]
        s = matcher(fe, fu)
        loss = F.binary_cross_entropy(s, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        stats["loss_sum"] += float(loss.item())
        stats["n_step"] += 1

    for ci, fi, frame in iter_frames(dataset, args.max_cases, args.frame_stride):
        if EGO_ID not in frame or UAV_ID not in frame:
            continue
        try:
            _, b_u, fe, fu = run_pair(ego_perc, uav_perc, frame)
        except Exception as e:
            print("[warn] case {} frame {} run failed: {}".format(ci, fi, e), flush=True)
            continue
        if fe is None or fu is None:
            print("[warn] missing BEV feat case {} frame {}".format(ci, fi), flush=True)
            continue
        gt, _ = build_v2u4_eval_gt(frame, EGO_ID, UAV_ID)
        ego_lidar = frame[EGO_ID].get("lidar")
        labeled = label_uav_boxes(b_u, gt, ego_lidar)
        picked = balance_frame(labeled, n_pos_cap=args.n_pos_cap)
        if not picked:
            continue
        fe = fe.float()
        fu = fu.float()
        for box, kind in picked:
            pe, pu, y = crops_for_kind(fe, fu, box, kind, args.k, grid)
            if pe.shape[0] == 0 or pu.shape[0] == 0:
                continue
            buf_e.append(pe.detach())
            buf_u.append(pu.detach())
            buf_y.append(y)
            if y >= 0.5:
                stats["n_pos"] += 1
            else:
                stats["n_neg"] += 1
        stats["n_frame"] += 1
        flush()
        if args.max_frames and stats["n_frame"] >= int(args.max_frames):
            break
    return stats


@torch.no_grad()
def eval_scores(matcher, ego_perc, uav_perc, dataset, device, args, grid, max_frames=40):
    matcher.eval()
    pos, neg = [], []
    n = 0
    for ci, fi, frame in iter_frames(dataset, args.val_max_cases, args.frame_stride):
        if EGO_ID not in frame or UAV_ID not in frame:
            continue
        try:
            _, b_u, fe, fu = run_pair(ego_perc, uav_perc, frame)
        except Exception:
            continue
        if fe is None or fu is None:
            continue
        gt, _ = build_v2u4_eval_gt(frame, EGO_ID, UAV_ID)
        ego_lidar = frame[EGO_ID].get("lidar")
        labeled = label_uav_boxes(b_u, gt, ego_lidar)
        if not labeled:
            continue
        fe = fe.float().to(device)
        fu = fu.float().to(device)
        for box, kind in labeled:
            if kind == "pos":
                pe, pu, _ = crops_for_kind(fe, fu, box, "pos", args.k, grid)
                s = float(matcher(pe.to(device), pu.to(device)).mean().item())
                pos.append(s)
            elif kind in ("hard", "fp"):
                pe, pu, _ = crops_for_kind(fe, fu, box, kind, args.k, grid)
                s = float(matcher(pe.to(device), pu.to(device)).mean().item())
                neg.append(s)
        n += 1
        if n >= int(max_frames):
            break
    return {
        "n_frames": n,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "mean_pos": float(np.mean(pos)) if pos else 0.0,
        "mean_neg": float(np.mean(neg)) if neg else 0.0,
    }


def pick_tau(mean_pos, mean_neg) -> float:
    if mean_pos <= mean_neg:
        return 0.5
    return float(np.clip(0.5 * (mean_pos + mean_neg), 0.2, 0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--in_channels", type=int, default=DEFAULT_IN_CH)
    ap.add_argument("--det_score", type=float, default=0.1)
    ap.add_argument("--max_cases", type=int, default=None, help="limit train cases")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--val_max_cases", type=int, default=8)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--n_pos_cap", type=int, default=6)
    ap.add_argument("--ego_ckpt", type=str, default=EGO_POINTPILLAR_DIR)
    ap.add_argument("--uav_ckpt", type=str, default=UAV_POINTPILLAR_DIR)
    ap.add_argument(
        "--out",
        type=str,
        default=os.path.join(_ROOT, "logs", "bev_matcher"),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="1 epoch, 4 cases")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dry_run:
        args.epochs = min(args.epochs, 1)
        args.max_cases = args.max_cases or 4
        args.max_frames = args.max_frames or 8
        args.val_max_cases = min(args.val_max_cases, 2)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, flush=True)
    print("[data] train", train_dir(), flush=True)
    print("[ckpt] ego", args.ego_ckpt, flush=True)
    print("[ckpt] uav", args.uav_ckpt, flush=True)

    train_ds = V2U4RealDataset(train_dir(), mode="train")
    val_ds = V2U4RealDataset(val_dir(), mode="val")
    print(
        "[data] train cases",
        train_ds.case_number("multi_frame"),
        "val cases",
        val_ds.case_number("multi_frame"),
        flush=True,
    )

    print("[load] frozen late PointPillars ...", flush=True)
    ego_perc = load_late(args.ego_ckpt, args.det_score)
    uav_perc = load_late(args.uav_ckpt, args.det_score)

    in_ch = int(args.in_channels)
    for ci, fi, frame in iter_frames(train_ds, 1, 1):
        try:
            _, _, fe, fu = run_pair(ego_perc, uav_perc, frame)
        except Exception as e:
            print("[warn] probe forward failed:", e, flush=True)
            break
        if fe is not None:
            t = fe[0] if torch.is_tensor(fe) and fe.dim() == 4 else fe
            in_ch = int(t.shape[0])
            print("[feat] spatial_features_2d C,H,W =", tuple(t.shape), flush=True)
        break
    if in_ch != int(args.in_channels):
        print("[feat] override in_channels {} → {}".format(args.in_channels, in_ch), flush=True)
        args.in_channels = in_ch

    matcher = BevLocalMatcher(
        in_channels=int(args.in_channels),
        hidden=int(args.hidden),
        nhead=int(args.nhead),
        k=int(args.k),
    ).to(device)
    opt = torch.optim.AdamW(matcher.parameters(), lr=float(args.lr), weight_decay=1e-4)
    grid = BevGrid()

    hist = []
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        st = train_one_epoch(
            matcher, opt, ego_perc, uav_perc, train_ds, device, args, grid
        )
        ev = eval_scores(matcher, ego_perc, uav_perc, val_ds, device, args, grid)
        tau = pick_tau(ev["mean_pos"], ev["mean_neg"])
        rec = {
            "epoch": epoch,
            "train": st,
            "val": ev,
            "tau": tau,
            "sec": round(time.time() - t0, 1),
        }
        hist.append(rec)
        print(
            "[epoch {:>2}] steps={} loss={:.4f} pos/neg={}/{} | "
            "val meanS pos={:.3f} neg={:.3f} tau~{:.2f}  ({:.1f}s)".format(
                epoch,
                st["n_step"],
                (st["loss_sum"] / max(1, st["n_step"])),
                st["n_pos"],
                st["n_neg"],
                ev["mean_pos"],
                ev["mean_neg"],
                tau,
                rec["sec"],
            ),
            flush=True,
        )
        ckpt = {
            "state_dict": matcher.state_dict(),
            "meta": {
                "in_channels": int(args.in_channels),
                "hidden": int(args.hidden),
                "nhead": int(args.nhead),
                "k": int(args.k),
                "tau": tau,
                "det_score": float(args.det_score),
                "epoch": epoch,
            },
        }
        torch.save(ckpt, os.path.join(args.out, "matcher_ep{:02d}.pth".format(epoch)))
        torch.save(ckpt, os.path.join(args.out, "matcher_last.pth"))
        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump(hist, f, indent=2)

    last = hist[-1] if hist else {}
    tau = float(last.get("tau", 0.5)) if last else 0.5
    with open(os.path.join(args.out, "tau.json"), "w") as f:
        json.dump({"tau": tau, "val": last.get("val")}, f, indent=2)
    print("[done] ckpt →", os.path.join(args.out, "matcher_last.pth"), "tau", tau, flush=True)


if __name__ == "__main__":
    main()
