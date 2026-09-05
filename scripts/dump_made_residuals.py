"""Dump clean R = Z_fused - Z_ego for MADE AE training (no attack)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.paths import (  # noqa: E402
    EGO_ID,
    MODEL_CKPTS,
    V2U4REAL_ROOT,
    setup_import_paths,
    train_dir,
    val_dir,
)

setup_import_paths()

from mvp.data.v2u4real_dataset import V2U4RealDataset  # noqa: E402

from defense.baselines.made_ae import DEFAULT_AE_DIR, residual_from_z  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attfuse", choices=list(MODEL_CKPTS))
    ap.add_argument("--n_cases", type=int, default=20)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))

    from mvp.perception.opencood_perception import OpencoodPerception

    perc = OpencoodPerception(
        fusion_method="intermediate",
        model_name=args.model,
        model_dir=MODEL_CKPTS[args.model],
        opencood_root=V2U4REAL_ROOT,
        root_dir=train_dir(),
        validate_dir=val_dir(),
    )
    out_dir = args.out or os.path.join(DEFAULT_AE_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)
    print("[dump] model={} out={}".format(args.model, out_dir), flush=True)
    dataset = V2U4RealDataset(root_path=val_dir(), mode="val")
    n_total = dataset.case_number("multi_frame")
    saved = 0
    for ci in range(args.case, min(n_total, args.case + args.n_cases)):
        case = dataset.get_case(ci, "multi_frame")
        frames = case if isinstance(case, (list, tuple)) else [
            case[k] for k in sorted(case.keys())
        ]
        for fi, frame in enumerate(frames):
            try:
                perc.run(frame, EGO_ID)
            except Exception as exc:
                print("[dump] skip case {} frame {}: {}".format(ci, fi, exc))
                continue
            z = getattr(perc, "last_fusion_z", None) or {}
            ze, zf = z.get("Z_ego"), z.get("Z_fused")
            r = residual_from_z(ze, zf)
            if r is None:
                print("[dump] no R case {} frame {} Z_ego={} Z_fused={}".format(
                    ci, fi,
                    None if ze is None else tuple(np.asarray(ze).shape),
                    None if zf is None else tuple(np.asarray(zf).shape),
                ), flush=True)
                continue
            if saved == 0:
                print("[dump] first R shape={} mean={:.4f}".format(
                    tuple(r.shape), float(r.mean())), flush=True)
            path = os.path.join(out_dir, "r_{:03d}_{:03d}.npy".format(ci, int(fi)))
            np.save(path, r.astype(np.float32))
            saved += 1
        print("[dump] case {} saved_total={}".format(ci, saved), flush=True)
    print("[dump] done n={}".format(saved))


if __name__ == "__main__":
    main()
