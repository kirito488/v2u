"""Train MADE residual AE on dumped clean R maps."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from defense.baselines.made_ae import (  # noqa: E402
    DEFAULT_AE_DIR,
    ResidualAutoencoder,
    default_ckpt_path,
    pool_residual,
)


class ResidualNpy(Dataset):
    def __init__(self, paths, tag="data"):
        self.items = []
        n = len(paths)
        for i, p in enumerate(paths):
            t = pool_residual(np.load(p, mmap_mode="r"))
            scale = t.flatten().std().clamp(min=1e-3)
            self.items.append(torch.tanh(t.squeeze(0) / scale).contiguous().cpu())
            if (i + 1) % 40 == 0 or (i + 1) == n:
                print("[train] pooled {} {}/{}".format(tag, i + 1, n), flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attfuse")
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))

    torch.set_num_threads(4)
    data_dir = args.data or os.path.join(DEFAULT_AE_DIR, args.model)
    paths = sorted(glob.glob(os.path.join(data_dir, "r_*.npy")))
    if len(paths) < 8:
        raise SystemExit("need at least 8 residuals in {}".format(data_dir))
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(paths))
    n_cal = max(8, int(0.2 * len(paths)))
    cal_paths = [paths[i] for i in idx[:n_cal]]
    tr_paths = [paths[i] for i in idx[n_cal:]]
    sample = np.load(tr_paths[0])
    if sample.ndim == 3:
        in_ch = int(sample.shape[0])
    else:
        in_ch = int(sample.shape[1])
    print("[train] {} residuals={} train={} cal={} C={}".format(
        args.model, len(paths), len(tr_paths), len(cal_paths), in_ch), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = ResidualAutoencoder(in_ch).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    print("[train] pooling train residuals ...", flush=True)
    loader = DataLoader(
        ResidualNpy(tr_paths, tag="train"),
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    ae.train()
    for ep in range(1, args.epochs + 1):
        losses = []
        for batch in loader:
            batch = batch.to(device)
            recon, _ = ae(batch)
            loss = F.mse_loss(batch, recon)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print("[train] epoch {} mse={:.5f}".format(ep, float(np.mean(losses))), flush=True)

    ae.eval()
    cal_losses = []
    with torch.no_grad():
        print("[train] pooling cal residuals ...", flush=True)
        cal_ds = ResidualNpy(cal_paths, tag="cal")
        for i in range(len(cal_ds)):
            x = cal_ds[i].unsqueeze(0).to(device)
            recon, _ = ae(x)
            cal_losses.append(float(F.mse_loss(x, recon, reduction="none").sum().item()))
    cal_losses = np.asarray(cal_losses, dtype=np.float64)
    th = float(np.max(cal_losses) * 1.05)
    os.makedirs(DEFAULT_AE_DIR, exist_ok=True)
    ckpt = default_ckpt_path(args.model)
    torch.save(
        {
            "state_dict": ae.cpu().state_dict(),
            "in_channels": in_ch,
            "threshold": th,
            "cal_losses": cal_losses,
            "model": args.model,
        },
        ckpt,
    )
    print("[train] saved {} th={:.4f} (1.05*cal_max) cal_mean={:.4f}".format(
        ckpt, th, float(cal_losses.mean())))


if __name__ == "__main__":
    main()
