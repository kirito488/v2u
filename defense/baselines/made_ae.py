"""Residual autoencoder for MADE collaborative reconstruction loss.

Official structure: made_simluation/utils/attack_detection/autoencoder.py
(DiscoNet 256ch / ~32x32). We pool Z residual to 32x32 and set in_channels=C.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MADE_AE_SPATIAL = 32
MADE_AE_LATENT = 256
MADE_AE_BASE = 32
DEFAULT_AE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "made_ae",
)


class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        return self.net(x)


class ResidualAutoencoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int = MADE_AE_BASE, latent_dim: int = MADE_AE_LATENT):
        super().__init__()
        self.num_input_channels = int(num_input_channels)
        self.encoder = Encoder(self.num_input_channels, base_channel_size, latent_dim)
        self.decoder = Decoder(self.num_input_channels, base_channel_size, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def residual_from_z(z_ego, z_fused) -> Optional[np.ndarray]:
    if z_ego is None or z_fused is None:
        return None
    a = np.asarray(z_ego, dtype=np.float32)
    b = np.asarray(z_fused, dtype=np.float32)
    if a.ndim == 3:
        a = a[None]
    if b.ndim == 3:
        b = b[None]
    if a.shape != b.shape:
        return None
    return b - a


def compact_fusion_z(fusion_z, spatial: int = MADE_AE_SPATIAL):
    """Keep Z at AE resolution so long case lists do not OOM (full Z is ~80MB/frame)."""
    if not fusion_z:
        return None
    out = {}
    for key in ("Z_ego", "Z_fused"):
        t = fusion_z.get(key)
        if t is None:
            continue
        out[key] = pool_residual(t, spatial=spatial).cpu().numpy().astype(np.float32)
    return out or None


def pool_residual(residual: np.ndarray, spatial: int = MADE_AE_SPATIAL) -> torch.Tensor:
    x = torch.from_numpy(np.array(residual, dtype=np.float32, copy=True))
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return F.adaptive_avg_pool2d(x, (spatial, spatial))


def reconstruction_loss(ae: ResidualAutoencoder, residual: np.ndarray) -> float:
    with torch.no_grad():
        x = pool_residual(residual).to(next(ae.parameters()).device)
        # Official Normalize uses data-center mean; we use per-sample std scale.
        scale = x.flatten(1).std(dim=1, keepdim=True).clamp(min=1e-3).view(-1, 1, 1, 1)
        xn = torch.tanh(x / scale)
        recon, _ = ae(xn)
        loss = F.mse_loss(xn, recon, reduction="none").sum(dim=(1, 2, 3))
        return float(loss.mean().item())


def default_ckpt_path(model_name: str, ae_dir: str = DEFAULT_AE_DIR) -> str:
    return os.path.join(ae_dir, "{}_residual_ae.pt".format(model_name))


def load_ae(model_name: str, in_channels: int, ae_dir: str = DEFAULT_AE_DIR, map_location="cpu"):
    path = default_ckpt_path(model_name, ae_dir)
    if not os.path.isfile(path):
        return None, None
    blob = torch.load(path, map_location=map_location)
    ch = int(blob.get("in_channels") or in_channels)
    ae = ResidualAutoencoder(ch)
    ae.load_state_dict(blob["state_dict"])
    ae.eval()
    th = float(blob.get("threshold") or 0.0)
    return ae, th
