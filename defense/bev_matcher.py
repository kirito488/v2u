"""Local BEV consensus: UAV crop (query) vs Ego crop (key/value) → S∈[0,1].

Architecture
  1) shared 1×1 conv → 64 ch
  2) UAV tokens attend to Ego tokens (light MHA)
  3) attended map → conv + GAP → sigmoid

Inference: non-Certain gate objects that have UAV and/or Init detection can be
promoted when local Ego↔UAV BEV agrees at the object's box:
  - UAV-backed: need C_uav≥c_min
  - Init-backed (incl. Init-only leftovers): need C_init≥c_min
Crops always come from Ego BEV and UAV BEV at that (x,y), even if the box
itself came from Init. Disagreement leaves gating unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# V2U4Real PointPillar yaml: cav_lidar_range / voxel / feature_stride
DEFAULT_X_MIN = -100.8
DEFAULT_Y_MIN = -80.0
DEFAULT_VOXEL = 0.4
DEFAULT_STRIDE = 2
DEFAULT_IN_CH = 384  # 128*3 BaseBEVBackbone concat
DEFAULT_K = 7
DEFAULT_TAU = 0.5
DEFAULT_C_MIN = 0.2


@dataclass
class BevGrid:
    x_min: float = DEFAULT_X_MIN
    y_min: float = DEFAULT_Y_MIN
    voxel: float = DEFAULT_VOXEL
    stride: int = DEFAULT_STRIDE

    @property
    def cell(self) -> float:
        return float(self.voxel) * float(self.stride)

    def xy_to_uv(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        u = int(np.floor((float(x) - float(self.x_min)) / self.cell))
        v = int(np.floor((float(y) - float(self.y_min)) / self.cell))
        u = int(np.clip(u, 0, max(0, int(width) - 1)))
        v = int(np.clip(v, 0, max(0, int(height) - 1)))
        return u, v


class BevLocalMatcher(nn.Module):
    """UAV query × Ego key/value local cross-attention matcher."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CH,
        hidden: int = 64,
        nhead: int = 4,
        k: int = DEFAULT_K,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.k = int(k)
        self.proj = nn.Conv2d(self.in_channels, self.hidden, kernel_size=1, bias=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden, num_heads=int(nhead), batch_first=True
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, feat_ego: torch.Tensor, feat_uav: torch.Tensor) -> torch.Tensor:
        """feat_*: (B, C, k, k) → S (B,) in (0,1)."""
        q = self.proj(feat_uav)
        kv = self.proj(feat_ego)
        b, c, h, w = q.shape
        q_tok = q.flatten(2).transpose(1, 2)
        kv_tok = kv.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(q_tok, kv_tok, kv_tok, need_weights=False)
        corr = attn_out.transpose(1, 2).reshape(b, c, h, w)
        logit = self.head(corr).reshape(b)
        return torch.sigmoid(logit)


def _as_chw(feat) -> torch.Tensor:
    if feat is None:
        raise ValueError("missing BEV feature")
    if not torch.is_tensor(feat):
        feat = torch.as_tensor(feat)
    if feat.dim() == 4:
        feat = feat[0]
    if feat.dim() != 3:
        raise ValueError("BEV feature must be (C,H,W) or (1,C,H,W), got {}".format(tuple(feat.shape)))
    return feat


def crop_patches(
    feat,
    boxes,
    k: int = DEFAULT_K,
    grid: Optional[BevGrid] = None,
) -> torch.Tensor:
    """Crop k×k patches at box xy. feat (C,H,W) or (1,C,H,W). boxes (N,7).

    Returns (N, C, k, k) on the same device/dtype as feat. Empty → (0,C,k,k).
    """
    grid = grid or BevGrid()
    f = _as_chw(feat)
    c, h, w = int(f.shape[0]), int(f.shape[1]), int(f.shape[2])
    boxes = np.asarray(boxes) if not torch.is_tensor(boxes) else boxes.detach().cpu().numpy()
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return f.new_zeros((0, c, int(k), int(k)))
    pad = int(k) // 2
    padded = F.pad(f, (pad, pad, pad, pad))
    rows = []
    for b in boxes:
        u, v = grid.xy_to_uv(float(b[0]), float(b[1]), w, h)
        patch = padded[:, v : v + int(k), u : u + int(k)]
        if patch.shape[-2] != int(k) or patch.shape[-1] != int(k):
            patch = f.new_zeros((c, int(k), int(k)))
        rows.append(patch)
    return torch.stack(rows, dim=0)


def last_bev_tensor(perception) -> Optional[torch.Tensor]:
    """GPU tensor (1,C,H,W) or (C,H,W) from OpencoodPerception / PointPillar."""
    if perception is None:
        return None
    model = getattr(perception, "model", None)
    z = getattr(model, "_defense_z", None) if model is not None else None
    if isinstance(z, dict):
        for key in ("spatial_features_2d", "Z_ego", "Z_fused"):
            t = z.get(key)
            if torch.is_tensor(t) and t.numel() > 0:
                return t
    packed = getattr(perception, "last_fusion_z", None)
    if isinstance(packed, dict):
        for key in ("spatial_features_2d", "Z_ego", "Z_fused"):
            t = packed.get(key)
            if t is None:
                continue
            if torch.is_tensor(t):
                return t
            return torch.as_tensor(t)
    return None


@torch.no_grad()
def score_boxes(
    matcher: BevLocalMatcher,
    feat_ego,
    feat_uav,
    boxes,
    k: Optional[int] = None,
    grid: Optional[BevGrid] = None,
) -> np.ndarray:
    """S for each box. Empty boxes → shape (0,)."""
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    k = int(matcher.k if k is None else k)
    device = next(matcher.parameters()).device
    fe = _as_chw(feat_ego).to(device=device, dtype=torch.float32)
    fu = _as_chw(feat_uav).to(device=device, dtype=torch.float32)
    pe = crop_patches(fe, boxes, k=k, grid=grid)
    pu = crop_patches(fu, boxes, k=k, grid=grid)
    matcher.eval()
    s = matcher(pe, pu).detach().float().cpu().numpy()
    return np.asarray(s, dtype=np.float64).reshape(-1)


def promote_uav_bev_certain(
    gating: Any,
    feat_ego,
    feat_uav,
    matcher: BevLocalMatcher,
    *,
    tau: float = DEFAULT_TAU,
    c_min: float = DEFAULT_C_MIN,
    grid: Optional[BevGrid] = None,
) -> int:
    """Promote UAV/Init-backed non-Certain objects when local Ego↔UAV BEV agrees.

    Candidates (any of):
      - D_uav=1 and C_uav≥c_min
      - D_init=1 and C_init≥c_min
    Box may be Ego/UAV/Init geometry; crops are always taken from Ego & UAV
    BEV feature maps at that box center. Returns # newly promoted.
    """
    from .gating import CERTAIN

    if gating is None or matcher is None or feat_ego is None or feat_uav is None:
        return 0
    objs: Sequence[Any] = getattr(gating, "objects", None) or []
    cand_i: List[int] = []
    boxes: List[np.ndarray] = []
    c_thr = float(c_min)
    for i, o in enumerate(objs):
        if getattr(o, "state", None) == CERTAIN:
            continue
        if o.box is None:
            continue
        d_uav = int(getattr(o, "d_uav", 0) or 0) == 1
        d_init = int(getattr(o, "d_init", 0) or 0) == 1
        if not d_uav and not d_init:
            continue
        ok_uav = d_uav and float(getattr(o, "c_uav", 0.0) or 0.0) >= c_thr
        ok_init = d_init and float(getattr(o, "c_init", 0.0) or 0.0) >= c_thr
        if not ok_uav and not ok_init:
            continue
        cand_i.append(i)
        boxes.append(np.asarray(o.box, dtype=np.float64)[:7])
    if not cand_i:
        return 0
    scores = score_boxes(matcher, feat_ego, feat_uav, np.stack(boxes), grid=grid)
    n = 0
    for j, i in enumerate(cand_i):
        s = float(scores[j]) if j < len(scores) else 0.0
        objs[i].bev_s = s
        if s >= float(tau):
            objs[i].state = CERTAIN
            objs[i].from_bev = True
            n += 1
    return n


# Alias: name kept for call sites; now covers Init-backed boxes too.
promote_bev_certain = promote_uav_bev_certain


def load_matcher(
    path: str,
    device: Optional[torch.device] = None,
    in_channels: Optional[int] = None,
) -> Tuple[BevLocalMatcher, dict]:
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    ch = int(in_channels or meta.get("in_channels") or DEFAULT_IN_CH)
    hidden = int(meta.get("hidden") or 64)
    nhead = int(meta.get("nhead") or 4)
    k = int(meta.get("k") or DEFAULT_K)
    net = BevLocalMatcher(in_channels=ch, hidden=hidden, nhead=nhead, k=k)
    net.load_state_dict(state, strict=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    return net, meta
