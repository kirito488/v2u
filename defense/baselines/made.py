"""MADE (IROS 2024 / arXiv:2310.11901) on dual-agent V2U4Real.

Match loss: HungarianMatcher on Y_ego vs Y_ego+UAV (Init).
Reconstruction: residual AE on R = Z_fused - Z_ego when a trained checkpoint
exists (see made_ae.py). Official DiscoNet weights are not reused.

Dual-agent: reject UAV (output Ego) if either statistic exceeds its threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..geometry import iou_bev
from .common import BaselineResult, as_boxes7, as_scores, source_boxes_scores
from .made_ae import (
    DEFAULT_AE_DIR,
    load_ae,
    reconstruction_loss,
    residual_from_z,
)

MADE_MATCH_TH = 0.83
MADE_PHI = 1.0

_AE_CACHE = {}


def match_loss(
    ego_boxes,
    ego_scores,
    fuse_boxes,
    fuse_scores,
    phi: float = MADE_PHI,
) -> float:
    ego_b = as_boxes7(ego_boxes)
    fuse_b = as_boxes7(fuse_boxes)
    n = int(ego_b.shape[0])
    m = int(fuse_b.shape[0])
    if n == 0:
        return 0.0
    ego_s = as_scores(ego_scores, n)
    fuse_s = as_scores(fuse_scores, m)

    dim = max(m, n)
    score_cost = np.zeros((dim, dim), dtype=np.float64)
    box_cost = np.zeros((dim, dim), dtype=np.float64)
    if m < n:
        score_cost[m:, :] = ego_s[np.newaxis, :]
        box_cost[m:, :] = 1.0

    for i in range(m):
        for j in range(n):
            score_cost[i, j] = max(0.0, float(ego_s[j]) - float(fuse_s[i]))
            box_cost[i, j] = 1.0 - float(iou_bev(ego_b[j], fuse_b[i]))

    cost = float(phi) * box_cost + score_cost
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum() / float(n))


def _ae_recon(fusion_z, model_name: str, ae_dir: str):
    if not fusion_z:
        return None, None
    residual = residual_from_z(fusion_z.get("Z_ego"), fusion_z.get("Z_fused"))
    if residual is None:
        return None, None
    in_ch = int(residual.shape[1] if residual.ndim == 4 else residual.shape[0])
    key = (model_name, ae_dir, in_ch)
    if key not in _AE_CACHE:
        _AE_CACHE[key] = load_ae(model_name, in_ch, ae_dir=ae_dir)
    ae, th = _AE_CACHE[key]
    if ae is None:
        return None, th
    return reconstruction_loss(ae, residual), th


def defend_made(
    ego,
    uav,
    init,
    made_th: float = MADE_MATCH_TH,
    phi: float = MADE_PHI,
    fusion_z=None,
    made_model: str = "attfuse",
    made_ae_dir: str = DEFAULT_AE_DIR,
    **_kwargs,
) -> BaselineResult:
    ego_b, ego_s = source_boxes_scores(ego)
    init_b, init_s = source_boxes_scores(init)
    loss = match_loss(ego_b, ego_s, init_b, init_s, phi=phi)
    recon, recon_th = _ae_recon(fusion_z, made_model, made_ae_dir)
    match_hit = loss > float(made_th)
    recon_hit = recon is not None and recon_th is not None and recon > float(recon_th)
    malicious = bool(match_hit or recon_hit)
    if malicious:
        boxes, scores, src = ego_b, ego_s, "ego"
    else:
        boxes, scores, src = init_b, init_s, "init"
    return BaselineResult(
        boxes=boxes,
        scores=scores,
        info={
            "name": "made",
            "match_loss": loss,
            "made_th": float(made_th),
            "phi": float(phi),
            "recon_loss": recon,
            "recon_th": recon_th,
            "malicious": bool(malicious),
            "match_hit": bool(match_hit),
            "recon_hit": bool(recon_hit),
            "output": src,
            "branch": "match+ae" if recon is not None else "match_cost",
        },
    )
