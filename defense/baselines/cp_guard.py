"""CP-Guard / PASAC (AAAI 2025 Oral) on dual-agent 3D detection.

Official CP-Guard CCLoss is on BEV segmentation maps:
  CCLoss = sum(ego * fused) / sum(ego + fused), accept if > --th (0.08).

Here: rasterize Ego / Init boxes to BEV occupancy (detection analog of
ego-only vs fused-with-UAV). One collaborator → PASAC is a single test:
accept UAV (output Init) iff CCLoss > th, else Ego-only.
"""
from __future__ import annotations

from .common import (
    BaselineResult,
    occupancy_ccloss,
    raster_occupancy,
    source_boxes_scores,
)

# Official default --th in CP-Guard cp_guard.py.
CP_GUARD_TH = 0.08


def defend_cp_guard(
    ego,
    uav,
    init,
    th: float = CP_GUARD_TH,
    **_kwargs,
) -> BaselineResult:
    ego_b, ego_s = source_boxes_scores(ego)
    init_b, init_s = source_boxes_scores(init)
    occ_ego = raster_occupancy(ego_b)
    occ_init = raster_occupancy(init_b)
    ccloss = occupancy_ccloss(occ_ego, occ_init)
    accept_uav = ccloss > float(th)
    if accept_uav:
        boxes, scores, src = init_b, init_s, "init"
    else:
        boxes, scores, src = ego_b, ego_s, "ego"
    return BaselineResult(
        boxes=boxes,
        scores=scores,
        info={
            "name": "cp_guard",
            "ccloss": ccloss,
            "th": float(th),
            "accept_uav": bool(accept_uav),
            "output": src,
        },
    )
