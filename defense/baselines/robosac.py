"""ROBOSAC (ICCV 2023) on dual-agent V2U4Real.

Official code (coperception/tools/det/robosac.py): sample a collaborator subset,
compare fused detections vs ego-only with set Jaccard; if Jaccard < 0.3 fall
back to ego-only.

With one collaborator (UAV) this is a single test: Init vs Ego.
Consensus → Init; dissensus → Ego.
"""
from __future__ import annotations

from .common import (
    ROBOSAC_JAC_THRES,
    ROBOSAC_PAIR_IOU,
    BaselineResult,
    set_jaccard,
    source_boxes_scores,
)


def defend_robosac(
    ego,
    uav,
    init,
    jac_thres: float = ROBOSAC_JAC_THRES,
    pair_iou: float = ROBOSAC_PAIR_IOU,
    **_kwargs,
) -> BaselineResult:
    ego_b, ego_s = source_boxes_scores(ego)
    init_b, init_s = source_boxes_scores(init)
    jac = set_jaccard(ego_b, init_b, pair_iou=pair_iou)
    consensus = jac >= float(jac_thres)
    if consensus:
        boxes, scores, src = init_b, init_s, "init"
    else:
        boxes, scores, src = ego_b, ego_s, "ego"
    return BaselineResult(
        boxes=boxes,
        scores=scores,
        info={
            "name": "robosac",
            "jaccard": jac,
            "jac_thres": float(jac_thres),
            "pair_iou": float(pair_iou),
            "consensus": bool(consensus),
            "output": src,
        },
    )
