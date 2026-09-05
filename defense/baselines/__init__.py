"""Algorithm-level baseline defenses on the three-source (Ego / UAV / Init) outputs."""
from __future__ import annotations

from typing import Optional

from .cad import CAD_THRES, defend_cad
from .common import BaselineResult, empty_result, source_boxes_scores
from .cp_guard import CP_GUARD_TH, defend_cp_guard
from .made import MADE_MATCH_TH, defend_made
from .made_ae import DEFAULT_AE_DIR
from .robosac import defend_robosac

DEFENSE_NAMES = ("ours", "none", "robosac", "cad", "cp_guard", "made")


def defend_none(ego, uav, init, **_kwargs) -> BaselineResult:
    boxes, scores = source_boxes_scores(init)
    return BaselineResult(boxes=boxes, scores=scores, info={"name": "none", "output": "init"})


def apply_baseline(
    name: str,
    ego,
    uav,
    init,
    ego_lidar=None,
    jac_thres: float = 0.3,
    cp_guard_th: float = CP_GUARD_TH,
    cad_thres: float = CAD_THRES,
    made_th: float = MADE_MATCH_TH,
    fusion_z=None,
    made_model: str = "attfuse",
    made_ae_dir: str = DEFAULT_AE_DIR,
) -> Optional[BaselineResult]:
    """Return boxes for non-ours defenses. ``ours`` returns None (use pool+buffer)."""
    name = str(name or "ours")
    if name in ("ours", ""):
        return None
    if name == "none":
        return defend_none(ego, uav, init)
    if name == "robosac":
        return defend_robosac(ego, uav, init, jac_thres=jac_thres)
    if name == "cad":
        return defend_cad(ego, uav, init, ego_lidar=ego_lidar, thres=cad_thres)
    if name == "cp_guard":
        return defend_cp_guard(ego, uav, init, th=cp_guard_th)
    if name == "made":
        return defend_made(
            ego, uav, init,
            made_th=made_th,
            fusion_z=fusion_z,
            made_model=made_model,
            made_ae_dir=made_ae_dir,
        )
    raise ValueError("unknown defense {!r}, choose {}".format(name, DEFENSE_NAMES))


__all__ = [
    "DEFENSE_NAMES",
    "BaselineResult",
    "apply_baseline",
    "defend_none",
    "defend_robosac",
    "defend_cad",
    "defend_cp_guard",
    "defend_made",
    "empty_result",
]
