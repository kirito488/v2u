"""UAV-UGV collaborative perception defense (three-source first)."""

from .three_source import ThreeSourceDetector, FrameThreeSource
from .gating import (
    Q_H,
    Q_L,
    THETA_P,
    gate_frame,
    FrameGating,
    spatial_state,
    resolve_ambiguous,
    dual_consensus,
    detection_quality,
)
from .buffer import TentativeBuffer, FrameBuffer

__all__ = [
    "ThreeSourceDetector",
    "FrameThreeSource",
    "THETA_P",
    "Q_H",
    "Q_L",
    "gate_frame",
    "FrameGating",
    "spatial_state",
    "resolve_ambiguous",
    "dual_consensus",
    "detection_quality",
    "TentativeBuffer",
    "FrameBuffer",
]
