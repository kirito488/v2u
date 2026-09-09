"""UAV-UGV collaborative perception defense (three-source first)."""

from .three_source import ThreeSourceDetector, FrameThreeSource
from .gating import (
    Q_H,
    Q_L,
    THETA_P,
    THETA_SOFT,
    gate_frame,
    FrameGating,
    spatial_state,
    resolve_ambiguous,
    dual_consensus,
    detection_quality,
)
from .buffer import TentativeBuffer, FrameBuffer
from .bev_matcher import BevLocalMatcher

__all__ = [
    "ThreeSourceDetector",
    "FrameThreeSource",
    "THETA_P",
    "THETA_SOFT",
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
    "BevLocalMatcher",
]
