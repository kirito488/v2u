"""Resolve V2U4Real / attack-repo paths (bundled monorepo first, then absolute)."""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_BUNDLED_V2U4 = os.path.join(_REPO_ROOT, "V2U4Real")
_BUNDLED_ATTACK = os.path.join(_REPO_ROOT, "AdvCollaborativePerception")


def first_existing(candidates, require_subdir=None):
    for p in candidates:
        if not p or not os.path.isdir(p):
            continue
        if require_subdir and not os.path.isdir(os.path.join(p, require_subdir)):
            continue
        return os.path.normpath(p)
    return os.path.normpath(candidates[0])


V2U4REAL_ROOT = first_existing([
    _BUNDLED_V2U4,
    "/data/hzy/lxt/V2U4Real-main/V2U4Real-main",
    "/data/hzy/lxt/V2U4Real-main",
    "D:/agent_project/V2U4Real-main/V2U4Real-main",
    "D:/agent_project/V2U4Real-main",
])

CKPT_ROOT = first_existing([
    os.path.join(_BUNDLED_V2U4, "checkpoints"),
    "/data/hzy/lxt/V2U4Real-main/checkpoints",
    os.path.join(V2U4REAL_ROOT, "checkpoints"),
    "D:/agent_project/V2U4Real-main/checkpoints",
])

ATTACK_ROOT = first_existing(
    [
        _BUNDLED_ATTACK,
        "/data/hzy/lxt/AdvCollaborativePerception-master",
        "D:/agent_project/AdvCollaborativePerception-master",
    ],
    require_subdir=os.path.join("mvp", "data"),
)

MODEL_CKPTS = {
    "attfuse": os.path.join(CKPT_ROOT, "attfuse_checkpoint"),
    "where2comm": os.path.join(CKPT_ROOT, "where2comm_checkpoint"),
    "coalign": os.path.join(CKPT_ROOT, "coalign_checkpoint"),
}

# Single-agent late PointPillars (vehicle / UAV), trained with only_cav_id 1 / 2.
_OPENCOOD_LOGS = first_existing([
    os.path.join(_BUNDLED_V2U4, "opencood", "logs"),
    "/data/hzy/lxt/V2U4Real-main/opencood/logs",
    os.path.join(V2U4REAL_ROOT, "opencood", "logs"),
    os.path.join(os.path.dirname(V2U4REAL_ROOT), "opencood", "logs"),
    "D:/agent_project/V2U4Real-main/V2U4Real-main/opencood/logs",
])
EGO_POINTPILLAR_DIR = os.path.join(
    _OPENCOOD_LOGS, "point_pillar_vehicle_only_2026_08_26_17_26_19"
)
UAV_POINTPILLAR_DIR = os.path.join(
    _OPENCOOD_LOGS, "point_pillar_uav_only_2026_08_27_00_38_53"
)

EGO_ID = "1"
UAV_ID = "2"


def setup_import_paths():
    """Put attack repo + V2U4Real OpenCOOD on sys.path."""
    for p in (ATTACK_ROOT, V2U4REAL_ROOT):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def val_dir():
    for p in (
        os.path.join(V2U4REAL_ROOT, "v2u4real", "val"),
        os.path.join(os.path.dirname(V2U4REAL_ROOT), "v2u4real", "val"),
        "/data/hzy/lxt/V2U4Real-main/v2u4real/val",
        "D:/agent_project/V2U4Real-main/v2u4real/val",
    ):
        if os.path.isdir(p):
            return os.path.normpath(p)
    return os.path.normpath(os.path.join(V2U4REAL_ROOT, "v2u4real", "val"))


def train_dir():
    for p in (
        os.path.join(V2U4REAL_ROOT, "v2u4real", "train"),
        os.path.join(os.path.dirname(V2U4REAL_ROOT), "v2u4real", "train"),
        "/data/hzy/lxt/V2U4Real-main/v2u4real/train",
        "D:/agent_project/V2U4Real-main/v2u4real/train",
    ):
        if os.path.isdir(p):
            return os.path.normpath(p)
    return os.path.normpath(os.path.join(V2U4REAL_ROOT, "v2u4real", "train"))
