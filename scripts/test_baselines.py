"""CPU-only checks for baseline Jaccard / CCLoss / CAD filter."""
from __future__ import annotations

import numpy as np

from defense.paths import setup_import_paths

setup_import_paths()

from defense.baselines.common import occupancy_ccloss, raster_occupancy, set_jaccard
from defense.baselines.robosac import defend_robosac
from defense.baselines.cp_guard import defend_cp_guard
from defense.baselines.cad import defend_cad
from defense.baselines.made import defend_made, match_loss
from defense.baselines import apply_baseline
from defense.three_source import SourceResult


def _src(name, boxes, scores=None):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2:
        boxes = np.zeros((0, 7), dtype=np.float64)
    n = int(boxes.shape[0])
    scores = np.ones((n,), dtype=np.float64) if scores is None else np.asarray(scores, dtype=np.float64)
    return SourceResult(name, boxes, scores, np.ones((n,), dtype=np.float64), np.zeros((n,), dtype=np.int32))


def test_jaccard_identical():
    b = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    assert set_jaccard(b, b) == 1.0


def test_jaccard_disjoint():
    a = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    b = np.array([[40.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    assert set_jaccard(a, b) == 0.0


def test_robosac_dissensus_falls_back_to_ego():
    ego = _src("ego", [[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    uav = _src("uav", [[80.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    init = _src("init", [[80.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    out = defend_robosac(ego, uav, init)
    assert out.info["consensus"] is False
    assert out.info["output"] == "ego"
    assert abs(out.boxes[0, 0] - 10.0) < 1e-6


def test_robosac_consensus_keeps_init():
    box = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    ego = _src("ego", [box])
    init = _src("init", [box])
    out = defend_robosac(ego, ego, init)
    assert out.info["consensus"] is True
    assert out.info["output"] == "init"


def test_cp_guard_formula():
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[1.0, 1.0], [0.0, 0.0]])
    # sum(A*B)=1, sum(A+B)=3 → 1/3
    assert abs(occupancy_ccloss(a, b) - 1.0 / 3.0) < 1e-9


def test_cad_drops_box_in_empty_space():
    # Ground plane + elevated cluster at x=10; ghost off-axis in open free space.
    ego = _src("ego", [[10.0, 0.0, 0.0, 4.5, 1.8, 1.5, 0.0]])
    ghost = [15.0, 25.0, 0.0, 4.5, 1.8, 1.5, 0.0]
    init = _src("init", [[10.0, 0.0, 0.0, 4.5, 1.8, 1.5, 0.0], ghost])
    gx, gy = np.meshgrid(np.linspace(-20.0, 20.0, 40), np.linspace(-20.0, 20.0, 40))
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    xs, ys = np.meshgrid(np.linspace(8.0, 12.0, 40), np.linspace(-0.6, 0.6, 12))
    obj = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size) * 1.0], axis=1)
    lidar = np.vstack([ground, obj])
    # Official get_free_space needs Velodyne rings; use polar fallback here.
    out = defend_cad(ego, ego, init, ego_lidar=lidar, thres=1.7, occupancy_backend="fallback")
    assert out.n >= 1
    ys_keep = out.boxes[:, 1]
    assert not np.any(np.abs(ys_keep - 25.0) < 1.0), out.boxes


def test_apply_none():
    init = _src("init", [[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    out = apply_baseline("none", init, init, init)
    assert out.info["output"] == "init"
    assert apply_baseline("ours", init, init, init) is None


def test_made_identical_keeps_init():
    box = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    ego = _src("ego", [box], scores=[0.9])
    init = _src("init", [box], scores=[0.9])
    loss = match_loss(ego.boxes, ego.scores, init.boxes, init.scores)
    assert loss < 1e-6
    out = defend_made(ego, ego, init)
    assert out.info["malicious"] is False
    assert out.info["output"] == "init"


def test_made_remove_falls_back_to_ego():
    # Ego sees a real car; fused Init lost it (remove). Official pad: p_ego + φ.
    ego = _src("ego", [[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], scores=[0.9])
    init = _src("init", [])
    out = defend_made(ego, ego, init)
    assert out.info["match_loss"] > 0.83
    assert out.info["malicious"] is True
    assert out.info["output"] == "ego"
    assert abs(out.boxes[0, 0] - 10.0) < 1e-6


def test_apply_made():
    box = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    src = _src("ego", [box])
    out = apply_baseline("made", src, src, src)
    assert out.info["name"] == "made"
    assert out.info["output"] == "init"


def test_made_residual_shape():
    from defense.baselines.made_ae import residual_from_z
    a = np.ones((1, 4, 8, 8), dtype=np.float32)
    b = np.ones((1, 4, 8, 8), dtype=np.float32) * 2
    r = residual_from_z(a, b)
    assert r is not None and r.shape == a.shape
    assert abs(float(r.mean()) - 1.0) < 1e-6


if __name__ == "__main__":
    test_jaccard_identical()
    test_jaccard_disjoint()
    test_robosac_dissensus_falls_back_to_ego()
    test_robosac_consensus_keeps_init()
    test_cp_guard_formula()
    test_cad_drops_box_in_empty_space()
    test_apply_none()
    test_made_identical_keeps_init()
    test_made_remove_falls_back_to_ego()
    test_apply_made()
    test_made_residual_shape()
    print("ok", raster_occupancy(np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])).sum())
