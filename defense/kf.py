"""Constant-velocity KF on BEV center (x, y, vx, vy). Shared by pool and buffer."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

DT = 1.0
_Q_DIAG = np.array([0.05, 0.05, 0.2, 0.2], dtype=np.float64)
_R_DIAG = np.array([0.25, 0.25], dtype=np.float64)
_P0_DIAG = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

F = np.array(
    [
        [1.0, 0.0, DT, 0.0],
        [0.0, 1.0, 0.0, DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
Q = np.diag(_Q_DIAG)
R = np.diag(_R_DIAG)
I4 = np.eye(4, dtype=np.float64)
P0 = np.diag(_P0_DIAG.copy())


def velocity_from_traj(
    traj: Optional[Sequence] = None,
    dt: float = DT,
) -> Tuple[float, float]:
    """Estimate (vx, vy) from a backfill / watch trajectory.

    `traj` entries are (frame_id, box) or bare boxes. Uses mean of consecutive
    frame-to-frame deltas (Δx/Δt, Δy/Δt). Needs ≥2 points; else (0, 0).
    """
    if not traj:
        return 0.0, 0.0
    pts = []
    for i, item in enumerate(traj):
        if item is None:
            continue
        fid = None
        box = item
        if isinstance(item, (tuple, list)) and len(item) == 2:
            a, b = item[0], item[1]
            if np.isscalar(a) and b is not None and np.asarray(b).size >= 2:
                fid = int(a)
                box = b
        b = np.asarray(box, dtype=np.float64).reshape(-1)
        if b.size < 2:
            continue
        if fid is None:
            fid = int(i)
        pts.append((int(fid), float(b[0]), float(b[1])))
    if len(pts) < 2:
        return 0.0, 0.0
    vxs, vys = [], []
    for i in range(1, len(pts)):
        df = float(pts[i][0] - pts[i - 1][0])
        if df <= 0:
            continue
        t_span = df * float(dt)
        vxs.append((pts[i][1] - pts[i - 1][1]) / t_span)
        vys.append((pts[i][2] - pts[i - 1][2]) / t_span)
    if not vxs:
        return 0.0, 0.0
    return float(np.mean(vxs)), float(np.mean(vys))


def birth_state(
    box,
    vx: float = 0.0,
    vy: float = 0.0,
    traj: Optional[Sequence] = None,
) -> tuple:
    """Initial KF state at pool/buffer birth.

    If `traj` is given (backfill / watch history), (vx, vy) are estimated from
    it and override the explicit vx/vy defaults.
    """
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if traj:
        vx, vy = velocity_from_traj(traj, dt=DT)
    x = np.array(
        [float(b[0]), float(b[1]), float(vx), float(vy)],
        dtype=np.float64,
    )
    return x, P0.copy()


def kf_predict(x: np.ndarray, P: np.ndarray):
    xn = F @ x
    Pn = F @ P @ F.T + Q
    Pn = 0.5 * (Pn + Pn.T)
    return xn, Pn


def kf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray):
    y = z - H @ x
    s = H @ P @ H.T + R
    k = P @ H.T @ np.linalg.inv(s)
    xn = x + k @ y
    pn = (I4 - k @ H) @ P
    pn = 0.5 * (pn + pn.T)
    return xn, pn
