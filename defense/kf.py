"""Constant-velocity KF on BEV center (x, y, vx, vy). Shared by pool and buffer."""
from __future__ import annotations

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


def birth_state(box) -> tuple:
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    x = np.array([float(b[0]), float(b[1]), 0.0, 0.0], dtype=np.float64)
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
