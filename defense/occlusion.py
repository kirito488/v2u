"""Visibility / occlusion helpers.

Two backends:
  1) Box-ray (legacy): BEV rays vs other detection boxes as occluders.
  2) LiDAR polar depth: real point-cloud occlusion via a min-range map in
     (azimuth, elevation). Works for both ego (sideways) and UAV (top-down)
     because it is sensor-frame depth, not BEV box blocking.

v = free_sample_rays / M. Out of lidar range → v = 0 (ego-range check only
when requested).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .metrics import lidar_range_mask

# Visibility threshold: fraction of free sample rays (with 75-ray grid,
# 0.25 ≈ ≥19/75 free). Same ratio as the old ≥2/8 corner rule.
V_MIN = 0.25
K_ATK = 10
# Range slack when comparing polar returns to near-face cutoff (metres).
# Ouster range resolution is cm-scale; 0.3 m absorbs calibration / motion blur.
_HIT_EPS = 0.3

# ---------------------------------------------------------------------------
# Polar depth map defaults (sensor frame).
# Previously AZ=0.5° / EL=1.0° / 5×5×3 grid were engineering guesses.
# Values below follow published spherical occlusion discretisation + V2U4Real
# sensor (config: lidar_type='ouster').
#
# Angular bins — BtcDet (Xu et al., AAAI 2022 / arXiv:2112.02205) uses
# spherical voxels for LiDAR occlusion; KITTI setting (φ, θ) = (0.52°, 0.42°).
# (WOD setting 0.81°/0.31° is coarser in azimuth; we take KITTI as default.)
# Closest Ouster OS1 physical spacings for reference:
#   az ≈ 360°/1024 ≈ 0.35° (common 1024 mode), el ≈ 42.4°/63 ≈ 0.67° (64 ch).
# ---------------------------------------------------------------------------
AZ_RES_DEG = 0.52
EL_RES_DEG = 0.42
# Fallback self-clear distance when OBB entry fails (metres).
# Ouster OS1 default minimum range is 0.5 m (datasheet); half-BEV-diagonal
# still applied in `_clear_dist_for_box` so large vehicles get a larger margin.
CLEAR_DIST_M = 0.5
# Neighbour bins (±nb) to reduce polar discretisation holes (same spirit as
# OctoMap / WYSIWYG raycasting aggregation robustness).
POLAR_NB = 1
# Query-box ray samples. Default = dense OBB grid 5×5×3 = 75 rays
# (replaces 8-corner probes for smoother partial-occlusion V).
# Set SAMPLE_MODE="corners8" for the classic 8-corner ablation.
SAMPLE_MODE = "grid"  # "corners8" | "grid"
SAMPLE_L = 5
SAMPLE_W = 5
SAMPLE_H = 3  # 5*5*3 = 75


def _as_boxes(boxes) -> np.ndarray:
    a = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if a.ndim == 1 and a.size >= 7:
        return np.asarray(a, dtype=np.float64).reshape(1, -1)[:, :7]
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)[:, :7]


def _sample_points(box) -> np.ndarray:
    """Dense BEV samples on the box footprint (xy only, legacy box-ray)."""
    return _sample_points_3d(box)[:, :2]


def _sample_points_3d(
    box,
    n_l: int = SAMPLE_L,
    n_w: int = SAMPLE_W,
    n_h: int = SAMPLE_H,
    mode: Optional[str] = None,
) -> np.ndarray:
    """Sample points on/in the OBB (sensor / ego frame).

    Default ``corners8``: the 8 geometric corners (standard bbox-corner
    visibility probes). ``grid``: uniform L×W×H lattice (ablation only).
    """
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    x, y, z = float(b[0]), float(b[1]), float(b[2])
    l, w, h = float(b[3]), float(b[4]), float(b[5])
    yaw = float(b[6]) if b.size > 6 else 0.0
    use = (mode or SAMPLE_MODE or "corners8").lower()

    if use in ("corners8", "corners", "8"):
        # Local corners of the OBB (±l/2, ±w/2, ±h/2).
        signs = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        local = signs * np.array([l, w, h], dtype=np.float64)
    else:
        n_l = max(int(n_l), 1)
        n_w = max(int(n_w), 1)
        n_h = max(int(n_h), 1)
        xs = np.linspace(-0.5 * l, 0.5 * l, n_l, dtype=np.float64)
        ys = np.linspace(-0.5 * w, 0.5 * w, n_w, dtype=np.float64)
        zs = np.linspace(-0.5 * h, 0.5 * h, n_h, dtype=np.float64)
        lx, ly, lz = np.meshgrid(xs, ys, zs, indexing="ij")
        local = np.column_stack(
            [lx.reshape(-1), ly.reshape(-1), lz.reshape(-1)]
        )

    c, s = np.cos(yaw), np.sin(yaw)
    xr = c * local[:, 0] - s * local[:, 1]
    yr = s * local[:, 0] + c * local[:, 1]
    return np.column_stack([xr + x, yr + y, local[:, 2] + z])


def _ray_hits_obb(origin, direction, bbox, t_max: float) -> bool:
    """Unit-direction ray vs BEV OBB. Hit if first intersection is in [0, t_max)."""
    x, y = float(bbox[0]), float(bbox[1])
    l, w = float(bbox[3]), float(bbox[4])
    yaw = float(bbox[6]) if len(bbox) > 6 else 0.0
    c, s = np.cos(-yaw), np.sin(-yaw)
    ox, oy = float(origin[0]) - x, float(origin[1]) - y
    lx = c * ox - s * oy
    ly = s * ox + c * oy
    dx, dy = float(direction[0]), float(direction[1])
    ldx = c * dx - s * dy
    ldy = s * dx + c * dy
    tmin = 0.0
    tmax = float(t_max)
    for orig_i, dir_i, half in ((lx, ldx, l / 2.0), (ly, ldy, w / 2.0)):
        if abs(dir_i) < 1e-12:
            if abs(orig_i) > half:
                return False
            continue
        t1 = (-half - orig_i) / dir_i
        t2 = (half - orig_i) / dir_i
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        tmin = max(tmin, lo)
        tmax = min(tmax, hi)
        if tmin > tmax:
            return False
    return tmax >= 0.0 and tmin < float(t_max) - _HIT_EPS


def _is_query(box, other, dist=0.5) -> bool:
    """True if `other` is a duplicate of the query, not a separate occluder."""
    a = np.asarray(box, dtype=np.float64).reshape(-1)
    b = np.asarray(other, dtype=np.float64).reshape(-1)
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))) <= float(dist)


def visibility_ratio(
    box,
    occluders=None,
    origin=(0.0, 0.0),
) -> float:
    """Fraction of BEV rays from origin to the box not blocked by occluder boxes."""
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
    if not bool(lidar_range_mask(b.reshape(1, -1))[0]):
        return 0.0
    pts = _sample_points(b)
    ox, oy = float(origin[0]), float(origin[1])
    occ = _as_boxes(occluders)
    others = []
    for o in occ:
        if _is_query(b, o):
            continue
        others.append(o)

    n_free = 0
    n = int(pts.shape[0])
    for p in pts:
        dx, dy = float(p[0]) - ox, float(p[1]) - oy
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            n_free += 1
            continue
        direction = np.array([dx / dist, dy / dist], dtype=np.float64)
        blocked = False
        for o in others:
            if _ray_hits_obb((ox, oy), direction, o, dist):
                blocked = True
                break
        if not blocked:
            n_free += 1
    return float(n_free) / float(n) if n else 0.0


def is_visible(box, occluders=None, v_min: float = V_MIN, origin=(0.0, 0.0)) -> bool:
    return visibility_ratio(box, occluders=occluders, origin=origin) >= float(v_min)


# ---------------------------------------------------------------------------
# Point-cloud occlusion (polar min-range depth)
# ---------------------------------------------------------------------------


@dataclass
class PolarDepthMap:
    """Sensor-frame min-range map indexed by azimuth / elevation bins.

    Points and query boxes must be in the **same** frame as `origin`
    (usually the LiDAR frame with origin ≈ (0,0,0)).
    """

    min_range: np.ndarray  # (n_az, n_el), inf = empty
    az_res: float
    el_res: float
    az_min: float = -180.0
    el_min: float = -90.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def n_az(self) -> int:
        return int(self.min_range.shape[0])

    @property
    def n_el(self) -> int:
        return int(self.min_range.shape[1])

    def _bin(self, az_deg: float, el_deg: float) -> Tuple[int, int]:
        ia = int(np.floor((float(az_deg) - self.az_min) / self.az_res))
        ie = int(np.floor((float(el_deg) - self.el_min) / self.el_res))
        ia = int(np.clip(ia, 0, self.n_az - 1))
        ie = int(np.clip(ie, 0, self.n_el - 1))
        return ia, ie

    def min_range_near(self, az_deg: float, el_deg: float, nb: int = POLAR_NB) -> float:
        """Minimum stored range in a (2*nb+1)^2 neighbourhood of (az, el)."""
        ia, ie = self._bin(az_deg, el_deg)
        rmin = float("inf")
        for da in range(-int(nb), int(nb) + 1):
            for de in range(-int(nb), int(nb) + 1):
                ja = ia + da
                je = ie + de
                if ja < 0 or ja >= self.n_az or je < 0 or je >= self.n_el:
                    continue
                r = float(self.min_range[ja, je])
                if r < rmin:
                    rmin = r
        return rmin


def build_polar_depth(
    lidar,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    az_res: float = AZ_RES_DEG,
    el_res: float = EL_RES_DEG,
    max_range: float = 150.0,
) -> PolarDepthMap:
    """Build a polar min-range map from a raw LiDAR cloud (N, ≥3)."""
    az_res = float(az_res)
    el_res = float(el_res)
    n_az = max(int(np.ceil(360.0 / az_res)), 1)
    n_el = max(int(np.ceil(180.0 / el_res)), 1)
    grid = np.full((n_az, n_el), np.inf, dtype=np.float64)

    pts = np.asarray(lidar)
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 3:
        xyz = pts[:, :3].astype(np.float64, copy=False)
        dx = xyz[:, 0] - ox
        dy = xyz[:, 1] - oy
        dz = xyz[:, 2] - oz
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        valid = (r > 1e-3) & (r <= float(max_range))
        if np.any(valid):
            dx, dy, dz, r = dx[valid], dy[valid], dz[valid], r[valid]
            az = np.degrees(np.arctan2(dy, dx))
            el = np.degrees(np.arctan2(dz, np.maximum(np.hypot(dx, dy), 1e-6)))
            ia = np.floor((az - (-180.0)) / az_res).astype(np.int32)
            ie = np.floor((el - (-90.0)) / el_res).astype(np.int32)
            ia = np.clip(ia, 0, n_az - 1)
            ie = np.clip(ie, 0, n_el - 1)
            # Vectorised scatter-min into the grid.
            flat = ia * n_el + ie
            order = np.argsort(r)
            flat_s = flat[order]
            r_s = r[order]
            _, first = np.unique(flat_s, return_index=True)
            grid.ravel()[flat_s[first]] = r_s[first]

    return PolarDepthMap(
        min_range=grid,
        az_res=az_res,
        el_res=el_res,
        az_min=-180.0,
        el_min=-90.0,
        origin=(ox, oy, oz),
    )


def _ray_obb_entry(origin, direction, bbox) -> float:
    """Distance along unit ray to first hit of BEV OBB; inf if miss."""
    x, y = float(bbox[0]), float(bbox[1])
    l, w = float(bbox[3]), float(bbox[4])
    yaw = float(bbox[6]) if len(bbox) > 6 else 0.0
    c, s = np.cos(-yaw), np.sin(-yaw)
    ox, oy = float(origin[0]) - x, float(origin[1]) - y
    lx = c * ox - s * oy
    ly = s * ox + c * oy
    dx, dy = float(direction[0]), float(direction[1])
    ldx = c * dx - s * dy
    ldy = s * dx + c * dy
    tmin = -1e30
    tmax = 1e30
    for orig_i, dir_i, half in ((lx, ldx, l / 2.0), (ly, ldy, w / 2.0)):
        if abs(dir_i) < 1e-12:
            if abs(orig_i) > half:
                return float("inf")
            continue
        t1 = (-half - orig_i) / dir_i
        t2 = (half - orig_i) / dir_i
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        tmin = max(tmin, lo)
        tmax = min(tmax, hi)
        if tmin > tmax:
            return float("inf")
    if tmax < 0.0:
        return float("inf")
    return float(max(tmin, 0.0))


def _clear_dist_for_box(box, default: float = CLEAR_DIST_M) -> float:
    """Fallback clear distance when OBB entry cannot be computed."""
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.size < 5:
        return float(default)
    # Half diagonal in BEV — covers self-points on the same vehicle.
    return float(max(default, 0.5 * float(np.hypot(b[3], b[4]))))


def visibility_ratio_lidar(
    box,
    depth: Optional[Union[PolarDepthMap, np.ndarray]] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    clear_dist: Optional[float] = None,
    check_ego_range: bool = False,
    az_res: float = AZ_RES_DEG,
    el_res: float = EL_RES_DEG,
    nb: int = POLAR_NB,
    bottom_center: bool = False,
) -> float:
    """Fraction of sample rays to `box` not occluded by the LiDAR depth map.

    `depth` may be a PolarDepthMap or a raw (N,≥3) cloud (built on the fly).
    `box` and `depth` must share the same sensor frame as `origin`.

    If `bottom_center` is True (OpenCOOD detector boxes), lift z by h/2 to the
    geometric center before sampling so low bottom-z rays do not false-hit ground.

    A ray to sample point p (range r) is blocked iff some return in a nearby
    (az, el) bin has range < r_enter − eps, where r_enter is the distance to
    the near face of the query OBB (so points on the target itself do not
    count). If `clear_dist` is set, use min(r_enter, r − clear_dist) as the
    cutoff instead.
    """
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7].copy()
    if bottom_center and b.size >= 6:
        b[2] = float(b[2]) + 0.5 * float(b[5])
    if check_ego_range and not bool(lidar_range_mask(b.reshape(1, -1))[0]):
        return 0.0

    if depth is None:
        return 1.0
    if not isinstance(depth, PolarDepthMap):
        depth = build_polar_depth(depth, origin=origin, az_res=az_res, el_res=el_res)

    ox = float(origin[0]) if origin is not None else float(depth.origin[0])
    oy = float(origin[1]) if origin is not None else float(depth.origin[1])
    oz = (
        float(origin[2])
        if origin is not None and len(origin) > 2
        else float(depth.origin[2])
    )

    clear_fb = (
        float(clear_dist)
        if clear_dist is not None
        else _clear_dist_for_box(b)
    )
    samples = _sample_points_3d(b)
    n_free = 0
    n = int(samples.shape[0])
    for p in samples:
        dx = float(p[0]) - ox
        dy = float(p[1]) - oy
        dz = float(p[2]) - oz
        r = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if r < 1e-6:
            n_free += 1
            continue
        # Near-face of the query box along this ray (BEV OBB).
        rh = float(np.hypot(dx, dy))
        if rh < 1e-6:
            r_enter = max(r - clear_fb, 0.0)
        else:
            direction = (dx / rh, dy / rh)
            entry = _ray_obb_entry((ox, oy), direction, b)
            if not np.isfinite(entry):
                r_enter = max(r - clear_fb, 0.0)
            else:
                r_enter = float(entry)
                if clear_dist is not None:
                    r_enter = min(r_enter, max(r - float(clear_dist), 0.0))
        az = float(np.degrees(np.arctan2(dy, dx)))
        el = float(np.degrees(np.arctan2(dz, max(rh, 1e-6))))
        # Compare horizontal ranges: polar map stores 3D range; convert cutoff
        # approximately with the sample's elevation factor r/rh.
        scale = r / max(rh, 1e-6)
        cutoff = r_enter * scale - _HIT_EPS
        r_occ = depth.min_range_near(az, el, nb=nb)
        if np.isfinite(r_occ) and r_occ < cutoff:
            continue  # blocked
        n_free += 1
    return float(n_free) / float(n) if n else 0.0


def is_visible_lidar(
    box,
    depth=None,
    v_min: float = V_MIN,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    **kwargs,
) -> bool:
    return visibility_ratio_lidar(box, depth=depth, origin=origin, **kwargs) >= float(
        v_min
    )
