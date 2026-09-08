"""Three-source detection: Ego-only / UAV-only / Init (equal-weight fusion).

Aligns with SYSTEM_DESIGN.md step 1:
- Ego-only: late PointPillars trained on vehicle (only_cav_id=1)
- UAV-only: late PointPillars trained on UAV (only_cav_id=2);
  LateFusionDataset projects UAV points into the vehicle (ego) frame
- Init: both lidars, intermediate fusion (AttFuse / Where2Comm / CoAlign)

If dedicated ego/uav models are not passed, falls back to the old shared
intermediate-fusion head: ego keeps one CAV; UAV lidar is warped into the
vehicle frame and placed on the ego slot (AttFuse z-range is vehicle BEV).

Each source returns (B, P, C):
- B: boxes (N,7) in ego lidar frame [x,y,z,l,w,h,yaw]
- P: existence scores from the detector
- C: perception confidence C_abs * V (V replaces C_n; polar-depth rays)
"""
from __future__ import annotations

import copy
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .confidence import (
    N_REF_EGO,
    N_REF_UAV,
    R0,
    R_MIN,
    boxes_geometric_center,
    confidences_for_boxes,
)
from .geometry import blank_lidar, iou_bev, pack_dets
from .gating import CERTAIN, Q_H, Q_L, THETA_P, force_certain_for_gt_id, gate_frame
from .far_certain import FarCertainTracker, T_FAR, backfill_tentative_to_certain
from .buffer import CONFIRM, SoftEgoDet, THETA_CONFIRM, THETA_SOFT, TentativeBuffer
from .trust_pool import TrustPool
from .occlusion import build_polar_depth, visibility_ratio_lidar
from .paths import EGO_ID, UAV_ID
from .attack_gt import is_car_oid
from .uav_fov import box_in_uav_fov


@dataclass
class SourceResult:
    """One view: Ego / UAV / Init."""

    name: str
    boxes: np.ndarray  # (N, 7)
    scores: np.ndarray  # (N,) P
    confidences: np.ndarray  # (N,) C = C_abs * V
    point_counts: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    visibilities: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64)
    )  # (N,) polar V
    c_abs: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64)
    )  # (N,) C_abs = clip(n/n_ref)

    def filter_by_score(self, score_thres: float) -> "SourceResult":
        if self.boxes.shape[0] == 0:
            return self
        m = self.scores >= float(score_thres)
        return SourceResult(
            name=self.name,
            boxes=self.boxes[m],
            scores=self.scores[m],
            confidences=self.confidences[m],
            point_counts=self.point_counts[m]
            if len(self.point_counts) == len(m)
            else self.point_counts,
            visibilities=self.visibilities[m]
            if len(self.visibilities) == len(m)
            else self.visibilities,
            c_abs=self.c_abs[m] if len(self.c_abs) == len(m) else self.c_abs,
        )

    def score_band(self, lo: float, hi: float) -> "SourceResult":
        """Keep boxes with lo ≤ P < hi (buffer soft Ego band)."""
        if self.boxes.shape[0] == 0:
            return self
        m = (self.scores >= float(lo)) & (self.scores < float(hi))
        return SourceResult(
            name=self.name,
            boxes=self.boxes[m],
            scores=self.scores[m],
            confidences=self.confidences[m],
            point_counts=self.point_counts[m]
            if len(self.point_counts) == len(m)
            else self.point_counts,
            visibilities=self.visibilities[m]
            if len(self.visibilities) == len(m)
            else self.visibilities,
            c_abs=self.c_abs[m] if len(self.c_abs) == len(m) else self.c_abs,
        )

    def dump_lines(
        self,
        gt_boxes=None,
        gt_ids=None,
        iou_thres=0.3,
        dist_thres=4.0,
        gate_states=None,
    ) -> List[str]:
        """One line per box: B, P, C, GT hit (IoU∨d), optional §5.1 state."""
        lines = []
        if self.n == 0:
            return ["  (none)"]
        gt_boxes = np.asarray(gt_boxes) if gt_boxes is not None else np.zeros((0, 7))
        order = np.argsort(-self.scores)
        for k, i in enumerate(order):
            b = self.boxes[i]
            p = float(self.scores[i])
            c = float(self.confidences[i]) if i < len(self.confidences) else 0.0
            extra = ""
            if gt_boxes.ndim == 2 and gt_boxes.shape[0] > 0:
                dxy = np.hypot(gt_boxes[:, 0] - b[0], gt_boxes[:, 1] - b[1])
                jn = int(np.argmin(dxy))
                ious = np.array([iou_bev(b, g) for g in gt_boxes], dtype=np.float64)
                j = int(np.argmax(ious))
                # prefer nearest center; if a farther GT actually overlaps, use that
                if float(ious[j]) > float(ious[jn]) + 1e-6:
                    use = j
                else:
                    use = jn
                iou = float(ious[use])
                dist = float(dxy[use])
                gid = gt_ids[use] if gt_ids is not None and use < len(gt_ids) else use
                tag = (
                    "hit"
                    if (iou >= float(iou_thres) or dist <= float(dist_thres))
                    else "miss"
                )
                extra = "  {} id={} iou={:.2f} dxy={:.1f}m".format(tag, gid, iou, dist)
            n_pts = int(self.point_counts[i]) if i < len(self.point_counts) else -1
            extra_n = "  n={}".format(n_pts) if n_pts >= 0 else ""
            v = float(self.visibilities[i]) if i < len(self.visibilities) else float("nan")
            extra_v = "  V={:.3f}".format(v) if v == v else ""
            q = float(p) * float(c)
            gate_extra = ""
            if gate_states is not None:
                st = gate_states.get(int(i))
                gate_extra = "  state={}".format(st if st else "—")
            lines.append(
                "  #{:<2d}  B=[{:7.2f} {:7.2f} {:6.2f}  {:5.2f} {:5.2f} {:5.2f} {:6.3f}]  "
                "P={:.3f}  C={:.3f}  Q={:.3f}{}{}{}{}".format(
                    k,
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    b[4],
                    b[5],
                    b[6],
                    p,
                    c,
                    q,
                    extra_n,
                    extra_v,
                    extra,
                    gate_extra,
                )
            )
        return lines

    @property
    def n(self) -> int:
        return int(self.boxes.shape[0])


def dump_boxes(boxes, ids=None, extras=None):
    """Print GT / extra boxes in the same B layout (no P/C).

    `extras`: optional per-row suffix strings (e.g. soft-ego P/V annotation).
    """
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return ["  (none)"]
    lines = []
    for k, b in enumerate(boxes):
        tag = ""
        if ids is not None and k < len(ids):
            tag = "  id={}".format(ids[k])
        extra = ""
        if extras is not None and k < len(extras) and extras[k]:
            extra = str(extras[k])
        lines.append(
            "  #{:<2d}{}  B=[{:7.2f} {:7.2f} {:6.2f}  {:5.2f} {:5.2f} {:5.2f} {:6.3f}]{}".format(
                k, tag, b[0], b[1], b[2], b[3], b[4], b[5], b[6], extra
            )
        )
    return lines


def soft_ego_gt_extras(
    gt_boxes,
    gt_ids=None,
    ego_raw: Optional[SourceResult] = None,
    p_lo: float = THETA_SOFT,
    p_hi: float = 0.3,
    iou_thres: float = 0.3,
    ego_depth=None,
) -> List[str]:
    """Per-GT suffix when a soft-ego det (p_lo ≤ P < p_hi) matches IoU ≥ iou_thres.

    Annotates `` soft-ego P=.. V=.. iou=..``. V from ego LiDAR polar depth when
    available, else ``V=—``. If several soft dets hit the same GT, keep the
    highest-IoU one (tie → higher P).
    """
    gts = np.asarray(gt_boxes) if gt_boxes is not None else np.zeros((0, 7))
    n = int(gts.shape[0]) if gts.ndim == 2 else 0
    extras = [""] * n
    if n == 0 or ego_raw is None or getattr(ego_raw, "n", 0) <= 0:
        return extras
    hi = float(p_hi)
    lo = float(p_lo)
    if hi <= lo:
        return extras
    band = ego_raw.score_band(lo, hi)
    if band.n <= 0:
        return extras

    # best[gt_i] = (iou, p, v)
    best: Dict[int, Tuple[float, float, float]] = {}
    for i in range(int(band.n)):
        b = np.asarray(band.boxes[i], dtype=np.float64)[:7]
        p = float(band.scores[i])
        ious = np.array([float(iou_bev(b, g)) for g in gts], dtype=np.float64)
        j = int(np.argmax(ious))
        iou = float(ious[j])
        if iou < float(iou_thres):
            continue
        if ego_depth is not None:
            try:
                v = float(
                    visibility_ratio_lidar(
                        b,
                        depth=ego_depth,
                        origin=(0.0, 0.0, 0.0),
                        check_ego_range=True,
                        bottom_center=True,
                    )
                )
            except Exception:
                v = float("nan")
        else:
            v = float("nan")
        prev = best.get(j)
        if prev is None or iou > prev[0] + 1e-9 or (
            abs(iou - prev[0]) <= 1e-9 and p > prev[1]
        ):
            best[j] = (iou, p, v)

    for j, (iou, p, v) in best.items():
        if v == v:  # not NaN
            extras[j] = "  soft-ego P={:.3f} V={:.3f} iou={:.2f}".format(p, v, iou)
        else:
            extras[j] = "  soft-ego P={:.3f} V=— iou={:.2f}".format(p, iou)
    return extras


@dataclass
class FrameThreeSource:
    frame_id: int
    ego: SourceResult
    uav: SourceResult
    init: SourceResult
    gt_ego: Optional[np.ndarray] = None
    gt_uav: Optional[np.ndarray] = None
    gt_ego_ids: Optional[List] = None
    gt_uav_ids: Optional[List] = None
    gt_bboxes: Optional[np.ndarray] = None  # alias of gt_ego
    gt_eval: Optional[np.ndarray] = None  # V2U4 Car + range, ego frame
    gt_eval_ids: Optional[List] = None
    gt_ego_types: Optional[Dict] = None
    gt_uav_types: Optional[Dict] = None
    gating: Any = None
    buffer: Any = None  # FrameBuffer from defense.buffer (§7)
    pool: Any = None  # FramePool from defense.trust_pool (§4.3)
    attack_target: Optional[np.ndarray] = None  # ego-frame spoof ghost / remove box
    # Algorithm-level baselines (--defense none/robosac/cad/cp_guard/made)
    baseline_name: Optional[str] = None
    baseline_boxes: Optional[np.ndarray] = None
    baseline_scores: Optional[np.ndarray] = None
    baseline_info: Optional[Dict[str, Any]] = None
    fusion_z: Optional[Dict[str, Any]] = None  # MADE Z_ego / Z_fused from Init forward
    ego_raw: Optional[SourceResult] = None  # pre-score_thres Ego, for buffer soft band
    ego_depth: Any = None  # PolarDepthMap for soft-ego V dump (optional)

    def summary(self, iou_thres: float = 0.3) -> str:
        """Counts aligned with --verbose dump: Car+range GT, hit dets vs gt_eval."""
        from .metrics import hit_boxes

        ge, _ = filter_car_range_gt(self.gt_ego, self.gt_ego_ids, self.gt_ego_types)
        gu, _ = filter_car_range_gt(self.gt_uav, self.gt_uav_ids, self.gt_uav_types)
        gts = np.asarray(self.gt_eval) if self.gt_eval is not None else np.zeros((0, 7))
        n_ego = int(hit_boxes(self.ego.boxes, gts, iou_thres).shape[0])
        n_uav = int(hit_boxes(self.uav.boxes, gts, iou_thres).shape[0])
        n_init = int(hit_boxes(self.init.boxes, gts, iou_thres).shape[0])
        n_ev = 0 if gts.ndim != 2 else int(gts.shape[0])
        line = (
            "frame {:>2} | ego={:>3} uav={:>3} init={:>3} | gt_ego={:>3} gt_uav={:>3} gt_eval={:>3}".format(
                self.frame_id, n_ego, n_uav, n_init, len(ge), len(gu), n_ev
            )
        )
        if self.gating is not None:
            line = line + " | " + self.gating.summary()
        if self.pool is not None:
            line = line + " | " + self.pool.summary()
        if self.buffer is not None:
            line = line + " | " + self.buffer.summary()
        return line


def _obj_type_map(frame, cav_id):
    params = frame[cav_id].get("params") or {}
    vehicles = params.get("vehicles") or {}
    out = {}
    for oid, info in vehicles.items():
        out[str(oid)] = (info or {}).get("obj_type", "")
    return out


def filter_car_range_gt(boxes, ids, type_map):
    """Keep obj_type==Car boxes inside lidar range (eval / dump filter)."""
    from .metrics import lidar_range_mask

    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    ids = list(ids or [])
    type_map = type_map or {}
    kept_b, kept_i = [], []
    for i, oid in enumerate(ids):
        if i >= len(boxes):
            break
        if not is_car_oid(oid, type_map):
            continue
        kept_b.append(np.asarray(boxes[i], dtype=np.float64))
        kept_i.append(oid)
    if not kept_b:
        return np.zeros((0, 7), dtype=np.float64), []
    stacked = np.stack(kept_b)
    mask = lidar_range_mask(stacked)
    return stacked[mask], [kept_i[j] for j in range(len(kept_i)) if mask[j]]


def build_v2u4_eval_gt(frame, ego_id, uav_id):
    """Car-only unique boxes in ego frame, inside V2U4 lidar range.

    Matches OpenCOOD generate_object_center (obj_type=='Car') + GT_RANGE mask.
    Returns (boxes, object_ids).
    """
    from .metrics import lidar_range_mask

    cars = {}
    for cav_id, warp in ((ego_id, False), (uav_id, True)):
        if cav_id not in frame:
            continue
        types = _obj_type_map(frame, cav_id)
        boxes = np.asarray(frame[cav_id].get("gt_bboxes", np.zeros((0, 7))))
        ids = list(frame[cav_id].get("object_ids") or [])
        pose_from = frame[cav_id]["lidar_pose"]
        pose_to = frame[ego_id]["lidar_pose"]
        for i, oid in enumerate(ids):
            if i >= len(boxes):
                break
            if not is_car_oid(oid, types):
                continue
            key = str(oid)
            if key in cars:
                continue
            b = np.asarray(boxes[i], dtype=np.float64)
            if warp:
                b = _bbox_to_pose(b, pose_from, pose_to)
            cars[key] = b
    if not cars:
        return np.zeros((0, 7), dtype=np.float64), []
    keys = list(cars.keys())
    stacked = np.stack([cars[k] for k in keys])
    mask = lidar_range_mask(stacked)
    return stacked[mask], [keys[j] for j in range(len(keys)) if mask[j]]


def _bbox_to_pose(bbox, pose_from, pose_to):
    """Move one box with the same x1_to_x2 used for lidar / OpenCOOD."""
    from opencood.utils.transformation_utils import x1_to_x2

    T = x1_to_x2(
        np.asarray(pose_from).reshape(-1).tolist(),
        np.asarray(pose_to).reshape(-1).tolist(),
    )
    b = np.asarray(bbox, dtype=np.float64).copy()
    q = T.dot(np.array([b[0], b[1], b[2], 1.0]))
    b[0], b[1], b[2] = q[0], q[1], q[2]
    b[6] = b[6] + float(np.arctan2(T[1, 0], T[0, 0]))
    return b


def _boxes_to_pose(boxes, pose_from, pose_to):
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 7))
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.stack([_bbox_to_pose(b, pose_from, pose_to) for b in boxes])


def _lidar_to_pose(lidar, pose_from, pose_to):
    """Move a lidar cloud from pose_from sensor frame into pose_to (OpenCOOD x1_to_x2)."""
    from opencood.utils.transformation_utils import x1_to_x2

    pts = np.asarray(lidar)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return blank_lidar(lidar)
    T = x1_to_x2(
        np.asarray(pose_from).reshape(-1).tolist(),
        np.asarray(pose_to).reshape(-1).tolist(),
    )
    homo = np.c_[pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)]
    out = pts.copy()
    out[:, :3] = (T.dot(homo.T)).T[:, :3]
    return out


def _keep_one_cav(frame, keep_id, lidar=None):
    """Single-agent forward: CoAlign warp needs record_len == feature batch.

    Blanking the other lidar still leaves 2 CAVs in pairwise_t_matrix but 1
    voxel batch (empty cloud cropped), which crashes affine_grid ([1,3] vs [2,3]).
    """
    one = OrderedDict()
    one[keep_id] = copy.deepcopy(frame[keep_id])
    if lidar is not None:
        one[keep_id]["lidar"] = lidar
    return one


_SAFE_RUN_TRACED = set()


def _safe_run(perception, frame, ego_id, tag="") -> Tuple[np.ndarray, np.ndarray]:
    try:
        return pack_dets(*perception.run(frame, ego_id))
    except Exception as e:
        print("[three-source] {} run failed (ego_id={}): {}".format(tag or "det", ego_id, e))
        key = tag or "det"
        if key not in _SAFE_RUN_TRACED:
            _SAFE_RUN_TRACED.add(key)
            traceback.print_exc()
        return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.float64)


def _c_from_ego_lidar(boxes, lidar_ego, n_ref: float, r0: float, r_min: float = R_MIN):
    """C_ego = C_abs * V_ego (polar depth on ego cloud). Returns C,n,V,C_abs."""
    return confidences_for_boxes(
        boxes, lidar_ego, source="ego", n_ref=n_ref, r0=r0, r_min=r_min
    )


def _c_from_uav_lidar(
    boxes_ego, frame, ego_id, uav_id, n_ref: float, r0: float, r_min: float = R_MIN
):
    """C_uav = C_abs * V_uav on UAV cloud (boxes warped to UAV frame).

    Lift z to geometric center in ego frame, then warp, so height is not
    re-applied along UAV z after the pose transform.
    """
    boxes_ego = np.asarray(boxes_ego)
    if boxes_ego.ndim != 2 or boxes_ego.shape[0] == 0:
        zc = np.zeros((0,), dtype=np.float64)
        zn = np.zeros((0,), dtype=np.int32)
        return zc, zn, zc.copy(), zc.copy()
    pose_e = frame[ego_id]["lidar_pose"]
    pose_u = frame[uav_id]["lidar_pose"]
    lidar_u = frame[uav_id]["lidar"]
    boxes_u = _boxes_to_pose(boxes_geometric_center(boxes_ego), pose_e, pose_u)
    return confidences_for_boxes(
        boxes_u,
        lidar_u,
        source="uav",
        n_ref=n_ref,
        r0=r0,
        r_min=r_min,
        bottom_center=False,
    )


def _c_init(
    boxes_ego,
    frame,
    ego_id,
    uav_id,
    n_ref_ego: float,
    n_ref_uav: float,
    r0: float,
    r_min: float = R_MIN,
    r0_uav: Optional[float] = None,
):
    """C_init = mean(C_ego, C_uav); also mean V and C_abs."""
    boxes_ego = np.asarray(boxes_ego)
    if boxes_ego.ndim != 2 or boxes_ego.shape[0] == 0:
        zc = np.zeros((0,), dtype=np.float64)
        zn = np.zeros((0,), dtype=np.int32)
        return zc, zn, zc.copy(), zc.copy()
    c_e, n_e, v_e, ca_e = _c_from_ego_lidar(
        boxes_ego, frame[ego_id]["lidar"], n_ref_ego, r0, r_min
    )
    c_u, n_u, v_u, ca_u = _c_from_uav_lidar(
        boxes_ego,
        frame,
        ego_id,
        uav_id,
        n_ref_uav,
        float(r0 if r0_uav is None else r0_uav),
        r_min,
    )
    cs = (c_e + c_u) / 2.0
    ns = ((n_e.astype(np.float64) + n_u.astype(np.float64)) / 2.0).astype(np.int32)
    vs = (v_e + v_u) / 2.0
    cas = (ca_e + ca_u) / 2.0
    return cs, ns, vs, cas


def _concat_boxes(*arrs) -> np.ndarray:
    rows = []
    for a in arrs:
        b = np.asarray(a) if a is not None else np.zeros((0, 7))
        if b.ndim == 2 and b.shape[0] > 0:
            rows.append(b[:, :7])
    if not rows:
        return np.zeros((0, 7), dtype=np.float64)
    return np.vstack(rows)


def _pool_birth_from(gating, buf, gt_lists=None):
    """Confirmed leftover output for §4.3.1.

    Returns (boxes, c_egos, allow_update, trajs). allow_update=True only for spatial
    Certain (KF update on merge); Tentative confirm can birth but not update
    existing tracks. `trajs[j]` is the watch trajectory for KF birth velocity
    (ego > fuse), or None.
    Skips GT id=1 (self-car): never enter the trust pool.
    """
    from .gating import FORCE_CERTAIN_GT_ID, box_matches_gt_id

    def _is_self(box) -> bool:
        for gt_boxes, gt_ids in gt_lists or ():
            if box_matches_gt_id(box, gt_boxes, gt_ids, oid=FORCE_CERTAIN_GT_ID):
                return True
        return False

    boxes, cs, flags, trajs = [], [], [], []
    for o in getattr(gating, "objects", None) or []:
        if getattr(o, "from_far", False):
            continue
        if getattr(o, "state", None) == CERTAIN and o.box is not None:
            if _is_self(o.box):
                continue
            boxes.append(np.asarray(o.box, dtype=np.float64)[:7])
            cs.append(float(o.c_ego))
            flags.append(True)
            trajs.append(None)
    # Prefer per-confirm trajs collected this frame (aligned with CONFIRM events).
    confirm_trajs = list(getattr(buf, "last_confirm_trajs", None) or [])
    ci = 0
    for e in getattr(buf, "events", None) or []:
        if getattr(e, "decision", None) == CONFIRM and e.box is not None:
            if _is_self(e.box):
                continue
            boxes.append(np.asarray(e.box, dtype=np.float64)[:7])
            cs.append(float(getattr(e, "c_ego", 0.0) or 0.0))
            flags.append(False)
            tr = None
            if ci < len(confirm_trajs):
                tr = confirm_trajs[ci].get("traj") or confirm_trajs[ci].get("hist")
            elif getattr(e, "traj", None):
                # snapshot may omit traj; live confirms list is preferred
                tr = None
            trajs.append(list(tr) if tr else None)
            ci += 1
    if not boxes:
        return np.zeros((0, 7), dtype=np.float64), [], [], []
    return np.stack(boxes), cs, flags, trajs


def _soft_ego_dets(
    ego_raw, hard_thres, soft_thres: float = THETA_SOFT, buf_q_mode: str = "pc_abs"
):
    """Ego boxes with θ_soft ≤ P < score_thres — buffer evidence only."""
    if ego_raw is None or hard_thres is None:
        return []
    hi = float(hard_thres)
    lo = float(soft_thres)
    if hi <= lo or getattr(ego_raw, "n", 0) <= 0:
        return []
    band = ego_raw.score_band(lo, hi)
    use_p = str(buf_q_mode or "pc_abs").lower() in ("p", "p_only", "score")
    out = []
    for i in range(int(band.n)):
        p = float(band.scores[i])
        ca = float(band.c_abs[i]) if i < len(band.c_abs) else 0.0
        q = float(p) if use_p else float(p) * float(ca)
        out.append(
            SoftEgoDet(
                box=np.asarray(band.boxes[i], dtype=np.float64)[:7].copy(),
                q=q,
                p=float(p),
            )
        )
    return out


class ThreeSourceDetector:
    """Run Ego / UAV / Init detections for each frame of a case."""

    def __init__(
        self,
        perception,
        ego_id: str = EGO_ID,
        uav_id: str = UAV_ID,
        n_ref: float = N_REF_EGO,
        n_ref_uav: float = N_REF_UAV,
        r0: float = R0,
        r0_uav: Optional[float] = None,
        r_min: float = R_MIN,
        score_thres: Optional[float] = None,
        theta_p: Optional[float] = None,
        q_h: float = Q_H,
        q_l: float = Q_L,
        ego_perception=None,
        uav_perception=None,
        gate_q_mode: str = "pc",
        theta_confirm: float = THETA_CONFIRM,
        buf_q_mode: str = "pc_abs",
    ):
        self.perception = perception
        self.ego_perception = ego_perception
        self.uav_perception = uav_perception
        self.ego_id = ego_id
        self.uav_id = uav_id
        self.n_ref = float(n_ref)
        self.n_ref_uav = float(n_ref_uav)
        self.r0 = float(r0)
        self.r0_uav = float(r0 if r0_uav is None else r0_uav)
        self.r_min = float(r_min)
        self.score_thres = score_thres
        if theta_p is not None:
            self.theta_p = float(theta_p)
        elif score_thres is not None:
            self.theta_p = float(score_thres)
        else:
            self.theta_p = float(THETA_P)
        self.q_h = float(q_h)
        self.q_l = float(q_l)
        self.gate_q_mode = str(gate_q_mode or "pc")
        self.theta_confirm = float(theta_confirm)
        self.buf_q_mode = str(buf_q_mode or "pc_abs")
        self.tent_buffer = TentativeBuffer(
            q_l=self.q_l,
            theta_confirm=self.theta_confirm,
            buf_q_mode=self.buf_q_mode,
        )
        self.far_tracker = FarCertainTracker()
        self.trust_pool = TrustPool(
            theta_p=self.theta_p,
            q_h=self.q_h,
            n_ref_ego=self.n_ref,
            r0=self.r0,
            r_min=self.r_min,
        )

    def apply_defense(
        self,
        ego,
        uav,
        init,
        gt_eval,
        gt_eval_ids,
        gt_ego=None,
        gt_ego_ids=None,
        ego_lidar=None,
        frame_id: int = 0,
        frame: Optional[Dict[str, Any]] = None,
        ego_raw=None,
    ):
        """§9: gate → far-bypass → pool → leftover §7.

        Pool ATTACK uses a UAV-observation callback (LiDAR polar-depth occlusion
        from the UAV sensor, OR UAV point density at the predicted box), not the
        ego-origin box-ray: an ego miss caused by ego occlusion or range should not
        force MISS when the UAV physically sees the object, and a point-erasing
        attack (early/point-cloud remove) still counts as ATTACK when the UAV
        depth map still has a free line of sight (or leftover points).

        Buffer visibility uses the same polar-depth test from the **ego** LiDAR.
        """
        gating = gate_frame(
            ego,
            uav,
            init,
            theta_p=self.theta_p,
            q_h=self.q_h,
            q_l=self.q_l,
            gate_q_mode=self.gate_q_mode,
        )
        gating = force_certain_for_gt_id(gating, gt_eval, gt_eval_ids)
        gating = force_certain_for_gt_id(gating, gt_ego, gt_ego_ids)
        self.far_tracker.step(gating, frame_id=frame_id)
        gt_lists = ((gt_eval, gt_eval_ids), (gt_ego, gt_ego_ids))

        # Per-frame polar depth maps (sensor frames, origin ≈ 0).
        ego_depth = None
        uav_depth = None
        if frame is not None and self.ego_id in frame:
            try:
                ego_depth = build_polar_depth(frame[self.ego_id].get("lidar"))
            except Exception:
                ego_depth = None
        if frame is not None and self.uav_id in frame:
            try:
                uav_depth = build_polar_depth(frame[self.uav_id].get("lidar"))
            except Exception:
                uav_depth = None

        pose_ego = None
        pose_uav = None
        if frame is not None and self.uav_id in frame and self.ego_id in frame:
            pose_ego = frame[self.ego_id].get("lidar_pose")
            pose_uav = frame[self.uav_id].get("lidar_pose")

        def _uav_sight(box, occ=None) -> float:
            """0..1 UAV-observation strength for pool ATTACK.

            Inside percentile FOV (r / az / el p5–p95 in UAV frame):
                V = max(LOS_lidar_from_UAV, C_uav)
            Outside FOV (blind zone; empty depth often spuriously ≈1):
                V = C_uav only
            """
            del occ  # box-occluders unused; LiDAR depth replaces them
            c_uav = 0.0
            try:
                b = np.asarray(box, dtype=np.float64).reshape(-1)
                if b.size >= 7 and frame is not None and self.uav_id in frame:
                    c, _, _, _ = _c_from_uav_lidar(
                        b[:7].reshape(1, 7),
                        frame,
                        self.ego_id,
                        self.uav_id,
                        self.n_ref_uav,
                        self.r0_uav,
                        self.r_min,
                    )
                    if c.size:
                        c_uav = float(c[0])
            except Exception:
                c_uav = 0.0

            in_fov = box_in_uav_fov(
                box, frame, ego_id=self.ego_id, uav_id=self.uav_id
            )
            # Unknown pose → treat as outside FOV (density-only; safer than fake LOS).
            if in_fov is not True:
                return float(c_uav)

            los = 0.0
            if uav_depth is not None and pose_ego is not None and pose_uav is not None:
                try:
                    b0 = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
                    # Detector boxes are bottom-center; lift before ego→uav warp.
                    b0 = boxes_geometric_center(b0.reshape(1, 7))[0]
                    b_u = _bbox_to_pose(b0, pose_ego, pose_uav)
                    los = float(
                        visibility_ratio_lidar(
                            b_u,
                            depth=uav_depth,
                            origin=(0.0, 0.0, 0.0),
                            check_ego_range=False,
                            bottom_center=False,
                        )
                    )
                except Exception:
                    los = 0.0
            return float(max(los, c_uav))

        def _ego_vis(box) -> float:
            """Ego LiDAR polar-depth visibility (buffer ATTACK / evidence)."""
            if ego_depth is None:
                return 1.0
            try:
                b0 = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
                return float(
                    visibility_ratio_lidar(
                        b0,
                        depth=ego_depth,
                        origin=(0.0, 0.0, 0.0),
                        check_ego_range=True,
                        bottom_center=True,
                    )
                )
            except Exception:
                return 0.0

        pool = self.trust_pool.step(
            gating,
            ego,
            uav,
            init,
            gt_boxes=gt_eval,
            gt_ids=gt_eval_ids,
            gt_lists=gt_lists,
            uav_sight=_uav_sight if frame is not None else None,
            birth_traj_hints=[
                {
                    "box": np.asarray(e.box, dtype=np.float64)[:7].copy(),
                    "traj": list(e.velocity_traj()),
                    "hist": list(e.hist or []),
                }
                for e in (self.tent_buffer.entries or [])
                if e.box is not None
            ],
        )
        leftover = replace(
            gating,
            objects=[o for o in gating.objects if not pool.claimed_object(o)],
        )
        buf = self.tent_buffer.step(
            leftover,
            gt_eval,
            gt_eval_ids,
            frame_id=frame_id,
            match_objects=list(gating.objects),
            soft_ego=_soft_ego_dets(
                ego_raw, self.score_thres, buf_q_mode=self.buf_q_mode
            ),
            visibility_fn=_ego_vis if ego_depth is not None else None,
            ego=ego,
            uav=uav,
            init=init,
        )
        birth_boxes, birth_c, birth_upd, birth_trajs = _pool_birth_from(
            leftover,
            buf,
            gt_lists=gt_lists,
        )
        self.trust_pool.ingest(
            birth_boxes, birth_c, allow_update=birth_upd, trajs=birth_trajs
        )
        # Watches that matched Certain this frame: pool may have birthed/updated
        # before buffer ran — seed vx,vy from the tentative traj if still ~0.
        self.trust_pool.seed_velocity_from_trajs(
            list(getattr(buf, "last_certain_trajs", None) or [])
            + list(getattr(buf, "last_confirm_trajs", None) or [])
        )
        self.trust_pool.label_living_sources(ego, uav, init)
        full = _concat_boxes(pool.output_boxes, buf.output_boxes)
        self.tent_buffer.prev_output = full
        buf.output_boxes = full
        return gating, buf, pool

    def run_frame(self, frame: Dict[str, Any], frame_id: int = 0) -> FrameThreeSource:
        ego_id, uav_id = self.ego_id, self.uav_id

        # Init = both lidars, normal intermediate fusion (vehicle is fusion ego)
        b_init, p_init = _safe_run(self.perception, frame, ego_id, tag="init")
        from .baselines.made_ae import compact_fusion_z

        fusion_z = compact_fusion_z(getattr(self.perception, "last_fusion_z", None))
        try:
            self.perception.last_fusion_z = fusion_z
            if getattr(self.perception, "model", None) is not None:
                self.perception.model._defense_z = None
        except Exception:
            pass
        c_init, n_init, v_init, ca_init = _c_init(
            b_init,
            frame,
            ego_id,
            uav_id,
            self.n_ref,
            self.n_ref_uav,
            self.r0,
            self.r_min,
            r0_uav=self.r0_uav,
        )

        # Ego-only: dedicated vehicle PointPillars, or shared fusion head
        if self.ego_perception is not None:
            b_ego, p_ego = _safe_run(self.ego_perception, frame, ego_id, tag="ego")
        else:
            fe = _keep_one_cav(frame, ego_id)
            b_ego, p_ego = _safe_run(self.perception, fe, ego_id, tag="ego")
        c_ego, n_ego, v_ego, ca_ego = _c_from_ego_lidar(
            b_ego, frame[ego_id]["lidar"], self.n_ref, self.r0, self.r_min
        )

        # UAV-only: dedicated UAV PointPillars (projects to ego frame internally),
        # or warp UAV cloud onto the ego slot of the shared fusion head
        if self.uav_perception is not None:
            b_uav, p_uav = _safe_run(self.uav_perception, frame, ego_id, tag="uav")
        else:
            uav_in_ego = _lidar_to_pose(
                frame[uav_id]["lidar"],
                frame[uav_id]["lidar_pose"],
                frame[ego_id]["lidar_pose"],
            )
            fu = _keep_one_cav(frame, ego_id, lidar=uav_in_ego)
            b_uav, p_uav = _safe_run(self.perception, fu, ego_id, tag="uav")
        c_uav, n_uav, v_uav, ca_uav = _c_from_uav_lidar(
            b_uav, frame, ego_id, uav_id, self.n_ref_uav, self.r0_uav, self.r_min
        )

        ego = SourceResult("ego", b_ego, p_ego, c_ego, n_ego, v_ego, ca_ego)
        uav = SourceResult("uav", b_uav, p_uav, c_uav, n_uav, v_uav, ca_uav)
        init = SourceResult("init", b_init, p_init, c_init, n_init, v_init, ca_init)
        ego_raw = ego

        def _maxp(src):
            return float(src.scores.max()) if src.n else 0.0

        print(
            "[raw] frame {:>2} | ego n={:<3} maxP={:.3f} | uav n={:<3} maxP={:.3f} | init n={:<3} maxP={:.3f}".format(
                frame_id, ego.n, _maxp(ego), uav.n, _maxp(uav), init.n, _maxp(init)
            )
        )

        if self.score_thres is not None:
            n_e, n_u, n_i = ego.n, uav.n, init.n
            ego = ego.filter_by_score(self.score_thres)
            uav = uav.filter_by_score(self.score_thres)
            init = init.filter_by_score(self.score_thres)
            dropped = (n_e - ego.n) + (n_u - uav.n) + (n_i - init.n)
            if dropped:
                print(
                    "[filter] P<{} dropped ego {}→{} uav {}→{} init {}→{}".format(
                        self.score_thres, n_e, ego.n, n_u, uav.n, n_i, init.n
                    )
                )

        gt_ego = np.asarray(frame[ego_id].get("gt_bboxes", np.zeros((0, 7))))
        gt_uav_local = np.asarray(frame[uav_id].get("gt_bboxes", np.zeros((0, 7))))
        # UAV yaml GT is in UAV lidar frame; warp to vehicle frame to match dets
        gt_uav = _boxes_to_pose(
            gt_uav_local,
            frame[uav_id]["lidar_pose"],
            frame[ego_id]["lidar_pose"],
        )
        gt_ego_ids = list(frame[ego_id].get("object_ids") or [])
        gt_uav_ids = list(frame[uav_id].get("object_ids") or [])
        gt_ego_types = _obj_type_map(frame, ego_id)
        gt_uav_types = _obj_type_map(frame, uav_id)

        gt_eval, gt_eval_ids = build_v2u4_eval_gt(frame, ego_id, uav_id)

        gating, buf, pool = self.apply_defense(
            ego,
            uav,
            init,
            gt_eval,
            gt_eval_ids,
            gt_ego=gt_ego,
            gt_ego_ids=gt_ego_ids,
            ego_lidar=frame[ego_id]["lidar"],
            frame_id=frame_id,
            frame=frame,
            ego_raw=ego_raw,
        )

        ego_depth = None
        try:
            ego_depth = build_polar_depth(frame[ego_id].get("lidar"))
        except Exception:
            ego_depth = None

        return FrameThreeSource(
            frame_id=frame_id,
            ego=ego,
            uav=uav,
            init=init,
            gt_ego=gt_ego,
            gt_uav=gt_uav,
            gt_ego_ids=gt_ego_ids,
            gt_uav_ids=gt_uav_ids,
            gt_bboxes=gt_ego,
            gt_eval=gt_eval,
            gt_eval_ids=gt_eval_ids,
            gt_ego_types=gt_ego_types,
            gt_uav_types=gt_uav_types,
            gating=gating,
            buffer=buf,
            pool=pool,
            fusion_z=fusion_z,
            ego_raw=ego_raw,
            ego_depth=ego_depth,
        )

    def run_case(
        self,
        multi_frame_case: Dict[int, Any],
        frame_ids: Optional[List[int]] = None,
    ) -> List[FrameThreeSource]:
        if frame_ids is None:
            if hasattr(multi_frame_case, "keys"):
                frame_ids = sorted(multi_frame_case.keys())
            else:
                frame_ids = list(range(len(multi_frame_case)))
        self.tent_buffer.reset()
        self.trust_pool.reset()
        self.far_tracker.reset()
        out = []
        for f in frame_ids:
            out.append(self.run_frame(multi_frame_case[f], frame_id=f))
        hists = list(self.far_tracker.promoted_hists())
        backfill_tentative_to_certain(out, hists, t_far=T_FAR)
        backfill_tentative_to_certain(
            out, getattr(self.tent_buffer, "confirmed_hists", None) or [], t_far=None
        )
        backfill_tentative_to_certain(
            out, getattr(self.tent_buffer, "gate_certain_hists", None) or [], t_far=None
        )
        return out


def results_to_dict(frames: List[FrameThreeSource]) -> List[Dict[str, Any]]:
    """JSON-serializable dump (lists, not numpy)."""
    rows = []
    for fr in frames:
        row = {
            "frame_id": fr.frame_id,
            "gt_ego": None if fr.gt_ego is None else np.asarray(fr.gt_ego).tolist(),
            "gt_uav": None if fr.gt_uav is None else np.asarray(fr.gt_uav).tolist(),
            "gt_ego_ids": fr.gt_ego_ids,
            "gt_uav_ids": fr.gt_uav_ids,
        }
        for key, src in (("ego", fr.ego), ("uav", fr.uav), ("init", fr.init)):
            qs = (
                (np.asarray(src.scores, dtype=np.float64) * np.asarray(src.confidences, dtype=np.float64)).tolist()
                if src.n
                else []
            )
            row[key] = {
                "n": src.n,
                "boxes": src.boxes.tolist(),
                "scores": src.scores.tolist(),
                "confidences": src.confidences.tolist(),
                "point_counts": src.point_counts.tolist(),
                "visibilities": np.asarray(src.visibilities, dtype=np.float64).tolist()
                if src.n and len(src.visibilities) == src.n
                else [],
                "c_abs": np.asarray(src.c_abs, dtype=np.float64).tolist()
                if src.n and len(src.c_abs) == src.n
                else [],
                "qualities": qs,
            }
        if fr.gating is not None:
            row["gating"] = fr.gating.to_dict()
        if fr.pool is not None:
            row["pool"] = fr.pool.to_dict()
        if fr.buffer is not None:
            row["buffer"] = fr.buffer.to_dict()
        if getattr(fr, "attack_target", None) is not None:
            row["attack_target"] = np.asarray(fr.attack_target).tolist()
        rows.append(row)
    return rows
