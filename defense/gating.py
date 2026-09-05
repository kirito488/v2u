"""Spatial gating Certain / Ambiguous / Tentative (scheme §5.1 + §6.3).

Only leftover detections that did not match the trust pool (scheme §4.3.0).

Q_ego = P_ego * C_ego
Q_h = theta_P * mu_h = 0.35
Q_l = theta_P * mu_l = 0.25

D_src = 1 iff that source has P >= theta_P and Hungarian-matches this object.

§6.3: Ambiguous → Certain iff Dual=(D_uav=1 and D_init=1), else Tentative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from .associate import THETA_IOU, hungarian_match
from .geometry import iou_bev

THETA_P = 0.5
MU_H = 0.7
MU_L = 0.5
Q_H = THETA_P * MU_H  # 0.35
Q_L = THETA_P * MU_L  # 0.25

CERTAIN = "certain"
AMBIGUOUS = "ambiguous"
TENTATIVE = "tentative"

# GT object_id forced to Certain (self-car / id=1 in V2U4Real dumps).
FORCE_CERTAIN_GT_ID = "1"


def detection_quality(p, c) -> float:
    """Q = P * C (scheme §5.1)."""
    return float(p) * float(c)


def spatial_state(d_ego: int, q_ego: float, q_h: float = Q_H, q_l: float = Q_L) -> str:
    """§5.1 hard gate. Ambiguous is resolved by resolve_ambiguous (§6.3)."""
    if int(d_ego) == 1 and q_ego >= q_h:
        return CERTAIN
    if int(d_ego) == 1 and q_l <= q_ego < q_h:
        return AMBIGUOUS
    return TENTATIVE


def dual_consensus(d_uav: int, d_init: int) -> bool:
    """§6.2 Dual = (D_uav=1) AND (D_init=1)."""
    return int(d_uav) == 1 and int(d_init) == 1


def resolve_ambiguous(state: str, d_uav: int, d_init: int) -> str:
    """§6.3: Ambiguous → Certain iff UAV and Init both match Ego, else Tentative."""
    if state != AMBIGUOUS:
        return state
    if dual_consensus(d_uav, d_init):
        return CERTAIN
    return TENTATIVE


def original_oid_local(oid) -> str:
    s = str(oid)
    if s.endswith("(remove)"):
        return s[: -len("(remove)")]
    return s


def gt_boxes_for_id(gt_boxes, gt_ids, oid: str = FORCE_CERTAIN_GT_ID):
    """GT boxes whose object id equals oid (after stripping '(remove)')."""
    boxes = np.asarray(gt_boxes) if gt_boxes is not None else np.zeros((0, 7))
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return []
    ids = [str(x) for x in (gt_ids or [])]
    want = str(oid)
    out = []
    for i, gid in enumerate(ids):
        if i >= len(boxes):
            break
        if gid == want or original_oid_local(gid) == want:
            out.append(np.asarray(boxes[i], dtype=np.float64))
    return out


def box_matches_gt_id(
    box,
    gt_boxes,
    gt_ids,
    oid: str = FORCE_CERTAIN_GT_ID,
    iou_thres: float = THETA_IOU,
    max_dist: float = 4.0,
) -> bool:
    """True if box matches GT oid by IoU or center distance (same as force-Certain)."""
    if box is None:
        return False
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
    for t in gt_boxes_for_id(gt_boxes, gt_ids, oid=oid):
        iou = float(iou_bev(b, t))
        dist = float(np.hypot(b[0] - t[0], b[1] - t[1]))
        if iou >= float(iou_thres) or dist <= float(max_dist):
            return True
    return False


def force_certain_for_gt_id(
    gating: "FrameGating",
    gt_boxes,
    gt_ids,
    oid: str = FORCE_CERTAIN_GT_ID,
    iou_thres: float = THETA_IOU,
    max_dist: float = 4.0,
) -> "FrameGating":
    """Unconditionally mark gated objects that match GT id as Certain.

    Match by BEV IoU >= iou_thres OR center distance <= max_dist (same spirit as
    verbose hit tagging), so self-car id=1 is not missed when IoU is soft.
    """
    if gating is None or not gating.objects:
        return gating
    targets = gt_boxes_for_id(gt_boxes, gt_ids, oid=oid)
    if not targets:
        return gating
    for o in gating.objects:
        if o.box is None:
            continue
        b = np.asarray(o.box, dtype=np.float64)
        for t in targets:
            iou = float(iou_bev(b, t))
            dist = float(np.hypot(b[0] - t[0], b[1] - t[1]))
            if iou >= float(iou_thres) or dist <= float(max_dist):
                o.state = CERTAIN
                break
    gating.objects.sort(key=lambda o: (-int(o.d_ego), -o.q_ego, -o.p_ego))
    return gating


def _keep_detected(src: Any, theta_p: float, exclude=None) -> np.ndarray:
    """Indices with P >= theta_P (scheme D=1 candidate before matching)."""
    n = int(np.asarray(src.boxes).shape[0]) if getattr(src, "boxes", None) is not None else 0
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    idx = np.flatnonzero(np.asarray(src.scores) >= float(theta_p)).astype(np.int32)
    if exclude:
        ban = set(int(x) for x in exclude)
        idx = np.asarray([i for i in idx.tolist() if int(i) not in ban], dtype=np.int32)
    return idx


@dataclass
class GatedObject:
    """One associated object after §4 matching + §5.1 + §6.3."""

    state: str
    d_ego: int
    d_uav: int
    d_init: int
    q_ego: float
    p_ego: float
    c_ego: float
    ego_i: int = -1
    uav_i: int = -1
    init_i: int = -1
    iou_uav: float = 0.0
    iou_init: float = 0.0
    dual: int = 0  # §6.2 Dual = D_uav AND D_init
    from_ambiguous: bool = False  # §6.3 transferred from Ambiguous
    from_far: bool = False  # far-range bypass Certain (not pooled)
    backfill_certain: bool = False  # dump-only: later confirmed
    show_certain: bool = False  # dump-only: confirm frame
    p_uav: float = 0.0
    c_uav: float = 0.0
    p_init: float = 0.0
    c_init: float = 0.0
    box: Optional[np.ndarray] = None  # ego box if D_ego=1, else UAV then Init

    @property
    def display_state(self) -> str:
        """Dump label: spatial state, Dual transfer, far/buffer confirm backfill."""
        if self.backfill_certain:
            return "tentative -> certain"
        if self.show_certain or (self.from_far and self.state == CERTAIN):
            return CERTAIN
        if self.from_ambiguous and self.state in (CERTAIN, TENTATIVE):
            return "ambiguous -> {}".format(self.state)
        return self.state

    def dump_line(self, k: int) -> str:
        b = self.box if self.box is not None else np.zeros(7)
        return (
            "  #{:<2d}  {:<26s}  D=({}{}{})  Dual={}  Q={:.3f}  P={:.3f}  C={:.3f}  "
            "B=[{:7.2f} {:7.2f} {:6.2f}  {:5.2f} {:5.2f} {:5.2f} {:6.3f}]".format(
                k,
                self.display_state,
                self.d_ego,
                self.d_uav,
                self.d_init,
                int(self.dual),
                self.q_ego,
                self.p_ego,
                self.c_ego,
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
                b[6],
            )
        )


@dataclass
class FrameGating:
    """§5.1 + §6.3 result for one frame."""

    objects: List[GatedObject] = field(default_factory=list)
    theta_p: float = THETA_P
    q_h: float = Q_H
    q_l: float = Q_L

    @property
    def n_certain(self) -> int:
        return sum(1 for o in self.objects if o.state == CERTAIN)

    @property
    def n_ambiguous(self) -> int:
        return sum(1 for o in self.objects if o.state == AMBIGUOUS)

    @property
    def n_tentative(self) -> int:
        return sum(1 for o in self.objects if o.state == TENTATIVE)

    def summary(self) -> str:
        return "gate §5.1+§6.3 | certain={:>3} ambiguous={:>3} tentative={:>3}".format(
            self.n_certain, self.n_ambiguous, self.n_tentative
        )

    def dump_lines(self) -> List[str]:
        if not self.objects:
            return ["  (none)"]
        return [o.dump_line(k) for k, o in enumerate(self.objects)]

    def label_for_box(self, box, iou_thres: float = THETA_IOU) -> str:
        """Display state of the gate object at this box, or '' if none."""
        if box is None:
            return ""
        b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
        best, best_iou = "", float(iou_thres)
        for o in self.objects:
            if o.box is None:
                continue
            iou = float(iou_bev(b, np.asarray(o.box, dtype=np.float64).reshape(-1)[:7]))
            if iou >= best_iou:
                best, best_iou = o.display_state, iou
        return best

    def source_state_maps(self) -> dict:
        """Map source box index -> display state for ego/uav/init dump_lines."""
        out = {"ego": {}, "uav": {}, "init": {}}
        for o in self.objects:
            label = o.display_state
            if o.ego_i >= 0:
                out["ego"][int(o.ego_i)] = label
            if o.uav_i >= 0:
                out["uav"][int(o.uav_i)] = label
            if o.init_i >= 0:
                out["init"][int(o.init_i)] = label
        return out

    def to_dict(self) -> dict:
        rows = []
        for o in self.objects:
            rows.append(
                {
                    "state": o.state,
                    "display_state": o.display_state,
                    "from_ambiguous": bool(o.from_ambiguous),
                    "from_far": bool(getattr(o, "from_far", False)),
                    "backfill_certain": bool(getattr(o, "backfill_certain", False)),
                    "d_ego": o.d_ego,
                    "d_uav": o.d_uav,
                    "d_init": o.d_init,
                    "dual": int(o.dual),
                    "q_ego": o.q_ego,
                    "p_ego": o.p_ego,
                    "c_ego": o.c_ego,
                    "ego_i": o.ego_i,
                    "uav_i": o.uav_i,
                    "init_i": o.init_i,
                    "box": None if o.box is None else o.box.tolist(),
                }
            )
        return {
            "theta_p": self.theta_p,
            "q_h": self.q_h,
            "q_l": self.q_l,
            "n_certain": self.n_certain,
            "n_ambiguous": self.n_ambiguous,
            "n_tentative": self.n_tentative,
            "objects": rows,
        }


def _src_pc(src: Any, i: int):
    scores = np.asarray(src.scores)
    confs = np.asarray(src.confidences)
    p = float(scores[i]) if i < len(scores) else 0.0
    c = float(confs[i]) if i < len(confs) else 0.0
    return p, c


def gate_frame(
    ego: Any,
    uav: Any,
    init: Any,
    theta_p: float = THETA_P,
    theta_iou: float = THETA_IOU,
    q_h: float = Q_H,
    q_l: float = Q_L,
    exclude_ego=None,
    exclude_uav=None,
    exclude_init=None,
) -> FrameGating:
    """Associate three sources, apply §5.1 spatial gating, then §6.3.

    Ego-anchored objects: D_ego=1; state from Q_ego, then Ambiguous
    is resolved by Dual=(D_uav=1 and D_init=1) → Certain, else Tentative.
    UAV/Init leftovers (no Ego match): D_ego=0 → Tentative.

    Pool-matched detections (scheme §4.3.0) are excluded via exclude_*.
    """
    ei = _keep_detected(ego, theta_p, exclude=exclude_ego)
    ui = _keep_detected(uav, theta_p, exclude=exclude_uav)
    ii = _keep_detected(init, theta_p, exclude=exclude_init)

    ego_boxes = ego.boxes[ei] if len(ei) else np.zeros((0, 7))
    uav_boxes = uav.boxes[ui] if len(ui) else np.zeros((0, 7))
    init_boxes = init.boxes[ii] if len(ii) else np.zeros((0, 7))

    ego_uav = {a: (b, iou) for a, b, iou in hungarian_match(ego_boxes, uav_boxes, theta_iou)}
    ego_init = {a: (b, iou) for a, b, iou in hungarian_match(ego_boxes, init_boxes, theta_iou)}
    used_uav = {b for b, _ in ego_uav.values()}
    used_init = {b for b, _ in ego_init.values()}

    objects: List[GatedObject] = []

    for a in range(len(ei)):
        gi = int(ei[a])
        p, c = _src_pc(ego, gi)
        q = detection_quality(p, c)
        ju, iou_u = ego_uav.get(a, (-1, 0.0))
        ji, iou_i = ego_init.get(a, (-1, 0.0))
        d_uav = 1 if ju >= 0 else 0
        d_init = 1 if ji >= 0 else 0
        dual = 1 if dual_consensus(d_uav, d_init) else 0
        uav_i = int(ui[ju]) if d_uav else -1
        init_i = int(ii[ji]) if d_init else -1
        st0 = spatial_state(1, q, q_h=q_h, q_l=q_l)
        from_amb = st0 == AMBIGUOUS
        st = resolve_ambiguous(st0, d_uav, d_init)
        pu, cu = _src_pc(uav, uav_i) if d_uav else (0.0, 0.0)
        pi, ci = _src_pc(init, init_i) if d_init else (0.0, 0.0)
        objects.append(
            GatedObject(
                state=st,
                d_ego=1,
                d_uav=d_uav,
                d_init=d_init,
                q_ego=q,
                p_ego=p,
                c_ego=c,
                ego_i=gi,
                uav_i=uav_i,
                init_i=init_i,
                iou_uav=float(iou_u),
                iou_init=float(iou_i),
                dual=dual,
                from_ambiguous=from_amb,
                p_uav=pu,
                c_uav=cu,
                p_init=pi,
                c_init=ci,
                box=np.asarray(ego.boxes[gi], dtype=np.float64),
            )
        )

    leftover_u = [b for b in range(len(ui)) if b not in used_uav]
    leftover_i = [c for c in range(len(ii)) if c not in used_init]
    uav_left = uav_boxes[leftover_u] if leftover_u else np.zeros((0, 7))
    init_left = init_boxes[leftover_i] if leftover_i else np.zeros((0, 7))
    u2i = {a: (b, iou) for a, b, iou in hungarian_match(uav_left, init_left, theta_iou)}
    used_i2 = {b for b, _ in u2i.values()}

    for local_u, b in enumerate(leftover_u):
        gi = int(ui[b])
        lj, iou_i = u2i.get(local_u, (-1, 0.0))
        d_init = 1 if lj >= 0 else 0
        init_i = int(ii[leftover_i[lj]]) if d_init else -1
        pu, cu = _src_pc(uav, gi)
        pi, ci = _src_pc(init, init_i) if d_init else (0.0, 0.0)
        objects.append(
            GatedObject(
                state=TENTATIVE,
                d_ego=0,
                d_uav=1,
                d_init=d_init,
                q_ego=0.0,
                p_ego=0.0,
                c_ego=0.0,
                uav_i=gi,
                init_i=init_i,
                iou_init=float(iou_i),
                p_uav=pu,
                c_uav=cu,
                p_init=pi,
                c_init=ci,
                box=np.asarray(uav.boxes[gi], dtype=np.float64),
            )
        )

    for local_i, cidx in enumerate(leftover_i):
        if local_i in used_i2:
            continue
        gi = int(ii[cidx])
        pi, ci = _src_pc(init, gi)
        objects.append(
            GatedObject(
                state=TENTATIVE,
                d_ego=0,
                d_uav=0,
                d_init=1,
                q_ego=0.0,
                p_ego=0.0,
                c_ego=0.0,
                init_i=gi,
                p_init=pi,
                c_init=ci,
                box=np.asarray(init.boxes[gi], dtype=np.float64),
            )
        )

    objects.sort(key=lambda o: (-int(o.d_ego), -o.q_ego, -o.p_ego))
    return FrameGating(
        objects=objects,
        theta_p=theta_p,
        q_h=q_h,
        q_l=q_l,
    )
