"""Confirmed-object trust pool: KF coast + source-split evidence (scheme §4.3).

Per-object first split: match pool → pool rules only; else §5–§7.
Birth: any this-frame confirmed output (Certain / Ambiguous Dual / Tentative confirm).

Unmatched → ATTACK when the UAV-observation callback (or ego-ray fallback)
says the object should be visible but nothing detected it; coast KF into Ŷ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set

import numpy as np

from .associate import THETA_DIST, THETA_IOU, hungarian_match_pool, same_object
from .buffer import match_gt_tag
from .confidence import N_REF_EGO, R0, R_MIN
from .gating import (
    CERTAIN,
    FORCE_CERTAIN_GT_ID,
    Q_H,
    THETA_P,
    box_matches_gt_id,
)
from .kf import birth_state, kf_predict, kf_update
from .occlusion import K_ATK, V_MIN, visibility_ratio

# Max frames a track may coast without a match when occluded.
POOL_K = 3

UPDATE = "update"  # Certain Ego → KF update
HOLD = "hold"      # Ego/Fusion match, no KF update
MISS = "miss"
DROP = "drop"
BIRTH = "birth"
ATTACK = "attack"  # unmatched and not occluded: remove, coast predict

DUMP_POOL = "pool"
DUMP_BIRTH = "certain"  # first frame a track enters the pool
DUMP_ATTACK = "attack"  # unmatched visible: remove detected, coast box


def _as_box(box) -> Optional[np.ndarray]:
    if box is None:
        return None
    a = np.asarray(box, dtype=np.float64).reshape(-1)
    if a.size < 7:
        return None
    return a[:7].copy()


def _as_boxes(boxes) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 7), dtype=np.float64)
    a = np.asarray(boxes, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return a[:, :7]


def _keep_detected(src: Any, theta_p: float) -> np.ndarray:
    n = int(np.asarray(src.boxes).shape[0]) if getattr(src, "boxes", None) is not None else 0
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.flatnonzero(np.asarray(src.scores) >= float(theta_p)).astype(np.int32)


def _is_self(box, gt_lists) -> bool:
    for gt_boxes, gt_ids in gt_lists or ():
        if box_matches_gt_id(box, gt_boxes, gt_ids, oid=FORCE_CERTAIN_GT_ID):
            return True
    return False


@dataclass
class TrustTrack:
    """One confirmed object with its own KF."""

    tid: int
    x: np.ndarray
    P: np.ndarray
    box: np.ndarray
    age: int = 0  # consecutive misses; hit (update/hold) does not increment
    last_c_ego: float = 0.0
    updated: bool = False
    action: str = MISS
    iou: float = 0.0
    ego_i: int = -1
    uav_i: int = -1
    init_i: int = -1
    v: float = 1.0

    def pred_box(self) -> np.ndarray:
        b = self.box.copy()
        b[0] = float(self.x[0])
        b[1] = float(self.x[1])
        return b


@dataclass
class PoolRecord:
    tid: int
    action: str
    iou: float = 0.0
    iou_gate: float = THETA_IOU
    dist_gate: float = THETA_DIST
    age: int = 0
    k_max: int = POOL_K
    box: Optional[np.ndarray] = None
    ego_i: int = -1
    uav_i: int = -1
    init_i: int = -1
    gt_tag: str = ""
    v: float = 1.0

    def dump_line(self, k: int) -> str:
        b = self.box if self.box is not None else np.zeros(7)
        return (
            "  #{:<2d}  {:<8s}  tid={:<3d}  IoU={:.2f}  v={:.2f}  θ_iou={:.2f}  "
            "θ_d={:.1f}m  age={}/{}  idx=(e{} u{} i{})  "
            "B=[{:7.2f} {:7.2f} {:6.2f}  {:5.2f} {:5.2f} {:5.2f} {:6.3f}]{}".format(
                k,
                self.action,
                self.tid,
                self.iou,
                self.v,
                self.iou_gate,
                self.dist_gate,
                self.age,
                self.k_max,
                self.ego_i,
                self.uav_i,
                self.init_i,
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
                b[6],
                self.gt_tag or "",
            )
        )

    def to_dict(self) -> dict:
        return {
            "tid": self.tid,
            "action": self.action,
            "iou": self.iou,
            "iou_gate": self.iou_gate,
            "dist_gate": self.dist_gate,
            "age": self.age,
            "k_max": self.k_max,
            "ego_i": self.ego_i,
            "uav_i": self.uav_i,
            "init_i": self.init_i,
            "gt_tag": self.gt_tag,
            "v": self.v,
            "box": None if self.box is None else np.asarray(self.box).tolist(),
        }


@dataclass
class FramePool:
    """One-frame trust-pool snapshot."""

    records: List[PoolRecord] = field(default_factory=list)
    output_boxes: Optional[np.ndarray] = None
    used_ego: Set[int] = field(default_factory=set)
    used_uav: Set[int] = field(default_factory=set)
    used_init: Set[int] = field(default_factory=set)
    labels_ego: dict = field(default_factory=dict)
    labels_uav: dict = field(default_factory=dict)
    labels_init: dict = field(default_factory=dict)
    n_track: int = 0
    n_output: int = 0
    n_update: int = 0
    n_hold: int = 0
    n_drop: int = 0
    n_birth: int = 0
    n_attack: int = 0
    theta_iou: float = THETA_IOU
    theta_d: float = THETA_DIST
    k_max: int = POOL_K
    k_atk: int = K_ATK
    v_min: float = V_MIN

    def summary(self) -> str:
        return (
            "pool §4.3 | tracks={:>3} update={:>3} hold={:>3} drop={:>3} "
            "birth={:>3} attack={:>3} | θ_iou={:.2f} θ_d={:.1f}m k={} k_atk={} v_min={:.2f}".format(
                self.n_track,
                self.n_update,
                self.n_hold,
                self.n_drop,
                self.n_birth,
                self.n_attack,
                self.theta_iou,
                self.theta_d,
                self.k_max,
                self.k_atk,
                self.v_min,
            )
        )

    def dump_lines(self) -> List[str]:
        if not self.records:
            return ["  (none)"]
        return [r.dump_line(k) for k, r in enumerate(self.records)]

    def source_state_maps(self) -> dict:
        return {
            "ego": dict(self.labels_ego),
            "uav": dict(self.labels_uav),
            "init": dict(self.labels_init),
        }

    def claimed_object(self, obj) -> bool:
        ei = int(getattr(obj, "ego_i", -1))
        ui = int(getattr(obj, "uav_i", -1))
        ii = int(getattr(obj, "init_i", -1))
        if ei >= 0 and ei in self.used_ego:
            return True
        if ui >= 0 and ui in self.used_uav:
            return True
        if ii >= 0 and ii in self.used_init:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "n_track": self.n_track,
            "n_output": self.n_output,
            "n_update": self.n_update,
            "n_hold": self.n_hold,
            "n_drop": self.n_drop,
            "n_birth": self.n_birth,
            "n_attack": self.n_attack,
            "theta_iou": self.theta_iou,
            "theta_d": self.theta_d,
            "k_max": self.k_max,
            "k_atk": self.k_atk,
            "v_min": self.v_min,
            "used_ego": sorted(self.used_ego),
            "used_uav": sorted(self.used_uav),
            "used_init": sorted(self.used_init),
            "output_boxes": None
            if self.output_boxes is None
            else np.asarray(self.output_boxes).tolist(),
            "records": [r.to_dict() for r in self.records],
        }


class TrustPool:
    """Cross-frame confirmed-object tracks (scheme §4.3)."""

    def __init__(
        self,
        theta_iou: float = THETA_IOU,
        theta_p: float = THETA_P,
        q_h: float = Q_H,
        n_ref_ego: float = N_REF_EGO,
        r0: float = R0,
        r_min: float = R_MIN,
        k_max: int = POOL_K,
        k_atk: int = K_ATK,
        v_min: float = V_MIN,
        theta_d: float = THETA_DIST,
    ):
        self.theta_iou = float(theta_iou)
        self.theta_d = float(theta_d)
        self.theta_p = float(theta_p)
        self.q_h = float(q_h)
        self.n_ref_ego = float(n_ref_ego)
        self.r0 = float(r0)
        self.r_min = float(r_min)
        self.k_max = int(k_max)
        self.k_atk = int(k_atk)
        self.v_min = float(v_min)
        self.reset()

    def reset(self) -> None:
        self.tracks: List[TrustTrack] = []
        self._next_id = 1
        self._last_snap: Optional[FramePool] = None

    def living_boxes(self) -> np.ndarray:
        rows = [t.box[:7] for t in self.tracks if t.action != DROP and t.box is not None]
        if not rows:
            return np.zeros((0, 7), dtype=np.float64)
        return np.stack(rows)

    def _birth_track(self, box: np.ndarray, c_ego: float = 0.0) -> TrustTrack:
        b = _as_box(box)
        x, P = birth_state(b)
        t = TrustTrack(
            tid=self._next_id,
            x=x,
            P=P,
            box=b,
            age=0,
            last_c_ego=float(c_ego),
            updated=True,
            action=BIRTH,
        )
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _apply_meas(self, t: TrustTrack, box: np.ndarray) -> None:
        b = _as_box(box)
        t.x, t.P = kf_update(t.x, t.P, np.array([b[0], b[1]], dtype=np.float64))
        t.box = b.copy()
        t.box[0] = float(t.x[0])
        t.box[1] = float(t.x[1])
        t.updated = True
        t.age = 0

    def _label(self, src_name: str, i: int, tag: str = DUMP_POOL) -> None:
        snap = self._last_snap
        if snap is None or i < 0:
            return
        tag = str(tag)
        if src_name == "ego":
            snap.labels_ego[int(i)] = tag
            snap.used_ego.add(int(i))
        elif src_name == "uav":
            snap.labels_uav[int(i)] = tag
            snap.used_uav.add(int(i))
        else:
            snap.labels_init[int(i)] = tag
            snap.used_init.add(int(i))

    def _claim_all(
        self, box, src, src_name: str, iou_thr: float, dist_thr: float, tag: str = DUMP_POOL
    ) -> None:
        if src is None or box is None or getattr(src, "boxes", None) is None:
            return
        boxes = np.asarray(src.boxes)
        if boxes.ndim != 2:
            return
        for ji in range(int(boxes.shape[0])):
            if same_object(box, boxes[ji], iou_thr, dist_thr):
                self._label(src_name, ji, tag)

    def _stamp_track(self, t: TrustTrack, tag: str, ego: Any, uav: Any, init: Any) -> None:
        if t.ego_i >= 0:
            self._label("ego", t.ego_i, tag)
        if t.uav_i >= 0:
            self._label("uav", t.uav_i, tag)
        if t.init_i >= 0:
            self._label("init", t.init_i, tag)
        self._claim_all(t.box, ego, "ego", self.theta_iou, self.theta_d, tag)
        self._claim_all(t.box, uav, "uav", self.theta_iou, self.theta_d, tag)
        self._claim_all(t.box, init, "init", self.theta_iou, self.theta_d, tag)

    def label_living_sources(self, ego: Any, uav: Any, init: Any) -> None:
        if self._last_snap is None:
            return
        living = [t for t in self.tracks if t.action != DROP]
        for t in living:
            if t.action == ATTACK:
                # Claim nearby source dets as "attack"; coast box still in output_boxes.
                self._stamp_track(t, DUMP_ATTACK, ego, uav, init)
            elif t.action != BIRTH:
                self._stamp_track(t, DUMP_POOL, ego, uav, init)
        for t in living:
            if t.action == BIRTH:
                self._stamp_track(t, DUMP_BIRTH, ego, uav, init)

    def attack_boxes(self) -> np.ndarray:
        """KF coast boxes for this-frame ATTACK tracks (for dump / Ŷ)."""
        rows = [
            np.asarray(t.pred_box(), dtype=np.float64)[:7]
            for t in self.tracks
            if t.action == ATTACK and t.box is not None
        ]
        if not rows:
            return np.zeros((0, 7), dtype=np.float64)
        return np.stack(rows)

    def _record(self, t: TrustTrack, gt_boxes, gt_ids) -> PoolRecord:
        k_lim = self.k_atk if t.action == ATTACK else self.k_max
        rec = PoolRecord(
            tid=t.tid,
            action=t.action,
            iou=float(t.iou),
            iou_gate=self.theta_iou,
            dist_gate=self.theta_d,
            age=int(t.age),
            k_max=k_lim,
            box=t.box.copy(),
            ego_i=int(t.ego_i),
            uav_i=int(t.uav_i),
            init_i=int(t.init_i),
            v=float(getattr(t, "v", 1.0)),
        )
        if t.box is not None:
            rec.gt_tag = match_gt_tag(t.box, gt_boxes, gt_ids, iou_thres=self.theta_iou)
        return rec

    def _match_dets(self, pred_boxes, boxes, src_idx, used: Set[int]):
        """Hungarian tracks vs a subset of detections. src_idx[j] is original index."""
        n_t = len(self.tracks)
        hit = [None] * n_t
        if n_t == 0 or boxes is None or len(src_idx) == 0:
            return hit
        dets = np.asarray(boxes, dtype=np.float64)
        if dets.ndim != 2 or dets.shape[0] == 0:
            return hit
        keep_j = [j for j, si in enumerate(src_idx) if int(si) not in used]
        if not keep_j:
            return hit
        sub = dets[np.asarray(keep_j, dtype=np.int32)]
        orig = [int(src_idx[j]) for j in keep_j]
        for ti, dj, iou in hungarian_match_pool(
            pred_boxes, sub, iou_thres=self.theta_iou, dist_thres=self.theta_d
        ):
            hit[int(ti)] = (orig[int(dj)], float(iou))
        return hit

    def _occluders_for(self, ti: int, extra) -> np.ndarray:
        rows = []
        for j, t in enumerate(self.tracks):
            if j == ti or t.box is None:
                continue
            rows.append(np.asarray(t.pred_box(), dtype=np.float64)[:7])
        extra = np.asarray(extra) if extra is not None else np.zeros((0, 7))
        if extra.ndim == 2 and extra.shape[0] > 0:
            rows.extend(np.asarray(extra, dtype=np.float64)[:, :7])
        if not rows:
            return np.zeros((0, 7), dtype=np.float64)
        return np.stack(rows)

    def step(
        self,
        gating: Any,
        ego: Any,
        uav: Any,
        init: Any,
        gt_boxes=None,
        gt_ids=None,
        gt_lists=None,
        occluders=None,
        uav_sight=None,
    ) -> FramePool:
        """Predict → Certain-Ego update / birth → other Ego/Fusion hold.

        Unmatched → ATTACK iff the object is "seeable but not detected".
        By default seeable = ego-ray visibility v >= v_min (ego origin).
        If uav_sight(box, occluders) is given, that callback decides (caller
        ORs UAV line-of-sight from the UAV origin with UAV point density, so an
        ego miss caused by ego occlusion no longer forces MISS and a pure
        point-erasing attack still counts as ATTACK). Otherwise → miss/drop.
        """
        lists = gt_lists or ((gt_boxes, gt_ids),)
        snap = FramePool(
            theta_iou=self.theta_iou,
            theta_d=self.theta_d,
            k_max=self.k_max,
            k_atk=self.k_atk,
            v_min=self.v_min,
        )
        self._last_snap = snap
        records = snap.records
        outputs: List[np.ndarray] = []
        n_update = n_hold = n_drop = n_birth = n_attack = 0

        keep: List[TrustTrack] = []
        for t in self.tracks:
            if _is_self(t.box, lists):
                t.action = DROP
                n_drop += 1
                records.append(self._record(t, gt_boxes, gt_ids))
            else:
                keep.append(t)
        self.tracks = keep

        for t in self.tracks:
            t.x, t.P = kf_predict(t.x, t.P)
            t.box[0] = float(t.x[0])
            t.box[1] = float(t.x[1])
            t.updated = False
            t.action = MISS
            t.iou = 0.0
            t.ego_i = t.uav_i = t.init_i = -1
            t.v = 1.0

        pred_boxes = (
            np.stack([t.pred_box() for t in self.tracks])
            if self.tracks
            else np.zeros((0, 7), dtype=np.float64)
        )

        certain_obj = []
        other_ego_i: List[int] = []
        other_init_i: List[int] = []
        other_uav_i: List[int] = []
        for o in getattr(gating, "objects", None) or []:
            ei = int(getattr(o, "ego_i", -1))
            ii = int(getattr(o, "init_i", -1))
            ui = int(getattr(o, "uav_i", -1))
            if getattr(o, "from_far", False):
                continue
            if getattr(o, "state", None) == CERTAIN and ei >= 0:
                box = np.asarray(ego.boxes[ei], dtype=np.float64)[:7]
                if _is_self(box, lists):
                    continue
                certain_obj.append((ei, box, float(getattr(o, "c_ego", 0.0) or 0.0)))
            else:
                if ei >= 0:
                    other_ego_i.append(ei)
                if ii >= 0:
                    other_init_i.append(ii)
                if ui >= 0:
                    other_uav_i.append(ui)

        used_certain: Set[int] = set()
        if certain_obj and self.tracks:
            c_boxes = np.stack([b for _, b, _ in certain_obj])
            c_idx = [ei for ei, _, _ in certain_obj]
            ego_hit = self._match_dets(pred_boxes, c_boxes, c_idx, used_certain)
        else:
            ego_hit = [None] * len(self.tracks)

        claimed_t: Set[int] = set()
        for ti, t in enumerate(self.tracks):
            h = ego_hit[ti] if ti < len(ego_hit) else None
            if h is None:
                continue
            ei, iou = h
            meas = np.asarray(ego.boxes[ei], dtype=np.float64)
            if _is_self(meas, lists):
                continue
            t.ego_i = int(ei)
            t.iou = float(iou)
            t.last_c_ego = float(ego.confidences[ei]) if ei < len(ego.confidences) else 0.0
            self._apply_meas(t, meas)
            t.action = UPDATE
            claimed_t.add(ti)
            used_certain.add(int(ei))
            snap.used_ego.add(int(ei))
            n_update += 1
            outputs.append(t.box.copy())
            records.append(self._record(t, gt_boxes, gt_ids))

        used_certain_set = {ei for ei, _, _ in certain_obj if ei in used_certain}
        for ei, box, c_ego in certain_obj:
            if ei in used_certain_set:
                continue
            t = self._birth_track(box, c_ego=c_ego)
            t.ego_i = int(ei)
            n_birth += 1
            snap.used_ego.add(int(ei))
            outputs.append(t.box.copy())
            records.append(self._record(t, gt_boxes, gt_ids))

        # Other-state Ego / Fusion: keep track, no KF update.
        rest = [i for i in range(len(self.tracks)) if i not in claimed_t and self.tracks[i].action != BIRTH]
        if rest:
            rest_pred = np.stack([self.tracks[i].pred_box() for i in rest])

            def _subset_hit(src, idxs, used):
                hit = [None] * len(rest)
                if src is None or not idxs:
                    return hit
                boxes = np.asarray(src.boxes)
                keep_j = [j for j, si in enumerate(idxs) if int(si) not in used]
                if not keep_j:
                    return hit
                sub = boxes[np.asarray([idxs[j] for j in keep_j], dtype=np.int32)]
                orig = [int(idxs[j]) for j in keep_j]
                for ti, dj, iou in hungarian_match_pool(
                    rest_pred, sub, iou_thres=self.theta_iou, dist_thres=self.theta_d
                ):
                    hit[int(ti)] = (orig[int(dj)], float(iou))
                return hit

            e_hit = _subset_hit(ego, other_ego_i, set(snap.used_ego))
            i_hit = _subset_hit(init, other_init_i, set(snap.used_init))
            u_hit = _subset_hit(uav, other_uav_i, set(snap.used_uav))

            for local, ti in enumerate(rest):
                t = self.tracks[ti]
                eh, ih, uh = e_hit[local], i_hit[local], u_hit[local]
                if eh is not None:
                    t.ego_i, t.iou = eh
                    snap.used_ego.add(int(t.ego_i))
                if ih is not None:
                    t.init_i, i_iou = ih
                    snap.used_init.add(int(t.init_i))
                    if eh is None:
                        t.iou = i_iou
                if uh is not None:
                    t.uav_i, u_iou = uh
                    snap.used_uav.add(int(t.uav_i))
                    if eh is None and ih is None:
                        t.iou = u_iou
                hit = eh is not None or ih is not None or uh is not None
                if hit:
                    t.action = HOLD
                    t.age = 0
                    n_hold += 1
                    outputs.append(t.box.copy())
                else:
                    occ = self._occluders_for(ti, occluders)
                    if uav_sight is not None:
                        # UAV-observation rule: "clear" means either the UAV's
                        # own line of sight is unblocked, or enough UAV points
                        # sit at the predicted location. 0..1 like v.
                        try:
                            t.v = float(uav_sight(t.pred_box(), occ))
                        except Exception:
                            t.v = 0.0
                    else:
                        t.v = visibility_ratio(t.pred_box(), occ)
                    t.age = int(t.age) + 1
                    if t.v >= self.v_min:
                        t.action = ATTACK
                        if t.age >= self.k_atk:
                            t.action = DROP
                            n_drop += 1
                            records.append(self._record(t, gt_boxes, gt_ids))
                            continue
                        n_attack += 1
                        outputs.append(t.pred_box())
                    else:
                        if t.age >= self.k_max:
                            t.action = DROP
                            n_drop += 1
                            records.append(self._record(t, gt_boxes, gt_ids))
                            continue
                        t.action = MISS
                records.append(self._record(t, gt_boxes, gt_ids))

        alive: List[TrustTrack] = []
        for t in self.tracks:
            if t.action == DROP:
                continue
            alive.append(t)
        self.tracks = alive

        out = np.stack(outputs) if outputs else np.zeros((0, 7), dtype=np.float64)
        snap.output_boxes = out
        snap.n_track = len(self.tracks)
        snap.n_output = int(out.shape[0])
        snap.n_update = n_update
        snap.n_hold = n_hold
        snap.n_drop = n_drop
        snap.n_birth = n_birth
        snap.n_attack = n_attack
        return snap

    def ingest(
        self,
        boxes,
        c_egos: Optional[Sequence[float]] = None,
        allow_update: Optional[Sequence[bool]] = None,
    ) -> int:
        """Buffer-confirm boxes: birth if new. Never KF-update (only Certain does)."""
        del allow_update  # Certain-only update is handled in step()
        new = _as_boxes(boxes)
        if new.shape[0] == 0:
            return 0
        cs = list(c_egos) if c_egos is not None else [0.0] * int(new.shape[0])
        while len(cs) < int(new.shape[0]):
            cs.append(0.0)
        n_birth = 0
        used_new: Set[int] = set()
        if self.tracks:
            tboxes = np.stack([t.box for t in self.tracks])
            for _ti, bj, _iou in hungarian_match_pool(
                tboxes, new, iou_thres=self.theta_iou, dist_thres=self.theta_d
            ):
                used_new.add(int(bj))
        for j in range(int(new.shape[0])):
            if j in used_new:
                continue
            t = self._birth_track(new[j], c_ego=float(cs[j]))
            n_birth += 1
            if self._last_snap is not None:
                self._last_snap.records.append(
                    PoolRecord(
                        tid=t.tid,
                        action=BIRTH,
                        iou_gate=self.theta_iou,
                        dist_gate=self.theta_d,
                        age=0,
                        k_max=self.k_max,
                        box=t.box.copy(),
                    )
                )
        if self._last_snap is not None:
            self._last_snap.n_birth += n_birth
            self._last_snap.n_track = len(self.tracks)
        return n_birth
