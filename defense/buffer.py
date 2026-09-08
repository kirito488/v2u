"""Tentative buffer: init, Bayesian update, confirm / reject / timeout (scheme §7).

No global W_trust. P0 uses the timeout midpoint 0.5 as a prior (scheme §7.2).
Spatial evidence is Ego-only:
  hit  → +κ_pos · ψ(Q_buf) · w(v),  w(v)=1+σ((0.5−v)/τ)
  miss → −κ_neg · v
where Q_buf = P·C_abs (not P·C). Full C = C_abs·V is for gating Q=P·C;
buffer keeps geometric v in w(v)/e_- so V is not multiplied twice into ψ.

Hard P≥score_thres still gates Certain/pool. Buffer may also match soft Ego
(θ_soft ≤ P < score_thres) to existing watches; soft Ego never births a watch
and never enters Certain. UAV/Init never add positive evidence.

New watches are leftover-only (scheme §4.3.0). If an existing watch matches a
gate/pool Certain this frame, drop it silently (not buffer confirm); hist is kept
for dump backfill only.

Each watch has a KF. Unmatched + ray-visible → ATTACK (label only; do not
emit into Ŷ — Tentative is still uncertain). Pool ATTACK still coasts.

Visibility `v` defaults to box-ray occluders; pass `visibility_fn(box)->float`
(e.g. LiDAR polar depth from ego) to override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from .associate import THETA_DIST, THETA_IOU, hungarian_match_pool, same_object
from .gating import CERTAIN, Q_L, TENTATIVE, detection_quality, traj_ref_box
from .geometry import iou_bev
from .kf import birth_state, kf_predict, kf_update
from .occlusion import K_ATK, V_MIN, visibility_ratio

P_MIN = 0.15
THETA_SOFT = 0.05  # buffer-only Ego band; gating/pool still use score_thres
THETA_CONFIRM = 0.7
THETA_REJECT = 0.1
T_TIMEOUT = 10
KAPPA_POS = 1.4
KAPPA_NEG = 1.0
TAU = 0.1
V_PIVOT = 0.5  # same midpoint as TIMEOUT / P0; reused by w(v), not a new knob
EPS = 1e-6

WATCH = "watch"
CONFIRM = "confirm"
REJECT = "reject"
TIMEOUT = "timeout"
ATTACK = "attack"


def _clip01(x, lo=EPS, hi=1.0 - EPS) -> float:
    return float(np.clip(x, lo, hi))


def sigmoid(x: float) -> float:
    z = float(np.clip(x, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def phi_ego(q_ego: float, q_ref: float = Q_L, tau: float = TAU) -> float:
    """φ(Q_ego) = σ((Q_l - Q_ego) / τ)."""
    return sigmoid((float(q_ref) - float(q_ego)) / float(tau))


def psi_ego(q_ego: float, q_ref: float = Q_L, tau: float = TAU) -> float:
    """ψ(Q) = 1 - φ(Q) = σ((Q - Q_l) / τ).

    Buffer passes Q = P·C_abs (see obj_q_buf_ego). Do not feed P·C_abs·V here.
    """
    return 1.0 - phi_ego(q_ego, q_ref=q_ref, tau=tau)


def occ_boost(v: float, tau: float = TAU, v0: float = V_PIVOT) -> float:
    """Occlusion boost on a positive Ego hit: w(v)=1+σ((v0−v)/τ) ∈ (1, 2).

    Same τ as ψ so the slope near the midpoint is comparable (linear 2−v is too
    flat). v0=0.5 reuses the buffer pivot — not a separately tuned parameter.
    Open (v≈1) → w≈1; occluded (v≈0) → w≈2.
    """
    return 1.0 + sigmoid((float(v0) - float(np.clip(v, 0.0, 1.0))) / float(tau))


def obj_q_ego(obj) -> float:
    """Gate Q_ego = P·C (full C); used only if buffer abs-Q unavailable."""
    return float(getattr(obj, "q_ego", 0.0) or 0.0)


def obj_q_buf_ego(obj, mode: str = "pc_abs") -> float:
    """Tentative evidence Q.

    pc_abs (default): P·C_abs  (V stays in w(v)/e_-)
    p: P only (ablation: drop C from buffer evidence)
    """
    p = float(getattr(obj, "p_ego", 0.0) or 0.0)
    m = str(mode or "pc_abs").lower()
    if m in ("p", "p_only", "score"):
        return float(p)
    ca = getattr(obj, "c_abs_ego", None)
    if ca is not None:
        return float(p) * float(ca)
    c = float(getattr(obj, "c_ego", 0.0) or 0.0)
    return float(p) * float(c)


def obj_q_uav(obj) -> float:
    if int(getattr(obj, "d_uav", 0)):
        return detection_quality(obj.p_uav, obj.c_uav)
    return 0.0


def obj_q_init(obj) -> float:
    if int(getattr(obj, "d_init", 0)):
        return detection_quality(obj.p_init, obj.c_init)
    return 0.0


def obj_q_collab(obj) -> float:
    """Q of the collab source used in g_uav ratio (UAV preferred, else Init)."""
    q_u = obj_q_uav(obj)
    if q_u > 0.0:
        return q_u
    return obj_q_init(obj)


def g_uav(q_collab: float, q_ego: float) -> float:
    denom = float(q_ego) + float(q_collab)
    ratio = 1.0 if denom <= 0 else float(q_collab) / denom
    return ratio * phi_ego(q_ego)


def strength(q: float, q_ref: float = Q_L, p_min: float = P_MIN) -> float:
    if q_ref <= 0:
        return float(p_min)
    return float(np.clip(float(q) / float(q_ref), p_min, 1.0))


def logit(p: float) -> float:
    p = _clip01(p)
    return float(np.log(p / (1.0 - p)))


def inv_logit(z: float) -> float:
    return _clip01(sigmoid(z))


def source_q(obj) -> float:
    if int(getattr(obj, "d_uav", 0)):
        return detection_quality(obj.p_uav, obj.c_uav)
    if int(getattr(obj, "d_init", 0)):
        return detection_quality(obj.p_init, obj.c_init)
    return float(getattr(obj, "q_ego", 0.0) or 0.0)


def spatial_evidence(
    ego_hit: bool,
    q_ego: float,
    v: float = 1.0,
    kappa_pos: float = KAPPA_POS,
    kappa_neg: float = KAPPA_NEG,
) -> float:
    """Spatial log-odds increment (Ego-only).

    Ego hit (hard or soft): +κ_pos · ψ(Q) · w(v),  Q = P·C_abs.
    Ego miss (unmatched, or matched UAV/Init only): −κ_neg · v.
    """
    vv = float(np.clip(v, 0.0, 1.0))
    if ego_hit:
        return float(kappa_pos) * psi_ego(q_ego) * occ_boost(vv)
    return -float(kappa_neg) * vv


def _stack_boxes(items) -> tuple:
    """Return (N,7) boxes and original indices (skip missing)."""
    rows, idx = [], []
    for i, x in enumerate(items):
        b = getattr(x, "box", x)
        if b is None:
            continue
        rows.append(np.asarray(b, dtype=np.float64).reshape(-1)[:7])
        idx.append(i)
    if not rows:
        return np.zeros((0, 7), dtype=np.float64), []
    return np.stack(rows), idx


def match_gt_tag(
    box,
    gt_boxes,
    gt_ids,
    iou_thres: float = THETA_IOU,
    dist_thres: float = THETA_DIST,
) -> str:
    """'hit id=…' if IoU≥θ_iou OR dxy≤θ_d; else 'miss' (nearest GT)."""
    gts = np.asarray(gt_boxes) if gt_boxes is not None else np.zeros((0, 7))
    if box is None or gts.ndim != 2 or gts.shape[0] == 0:
        return ""
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
    ious = np.array([float(iou_bev(b, g)) for g in gts], dtype=np.float64)
    dxy = np.hypot(gts[:, 0] - b[0], gts[:, 1] - b[1])
    j_iou = int(np.argmax(ious))
    j_n = int(np.argmin(dxy))
    use = j_iou if float(ious[j_iou]) > float(ious[j_n]) + 1e-6 else j_n
    iou = float(ious[use])
    dist = float(dxy[use])
    ids = list(gt_ids or [])
    gid = ids[use] if use < len(ids) else use
    if iou >= float(iou_thres) or dist <= float(dist_thres):
        return "  hit id={} iou={:.2f} dxy={:.1f}m".format(gid, iou, dist)
    return "  miss id={} iou={:.2f} dxy={:.1f}m".format(gid, iou, dist)


@dataclass
class SoftEgoDet:
    """Low-score Ego box used only as buffer evidence (not gating/pool)."""

    box: np.ndarray
    q: float
    p: float = 0.0


@dataclass
class BufferEntry:
    box: np.ndarray
    p: float
    age: int = 1
    d_uav0: int = 0
    from_ambiguous: bool = False
    q_ego: float = 0.0
    q_uav: float = 0.0
    decision: str = WATCH
    evidence: float = 0.0
    e_temp: float = 0.0
    gt_tag: str = ""  # filled on confirm/reject/timeout
    hist: list = field(default_factory=list)  # (frame_id, box) while watching
    # Measurement traj for KF birth velocity: ego preferred, else fuse (init).
    # (frame_id, box); miss / pure-UAV frames omitted unless no ego/fuse ever seen.
    traj: list = field(default_factory=list)
    x: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    v: float = 1.0
    miss_age: int = 0

    def ensure_kf(self) -> None:
        if self.x is None or self.P is None:
            self.x, self.P = birth_state(self.box, traj=self.velocity_traj())

    def append_traj(self, frame_id: int, box, source: str = "") -> None:
        """Record traj point: (frame_id, box, source_tag)."""
        if box is None:
            return
        self.traj.append(
            (int(frame_id), np.asarray(box, dtype=np.float64)[:7].copy(), str(source or ""))
        )

    def velocity_traj(self) -> list:
        """Points for KF birth velocity: ego/fuse preferred; else uav/box/hist.

        Returns list of (frame_id, box).
        """
        rows = list(self.traj or [])
        pref = []
        for item in rows:
            if len(item) >= 3:
                fid, b, src = item[0], item[1], str(item[2])
            elif len(item) == 2:
                fid, b, src = item[0], item[1], ""
            else:
                continue
            if src in ("ego", "fuse"):
                pref.append((int(fid), b))
        if len(pref) >= 2:
            return pref
        # Fall back: any traj tag, then hist (may include KF coast).
        any_t = []
        for item in rows:
            if len(item) >= 2:
                any_t.append((int(item[0]), item[1]))
        if len(any_t) >= 2:
            return any_t
        return [(int(fid), b) for fid, b in (self.hist or [])]

    def snapshot(self) -> "BufferEntry":
        """Immutable copy for per-frame dump (entries are mutated across frames)."""
        return BufferEntry(
            box=np.asarray(self.box, dtype=np.float64).copy(),
            p=float(self.p),
            age=int(self.age),
            d_uav0=int(self.d_uav0),
            from_ambiguous=bool(self.from_ambiguous),
            q_ego=float(self.q_ego),
            q_uav=float(self.q_uav),
            decision=str(self.decision),
            evidence=float(self.evidence),
            e_temp=float(self.e_temp),
            gt_tag=str(self.gt_tag or ""),
            v=float(self.v),
            miss_age=int(self.miss_age),
        )

    def dump_line(self, k: int) -> str:
        b = np.asarray(self.box)
        src = "amb" if self.from_ambiguous else "direct"
        return (
            "  #{:<2d}  {:<8s}  P={:.3f}  Q_ego={:.3f}  Q_uav={:.3f}  "
            "age={:<2d}  miss={:<2d}  from={:<6s}  "
            "e={:+.2f}  e_temp={:+.2f}  V={:.3f}  Duav0={}  "
            "B=[{:7.2f} {:7.2f} {:6.2f}  {:5.2f} {:5.2f} {:5.2f} {:6.3f}]{}".format(
                k,
                self.decision,
                self.p,
                self.q_ego,
                self.q_uav,
                self.age,
                self.miss_age,
                src,
                self.evidence,
                self.e_temp,
                self.v,
                int(self.d_uav0),
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
            "decision": self.decision,
            "p": self.p,
            "age": self.age,
            "from_ambiguous": self.from_ambiguous,
            "d_uav0": int(self.d_uav0),
            "evidence": self.evidence,
            "e_temp": self.e_temp,
            "v": self.v,
            "gt_tag": self.gt_tag,
            "box": None if self.box is None else np.asarray(self.box).tolist(),
        }


@dataclass
class FrameBuffer:
    """One-frame buffer snapshot after §7 step."""

    watch: List[BufferEntry] = field(default_factory=list)
    events: List[BufferEntry] = field(default_factory=list)
    output_boxes: Optional[np.ndarray] = None
    n_confirm: int = 0
    n_reject: int = 0
    n_timeout: int = 0
    n_attack: int = 0

    @property
    def n_watch(self) -> int:
        return len(self.watch)

    @property
    def conservative(self) -> bool:
        return self.n_watch > 0

    def summary(self) -> str:
        return (
            "buf §7 | watch={:>3} confirm={:>3} reject={:>3} timeout={:>3} "
            "attack={:>3}".format(
                self.n_watch,
                self.n_confirm,
                self.n_reject,
                self.n_timeout,
                self.n_attack,
            )
        )

    def dump_lines(self) -> List[str]:
        rows = list(self.events) if self.events else list(self.watch)
        if not rows:
            return ["  (none)"]
        return [e.dump_line(k) for k, e in enumerate(rows)]

    def to_dict(self) -> dict:
        return {
            "n_watch": self.n_watch,
            "n_confirm": self.n_confirm,
            "n_reject": self.n_reject,
            "n_timeout": self.n_timeout,
            "n_attack": self.n_attack,
            "conservative": self.conservative,
            "output_boxes": None
            if self.output_boxes is None
            else np.asarray(self.output_boxes).tolist(),
            "events": [e.to_dict() for e in self.events],
        }


class TentativeBuffer:
    """Cross-frame Tentative buffer (no global W_trust)."""

    def __init__(
        self,
        q_l: float = Q_L,
        theta_iou: float = THETA_IOU,
        theta_d: float = THETA_DIST,
        theta_confirm: float = THETA_CONFIRM,
        theta_reject: float = THETA_REJECT,
        t_timeout: int = T_TIMEOUT,
        kappa_pos: float = KAPPA_POS,
        kappa_neg: float = KAPPA_NEG,
        p_min: float = P_MIN,
        k_atk: int = K_ATK,
        v_min: float = V_MIN,
        buf_q_mode: str = "pc_abs",
    ):
        self.q_l = float(q_l)
        self.theta_iou = float(theta_iou)
        self.theta_d = float(theta_d)
        self.theta_confirm = float(theta_confirm)
        self.theta_reject = float(theta_reject)
        self.t_timeout = int(t_timeout)
        self.kappa_pos = float(kappa_pos)
        self.kappa_neg = float(kappa_neg)
        self.p_min = float(p_min)
        self.k_atk = int(k_atk)
        self.v_min = float(v_min)
        self.buf_q_mode = str(buf_q_mode or "pc_abs")
        self.reset()

    def reset(self) -> None:
        self.entries: List[BufferEntry] = []
        self.prev_output = np.zeros((0, 7), dtype=np.float64)
        self.confirmed_hists: list = []
        self.gate_certain_hists: list = []
        self.last_certain_trajs: list = []
        self.last_confirm_trajs: list = []

    def _init_from_obj(
        self, obj, frame_id: int = 0, ego=None, uav=None, init=None
    ) -> BufferEntry:
        st = strength(source_q(obj), q_ref=self.q_l, p_min=self.p_min)
        q_ego = obj_q_buf_ego(obj, mode=self.buf_q_mode)
        q_collab = obj_q_collab(obj)
        # 0.5 is the same pivot as timeout (P <= 0.5), not a new threshold.
        half = 0.5
        if obj.from_ambiguous:
            gamma = q_ego / self.q_l if self.q_l > 0 else 1.0
            p0 = max(half * st * gamma, self.p_min)
        else:
            p0 = max(half * g_uav(q_collab, q_ego) * st, self.p_min)
        ref, src = traj_ref_box(obj, ego=ego, uav=uav, init=init)
        b = (
            np.asarray(ref, dtype=np.float64)[:7].copy()
            if ref is not None
            else np.asarray(obj.box, dtype=np.float64)[:7].copy()
        )
        traj = [(int(frame_id), b.copy(), src or "box")]
        x, P = birth_state(b, traj=[(int(frame_id), b.copy())])
        return BufferEntry(
            box=b,
            p=_clip01(p0),
            age=1,
            d_uav0=int(obj.d_uav),
            from_ambiguous=bool(obj.from_ambiguous),
            q_ego=q_ego,
            q_uav=obj_q_uav(obj),
            decision=WATCH,
            hist=[(int(frame_id), b.copy())],
            traj=traj,
            x=x,
            P=P,
        )

    def _update(
        self,
        entry: BufferEntry,
        ego_hit: bool,
        curr_box=None,
        frame_id: int = 0,
        v: float = 1.0,
    ) -> None:
        """Update P with Ego hit / visibility-weighted Ego miss; no e_temp."""
        ev = spatial_evidence(
            ego_hit,
            entry.q_ego,
            v=v,
            kappa_pos=self.kappa_pos,
            kappa_neg=self.kappa_neg,
        )
        if curr_box is not None:
            curr = np.asarray(curr_box, dtype=np.float64).copy()
            entry.box = curr[:7]
            entry.hist.append((int(frame_id), curr[:7].copy()))
        entry.evidence = ev
        entry.e_temp = 0.0
        entry.p = inv_logit(logit(entry.p) + ev)
        entry.age = int(entry.age) + 1

    def _kf_meas(self, entry: BufferEntry, curr_box) -> None:
        curr = np.asarray(curr_box, dtype=np.float64).reshape(-1)[:7]
        entry.ensure_kf()
        entry.x, entry.P = kf_update(
            entry.x, entry.P, np.array([curr[0], curr[1]], dtype=np.float64)
        )
        entry.box = curr.copy()
        entry.box[0] = float(entry.x[0])
        entry.box[1] = float(entry.x[1])

    def _decide(self, entry: BufferEntry) -> str:
        if entry.p >= self.theta_confirm:
            return CONFIRM
        if entry.p <= self.theta_reject:
            return REJECT
        if entry.age >= self.t_timeout and entry.p <= 0.5:
            return TIMEOUT
        return WATCH

    def step(
        self,
        gating: Any,
        gt_boxes=None,
        gt_ids=None,
        frame_id: int = 0,
        match_objects=None,
        occluders=None,
        pool_boxes=None,
        soft_ego=None,
        visibility_fn: Optional[Callable[[np.ndarray], float]] = None,
        ego=None,
        uav=None,
        init=None,
    ) -> FrameBuffer:
        """Consume this-frame leftover gating; return snapshot.

        New watches come only from leftover Tentative. Existing watches may match
        `match_objects` (full gate) to detect gate/pool Certain and exit silently.
        Unmatched watches may still match `soft_ego` (P in [θ_soft, score_thres))
        for a positive Ego update. Unmatched + ray-visible → ATTACK label only.
        Watches that match a living pool track exit silently (pool owns them).

        `visibility_fn`, when set, replaces box-occluder `visibility_ratio`
        (intended for ego LiDAR polar-depth occlusion).

        `ego` / `uav` / `init` SourceResults select traj boxes (ego > fuse > uav)
        for KF birth-velocity history.
        """
        objects = list(getattr(gating, "objects", None) or [])
        match_src = list(match_objects) if match_objects is not None else objects
        soft_src = list(soft_ego or [])
        old = list(self.entries)
        for entry in old:
            entry.ensure_kf()
            entry.x, entry.P = kf_predict(entry.x, entry.P)
            entry.box = np.asarray(entry.box, dtype=np.float64)[:7].copy()
            entry.box[0] = float(entry.x[0])
            entry.box[1] = float(entry.x[1])
            entry.v = 1.0
        old_boxes, old_idx = _stack_boxes(old)
        obj_boxes, obj_idx = _stack_boxes(match_src)
        matched = {}
        for i, j, _ in hungarian_match_pool(
            old_boxes, obj_boxes, iou_thres=self.theta_iou, dist_thres=self.theta_d
        ):
            matched[int(old_idx[i])] = int(obj_idx[j])
        used_ids = {id(match_src[oj]) for oj in matched.values()}

        rest = [int(i) for i in old_idx if int(i) not in matched]
        soft_hit: dict = {}
        if rest and soft_src:
            rest_boxes = np.stack(
                [np.asarray(old[i].box, dtype=np.float64)[:7] for i in rest]
            )
            s_boxes, s_idx = _stack_boxes(soft_src)
            for li, sj, _ in hungarian_match_pool(
                rest_boxes, s_boxes, iou_thres=self.theta_iou, dist_thres=self.theta_d
            ):
                soft_hit[int(rest[int(li)])] = int(s_idx[int(sj)])

        events: List[BufferEntry] = []
        keep: List[BufferEntry] = []
        confirmed: List[np.ndarray] = []
        confirms: List[BufferEntry] = []
        n_c = n_r = n_t = n_a = 0
        # Trajectories of watches that just became gate/pool Certain (for pool KF).
        self.last_certain_trajs: list = []
        self.last_confirm_trajs: list = []

        def finish(entry: BufferEntry, decision: str) -> None:
            nonlocal n_c, n_r, n_t, n_a
            entry.decision = decision
            if decision in (CONFIRM, REJECT, TIMEOUT, ATTACK):
                entry.gt_tag = match_gt_tag(
                    entry.box,
                    gt_boxes,
                    gt_ids,
                    iou_thres=self.theta_iou,
                    dist_thres=self.theta_d,
                )
            else:
                entry.gt_tag = ""
            # Snapshot: run_case finishes all frames before printing; live entries
            # would otherwise show later-frame decision/age on earlier dumps.
            events.append(entry.snapshot())
            if decision == WATCH:
                keep.append(entry)
            elif decision == ATTACK:
                # Label + keep tracking; do NOT emit into Ŷ (still Tentative).
                n_a += 1
                keep.append(entry)
            elif decision == CONFIRM:
                n_c += 1
                confirmed.append(np.asarray(entry.box, dtype=np.float64).copy())
                confirms.append(entry)
                if getattr(entry, "hist", None):
                    self.confirmed_hists.append(list(entry.hist))
                self.last_confirm_trajs.append(
                    {
                        "box": np.asarray(entry.box, dtype=np.float64)[:7].copy(),
                        "traj": list(entry.velocity_traj()),
                        "hist": list(entry.hist or []),
                    }
                )
            elif decision == REJECT:
                n_r += 1
            else:
                n_t += 1

        pool = np.asarray(pool_boxes) if pool_boxes is not None else np.zeros((0, 7))
        if pool.ndim != 2:
            pool = np.zeros((0, 7), dtype=np.float64)
        occ = occluders

        def _vis(box) -> float:
            b = np.asarray(box, dtype=np.float64).reshape(-1)[:7]
            if visibility_fn is not None:
                try:
                    return float(visibility_fn(b))
                except Exception:
                    return 0.0
            return float(visibility_ratio(b, occ))

        def _meas_traj(entry: BufferEntry, obj, frame_id: int, fallback_box) -> np.ndarray:
            ref, src = traj_ref_box(obj, ego=ego, uav=uav, init=init)
            if ref is None:
                ref = np.asarray(fallback_box, dtype=np.float64)[:7]
                src = "box"
            entry.append_traj(frame_id, ref, src)
            return np.asarray(ref, dtype=np.float64)[:7].copy()

        for ei, entry in enumerate(old):
            oj = matched.get(ei, -1)
            if oj >= 0:
                obj = match_src[oj]
                if getattr(obj, "from_far", False):
                    continue
                curr = _meas_traj(entry, obj, frame_id, obj.box)
                if int(obj.d_ego) == 1:
                    entry.q_ego = obj_q_buf_ego(obj, mode=self.buf_q_mode)
                if int(obj.d_uav) == 1:
                    entry.q_uav = obj_q_uav(obj)
                if obj.state == CERTAIN:
                    # Gate/pool Certain: leave buffer; dump backfill only.
                    entry.hist.append((int(frame_id), curr[:7].copy()))
                    self.gate_certain_hists.append(list(entry.hist))
                    self.last_certain_trajs.append(
                        {
                            "box": curr[:7].copy(),
                            "traj": list(entry.velocity_traj()),
                            "hist": list(entry.hist),
                        }
                    )
                    continue
                entry.v = _vis(entry.box)
                ego_hit = int(obj.d_ego) == 1
                self._update(
                    entry,
                    ego_hit=ego_hit,
                    curr_box=curr,
                    frame_id=frame_id,
                    v=entry.v,
                )
                self._kf_meas(entry, curr)
                entry.miss_age = 0
                finish(entry, self._decide(entry))
                continue
            owned = False
            if pool.shape[0] > 0:
                for pb in pool:
                    if same_object(entry.box, pb, self.theta_iou, self.theta_d):
                        owned = True
                        break
            if owned:
                continue
            sj = soft_hit.get(ei, -1)
            if sj >= 0:
                det = soft_src[sj]
                curr = np.asarray(det.box, dtype=np.float64).copy()
                entry.q_ego = float(getattr(det, "q", 0.0) or 0.0)
                entry.append_traj(frame_id, curr, "ego")
                entry.v = _vis(entry.box)
                self._update(
                    entry,
                    ego_hit=True,
                    curr_box=curr,
                    frame_id=frame_id,
                    v=entry.v,
                )
                self._kf_meas(entry, curr)
                entry.miss_age = 0
                finish(entry, self._decide(entry))
                continue
            entry.v = _vis(entry.box)
            entry.miss_age = int(entry.miss_age) + 1
            self._update(
                entry,
                ego_hit=False,
                curr_box=None,
                frame_id=frame_id,
                v=entry.v,
            )
            if entry.v >= self.v_min:
                entry.hist.append((int(frame_id), np.asarray(entry.box)[:7].copy()))
                decided = self._decide(entry)
                if decided != WATCH:
                    finish(entry, decided)
                elif entry.miss_age >= self.k_atk:
                    finish(entry, TIMEOUT)
                else:
                    finish(entry, ATTACK)
            else:
                finish(entry, self._decide(entry))

        for obj in objects:
            if id(obj) in used_ids or obj.state != TENTATIVE or obj.box is None:
                continue
            entry = self._init_from_obj(
                obj, frame_id=frame_id, ego=ego, uav=uav, init=init
            )
            finish(entry, self._decide(entry))

        self.entries = keep
        certain = [
            np.asarray(o.box, dtype=np.float64)
            for o in objects
            if o.state == CERTAIN and o.box is not None
        ]
        out = certain + confirmed
        if out:
            output = np.stack(out)
        else:
            output = np.zeros((0, 7), dtype=np.float64)
        self.prev_output = output
        snap = FrameBuffer(
            watch=[e.snapshot() for e in keep],
            events=events,
            output_boxes=output,
            n_confirm=n_c,
            n_reject=n_r,
            n_timeout=n_t,
            n_attack=n_a,
        )
        return snap
