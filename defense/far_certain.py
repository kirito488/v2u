"""Far-range no-threat bypass: Certain for dump/Ŷ, never enter the trust pool.

Promote a leftover (no-Ego) UAV/Init object when:
  r >= 60 m for 5 consecutive frames and Δr = r_now - r_start >= 0.
Demote if it approaches (r < 60 or Δr < 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from .associate import THETA_DIST, THETA_IOU, hungarian_match_pool, same_object
from .gating import CERTAIN, TENTATIVE

R_FAR = 60.0
T_FAR = 5

Hist = List[Tuple[int, np.ndarray]]  # (frame_id, box)


def bev_range(box) -> float:
    a = np.asarray(box, dtype=np.float64).reshape(-1)
    return float(np.hypot(a[0], a[1]))


def _is_far_candidate(obj) -> bool:
    """No Ego, UAV and/or Init present, currently far (or already promoted)."""
    if getattr(obj, "box", None) is None:
        return False
    if int(getattr(obj, "d_ego", 0)) == 1:
        return False
    if int(getattr(obj, "d_init", 0)) != 1 and int(getattr(obj, "d_uav", 0)) != 1:
        return False
    if getattr(obj, "from_far", False):
        return True
    if getattr(obj, "state", None) != TENTATIVE:
        return False
    return bev_range(obj.box) >= R_FAR


@dataclass
class _FarTrack:
    box: np.ndarray
    r0: float
    rs: List[float] = field(default_factory=list)
    hist: Hist = field(default_factory=list)
    promoted: bool = False
    dead: bool = False

    @property
    def age(self) -> int:
        return len(self.rs)


class FarCertainTracker:
    def __init__(self, r_far: float = R_FAR, t_far: int = T_FAR):
        self.r_far = float(r_far)
        self.t_far = int(t_far)
        self.reset()

    def reset(self) -> None:
        self.tracks: List[_FarTrack] = []
        self.closed_promoted: List[Hist] = []

    def promoted_hists(self) -> List[Hist]:
        out = list(self.closed_promoted)
        for t in self.tracks:
            if t.promoted and len(t.hist) >= self.t_far:
                out.append(list(t.hist))
        return out

    def step(self, gating: Any, frame_id: int = 0) -> int:
        """Promote / demote this-frame gating objects. Returns n newly promoted."""
        objs = [o for o in (getattr(gating, "objects", None) or []) if o.box is not None]
        cand_i = [i for i, o in enumerate(objs) if _is_far_candidate(o)]
        n_new = 0

        used_t: set = set()
        used_o: set = set()
        if self.tracks and cand_i:
            tboxes = np.stack([t.box for t in self.tracks])
            cboxes = np.stack([objs[i].box for i in cand_i])
            for ti, cj, _iou in hungarian_match_pool(
                tboxes, cboxes, iou_thres=THETA_IOU, dist_thres=THETA_DIST
            ):
                oi = cand_i[int(cj)]
                used_t.add(int(ti))
                used_o.add(oi)
                n_new += self._feed(self.tracks[int(ti)], objs[oi], frame_id)

        alive: List[_FarTrack] = []
        for ti, t in enumerate(self.tracks):
            if getattr(t, "dead", False):
                continue
            if ti in used_t:
                alive.append(t)
                continue
            if t.promoted and len(t.hist) >= self.t_far:
                self.closed_promoted.append(list(t.hist))
        self.tracks = alive

        for i, o in enumerate(objs):
            if i in used_o or not _is_far_candidate(o):
                continue
            t = _FarTrack(
                box=np.asarray(o.box, dtype=np.float64)[:7].copy(),
                r0=bev_range(o.box),
            )
            self._feed(t, o, frame_id)
            self.tracks.append(t)
        return n_new

    def _feed(self, t: _FarTrack, obj, frame_id: int) -> int:
        box = np.asarray(obj.box, dtype=np.float64)[:7].copy()
        r = bev_range(box)
        t.box = box
        t.rs.append(r)
        t.hist.append((int(frame_id), box))
        dr = r - float(t.r0)
        recent = t.rs[-self.t_far :]
        far_ok = (
            t.age >= self.t_far
            and all(x >= self.r_far for x in recent)
            and dr >= 0.0
        )
        if r < self.r_far or dr < 0.0:
            if t.promoted:
                obj.state = TENTATIVE
                obj.from_far = False
                prev = t.hist[:-1]
                if len(prev) >= self.t_far:
                    self.closed_promoted.append(prev)
            t.promoted = False
            if r >= self.r_far:
                t.r0 = r
                t.rs = [r]
                t.hist = [(int(frame_id), box)]
            else:
                t.dead = True
            return 0
        if far_ok:
            newborn = not t.promoted
            t.promoted = True
            obj.state = CERTAIN
            obj.from_far = True
            return 1 if newborn else 0
        return 0


def backfill_tentative_to_certain(
    frames, hists: List[Hist], t_far: Optional[int] = T_FAR
) -> int:
    """After a scene: label pre-confirm Tentative dumps as 'tentative -> certain'.

    t_far: far-bypass uses the 5th frame as confirm. None → last hist frame
    (buffer confirm).
    """
    by_fid = {}
    for fr in frames or []:
        by_fid[int(getattr(fr, "frame_id", -1))] = fr
    n = 0
    for hist in hists or []:
        if not hist:
            continue
        if t_far is not None and len(hist) >= int(t_far):
            confirm_fid = int(hist[int(t_far) - 1][0])
        else:
            confirm_fid = int(hist[-1][0])
        for fid, box in hist:
            fr = by_fid.get(int(fid))
            if fr is None or getattr(fr, "gating", None) is None:
                continue
            for o in fr.gating.objects:
                if o.box is None:
                    continue
                if not same_object(o.box, box):
                    continue
                if int(fid) >= confirm_fid:
                    if o.state != CERTAIN:
                        o.show_certain = True
                    n += 1
                elif o.state == TENTATIVE:
                    o.backfill_certain = True
                    n += 1
    return n
