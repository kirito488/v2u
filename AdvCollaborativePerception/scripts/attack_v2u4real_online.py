"""
V2U4Real 攻击（单物体/帧）+ 指标评估

``--level intermediate``（默认）：改 UAV 的 BEV 特征（PGD），再走中间融合检测。
``--level early``：只改 UAV 原始点云；然后用同一套 attfuse/where2comm/coalign
把各智能体点云提成 BEV、再中间融合检测。**不再做特征扰动。**

用法:
    python scripts/attack_v2u4real_online.py --model attfuse --mode spoof --level intermediate --iters 20
    python scripts/attack_v2u4real_online.py --model attfuse --mode spoof --level early
    python scripts/attack_v2u4real_online.py --model attfuse --mode remove --level early
    python scripts/attack_v2u4real_online.py --model where2comm --mode remove --iters 30
"""
import os
import sys
import argparse
import copy
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="invalid value encountered in intersection")

VERBOSE = False


def log(msg):
    print(msg)


def dbg(msg):
    if VERBOSE:
        print(msg)


def _first_existing(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return os.path.normpath(p)
    return os.path.normpath(candidates[0])


# 服务器优先，本机路径作后备
V2U4REAL_ROOT = _first_existing([
    "/data/hzy/lxt/V2U4Real-main/V2U4Real-main",
    "/data/hzy/lxt/V2U4Real-main",
    "D:/agent_project/V2U4Real-main/V2U4Real-main",
])
CKPT_ROOT = _first_existing([
    "/data/hzy/lxt/V2U4Real-main/checkpoints",
    os.path.join(V2U4REAL_ROOT, "checkpoints"),
    "D:/agent_project/V2U4Real-main/checkpoints",
])

root = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../"))
sys.path.insert(0, root)
sys.path.insert(0, V2U4REAL_ROOT)

from mvp.data.v2u4real_dataset import V2U4RealDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, pcd_sensor_to_map, pcd_map_to_sensor
from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
from mvp.attack.lidar_spoof_early_attacker import LidarSpoofEarlyAttacker
from mvp.attack.lidar_remove_intermediate_attacker import LidarRemoveIntermediateAttacker
from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
from mvp.attack.donor_selector import select_attack_positions
from mvp.evaluate.attack_metrics import (
    evaluate_spoof, evaluate_remove, print_spoof_report, print_remove_report,
)
from opencood.utils.transformation_utils import x1_to_x2

MODEL_CKPTS = {
    "attfuse":    os.path.join(CKPT_ROOT, "attfuse_checkpoint"),
    "where2comm": os.path.join(CKPT_ROOT, "where2comm_checkpoint"),
    "coalign":    os.path.join(CKPT_ROOT, "coalign_checkpoint"),
}


def load_model(name):
    return OpencoodPerception(
        fusion_method="intermediate",
        model_name=name,
        model_dir=MODEL_CKPTS[name],
        opencood_root=V2U4REAL_ROOT,
        root_dir=os.path.join(V2U4REAL_ROOT, "v2u4real", "train"),
        validate_dir=os.path.join(V2U4REAL_ROOT, "v2u4real", "val"),
    )


def run_clean(perception, multi_frame_case, frame_ids, ego_id):
    clean = []
    gt_list = []
    for f in frame_ids:
        frame = multi_frame_case[f]
        pb, ps = perception.run(frame, ego_id)
        if pb is None or len(pb) == 0:
            clean.append((np.zeros((0, 7)), np.zeros((0,))))
        else:
            clean.append((np.asarray(pb), np.asarray(ps).reshape(-1)))
        gt_list.append(frame[ego_id]["gt_bboxes"])
    return clean, gt_list


def _select_spoof_ghost(multi_frame_case, frame_ids, ego_id):
    """与中间 spoof 相同的幽灵选点：ego 前方危险/盲区，z 用 GT 中位数。"""
    ego_frame0 = multi_frame_case[frame_ids[0]][ego_id]
    pos, _ = select_attack_positions(
        ego_frame0["lidar"], ego_frame0["gt_bboxes"], 1, np.random.RandomState(0),
        x_min=15.0, x_max=35.0)
    if not pos:
        pos = [[25.0, 0.0, 0.0]]
    fx, fy = pos[0][0], pos[0][1]
    gz = -1.0
    gts = ego_frame0["gt_bboxes"]
    if gts is not None and len(gts) > 0:
        gz = float(np.median(gts[:, 2]))
    bbox_ego = np.array([fx, fy, gz, 4.7, 1.9, 1.7, 0], dtype=np.float64)
    dbg("[spoof] 幽灵 ego ({:.1f}, {:.1f}, z={:.2f})".format(fx, fy, gz))
    return bbox_ego


def _transform_bbox_opencood(bbox, pose_from, pose_to):
    """pose_from 雷达系 -> pose_to 雷达系，与训练/推理的 x1_to_x2 一致。"""
    T = x1_to_x2(
        np.asarray(pose_from).reshape(-1).tolist(),
        np.asarray(pose_to).reshape(-1).tolist())
    b = np.asarray(bbox, dtype=np.float64).copy()
    q = T.dot(np.array([b[0], b[1], b[2], 1.0]))
    b[0], b[1], b[2] = q[0], q[1], q[2]
    b[6] = b[6] + float(np.arctan2(T[1, 0], T[0, 0]))
    return b


def _snap_bbox_z_to_lidar(bbox, lidar, z_min=-3.5, z_max=2.2):
    """优先用已在 [-4,4] 内的附近点定 z；没有才夹紧，避免框浮在空中导致射线命中为 0。"""
    b = np.asarray(bbox, dtype=np.float64).copy()
    h = float(b[5])
    pcd = np.asarray(lidar)
    if pcd.ndim == 2 and pcd.shape[0] > 0:
        in_rng = pcd[(pcd[:, 2] >= -4.0) & (pcd[:, 2] <= 4.0)]
        src = in_rng if in_rng.shape[0] >= 50 else pcd
        d = np.hypot(src[:, 0] - b[0], src[:, 1] - b[1])
        near = src[d < 12.0]
        if near.shape[0] >= 10:
            b[2] = float(np.median(near[:, 2]))
        elif src.shape[0] > 0:
            b[2] = float(np.median(src[:, 2]))
    b[2] = float(np.clip(b[2], z_min, z_max - h))
    return b


def _ghost_in_attacker_frame(bbox_ego, multi_frame_case, frame_ids, ego_id, att_id,
                             opencood=False, snap_z=False):
    positions = [None] * len(multi_frame_case)
    xf = _transform_bbox_opencood if opencood else _transform_bbox
    for f in frame_ids:
        frame = multi_frame_case[f]
        b = xf(bbox_ego, frame[ego_id]["lidar_pose"], frame[att_id]["lidar_pose"])
        if snap_z:
            b = _snap_bbox_z_to_lidar(b, frame[att_id]["lidar"])
        # 用同一 T 把 UAV 点投回 ego，确认评估坐标一致
        T_back = x1_to_x2(
            np.asarray(frame[att_id]["lidar_pose"]).reshape(-1).tolist(),
            np.asarray(frame[ego_id]["lidar_pose"]).reshape(-1).tolist())
        back = T_back.dot(np.array([b[0], b[1], b[2], 1.0]))
        positions[f] = b
        dbg("[spoof] frame {} UAV=({:.1f},{:.1f}) ego回投=({:.1f},{:.1f})".format(
            f, b[0], b[1], back[0], back[1]))
    return positions


def _pack_dets(pb, ps):
    if pb is None or len(pb) == 0:
        return np.zeros((0, 7)), np.zeros((0,))
    return np.asarray(pb), np.asarray(ps).reshape(-1)


def _points_in_obb(pts, bbox, pad=0.25):
    x, y, z, l, w, h, yaw = [float(v) for v in bbox[:7]]
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    lz = pts[:, 2] - z
    return ((np.abs(lx) <= l / 2.0 + pad) &
            (np.abs(ly) <= w / 2.0 + pad) &
            (lz >= -pad) & (lz <= h + pad))


def _warp_cloud(src_xyz, src_bbox, dst_bbox):
    """把 src 框内点按长宽高归一化后贴到 dst 框（含 yaw）。"""
    sx, sy, sz, sl, sw, sh, syaw = [float(v) for v in src_bbox[:7]]
    dx, dy, dz, dl, dw, dh, dyaw = [float(v) for v in dst_bbox[:7]]
    c, s = np.cos(-syaw), np.sin(-syaw)
    p = src_xyz - np.array([sx, sy, sz])
    loc = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], 1)
    loc[:, 0] *= dl / max(sl, 1e-3)
    loc[:, 1] *= dw / max(sw, 1e-3)
    loc[:, 2] *= dh / max(sh, 1e-3)
    c2, s2 = np.cos(dyaw), np.sin(dyaw)
    out = np.stack([
        c2 * loc[:, 0] - s2 * loc[:, 1] + dx,
        s2 * loc[:, 0] + c2 * loc[:, 1] + dy,
        loc[:, 2] + dz,
    ], 1)
    return out


def _append_xyz(lidar, xyz, intensity=None):
    lidar = np.asarray(lidar)
    xyz = np.asarray(xyz, dtype=lidar.dtype)
    if xyz.shape[0] == 0:
        return lidar
    pad = np.ones((xyz.shape[0], lidar.shape[1]), dtype=lidar.dtype)
    pad[:, :3] = xyz
    if lidar.shape[1] > 3:
        if intensity is None:
            val = float(np.median(lidar[:, 3])) if lidar.shape[0] else 1.0
            pad[:, 3] = val
        else:
            inten = np.asarray(intensity, dtype=lidar.dtype).reshape(-1)
            if inten.shape[0] == xyz.shape[0]:
                pad[:, 3] = inten
            else:
                pad[:, 3] = float(inten[0]) if inten.shape[0] else 1.0
    return np.vstack([lidar, pad])


def _harvest_donor(frame, ego_id, att_id, ghost_ego):
    """从真实车辆上抠点，贴到幽灵框。优先 UAV 顶视回波（与攻击通道同分布）。"""
    att = frame[att_id]
    ego = frame[ego_id]
    T_e2a = x1_to_x2(
        np.asarray(ego["lidar_pose"]).reshape(-1).tolist(),
        np.asarray(att["lidar_pose"]).reshape(-1).tolist())
    ghost_uav = _transform_bbox_opencood(
        ghost_ego, ego["lidar_pose"], att["lidar_pose"])
    best = None
    for src_id, src_bbox_list, src_lidar, to_uav in (
            (att_id, att["gt_bboxes"], att["lidar"], None),
            (ego_id, ego["gt_bboxes"], ego["lidar"], T_e2a),
    ):
        if src_bbox_list is None or len(src_bbox_list) == 0:
            continue
        pts = np.asarray(src_lidar)
        for b in src_bbox_list:
            b = np.asarray(b, dtype=np.float64)
            mask = _points_in_obb(pts[:, :3], b)
            n = int(mask.sum())
            if n < 40:
                continue
            xyz = pts[mask, :3].astype(np.float64)
            inten = pts[mask, 3] if pts.shape[1] > 3 else None
            if to_uav is not None:
                xyz = (to_uav.dot(np.c_[xyz, np.ones((xyz.shape[0], 1))].T)).T[:, :3]
                b_uav = _transform_bbox_opencood(b, ego["lidar_pose"], att["lidar_pose"])
                xyz = _warp_cloud(xyz, b_uav, ghost_uav)
            else:
                xyz = _warp_cloud(xyz, b, ghost_uav)
            score = n
            if best is None or score > best[0]:
                best = (score, src_id, xyz, inten, ghost_uav)
    if best is None:
        dbg("[early-spoof] 无 donor 点云")
        return None, ghost_uav
    n, src_id, xyz, inten, ghost_uav = best
    dbg("[early-spoof] donor={} 点数={}".format(src_id, n))
    return (xyz, inten), ghost_uav


def _offset_box(src, dist):
    dst = np.asarray(src, dtype=np.float64).copy()
    yaw = float(dst[6])
    dst[0] += dist * np.cos(yaw)
    dst[1] += dist * np.sin(yaw)
    return dst


def _select_early_ghost(clean, ego0_lidar):
    """源车必须是真实外部车辆：score>=0.3、距 ego 12-65m、框内立体点>=50。
    帧 3-9 失败是 argmax 落到 ego 自车(0.7,-0.2)，抠出的是贴地地面点，warp 后是平的。
    """
    boxes, scores = clean[0]
    boxes = np.asarray(boxes)
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(-scores)
    for j in order:
        if scores[j] < 0.3:
            break
        src = boxes[j].astype(np.float64)
        r = float(np.hypot(src[0], src[1]))
        if r < 10.0 or r > 75.0:
            continue
        _, _, n = _extract_box_cloud(ego0_lidar, src)
        if n < 15:
            dbg("[spoof] 跳过 ({:.1f},{:.1f}) 点={}".format(src[0], src[1], n))
            continue
        others = np.delete(boxes, j, axis=0) if len(boxes) > 1 else np.zeros((0, 7))
        for dist in (8.0, 10.0, 12.0, -8.0, 6.0, -6.0):
            dst = _offset_box(src, dist)
            rr = float(np.hypot(dst[0], dst[1]))
            if rr < 4.0 or rr > 70.0:
                continue
            if others.shape[0] > 0:
                dmin = float(np.min(np.hypot(others[:, 0] - dst[0], others[:, 1] - dst[1])))
                if dmin < 5.0:
                    continue
            dbg("[spoof] 幽灵 +{:.0f}m ({:.1f},{:.1f})←({:.1f},{:.1f}) 点={}".format(
                dist, dst[0], dst[1], src[0], src[1], n))
            return dst, src, dist
    j = int(np.argmax(scores))
    src = boxes[j].astype(np.float64)
    dst = _offset_box(src, 8.0)
    dbg("[spoof] fallback +8m -> ({:.1f},{:.1f})".format(dst[0], dst[1]))
    return dst, src, 8.0


def _extract_box_cloud(lidar, bbox):
    raw = np.asarray(lidar)
    mask = _points_in_obb(raw[:, :3], bbox, pad=0.4)
    if int(mask.sum()) < 30:
        mask = _points_in_obb(raw[:, :3], bbox, pad=2.0)
    xyz = raw[mask, :3].astype(np.float64)
    inten = raw[mask, 3] if raw.shape[1] > 3 else None
    return xyz, inten, int(mask.sum())


def run_spoof_early_points(perception, dataset, multi_frame_case, frame_ids,
                           ego_id, att_id, dense=3, sync=0, also_ego=False,
                           sanity="off", clean=None, ghost="near"):
    """early spoof。ghost=near：把已检车辆点云平移后写入 UAV（与 near_clone 同几何）。
    ghost=danger：旧的 (15,-3) 射线注入。"""
    if sanity in ("ego_clone", "near_clone"):
        bbox_ego = _select_spoof_ghost(multi_frame_case, frame_ids, ego_id)
        src_box = None
        if clean is None or len(clean) == 0 or len(clean[0][0]) == 0:
            log("[early-spoof] 需要干净检测，放弃")
            return (
                [(np.zeros((0, 7)), np.zeros((0,))) for _ in frame_ids],
                [bbox_ego.copy() for _ in frame_ids],
                multi_frame_case,
            )
        scores0 = np.asarray(clean[0][1]).reshape(-1)
        src_box = np.asarray(clean[0][0][int(np.argmax(scores0))], dtype=np.float64)
        if sanity == "near_clone":
            bbox_ego = src_box.copy()
            bbox_ego[0] = src_box[0] + (8.0 if src_box[0] < 40 else -8.0)
            dbg("[early-spoof] near_clone ({:.1f},{:.1f})→({:.1f},{:.1f})".format(
                src_box[0], src_box[1], bbox_ego[0], bbox_ego[1]))
        else:
            dbg("[early-spoof] ego_clone ({:.1f},{:.1f})→({:.1f},{:.1f})".format(
                src_box[0], src_box[1], bbox_ego[0], bbox_ego[1]))
        attacked_case = copy.deepcopy(multi_frame_case)
        dist = None
        src0 = src_box
    elif ghost == "danger":
        bbox_ego = _select_spoof_ghost(multi_frame_case, frame_ids, ego_id)
        positions = _ghost_in_attacker_frame(
            bbox_ego, multi_frame_case, frame_ids, ego_id, att_id,
            opencood=True, snap_z=False)
        attacker = LidarSpoofEarlyAttacker(dataset, dense=dense, sync=sync)
        attacked_case, _ = attacker.run(multi_frame_case, {
            "frame_ids": frame_ids,
            "attacker_vehicle_id": att_id,
            "positions": positions,
        })
        dist = None
        src0 = None
    else:
        if clean is None or len(clean) == 0 or len(clean[0][0]) == 0:
            log("[early-spoof] 无干净检测，回退 danger 幽灵注入")
            ghost = "danger"
            bbox_ego = _select_spoof_ghost(multi_frame_case, frame_ids, ego_id)
            positions = _ghost_in_attacker_frame(
                bbox_ego, multi_frame_case, frame_ids, ego_id, att_id,
                opencood=True, snap_z=False)
            attacker = LidarSpoofEarlyAttacker(dataset, dense=dense, sync=sync)
            attacked_case, _ = attacker.run(multi_frame_case, {
                "frame_ids": frame_ids,
                "attacker_vehicle_id": att_id,
                "positions": positions,
            })
            dist = None
            src0 = None
        else:
            bbox_eval0, src0, dist = _select_early_ghost(
                clean, multi_frame_case[frame_ids[0]][ego_id]["lidar"])
            log("  幽灵 ({:.1f},{:.1f}) ← 检出车平移 {:.0f}m".format(
                bbox_eval0[0], bbox_eval0[1], dist))
            ego0_pose = np.asarray(multi_frame_case[frame_ids[0]][ego_id]["lidar_pose"])
            cloud_world = np.zeros((0, 3))
            inten0 = None
            cloud0, inten0, n0 = _extract_box_cloud(
                multi_frame_case[frame_ids[0]][ego_id]["lidar"], src0)
            if n0 >= 15:
                cloud_eval0 = _warp_cloud(cloud0, src0, bbox_eval0)
                cloud_world = pcd_sensor_to_map(cloud_eval0, ego0_pose)
            else:
                log("[early-spoof] 源车点过少 n={}，改 danger 注入".format(n0))
                ghost = "danger"
                bbox_ego = _select_spoof_ghost(multi_frame_case, frame_ids, ego_id)
                positions = _ghost_in_attacker_frame(
                    bbox_ego, multi_frame_case, frame_ids, ego_id, att_id,
                    opencood=True, snap_z=False)
                attacker = LidarSpoofEarlyAttacker(dataset, dense=dense, sync=sync)
                attacked_case, _ = attacker.run(multi_frame_case, {
                    "frame_ids": frame_ids,
                    "attacker_vehicle_id": att_id,
                    "positions": positions,
                })
                dist = None
                src0 = None
            if ghost != "danger":
                ghost_world = bbox_sensor_to_map(bbox_eval0, ego0_pose)
                bbox_ego = bbox_eval0
                attacked_case = copy.deepcopy(multi_frame_case)

    attacked = []
    ghosts = []
    prev_src = src0 if src0 is not None else None
    frame_notes = []
    for f in frame_ids:
        frame = attacked_case[f]
        bbox_eval = bbox_ego
        if sanity in ("ego_clone", "near_clone"):
            raw = np.asarray(multi_frame_case[f][ego_id]["lidar"])
            src_f = src_box
            if clean is not None and f < len(clean) and len(clean[f][0]) > 0:
                if prev_src is not None:
                    cand = np.asarray(clean[f][0])
                    dc = np.hypot(cand[:, 0] - prev_src[0], cand[:, 1] - prev_src[1])
                    k = int(np.argmin(dc))
                    if dc[k] < 15.0:
                        src_f = cand[k].astype(np.float64)
                else:
                    sc = np.asarray(clean[f][1]).reshape(-1)
                    src_f = np.asarray(clean[f][0][int(np.argmax(sc))], dtype=np.float64)
                prev_src = src_f
                if sanity == "near_clone":
                    bbox_f = src_f.copy()
                    bbox_f[0] = src_f[0] + (8.0 if src_f[0] < 40 else -8.0)
                else:
                    bbox_f = bbox_ego
            else:
                bbox_f = bbox_ego
            xyz, inten, n_pick = _extract_box_cloud(raw, src_f)
            dbg("[early-spoof] frame {} 抠点={}".format(f, n_pick))
            if xyz.shape[0] > 0:
                xyz = _warp_cloud(xyz, src_f, bbox_f)
                frame[ego_id]["lidar"] = _append_xyz(frame[ego_id]["lidar"], xyz, inten)
            bbox_eval = bbox_f
        elif ghost == "near":
            ego_pose_f = np.asarray(frame[ego_id]["lidar_pose"])
            uav_pose_f = np.asarray(frame[att_id]["lidar_pose"])
            bbox_eval = bbox_map_to_sensor(ghost_world, ego_pose_f)
            if cloud_world.shape[0] > 0:
                xyz_uav = pcd_map_to_sensor(cloud_world, uav_pose_f)
                frame[att_id]["lidar"] = _append_xyz(frame[att_id]["lidar"], xyz_uav, inten0)
                if also_ego:
                    xyz_ego = pcd_map_to_sensor(cloud_world, ego_pose_f)
                    frame[ego_id]["lidar"] = _append_xyz(frame[ego_id]["lidar"], xyz_ego, inten0)
            dbg("[early-spoof] frame {} 点={} 目标=({:.1f},{:.1f})".format(
                f, int(cloud_world.shape[0]), bbox_eval[0], bbox_eval[1]))
        else:
            donor, _ = _harvest_donor(multi_frame_case[f], ego_id, att_id, bbox_ego)
            if donor is not None:
                xyz, inten = donor
                frame[att_id]["lidar"] = _append_xyz(frame[att_id]["lidar"], xyz, inten)

        T = x1_to_x2(
            np.asarray(frame[att_id]["lidar_pose"]).reshape(-1).tolist(),
            np.asarray(frame[ego_id]["lidar_pose"]).reshape(-1).tolist())
        p = np.asarray(frame[att_id]["lidar"][:, :3], dtype=np.float64)
        ego_pts = (T.dot(np.c_[p, np.ones((p.shape[0], 1))].T)).T[:, :3]
        dxy = np.hypot(ego_pts[:, 0] - bbox_eval[0], ego_pts[:, 1] - bbox_eval[1])
        n_keep = int(
            ((ego_pts[:, 0] > -100.8) & (ego_pts[:, 0] < 100.8) &
             (ego_pts[:, 1] > -80.0) & (ego_pts[:, 1] < 80.0) &
             (ego_pts[:, 2] > -4.0) & (ego_pts[:, 2] < 4.0) &
             (dxy < 3.0)).sum())
        if also_ego and sanity == "off" and ghost == "danger":
            near = ego_pts[dxy < 3.0]
            if near.shape[0] > 0:
                frame[ego_id]["lidar"] = _append_xyz(frame[ego_id]["lidar"], near)
        ego_lidar = np.asarray(frame[ego_id]["lidar"][:, :3], dtype=np.float64)
        n_ego = int((np.hypot(ego_lidar[:, 0] - bbox_eval[0],
                              ego_lidar[:, 1] - bbox_eval[1]) < 3.0).sum())
        dbg("[early-spoof] frame {} UAV近点={} ego近点={}".format(f, n_keep, n_ego))
        pb, ps = perception.run(frame, ego_id)
        attacked.append(_pack_dets(pb, ps))
        ab, ascore = attacked[-1]
        atk_sc = 0.0
        if len(ab) > 0:
            dd = np.hypot(ab[:, 0] - bbox_eval[0], ab[:, 1] - bbox_eval[1])
            j = int(np.argmin(dd))
            if float(dd[j]) <= 4.0:
                atk_sc = float(ascore[j])
        cl_sc = _score_near(clean[f] if clean and f < len(clean) else None, bbox_eval)
        frame_notes.append((f, cl_sc, atk_sc))
        dbg("[early-spoof] frame {} 干净={:.3f} 攻击={:.3f} 检测数={}".format(
            f, cl_sc, atk_sc, len(ab)))
        ghosts.append(np.asarray(bbox_eval, dtype=np.float64).copy())
    if frame_notes and not VERBOSE:
        ok = sum(1 for _, c, a in frame_notes if c < 0.3 and a >= 0.3)
        elig = sum(1 for _, c, _ in frame_notes if c < 0.3)
        log("  攻击 {} 帧 | 幽灵 ({:.1f},{:.1f}) | 即时成功 {}/{}".format(
            len(frame_notes), ghosts[0][0], ghosts[0][1], ok, elig))
    elif VERBOSE and frame_notes:
        log("  f   clean  attack")
        for f, cl, atk in frame_notes:
            log("  {:>2}  {:.3f}  {:.3f}".format(f, cl, atk))
    return attacked, ghosts, attacked_case


def run_spoof_online(perception, dataset, multi_frame_case, frame_ids, ego_id, att_id, iters):
    bbox_ego = _select_spoof_ghost(multi_frame_case, frame_ids, ego_id)
    log("  幽灵 ({:.1f},{:.1f})".format(bbox_ego[0], bbox_ego[1]))
    positions = _ghost_in_attacker_frame(
        bbox_ego, multi_frame_case, frame_ids, ego_id, att_id)

    attacker = LidarSpoofIntermediateAttacker(
        perception, dataset, step=iters, sync=0, init=False, online=True)
    _, info = attacker.run(multi_frame_case, {
        "frame_ids": frame_ids,
        "attacker_vehicle_id": att_id,
        "victim_vehicle_id": ego_id,
        "positions": positions,
        "attack_info": [{} for _ in range(len(multi_frame_case))],
    })
    attacked = []
    for f in frame_ids:
        pb = info[f][ego_id]["pred_bboxes"]
        ps = info[f][ego_id]["pred_scores"]
        if pb is None or len(pb) == 0:
            attacked.append((np.zeros((0, 7)), np.zeros((0,))))
        else:
            attacked.append((np.asarray(pb), np.asarray(ps).reshape(-1)))
    ghosts = [bbox_ego.copy() for _ in frame_ids]
    return attacked, ghosts


def _transform_bbox(bbox, pose_from, pose_to):
    """把 [x,y,z,l,w,h,yaw] 从 pose_from 的雷达系转到 pose_to 的雷达系。"""
    world = bbox_sensor_to_map(np.asarray(bbox, dtype=np.float64), pose_from)
    return bbox_map_to_sensor(world, pose_to)


def _uav_points_on_ego_box(frame, ego_id, att_id, box_ego, pad=0.5):
    """ego 系检测框投到 UAV 系后，统计无人机点云落在框内的点数。"""
    box_att = _transform_bbox(
        box_ego, frame[ego_id]["lidar_pose"], frame[att_id]["lidar_pose"])
    pts = np.asarray(frame[att_id]["lidar"])
    n = 0
    if pts.ndim == 2 and pts.shape[0] > 0:
        n = int(_points_in_obb(pts[:, :3], box_att, pad=pad).sum())
        if n < 8:
            n = int(_points_in_obb(pts[:, :3], box_att, pad=2.0).sum())
    return n, box_att


def _blank_lidar(lidar):
    """留 1 个远处点，避免空点云把 voxel 预处理打崩。"""
    x = np.asarray(lidar)
    if x.ndim != 2 or x.shape[1] < 3:
        return np.zeros((1, 4), dtype=np.float32)
    row = np.zeros((1, x.shape[1]), dtype=x.dtype)
    row[0, 0], row[0, 1], row[0, 2] = 200.0, 200.0, 0.0
    if x.shape[1] > 3:
        row[0, 3] = 1.0
    return row


def _score_near(dets, box, max_dist=4.0):
    if dets is None or box is None:
        return 0.0
    boxes, scores = dets
    if boxes is None or len(boxes) == 0:
        return 0.0
    boxes = np.asarray(boxes)
    scores = np.asarray(scores).reshape(-1)
    d = np.hypot(boxes[:, 0] - box[0], boxes[:, 1] - box[1])
    j = int(np.argmin(d))
    return float(scores[j]) if float(d[j]) <= max_dist else 0.0


def run_split_views(perception, multi_frame_case, frame_ids, ego_id, att_id):
    """融合检出之外，再跑 ego-only / UAV-only（清空另一路点云）。"""
    ego_only, uav_only = [], []
    for f in frame_ids:
        frame = multi_frame_case[f]
        fe = copy.deepcopy(frame)
        fe[att_id]["lidar"] = _blank_lidar(fe[att_id]["lidar"])
        fu = copy.deepcopy(frame)
        fu[ego_id]["lidar"] = _blank_lidar(fu[ego_id]["lidar"])
        try:
            ego_only.append(_pack_dets(*perception.run(fe, ego_id)))
        except Exception as e:
            dbg("[remove] ego-only 失败: {}".format(e))
            ego_only.append((np.zeros((0, 7)), np.zeros((0,))))
        try:
            uav_only.append(_pack_dets(*perception.run(fu, ego_id)))
        except Exception as e:
            dbg("[remove] UAV-only 失败: {}".format(e))
            uav_only.append((np.zeros((0, 7)), np.zeros((0,))))
    return ego_only, uav_only


def _oid_is_car(oid, type_map) -> bool:
    """Prefer defense.attack_gt.is_car_oid; else obj_type==Car / unknown OK."""
    try:
        from defense.attack_gt import is_car_oid as _is_car

        return bool(_is_car(oid, type_map))
    except Exception:
        t = (type_map or {}).get(str(oid), "")
        return (not t) or t == "Car"


def _gt_cars_in_ego(frame, ego_id, att_id, min_range=3.0, max_range=75.0):
    """Ego-frame Car GT boxes in range (for last-resort remove targets)."""
    rows = []
    ego = frame[ego_id]
    types = {}
    vehicles = (ego.get("params") or {}).get("vehicles") or {}
    for oid, info in vehicles.items():
        types[str(oid)] = (info or {}).get("obj_type", "")
    boxes = np.asarray(ego.get("gt_bboxes", np.zeros((0, 7))))
    ids = list(ego.get("object_ids") or [])
    for i, oid in enumerate(ids):
        if i >= len(boxes):
            break
        if not _oid_is_car(oid, types):
            continue
        b = np.asarray(boxes[i], dtype=np.float64)[:7]
        r = float(np.hypot(b[0], b[1]))
        if r < float(min_range) or r > float(max_range):
            continue
        rows.append(b.copy())
    if att_id in frame:
        att = frame[att_id]
        atypes = {}
        avehicles = (att.get("params") or {}).get("vehicles") or {}
        for oid, info in avehicles.items():
            atypes[str(oid)] = (info or {}).get("obj_type", "")
        aboxes = np.asarray(att.get("gt_bboxes", np.zeros((0, 7))))
        aids = list(att.get("object_ids") or [])
        seen = {tuple(np.round(b[:2], 1)) for b in rows}
        for i, oid in enumerate(aids):
            if i >= len(aboxes):
                break
            if not _oid_is_car(oid, atypes):
                continue
            b = _transform_bbox(
                aboxes[i], att["lidar_pose"], ego["lidar_pose"]
            )
            b = np.asarray(b, dtype=np.float64)[:7]
            r = float(np.hypot(b[0], b[1]))
            if r < float(min_range) or r > float(max_range):
                continue
            key = tuple(np.round(b[:2], 1))
            if key in seen:
                continue
            seen.add(key)
            rows.append(b.copy())
    return rows


def _force_any_remove_box(
    clean_boxes,
    clean_scores,
    frame,
    ego_id,
    att_id,
    min_range=3.0,
    prefer_near=None,
):
    """Last-resort: any fused det in range, else nearest Car GT. Never None if possible."""
    boxes = np.asarray(clean_boxes) if clean_boxes is not None else np.zeros((0, 7))
    scores = (
        np.asarray(clean_scores).reshape(-1)
        if clean_scores is not None
        else np.zeros((0,))
    )
    cands = []
    if boxes.ndim == 2 and boxes.shape[0] > 0:
        for i in range(len(scores) if len(scores) else len(boxes)):
            b = boxes[i][:7]
            r = float(np.hypot(b[0], b[1]))
            if r < float(min_range) or r > 75.0:
                continue
            sc = float(scores[i]) if i < len(scores) else 0.0
            # Prefer higher score; among equals prefer closer to prefer_near
            near = 0.0
            if prefer_near is not None:
                near = -float(np.hypot(b[0] - prefer_near[0], b[1] - prefer_near[1]))
            cands.append((sc, near, b.copy()))
    if cands:
        cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
        b = cands[0][2]
        dbg("[remove] force-det ({:.1f},{:.1f}) score={:.2f}".format(
            b[0], b[1], cands[0][0]))
        return b, float(cands[0][0]), "force-det"
    gts = _gt_cars_in_ego(frame, ego_id, att_id, min_range=min_range)
    if not gts:
        return None, 0.0, "none"
    if prefer_near is not None:
        gts.sort(
            key=lambda b: float(np.hypot(b[0] - prefer_near[0], b[1] - prefer_near[1]))
        )
    else:
        gts.sort(key=lambda b: float(np.hypot(b[0], b[1])))
    b = gts[0]
    dbg("[remove] force-gt ({:.1f},{:.1f})".format(b[0], b[1]))
    return b.copy(), 1.0, "force-gt"


def _select_remove_target(clean0_boxes, clean0_scores, frame0, ego_id, att_id,
                          require_uav_pts=0, min_range=3.0,
                          ego_only=None, uav_only=None, prefer_uav_gap=False,
                          max_ego_score=None, ensure_one=False):
    """选 remove 目标。early+prefer_uav_gap：融合>=0.3、UAV有点，优先「机强车弱」。

    max_ego_score: 优先跳过车端过强目标；ensure_one=True 时仍逐级回退，
    保证尽量每帧至少一个目标（机强车弱 → min-ego → 放宽 UAV 点数 →
    任意融合框 → GT）。
    """
    ego0 = frame0[ego_id]
    att0 = frame0[att_id]
    att_in_ego = []
    for b in att0["gt_bboxes"]:
        att_in_ego.append(_transform_bbox(b, att0["lidar_pose"], ego0["lidar_pose"]))
    att_in_ego = np.array(att_in_ego) if len(att_in_ego) else np.zeros((0, 7))

    clean0_boxes = np.asarray(clean0_boxes) if clean0_boxes is not None else np.zeros((0, 7))
    clean0_scores = np.asarray(clean0_scores).reshape(-1) if clean0_scores is not None else np.zeros((0,))

    def _collect(req_pts, enforce_max_ego):
        order = np.argsort(-clean0_scores) if len(clean0_scores) else np.arange(0)
        uav_hit = None
        fallback = None
        ranked = []
        ego_strong = []
        weak_uav = []  # dets that fail UAV-pts but otherwise OK
        for i in order:
            if i >= len(clean0_boxes):
                continue
            box = clean0_boxes[i]
            dist = float(np.hypot(box[0], box[1]))
            if dist < min_range or dist > 75.0:
                continue
            if i < len(clean0_scores) and float(clean0_scores[i]) < 0.3:
                continue
            n_uav, _ = _uav_points_on_ego_box(frame0, ego_id, att_id, box)
            if req_pts > 0 and n_uav < req_pts:
                weak_uav.append((n_uav, float(clean0_scores[i]) if i < len(clean0_scores) else 0.0, i))
                dbg("[remove] 跳过 ({:.1f},{:.1f}) uav_pts={}".format(box[0], box[1], n_uav))
                continue
            ego_s = _score_near(ego_only, box)
            uav_s = _score_near(uav_only, box)
            uav_pts_s = min(1.0, float(n_uav) / 80.0)
            uav_s = max(uav_s, uav_pts_s)
            util = uav_s - ego_s
            fused = float(clean0_scores[i]) if i < len(clean0_scores) else 0.0
            dbg("[remove] 候选 ({:.1f},{:.1f}) fused={:.2f} ego={:.2f} uav={:.2f} util={:.2f}".format(
                box[0], box[1], fused, ego_s, uav_s, util))
            if prefer_uav_gap and util <= 0.05:
                dbg("[remove] 机弱车强 ({:.1f},{:.1f}) ego={:.2f}".format(box[0], box[1], ego_s))
                ego_strong.append((-ego_s, n_uav, fused, i, ego_s, uav_s, util))
                continue
            ranked.append((util, uav_s, -ego_s, fused, i, ego_s, n_uav))
            if fallback is None:
                fallback = i
            if att_in_ego.shape[0] > 0:
                dmin = np.min(np.hypot(att_in_ego[:, 0] - box[0], att_in_ego[:, 1] - box[1]))
                if dmin < 8.0:
                    uav_hit = i
                    if not prefer_uav_gap:
                        break
        if prefer_uav_gap:
            if ranked:
                ranked.sort(reverse=True)
                util, uav_s, _, fused, pick, ego_s, n_uav = ranked[0]
                dbg("[remove] 选 uav-over-ego util={:.2f} xy=({:.1f},{:.1f})".format(
                    util, float(clean0_boxes[pick][0]), float(clean0_boxes[pick][1])))
                return clean0_boxes[pick], fused, "uav-over-ego"
            if ego_strong:
                ego_strong.sort(reverse=True)
                _, n_uav, fused, pick, ego_s, uav_s, util = ego_strong[0]
                if enforce_max_ego and max_ego_score is not None and float(ego_s) > float(max_ego_score):
                    dbg("[remove] 暂缓 min-ego ego={:.2f} > {:.2f} xy=({:.1f},{:.1f})".format(
                        ego_s, float(max_ego_score),
                        float(clean0_boxes[pick][0]), float(clean0_boxes[pick][1])))
                else:
                    dbg("[remove] 选 min-ego ego={:.2f} xy=({:.1f},{:.1f})".format(
                        ego_s, float(clean0_boxes[pick][0]), float(clean0_boxes[pick][1])))
                    return clean0_boxes[pick], fused, "min-ego"
            return None, 0.0, "none", ego_strong, weak_uav
        pick = uav_hit if uav_hit is not None else fallback
        if pick is None:
            return None, 0.0, "none", ego_strong, weak_uav
        src = "uav-visible" if uav_hit is not None else "ego-det-fallback"
        if req_pts > 0:
            src = "uav-points"
        return clean0_boxes[pick], float(clean0_scores[pick]) if pick < len(clean0_scores) else 0.0, src, ego_strong, weak_uav

    # Pass 1: strict
    out = _collect(require_uav_pts, enforce_max_ego=True)
    if len(out) == 3:
        return out
    box, sc, src, ego_strong, weak_uav = out
    if box is not None:
        return box, sc, src

    if not ensure_one:
        dbg("[remove] 无可打目标")
        return None, 0.0, "none"

    # Pass 2: allow ego-strong min-ego (ignore max_ego_score)
    if ego_strong:
        ego_strong.sort(reverse=True)
        _, n_uav, fused, pick, ego_s, uav_s, util = ego_strong[0]
        dbg("[remove] ensure min-ego-strong ego={:.2f} xy=({:.1f},{:.1f})".format(
            ego_s, float(clean0_boxes[pick][0]), float(clean0_boxes[pick][1])))
        return clean0_boxes[pick], fused, "min-ego-strong"

    # Pass 3: relax UAV point requirement
    if require_uav_pts > 0:
        out2 = _collect(0, enforce_max_ego=False)
        if len(out2) == 3:
            return out2
        box2, sc2, src2, ego_strong2, _ = out2
        if box2 is not None:
            dbg("[remove] ensure relax-uav-pts src={}".format(src2))
            return box2, sc2, src2 + "+relax-pts"
        if ego_strong2:
            ego_strong2.sort(reverse=True)
            _, _, fused, pick, ego_s, _, _ = ego_strong2[0]
            dbg("[remove] ensure min-ego after relax-pts ego={:.2f}".format(ego_s))
            return clean0_boxes[pick], fused, "min-ego-relax-pts"

    # Pass 4: any fused det / GT
    box3, sc3, src3 = _force_any_remove_box(
        clean0_boxes, clean0_scores, frame0, ego_id, att_id, min_range=min_range
    )
    return box3, sc3, src3


def _erase_box_points(lidar, bbox, pad=0.5):
    """删掉定向框内的点。UAV 俯视时 Adv 四面墙经常 0 命中，这是物理删除回波。"""
    lidar = np.asarray(lidar)
    if lidar.ndim != 2 or lidar.shape[0] == 0 or bbox is None:
        return lidar, 0
    mask = _points_in_obb(lidar[:, :3], bbox, pad=pad)
    n = int(mask.sum())
    if n < 8:
        mask = _points_in_obb(lidar[:, :3], bbox, pad=2.0)
        n = int(mask.sum())
    return lidar[~mask], n


def _uav_overhead_mask(pts, bbox, xy_scale=1.25, pad_xy=2.0, pad_z=1.5):
    """UAV 俯视：放大 BEV 脚印并加高，盖住车顶回波。"""
    x, y, z, l, w, h, yaw = [float(v) for v in bbox[:7]]
    l = abs(l) * float(xy_scale)
    w = abs(w) * float(xy_scale)
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    lz = pts[:, 2] - z
    return ((np.abs(lx) <= l / 2.0 + pad_xy) &
            (np.abs(ly) <= w / 2.0 + pad_xy) &
            (lz >= -pad_z) & (lz <= h + pad_z))


def _fill_ground_footprint(lidar, bbox, n=32, pad_xy=2.0, xy_scale=1.25):
    """删点后在脚印里铺稀疏地面点，避免空洞被当成物体。"""
    lidar = np.asarray(lidar)
    if bbox is None or n <= 0:
        return lidar
    x, y, z, l, w, h, yaw = [float(v) for v in bbox[:7]]
    l = abs(l) * float(xy_scale) + 2.0 * float(pad_xy)
    w = abs(w) * float(xy_scale) + 2.0 * float(pad_xy)
    rng = np.random.RandomState(0)
    xs = rng.uniform(-l / 2.0, l / 2.0, int(n))
    ys = rng.uniform(-w / 2.0, w / 2.0, int(n))
    c, s = np.cos(yaw), np.sin(yaw)
    xyz = np.stack([
        c * xs - s * ys + x,
        s * xs + c * ys + y,
        np.full(int(n), z, dtype=np.float64),
    ], 1)
    return _append_xyz(lidar, xyz)


def _erase_uav_overhead(lidar, bbox, pad_xy=2.0, pad_z=1.5, fill_n=32):
    """UAV-only 点云 remove：俯视 OBB 硬删 + 地面填充。不用车载 AdvShape。"""
    lidar = np.asarray(lidar)
    if lidar.ndim != 2 or lidar.shape[0] == 0 or bbox is None:
        return lidar, 0
    mask = _uav_overhead_mask(lidar[:, :3], bbox, pad_xy=pad_xy, pad_z=pad_z)
    n = int(mask.sum())
    if n < 8:
        mask = _uav_overhead_mask(
            lidar[:, :3], bbox, xy_scale=1.5, pad_xy=pad_xy + 1.5, pad_z=pad_z + 1.0)
        n = int(mask.sum())
    kept = lidar[~mask]
    if kept.shape[0] == 0:
        kept = _blank_lidar(lidar)
    if fill_n > 0:
        kept = _fill_ground_footprint(kept, bbox, n=fill_n, pad_xy=pad_xy)
    return kept, n


def _plan_remove_targets(multi_frame_case, frame_ids, ego_id, att_id, clean,
                         require_uav_pts=0, min_range=3.0, hold_position=True,
                         ego_only_list=None, uav_only_list=None, prefer_uav_gap=False,
                         max_ego_score=None, ensure_every_frame=True):
    """选 remove 目标并按帧跟踪。

    ensure_every_frame=True（默认）: 每帧至少打一个目标。优先机强车弱 /
    跟踪；跟丢则逐级放宽；仍没有则 force-det / GT。只有场景里完全没有
    融合框也没有 Car GT 时才会 None。
    """
    if clean is None or len(clean) == 0:
        dbg("[remove] 无 clean 检测")
        return None, None, None

    world_tgt = None
    pred_ego = None
    target_att, target_ego = [], []
    reset_perturbation = [False] * len(multi_frame_case)
    n_forced = 0
    for fi, f in enumerate(frame_ids):
        att = multi_frame_case[f][att_id]
        ego = multi_frame_case[f][ego_id]
        boxes, scores = clean[fi]
        chosen = None
        snapped = False
        switched = False
        dmin = float("nan")
        n_uav = -1
        if pred_ego is not None and boxes is not None and len(boxes) > 0:
            boxes = np.asarray(boxes)
            scores = np.asarray(scores).reshape(-1)
            dist = np.hypot(boxes[:, 0] - pred_ego[0], boxes[:, 1] - pred_ego[1])
            cand = [j for j in range(len(boxes)) if scores[j] >= 0.3]
            for j in sorted(cand, key=lambda k: dist[k]):
                if float(dist[j]) > 15.0:
                    break
                if float(np.hypot(boxes[j][0], boxes[j][1])) < min_range:
                    continue
                n_uav, _ = _uav_points_on_ego_box(
                    multi_frame_case[f], ego_id, att_id, boxes[j])
                # Tracking snap: prefer UAV pts, but still allow when ensuring
                if require_uav_pts > 0 and n_uav < require_uav_pts and not ensure_every_frame:
                    continue
                if (
                    max_ego_score is not None
                    and ego_only_list is not None
                    and not ensure_every_frame
                ):
                    eo = ego_only_list[fi]
                    if _score_near(eo, boxes[j]) > float(max_ego_score):
                        continue
                dmin = float(dist[j])
                chosen = boxes[j]
                snapped = True
                break
        if boxes is not None and len(boxes) > 0:
            eo = ego_only_list[fi] if ego_only_list is not None else None
            uo = uav_only_list[fi] if uav_only_list is not None else None
            new_box, new_score, new_src = _select_remove_target(
                boxes, scores, multi_frame_case[f], ego_id, att_id,
                require_uav_pts=require_uav_pts, min_range=min_range,
                ego_only=eo, uav_only=uo, prefer_uav_gap=prefer_uav_gap,
                max_ego_score=max_ego_score, ensure_one=bool(ensure_every_frame),
            )
            if new_box is not None and (
                float(new_score) >= 0.3 or str(new_src).startswith("force")
            ):
                n_new, _ = _uav_points_on_ego_box(
                    multi_frame_case[f], ego_id, att_id, new_box)
                take = False
                if new_src == "uav-over-ego":
                    if chosen is None:
                        take = True
                    else:
                        gap = float(np.hypot(chosen[0] - new_box[0], chosen[1] - new_box[1]))
                        take = gap > 6.0
                elif chosen is None:
                    take = True
                if take:
                    if chosen is not None:
                        switched = True
                        reset_perturbation[f] = True
                        dbg("[remove] frame {} 改打 ({:.1f},{:.1f}) src={}".format(
                            f, new_box[0], new_box[1], new_src))
                    else:
                        dbg("[remove] frame {} 选 ({:.1f},{:.1f}) src={}".format(
                            f, new_box[0], new_box[1], new_src))
                    chosen = new_box
                    n_uav = n_new
                    if new_src == "uav-over-ego":
                        snapped = False
                    if str(new_src).startswith("force") or "relax" in str(new_src) or "strong" in str(new_src):
                        n_forced += 1
        if chosen is None:
            if hold_position and world_tgt is not None:
                chosen = bbox_map_to_sensor(world_tgt, ego["lidar_pose"])
                n_uav, _ = _uav_points_on_ego_box(
                    multi_frame_case[f], ego_id, att_id, chosen)
                dbg("[remove] frame {} 沿用预测 ({:.1f},{:.1f})".format(
                    f, chosen[0], chosen[1]))
            elif ensure_every_frame:
                forced, fsc, fsrc = _force_any_remove_box(
                    boxes, scores, multi_frame_case[f], ego_id, att_id,
                    min_range=min_range, prefer_near=pred_ego,
                )
                if forced is not None:
                    chosen = forced
                    n_forced += 1
                    dbg("[remove] frame {} {} ({:.1f},{:.1f})".format(
                        f, fsrc, chosen[0], chosen[1]))
                else:
                    dbg("[remove] frame {} skip (no det/GT)".format(f))
                    target_ego.append(None)
                    target_att.append(None)
                    if world_tgt is not None:
                        pred_ego = bbox_map_to_sensor(world_tgt, ego["lidar_pose"])
                    continue
            else:
                dbg("[remove] frame {} skip".format(f))
                target_ego.append(None)
                target_att.append(None)
                if world_tgt is not None:
                    pred_ego = bbox_map_to_sensor(world_tgt, ego["lidar_pose"])
                continue
        pred_ego = np.asarray(chosen, dtype=np.float64)
        world_tgt = bbox_sensor_to_map(pred_ego, ego["lidar_pose"])
        if n_uav < 0:
            n_uav, _ = _uav_points_on_ego_box(
                multi_frame_case[f], ego_id, att_id, pred_ego)
        tag = "switch" if switched else ("snap" if snapped else "pick")
        dbg("[remove] frame {} {} ({:.1f},{:.1f}) uav_pts={}".format(
            f, tag, pred_ego[0], pred_ego[1], n_uav))
        target_ego.append(pred_ego)
        target_att.append(_transform_bbox(pred_ego, ego["lidar_pose"], att["lidar_pose"]))
    if not any(t is not None for t in target_ego):
        dbg("[remove] 全程无可打目标")
        return None, None, None
    n_attack = sum(1 for t in target_ego if t is not None)
    n_skip = len(target_ego) - n_attack
    first = next(t for t in target_ego if t is not None)
    if not VERBOSE:
        log("  目标 ({:.1f},{:.1f}) | 攻击 {} 帧 / 跳过 {} 帧 | 回退选 {}".format(
            first[0], first[1], n_attack, n_skip, n_forced))
    return target_att, target_ego, reset_perturbation


def run_remove_early_points(perception, dataset, multi_frame_case, frame_ids,
                            ego_id, att_id, dense=3, also_ego=False, clean=None,
                            remove_mode="box", remove_select="ensure"):
    """点云级 remove。

    remove_mode:
      - box（默认）：UAV 俯视 OBB 硬删 + 地面填充，不用车载 AdvShape
      - adv：Zhang 射线 / AdvShape，再补一层框内删点

    remove_select（选目标策略）:
      - ensure（默认）：每帧至少 1 个目标；机强车弱优先，选不到则逐级回退
        （含 ego>0.35 / GT force）
      - prefer：宽松。融合 P≥0.3、距离≥8m、UAV 框内点数≥15 即可；
        **不要求**机强车弱，**不筛** ego-only>0.35。可跳帧；
        跟踪沿用；不 force GT。UAV 点数不够则该帧 skip。
    """
    remove_mode = str(remove_mode or "box").lower()
    if remove_mode not in ("box", "adv"):
        raise ValueError("remove_mode must be 'box' or 'adv', got {!r}".format(remove_mode))
    remove_select = str(remove_select or "ensure").lower()
    if remove_select not in ("ensure", "prefer"):
        raise ValueError(
            "remove_select must be 'ensure' or 'prefer', got {!r}".format(remove_select)
        )
    ensure_every = remove_select == "ensure"
    dbg("计算 ego-only / UAV-only ...")
    ego_only_list, uav_only_list = run_split_views(
        perception, multi_frame_case, frame_ids, ego_id, att_id)
    if ensure_every:
        target_att, target_ego, _ = _plan_remove_targets(
            multi_frame_case, frame_ids, ego_id, att_id, clean,
            require_uav_pts=15, min_range=8.0,
            hold_position=True,
            ego_only_list=ego_only_list, uav_only_list=uav_only_list,
            prefer_uav_gap=True, max_ego_score=0.35,
            ensure_every_frame=True)
    else:
        # prefer: no UAV-over-ego gate, no max_ego_score; skip if no UAV-visible fused det
        target_att, target_ego, _ = _plan_remove_targets(
            multi_frame_case, frame_ids, ego_id, att_id, clean,
            require_uav_pts=15, min_range=8.0,
            hold_position=True,
            ego_only_list=ego_only_list, uav_only_list=uav_only_list,
            prefer_uav_gap=False, max_ego_score=None,
            ensure_every_frame=False)
    if target_att is None:
        return None, None, None

    if remove_mode == "adv":
        positions = [None] * len(multi_frame_case)
        for fi, f in enumerate(frame_ids):
            positions[f] = target_att[fi]
        attacker = LidarRemoveEarlyAttacker(dataset, advshape=1, dense=dense, sync=0)
        attacked_case, infos = attacker.run(multi_frame_case, {
            "frame_ids": frame_ids,
            "attacker_vehicle_id": att_id,
            "bboxes": positions,
        })
        log("[early-remove] mode=adv (Zhang ray+AdvShape) select={}".format(remove_select))
    else:
        attacked_case = copy.deepcopy(multi_frame_case)
        infos = [{} for _ in frame_ids]
        log("[early-remove] mode=box (UAV overhead OBB delete, no car mesh) select={}".format(
            remove_select))

    attacked = []
    frame_rows = []
    for fi, f in enumerate(frame_ids):
        frame = attacked_case[f]
        tgt = target_ego[fi]
        if tgt is None:
            pb, ps = perception.run(frame, ego_id)
            attacked.append(_pack_dets(pb, ps))
            frame_rows.append((f, None, 0, 0, 0, 0.0, 0.0))
            continue
        info = infos[fi] if fi < len(infos) else {}
        n_rep = 0
        if info.get("replace_indices") is not None:
            n_rep = int(np.asarray(info["replace_indices"]).reshape(-1).shape[0])

        if remove_mode == "box":
            frame[att_id]["lidar"], n_uav = _erase_uav_overhead(
                frame[att_id]["lidar"], target_att[fi])
        else:
            frame[att_id]["lidar"], n_uav = _erase_box_points(
                frame[att_id]["lidar"], target_att[fi])
        n_ego = 0
        if also_ego:
            if remove_mode == "box":
                frame[ego_id]["lidar"], n_ego = _erase_uav_overhead(
                    frame[ego_id]["lidar"], tgt)
            else:
                frame[ego_id]["lidar"], n_ego = _erase_box_points(
                    frame[ego_id]["lidar"], tgt)
        dbg("[early-remove] f{} mode={} ray={} del_uav={} del_ego={}".format(
            f, remove_mode, n_rep, n_uav, n_ego))

        pb, ps = perception.run(frame, ego_id)
        attacked.append(_pack_dets(pb, ps))
        cl = _score_near(clean[fi] if clean else None, tgt)
        atk = _score_near(attacked[-1], tgt)
        frame_rows.append((f, tgt, n_rep, n_uav, n_ego, cl, atk))

    if frame_rows and not VERBOSE:
        active = [r for r in frame_rows if r[1] is not None]
        if active:
            avg_del = int(np.mean([r[3] for r in active]))
            ok = sum(1 for r in active if r[5] >= 0.3 and r[6] < 0.3)
            valid = sum(1 for r in active if r[5] >= 0.3)
            log("  删点 ~{} / 帧 | 射线替换 {} | 即时成功 {}/{}".format(
                avg_del, sum(r[2] for r in active), ok, valid))
    elif VERBOSE and frame_rows:
        log("  f   clean  attack  del_uav  ray")
        for f, tgt, n_rep, n_uav, n_ego, cl, atk in frame_rows:
            if tgt is None:
                log("  {:>2}  skip".format(f))
            else:
                log("  {:>2}  {:.3f}  {:.3f}  {:>4}  {:>3}".format(
                    f, cl, atk, n_uav, n_rep))
    return attacked, target_ego, attacked_case


def run_remove_online(perception, dataset, multi_frame_case, frame_ids, ego_id, att_id, iters, clean=None):
    target_att, target_ego, reset_perturbation = _plan_remove_targets(
        multi_frame_case, frame_ids, ego_id, att_id, clean)
    if target_att is None:
        return None, None

    attacker = LidarRemoveIntermediateAttacker(
        perception, dataset, step=iters, sync=0, init=False, online=True)
    _, info = attacker.run(multi_frame_case, {
        "frame_ids": frame_ids,
        "attacker_vehicle_id": att_id,
        "victim_vehicle_id": ego_id,
        "bboxes": target_att,
        "reset_perturbation": reset_perturbation,
        "attack_info": [{} for _ in range(len(multi_frame_case))],
    })
    attacked = [(info[f][ego_id]["pred_bboxes"], info[f][ego_id]["pred_scores"])
                for f in frame_ids]
    return attacked, target_ego


def main():
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CKPTS.keys()), default="attfuse")
    parser.add_argument("--mode", choices=["remove", "spoof"], default="remove")
    parser.add_argument("--level", choices=["intermediate", "early"], default="intermediate",
                        help="intermediate=特征扰动; early=只改点云，再提BEV并中间融合")
    parser.add_argument("--dense", type=int, default=3, choices=[0, 1, 2, 3],
                        help="仅 --level early：点云注入密度模式")
    parser.add_argument(
        "--early_remove_mode",
        choices=["box", "adv"],
        default="box",
        help="early remove: box=UAV overhead OBB delete; adv=Zhang ray+AdvShape",
    )
    parser.add_argument("--also_ego", action="store_true",
                        help="early：spoof 把幽灵点也写入 ego；remove 同时删 ego 框内点")
    parser.add_argument("--sanity", choices=["off", "ego_clone", "near_clone"], default="off",
                        help="off=点云攻击; near_clone=把已检出的车复制到旁边8m; ego_clone=复制到幽灵位置")
    parser.add_argument("--ghost", choices=["near", "danger"], default="near",
                        help="early: near=沿已检车辆平移(与near_clone同几何,写入UAV); danger=旧(15,-3)")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印逐帧调试信息")
    parser.add_argument("--data", default=os.path.join(V2U4REAL_ROOT, "v2u4real"))
    args = parser.parse_args()
    VERBOSE = args.verbose

    ego_id, att_id = "1", "2"
    frame_ids = list(range(10))

    dataset = V2U4RealDataset(root_path=os.path.join(args.data, "val"), mode="val")
    perception = load_model(args.model)

    log("=== {} | {} | {} | case={} | iters={} ===".format(
        args.model, args.mode, args.level, args.case,
        args.iters if args.level == "intermediate" else "n/a"))

    if args.mode == "spoof":
        multi_frame_case = dataset.get_case(args.case, tag="multi_frame")
        meta = dataset.cases["multi_frame"][args.case]
        log("场景: {} | 帧数: {}".format(meta["scenario_name"], len(frame_ids)))
        log("clean 基线 ...")
        clean, gt_list = run_clean(perception, multi_frame_case, frame_ids, ego_id)
        log("攻击 ...")
        if args.level == "early":
            attacked, injected, _ = run_spoof_early_points(
                perception, dataset, multi_frame_case, frame_ids, ego_id, att_id,
                dense=args.dense, sync=0, also_ego=args.also_ego,
                sanity=args.sanity, clean=clean, ghost=args.ghost)
            mode_name = "spoof-early"
        else:
            attacked, injected = run_spoof_online(
                perception, dataset, multi_frame_case, frame_ids, ego_id, att_id, args.iters)
            mode_name = "spoof-intermediate"
        print_spoof_report(
            evaluate_spoof(clean, attacked, gt_list, injected),
            mode_name=mode_name, verbose=VERBOSE)
    else:
        case = args.case
        attacked = target_ego = None
        clean = gt_list = None
        for _ in range(20):
            multi_frame_case = dataset.get_case(case, tag="multi_frame")
            meta = dataset.cases["multi_frame"][case]
            log("场景: {} | case={} | 帧数: {}".format(
                meta["scenario_name"], case, len(frame_ids)))
            log("clean 基线 ...")
            clean, gt_list = run_clean(perception, multi_frame_case, frame_ids, ego_id)
            log("选目标 + 攻击 ...")
            if args.level == "early":
                attacked, target_ego, _ = run_remove_early_points(
                    perception, dataset, multi_frame_case, frame_ids, ego_id, att_id,
                    dense=args.dense, also_ego=args.also_ego, clean=clean,
                    remove_mode=args.early_remove_mode)
            else:
                attacked, target_ego = run_remove_online(
                    perception, dataset, multi_frame_case, frame_ids, ego_id, att_id,
                    args.iters, clean=clean)
            if attacked is not None:
                break
            log("  case {} 无目标，尝试 case {} ...".format(case, case + 1))
            case += 1
        if attacked is None:
            log("20 个 case 均无可用目标，退出。")
            return
        mode_name = "remove-early" if args.level == "early" else "remove-intermediate"
        print_remove_report(
            evaluate_remove(clean, attacked, gt_list, target_ego),
            mode_name=mode_name, verbose=VERBOSE)


if __name__ == "__main__":
    main()
