"""
V2U4Real 中间攻击演示脚本

作用：把 AdvCollaborativePerception 的中间攻击用到 V2U4Real 的
      AttFuse / Where2comm / CoAlign 三个模型上。

用法:
    python scripts/attack_v2u4real.py --model attfuse   --mode remove --iters 20
    python scripts/attack_v2u4real.py --model where2comm --mode spoof  --iters 20
    python scripts/attack_v2u4real.py --model coalign   --mode remove --iters 50

参数说明:
    --model  三个模型名之一: attfuse / where2comm / coalign
    --mode   remove(让车看不见目标) 或 spoof(让车看到幽灵目标)
    --iters  攻击优化迭代次数（越大效果越好、越慢）
"""
import os
import sys
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# 路径配置：改成你自己机器上的路径
# ---------------------------------------------------------------------------
V2U4REAL_ROOT = "D:/agent_project/V2U4Real-main/V2U4Real-main"
CKPT_ROOT = "D:/agent_project/V2U4Real-main/checkpoints"

# 让 import opencood 能找到 V2U4Real 的 opencood 代码
root = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../"))
sys.path.insert(0, root)
sys.path.insert(0, V2U4REAL_ROOT)

from mvp.data.v2u4real_dataset import V2U4RealDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor

MODEL_CKPTS = {
    "attfuse":    os.path.join(CKPT_ROOT, "attfuse_checkpoint"),
    "where2comm": os.path.join(CKPT_ROOT, "where2comm_checkpoint"),
    "coalign":    os.path.join(CKPT_ROOT, "coalign_checkpoint"),
}


def load_model(name):
    """用 V2U4Real 的 checkpoint 构造一个感知对象。"""
    return OpencoodPerception(
        fusion_method="intermediate",
        model_name=name,
        model_dir=MODEL_CKPTS[name],
        opencood_root=V2U4REAL_ROOT,
        root_dir=os.path.join(V2U4REAL_ROOT, "v2u4real", "train"),
        validate_dir=os.path.join(V2U4REAL_ROOT, "v2u4real", "val"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CKPTS.keys()), default="attfuse")
    parser.add_argument("--mode", choices=["remove", "spoof"], default="remove")
    parser.add_argument("--case", type=int, default=0, help="第几个 multi_frame case")
    parser.add_argument("--frame", type=int, default=0, help="case 里的第几帧")
    parser.add_argument("--iters", type=int, default=20, help="攻击迭代次数")
    parser.add_argument("--data", default=os.path.join(V2U4REAL_ROOT, "v2u4real"))
    args = parser.parse_args()

    ego_id = "1"   # 车 = 受害者 ego
    att_id = "2"   # 无人机 = 攻击者 attacker

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    dataset = V2U4RealDataset(root_path=os.path.join(args.data, "val"), mode="val")
    print("[1/5] 加载数据完成，multi_frame case 数 =", dataset.case_number("multi_frame"))

    # ------------------------------------------------------------------
    # 2. 加载模型（这一步会构建数据集 + 读 checkpoint）
    # ------------------------------------------------------------------
    print("[2/5] 加载模型:", args.model)
    perception = load_model(args.model)

    # ------------------------------------------------------------------
    # 3. 取一帧
    # ------------------------------------------------------------------
    multi_frame_case = dataset.get_case(args.case, tag="multi_frame")
    case = multi_frame_case[args.frame]
    meta = dataset.cases["multi_frame"][args.case]
    print("[3/5] 场景 =", meta["scenario_name"],
          " 帧 =", meta["frame_ids"][args.frame])

    # ------------------------------------------------------------------
    # 4. 攻击前：正常检测（baseline）
    # ------------------------------------------------------------------
    pred_before, score_before = perception.run(case, ego_id)
    print("[4/5] 攻击前检测到 {} 个目标".format(len(pred_before)))

    # ------------------------------------------------------------------
    # 5. 选攻击目标
    # ------------------------------------------------------------------
    att_case = case[att_id]
    if args.mode == "remove":
        if att_case["gt_bboxes"].shape[0] == 0:
            print("攻击车看不到任何目标，换一个 case 试试")
            return
        # 选离 ego 最近的一辆车作为 remove 目标
        best_i, best_d = -1, 1e9
        for i in range(att_case["gt_bboxes"].shape[0]):
            bbox_map = bbox_sensor_to_map(att_case["gt_bboxes"][i], att_case["lidar_pose"])
            d = float(np.linalg.norm(bbox_map[:2] - case[ego_id]["lidar_pose"][:2]))
            if d < best_d:
                best_d, best_i = d, i
        # 把目标框转到 ego 坐标系（攻击代码要求这个格式）
        target_bbox_ego = bbox_map_to_sensor(
            bbox_sensor_to_map(att_case["gt_bboxes"][best_i], att_case["lidar_pose"]),
            case[ego_id]["lidar_pose"])
        print("[5/5] remove 目标 object_id =", att_case["object_ids"][best_i],
              " 距 ego {:.1f} m".format(best_d))
    else:
        # spoof：在 ego 前方 30m 偏右 5m 放一个幽灵车框
        target_bbox_ego = np.array([30, 5, 0, 4.7, 1.9, 1.7, 0])
        print("[5/5] spoof 幽灵车位置 = ego 前方 30m")

    # ------------------------------------------------------------------
    # 6. 跑中间攻击（核心）
    # ------------------------------------------------------------------
    print("正在攻击（{} 次迭代）...".format(args.iters))
    result = perception.attack_intermediate(
        case, ego_id, att_id,
        max_perturb=3, lr=0.05, max_iteration=args.iters,
        bbox=target_bbox_ego, mode=args.mode, feature_size=5,
    )

    # ------------------------------------------------------------------
    # 7. 攻击后结果
    # ------------------------------------------------------------------
    pred_after, score_after = result["pred_bboxes"], result["pred_scores"]
    print("攻击后检测到 {} 个目标".format(len(pred_after)))

    if args.mode == "remove":
        from mvp.tools.iou import iou3d
        if len(pred_after) > 0:
            best_iou = max(iou3d(b, target_bbox_ego) for b in pred_after)
            print("攻击后目标位置最大 IoU = {:.3f}  (越小说明越成功)".format(best_iou))
        else:
            print("攻击后没有任何检测框（目标被完全移除）")
    else:
        from mvp.tools.iou import iou3d
        if len(pred_after) > 0:
            best_iou = max(iou3d(b, target_bbox_ego) for b in pred_after)
            print("攻击后目标位置最大 IoU = {:.3f}  (越大说明幽灵车越成功)".format(best_iou))
        else:
            print("攻击后没有检测到幽灵车")

    print("==== 完成 ====")


if __name__ == "__main__":
    main()
