"""
V2U4Real 中间级多物体伪造攻击演示脚本（单帧）

作用：在每一帧独立地伪造 1-3 个"幽灵目标"。物体的尺寸/朝向借用数据集里
      真实 GT 物体（donor），位置选择车的危险区（正前方车道）或盲区
      （ego LiDAR 点稀疏/被遮挡处）。

用法:
    python scripts/attack_v2u4real_multi.py --model attfuse   --num 2 --iters 30
    python scripts/attack_v2u4real_multi.py --model where2comm --num 3 --iters 30
    python scripts/attack_v2u4real_multi.py --model coalign   --num 1 --iters 50

参数说明:
    --model   attfuse / where2comm / coalign
    --num     每帧伪造的物体数（1-3）
    --iters   攻击优化迭代次数（越大越好、越慢）
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

root = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../"))
sys.path.insert(0, root)
sys.path.insert(0, V2U4REAL_ROOT)

from mvp.data.v2u4real_dataset import V2U4RealDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_spoof_intermediate_multi_attacker import LidarSpoofIntermediateMultiAttacker
from mvp.tools.iou import iou3d

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CKPTS.keys()), default="attfuse")
    parser.add_argument("--case", type=int, default=0, help="第几个 multi_frame case")
    parser.add_argument("--frame", type=int, default=9, help="要攻击的帧（case 内索引）")
    parser.add_argument("--num", type=int, default=2, help="每帧伪造物体数（1-3）")
    parser.add_argument("--iters", type=int, default=30, help="攻击迭代次数")
    parser.add_argument("--data", default=os.path.join(V2U4REAL_ROOT, "v2u4real"))
    args = parser.parse_args()

    ego_id = "1"   # 车 = 受害者 ego
    att_id = "2"   # 无人机 = 攻击者 attacker

    dataset = V2U4RealDataset(root_path=os.path.join(args.data, "val"), mode="val")
    print("[1/5] 加载数据完成，multi_frame case 数 =", dataset.case_number("multi_frame"))

    print("[2/5] 加载模型:", args.model)
    perception = load_model(args.model)

    multi_frame_case = dataset.get_case(args.case, tag="multi_frame")
    meta = dataset.cases["multi_frame"][args.case]
    print("[3/5] 场景 =", meta["scenario_name"],
          " 帧 =", meta["frame_ids"][args.frame])

    # 攻击前基线（只针对目标帧）
    frame = multi_frame_case[args.frame]
    pred_before, score_before = perception.run(frame, ego_id)
    print("[4/5] 攻击前检测到 {} 个目标".format(len(pred_before)))

    # 构造多物体伪造攻击器并执行（单帧）
    attacker = LidarSpoofIntermediateMultiAttacker(
        perception, dataset, num_objects=args.num, step=args.iters)
    print("正在伪造 {} 个幽灵目标（{} 次迭代）...".format(args.num, args.iters))
    _, attack_info = attacker.run(multi_frame_case, {
        "frame_ids": [args.frame],
        "attacker_vehicle_id": att_id,
        "victim_vehicle_id": ego_id,
        "num_objects": args.num,
    })

    info = attack_info[args.frame][ego_id]
    injected = np.array(info["injected_bboxes"])
    pred_after = info["pred_bboxes"]
    print("[5/5] 攻击后检测到 {} 个目标".format(len(pred_after)))

    print("\n选中的注入位置 / donor 尺寸:")
    for i, (pos, size) in enumerate(zip(info["positions"], info["donor_sizes"])):
        yaw = injected[i][6]
        print("  目标 {}: 位置 ego({:6.1f}, {:6.1f}) 尺寸 lwh=({:.1f},{:.1f},{:.1f}) 朝向 {:.1f}°"
              .format(i, pos[0], pos[1], size[0], size[1], size[2], np.degrees(yaw)))

    print("\n各注入位置最大 IoU（越大说明幽灵目标越成功）:")
    for i, b in enumerate(injected):
        best = 0.0
        if len(pred_after) > 0:
            best = max(iou3d(p, b) for p in pred_after)
        print("  目标 {}: 最大 IoU = {:.3f}".format(i, best))

    print("\n==== 完成 ====")


if __name__ == "__main__":
    main()
