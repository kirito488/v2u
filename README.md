# CollaborativePerceptionDefense

UAV-UGV 协同感知**防御**实现仓库。权威设计见 `V2U4Real-main/SYSTEM_DESIGN.md`。

本阶段只落地 **三源检测**（后续：关联 / 状态 / 共识 / 缓冲 / Beta 信任）。

## 与兄弟仓库的关系

| 路径 | 用途 |
|------|------|
| `AdvCollaborativePerception-master` | 攻击代码；本仓库复用其 `mvp/`（数据集 + OpenCOOD 封装） |
| `V2U4Real-main` | val 数据、checkpoints、OpenCOOD、`SYSTEM_DESIGN.md` |
| **本仓库** | 防御流水线 |

角色固定：Ego=`"1"`，UAV=`"2"`。检测框一律在 **ego 雷达系**。

## 三源检测在做什么

每帧并行得到三组 `(B, P, C)`：

| 源 | 做法 | 对齐攻击侧 |
|----|------|------------|
| **Ego** | 清空 UAV 点云后中间融合推理 | `run_split_views` ego-only |
| **UAV** | 清空 Ego 点云后中间融合推理 | `run_split_views` UAV-only |
| **Init** | 双路上正常中间融合 | `run_clean` |

- `B`：框 `[x,y,z,l,w,h,yaw]`
- `P`：检测头存在概率（score）
- `C`：感知置信度，由对应源点云落在框内的点数归一化得到（`n_ref=40` → C=1）

空点云用远处单点占位，避免 voxel 预处理崩溃（与攻击脚本一致）。

## 运行

依赖与攻击相同：conda `v2u4`，能 import 攻击仓 `mvp` 与 V2U4Real OpenCOOD。

```bash
# 本机 / Linux 项目根
cd CollaborativePerceptionDefense
python scripts/run_three_source.py --model attfuse --case 0
python scripts/run_three_source.py --model attfuse --case 0 --score_thres 0.3 -v
python scripts/run_three_source.py --model where2comm --case 1 --save outputs/case1.json
```

`--score_thres`：可选按 P 过滤（攻击评估常用 `0.3`；设计文档 θ_P=`0.5`）。默认不过滤，保留检测头全部输出。

路径解析顺序：Linux `/data/hzy/lxt/...` → 本机 `D:/agent_project/...`。

## 目录

```text
defense/
  paths.py          # 双仓路径
  geometry.py       # blank lidar / OBB 点计数
  confidence.py     # C = clip(n/n_ref)
  three_source.py   # ThreeSourceDetector
scripts/
  run_three_source.py
```

## 下一步（未实现）

数据关联（匈牙利）、Certain/Ambiguous/Tentative、ROBOSAC 共识、缓冲区、Beta `W_trust`。
