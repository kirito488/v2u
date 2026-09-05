# -*- coding: utf-8 -*-
"""
自定义 Intermediate Fusion 模型 —— 完整示例
假设你的创新点是：在融合阶段使用多头注意力代替原来的 V2VNet Fusion。
"""

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class MyIntermediate(nn.Module):
    """
    自定义 Intermediate Fusion 模型。

    核心流程：
      车和无人机各自体素化 → 各自做 PillarVFE+Scatter → 各自得到 BEV 特征
      → 你的融合模块把两份 BEV 特征融合成一份 → Backbone → 检测头

    和 Early Fusion(my_pillar.py) 的关键区别：
      - Early：点云先拼接，再一次性体素化 → 一份特征直接进 backbone
      - Intermediate：各自体素化 → 两份特征先融合，再进 backbone
    """
    def __init__(self, args):
        super(MyIntermediate, self).__init__()

        self.max_cav = args['max_cav']

        # ========== 1. Pillar VFE（每辆车独立调用）==========
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )

        # ========== 2. Scatter（每辆车独立调用）==========
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        # ========== 3. ★ 你的融合模块（这是 intermediate 的核心）==========
        # 输入：多辆车的 BEV 特征拼在一起 (B, C, H, W)
        # 输出：融合后的一张特征图 (B, C, H, W)
        in_channels = args['base_bev_backbone']['num_filters'][0]  # 64
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        # ========== 4. BEV Backbone ==========
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # ========== 5. 检测头 ==========
        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'], kernel_size=1)

    def forward(self, data_dict):
        """
        输入：data_dict['ego']，结构和 early fusion 完全不同。

        关键区别：
          processed_lidar 里的每个字段是 LIST，不是单个 tensor。
          因为每个 agent 的点云是各自体素化的，没有预先拼接。
        """
        # ============================================================
        # 解包：全是 list，因为 intermediate 的 data_dict 不提前拼点云
        # ============================================================
        voxel_features = data_dict['processed_lidar']['voxel_features']    # [tensor, tensor]
        voxel_coords = data_dict['processed_lidar']['voxel_coords']        # [tensor, tensor]
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'] # [tensor, tensor]
        record_len = data_dict['record_len']                                # tensor([2])

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len,
        }

        # ============================================================
        # Step 1: PillarVFE —— 内部自动遍历 list，每个 agent 各自编码
        #   输入 [agent1的体素, agent2的体素]
        #   输出 dict: {'pillar_features': 拼在一起的体素特征}
        #   内部逻辑：for each agent → PointNet → cat 在一起
        # ============================================================
        batch_dict = self.pillar_vfe(batch_dict)

        # ============================================================
        # Step 2: Scatter —— 内部自动遍历 list，每个 agent 各自投影到 BEV
        #   输入：拼在一起的体素特征 + voxel_coords
        #   输出：'spatial_features'：(B, 64, H, W)
        #   内部逻辑：遍历每个 agent 的 voxel_coords，填入同一张 BEV 格网
        #            每个 agent 的体素落在自己的坐标区域，互不重叠或自然重叠
        # ============================================================
        batch_dict = self.scatter(batch_dict)
        spatial_features = batch_dict['spatial_features']   # (1, 64, H, W)

        # ============================================================
        # Step 3: ★ 你的融合模块 ★
        #   把拼接后的 BEV 特征按 agent 拆分，各自做变换对齐，再融合
        # ============================================================
        split_features = self.regroup(spatial_features, record_len)
        # split_features = [agent1的BEV特征 (1, 64, H, W),
        #                   agent2的BEV特征 (1, 64, H, W)]
        # 每份是同一个 BEV 图的不同区域（或重叠区域的不同视角）

        if len(split_features) >= 2:
            # 简单融合策略：沿通道拼接 → 卷积降维
            fused_feature = torch.cat(split_features[:2], dim=1)  # (1, 128, H, W)
            fused_feature = self.fusion_conv(fused_feature)       # (1, 64, H, W)
        else:
            # 只有一个 agent（比如测试时某帧只有车辆端数据）
            fused_feature = split_features[0]

        # ============================================================
        # Step 4: Backbone（和 early fusion 一样）
        # ============================================================
        batch_dict['spatial_features_2d'] = fused_feature
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        # ============================================================
        # Step 5: 检测头（和 early fusion 完全一样）
        # ============================================================
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        return {'psm': psm, 'rm': rm}

    @staticmethod
    def regroup(x, record_len):
        """
        把拼在一起的多 agent 特征按 record_len 拆分。

        例：record_len = [2, 3]（batch中第1帧2个agent，第2帧3个agent）
            → x 切成 5 份 → 按帧分回 [(ag1,ag2), (ag1,ag2,ag3)]
        """
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
