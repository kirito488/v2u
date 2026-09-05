# -*- coding: utf-8 -*-
"""
自定义模型 MyPillar —— 这是一个示例，展示如何接入你自己的模型。
假设你的创新点是：在 BEV backbone 之后加了一个额外的 refinement 卷积层。
"""

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class MyPillar(nn.Module):
    """
    自定义 PointPillar 变体：
    - 和原始 PointPillar 一样使用 PillarVFE + Scatter + BaseBEVBackbone
    - 创新点：在 backbone 之后加了一个 refinement 卷积块 (extra_conv)
    """
    def __init__(self, args):
        super(MyPillar, self).__init__()

        # ========== 1. Pillar Feature Encoder ==========
        # 将每个 pillar 内的点云编码为特征向量
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,         # 每个点有 (x, y, z, intensity) 4 个特征
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )

        # ========== 2. Scatter ==========
        # 将 pillar 特征投影到 2D BEV 平面
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        # ========== 3. BEV Backbone ==========
        # 2D 卷积 backbone，输出 3 个尺度的特征 (各 128 通道 → 拼接为 384 通道)
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # ========== 4. 你的创新：Refinement 卷积块 ==========
        # 在 backbone 输出后加一层额外处理
        self.extra_conv = nn.Sequential(
            nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(128 * 3),
            nn.ReLU(),
        )

        # ========== 5. 检测头 ==========
        # 分类头：输出每个 anchor 的置信度
        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'], kernel_size=1)
        # 回归头：输出每个 anchor 的 7 个 box 参数 (x, y, z, w, l, h, yaw)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'], kernel_size=1)

    def forward(self, data_dict):
        """
        前向传播 —— 这是框架调用的入口。

        输入 data_dict 包含:
          - 'processed_lidar': dict
              - 'voxel_features':  (N_voxels, max_points_per_voxel, 4)
              - 'voxel_coords':     (N_voxels, 3)
              - 'voxel_num_points': (N_voxels,)

        输出必须是一个 dict:
          - 'psm': 分类预测 (B, anchor_num, H, W)
          - 'rm':  回归预测 (B, 7*anchor_num, H, W)
        """
        # 从 data_dict 中取出体素化后的点云
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
        }

        # Step 1: Pillar VFE → 每个 pillar → 特征向量
        batch_dict = self.pillar_vfe(batch_dict)

        # Step 2: Scatter → pillar 特征散布到 BEV 网格
        batch_dict = self.scatter(batch_dict)

        # Step 3: BEV Backbone → 多尺度特征提取
        batch_dict = self.backbone(batch_dict)

        # Step 4: 你的创新模块 —— 额外 refinement
        spatial_features_2d = batch_dict['spatial_features_2d']
        spatial_features_2d = self.extra_conv(spatial_features_2d)

        # Step 5: 检测头
        psm = self.cls_head(spatial_features_2d)   # (B, anchor_num, H, W)
        rm = self.reg_head(spatial_features_2d)     # (B, 7*anchor_num, H, W)

        output_dict = {
            'psm': psm,
            'rm': rm,
        }

        return output_dict
