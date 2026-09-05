# -*- coding: utf-8 -*-
"""
微调大模型 —— 完整示例
用预训练的 Swin Transformer 替代原有的小 backbone，只训练检测头。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class MyLargeFinetune(nn.Module):
    """
    微调大模型做 3D 检测。

    结构：
      固定部分（冻结）                可训练部分
      ─────────────               ──────────
      PillarVFE + Scatter         融合模块
      预训练 Swin Transformer      检测头 (cls_head + reg_head)

    你训练时只更新右边两列，左边两列不动。
    """
    def __init__(self, args):
        super(MyLargeFinetune, self).__init__()

        # ================================================================
        # ① PillarVFE + Scatter：固定前处理（冻结）
        #    导入 timm 里预训练的 Swin Transformer 作为 backbone
        # ================================================================
        try:
            import timm
        except ImportError:
            raise ImportError("请先 pip install timm")

        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        # ================================================================
        # ② 融合模块（可训练）
        # ================================================================
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # ================================================================
        # ③ ★ 加载预训练大模型 backbone ★
        #    因为预训练模型输入是 3 通道 RGB 图，BEV 是 64 通道
        #    所以加一个适配层把 64 通道转成 3 通道
        # ================================================================
        self.input_adapter = nn.Conv2d(64, 3, kernel_size=1)  # 通道适配器

        # 从 timm 加载预训练的 Swin-Tiny（ImageNet 预训练权重）
        self.large_backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',   # Swin Transformer
            pretrained=True,                   # ★ 加载 ImageNet 预训练权重
            features_only=True,                # 输出多尺度特征
        )

        # ================================================================
        # ④ ★ 冻结大模型 ★
        # ================================================================
        for param in self.large_backbone.parameters():
            param.requires_grad = False

        # ================================================================
        # ⑤ 检测头（可训练）
        # ================================================================
        # Swin-Tiny 多尺度输出通道：[96, 192, 384, 768]
        # 取后 3 个尺度上采样拼接：192+384+768 = 1344
        self.neck = nn.Sequential(
            nn.Conv2d(1344, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.cls_head = nn.Conv2d(384, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(384, 7 * args['anchor_number'], kernel_size=1)

    def forward(self, data_dict):
        # ================================================================
        # Step 1: 体素 → BEV（固定流程，不训练）
        # ================================================================
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len,
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        spatial_features = batch_dict['spatial_features']     # (N, 64, H, W)

        # ================================================================
        # Step 2: 融合（可训练）
        # ================================================================
        split = self.regroup(spatial_features, record_len)
        if len(split) >= 2:
            fused = torch.cat(split[:2], dim=1)               # (1, 128, H, W)
            fused = self.fusion_conv(fused)                   # (1, 64, H, W)
        else:
            fused = split[0]

        # ================================================================
        # Step 3: 通道适配（64 → 3），再插值到 224×224 匹配预训练模型
        # ================================================================
        x = self.input_adapter(fused)                         # (1, 3, H, W)
        x = F.interpolate(x, size=(224, 224))                 # → (1, 3, 224, 224)

        # ================================================================
        # Step 4: ★ 大模型提取特征（冻结，不训练）★
        # ================================================================
        with torch.no_grad():
            multi_scale = self.large_backbone(x)
        # multi_scale: list of 4 tensors
        #   [0]: (1, 96, 56, 56)
        #   [1]: (1, 192, 28, 28)
        #   [2]: (1, 384, 14, 14)
        #   [3]: (1, 768, 7, 7)

        # 上采样后 3 层到同一尺寸再拼接
        f1 = F.interpolate(multi_scale[1], size=(200, 252))
        f2 = F.interpolate(multi_scale[2], size=(200, 252))
        f3 = F.interpolate(multi_scale[3], size=(200, 252))
        features = torch.cat([f1, f2, f3], dim=1)             # (1, 1344, 200, 252)

        # ================================================================
        # Step 5: 检测头（可训练）
        # ================================================================
        features = self.neck(features)                        # (1, 384, 200, 252)
        psm = self.cls_head(features)                         # (1, 2, 200, 252)
        rm = self.reg_head(features)                          # (1, 14, 200, 252)

        return {'psm': psm, 'rm': rm}

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
