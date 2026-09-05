# -*- coding: utf-8 -*-
"""
方案：LLM 做 3D 目标检测

完整流程：
  点云 → 体素化(PillarVFE) → BEV(Scatter) → 压缩 → 转token → LLM(LoRA) → 检测头 → psm+rm

你需要：
  - GPU ≥ 16GB 显存（RTX 3090/4090/A6000）
  - pip install transformers peft accelerate bitsandbytes
  - HuggingFace 模型会下载到本地缓存，不联网也能用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class MyLLMDetector(nn.Module):
    """
    用 LLM 做 3D 检测的一个完整方案。

    结构一览：

      ┌──────────────────────────────────────────────┐
      │ ① 前处理（冻结，照抄不动）                      │
      │    PillarVFE → Scatter → BEV (64, 400, 504)   │
      ├──────────────────────────────────────────────┤
      │ ② 融合 + 压缩（可训练）                         │
      │    BEV → 融合 → Conv下采样 → (256, 50, 63)     │
      │    3150 个 token，每个 256 维                   │
      ├──────────────────────────────────────────────┤
      │ ③ Token 投影（可训练）                          │
      │    Linear(256, 1536) → 适配 LLM 输入维度         │
      ├──────────────────────────────────────────────┤
      │ ④ ★ LLM 处理（LoRA 微调）★                     │
      │    Qwen2.5-1.5B → 28 层 Transformer            │
      │    只训练 LoRA 参数（约 10M，冻结 LLM 主体）      │
      ├──────────────────────────────────────────────┤
      │ ⑤ 解码（可训练）                               │
      │    Linear(1536, 256) → 上采样 → (384, 200, 252) │
      ├──────────────────────────────────────────────┤
      │ ⑥ 检测头（可训练）                              │
      │    cls_head → psm,  reg_head → rm              │
      └──────────────────────────────────────────────┘

    训练时梯度流：
      LLM 主体: ✗ 冻结（不更新）
      LoRA 参数: ✓ 可训练
      其他所有模块: ✓ 可训练

    显存估算：
      Qwen2.5-1.5B fp16: ~3GB
      LoRA 参数: ~0.04GB
      其他模块: ~1GB
      训练总计: ~12-16GB（单卡 4090 可跑）
    """

    def __init__(self, args):
        super(MyLLMDetector, self).__init__()

        # ============================================================
        # ① 固定前处理：点云 → BEV（不可训练，照抄）
        # ============================================================
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        # ============================================================
        # ② 融合模块 + BEV 压缩（可训练）
        #    400×504 → 50×63（64 倍压缩，token 数量从 20 万降到 3150）
        # ============================================================
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.bev_compressor = nn.Sequential(
            # 400×504 → 200×252
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 200×252 → 100×126
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 100×126 → 50×63
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # ============================================================
        # ③ Token 投影器：BEV 特征通道 → LLM 隐藏维度（可训练）
        #    Qwen2.5-1.5B hidden_dim = 1536
        #    换其他 LLM 时改这个数字就行
        # ============================================================
        self.llm_hidden_dim = args.get('llm_hidden_dim', 1536)
        self.token_projector = nn.Linear(256, self.llm_hidden_dim)

        # ============================================================
        # ④ ★ 加载 LLM + LoRA ★
        # ============================================================
        self.llm = self._build_llm(args.get('llm_model_name', 'Qwen/Qwen2.5-1.5B'))

        # ============================================================
        # ⑤ Token 反投影 + 上采样解码器（可训练）
        #    LLM 输出 → 还原到 BEV 空间 → 匹配 backbone 输出尺寸
        # ============================================================
        self.token_deprojector = nn.Linear(self.llm_hidden_dim, 256)

        self.bev_decoder = nn.Sequential(
            # 50×63 → 100×126
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 100×126 → 200×252
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 → 384（匹配 backbone 输出 384 = 128×3）
            nn.Conv2d(128, 384, kernel_size=1),
        )

        # ============================================================
        # ⑥ 检测头（可训练，和之前完全一样）
        # ============================================================
        self.cls_head = nn.Conv2d(384, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(384, 7 * args['anchor_number'], kernel_size=1)

        # 统计参数量
        self._print_param_count()

    def _build_llm(self, model_name):
        """
        加载 LLM 并添加 LoRA 适配器。
        模型权重自动从 HuggingFace 下载到本地 ~/.cache/huggingface/。
        第一次运行时需要网络，之后离线运行。
        """
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        # 加载 LLM（只加载 encoder/backbone，不需要 lm_head）
        llm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,         # fp16 节省显存
            trust_remote_code=True,             # Qwen 需要这个
        )

        # 冻结 LLM 主体
        for param in llm.parameters():
            param.requires_grad = False

        # 添加 LoRA（只训练这 ~10M 参数）
        lora_config = LoraConfig(
            r=8,                                # LoRA rank
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        llm = get_peft_model(llm, lora_config)

        return llm

    def forward(self, data_dict):
        # ============================================================
        # Step 1: 体素 → BEV（照抄不动）
        # ============================================================
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
        spatial_features = batch_dict['spatial_features']      # (N_agent, 64, 400, 504)

        # ============================================================
        # Step 2: 融合 + 压缩 BEV → 减少 token 数量
        # ============================================================
        split = self.regroup(spatial_features, record_len)
        if len(split) >= 2:
            fused = torch.cat(split[:2], dim=1)                # (1, 128, 400, 504)
            fused = self.fusion_conv(fused)                    # (1, 64, 400, 504)
        else:
            fused = split[0]                                    # (1, 64, 400, 504)

        compressed = self.bev_compressor(fused)                # (1, 256, 50, 63)
        B, C, H_comp, W_comp = compressed.shape

        # ============================================================
        # Step 3: BEV 空间 → token 序列
        #         50×63 = 3150 个 token，每个 256 维
        #         投影到 LLM 维度 1536
        # ============================================================
        tokens = compressed.flatten(2).transpose(1, 2)         # (1, 3150, 256)
        tokens = self.token_projector(tokens)                   # (1, 3150, 1536)

        # ============================================================
        # Step 4: ★ LLM 处理 ★
        #         每个 token 通过 28 层 Transformer 和其他 token 交互
        #         LLM 主体冻结，只 LoRA 参数有梯度
        # ============================================================
        llm_output = self.llm(inputs_embeds=tokens)
        # 取最后一层的 hidden states
        llm_features = llm_output.last_hidden_state           # (1, 3150, 1536)

        # ============================================================
        # Step 5: Token → BEV 空间
        #         1536 → 256 → 重塑 → 上采样到检测头需要的尺寸
        # ============================================================
        decoded = self.token_deprojector(llm_features)         # (1, 3150, 256)
        decoded = decoded.transpose(1, 2)                      # (1, 256, 3150)
        decoded = decoded.view(B, 256, H_comp, W_comp)         # (1, 256, 50, 63)

        spatial_2d = self.bev_decoder(decoded)                 # (1, 384, 200, 252)

        # ============================================================
        # Step 6: 检测头 → psm + rm（和原来完全一样）
        # ============================================================
        psm = self.cls_head(spatial_2d)                       # (1, 2, 200, 252)
        rm = self.reg_head(spatial_2d)                        # (1, 14, 200, 252)

        return {'psm': psm, 'rm': rm}

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MyLLMDetector] 总参数: {total/1e6:.1f}M  |  可训练: {trainable/1e6:.1f}M")
