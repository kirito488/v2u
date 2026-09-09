from concurrent.futures import process
import os, sys
from mvp.config import third_party_root
opencood_root = os.path.join(third_party_root, "OpenCOOD")
sys.path.append(opencood_root)
import numpy as np
from collections import OrderedDict
import torch
import math
import copy
import random
import logging
import torch
import torch.nn.functional as F

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import torch_tensor_to_numpy

from .perception import Perception
from mvp.config import model_root, data_root
from mvp.tools.iou import iou3d
from mvp.evaluate.detection import iou3d_batch
from .iou_util import oriented_box_intersection_2d
from mvp.data.util import pcd_sensor_to_map, pcd_map_to_sensor, pose_to_transformation


class OpencoodPerception(Perception):
    def __init__(self, fusion_method="early", model_name="pointpillar",
                 model_dir=None, opencood_root=None,
                 root_dir=None, validate_dir=None, score_threshold=None):
        super().__init__()
        # pointpillar = AttFuse 模型；attfuse/where2comm/coalign 为 V2U4Real 的三个模型
        assert(model_name in ["pixor", "voxelnet", "second", "pointpillar", "v2vnet", "fpvrcnn",
                              "attfuse", "where2comm", "coalign"])
        assert(fusion_method in ["early", "intermediate", "late"])
        self.name = "{}_{}".format(model_name, fusion_method)
        self.devices = "cuda:0"
        self.model_name = model_name
        self.fusion_method = fusion_method

        # 如果传了 opencood_root，就指向这套 OpenCOOD 代码（例如 V2U4Real 的 opencood）
        if opencood_root is not None and opencood_root not in sys.path:
            sys.path.insert(0, opencood_root)

        # 三个 V2U4Real 模型：直接指定 checkpoint 目录
        if model_dir is not None:
            self.model_dir = model_dir
        elif self.model_name == "v2vnet":
            self.model_dir = os.path.join(model_root, "OpenCOOD/v2vnet")
            self.fusion_method = "intermediate"
        else:
            self.model_dir = os.path.join(model_root, "OpenCOOD/{}_{}_fusion".format(self.model_name, self.fusion_method if self.fusion_method != "intermediate" else "attentive"))
        self.config_file = os.path.join(self.model_dir, "config.yaml")
        self.preprocessors = {
            "early": self.early_preprocess,
            "intermediate": self.intermediate_preprocess,
            "late": self.late_preprocess,
        }
        self.inference_processors = {
            "early": inference_utils.inference_early_fusion,
            "intermediate": inference_utils.inference_intermediate_fusion,
            "late": inference_utils.inference_late_fusion,
        }

        hypes = yaml_utils.load_yaml(self.config_file, None)
        if score_threshold is not None:
            hypes.setdefault("postprocess", {}).setdefault("target_args", {})
            hypes["postprocess"]["target_args"]["score_threshold"] = float(score_threshold)
        # 数据路径：V2U4Real 时由外部传入
        if root_dir is not None:
            hypes["root_dir"] = root_dir
        else:
            hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
        if validate_dir is not None:
            hypes["validate_dir"] = validate_dir
        else:
            hypes["validate_dir"] = os.path.join(data_root, "OPV2V/validate")
        self.dataset = build_dataset(hypes, visualize=False, train=False)
        self.model = train_utils.create_model(hypes)
        # we assume gpu is available
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = train_utils.load_saved_model(self.model_dir, self.model)
        self.model = ret[1]
        self.model.eval()

    def run(self, multi_vehicle_case, ego_id):
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)
            # V2U4Real 的 inference 函数返回 4 个值（多一个 gt_ids），用 _ 忽略
            pred_box_tensor, pred_score, gt_box_tensor, _ = \
                self.inference_processors[self.fusion_method](batch_data,
                                                              self.model,
                                                              self.dataset)
        if pred_box_tensor is None:
            try:
                n_vox = int(batch_data["ego"]["processed_lidar"]["voxel_coords"].shape[0])
            except Exception:
                n_vox = -1
                n_keys = list(batch_data.keys()) if isinstance(batch_data, dict) else type(batch_data)
                print("[opencood] {} empty preds, batch_keys={}".format(self.name, n_keys))
            else:
                print("[opencood] {} empty preds, voxels={}".format(self.name, n_vox))
            self.last_fusion_z = self._read_defense_z()
            return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        pred_bboxes = pred_box_tensor.cpu().numpy()
        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
        pred_scores = pred_score.cpu().numpy()
        self.last_fusion_z = self._read_defense_z()
        return pred_bboxes, pred_scores

    def _read_defense_z(self):
        """MADE Z_ego / Z_fused left on the fusion module after forward."""
        z = getattr(self.model, "_defense_z", None)
        if not z:
            self.last_fusion_z = None
            return None
        out = {}
        for key in ("Z_ego", "Z_fused", "spatial_features_2d"):
            t = z.get(key)
            if t is None:
                continue
            out[key] = t.detach().float().cpu().numpy()
        self.last_fusion_z = out or None
        return self.last_fusion_z
    
    def run_multi_vehicle(self, multi_vehicle_case, ego_id):
        pred_bboxes, pred_scores = self.run(multi_vehicle_case, ego_id)
        if pred_bboxes.shape[0] == 0:
            multi_vehicle_case[ego_id]["pred_bboxes"] = np.array([])
            multi_vehicle_case[ego_id]["pred_scores"] = np.array([])
        else:
            multi_vehicle_case[ego_id]["pred_bboxes"] = pred_bboxes
            multi_vehicle_case[ego_id]["pred_scores"] = pred_scores
        return multi_vehicle_case

    def attack_late(self, multi_vehicle_case, ego_id, attacker_id, bbox=None, mode="spoof"):
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        if bbox is not None:
            bbox = np.copy(bbox)
            bbox[3:6] = bbox[[5,4,3]]
            bbox[2] += 0.5 * bbox[3]
            bbox = torch.from_numpy(bbox).type(torch.float32).to(self.device)

        with torch.no_grad():
            data_dict = train_utils.to_device(batch_data, self.device)
            output_dict = OrderedDict()
            for cav_id, cav_content in data_dict.items():
                output_dict[cav_id] = self.model(cav_content)

            # the final bounding box list
            pred_box3d_list = []
            pred_box2d_list = []

            for cav_id, cav_content in data_dict.items():
                transformation_matrix = cav_content['transformation_matrix']
                anchor_box = cav_content['anchor_box']
                prob = output_dict[cav_id]['psm']
                prob = F.sigmoid(prob.permute(0, 2, 3, 1))
                prob = prob.reshape(1, -1)
                reg = output_dict[cav_id]['rm']
                batch_box3d = self.dataset.post_processor.delta_to_boxes3d(reg, anchor_box)
                mask = \
                    torch.gt(prob, self.dataset.post_processor.params['target_args']['score_threshold'])
                mask = mask.view(1, -1)
                mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

                boxes3d = torch.masked_select(batch_box3d[0],
                                            mask_reg[0]).view(-1, 7)
                scores = torch.masked_select(prob[0], mask[0])

                # convert output to bounding box
                if len(boxes3d) != 0:
                    if cav_id == attacker_id:
                        if mode == "spoof":
                            boxes3d = torch.vstack([boxes3d, torch.reshape(bbox, (1, 7))])
                            scores = torch.hstack([scores, torch.tensor([1.0]).type(scores.dtype).to(self.device)])
                        elif mode == "remove":
                            keep_index = torch.sum((boxes3d[:, :2] - bbox[:2]) ** 2, dim=1) > 4
                            boxes3d = boxes3d[keep_index]
                            scores = scores[keep_index]

                    # (N, 8, 3)
                    boxes3d_corner = \
                        box_utils.boxes_to_corners_3d(boxes3d,
                                                    order=self.dataset.post_processor.params['order'])
                    # (N, 8, 3)
                    projected_boxes3d = \
                        box_utils.project_box3d(boxes3d_corner,
                                                transformation_matrix)
                    # convert 3d bbx to 2d, (N,4)
                    projected_boxes2d = \
                        box_utils.corner_to_standup_box_torch(projected_boxes3d)
                    # (N, 5)
                    boxes2d_score = \
                        torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                    pred_box2d_list.append(boxes2d_score)
                    pred_box3d_list.append(projected_boxes3d)

            if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
                raise Exception("no detection result")
            # shape: (N, 5)
            pred_box2d_list = torch.vstack(pred_box2d_list)
            # scores
            scores = pred_box2d_list[:, -1]
            # predicted 3d bbx
            pred_box3d_tensor = torch.vstack(pred_box3d_list)
            # remove large bbx
            keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
            keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
            keep_index = torch.logical_and(keep_index_1, keep_index_2)
            pred_box3d_tensor = pred_box3d_tensor[keep_index]
            scores = scores[keep_index]

            # nms
            keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                            scores,
                                            self.dataset.post_processor.params['nms_thresh']
                                            )
            pred_box3d_tensor = pred_box3d_tensor[keep_index]

            # select cooresponding score
            scores = scores[keep_index]

            # filter out the prediction out of the range.
            mask = \
                box_utils.get_mask_for_boxes_within_range_torch(
                    pred_box3d_tensor,
                    self.dataset.post_processor.params['GT_RANGE'])
            pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
            scores = scores[mask]
            assert scores.shape[0] == pred_box3d_tensor.shape[0]

        pred_box = pred_box3d_tensor.cpu().numpy()
        pred_box = box_utils.corner_to_center(pred_box, order="lwh")
        pred_box[:,2] -= 0.5 * pred_box[:,5]
        return {
            "pred_bboxes": pred_box,
            "pred_scores": scores.cpu().numpy()
        }

    def attack_intermediate_forward(self, batch_data, attacker_index, perturbation=None, feature=None, max_perturb=3, center=[0, 0, 0], feature_size=5, perturb_func=None, centers=None):
        """Forward pass of the intermediate fusion model with an optional feature attack.

        Supports both single-object (backward compatible) and multi-object perturbation:

        - ``centers`` is None: the legacy single-object path, perturbing a
          ``feature_size`` block around ``center``.
        - ``centers`` is a list of N voxel indices: ``perturbation`` is treated as a
          tensor of shape ``(N, C, 2S, 2S)`` and each block is placed around its own
          center, then summed onto the attacker's spatial feature map.
        """
        if perturbation is not None:
            clipped_perturbation = torch.clip(perturbation, min=-max_perturb, max=max_perturb)
        else:
            clipped_perturbation = None

        voxel_features = batch_data['ego']['processed_lidar']['voxel_features']
        voxel_coords = batch_data['ego']['processed_lidar']['voxel_coords']
        voxel_num_points = batch_data['ego']['processed_lidar']['voxel_num_points']
        record_len = batch_data['ego']['record_len']

        pairwise_t_matrix = batch_data['ego']['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points,
                    'record_len': record_len,
                    'pairwise_t_matrix': pairwise_t_matrix}

        if self.model_name == "v2vnet":
            batch_dict['voxel_features'] = batch_dict['voxel_features'].float()
        
        if self.model_name in ["pointpillar", "v2vnet", "attfuse", "where2comm", "coalign"]:
            # n, 4 -> n, c
            self.model.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            self.model.scatter(batch_dict)

            spatial_features = batch_dict['spatial_features']
        elif self.model_name == "voxelnet":
            if voxel_coords.is_cuda:
                record_len_tmp = record_len.cpu()

            record_len_tmp = list(record_len_tmp.numpy())

            self.model.N = sum(record_len_tmp)

            # feature learning network
            vwfs = self.model.svfe(batch_dict)['pillar_features']

            voxel_coords = torch_tensor_to_numpy(voxel_coords)
            vwfs = self.model.voxel_indexing(vwfs, voxel_coords)

            # convolutional middle network
            vwfs = self.model.cml(vwfs)
            # convert from 3d to 2d N C H W
            vmfs = vwfs.view(self.model.N, -1, self.model.H, self.model.W)

            # compression layer
            if self.model.compression:
                vmfs = self.model.compression_layer(vmfs)
            
            spatial_features = vmfs
        else:
            raise NotImplementedError()

        if perturb_func is not None:
            x = torch.clone(spatial_features).detach()
            spatial_features[attacker_index] = perturb_func(x[attacker_index].unsqueeze(0))[0]
        elif feature is not None:
            # Or directly set the feature.
            spatial_features[attacker_index][:, center[1]-feature_size:center[1]+feature_size, center[0]-feature_size:center[0]+feature_size] = feature
            clipped_perturbation = None
        elif perturbation is not None:
            # Appends the perturbation.
            x = torch.clone(spatial_features).detach()
            if centers is not None and len(centers) > 0:
                # Multi-object: perturbation is (N, C, 2S, 2S); place each block at
                # its own center and sum them onto the attacker feature map.
                # Accepts a legacy single-object (C, 2S, 2S) tensor too (auto-wrapped).
                perturbation_blocks = clipped_perturbation if clipped_perturbation.dim() == 4 else clipped_perturbation.unsqueeze(0)
                C, H, W = spatial_features[attacker_index].size()
                perturbation_features = torch.zeros_like(spatial_features[attacker_index]).to(self.device)
                for i, center_i in enumerate(centers):
                    # Interpolation of center indices
                    aligned_center = center_i.astype(np.int32)
                    block = torch.zeros_like(spatial_features[attacker_index]).to(self.device)
                    block[:, aligned_center[1] - feature_size:aligned_center[1] + feature_size,
                             aligned_center[0] - feature_size:aligned_center[0] + feature_size] = perturbation_blocks[i]
                    theta = torch.tensor([[[1, 0, (center_i[1] - aligned_center[1]) * 2 / W],
                                           [0, 1, (center_i[0] - aligned_center[0]) * 2 / H]]], dtype=torch.float).repeat(1, 1, 1).to(self.device)
                    grid = torch.nn.functional.affine_grid(theta, (1, C, H, W))
                    block = torch.nn.functional.grid_sample(block.unsqueeze(0), grid)[0]
                    perturbation_features = perturbation_features + block
                spatial_features[attacker_index] = x[attacker_index] + perturbation_features
            else:
                # Interpolation of center indices
                aligned_center = center.astype(np.int32)
                C, H, W = spatial_features[attacker_index].size()
                perturbation_features = torch.zeros_like(spatial_features[attacker_index]).to(self.device)
                perturbation_features[:, aligned_center[1]-feature_size:aligned_center[1]+feature_size,
                                         aligned_center[0]-feature_size:aligned_center[0]+feature_size] = clipped_perturbation
                theta = torch.tensor([[[1, 0, (center[1] - aligned_center[1]) * 2 / W],
                                       [0, 1, (center[0] - aligned_center[0]) * 2 / H]]], dtype=torch.float).repeat(1, 1, 1).to(self.device)
                grid = torch.nn.functional.affine_grid(theta, (1, C, H, W))
                perturbation_features = torch.nn.functional.grid_sample(perturbation_features.unsqueeze(0), grid)[0]
                spatial_features[attacker_index] = x[attacker_index] + perturbation_features

        if self.model_name in ["pointpillar", "v2vnet", "attfuse"]:
            # ------------------------------------------------------------------
            # AttFuse (point_pillar_intermediate)：融合发生在 backbone 内部，
            # backbone 一步就把所有 agent 的特征合好了，之后直接接两个 head。
            # ------------------------------------------------------------------
            batch_dict["spatial_features"] = spatial_features
            self.model.backbone(batch_dict)
            spatial_features_2d = batch_dict['spatial_features_2d']

            if self.model_name == "v2vnet":
                # downsample feature to reduce memory
                if self.model.shrink_flag:
                    spatial_features_2d = self.model.shrink_conv(spatial_features_2d)
                # compressor
                if self.model.compression:
                    spatial_features_2d = self.model.naive_compressor(spatial_features_2d)

                fused_feature = self.model.fusion_net(spatial_features_2d,
                                                record_len,
                                                pairwise_t_matrix)
                psm = self.model.cls_head(fused_feature)
                rm = self.model.reg_head(fused_feature)
            else:
                psm = self.model.cls_head(spatial_features_2d)
                rm = self.model.reg_head(spatial_features_2d)

        elif self.model_name == "where2comm":
            # ------------------------------------------------------------------
            # Where2comm：backbone 之后还要 shrink(缩小) -> 先算注意力图 psm_single
            # -> 压缩 -> 它的像素级融合网络 -> head。
            # 这一段就是把 point_pillar_where2comm.py 的 forward 抄进来。
            # ------------------------------------------------------------------
            self.model.backbone(batch_dict)
            spatial_features_2d = batch_dict['spatial_features_2d']
            if self.model.shrink_flag:
                spatial_features_2d = self.model.shrink_conv(spatial_features_2d)
            psm_single = self.model.cls_head(spatial_features_2d)
            if self.model.compression:
                spatial_features_2d = self.model.naive_compressor(spatial_features_2d)
            if self.model.multi_scale:
                fused_feature, _ = self.model.fusion_net(
                    batch_dict['spatial_features'], psm_single,
                    record_len, pairwise_t_matrix, self.model.backbone)
                if self.model.shrink_flag:
                    fused_feature = self.model.shrink_conv(fused_feature)
            else:
                fused_feature, _ = self.model.fusion_net(
                    spatial_features_2d, psm_single,
                    record_len, pairwise_t_matrix)
            psm = self.model.cls_head(fused_feature)
            rm = self.model.reg_head(fused_feature)

        elif self.model_name == "coalign":
            # ------------------------------------------------------------------
            # CoAlign：ResBEVBackbone 分层提特征（多尺度），每层用 Att_w_Warp
            # 做 warp+attention 融合，最后 decode_multiscale_feature 合并。
            # 这一段就是把 point_pillar_coalign.py 的 forward 抄进来。
            # ------------------------------------------------------------------
            from opencood.models.fuse_modules.coalign_fuse import normalize_pairwise_tfm
            _, _, H0, W0 = batch_dict['spatial_features'].shape
            normalized_affine_matrix = normalize_pairwise_tfm(
                pairwise_t_matrix, H0, W0, self.model.voxel_size[0])

            feature_list = []
            feature_list.append(self.model.backbone.get_layer_i_feature(
                batch_dict['spatial_features'], layer_i=0))
            if self.model.compression:
                feature_list[0] = self.model.naive_compressor(feature_list[0])
            for i in range(1, self.model.backbone.num_levels):
                feature_list.append(self.model.backbone.get_layer_i_feature(
                    feature_list[i - 1], layer_i=i))

            fused_feature_list = []
            for i, fuse_module in enumerate(self.model.fusion_net):
                fused_feature_list.append(fuse_module(
                    feature_list[i], record_len,
                    normalized_affine_matrix, batch_data['ego']))
            fused_feature = self.model.backbone.decode_multiscale_feature(
                fused_feature_list)
            if self.model.shrink_flag:
                fused_feature = self.model.shrink_conv(fused_feature)
            psm = self.model.cls_head(fused_feature)
            rm = self.model.reg_head(fused_feature)

        elif self.model_name == "voxelnet":
            # information naive fusion
            vmfs_fusion = self.model.fusion_net(spatial_features, record_len)
            # map and regression map
            psm, rm = self.model.rpn(vmfs_fusion)
        else:
            raise NotImplementedError()

        output_dict = OrderedDict()
        output_dict['ego'] = {'psm': psm,
                            'rm': rm}

        return output_dict, clipped_perturbation, spatial_features

    def attack_intermediate(self, multi_vehicle_case, ego_id, attacker_id, max_perturb=3, lr=0.2, max_iteration=25, bbox=None, mode="spoof", real_case=None, original_case=None, real_original_case=None, real_bbox=None, init_perturbation=None, feature_size=5, bboxes=None):
        """Intermediate fusion attack.

        Backward compatible: the legacy single-object ``bbox`` is internally treated
        as ``bboxes=[bbox]``. When ``bboxes`` (list of N ego-frame bboxes) is given,
        N perturbation blocks are optimized simultaneously (one per object center)
        and the loss is the sum of per-object terms.
        """
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        # Normalize to a list of target bboxes (ego frame).
        if bboxes is None:
            bboxes = [bbox] if bbox is not None else None
        if bboxes is not None and not isinstance(bboxes, (list, tuple)):
            bboxes = [bboxes]
        num_objects = len(bboxes) if bboxes is not None else 0

        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert(attacker_index >= 0)

        optimize_batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        optimize_batch_data = train_utils.to_device(self.dataset.collate_batch_test([optimize_batch]), self.device)
        anchor_box = optimize_batch_data['ego']['anchor_box']

        if self.model_name == "voxelnet":
            feature_dim = 128
        else:
            # 扰动加在 scatter 之后的 per-CAV 特征图上，Pillar 系列都是 64 通道
            feature_dim = 64

        bbox_tensors = None
        centers = None
        if bboxes is not None:
            bbox_tensors = []
            centers = []
            for b in bboxes:
                bbox_tensor = torch.from_numpy(b).to(self.device).type(torch.float32)
                bbox_tensor[2] += 0.5 * bbox_tensor[5]
                bbox_tensors.append(bbox_tensor)
                centers.append(self.point_to_voxel_index(b))
            # single-object shorthands kept for the legacy sync/remove paths
            bbox_tensor = bbox_tensors[0]
            center = centers[0]

        real_center = None
        if real_bbox is not None:
            real_center = self.point_to_voxel_index(real_bbox)

        with torch.no_grad():
            optimize_output_dict, _, optimize_feature = self.attack_intermediate_forward(optimize_batch_data, attacker_index)

            # 把 center 夹紧到特征图有效范围内，避免扰动窗口越界变空(size 0)
            FH = optimize_feature[attacker_index].shape[-2]
            FW = optimize_feature[attacker_index].shape[-1]

            def _clamp_center(c):
                c = np.asarray(c).astype(np.int32).copy()
                c[0] = int(np.clip(c[0], feature_size, max(feature_size, FW - feature_size - 1)))
                c[1] = int(np.clip(c[1], feature_size, max(feature_size, FH - feature_size - 1)))
                return c

            if centers is not None:
                centers = [_clamp_center(c) for c in centers]
                center = centers[0]
            if real_center is not None:
                real_center = _clamp_center(real_center)

            if real_case is not None:
                real_batch = self.preprocessors[self.fusion_method](real_case, ego_id)
                real_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_batch]), self.device)
                _, _, real_feature = self.attack_intermediate_forward(real_batch_data, attacker_index)

            if original_case is not None:
                original_batch = self.preprocessors[self.fusion_method](original_case, ego_id)
                original_batch_data = train_utils.to_device(self.dataset.collate_batch_test([original_batch]), self.device)
                _, _, original_feature = self.attack_intermediate_forward(original_batch_data, attacker_index)
                # TODO: interpolation of center indices
                if centers is not None and num_objects > 1:
                    base_perturbation = torch.stack([
                        (optimize_feature[attacker_index] - original_feature[attacker_index])[
                            :, c[1]-feature_size:c[1]+feature_size, c[0]-feature_size:c[0]+feature_size]
                        for c in centers
                    ]).detach()
                else:
                    base_perturbation = ((optimize_feature[attacker_index] - original_feature[attacker_index])[:, center[1]-feature_size:center[1]+feature_size, center[0]-feature_size:center[0]+feature_size]).detach()
            else:
                shape = (num_objects, feature_dim, 2 * feature_size, 2 * feature_size) if centers is not None and num_objects > 1 else (feature_dim, 2 * feature_size, 2 * feature_size)
                base_perturbation = torch.zeros(*shape).to(self.device).detach()

            if real_original_case is not None:
                real_original_batch = self.preprocessors[self.fusion_method](real_original_case, ego_id)
                real_original_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_original_batch]), self.device)
                _, _, real_original_feature = self.attack_intermediate_forward(real_original_batch_data, attacker_index)
                # TODO: interpolation of center indices
                real_base_perturbation = ((real_feature[attacker_index] - real_original_feature[attacker_index])[:, real_center[1]-feature_size:real_center[1]+feature_size, real_center[0]-feature_size:real_center[0]+feature_size]).detach()
            else:
                real_base_perturbation = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device).detach()

        if init_perturbation is not None:
            perturbation = torch.from_numpy(init_perturbation).to(self.device)
        else:
            shape = (num_objects, feature_dim, 2 * feature_size, 2 * feature_size) if centers is not None and num_objects > 1 else (feature_dim, 2 * feature_size, 2 * feature_size)
            perturbation = torch.zeros(*shape).to(self.device)
        perturbation.requires_grad = True
        optimizer = torch.optim.Adam([perturbation], lr=lr)

        best_loss = 0xffffffff
        best_perturbation = None
        best_pred_bboxes = None
        best_pred_scores = None
        no_progress_iters = 0

        for it in range(max_iteration):
            batch_data = self.detach_all(optimize_batch_data if original_case is None else original_batch_data)

            output_dict, clipped_perturbation, _ = self.attack_intermediate_forward(batch_data, attacker_index, perturbation=(base_perturbation + perturbation), max_perturb=max_perturb, center=center, feature_size=feature_size, centers=centers)
            prob = F.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
            proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0]

            # Per-object IoU masks; loss is the sum over all target objects.
            if bboxes is not None:
                iou_masks = []
                for bj in bbox_tensors:
                    iou_j = torch.clip(self.iou_torch(
                        proposals[:,[0,1,2,5,4,3,6]],
                        bj.tile((proposals.shape[0],1))
                    ), min=0, max=1)
                    mask_j = iou_j >= 0.01
                    # spoof 幽灵处常无 3D IoU（高度对不上 / 空区域无重叠 proposal），
                    # 用 BEV 距离把附近 anchor 纳入 loss，否则 n_hit=0、梯度为 0。
                    if mode == "spoof":
                        dxy = torch.sqrt(
                            (proposals[:, 0] - bj[0]) ** 2 + (proposals[:, 1] - bj[1]) ** 2)
                        mask_j = torch.logical_or(mask_j, dxy < 2.0)
                        if int(mask_j.sum().item()) == 0:
                            mask_j = dxy < 3.0
                        # 只对离幽灵最近的 K 个 anchor 反传，避免 150+ anchor 累加把扰动推爆
                        hit_idx = torch.nonzero(mask_j, as_tuple=False).reshape(-1)
                        spoof_topk = 32
                        if hit_idx.numel() > spoof_topk:
                            keep = hit_idx[torch.argsort(dxy[hit_idx])[:spoof_topk]]
                            narrow = torch.zeros_like(mask_j)
                            narrow[keep] = True
                            mask_j = narrow
                        iou_j = torch.maximum(iou_j, torch.exp(-dxy / 3.0))
                    iou_masks.append((iou_j, mask_j))
            else:
                iou = torch.ones(proposals.shape[0], dtype=torch.bool).to(self.device).detach()
                box_mask = (iou >= 0.01)

            with torch.no_grad():
                if real_case is not None:
                    real_batch_data = self.detach_all(real_batch_data)
                    real_output_dict, _, _ = self.attack_intermediate_forward(real_original_batch_data if real_original_case is not None else real_batch_data, attacker_index, perturbation=real_base_perturbation + perturbation, feature=None, max_perturb=max_perturb, center=real_center, feature_size=feature_size)

                    pred_box_tensor, pred_score_tensor, gt_box_tensor, _ = \
                        self.dataset.post_process(real_batch_data,
                                                  real_output_dict)
                    result_anchor_box = real_batch_data['ego']['anchor_box']
                    result_prob = F.sigmoid(real_output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                    result_proposals = self.dataset.post_processor.delta_to_boxes3d(real_output_dict['ego']['rm'], result_anchor_box)[0]
                else:
                    pred_box_tensor, pred_score_tensor, gt_box_tensor, _ = \
                        self.dataset.post_process(batch_data,
                                                  output_dict)
                    result_prob = prob
                    result_proposals = proposals

                if pred_box_tensor is None:
                    pred_bboxes = np.array([])
                    pred_scores = np.array([])
                else:
                    pred_bboxes = pred_box_tensor.cpu().numpy()
                    pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                    pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                    pred_scores = pred_score_tensor.cpu().numpy()

            if mode == "spoof":
                if bboxes is not None:
                    loss = torch.zeros((), device=self.device)
                    for iou_j, mask_j in iou_masks:
                        n_hit = max(int(mask_j.sum().item()), 1)
                        loss = loss + (iou_j[mask_j] * torch.log(1 - prob[mask_j])).sum() / n_hit
                else:
                    n_hit = max(int(box_mask.sum().item()), 1)
                    loss = (iou[box_mask] * torch.log(1 - prob[box_mask])).sum() / n_hit
            elif mode == "remove":
                if bboxes is not None:
                    loss = torch.zeros((), device=self.device)
                    for iou_j, mask_j in iou_masks:
                        n_hit = max(int(mask_j.sum().item()), 1)
                        loss = loss + (-iou_j[mask_j] * torch.log(1 - prob[mask_j])).sum() / n_hit
                else:
                    n_hit = max(int(box_mask.sum().item()), 1)
                    loss = (-iou[box_mask] * torch.log(1 - prob[box_mask])).sum() / n_hit
            else:
                raise NotImplementedError("Attack mode not supported.")

            if it == 0 and bboxes is not None and logging.getLogger().isEnabledFor(logging.DEBUG):
                n_hits = [int(m.sum().item()) for _, m in iou_masks]
                tgt = bboxes[0]
                logging.debug(
                    "obj[0] target=[{:.2f}, {:.2f}, {:.2f}] n_hit={} n_prop={}".format(
                        float(tgt[0]), float(tgt[1]), float(tgt[2]), n_hits[0], int(prob.numel())))

            if loss.item() < -0xffff:
                break

            spoof_stop = False
            spoof_peak = 0.0
            if mode == "spoof" and bboxes is not None:
                with torch.no_grad():
                    spoof_peak = max(float(prob[m].max().item()) for _, m in iou_masks if m.any())
                spoof_stop = spoof_peak >= 0.35

            if mode == "spoof" and spoof_stop:
                best_loss = loss.item()
                best_perturbation = clipped_perturbation.cpu().detach().numpy()
                best_pred_bboxes = pred_bboxes
                best_pred_scores = pred_scores
                best_proposals = result_proposals[:,[0,1,2,5,4,3,6]].cpu().detach().numpy()
                best_prob = result_prob.cpu().detach().numpy()
                break

            if loss.item() < best_loss or max_iteration <= 2:
                best_loss = loss.item()
                best_perturbation = clipped_perturbation.cpu().detach().numpy()
                best_pred_bboxes = pred_bboxes
                best_pred_scores = pred_scores
                best_proposals = result_proposals[:,[0,1,2,5,4,3,6]].cpu().detach().numpy()
                best_prob = result_prob.cpu().detach().numpy()
                no_progress_iters = 0
            else:
                no_progress_iters += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "Iteration {} - loss: {:.4f}, peak: {:.3f}, n_hit: {}".format(
                        it, loss.item(), spoof_peak,
                        int(iou_masks[0][1].sum().item()) if bboxes is not None else -1))

        return {
            "perturbation": best_perturbation,
            "loss": best_loss,
            "pred_bboxes": best_pred_bboxes,
            "pred_scores": best_pred_scores,
            "proposals": best_proposals,
            "prob": best_prob,
        }

    def retrieve_base_data(self, multi_vehicle_case, ego_id):
        data = OrderedDict()
        # x1_to_x2 needs a Python list pose; numpy (6,) hits np.linalg.inv on 1D.
        ego_pose = np.asarray(multi_vehicle_case[ego_id]["lidar_pose"]).reshape(-1).tolist()
        for vehicle_id, vehicle_data in multi_vehicle_case.items():
            data[vehicle_id] = OrderedDict()
            data[vehicle_id]['ego'] = (vehicle_id == ego_id)
            data[vehicle_id]["cav_id"] = vehicle_id
            data[vehicle_id]['time_delay'] = 0
            cav_pose = np.asarray(vehicle_data["lidar_pose"]).reshape(-1).tolist()
            if "params" in vehicle_data:
                data[vehicle_id]['params'] = vehicle_data["params"]
                data[vehicle_id]['params']["lidar_pose"] = cav_pose
                # 必须与训练时 basedataset 一致：scipy ZYX 的 x1_to_x2(cav, ego)
                data[vehicle_id]['params']["transformation_matrix"] = x1_to_x2(
                    cav_pose, ego_pose)
            else:
                data[vehicle_id]['params'] = {
                    "lidar_pose": cav_pose,
                    "vehicles": {},
                }
            # V2U4Real 数据集需要这两个字段
            data[vehicle_id]['params']["spatial_correction_matrix"] = np.eye(4)
            if "ego_speed" not in data[vehicle_id]['params']:
                data[vehicle_id]['params']["ego_speed"] = 0.0
            # V2U4Real 训练用 pcd 原始 intensity；不要改成全 1（那是 OPV2V PointPillar 的旧约定）
            lidar = np.asarray(vehicle_data["lidar"], dtype=np.float32)
            if lidar.ndim != 2 or lidar.shape[0] == 0:
                lidar = np.zeros((1, 4), dtype=np.float32)
                lidar[0, :2] = 200.0
            elif lidar.shape[1] < 4:
                pad = np.full((lidar.shape[0], 4 - lidar.shape[1]), 0.1, dtype=np.float32)
                lidar = np.hstack([lidar, pad])
            else:
                lidar = np.ascontiguousarray(lidar[:, :4])
            data[vehicle_id]['lidar_np'] = lidar
        return data

    def early_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_lidar_pose = base_data_dict[ego_id]["params"]['lidar_pose']

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > opencood.data_utils.datasets.COM_RANGE:
            #     continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            # all these lidar and object coordinates are projected to ego
            # already.
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # we do lidar filtering in the stacked lidar
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.dataset.params['postprocess'][
                                                         'order']
                                                     )
        # Two versions of OpenCOOD!
        if isinstance(object_bbx_center_valid, tuple):
            object_bbx_center_valid = object_bbx_center_valid[0]

        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.dataset.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        anchor_box = self.dataset.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.dataset.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'label_dict': label_dict})

        return processed_data_dict

    def intermediate_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.dataset.get_pairwise_transformation(base_data_dict,
                                             self.dataset.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > opencood.data_utils.datasets.COM_RANGE:
            #     continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                selected_cav_processed['processed_features'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.dataset.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.dataset.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.dataset.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'velocity': [0 for i in range(len(multi_vehicle_case))],
             'time_delay': [0 for i in range(len(multi_vehicle_case))],
             'infra': [0 for i in range(len(multi_vehicle_case))],
             'spatial_correction_matrix': [np.eye(4) for i in range(len(multi_vehicle_case))],
             "pairwise_t_matrix": pairwise_t_matrix})

        return processed_data_dict

    def late_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)
        reformat_data_dict = self.dataset.get_item_test(base_data_dict)

        return reformat_data_dict

    def points_to_voxel_torch(self, pcd):
        # https://github.com/DerrickXuNu/OpenCOOD/blob/main/opencood/data_utils/pre_processor/voxel_preprocessor.py
        # full_mean = False
        # block_filtering = False
        data_dict = {}
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        max_points_per_voxel = self.dataset.pre_processor.params["args"]["max_points_per_voxel"]

        voxel_coords = torch.floor((pcd[:, :3] - 
                torch.tensor(lidar_range[:3]).to(self.device)
            ) / torch.tensor(voxel_size).to(self.device)).int()

        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = torch.unique(voxel_coords, dim=0,
                                                           return_inverse=True,
                                                           return_counts=True)
        
        voxel_features = torch.zeros((len(voxel_coords), max_points_per_voxel, 4), dtype=torch.float32).to(self.device)

        for i in range(len(voxel_coords)):
            pts = pcd[inv_ind == i]
            if voxel_counts[i] > max_points_per_voxel:
                pts = pts[:max_points_per_voxel, :]
                voxel_counts[i] = max_points_per_voxel

            voxel_features[i, :pts.shape[0], :] = pts

        data_dict['voxel_features'] = voxel_features
        data_dict['voxel_coords'] = voxel_coords
        data_dict['voxel_num_points'] = voxel_counts

        return data_dict

    def point_to_voxel_index(self, point):
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        voxel_index = (np.floor(point[:3] - lidar_range[:3]) / voxel_size).astype(np.int32)
        return voxel_index

    def iou_torch(self, bboxes_a, bboxes_b):
        corners2d_a = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_a, order="lwh")[:,:,:2], 0)
        corners2d_b = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_b, order="lwh")[:,:,:2], 0)
        area_a = bboxes_a[:, 3] * bboxes_a[:, 4]
        area_b = bboxes_b[:, 3] * bboxes_b[:, 4]
        area_inter, _ = oriented_box_intersection_2d(corners2d_a, corners2d_b)
        area_inter = area_inter.squeeze()
        height_inter = torch.clip(
            torch.min(bboxes_a[:, 2] + 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] + 0.5 * bboxes_b[:, 5]) - \
            torch.max(bboxes_a[:, 2] - 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] - 0.5 * bboxes_b[:, 5]),
            min=0, max=5)
        iou = area_inter * height_inter / (area_a * bboxes_a[:, 5] + area_b * bboxes_b[:, 5] - area_inter * height_inter)
        return iou

    def pose_to_transformation_torch(self, pose, dim=2):
        x, y, z, roll, yaw, pitch = pose[0], pose[1], pose[2], torch.deg2rad(pose[3]), torch.deg2rad(pose[4]), torch.deg2rad(pose[5])
        if dim == 2:
            T = torch.zeros((3, 3)).to(torch.float32).to(self.device)
            T[0, 0] = torch.cos(yaw)
            T[0, 1] = 0 - torch.sin(yaw)
            T[0, 2] = x
            T[1, 0] = torch.sin(yaw)
            T[1, 1] = torch.cos(yaw)
            T[1, 2] = y
            T[2, 2] = 1
        elif dim == 3:
            T = torch.tensor([[torch.cos(yaw)*torch.cos(pitch), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.sin(roll)-torch.sin(yaw)*torch.cos(roll), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.cos(roll)+torch.sin(yaw)*torch.sin(roll),
                        x],
                        [torch.sin(yaw)*torch.cos(pitch), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.sin(roll)+torch.cos(yaw)*torch.cos(roll), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.cos(roll)-torch.cos(yaw)*torch.sin(roll),
                        y],
                        [-torch.sin(pitch), 
                        torch.cos(pitch)*torch.sin(roll), 
                        torch.cos(pitch)*torch.cos(roll),
                        z],
                        [0, 0, 0, 1]]).to(self.device)
        return T

    def attacker_to_origin_transformation(self, T, attacker_pose, origin_pose, dim=2):
        attacker_T = self.pose_to_transformation_torch(attacker_pose, dim=dim)
        origin_T = self.pose_to_transformation_torch(origin_pose, dim=dim)
        return torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.inverse(origin_T), attacker_T), T), torch.inverse(attacker_T)), origin_T)

    def detach_all(self, x):
        if isinstance(x, dict):
            y = {}
            for key, value in x.items():
                y[key] = self.detach_all(value)
        elif isinstance(x, list):
            y = []
            for value in x:
                y.append(self.detach_all(value))
        elif isinstance(x, torch.Tensor):
            y = x.detach()
        else:
            y = x
        return y
