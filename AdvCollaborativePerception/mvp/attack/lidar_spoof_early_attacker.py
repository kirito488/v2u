import copy
import os
import open3d as o3d
import numpy as np
import time
import random
import pickle

from .attacker import Attacker
from mvp.config import model_3d_path, model_3d_examples
from mvp.data.util import rotation_matrix, get_distance, pcd_map_to_sensor, sort_lidar_points, get_point_indices_in_bbox
from mvp.tools.ray_tracing import get_model_mesh, get_box_mesh, get_wall_mesh, ray_intersection


class LidarSpoofEarlyAttacker(Attacker):
    def __init__(self, dataset=None, dense=3, sync=1):
        super().__init__()
        self.name = "lidar_spoof"
        self.dataset = dataset
        self.load_benchmark_meta()
        self.name = "lidar_spoof_early"
        if dense == 1:
            self.name += "_DenseA"
        elif dense == 2:
            self.name += "_DenseAll"
        elif dense == 3:
            self.name += "_Sampled"
        if sync > 0:
            self.name += "_Async"
        self.dense = dense
        self.sync = sync

        # 优先用 AdvCollab 自带车辆 ply；没有则用长方体四面，保证 V2U4Real 也能注入点。
        self.default_car_model = "car_0200"
        self.use_car_ply = False
        self.mesh = None
        self.meshes = None
        ply_path = os.path.join(model_3d_path, "{}.ply".format(self.default_car_model))
        divide_path = os.path.join(model_3d_path, "spoof", "mesh_divide.pkl")
        if os.path.isfile(ply_path):
            self.mesh = o3d.io.read_triangle_mesh(ply_path)
            self.use_car_ply = True
            if os.path.isfile(divide_path):
                mesh_divide = pickle.load(open(divide_path, "rb"))
                self.meshes = [self.mesh.select_by_index(vi) for vi in mesh_divide]
            print("[early-spoof] 使用 3D 车模", ply_path)
        else:
            print("[early-spoof] 未找到 {}，改用长方体 mesh 做射线投射".format(ply_path))

        # In the mode of injecting dense points, fix the distance between the target and the sensor.
        self.dense_distance = 10 # (m)


    def run(self, multi_frame_case, attack_opts):
        """ attack_opts: {
                "frame_ids": [-1],
                "attacker_vehicle_id": int,
                "positions": list([7])
            }
        """
        new_case = copy.deepcopy(multi_frame_case)
        attack_info = []
        frame_ids = attack_opts["frame_ids"]
        if self.sync != 0:
            frame_ids = [frame_ids[0] - 1] + frame_ids
        
        last_info = None
        for frame_id in frame_ids:
            single_vehicle_case = multi_frame_case[frame_id][attack_opts["attacker_vehicle_id"]]
            lidar_poses = {vehicle_id: multi_frame_case[frame_id][vehicle_id]["lidar_pose"] for vehicle_id in multi_frame_case[frame_id]}
            core_function = self.run_core_sample if self.dense == 3 else self.run_core

            x, info = core_function(single_vehicle_case, {
                "position": attack_opts["positions"][frame_id],
                "lidar_poses": lidar_poses,
                "attacker_vehicle_id": attack_opts["attacker_vehicle_id"]
            })

            if self.sync != 0 and last_info is not None:
                x = copy.deepcopy(multi_frame_case[frame_id][attack_opts["attacker_vehicle_id"]])
                shift_vector = attack_opts["positions"][frame_id][:3] - attack_opts["positions"][frame_id - 1][:3]
                
                append_data = []
                if last_info["replace_data"] is not None and last_info["replace_data"].shape[0] > 0:
                    append_data.append(last_info["replace_data"] + shift_vector)
                if last_info["append_data"] is not None and last_info["append_data"].shape[0] > 0:
                    append_data.append(last_info["append_data"] + shift_vector)
                if len(append_data) > 1:
                    append_data = np.vstack(append_data)
                else:
                    append_data = None

                ignore_indices = []
                if info["replace_indices"] is not None and last_info["replace_indices"].shape[0] > 0:
                    ignore_indices.append(info["replace_indices"])
                if info["ignore_indices"] is not None and last_info["ignore_indices"].shape[0] > 0:
                    ignore_indices.append(info["ignore_indices"])
                if len(append_data) > 1:
                    ignore_indices = np.hstack(ignore_indices).reshape(-1)
                else:
                    ignore_indices = None

                info = {
                    "replace_indices": None,
                    "replace_data": None,
                    "ignore_indices": ignore_indices,
                    "append_data": append_data,
                }
                x["lidar"] = self.apply_ray_tracing(x["lidar"], **info)

            if frame_id in attack_opts["frame_ids"]:
                attack_info.append(info)
                new_case[frame_id][attack_opts["attacker_vehicle_id"]] = x

            last_info = info
        return new_case, attack_info

    def post_process_meshes(self, meshes, bbox):
        new_meshes = []
        for mesh in meshes:
            x = copy.deepcopy(mesh)
            scale = np.min(bbox[3:6] / model_3d_examples[self.default_car_model][3:6]) 
            x = x.scale(scale, np.array([0, 0, 0]).T)
            x = x.rotate(rotation_matrix(0, bbox[6], 0), np.zeros(3).T)
            x = x.translate(bbox[:3])
            new_meshes.append(x)
        return new_meshes

    def _side_meshes(self, bbox):
        """幽灵框四面薄墙，用于 sampled 射线投射（无 ply 时）。"""
        walls = []
        extend = 0.05
        specs = [
            (0, bbox[4] / 2 + extend, bbox[3], 0.02, bbox[5]),
            (0, -(bbox[4] / 2 + extend), bbox[3], 0.02, bbox[5]),
            (bbox[3] / 2 + extend, 0, 0.02, bbox[4], bbox[5]),
            (-(bbox[3] / 2 + extend), 0, 0.02, bbox[4], bbox[5]),
        ]
        c, s = np.cos(bbox[6]), np.sin(bbox[6])
        for dx, dy, l, w, h in specs:
            b = np.array(bbox, dtype=np.float64).copy()
            b[0] += c * dx - s * dy
            b[1] += s * dx + c * dy
            b[3], b[4], b[5] = l, w, h
            walls.append(get_wall_mesh(b))
        return walls

    def _spoof_meshes(self, bbox):
        if self.meshes is not None:
            return self.post_process_meshes(self.meshes, bbox)
        return self._side_meshes(bbox)

    def _car_mesh(self, bbox):
        if self.use_car_ply:
            return get_model_mesh(self.default_car_model, bbox)
        return get_box_mesh(bbox)

    @staticmethod
    def _sample_box_surface(bbox, n=400, rng=None):
        """在幽灵框表面均匀采样点，射线未打中时作为注入回退。"""
        rng = np.random.RandomState(0) if rng is None else rng
        x, y, z, l, w, h, yaw = [float(v) for v in bbox[:7]]
        areas = np.array([w * h, w * h, l * h, l * h, l * w, l * w], dtype=np.float64)
        areas = areas / areas.sum()
        face = rng.choice(6, size=n, p=areas)
        u = rng.rand(n) - 0.5
        v = rng.rand(n) - 0.5
        local = np.zeros((n, 3), dtype=np.float64)
        m0 = face == 0
        local[m0] = np.stack([np.full(m0.sum(), 0.5 * l), u[m0] * w, (v[m0] + 0.5) * h], 1)
        m1 = face == 1
        local[m1] = np.stack([np.full(m1.sum(), -0.5 * l), u[m1] * w, (v[m1] + 0.5) * h], 1)
        m2 = face == 2
        local[m2] = np.stack([u[m2] * l, np.full(m2.sum(), 0.5 * w), (v[m2] + 0.5) * h], 1)
        m3 = face == 3
        local[m3] = np.stack([u[m3] * l, np.full(m3.sum(), -0.5 * w), (v[m3] + 0.5) * h], 1)
        m4 = face == 4
        local[m4] = np.stack([u[m4] * l, v[m4] * w, np.full(m4.sum(), h)], 1)
        m5 = face == 5
        local[m5] = np.stack([u[m5] * l, v[m5] * w, np.zeros(m5.sum())], 1)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return local.dot(rot.T) + np.array([x, y, z])

    def run_core(self, single_vehicle_case, attack_opts):
        new_case = copy.deepcopy(single_vehicle_case)
        attacker_pcd = new_case["lidar"][:, :3]

        distance = np.sum(attacker_pcd ** 2, axis=1) ** 0.5
        direction = attacker_pcd / np.tile(distance.reshape((-1, 1)), (1, 3))
        rays = np.hstack([np.zeros((direction.shape[0], 3)), direction])

        extra_rays_list = []
        target_offset = attack_opts["position"][:2]
        if self.dense in [1, 2]:
            target_distance = get_distance(target_offset)
            lidar_offset = target_offset / target_distance * (target_distance - self.dense_distance)
            extra_rays = copy.deepcopy(rays)
            extra_rays[:, :2] = lidar_offset
            extra_rays_list.append(extra_rays)
        if self.dense == 2:
            attacker_lidar_pose = attack_opts["lidar_poses"][attack_opts["attacker_vehicle_id"]]
            for vehicle_id, lidar_pose in attack_opts["lidar_poses"].items():
                if vehicle_id == attack_opts["attacker_vehicle_id"]:
                    continue
                lidar_offset = pcd_map_to_sensor(lidar_pose[np.newaxis, :3], attacker_lidar_pose)[0, :2]
                target_distance = get_distance(target_offset, lidar_offset)
                lidar_offset = target_offset + (lidar_offset - target_offset) / target_distance * self.dense_distance
                extra_rays = copy.deepcopy(rays)
                extra_rays[:, :2] = lidar_offset
                extra_rays_list.append(extra_rays)

        car_mesh = self._car_mesh(attack_opts["position"])

        intersect_points = ray_intersection([car_mesh], rays)
        in_range_mask = (intersect_points[:,0] ** 2 < 10000)
        occlusion_mask =  (attacker_pcd[:,0] / intersect_points[:,0] > 1)

        extra_points_list = []
        for extra_rays in extra_rays_list:
            extra_intersect_points = ray_intersection([car_mesh], extra_rays)
            extra_index_mask = (extra_intersect_points[:,0] ** 2 < 10000)
            extra_points_list.append(extra_intersect_points[extra_index_mask])
        
        ignore_indices = None
        append_data = None
        replace_indices = None
        replace_data = None

        if self.dense == 0:
            replace_indices = np.argwhere(in_range_mask * occlusion_mask > 0).reshape(-1)
            replace_data = intersect_points[replace_indices]
            attacker_pcd[replace_indices, :3] = replace_data
        elif self.dense in [1, 2]:
            ignore_indices = np.argwhere(in_range_mask * occlusion_mask > 0).reshape(-1)
            if len(extra_points_list) > 0:
                append_data = np.vstack(extra_points_list)
            else:
                append_data = extra_points_list[0]
            attacker_pcd = np.delete(attacker_pcd, ignore_indices, axis=0)
            attacker_pcd = np.vstack([attacker_pcd, append_data])
            attacker_pcd, _ = sort_lidar_points(attacker_pcd)

        info = {
            "replace_indices": replace_indices,
            "replace_data": replace_data,
            "ignore_indices": ignore_indices,
            "append_data": append_data
        }

        new_case["lidar"] = np.hstack([attacker_pcd, np.ones((attacker_pcd.shape[0], 1))])
        return new_case, info

    def run_core_sample(self, single_vehicle_case, attack_opts):
        np.random.seed(0)
        new_case = copy.deepcopy(single_vehicle_case)
        attacker_pcd = new_case["lidar"][:, :3]
        bbox_to_spoof = attack_opts["position"]

        points = attacker_pcd
        distance = np.sum(points ** 2, axis=1) ** 0.5
        direction = points / np.tile(distance.reshape((-1, 1)), (1, 3))
        rays = np.hstack([np.zeros((direction.shape[0], 3)), direction])

        meshes = self._spoof_meshes(bbox_to_spoof)
        replace_mask_list = []
        replace_data_list = []
        for i in range(len(meshes)):
            intersect_points = ray_intersection([meshes[i]], rays)
            in_range_mask = (intersect_points[:,0] ** 2 < 10000)
            replace_mask_list.append(in_range_mask)
            replace_data_list.append(intersect_points)

        mesh_weight = np.zeros(len(meshes))
        attacker_lidar_pose = attack_opts["lidar_poses"][attack_opts["attacker_vehicle_id"]]
        for vehicle_id, lidar_pose in attack_opts["lidar_poses"].items():
            if vehicle_id == attack_opts["attacker_vehicle_id"]:
                continue
            lidar_offset = pcd_map_to_sensor(lidar_pose[np.newaxis, :3], attacker_lidar_pose)[0, :3]
            for i, mesh in enumerate(meshes):
                vertices = np.asarray(mesh.vertices)
                h_angle = np.arctan2(vertices[:, 1] - lidar_offset[1], vertices[:, 0] - lidar_offset[0])
                v_angle = (vertices[:, 2] - lidar_offset[2]) / get_distance(vertices[:, :2], lidar_offset[:2])
                mesh_weight[i] += ((h_angle.max() - h_angle.min()) / 0.005) * ((v_angle.max() - v_angle.min()) / 0.01)
        if float(mesh_weight.sum()) <= 0:
            mesh_weight[:] = 1.0

        replace_data = []
        point_sampling_weight = np.vstack(replace_mask_list).T * mesh_weight
        replace_indices = np.argwhere(np.logical_or.reduce(replace_mask_list)).reshape(-1).astype(np.int32)
        for i in replace_indices:
            w = point_sampling_weight[i]
            s = float(np.sum(w))
            if s <= 0:
                continue
            replace_data.append(
                replace_data_list[np.random.choice(mesh_weight.shape[0], p=w / s)][i]
            )
        replace_data = np.array(replace_data) if len(replace_data) else np.zeros((0, 3))
        if replace_indices.shape[0] != replace_data.shape[0]:
            replace_indices = replace_indices[:replace_data.shape[0]]
        n_hit = int(replace_data.shape[0])

        extra_hit = np.zeros((0, 3), dtype=np.float64)
        tgt = np.asarray(bbox_to_spoof[:3], dtype=np.float64)
        dist = float(np.linalg.norm(tgt))
        if dist > float(self.dense_distance) + 1.0:
            origin = tgt * (1.0 - float(self.dense_distance) / dist)
            extra_rays = np.hstack([
                np.repeat(origin.reshape(1, 3), direction.shape[0], axis=0),
                direction])
            chunks = []
            for mesh in meshes:
                ip = ray_intersection([mesh], extra_rays)
                ok = np.isfinite(ip).all(axis=1) & (np.sum(ip ** 2, axis=1) < 1e8)
                if np.any(ok):
                    chunks.append(ip[ok])
            if chunks:
                extra_hit = np.vstack(chunks)

        extra_surf = self._sample_box_surface(bbox_to_spoof, n=2500)
        extra_vol = self._sample_box_surface(bbox_to_spoof, n=2500)
        append_data = np.vstack([x for x in (extra_hit, extra_surf, extra_vol) if x.shape[0] > 0])
        print("[early-spoof] 射线替换 {}，近距加密 {}，采样追加 {}, z=[{:.2f},{:.2f}]".format(
            n_hit, extra_hit.shape[0], append_data.shape[0],
            float(append_data[:, 2].min()), float(append_data[:, 2].max())))

        lidar = new_case["lidar"]
        if n_hit > 0:
            lidar[replace_indices, :3] = replace_data.astype(lidar.dtype)
        pad = np.ones((append_data.shape[0], lidar.shape[1]), dtype=lidar.dtype)
        pad[:, :3] = append_data.astype(lidar.dtype)
        if lidar.shape[1] > 3:
            pad[:, 3] = 1
        new_case["lidar"] = np.vstack([lidar, pad])
        return new_case, {
            "replace_indices": replace_indices if n_hit > 0 else None,
            "replace_data": replace_data if n_hit > 0 else None,
            "ignore_indices": None,
            "append_data": append_data,
        }
