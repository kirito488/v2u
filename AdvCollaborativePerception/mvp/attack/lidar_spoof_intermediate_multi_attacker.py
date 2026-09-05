"""Intermediate-fusion spoofing attack that injects 1-3 fake objects per frame.

Each frame is attacked **independently** (single-frame, no temporal consistency).
Per frame we:

1. borrow the geometry (size + world orientation) of up to ``K`` real objects from
   the dataset (the "donor" objects), and
2. place the fake objects at locations chosen to be **danger regions** (ahead of the
   ego in its lane) or **blind spots** (sparse ego LiDAR / occlusion).

The injection itself reuses ``OpencoodPerception.attack_intermediate`` with a list of
ego-frame bboxes (N perturbation blocks optimized jointly).
"""

import copy
import numpy as np

from .attacker import Attacker
from .donor_selector import select_donor_boxes, select_attack_positions


class LidarSpoofIntermediateMultiAttacker(Attacker):
    def __init__(self, perception, dataset=None, num_objects=2, step=20,
                 max_perturb=3, feature_size=5, seed=0):
        super().__init__()
        self.name = "lidar_spoof_intermediate_multi"
        if perception.model_name != "pointpillar":
            self.name += "_{}".format(perception.model_name)
        self.perception = perception
        self.dataset = dataset
        self.num_objects = num_objects
        self.step = step
        self.max_perturb = max_perturb
        self.feature_size = feature_size
        self.rng = np.random.RandomState(seed)

    def run(self, multi_frame_case, attack_opts):
        """attack_opts: {
            "frame_ids": list[int],
            "attacker_vehicle_id": str,   # UAV
            "victim_vehicle_id": str,     # ego car
            "num_objects"?: int,          # override K per frame (1-3)
            "positions"?: dict[int, list],  # optional override of ego-frame positions
        }
        """
        new_case = copy.deepcopy(multi_frame_case)
        attack_info = [{} for _ in range(len(multi_frame_case))]

        ego_id = attack_opts["victim_vehicle_id"]
        attacker_id = attack_opts["attacker_vehicle_id"]
        num_objects = attack_opts.get("num_objects", self.num_objects)
        positions_override = attack_opts.get("positions", {})

        for frame_id in attack_opts["frame_ids"]:
            frame = multi_frame_case[frame_id]
            ego_case = frame[ego_id]

            # 1. Borrow geometry from up to K real objects in the scene.
            donors = select_donor_boxes(multi_frame_case, attacker_id, self.rng, num_objects)
            K = min(num_objects, len(donors))
            if K == 0:
                print("[multi-spoof] frame {}: no donor objects available, skipping".format(frame_id))
                continue

            # 2. Choose K dangerous / blind positions in the ego frame.
            if frame_id in positions_override and len(positions_override[frame_id]) >= K:
                positions = [list(p[:3]) for p in positions_override[frame_id]]
            else:
                positions, scores = select_attack_positions(
                    ego_case["lidar"], ego_case["gt_bboxes"], K, self.rng)

            # 3. Assemble ego-frame bboxes: position from blind/danger spot,
            #    geometry (l,w,h) and orientation from the donor.
            ego_yaw = np.deg2rad(ego_case["lidar_pose"][4])
            bboxes = []
            donor_sizes = []
            for i in range(K):
                l, w, h, world_yaw = donors[i]
                yaw_ego = world_yaw - ego_yaw
                x, y, z = positions[i]
                bboxes.append(np.array([x, y, z, l, w, h, yaw_ego], dtype=np.float64))
                donor_sizes.append([l, w, h])

            # 4. Run the multi-object intermediate spoofing optimization.
            result = self.perception.attack_intermediate(
                frame, ego_id, attacker_id,
                mode="spoof", bboxes=bboxes,
                max_iteration=self.step, max_perturb=self.max_perturb,
                feature_size=self.feature_size)

            new_case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
            new_case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
            attack_info[frame_id][ego_id] = {
                "pred_bboxes": result["pred_bboxes"],
                "pred_scores": result["pred_scores"],
                "injected_bboxes": [b.tolist() for b in bboxes],
                "donor_sizes": donor_sizes,
                "positions": [list(p) for p in positions],
            }

        return new_case, attack_info
