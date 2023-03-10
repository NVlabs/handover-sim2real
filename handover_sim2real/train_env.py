# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import numpy as np
import pybullet
import random

from handover.benchmark_wrapper import EpisodeStatus
from handover.handover_env import HandoverHandCameraPointStateEnv
from scipy.spatial.transform import Rotation as Rot

from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("OMG_PLANNER_DIR")

from omg.config import cfg as omg_cfg
from omg.core import PlanningScene
from ycb_render.robotPose import robot_pykdl

add_sys_path_from_env("GADDPG_DIR")

from core.utils import (
    anchor_seeds,
    hand_finger_point,
    inv_lookat,
    inv_relative_pose,
    mat2euler,
    mat2quat,
    pack_pose,
    pack_pose_rot_first,
    ros_quat,
    rotZ,
    se3_inverse,
    tf_quat,
    unpack_pose,
    wrap_value,
)


class OMGPlanner:
    def __init__(self, cfg):
        self._cfg = cfg

        for key, val in self._cfg.items():
            setattr(omg_cfg, key, val)

        omg_cfg.get_global_path()

        # Enforce determinism. This accounts for the call of random.sample() in
        # `Robot.load_collision_points()` in `OMG-Planner/omg/core.py`.
        random.seed(0)

        self._scene = PlanningScene(omg_cfg)

    def reset_scene(self, names, poses):
        for name in list(self._scene.env.names):
            self._scene.env.remove_object(name, lazy=True)
        assert len(self._scene.env.objects) == 0

        for name, pose in zip(names, poses):
            self._scene.env.add_object(name, pose[:3], pose[3:], compute_grasp=False)
        self._scene.env.combine_sdfs()

        self._grasp_computed = False

    def plan_to_target(self, start_conf, target_name, num_steps, scene_idx):
        self._scene.traj.start = start_conf
        self._scene.env.set_target(target_name)

        omg_cfg.timesteps = num_steps
        omg_cfg.get_global_param(steps=omg_cfg.timesteps)

        if not hasattr(self._scene, "planner"):
            self._scene.reset(scene_idx=scene_idx)
        else:
            if self._grasp_computed:
                self._scene.env.objects[0].compute_grasp = False
            self._scene.update_planner(scene_idx=scene_idx)

        if not self._grasp_computed:
            self._grasp_computed = True

        info = self._scene.step()
        traj = self._scene.planner.history_trajectories[-1]

        if len(info) == 0:
            traj = None

        return traj, info

    def get_grasp_poses(self):
        return self._scene.env.objects[self._scene.env.target_idx].grasps_poses


class HandoverSim2RealTrainEnv(HandoverHandCameraPointStateEnv):
    def init(self):
        super().init()

        self._panda_base_invert_transform = pybullet.invertTransform(
            self._cfg.ENV.PANDA_BASE_POSITION, self._cfg.ENV.PANDA_BASE_ORIENTATION
        )
        self._panda_base_pose = (
            self.cfg.ENV.PANDA_BASE_POSITION
            + self.cfg.ENV.PANDA_BASE_ORIENTATION[3:]
            + self.cfg.ENV.PANDA_BASE_ORIENTATION[:3]
        )

        self._panda_kinematics = robot_pykdl.robot_kinematics(None, data_path=self.cfg.ROOT_DIR)

        self._omg_planner = OMGPlanner(self._cfg.omg_config)

    def post_reset(self, env_ids, scene_id):
        self._omg_planner_goal_pose = None

        return super().post_reset(env_ids, scene_id)

    def callback_get_reward_post_status(self, reward, status):
        if status == EpisodeStatus.SUCCESS:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    def _get_ee_pose(self):
        pos = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3]
        orn = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 3:7]
        ee_pose = np.hstack((pos, tf_quat(orn)))
        return ee_pose

    def randomize_arm_init(self, near, far):
        pos = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3].tolist()
        orn = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 3:7].tolist()
        ycb_pose = pos + orn[3:] + orn[:3]
        panda_base_to_ycb_pose = inv_relative_pose(ycb_pose, self._panda_base_pose)
        panda_base_to_ycb_trans = panda_base_to_ycb_pose[:3, 3]

        outer_loop_num = 20
        inner_loop_num = 5

        for _ in range(outer_loop_num):
            theta = np.random.uniform(low=0, high=np.pi * 2 / 3)
            phi = np.random.uniform(low=np.pi / 2, high=np.pi * 3 / 2)
            r = np.random.uniform(low=near, high=far)
            pos = np.array(
                [
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                ]
            )

            position = (
                panda_base_to_ycb_trans + pos + np.random.uniform(low=-0.03, high=0.03, size=3)
            )
            position[0] = np.clip(position[0], +0.0, +0.5)
            position[1] = np.clip(position[1], -0.3, +0.3)
            position[2] = np.clip(position[2], +0.2, +0.6)

            pos = position - panda_base_to_ycb_trans
            up = np.array([0.0, 0.0, -1.0])

            for _ in range(inner_loop_num):
                R = np.matmul(inv_lookat(pos, 2 * pos, up), rotZ(-np.pi / 2)[:3, :3])
                orientation = ros_quat(mat2quat(R))
                anchor_idx = np.random.randint(len(anchor_seeds))
                q_out = self._panda_kinematics.inverse_kinematics(
                    position, orientation=orientation, seed=anchor_seeds[anchor_idx]
                )
                if q_out is not None:
                    break

        if q_out is not None:
            q_out = q_out.tolist() + [0.04, 0.04]

        return q_out

    def set_initial_joint_position(self, initial_joint_position):
        self.panda.body.initial_dof_position = initial_joint_position

    def get_ee_to_ycb_distance(self):
        ee_pos = self._get_ee_pose()[:3]
        ycb_pos = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        ee_to_ycb_distance = np.linalg.norm(ee_pos - ycb_pos)
        return ee_to_ycb_distance

    def run_omg_planner(self, num_steps, scene_idx, reset_scene=True):
        if reset_scene:
            names = []
            poses = []
            for i in range(len(self.ycb.ids)):
                names += [self.ycb.CLASSES[self.ycb.ids[i]]]
                pos = self.ycb.pose[-1, i, 0:3]
                orn = self.ycb.pose[-1, i, 3:6]
                orn = Rot.from_euler("XYZ", orn).as_quat()
                pos, orn = pybullet.multiplyTransforms(*self._panda_base_invert_transform, pos, orn)
                poses += [pos + orn[3:] + orn[:3]]

            self._omg_planner.reset_scene(names, poses)

        start_conf = self.panda.body.dof_state[0, :, 0]
        target_name = self.ycb.CLASSES[self.ycb.ids[0]]

        traj, info = self._omg_planner.plan_to_target(start_conf, target_name, num_steps, scene_idx)

        if traj is None:
            print("Planning not run due to empty goal set.")
        else:
            goal_joint_position = traj[-5]
            panda_base_to_goal_pose = self._panda_kinematics.forward_kinematics_parallel(
                joint_values=wrap_value(goal_joint_position)[None], offset=False
            )[0, 7]
            self._omg_planner_goal_pose = pack_pose(
                np.matmul(unpack_pose(self._panda_base_pose), panda_base_to_goal_pose)
            )

        return traj, info

    def convert_target_joint_position_to_action(self, target_joint_position):
        current_joint_position = self.panda.body.dof_state[0, :, 0]
        current_ee_pose = self._panda_kinematics.forward_kinematics_parallel(
            joint_values=wrap_value(current_joint_position)[None], offset=False
        )[0, 7]
        target_ee_pose = self._panda_kinematics.forward_kinematics_parallel(
            joint_values=wrap_value(target_joint_position)[None], offset=False
        )[0, 7]
        delta_ee_pose = np.matmul(se3_inverse(current_ee_pose), target_ee_pose)
        action = np.hstack((delta_ee_pose[:3, 3], mat2euler(delta_ee_pose[:3, :3])))
        return action

    def get_ee_to_goal_pose(self, nearest=False):
        if nearest:
            return self._get_ee_to_nearest_goal_pose()
        ee_pose = self._get_ee_pose()
        ee_to_goal_pose = pack_pose_rot_first(
            inv_relative_pose(self._omg_planner_goal_pose, ee_pose)
        )
        return ee_to_goal_pose

    def _get_ee_to_nearest_goal_pose(self):
        ycb_to_goal_poses = self._omg_planner.get_grasp_poses()

        pos = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3].tolist()
        orn = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 3:7].tolist()
        ycb_pose = pos + orn[3:] + orn[:3]
        ee_pose = self._get_ee_pose()
        goal_poses = np.matmul(unpack_pose(ycb_pose), ycb_to_goal_poses)
        ee_to_goal_poses = np.matmul(se3_inverse(unpack_pose(ee_pose)), goal_poses)

        point = hand_finger_point
        point_goal_poses = (
            np.matmul(ee_to_goal_poses[:, :3, :3], hand_finger_point) + ee_to_goal_poses[:, :3, 3:4]
        )
        index = np.argmin(np.mean(np.sum(np.abs(point - point_goal_poses), axis=1), axis=-1))
        ee_to_nearest_goal_pose = pack_pose_rot_first(ee_to_goal_poses[index])

        return ee_to_nearest_goal_pose
