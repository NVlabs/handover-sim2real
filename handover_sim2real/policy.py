# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import numpy as np
import pybullet
import ray

from handover.benchmark_runner import timer
from scipy.spatial.transform import Rotation as Rot

from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")

from core.utils import (
    regularize_pc_point_count,
    se3_inverse,
    se3_transform_pc,
    tf_quat,
    unpack_action,
    unpack_pose,
)


class PointListener:
    def __init__(self, cfg, seed=None):
        self._cfg = cfg
        self._seed = seed

        self._merge_ratios = [
            self._cfg.POLICY.POINT_STATE_YCB_RATIO,
            1.0 - self._cfg.POLICY.POINT_STATE_YCB_RATIO,
        ]

    @property
    def num_point_states(self):
        return len(self._merge_ratios)

    def reset(self):
        self._acc_points = [np.zeros((3, 0), dtype=np.float32) for _ in self._merge_ratios]

        if self._seed is not None:
            np.random.seed(self._seed)

    @property
    def acc_points(self):
        return self._acc_points

    def point_states_to_state(self, point_states, ee_pose):
        point_states = self._process_pointcloud(point_states, ee_pose)

        if all(point_state.shape[1] > 0 for point_state in point_states[1:]):
            point_states_to_merge = []
            for point_state, merge_ratio in zip(point_states, self._merge_ratios):
                point_state = point_state[
                    :, : int(self._cfg.RL_TRAIN.uniform_num_pts * merge_ratio)
                ]
                point_states_to_merge.append(point_state)
            point_state = np.concatenate(point_states_to_merge, axis=-1)
        else:
            point_state = point_states[0]

        return [(point_state, np.array([])), None, None, None]

    def _process_pointcloud(self, point_states, ee_pose):
        self._update_acc_points(point_states, ee_pose)

        inv_ee_pose = se3_inverse(ee_pose)
        point_states = []
        for i, acc_points in enumerate(self.acc_points):
            point_state = se3_transform_pc(inv_ee_pose, acc_points)
            if point_state.shape[1] > 0:
                point_state = regularize_pc_point_count(
                    point_state.T, self._cfg.RL_TRAIN.uniform_num_pts
                ).T

            point_state_ = np.zeros(
                (3 + len(self.acc_points), point_state.shape[1]), dtype=np.float32
            )
            point_state_[:3] = point_state
            point_state_[3 + i] = 1

            point_states.append(point_state_)

        return point_states

    def _update_acc_points(self, point_states, ee_pose):
        for i, point_state in enumerate(point_states):
            if point_state.shape[1] == 0:
                continue
            new_points = se3_transform_pc(ee_pose, point_state)
            index = np.random.choice(
                range(new_points.shape[1]),
                size=min(self._cfg.RL_TRAIN.uniform_num_pts, new_points.shape[1]),
                replace=False,
            )
            self.acc_points[i] = new_points[:, index]


class HandoverSim2RealPolicy:
    def __init__(self, cfg, agent, grasp_agent, grasp_pred_threshold, use_ray=False, seed=None):
        self._cfg = cfg
        self._agent = agent
        self._grasp_agent = grasp_agent
        self._grasp_pred_threshold = grasp_pred_threshold
        self._use_ray = use_ray

        self._point_listener = PointListener(cfg, seed=seed)

        self._panda_base_invert_transform = pybullet.invertTransform(
            self._cfg.ENV.PANDA_BASE_POSITION, self._cfg.ENV.PANDA_BASE_ORIENTATION
        )

        self._steps_action_repeat = int(
            self._cfg.POLICY.TIME_ACTION_REPEAT / self._cfg.SIM.TIME_STEP
        )
        self._steps_close_gripper = int(
            self._cfg.POLICY.TIME_CLOSE_GRIPPER / self._cfg.SIM.TIME_STEP
        )
        self._standoff_offset = np.array([0.0, 0.0, 0.08])

    @property
    def steps_action_repeat(self):
        return self._steps_action_repeat

    def reset(self):
        self._done_frame = None
        self._grasp = None
        self._back = None

        self._point_listener.reset()

    def get_state(self, obs):
        point_states, elapsed_time = self._get_point_states_from_callback(obs)
        ee_pose = self._get_ee_pose(obs, in_panda_base=True)
        state = self._point_listener.point_states_to_state(point_states, ee_pose)
        return state, elapsed_time

    @timer
    def _get_point_states_from_callback(self, obs):
        point_states = obs["callback_get_point_states"]()
        point_states = [point_state.T for point_state in point_states]
        return point_states

    def _get_ee_pose(self, obs, in_panda_base=False):
        pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3]
        orn = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 3:7]
        if in_panda_base:
            pos, orn = pybullet.multiplyTransforms(*self._panda_base_invert_transform, pos, orn)
        ee_pose = unpack_pose(np.hstack((pos, tf_quat(orn))))
        return ee_pose

    def select_action(self, state, expert_policy=False):
        if self._use_ray:
            action, _, _, _ = ray.get(
                self._agent.select_action.remote(
                    state, remain_timestep=1, expert_policy=expert_policy
                )
            )
        else:
            action, _, _, _ = self._agent.select_action(
                state, remain_timestep=1, expert_policy=expert_policy
            )
        return action

    def convert_action_to_target_joint_position(self, action, obs):
        ee_pose = self._get_ee_pose(obs)
        delta_ee_pose = unpack_action(action)
        target_ee_pose = np.matmul(ee_pose, delta_ee_pose)

        pos = target_ee_pose[:3, 3]
        orn = Rot.from_matrix(target_ee_pose[:3, :3]).as_quat()
        target_joint_position = pybullet.calculateInverseKinematics(
            obs["panda_body"].contact_id[0], obs["panda_link_ind_hand"] - 1, pos, orn
        )
        target_joint_position = np.array(target_joint_position)
        target_joint_position[7:9] = 0.04

        return target_joint_position

    def select_action_grasp(self, state):
        if self._use_ray:
            action = ray.get(
                self._grasp_agent.select_action_grasp.remote(state, self._grasp_pred_threshold)
            )
        else:
            action = self._grasp_agent.select_action_grasp(state, self._grasp_pred_threshold)
        return action

    def grasp_and_back(self, obs):
        if self._done_frame is None:
            self._done_frame = obs["frame"]

        done = False

        if obs["frame"] < self._done_frame + 4 * self._steps_action_repeat:
            if self._grasp is None:
                pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
                orn = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 3:7].numpy()
                R = Rot.from_quat(orn).as_matrix()
                reach_goal = np.matmul(R, self._standoff_offset) + pos
                reach_traj = np.linspace(pos, reach_goal, 5)[1:]

                self._grasp = []
                for pos in reach_traj:
                    conf = pybullet.calculateInverseKinematics(
                        obs["panda_body"].contact_id[0],
                        obs["panda_link_ind_hand"] - 1,
                        pos,
                        orn,
                    )
                    conf = np.array(conf)
                    conf[7:9] = 0.04
                    self._grasp.append(conf)

            i = (obs["frame"] - self._done_frame) // self._steps_action_repeat
            action = self._grasp[i].copy()
        elif (
            obs["frame"]
            < self._done_frame + 4 * self._steps_action_repeat + self._steps_close_gripper
        ):
            action = self._grasp[3].copy()
            action[7:9] = 0.0
        else:
            if self._back is None:
                self._back = []
                pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
                dpos_goal = self._cfg.BENCHMARK.GOAL_CENTER - pos
                dpos_step = dpos_goal / np.linalg.norm(dpos_goal) * self._cfg.POLICY.BACK_STEP_SIZE
                num_steps = int(
                    np.ceil(np.linalg.norm(dpos_goal) / self._cfg.POLICY.BACK_STEP_SIZE)
                )
                for _ in range(num_steps):
                    pos += dpos_step
                    conf = pybullet.calculateInverseKinematics(
                        obs["panda_body"].contact_id[0], obs["panda_link_ind_hand"] - 1, pos
                    )
                    conf = np.array(conf)
                    conf[7:9] = 0.0
                    self._back.append(conf)

            num_frames = (
                obs["frame"]
                - self._done_frame
                - 4 * self._steps_action_repeat
                - self._steps_close_gripper
            )
            i = num_frames // self._steps_action_repeat
            i = min(i, len(self._back) - 1)
            action = self._back[i].copy()
            done = i == len(self._back) - 1

        return action, done
