# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import abc
import argparse
import numpy as np
import os

from handover.benchmark_runner import timer, BenchmarkRunner

from handover_sim2real.config import get_cfg
from handover_sim2real.policy import HandoverSim2RealPolicy
from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")

from core.bc import BC
from core.ddpg import DDPG
from core.utils import make_nets_opts_schedulers
from env.panda_scene import PandaTaskSpace6D
from experiments.config import cfg_from_file

seed = 123456


def parse_args():
    parser = argparse.ArgumentParser(description="Test.")
    parser.add_argument("--model-dir", help="model directory")
    parser.add_argument("--without-hold", action="store_true", help="use without hold policy")
    parser.add_argument("--name", help="benchmark name")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help=(
            """modify config options at the end of the command; use space-separated """
            """"PATH.KEY VALUE" pairs; see handover_sim2real/config.py, """
            """handover-sim/handover/config.py, and easysim/src/easysim/config.py for all options"""
        ),
    )
    args = parser.parse_args()
    return args


class HandoverSim2RealPolicyTest(HandoverSim2RealPolicy, abc.ABC):
    def __init__(self, cfg, agent, grasp_agent, grasp_pred_threshold, name):
        super().__init__(cfg, agent, grasp_agent, grasp_pred_threshold, seed=seed)

        if name is not None:
            self._name += "_{}".format(name)

        self._start_conf = np.array(self._cfg.ENV.PANDA_INITIAL_POSITION)
        self._steps_wait = int(self._TIME_WAIT / self._cfg.SIM.TIME_STEP)
        self._max_policy_steps = (self._cfg.RL_MAX_STEP + 7) * self._steps_action_repeat

        # Warm up network. The first pass will be slower than later ones so we want to exclude it
        # from benchmark time.
        _, warm_up_time = self._warm_up_network()
        print("warn up time: {:6.2f}".format(warm_up_time))

    @timer
    def _warm_up_network(self):
        state = [
            (np.zeros((3 + self._point_listener.num_point_states, 1)), np.array([])),
            None,
            None,
            None,
        ]
        self.select_action(state)
        self.select_action_grasp(state)

    @property
    def name(self):
        return self._name

    def reset(self):
        super().reset()

        self._done = False
        self._traj = []

    def forward(self, obs):
        info = {}

        if obs["frame"] < self._steps_wait:
            # Wait.
            action = self._start_conf.copy()
        elif not self._done:
            # Approach object until reaching pre-grasp pose.
            action, done, info = self._plan(obs)
            if done:
                self._done = True
                self._done_frame = obs["frame"] + 1
        else:
            # Grasp and back.
            action, _ = self.grasp_and_back(obs)

        return action, info

    def _plan(self, obs):
        info = {}

        if (obs["frame"] - self._steps_wait) % self._steps_action_repeat == 0:
            state, info["obs_time"] = self.get_state(obs)
            action = self.select_action(state)
            action = self.convert_action_to_target_joint_position(action, obs)
            self._traj.append(action)
        else:
            action = self._traj[-1].copy()

        if (obs["frame"] - self._steps_wait + 1) % self._steps_action_repeat == 0:
            done = (obs["frame"] - self._steps_wait + 1) == self._max_policy_steps
            state, info["obs_time"] = self.get_state(obs)
            grasp_pred = self.select_action_grasp(state)
            done |= grasp_pred
        else:
            done = False

        return action, done, info


class HandoverSim2RealHoldPolicy(HandoverSim2RealPolicyTest):
    _name = "handover-sim2real-hold"
    _TIME_WAIT = 3.0


class HandoverSim2RealWithoutHoldPolicy(HandoverSim2RealPolicyTest):
    _name = "handover-sim2real-wo-hold"
    _TIME_WAIT = 1.5


def main():
    args = parse_args()

    model_cfg = get_cfg()
    cfg_from_file(
        filename=os.path.join(args.model_dir, "config.yaml"),
        dict=model_cfg,
        reset_model_spec=False,
        merge_to_cn_dict=True,
    )

    agent = DDPG(model_cfg.RL_TRAIN.feature_input_dim, PandaTaskSpace6D(), model_cfg.RL_TRAIN)
    net_dict = make_nets_opts_schedulers(model_cfg.RL_MODEL_SPEC, model_cfg.RL_TRAIN)
    agent.setup_feature_extractor(net_dict)
    agent.load_model(args.model_dir)

    grasp_pretrained_dir = os.path.join("output", "grasp_trigger_PRE_2")
    grasp_cfg = cfg_from_file(
        filename=os.path.join(grasp_pretrained_dir, "config.yaml"), no_merge=True
    )
    grasp_model_surfix = "epoch_20000"
    grasp_pred_threshold = 0.9

    grasp_agent = BC(grasp_cfg.RL_TRAIN.feature_input_dim, PandaTaskSpace6D(), grasp_cfg.RL_TRAIN)
    grasp_net_dict = make_nets_opts_schedulers(grasp_cfg.RL_MODEL_SPEC, grasp_cfg.RL_TRAIN)
    grasp_agent.setup_feature_extractor(grasp_net_dict)
    grasp_agent.load_model(grasp_pretrained_dir, surfix=grasp_model_surfix)

    if not args.without_hold:
        Policy = HandoverSim2RealHoldPolicy
    else:
        Policy = HandoverSim2RealWithoutHoldPolicy
    policy = Policy(model_cfg, agent, grasp_agent, grasp_pred_threshold, args.name)

    cfg = get_cfg(handover_config_only=True)
    cfg.merge_from_file(os.path.join("examples", "test.yaml"))
    cfg.merge_from_list(args.opts)

    benchmark_runner = BenchmarkRunner(cfg)
    benchmark_runner.run(policy)


if __name__ == "__main__":
    main()
