# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import argparse
import gym
import itertools
import numpy as np
import os
import ray

from datetime import datetime
from handover.benchmark_wrapper import EpisodeStatus, HandoverBenchmarkWrapper

from handover_sim2real.config import get_cfg
from handover_sim2real.policy import HandoverSim2RealPolicy
from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")

from experiments.config import cfg_from_file, save_cfg_to_file
from core.trainer import (
    AgentWrapper,
    AgentWrapperGPU05,
    ReplayMemoryWrapper,
    ReplayMemoryWrapperBase,
    RolloutAgentWrapperGPU1,
    Trainer,
    TrainerRemote,
)
from core.utils import get_noise_delta, get_valid_index, rand_sample_joint


def parse_args():
    parser = argparse.ArgumentParser(description="Train.")
    parser.add_argument("--cfg-file", help="path to config file")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--use-grasp-predictor", action="store_true", help="use grasp predictor")
    parser.add_argument("--use-ray", action="store_true", help="use Ray")
    parser.add_argument("--pretrained-dir", help="pretrained model directory")
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


class ActorWrapper:
    def __init__(
        self,
        stage,
        cfg,
        use_ray,
        rollout_agent,
        expert_buffer,
        online_buffer,
        actor_seed,
        grasp_agent,
        grasp_pred_threshold,
    ):
        self._stage = stage
        self._cfg = cfg
        self._use_ray = use_ray
        self._expert_buffer = expert_buffer
        self._online_buffer = online_buffer
        self._use_grasp_predictor = grasp_agent is not None

        self._env = HandoverBenchmarkWrapper(gym.make(self._cfg.ENV.ID, cfg=self._cfg))

        self._policy = HandoverSim2RealPolicy(
            cfg, rollout_agent, grasp_agent, grasp_pred_threshold, use_ray=self._use_ray
        )

        self._max_explore_steps = self._cfg.RL_MAX_STEP + 7

        if actor_seed is not None:
            np.random.seed(seed=actor_seed)

    def rollout(self, num_episodes, explore, test, noise_scale):
        for _ in range(num_episodes):
            self._rollout_one(explore, test, noise_scale)

    def _rollout_one(self, explore, test, noise_scale):
        scene_idx = np.random.randint(self._env.num_scenes)

        if self._stage == "pretrain":
            sample_initial_joint_position = (
                np.random.uniform()
                < self._cfg.RL_TRAIN.HANDOVER_SIM2REAL.sample_initial_joint_position_ratio
            )
        if self._stage == "finetune":
            sample_initial_joint_position = False
        reset_to_sample = False
        if sample_initial_joint_position:
            self._env.reset(idx=scene_idx)
            for _ in range(self._cfg.RL_TRAIN.ENV_RESET_TRIALS):
                initial_joint_position = rand_sample_joint(self._env, init_joints=None)
                if initial_joint_position is not None:
                    self._env.set_initial_joint_position(initial_joint_position)
                    obs = self._env.reset(idx=scene_idx)
                    if self._env.get_ee_to_ycb_distance() > self._cfg.RL_TRAIN.init_distance_low:
                        reset_to_sample = True
                        break
        if not reset_to_sample:
            self._env.set_initial_joint_position(self._cfg.ENV.PANDA_INITIAL_POSITION)
            obs = self._env.reset(idx=scene_idx)

        self._policy.reset()

        expert_plan, _ = self._env.run_omg_planner(self._cfg.RL_MAX_STEP, scene_idx)
        if expert_plan is None:
            return

        if self._stage == "pretrain" and explore:
            expert_initial = self._cfg.RL_TRAIN.expert_initial_state and not test
            if expert_initial:
                expert_initial_steps = np.random.randint(
                    self._cfg.RL_TRAIN.EXPERT_INIT_MIN_STEP,
                    high=self._cfg.RL_TRAIN.EXPERT_INIT_MAX_STEP,
                )

        step = 0
        done = False
        cur_episode = []

        while not done:
            state, _ = self._policy.get_state(obs)

            if self._stage == "pretrain":
                apply_dart = (
                    self._cfg.RL_TRAIN.dart
                    and not explore
                    and reset_to_sample
                    and step > self._cfg.RL_TRAIN.DART_MIN_STEP
                    and step < self._cfg.RL_TRAIN.DART_MAX_STEP
                    and np.random.uniform() < self._cfg.RL_TRAIN.DART_RATIO
                )
                apply_dagger = (
                    self._cfg.RL_TRAIN.dagger
                    and explore
                    and reset_to_sample
                    and step > self._cfg.RL_TRAIN.DAGGER_MIN_STEP
                    and step < self._cfg.RL_TRAIN.DAGGER_MAX_STEP
                    and np.random.uniform() < self._cfg.RL_TRAIN.DAGGER_RATIO
                )

                if apply_dart:
                    t = np.random.uniform(low=-0.04, high=+0.04, size=(3,))
                    r = np.random.uniform(low=-0.20, high=+0.20, size=(3,))
                    action = np.hstack([t, r])
                    target_joint_position = self._policy.convert_action_to_target_joint_position(
                        action, obs
                    )
                    obs, _, _, _ = self._step_env_repeat(target_joint_position)

                if apply_dart or apply_dagger:
                    num_steps = self._cfg.RL_MAX_STEP - step
                    expert_plan_dart, _ = self._env.run_omg_planner(
                        num_steps, scene_idx, reset_scene=False
                    )
                    expert_plan = np.concatenate((expert_plan[:step], expert_plan_dart))

            if self._stage == "pretrain":
                nearest = explore and not apply_dagger
                ee_to_goal_pose = self._env.get_ee_to_goal_pose(nearest=nearest)

            if self._stage == "pretrain":
                expert_flag = (
                    not explore or expert_initial and step < expert_initial_steps or apply_dagger
                )
                perturb_flag = apply_dart
            if self._stage == "finetune":
                expert_flag = not explore
                perturb_flag = False

            if self._stage == "pretrain" and expert_flag:
                expert_action = self._env.convert_target_joint_position_to_action(expert_plan[step])

            if (
                not explore
                or self._stage == "pretrain"
                and expert_initial
                and step < expert_initial_steps
            ):
                # Expert.
                if self._stage == "pretrain":
                    action = expert_action
                    target_joint_position = expert_plan[step]
                if self._stage == "finetune":
                    action = self._policy.select_action(state, expert_policy=True)
                    target_joint_position = self._policy.convert_action_to_target_joint_position(
                        action, obs
                    )
            else:
                # Online.
                action = self._policy.select_action(state)

                noise = get_noise_delta(
                    action, self._cfg.RL_TRAIN.action_noise, self._cfg.RL_TRAIN.noise_type
                )
                action = action + noise * noise_scale

                target_joint_position = self._policy.convert_action_to_target_joint_position(
                    action, obs
                )

            if self._stage == "finetune" and expert_flag:
                expert_action = action

            obs, reward, done, info = self._step_env_repeat(
                target_joint_position, break_if_done=True
            )

            run_grasp_and_back = False
            if not done:
                if (
                    step + 1 == self._max_explore_steps
                    or self._stage == "pretrain"
                    and not explore
                    and step == len(expert_plan) - 5
                ):
                    run_grasp_and_back = True
                elif self._use_grasp_predictor and (
                    self._stage == "pretrain" and explore or self._stage == "finetune"
                ):
                    state_grasp, _ = self._policy.get_state(obs)
                    grasp_pred = self._policy.select_action_grasp(state_grasp).item()
                    if grasp_pred:
                        run_grasp_and_back = True
            if run_grasp_and_back:
                back_done = False
                if self._stage == "pretrain" and not explore:
                    obs, _, done, _ = self._step_env_repeat(
                        target_joint_position, break_if_done=True
                    )
                    if done:
                        back_done = True
                while not back_done:
                    target_joint_position, back_done = self._policy.grasp_and_back(obs)
                    obs, reward, done, info = self._step_env_repeat(
                        target_joint_position, break_if_done=True
                    )
                    if done:
                        back_done = True
                if not done:
                    done = True

            failure_1 = (
                info["status"] & EpisodeStatus.FAILURE_HUMAN_CONTACT
                == EpisodeStatus.FAILURE_HUMAN_CONTACT
            )
            failure_2 = (
                info["status"] & EpisodeStatus.FAILURE_OBJECT_DROP
                == EpisodeStatus.FAILURE_OBJECT_DROP
            )
            failure_3 = (
                info["status"] & EpisodeStatus.FAILURE_TIMEOUT == EpisodeStatus.FAILURE_TIMEOUT
            )

            step_dict = {
                "timestep": step,
                "point_state": state[0][0],
                "expert_flags": expert_flag,
                "perturb_flags": perturb_flag,
                "action": action,
                "reward": reward,
                "returns": reward,
                "terminal": done,
                "target_name": "",
                "failure_case_1": failure_1,
                "failure_case_2": failure_2,
                "failure_case_3": failure_3,
            }
            if self._stage == "pretrain":
                step_dict["goal"] = ee_to_goal_pose
            if expert_flag:
                step_dict["expert_action"] = expert_action

            cur_episode.append(step_dict)
            step += 1

        if not explore:
            if self._use_ray:
                self._expert_buffer.add_episode.remote(cur_episode, explore, test)
            else:
                self._expert_buffer.add_episode(cur_episode, explore, test)
        else:
            if self._use_ray:
                self._online_buffer.add_episode.remote(cur_episode, explore, test)
            else:
                self._online_buffer.add_episode(cur_episode, explore, test)

    def _step_env_repeat(self, target_joint_position, break_if_done=False):
        for _ in range(self._policy.steps_action_repeat):
            obs, reward, done, info = self._env.step(target_joint_position)
            if break_if_done and done:
                break
        return obs, reward, done, info


@ray.remote(num_gpus=0.13)
class ActorWrapperRemote(ActorWrapper):
    pass


def main():
    args = parse_args()
    args.log = True
    args.policy = "DDPG"
    args.save_model = True

    cfg = get_cfg()
    cfg_from_file(filename=args.cfg_file, dict=cfg, merge_to_cn_dict=True)
    cfg.merge_from_list(args.opts)

    dt = datetime.now()
    dt = dt.strftime("%Y-%m-%d_%H-%M-%S")
    cfg_name = os.path.basename(args.cfg_file).split(".")[0]
    output_dir = os.path.join(
        "output",
        "{}_{}_{}_{}_{}".format(dt, cfg_name, args.seed, cfg.BENCHMARK.SETUP, cfg.BENCHMARK.SPLIT),
    )
    os.makedirs(output_dir, exist_ok=True)

    cfg.RL_MODEL_SPEC = os.path.join(os.environ["GADDPG_DIR"], cfg.RL_MODEL_SPEC)
    cfg.RL_TRAIN.output_time = output_dir
    cfg.RL_TRAIN.model_output_dir = output_dir
    cfg.RL_TRAIN.logdir = ""
    cfg.omg_config["valid_grasp_dict_path"] = os.path.join("examples", "valid_grasp_dict_005.pkl")

    filename = os.path.join(output_dir, "config.yaml")
    save_cfg_to_file(filename, cfg)

    np.random.seed(args.seed)

    if cfg.RL_TRAIN.HANDOVER_SIM2REAL.stage == "pretrain":
        add_expert = False
    if cfg.RL_TRAIN.HANDOVER_SIM2REAL.stage == "finetune":
        add_expert = True

    if args.use_grasp_predictor:
        grasp_args = type(args)()
        grasp_args.seed = args.seed
        grasp_args.policy = "BC"
        grasp_pretrained_dir = os.path.join("output", "grasp_trigger_PRE_2")
        grasp_cfg = cfg_from_file(
            filename=os.path.join(grasp_pretrained_dir, "config.yaml"), no_merge=True
        )
        grasp_model_surfix = "epoch_20000"
        grasp_pred_threshold = 0.9
    else:
        grasp_agent = None
        grasp_pred_threshold = None

    if args.use_ray:
        runtime_env = {"py_modules": [os.path.join(os.environ["GADDPG_DIR"], "core")]}
        ray.init(runtime_env=runtime_env)
        expert_buffer = ReplayMemoryWrapper.remote(cfg.RL_MEMORY_SIZE, cfg, "expert")
        online_buffer = ReplayMemoryWrapper.remote(cfg.RL_MEMORY_SIZE, cfg, "online")
        rollout_agent = RolloutAgentWrapperGPU1.remote(
            args,
            cfg,
            pretrained_path=args.pretrained_dir,
            input_dim=cfg.RL_TRAIN.feature_input_dim,
            add_expert=add_expert,
        )
        if args.use_grasp_predictor:
            grasp_agent = RolloutAgentWrapperGPU1.remote(
                grasp_args,
                grasp_cfg,
                pretrained_path=grasp_pretrained_dir,
                input_dim=cfg.RL_TRAIN.feature_input_dim,
                model_surfix=grasp_model_surfix,
            )
        actors = []
        for actor_idx in range(cfg.RL_TRAIN.num_remotes):
            actor_seed = args.seed + actor_idx
            actors.append(
                ActorWrapperRemote.remote(
                    cfg.RL_TRAIN.HANDOVER_SIM2REAL.stage,
                    cfg,
                    args.use_ray,
                    rollout_agent,
                    expert_buffer,
                    online_buffer,
                    actor_seed,
                    grasp_agent,
                    grasp_pred_threshold,
                )
            )
        learner_agent = AgentWrapperGPU05.remote(
            args, cfg, pretrained_path=args.pretrained_dir, input_dim=cfg.RL_TRAIN.feature_input_dim
        )
        trainer = TrainerRemote.remote(
            args, cfg, learner_agent, expert_buffer, online_buffer, logdir=output_dir
        )
        num_episodes = 1
        weights = ray.get(learner_agent.get_weight.remote())
    else:
        expert_buffer = ReplayMemoryWrapperBase(cfg.RL_MEMORY_SIZE, cfg, "expert")
        online_buffer = ReplayMemoryWrapperBase(cfg.RL_MEMORY_SIZE, cfg, "online")
        rollout_agent = AgentWrapper(
            args,
            cfg,
            pretrained_path=args.pretrained_dir,
            input_dim=cfg.RL_TRAIN.feature_input_dim,
            add_expert=add_expert,
        )
        actor_seed = args.seed
        if args.use_grasp_predictor:
            grasp_agent = AgentWrapper(
                grasp_args,
                grasp_cfg,
                pretrained_path=grasp_pretrained_dir,
                input_dim=cfg.RL_TRAIN.feature_input_dim,
                model_surfix=grasp_model_surfix,
            )
        actor = ActorWrapper(
            cfg.RL_TRAIN.HANDOVER_SIM2REAL.stage,
            cfg,
            args.use_ray,
            rollout_agent,
            expert_buffer,
            online_buffer,
            actor_seed,
            grasp_agent,
            grasp_pred_threshold,
        )
        learner_agent = AgentWrapper(
            args, cfg, pretrained_path=args.pretrained_dir, input_dim=cfg.RL_TRAIN.feature_input_dim
        )
        trainer = Trainer(args, cfg, learner_agent, expert_buffer, online_buffer, logdir=output_dir)
        num_episodes = cfg.RL_TRAIN.num_remotes
        weights = learner_agent.get_weight()

    for train_iter in itertools.count(start=1):
        print("train iter: {:05d}".format(train_iter))

        if args.use_ray:
            incr_update_step = ray.get(learner_agent.get_agent_incr_update_step.remote())
        else:
            incr_update_step = learner_agent.get_agent_incr_update_step()
        milestone_idx = (incr_update_step > np.array(cfg.RL_TRAIN.mix_milestones)).sum().item()
        explore_ratio = min(
            get_valid_index(cfg.RL_TRAIN.explore_ratio_list, milestone_idx),
            cfg.RL_TRAIN.explore_cap,
        )
        explore = np.random.uniform() < explore_ratio
        if cfg.RL_TRAIN.HANDOVER_SIM2REAL.stage == "pretrain":
            if explore:
                test = np.random.uniform() < 0.1
            else:
                test = False
        if cfg.RL_TRAIN.HANDOVER_SIM2REAL.stage == "finetune":
            test = False
        noise_scale = cfg.RL_TRAIN.action_noise * get_valid_index(
            cfg.RL_TRAIN.noise_ratio_list, milestone_idx
        )

        if args.use_ray:
            refs = []
            refs.extend(
                [actor.rollout.remote(num_episodes, explore, test, noise_scale) for actor in actors]
            )
            refs.extend([trainer.train_iter.remote()])
            refs.extend([rollout_agent.load_weight.remote(weights)])
            refs.extend([learner_agent.get_weight.remote()])
            res = ray.get(refs)
            weights = res[-1]
        else:
            rollout_agent.load_weight(weights)
            actor.rollout(num_episodes, explore, test, noise_scale)
            trainer.train_iter()
            weights = learner_agent.get_weight()

        if args.use_ray:
            expert_reward_info = np.array(ray.get(expert_buffer.reward_info.remote()))
            online_reward_info = np.array(ray.get(online_buffer.reward_info.remote()))
        else:
            expert_reward_info = np.array(expert_buffer.reward_info())
            online_reward_info = np.array(online_buffer.reward_info())
        buffer_log = [*zip(expert_reward_info, online_reward_info)]
        if args.use_ray:
            trainer.write_buffer_info.remote(buffer_log)
        else:
            trainer.write_buffer_info(buffer_log)

        if args.use_ray:
            update_step = ray.get(learner_agent.get_agent_update_step.remote())
        else:
            update_step = learner_agent.get_agent_update_step()
        if update_step > cfg.RL_TRAIN.max_epoch:
            if args.use_ray:
                trainer.save_latest_model.remote(update_step)
            else:
                trainer.save_latest_model(update_step)
            break


if __name__ == "__main__":
    main()
