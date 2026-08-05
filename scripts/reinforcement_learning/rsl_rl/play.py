# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
使用RSL-RL强化学习代理的播放脚本

该脚本用于运行和可视化训练好的RSL-RL强化学习代理，支持视频录制和策略导出。
"""

"""首先启动Isaac Sim模拟器"""

import argparse
import sys

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip

# 添加命令行参数
parser = argparse.ArgumentParser(description="使用RSL-RL训练强化学习代理。")
parser.add_argument("--video", action="store_true", default=False, help="在训练期间录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（以步数为单位）。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用fabric并使用USD I/O操作。"
)
parser.add_argument("--num_envs", type=int, default=None, help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务的名称。")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="强化学习代理配置入口点的名称。"
)
parser.add_argument("--seed", type=int, default=None, help="用于环境的种子")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="使用来自Nucleus的预训练检查点。",
)
parser.add_argument("--real-time", action="store_true", default=False, help="如果可能，以实时方式运行。")
# 添加RSL-RL命令行参数
cli_args.add_rsl_rl_args(parser)
# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli, hydra_args = parser.parse_known_args()
# 录制视频时始终启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 为Hydra清除sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的所有内容"""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import sys
sys.path.append("/workspace/isaaclab/scripts")
import my_ehand.config
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: 扩展模板（不要删除此注释）


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """
    使用RSL-RL代理播放
    
    参数:
        env_cfg: 环境配置
        agent_cfg: RSL-RL代理配置
    """
    # 获取任务名称用于检查点路径
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # 使用非Hydra命令行参数覆盖配置
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # 设置环境种子
    # 注意：某些随机化发生在环境初始化中，因此我们在此处设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 指定用于记录实验的目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 从目录加载实验: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] 遗憾的是，此任务目前没有可用的预训练检查点。")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 为环境设置日志目录（适用于所有环境类型）
    env_cfg.log_dir = log_dir

    # 创建isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果RL算法需要，转换为单代理实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装用于视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 在训练期间录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 为rsl-rl包装环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: 从以下位置加载模型检查点: {resume_path}")
    # 加载先前训练的模型
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"不支持的运行器类: {agent_cfg.class_name}")
    runner.load(resume_path)

    # 获取训练好的策略用于推理
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 提取神经网络模块
    # 我们在try-except中执行此操作以保持向后兼容性
    try:
        # 2.3及以上版本
        policy_nn = runner.alg.policy
    except AttributeError:
        # 2.2及以下版本
        policy_nn = runner.alg.actor_critic

    # 提取归一化器
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # 导出策略为onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # 重置环境
    obs = env.get_observations()
    timestep = 0
    # 模拟环境
    while simulation_app.is_running():
        start_time = time.time()
        # 在推理模式下运行所有内容
        with torch.inference_mode():
            # 代理步进
            actions = policy(obs)
            # 环境步进
            obs, _, dones, _ = env.step(actions)
            # 重置已终止 episode 的循环状态
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # 录制一个视频后退出播放循环
            if timestep == args_cli.video_length:
                break

        # 实时评估的时间延迟
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
