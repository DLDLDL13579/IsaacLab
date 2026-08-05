# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
使用RSL-RL训练强化学习代理的脚本

该脚本用于训练RSL-RL强化学习代理，支持视频录制、多GPU训练和策略导出。
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
parser.add_argument("--video_interval", type=int, default=2000, help="视频录制之间的间隔（以步数为单位）。")
parser.add_argument("--num_envs", type=int, default=None, help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务的名称。")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="强化学习代理配置入口点的名称。"
)
parser.add_argument("--seed", type=int, default=None, help="用于环境的种子")
parser.add_argument("--max_iterations", type=int, default=None, help="强化学习策略训练迭代次数。")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="使用多个GPU或节点运行训练。"
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="导出IO描述符。")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="由Ray集成自动配置，否则为None。"
)
# 添加RSL-RL命令行参数
cli_args.add_rsl_rl_args(parser)
# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录制视频时始终启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 为Hydra清除sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""检查最小支持的RSL-RL版本。"""

import importlib.metadata as metadata
import platform

from packaging import version

# 检查最小支持的rsl-rl版本
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"请安装正确版本的RSL-RL。\n现有版本是: '{installed_version}'"
        f" 所需版本是: '{RSL_RL_VERSION}'。\n要安装正确版本，请运行:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""接下来的所有内容"""

import logging
import os
import time
from datetime import datetime

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import sys
sys.path.append("/workspace/isaaclab/scripts")
import my_ehand.config
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入日志记录器
logger = logging.getLogger(__name__)

# PLACEHOLDER: 扩展模板（不要删除此注释）

# 设置PyTorch后端配置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """
    使用RSL-RL代理训练
    
    参数:
        env_cfg: 环境配置
        agent_cfg: RSL-RL代理配置
    """
    # 使用非Hydra命令行参数覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境种子
    # 注意：某些随机化发生在环境初始化中，因此我们在此处设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # 检查CPU设备与分布式训练的无效组合
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "使用CPU设备时不支持分布式训练。 "
            "请使用GPU设备（例如，--device cuda）进行分布式训练。"
        )

    # 多GPU训练配置
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 设置种子以在不同线程中具有多样性
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 指定用于记录实验的目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 在目录中记录实验: {log_root_path}")
    # 指定用于记录运行的目录: {时间戳}_{运行名称}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Ray Tune工作流使用下面的日志行提取实验名称，因此不要更改它（请参见PR #2346，comment-2819298849）
    print(f"从命令行请求的确切实验名称: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 如果请求，设置IO描述符导出标志
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO描述符仅支持基于管理器的强化学习环境。不会导出IO描述符。"
        )

    # 为环境设置日志目录（适用于所有环境类型）
    env_cfg.log_dir = log_dir

    # 创建isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果RL算法需要，转换为单代理实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 在创建新的log_dir之前保存恢复路径
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 包装用于视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 在训练期间录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # 为rsl-rl包装环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 从rsl-rl创建运行器
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"不支持的运行器类: {agent_cfg.class_name}")
    # 将git状态写入日志
    runner.add_git_repo_to_log(__file__)
    # 加载检查点
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: 从以下位置加载模型检查点: {resume_path}")
        # 加载先前训练的模型
        runner.load(resume_path)

    # 将配置转储到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 运行训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"训练时间: {round(time.time() - start_time, 2)} 秒")

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
