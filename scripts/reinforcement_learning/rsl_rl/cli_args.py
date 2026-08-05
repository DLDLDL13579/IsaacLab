# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL命令行参数处理模块

该模块提供了用于处理RSL-RL强化学习框架的命令行参数的函数，包括添加参数、解析配置和更新配置。
"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """
    向解析器添加RSL-RL参数

    参数:
        parser: 要添加参数的解析器
    """
    # 创建一个新的参数组
    arg_group = parser.add_argument_group("rsl_rl", description="RSL-RL代理的参数")
    # -- 实验参数
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="存储日志的实验文件夹名称"
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="日志目录的运行名称后缀")
    # -- 加载参数
    arg_group.add_argument("--resume", action="store_true", default=False, help="是否从检查点恢复")
    arg_group.add_argument("--load_run", type=str, default=None, help="要恢复的运行文件夹名称")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="要恢复的检查点文件")
    # -- 日志记录器参数
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="要使用的日志记录器模块"
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="使用wandb或neptune时的日志项目名称"
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """
    根据输入解析RSL-RL代理的配置

    参数:
        task_name: 环境的名称
        args_cli: 命令行参数

    返回:
        根据输入解析的RSL-RL代理配置
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # 加载默认配置
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """
    根据输入更新RSL-RL代理的配置

    参数:
        agent_cfg: RSL-RL代理的配置
        args_cli: 命令行参数

    返回:
        根据输入更新的RSL-RL代理配置
    """
    # 使用CLI参数覆盖默认配置
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # 如果seed = -1，则随机采样一个种子
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # 为wandb和neptune设置项目名称
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
