# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
学习环境课程学习函数

该模块包含可用于创建学习环境课程学习的通用函数。
这些函数可以传递给 :class:`isaaclab.managers.CurriculumTermCfg` 对象，以启用函数引入的课程学习。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    基于机器人在命令速度下行走距离的课程学习

    当机器人走得足够远时，该函数会增加地形难度；当机器人行走距离小于命令速度要求的一半时，会降低地形难度。

    .. note::
        只能在地形类型为 ``generator`` 时使用此函数。有关不同地形类型的更多信息，
        请查看 :class:`isaaclab.terrains.TerrainImporter` 类。

    返回:
        给定环境ID的平均地形级别。
    """
    # 提取使用的量（启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]  # 获取机器人资产
    terrain: TerrainImporter = env.scene.terrain  # 获取地形
    command = env.command_manager.get_command("base_velocity")  # 获取速度命令
    # 计算机器人行走的距离
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # 行走足够远的机器人进入更难的地形
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # 行走距离小于所需距离一半的机器人进入更简单的地形
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up  # 确保不与move_up重叠
    # 更新地形级别
    terrain.update_env_origins(env_ids, move_up, move_down)
    # 返回平均地形级别
    return torch.mean(terrain.terrain_levels.float())
