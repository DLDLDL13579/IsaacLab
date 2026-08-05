# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
学习环境终止条件函数

该模块包含可用于激活某些终止条件的通用函数。
这些函数可以传递给 :class:`isaaclab.managers.TerminationTermCfg` 对象，以启用函数引入的终止条件。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """
    当演员移动到地形边缘太近时终止

    如果演员移动到地形边缘太近，终止条件被激活。到地形边缘的距离
    是根据地形大小和距离缓冲计算的。
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # 平面是无限地形
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # 获取子地形的大小
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size  # 网格宽度和长度
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols  # 行数和列数
        border_width = terrain_gen_cfg.border_width  # 边界宽度
        # 计算地图大小
        map_width = n_rows * grid_width + 2 * border_width  # 地图宽度
        map_height = n_cols * grid_length + 2 * border_width  # 地图高度

        # 提取使用的量（启用类型提示）
        asset: RigidObject = env.scene[asset_cfg.name]  # 获取机器人资产

        # 检查代理是否越界
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer  # X轴越界
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer  # Y轴越界
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)  # 返回是否越界
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")  # 不支持的地形类型
