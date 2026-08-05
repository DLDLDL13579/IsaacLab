# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
学习环境奖励函数

该模块包含可用于定义学习环境奖励的通用函数。
这些函数可以传递给 :class:`isaaclab.managers.RewardTermCfg` 对象，以指定奖励函数及其参数。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    使用L2核奖励脚部迈出的长步

    此函数奖励代理迈出超过阈值的步数。这有助于确保机器人抬起脚并迈步。
    奖励计算为脚在空中的时间总和。

    如果命令很小（即代理不应该迈步），则奖励为零。
    """
    # 提取使用的量（启用类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 获取接触传感器
    # 计算奖励
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # 第一次接触
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # 上次空中时间
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)  # 计算奖励
    # 零命令无奖励
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    为双足机器人奖励脚部迈出的长步

    此函数奖励代理迈出达到指定阈值的步数，并且每次保持一只脚在空中。

    如果命令很小（即代理不应该迈步），则奖励为零。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 获取接触传感器
    # 计算奖励
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # 当前空中时间
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # 当前接触时间
    in_contact = contact_time > 0.0  # 是否接触
    in_mode_time = torch.where(in_contact, contact_time, air_time)  # 当前模式时间
    single_stance = torch.sum(in_contact.int(), dim=1) == 1  # 单脚站立
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]  # 计算奖励
    reward = torch.clamp(reward, max=threshold)  # 限制最大奖励
    # 零命令无奖励
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚脚滑动

    此函数惩罚代理在地面上滑动脚。奖励计算为脚的线性速度范数乘以二进制接触传感器。
    这确保只有当脚与地面接触时，代理才会受到惩罚。
    """
    # 惩罚脚滑动
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # 获取接触传感器
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0  # 接触状态
    asset = env.scene[asset_cfg.name]  # 获取机器人资产

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # 脚的线性速度
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)  # 计算奖励
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    在重力对齐的机器人坐标系中使用指数核奖励线性速度命令（xy轴）跟踪
    """
    # 提取使用的量（启用类型提示）
    asset = env.scene[asset_cfg.name]  # 获取机器人资产
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])  # 偏航坐标系中的速度
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )  # 线性速度误差
    return torch.exp(-lin_vel_error / std**2)  # 指数奖励


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    在世界坐标系中使用指数核奖励角速度命令（偏航）跟踪
    """
    # 提取使用的量（启用类型提示）
    asset = env.scene[asset_cfg.name]  # 获取机器人资产
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])  # 角速度误差
    return torch.exp(-ang_vel_error / std**2)  # 指数奖励


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    当命令非常小时，惩罚与默认关节位置的偏差
    """
    command = env.command_manager.get_command(command_name)  # 获取命令
    # 当命令接近零时惩罚运动
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
