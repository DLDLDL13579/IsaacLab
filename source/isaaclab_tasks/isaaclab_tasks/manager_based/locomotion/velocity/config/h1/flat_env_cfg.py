# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
H1机器人的平坦地形环境配置

该文件定义了H1机器人在平坦地形上的环境配置，继承自粗糙地形配置并进行了相应调整。
"""

from isaaclab.utils import configclass

from .rough_env_cfg import H1RoughEnvCfg


@configclass
class H1FlatEnvCfg(H1RoughEnvCfg):
    """H1机器人在平坦地形上的环境配置"""
    
    def __post_init__(self):
        """初始化后处理方法"""
        # 调用父类的初始化后处理
        super().__post_init__()

        # 将地形类型设置为平坦平面
        self.scene.terrain.terrain_type = "plane"
        # 禁用地形生成器
        self.scene.terrain.terrain_generator = None
        # 禁用高度扫描
        self.scene.height_scanner = None
        # 从观测中移除高度扫描数据
        self.observations.policy.height_scan = None
        # 禁用地形难度递进
        self.curriculum.terrain_levels = None
        # 调整脚部空中时间奖励权重
        self.rewards.feet_air_time.weight = 1.0
        # 设置脚部空中时间奖励阈值
        self.rewards.feet_air_time.params["threshold"] = 0.6


class H1FlatEnvCfg_PLAY(H1FlatEnvCfg):
    """H1机器人在平坦地形上的演示环境配置"""
    
    def __post_init__(self) -> None:
        """初始化后处理方法"""
        # 调用父类的初始化后处理
        super().__post_init__()

        # 为演示创建较小的场景
        self.scene.num_envs = 50
        # 设置环境间距
        self.scene.env_spacing = 2.5
        # 禁用演示时的观测噪声
        self.observations.policy.enable_corruption = False
        # 移除随机推力事件
        self.events.base_external_force_torque = None
        self.events.push_robot = None
