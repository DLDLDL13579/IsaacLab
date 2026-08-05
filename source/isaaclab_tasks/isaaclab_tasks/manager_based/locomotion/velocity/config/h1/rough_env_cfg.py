# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
H1机器人的粗糙地形环境配置

该文件定义了H1机器人在粗糙地形上的环境配置，包括奖励函数、场景设置、随机化参数等。
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# 预定义配置
##
from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip


@configclass
class H1Rewards(RewardsCfg):
    """H1机器人的奖励函数配置"""

    # 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # 禁用Z轴线性速度惩罚
    lin_vel_z_l2 = None
    # XY平面线性速度跟踪奖励（指数）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # Z轴角速度跟踪奖励（指数）
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # 脚部空中时间奖励（双足机器人）
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    # 脚部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    # 踝关节限制惩罚
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    # 非运动关键关节偏离默认位置惩罚
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    # 手臂关节偏离默认位置惩罚
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    # 躯干关节偏离默认位置惩罚
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )


@configclass
class H1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """H1机器人在粗糙地形上的环境配置"""
    
    # 奖励函数配置
    rewards: H1Rewards = H1Rewards()

    def __post_init__(self):
        """初始化后处理方法"""
        # 调用父类的初始化后处理
        super().__post_init__()
        # 场景设置
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 设置高度扫描器位置
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # 随机化设置
        # 禁用机器人推动
        self.events.push_robot = None
        # 禁用基础质量添加
        self.events.add_base_mass = None
        # 设置关节复位位置范围
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # 设置外部力 torque 作用点
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        # 设置基础复位参数
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # 禁用基础质心随机化
        self.events.base_com = None

        # 奖励设置
        # 禁用非期望接触惩罚
        self.rewards.undesired_contacts = None
        # 设置平面朝向惩罚权重
        self.rewards.flat_orientation_l2.weight = -1.0
        # 禁用关节力矩惩罚
        self.rewards.dof_torques_l2.weight = 0.0
        # 设置动作速率惩罚权重
        self.rewards.action_rate_l2.weight = -0.005
        # 设置关节加速度惩罚权重
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # 命令设置
        # X轴线性速度范围
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # Y轴线性速度范围（禁用）
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # Z轴角速度范围
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 终止条件设置
        # 设置基础接触检测的身体部位
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"


@configclass
class H1RoughEnvCfg_PLAY(H1RoughEnvCfg):
    """H1机器人在粗糙地形上的演示环境配置"""
    
    def __post_init__(self):
        """初始化后处理方法"""
        # 调用父类的初始化后处理
        super().__post_init__()

        # 为演示创建较小的场景
        self.scene.num_envs = 50
        # 设置环境间距
        self.scene.env_spacing = 2.5
        # 设置演示时长
        self.episode_length_s = 40.0
        # 在网格中随机生成机器人（而不是按照地形级别）
        self.scene.terrain.max_init_terrain_level = None
        # 减少地形数量以节省内存
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # 设置固定的X轴线性速度
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # Y轴线性速度范围（禁用）
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # Z轴角速度范围
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # 禁用航向随机化
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # 禁用演示时的观测噪声
        self.observations.policy.enable_corruption = False
        # 移除随机推力
        self.events.base_external_force_torque = None
        self.events.push_robot = None
