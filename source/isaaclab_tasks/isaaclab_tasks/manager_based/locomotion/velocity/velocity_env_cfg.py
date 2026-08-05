# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
速度跟踪移动环境的基础配置

该文件定义了所有机器人共用的速度跟踪移动环境配置，包括场景设置、MDP参数、奖励函数等。
这些配置被各个机器人（如H1、A1、Anymal等）的具体配置继承和覆盖。
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# 预定义配置
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# 场景定义
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """带腿部机器人的地形场景配置"""

    # 地面地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形的Prim路径
        terrain_type="generator",  # 地形类型为生成器
        terrain_generator=ROUGH_TERRAINS_CFG,  # 使用预定义的粗糙地形配置
        max_init_terrain_level=5,  # 最大初始地形级别
        collision_group=-1,  # 碰撞组
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦组合模式
            restitution_combine_mode="multiply",  #  restitution组合模式
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,  # 启用UVW投影
            texture_scale=(0.25, 0.25),  # 纹理缩放
        ),
        debug_vis=False,  # 禁用调试可视化
    )
    # 机器人
    robot: ArticulationCfg = MISSING  # 机器人配置，由具体机器人实现填充
    # 传感器
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # 高度扫描器的Prim路径
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 扫描器偏移
        ray_alignment="yaw",  # 射线对齐方式
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 网格模式配置
        debug_vis=False,  # 禁用调试可视化
        mesh_prim_paths=["/World/ground"],  # 扫描的网格路径
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)  # 接触力传感器
    # 灯光
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 天空光的Prim路径
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 光强度
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",  # 纹理文件
        ),
    )


##
# MDP设置
##


@configclass
class CommandsCfg:
    """MDP的命令规格"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",  # 资产名称
        resampling_time_range=(10.0, 10.0),  # 重采样时间范围
        rel_standing_envs=0.02,  # 站立环境的比例
        rel_heading_envs=1.0,  # 航向环境的比例
        heading_command=True,  # 启用航向命令
        heading_control_stiffness=0.5,  # 航向控制刚度
        debug_vis=True,  # 启用调试可视化
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # X轴线性速度范围
            lin_vel_y=(-1.0, 1.0),  # Y轴线性速度范围
            ang_vel_z=(-1.0, 1.0),  # Z轴角速度范围
            heading=(-math.pi, math.pi)  # 航向范围
        ),
    )


@configclass
class ActionsCfg:
    """MDP的动作规格"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",  # 资产名称
        joint_names=[".*"],  # 关节名称（所有关节）
        scale=0.5,  # 动作缩放因子
        use_default_offset=True  # 使用默认偏移
    )


@configclass
class ObservationsCfg:
    """MDP的观测规格"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略组的观测"""

        # 观测项（保持顺序）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 基础线性速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 基础角速度
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 投影重力
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 速度命令
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 关节位置（相对）
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))  # 关节速度（相对）
        actions = ObsTerm(func=mdp.last_action)  # 上一个动作
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},  # 高度扫描
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            """初始化后处理"""
            self.enable_corruption = True  # 启用观测噪声
            self.concatenate_terms = True  # 连接观测项

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置"""

    # 启动事件
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",  # 启动模式
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 资产配置
            "static_friction_range": (0.8, 0.8),  # 静摩擦范围
            "dynamic_friction_range": (0.6, 0.6),  # 动摩擦范围
            "restitution_range": (0.0, 0.0),  #  restitution范围
            "num_buckets": 64,  # 桶数量
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",  # 启动模式
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  # 资产配置
            "mass_distribution_params": (-5.0, 5.0),  # 质量分布参数
            "operation": "add",  # 操作类型
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",  # 启动模式
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  # 资产配置
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},  # 质心范围
        },
    )

    # 重置事件
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",  # 重置模式
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  # 资产配置
            "force_range": (0.0, 0.0),  # 力范围
            "torque_range": (-0.0, 0.0),  # 力矩范围
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",  # 重置模式
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},  # 姿态范围
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },  # 速度范围
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",  # 重置模式
        params={
            "position_range": (0.5, 1.5),  # 位置范围
            "velocity_range": (0.0, 0.0),  # 速度范围
        },
    )

    # 间隔事件
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",  # 间隔模式
        interval_range_s=(10.0, 15.0),  # 间隔范围（秒）
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # 速度范围
    )


@configclass
class RewardsCfg:
    """MDP的奖励项"""

    # -- 任务奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )  # XY平面线性速度跟踪奖励（指数）
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )  # Z轴角速度跟踪奖励（指数）
    # -- 惩罚项
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  # Z轴线性速度惩罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # XY平面角速度惩罚
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)  # 关节力矩惩罚
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 关节加速度惩罚
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # 动作速率惩罚
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )  # 脚部空中时间奖励
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )  # 非期望接触惩罚
    # -- 可选惩罚项
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)  # 平面朝向惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)  # 关节位置限制惩罚


@configclass
class TerminationsCfg:
    """MDP的终止项"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超时终止
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )  # 基础接触终止


@configclass
class CurriculumCfg:
    """MDP的课程学习项"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)  # 地形级别课程学习


##
# 环境配置
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """速度跟踪移动环境的配置"""

    # 场景设置
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)  # 场景配置，4096个环境，环境间距2.5
    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()  # 观测配置
    actions: ActionsCfg = ActionsCfg()  # 动作配置
    commands: CommandsCfg = CommandsCfg()  # 命令配置
    # MDP设置
    rewards: RewardsCfg = RewardsCfg()  # 奖励配置
    terminations: TerminationsCfg = TerminationsCfg()  # 终止配置
    events: EventCfg = EventCfg()  # 事件配置
    curriculum: CurriculumCfg = CurriculumCfg()  # 课程学习配置

    def __post_init__(self):
        """初始化后处理"""
        # 一般设置
        self.decimation = 4  # 抽取因子
        self.episode_length_s = 20.0  #  episode长度（秒）
        # 仿真设置
        self.sim.dt = 0.005  # 仿真时间步长
        self.sim.render_interval = self.decimation  # 渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material  # 物理材料
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # GPU最大刚体补丁计数
        # 更新传感器更新周期
        # 所有传感器基于最小更新周期（物理更新周期）进行更新
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # 检查是否启用了地形级别课程学习 - 如果是，启用地形生成器的课程学习
        # 这会生成难度递增的地形，对训练很有用
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
