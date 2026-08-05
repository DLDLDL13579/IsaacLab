from __future__ import annotations
import os
import sys
import torch
import math  # 新增 math 库用于角度转换

# 将 my_ehand 目录加入环境变量，确保导入正常
MY_EHAND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MY_EHAND_DIR not in sys.path:
    sys.path.append(MY_EHAND_DIR)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.envs.mdp as default_mdp
import mdp as custom_mdp
import isaaclab_tasks.manager_based.manipulation.inhand.mdp as inhand_mdp

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets.articulation import Articulation

URDF_FILE = os.path.join(MY_EHAND_DIR, "urdf", "eHand-6-R.urdf")
##
# 0. 完美适配参数表的机械连杆动作类
##
class EHandLinkageAction(ActionTerm):
    cfg: ActionTermCfg
    
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_name]
        self.joint_idx, self.joint_names = self.asset.find_joints("joint_.*")
        
        # 内部缓存变量 (带下划线)
        self._action_dim = 6
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, len(self.joint_idx)), device=self.device)
        
        self.mapped_actions = torch.zeros((self.num_envs, len(self.joint_idx)), device=self.device)
        self.joint_limits_high = torch.zeros(len(self.joint_idx), device=self.device)
        self.motor_to_joints = [[] for _ in range(6)]
        
        for idx, name in enumerate(self.joint_names):
            # 电机 1：拇指侧摆 (70°)
            if "joint_1_1" in name: 
                self.motor_to_joints[0].append(idx)
                self.joint_limits_high[idx] = math.radians(70) 
            # 电机 2：拇指弯曲联动 (32°, 44°, 44°)
            elif "joint_2_1" in name: 
                self.motor_to_joints[1].append(idx)
                self.joint_limits_high[idx] = math.radians(32)
            elif "joint_2_2" in name: 
                self.motor_to_joints[1].append(idx)
                self.joint_limits_high[idx] = math.radians(44)
            elif "joint_2_3" in name: 
                self.motor_to_joints[1].append(idx)
                self.joint_limits_high[idx] = math.radians(44)
            # 电机 3-6：四指弯曲联动 (60°, 110°, 170°)
            else:
                if "joint_3_" in name: self.motor_to_joints[2].append(idx)
                elif "joint_4_" in name: self.motor_to_joints[3].append(idx)
                elif "joint_5_" in name: self.motor_to_joints[4].append(idx)
                elif "joint_6_" in name: self.motor_to_joints[5].append(idx)
                
                if name.endswith("_1"): self.joint_limits_high[idx] = math.radians(60)
                elif name.endswith("_2"): self.joint_limits_high[idx] = math.radians(110)
                elif name.endswith("_3"): self.joint_limits_high[idx] = math.radians(170)

        self.joint_limits_low = torch.zeros_like(self.joint_limits_high)
        self.offset = (self.joint_limits_high + self.joint_limits_low) / 2.0
        self.scale = (self.joint_limits_high - self.joint_limits_low) / 2.0

    # ========== 核心修复：实现抽象属性 ==========
    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    # ============================================

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        for i in range(6):
            for j_idx in self.motor_to_joints[i]:
                self.mapped_actions[:, j_idx] = actions[:, i]
        # 修复：必须将结果赋值给内部变量，而不是 return
        self._processed_actions[:] = self.offset + self.scale * self.mapped_actions

    def apply_actions(self):
        # 修复：应用正确的内部变量到物理引擎
        self.asset.set_joint_position_target(self._processed_actions, joint_ids=self.joint_idx)
@configclass
class EHandLinkageActionCfg(ActionTermCfg):
    class_type: type = EHandLinkageAction
    asset_name: str = "robot"
##
# 1. 场景定义
##
@configclass
class EHandGraspSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_FILE,
            make_instanceable=False,
            fix_base=True,
            collider_type="convex_decomposition",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=10.0,
                    damping=1.0,
                )
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.7071, 0.0, 0.7071), 
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]_[1-3]"], 
                stiffness=10.0,  
                damping=1.0,   
                # 【完美匹配】10N 最大力矩上限
                effort_limit=10.0, 
            )
        }
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.SphereCfg(
            # 【修改 1】：将半径增加到 0.035 米（直径 7 厘米，类似网球）
            radius=0.035, 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,      
                solver_position_iteration_count=8,  
                solver_velocity_iteration_count=0,
            ),
            # 【修改 2】：体积变大了，依据 PLA 材料密度，质量需同步增加到 0.22 kg (220克)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.22), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)), 
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,  
                dynamic_friction=0.2,  
                restitution=0.5, 
            ),
        ),
        # 【修改 3】：球变大后，底盘高度必须抬升至 0.05，防止一出生就卡在手心网格里
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.07, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
    )

##
# 2. 指令生成
##
@configclass
class CommandsCfg:
    object_pose = inhand_mdp.InHandReOrientationCommandCfg(
        asset_name="object",
        update_goal_on_success=True, 
        orientation_success_threshold=0.1, 
        make_quat_unique=False,
    )

##
# 3. 动作空间配置 (挂载 6 自由度真实映射)
##
@configclass
class ActionsCfg:
    joint_pos = EHandLinkageActionCfg(asset_name="robot")

##
# 4. 观测空间
##
@configclass
class ObservationsCfg:
    @configclass
    class PolicyObsGroupCfg(ObsGroup):
        joint_pos = ObsTerm(func=default_mdp.joint_pos_rel, noise=Gnoise(std=0.01))
        joint_vel = ObsTerm(func=default_mdp.joint_vel_rel, noise=Gnoise(std=0.01))
        
        object_pos = ObsTerm(func=default_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(func=default_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False})
        object_lin_vel = ObsTerm(func=default_mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        
        goal_quat = ObsTerm(func=inhand_mdp.generated_commands, params={"command_name": "object_pose"})
        goal_quat_diff = ObsTerm(func=inhand_mdp.goal_quat_diff, params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False})

        last_action = ObsTerm(func=default_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyObsGroupCfg = PolicyObsGroupCfg()
##
# 5. 随机化重置
##
@configclass
class EventCfg:
    reset_object = EventTerm(
        func=default_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 【修改 4】：回合失败重置时，空投高度同样抬升至 0.045 ~ 0.055 之间
            "pose_range": {"x": [0.065, 0.075], "y": [-0.005, 0.005], "z": [0.045, 0.055]}, 
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_robot_joints = EventTerm(
        func=default_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
##
# 6. 奖励机制
##
@configclass
class RewardsCfg:
    survival_bonus = RewTerm(
        func=custom_mdp.object_survival,
        weight=2.0, 
        params={"object_cfg": SceneEntityCfg("object"), "min_height": -0.02} 
    )
    track_orientation = RewTerm(
        func=inhand_mdp.track_orientation_inv_l2,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "object_pose"}
    )
    success_bonus = RewTerm(
        func=inhand_mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"}
    )
    wrist_penalty = RewTerm(
        func=custom_mdp.wrist_zone_penalty,
        weight=-5.0,
        params={"object_cfg": SceneEntityCfg("object"), "wrist_x_threshold": 0.04}
    )
    action_l2 = RewTerm(func=default_mdp.action_l2, weight=-0.005)
    action_rate_l2 = RewTerm(func=default_mdp.action_rate_l2, weight=-0.01)

##
# 7. 结束条件
##
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=default_mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=custom_mdp.object_dropped, 
        params={"object_cfg": SceneEntityCfg("object"), "drop_height": -0.05}
    )
    max_consecutive_success = DoneTerm(
        func=inhand_mdp.max_consecutive_success, 
        params={"num_success": 20, "command_name": "object_pose"}
    )

@configclass
class EHandGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: EHandGraspSceneCfg = EHandGraspSceneCfg(num_envs=2048, env_spacing=0.8) 
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 15.0 
        self.sim.dt = 1.0 / 120.0
        self.viewer.eye = (1.5, 1.5, 1.5)