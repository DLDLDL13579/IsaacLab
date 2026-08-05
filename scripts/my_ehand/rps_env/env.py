import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .robot import EHAND_CFG

@configclass
class RPSHandSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = EHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class RPSHandEnvCfg(DirectRLEnvCfg):
    action_space: int = 6         
    observation_space: int = 25   # 16(关节位姿) + 3(目标指令) + 6(历史动作补偿)
    state_space: int = 0
    episode_length_s: float = 4.0 
    decimation: int = 2         
    
    scene: RPSHandSceneCfg = RPSHandSceneCfg(
        num_envs=2048, 
        env_spacing=2.0,
        replicate_physics=True,
    )

class RPSHandEnv(DirectRLEnv):
    cfg: RPSHandEnvCfg

    def __init__(self, cfg: RPSHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.robot = self.scene["robot"]
        self.actuated_joint_indices, self.joint_names = self.robot.find_joints("joint_.*")
        
        # 机械连杆矩阵
        self.action_to_joint_matrix = torch.zeros(6, len(self.actuated_joint_indices), device=self.device)
        for idx, name in enumerate(self.joint_names):
            if "joint_1_1" in name: 
                self.action_to_joint_matrix[0, idx] = math.radians(70)
            elif "joint_2_1" in name: 
                self.action_to_joint_matrix[1, idx] = math.radians(32)
            elif "joint_2_2" in name:
                self.action_to_joint_matrix[1, idx] = math.radians(44)
            elif "joint_2_3" in name:
                self.action_to_joint_matrix[1, idx] = math.radians(44)
            else:
                motor_idx = 0
                if "joint_3_" in name: motor_idx = 2
                elif "joint_4_" in name: motor_idx = 3
                elif "joint_5_" in name: motor_idx = 4
                elif "joint_6_" in name: motor_idx = 5
                
                if name.endswith("_1"): self.action_to_joint_matrix[motor_idx, idx] = math.radians(60)
                elif name.endswith("_2"): self.action_to_joint_matrix[motor_idx, idx] = math.radians(110)
                elif name.endswith("_3"): self.action_to_joint_matrix[motor_idx, idx] = math.radians(170)
        
        self.target_gestures = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_one_hot = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.target_joint_pos = torch.zeros(self.num_envs, len(self.actuated_joint_indices), dtype=torch.float32, device=self.device)
        
        # 动作缓存，用于网络推断和延迟模拟
        self.last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.delayed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        pose_rock_6d = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device)  
        pose_paper_6d = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device) 
        pose_scissors_6d = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], device=self.device) 
        
        self.pose_rock = torch.matmul(pose_rock_6d, self.action_to_joint_matrix)
        self.pose_paper = torch.matmul(pose_paper_6d, self.action_to_joint_matrix)
        self.pose_scissors = torch.matmul(pose_scissors_6d, self.action_to_joint_matrix)

    def _setup_scene(self):
        super()._setup_scene()
        cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        cfg.func("/World/Light", cfg)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        self.last_actions[env_ids] = 0.0
        self.delayed_actions[env_ids] = 0.0
        
        new_gestures = torch.randint(0, 3, (len(env_ids),), device=self.device)
        self.target_gestures[env_ids] = new_gestures
        self.target_one_hot[env_ids] = torch.nn.functional.one_hot(new_gestures, num_classes=3).float()
        
        mask_rock = (new_gestures == 0)
        mask_paper = (new_gestures == 1)
        mask_scissors = (new_gestures == 2)
        
        self.target_joint_pos[env_ids[mask_rock]] = self.pose_rock
        self.target_joint_pos[env_ids[mask_paper]] = self.pose_paper
        self.target_joint_pos[env_ids[mask_scissors]] = self.pose_scissors

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # 延迟缓冲：下发上一帧的动作，将当前动作推入缓存
        action_to_apply = self.delayed_actions.clone()
        self.delayed_actions = self.actions.clone()
        
        activation = (action_to_apply + 1.0) / 2.0
        target_positions = torch.matmul(activation, self.action_to_joint_matrix)
        self.robot.set_joint_position_target(target_positions, joint_ids=self.actuated_joint_indices)

    def _get_observations(self) -> dict:
        current_joint_pos = self.robot.data.joint_pos[:, self.actuated_joint_indices]
        # 域随机化：添加高斯传感器噪声，增强鲁棒性
        noise = torch.randn_like(current_joint_pos) * 0.02
        noisy_joint_pos = current_joint_pos + noise
        
        # 拼接观测：带噪位姿 + 目标指令 + 历史动作
        obs = torch.cat([noisy_joint_pos, self.target_one_hot, self.last_actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        current_joint_pos = self.robot.data.joint_pos[:, self.actuated_joint_indices]
        
        # 极速追踪奖励：缩小容忍分母(0.1)，倒逼网络以极快的速度收敛到目标角度
        pose_error = torch.sum(torch.square(current_joint_pos - self.target_joint_pos), dim=-1)
        reward_pose = torch.exp(-pose_error / 0.1) 
        
        action_diff = torch.sum(torch.square(self.actions - self.last_actions), dim=-1)
        penalty_smoothness = -0.05 * action_diff  # 降低平滑惩罚，允许爆发性扭矩输出
        
        self.last_actions = self.actions.clone()
        
        return reward_pose + penalty_smoothness

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out