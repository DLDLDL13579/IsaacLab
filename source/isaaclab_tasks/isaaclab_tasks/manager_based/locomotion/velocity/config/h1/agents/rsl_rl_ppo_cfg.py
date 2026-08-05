# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
H1机器人的RSL-RL PPO算法配置

该文件定义了H1机器人使用RSL-RL库的PPO算法配置，包括粗糙地形和平坦地形的不同配置。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class H1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """H1机器人在粗糙地形上的PPO运行器配置"""
    
    # 每个环境的步数
    num_steps_per_env = 24
    # 最大训练迭代次数
    max_iterations = 3000
    # 保存间隔
    save_interval = 50
    # 实验名称
    experiment_name = "h1_rough"
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始化噪声标准差
        actor_obs_normalization=False,  # 禁用演员网络观测归一化
        critic_obs_normalization=False,  # 禁用评论家网络观测归一化
        actor_hidden_dims=[512, 256, 128],  # 演员网络隐藏层维度
        critic_hidden_dims=[512, 256, 128],  # 评论家网络隐藏层维度
        activation="elu",  # 激活函数
    )
    # PPO算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 价值损失系数
        use_clipped_value_loss=True,  # 使用裁剪价值损失
        clip_param=0.2,  # 裁剪参数
        entropy_coef=0.01,  # 熵系数
        num_learning_epochs=5,  # 学习轮数
        num_mini_batches=4,  # 小批量数量
        learning_rate=1.0e-3,  # 学习率
        schedule="adaptive",  # 学习率调度
        gamma=0.99,  # 折扣因子
        lam=0.95,  # 广义优势估计参数
        desired_kl=0.01,  # 期望KL散度
        max_grad_norm=1.0,  # 梯度裁剪
    )


@configclass
class H1FlatPPORunnerCfg(H1RoughPPORunnerCfg):
    """H1机器人在平坦地形上的PPO运行器配置"""
    
    def __post_init__(self):
        """初始化后处理方法"""
        # 调用父类的初始化后处理
        super().__post_init__()

        # 减少最大训练迭代次数
        self.max_iterations = 1000
        # 设置实验名称
        self.experiment_name = "h1_flat"
        # 简化网络结构以适应平坦地形
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

