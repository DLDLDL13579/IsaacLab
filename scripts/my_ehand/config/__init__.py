import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .ehand_grasp_env_cfg import EHandGraspEnvCfg

def rsl_rl_ppo_cfg():
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,           
        # 【修改】：将训练量大幅提升至 10000 次，确保策略充分收敛
        max_iterations=5000,
        save_interval=500,  # 每 00 轮保存一次模型，避免硬盘占用过大
        experiment_name="ehand_reorient",  
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

gym.register(
    id="Isaac-eHand-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": EHandGraspEnvCfg, "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg},
)