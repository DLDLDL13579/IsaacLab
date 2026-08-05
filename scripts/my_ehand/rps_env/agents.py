from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

rsl_rl_ppo_cfg = RslRlOnPolicyRunnerCfg(
    seed=42,
    empirical_normalization=False,
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],  # 加深网络容量以应对域随机化
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,  # 降低学习率，提升抗噪稳定性
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    ),
    num_steps_per_env=24,
    max_iterations=5000,  # 完整的 5000 次大规模迭代训练周期
    save_interval=100,
    experiment_name="ehand_rps_sim2real",
    run_name="v_final",
)