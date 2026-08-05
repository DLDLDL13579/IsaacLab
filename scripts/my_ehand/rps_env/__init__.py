import gymnasium as gym
from . import env, agents

# 将你的猜拳环境注册到 Gymnasium 的全局环境变量库中
gym.register(
    id="Isaac-EHand-RPS-Direct-v0",
    entry_point="rps_env.env:RPSHandEnv",  # <--- 【核心修复】：指向你自定义的环境类
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:RPSHandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:rsl_rl_ppo_cfg",
    },
)