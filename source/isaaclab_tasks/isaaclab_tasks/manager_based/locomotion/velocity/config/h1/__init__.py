# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
H1机器人的环境注册模块

该模块注册了H1机器人的各种Gym环境，包括粗糙地形和平坦地形的训练和演示环境。
"""

import gymnasium as gym

from . import agents

##
# 注册Gym环境
##

# 注册H1机器人在粗糙地形上的训练环境
gym.register(
    id="Isaac-Velocity-Rough-H1-v0",  # 环境ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 环境入口点
    disable_env_checker=True,  # 禁用环境检查器
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H1RoughEnvCfg",  # 环境配置入口点
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",  # RSL-RL配置入口点
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",  # SKRL配置入口点
    },
)


# 注册H1机器人在粗糙地形上的演示环境
gym.register(
    id="Isaac-Velocity-Rough-H1-Play-v0",  # 环境ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 环境入口点
    disable_env_checker=True,  # 禁用环境检查器
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H1RoughEnvCfg_PLAY",  # 演示环境配置入口点
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",  # RSL-RL配置入口点
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",  # SKRL配置入口点
    },
)


# 注册H1机器人在平坦地形上的训练环境
gym.register(
    id="Isaac-Velocity-Flat-H1-v0",  # 环境ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 环境入口点
    disable_env_checker=True,  # 禁用环境检查器
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1FlatEnvCfg",  # 环境配置入口点
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",  # RSL-RL配置入口点
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",  # SKRL配置入口点
    },
)


# 注册H1机器人在平坦地形上的演示环境
gym.register(
    id="Isaac-Velocity-Flat-H1-Play-v0",  # 环境ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 环境入口点
    disable_env_checker=True,  # 禁用环境检查器
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1FlatEnvCfg_PLAY",  # 演示环境配置入口点
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",  # RSL-RL配置入口点
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",  # SKRL配置入口点
    },
)

