# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class X30RslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for X30 PPO."""

    # fmt: off
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-4
    schedule = "linear"
    gamma = 0.99
    lam = 0.95
    value_loss_coef = 1.0
    entropy_coef = 0.0
    clip_param = 0.2
    max_grad_norm = 1.0
    use_clipped_value_loss = True
    # fmt: on


@configclass
class X30RslRlRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for X30 runner."""

    experiment_name = "x30"
    algorithm: X30RslRlPpoAlgorithmCfg = X30RslRlPpoAlgorithmCfg()
    max_iterations = 10000
