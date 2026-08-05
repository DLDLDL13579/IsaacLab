# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""X30 locomotion velocity task."""

from isaaclab.envs import manager_based
from isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.config.x30 import (
    X30FlatEnvCfg,
    X30FlatEnvCfg_PLAY,
    X30RoughEnvCfg,
    X30RoughEnvCfg_PLAY,
)


##
# Register Gym environments.
##

manager_based.register_env(
    "Isaac-Velocity-Flat-X30-v0",
    X30FlatEnvCfg,
    X30FlatEnvCfg_PLAY,
)

manager_based.register_env(
    "Isaac-Velocity-Rough-X30-v0",
    X30RoughEnvCfg,
    X30RoughEnvCfg_PLAY,
)
