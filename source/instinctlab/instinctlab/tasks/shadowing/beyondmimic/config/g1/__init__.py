# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 robot in BeyondMimic task."""

import gymnasium as gym

from . import agents
from .beyondmimic_plane_newton_cfg import G1BeyondMimicNewtonEnvCfg, G1BeyondMimicNewtonEnvCfg_PLAY

task_entry = "instinctlab.tasks.shadowing.beyondmimic.config.g1"

gym.register(
    id="Instinct-BeyondMimic-Plane-G1-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.beyondmimic_plane_cfg:G1BeyondMimicPlaneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-BeyondMimic-Plane-G1-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.beyondmimic_plane_cfg:G1BeyondMimicPlaneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-BeyondMimic-Plane-G1-Newton-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.beyondmimic_plane_newton_cfg:G1BeyondMimicNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-BeyondMimic-Plane-G1-Newton-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.beyondmimic_plane_newton_cfg:G1BeyondMimicNewtonEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
    },
)
