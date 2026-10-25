# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 robot in BeyondMimic task."""

from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.shadowing.beyondmimic.config.g1"

register_backend_tasks(
    "Instinct-BeyondMimic-Plane-G1",
    env_cfg_entry_point=f"{task_entry}.beyondmimic_plane_cfg:G1BeyondMimicPlaneEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.beyondmimic_plane_cfg:G1BeyondMimicPlaneEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    rsl_rl_cfg_entry_point=f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.beyondmimic_ppo_cfg:G1BeyondMimicPPORunnerCfg",
)
