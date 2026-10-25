# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.parkour.config.g1"

register_backend_tasks(
    "Instinct-Parkour-Target-Amp-G1",
    env_cfg_entry_point=f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
)
