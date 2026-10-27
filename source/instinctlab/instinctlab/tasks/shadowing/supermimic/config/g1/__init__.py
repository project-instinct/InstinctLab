"""Configuration for G1 robot in the SuperMimic task."""

from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.shadowing.supermimic.config.g1"

register_backend_tasks(
    "Instinct-SuperMimic-Plane-G1",
    env_cfg_entry_point=f"{task_entry}.supermimic_plane_cfg:G1SuperMimicEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.supermimic_plane_cfg:G1SuperMimicEnvCfg",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.supermimic_agent_cfg:G1SuperMimicRunnerCfg",
)
