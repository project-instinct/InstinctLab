from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.shadowing.acp.config.g1"

register_backend_tasks(
    "Instinct-ACP-Tracker-G1",
    env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPTrackerPPORunnerCfg",
)
