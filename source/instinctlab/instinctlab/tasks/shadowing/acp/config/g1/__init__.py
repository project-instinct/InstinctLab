from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.shadowing.acp.config.g1"
agent_entry = f"{agents.__name__}.instinct_rl_ppo_cfg"

register_backend_tasks(
    "Instinct-ACP-Tracker-G1",
    env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agent_entry}:G1ACPTrackerPPORunnerCfg",
)

# Experiment variants: AME-2 depth encoder, BeyondMimic start sampling instead of CSI, the paper's action scale.
register_backend_tasks(
    "Instinct-ACP-Tracker-AME2-G1",
    env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agent_entry}:G1ACPTrackerAME2PPORunnerCfg",
)
register_backend_tasks(
    "Instinct-ACP-Tracker-BeyondSampling-G1",
    env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerBeyondSamplingEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agent_entry}:G1ACPTrackerPPORunnerCfg",
)
register_backend_tasks(
    "Instinct-ACP-Tracker-BeyondSampling-AME2-G1",
    env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerBeyondSamplingEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agent_entry}:G1ACPTrackerAME2PPORunnerCfg",
)
register_backend_tasks(
    "Instinct-ACP-Tracker-ActionScale025-G1",
    env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerPaperActionScaleEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerPaperActionScaleEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agent_entry}:G1ACPTrackerPPORunnerCfg",
)
