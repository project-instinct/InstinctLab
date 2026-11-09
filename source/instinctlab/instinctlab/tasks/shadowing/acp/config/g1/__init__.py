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

register_backend_tasks(
    "Instinct-ACP-Planner-G1",
    env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPPlannerPPORunnerCfg",
)

register_backend_tasks(
    "Instinct-ACP-Planner-SequentialHandoff-G1",
    env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerSequentialHandoffEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPPlannerPPORunnerCfg",
)

for variant, cfg_name in (
    ("RefLabels", "G1ACPPlannerReferenceLabelsEnvCfg"),
    ("LateHandoff", "G1ACPPlannerLateHandoffEnvCfg"),
    ("RefLabels-LateHandoff", "G1ACPPlannerReferenceLabelsLateHandoffEnvCfg"),
):
    register_backend_tasks(
        f"Instinct-ACP-Planner-{variant}-G1",
        env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:{cfg_name}",
        play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
        entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
        instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPPlannerPPORunnerCfg",
    )
