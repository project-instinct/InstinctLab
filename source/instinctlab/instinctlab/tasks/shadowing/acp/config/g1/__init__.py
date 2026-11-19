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
    "Instinct-ACP-Tracker-FootDropout-G1",
    env_cfg_entry_point=f"{task_entry}.acp_tracker_cfg:G1ACPTrackerFootDropoutEnvCfg",
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
    ("AdaptiveStart", "G1ACPPlannerAdaptiveStartEnvCfg"),
    ("RefLabels-AdaptiveStart", "G1ACPPlannerReferenceLabelsAdaptiveStartEnvCfg"),
    ("PartialHandoff", "G1ACPPlannerPartialHandoffEnvCfg"),
    ("LabelAll", "G1ACPPlannerLabelAllEnvCfg"),
):
    register_backend_tasks(
        f"Instinct-ACP-Planner-{variant}-G1",
        env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:{cfg_name}",
        play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
        entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
        instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPPlannerPPORunnerCfg",
    )

for variant, cfg_name in (
    ("BigReplay", "G1ACPPlannerEnvCfg"),
    ("PartialHandoff-BigReplay", "G1ACPPlannerPartialHandoffEnvCfg"),
):
    register_backend_tasks(
        f"Instinct-ACP-Planner-{variant}-G1",
        env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:{cfg_name}",
        play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
        entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
        instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPBigReplayPlannerPPORunnerCfg",
    )

register_backend_tasks(
    "Instinct-ACP-Planner-V2-G1",
    env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerV2EnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPPlannerV2PPORunnerCfg",
)

register_backend_tasks(
    "Instinct-ACP-Planner-V3-G1",
    env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerV3EnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPPlannerV2PPORunnerCfg",
)

for variant, env_cfg_name, runner_cfg_name in (
    ("V3-Occ5", "G1ACPPlannerV2EnvCfg", "G1ACPPlannerV3Occ5PPORunnerCfg"),
    ("V3-NoAct", "G1ACPPlannerV2EnvCfg", "G1ACPPlannerV3NoActPPORunnerCfg"),
    ("V3-NoAct-Occ5", "G1ACPPlannerV2EnvCfg", "G1ACPPlannerV3NoActOcc5PPORunnerCfg"),
    ("V3-RefLabels", "G1ACPPlannerV3RefLabelsEnvCfg", "G1ACPPlannerV2PPORunnerCfg"),
):
    register_backend_tasks(
        f"Instinct-ACP-Planner-{variant}-G1",
        env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:{env_cfg_name}",
        play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
        entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
        instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:{runner_cfg_name}",
    )

register_backend_tasks(
    "Instinct-ACP-Planner-Depth-G1",
    env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerDepthEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.acp_planner_cfg:G1ACPPlannerEnvCfg_PLAY",
    entry_point="instinctlab.tasks.shadowing.acp.labeling_env:HindsightLabelingEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ACPDepthPlannerPPORunnerCfg",
)
