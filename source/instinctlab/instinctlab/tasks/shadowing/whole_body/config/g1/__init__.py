from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.shadowing.whole_body.config.g1"

register_backend_tasks(
    "Instinct-Shadowing-WholeBody-Plane-G1",
    env_cfg_entry_point=f"{task_entry}.plane_shadowing_cfg:G1PlaneShadowingEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.plane_shadowing_cfg:G1PlaneShadowingEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    rsl_rl_cfg_entry_point=f"{agents.__name__}.rsl_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
)
