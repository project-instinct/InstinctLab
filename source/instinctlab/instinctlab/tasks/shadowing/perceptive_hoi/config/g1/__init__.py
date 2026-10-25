from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.shadowing.perceptive_hoi.config.g1"

register_backend_tasks(
    "Instinct-Perceptive-HOI-Shadowing-G1",
    env_cfg_entry_point=f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveHoiShadowingEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveHoiShadowingEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    rsl_rl_cfg_entry_point=f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveHoiShadowingPPORunnerCfg",
)
