import gymnasium as gym

from . import agents
from .plane_shadowing_newton_cfg import G1WholeBodyNewtonEnvCfg, G1WholeBodyNewtonEnvCfg_PLAY

task_entry = "instinctlab.tasks.shadowing.whole_body.config.g1"

gym.register(
    id="Instinct-Shadowing-WholeBody-Plane-G1-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.plane_shadowing_cfg:G1PlaneShadowingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Shadowing-WholeBody-Plane-G1-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.plane_shadowing_cfg:G1PlaneShadowingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Shadowing-WholeBody-Plane-G1-Newton-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.plane_shadowing_newton_cfg:G1WholeBodyNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Shadowing-WholeBody-Plane-G1-Newton-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.plane_shadowing_newton_cfg:G1WholeBodyNewtonEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1ShadowingPPORunnerCfg",
    },
)
