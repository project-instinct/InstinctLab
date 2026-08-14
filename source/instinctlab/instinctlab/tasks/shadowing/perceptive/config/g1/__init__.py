import gymnasium as gym

from . import agents
from .perceptive_shadowing_newton_cfg import G1PerceptiveNewtonEnvCfg, G1PerceptiveNewtonEnvCfg_PLAY
from .perceptive_vae_newton_cfg import G1PerceptiveVaeNewtonEnvCfg, G1PerceptiveVaeNewtonEnvCfg_PLAY

task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"

gym.register(
    id="Instinct-Perceptive-Shadowing-G1-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Vae-G1-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_cfg:G1PerceptiveVaeEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Vae-G1-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_cfg:G1PerceptiveVaeEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Newton-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_newton_cfg:G1PerceptiveNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Newton-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_newton_cfg:G1PerceptiveNewtonEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Vae-G1-Newton-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_newton_cfg:G1PerceptiveVaeNewtonEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Perceptive-Vae-G1-Newton-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_newton_cfg:G1PerceptiveVaeNewtonEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg",
    },
)
