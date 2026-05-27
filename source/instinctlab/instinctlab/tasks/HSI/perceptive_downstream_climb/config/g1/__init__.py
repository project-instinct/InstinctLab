import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.HSI.perceptive_downstream_climb.config.g1"

gym.register(
    id="Instinct-HSIDownstreamClimb-Perceptive-Shadowing-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-HSIDownstreamClimb-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-HSIDownstreamClimb-Perceptive-Vae-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_cfg:G1PerceptiveVaeEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg",
    },
)

gym.register(
    id="Instinct-HSIDownstreamClimb-Perceptive-Vae-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_cfg:G1PerceptiveVaeEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_cfg:G1PerceptiveVaePPORunnerCfg",
    },
)

gym.register(
    id="Instinct-HSIDownstreamClimb-Perceptive-Vae-Amp-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_amp_cfg:G1PerceptiveVaeAmpEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_amp_cfg:G1PerceptiveVaeAmpPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-HSIDownstreamClimb-Perceptive-Vae-Amp-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.perceptive_vae_amp_cfg:G1PerceptiveVaeAmpEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_vae_amp_cfg:G1PerceptiveVaeAmpPPORunnerCfg",
    },
)
