import gymnasium as gym

from . import agents
from .flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY
from .flat_newton_env_cfg import G1FlatNewtonEnvCfg, G1FlatNewtonEnvCfg_PLAY

gym.register(
    id="Instinct-Locomotion-Flat-G1-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Flat-G1-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Flat-G1-Newton-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatNewtonEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Locomotion-Flat-G1-Newton-Play-v0",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatNewtonEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)
