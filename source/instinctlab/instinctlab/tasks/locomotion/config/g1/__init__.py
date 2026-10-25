from instinctlab.tasks.utils.backends import register_backend_tasks

from . import agents

task_entry = "instinctlab.tasks.locomotion.config.g1"

register_backend_tasks(
    "Instinct-Locomotion-Flat-G1",
    env_cfg_entry_point=f"{task_entry}.flat_env_cfg:G1FlatEnvCfg",
    play_env_cfg_entry_point=f"{task_entry}.flat_env_cfg:G1FlatEnvCfg_PLAY",
    entry_point="instinctlab.envs.manager_based_rl_env:InstinctRlEnv",
    instinct_rl_cfg_entry_point=f"{agents.__name__}.instinct_rl_ppo_cfg:G1FlatPPORunnerCfg",
)
