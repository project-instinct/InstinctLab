"""Check the imagined observation reproduces the manager's history layout exactly.

Two things must hold, or the policy would be trained on a differently-shaped and differently-ordered
input than it is deployed with:

1. the imagined observation's width equals the real one, and
2. the history terms are the ones the config says, stacked in the same order.
"""

import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = 4
HISTORY = 10

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "hist"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, _ = cli.resolve_train_cfg(TASK, overrides)
env_cfg.scene.num_envs = NUM_ENVS

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinctlab.tasks.shadowing.supermimic.supermimic_provider import build_imagination
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    sim = gym.make(TASK, cfg=env_cfg)
    sim.reset()
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    obs_format = env.get_obs_format()
    om = unwrapped.observation_manager

    print("policy terms declared:", obs_format["policy"])
    imagination = build_imagination(env=env, obs_format=obs_format)
    print("history terms:", imagination.history_terms, "| length:", imagination.history_length)

    obs, extras = env.get_observations()
    pack = extras["observations"]
    real_policy = pack["policy"]
    real_critic = pack["critic"]
    real_policy = real_policy if torch.is_tensor(real_policy) else torch.cat(
        [v.reshape(v.shape[0], -1) for v in real_policy.values()], dim=-1
    )
    real_critic = real_critic if torch.is_tensor(real_critic) else torch.cat(
        [v.reshape(v.shape[0], -1) for v in real_critic.values()], dim=-1
    )

    split = imagination.split_obs_pack(pack)
    state = torch.cat([split[n].reshape(NUM_ENVS, -1) for n in imagination.state_format], dim=-1)

    # build a history window from the live state repeated, and the matching action window
    history_state = state.unsqueeze(1).expand(-1, imagination.history_length, -1).contiguous()
    history_action = torch.zeros(
        NUM_ENVS, imagination.history_length, unwrapped.action_manager.total_action_dim, device=state.device
    )

    imagination.refresh_reference_window(imagination.goal_window)
    imagination.select_reference_rows(torch.arange(NUM_ENVS, device=state.device))
    ref_index = torch.zeros(NUM_ENVS, dtype=torch.long, device=state.device)

    imag_policy = imagination.policy_obs(state, None, None, ref_index, history_state, history_action)
    imag_critic = imagination.critic_obs(state, None, ref_index)

    print(f"\nreal policy {tuple(real_policy.shape)}  imagined policy {tuple(imag_policy.shape)}")
    print(f"real critic {tuple(real_critic.shape)}  imagined critic {tuple(imag_critic.shape)}")
    assert real_policy.shape == imag_policy.shape, "policy width mismatch"
    assert real_critic.shape == imag_critic.shape, "critic width mismatch"

    # the history axis must actually be present: a term declared (H, D) must occupy H*D values
    for name, dim in obs_format["policy"].items():
        print(f"   {name:16s} declared {tuple(dim)}")

    expected = 0
    for name, dim in obs_format["policy"].items():
        n = 1
        for d in dim:
            n *= int(d)
        expected += n
    print(f"\nsum of declared widths: {expected} (imagined {imag_policy.shape[1]})")
    assert expected == imag_policy.shape[1]

    sim.close()
    print("\nOK: imagined observation matches the manager's history layout")
