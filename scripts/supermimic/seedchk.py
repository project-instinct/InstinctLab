"""Does a rollout seeded from a replayed state see the goal that state was actually tracking?

Records, at one step, each environment's world-model state, its reference cursor and the goal terms
the environment put in the policy observation. Then it keeps stepping -- long enough for environments
to move on and to reset onto other clips -- and rebuilds the goal from the recorded state and cursor
the way the algorithm does for a replayed start. The two must agree.

For contrast it also rebuilds the goal from the environment's *current* reference position, which is
what the imagination did before: that is the mismatch this check exists to rule out.

Also checks that the imagined policy and critic observations have the environment's widths.
"""

import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = 16
RECORD_AT = 5
LATER = 60

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "seed"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, _ = cli.resolve_train_cfg(TASK, overrides)
env_cfg.scene.num_envs = NUM_ENVS

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import build_reference_window, unpack_state
    from instinctlab.tasks.shadowing.supermimic.supermimic_provider import build_imagination
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    sim = gym.make(TASK, cfg=env_cfg)
    sim.reset()
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    obs_format = env.get_obs_format()
    imagination = build_imagination(env=env, obs_format=obs_format)
    action_dim = unwrapped.action_manager.total_action_dim
    device = unwrapped.device
    motion_reference = unwrapped.scene["motion_reference"]

    def policy_slices():
        out, start = {}, 0
        for name, dim in obs_format["policy"].items():
            width = 1
            for d in dim:
                width *= int(d)
            out[name] = slice(start, start + width)
            start += width
        return out

    slices = policy_slices()
    goal_terms = ("joint_pos_ref", "joint_vel_ref", "position_ref", "rotation_ref")

    resets = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
    recorded = None
    for step in range(LATER):
        obs, rew, dones, extras = env.step(torch.zeros(NUM_ENVS, action_dim, device=device))
        resets += dones.reshape(-1).long()
        if step == RECORD_AT:
            pack = extras["observations"]
            split = imagination.split_obs_pack(pack)
            recorded = {
                "state": torch.cat([split[n].reshape(NUM_ENVS, -1) for n in imagination.state_format], -1).clone(),
                "cursor": pack["ref_cursor"].clone(),
                "goal": {t: pack["policy"][:, slices[t]].clone() for t in goal_terms},
                "policy_width": pack["policy"].shape[-1],
                "critic_width": pack["critic"].shape[-1],
            }
            resets.zero_()
    print(f"resets between recording and rebuild: {int((resets > 0).sum())}/{NUM_ENVS} envs")

    s = unpack_state(recorded["state"], imagination.state_format)
    env_index = torch.arange(NUM_ENVS, device=device)

    # rebuilt from the recorded cursor, the way a replayed rollout is seeded now
    ref_index = imagination.begin_rollout({"ref_cursor": recorded["cursor"]}, env_index, horizon=8)
    seeded = imagination._goal_values(s["root_pos_w"], s["root_quat_w"], ref_index)
    # rebuilt from where each environment is *now*, the way it was seeded before
    imagination.set_reference_window(build_reference_window(motion_reference, horizon=8 + imagination.goal_window))
    current = imagination._goal_values(s["root_pos_w"], s["root_quat_w"], ref_index)

    print(f"\n{'goal term':16s}{'seeded from cursor':>22s}{'current env time':>20s}   (max |imag - recorded|)")
    worst = 0.0
    for t in goal_terms:
        real = recorded["goal"][t]
        a = float((seeded[t].reshape(NUM_ENVS, -1) - real).abs().max())
        b = float((current[t].reshape(NUM_ENVS, -1) - real).abs().max())
        worst = max(worst, a)
        print(f"{t:16s}{a:22.6f}{b:20.6f}")

    # widths of the imagined observations against the environment's
    context = torch.zeros(
        NUM_ENVS, sum(int(torch.tensor(v).prod()) for v in imagination.context_format.values()), device=device
    )
    hist_s = recorded["state"].unsqueeze(1).expand(-1, imagination.history_length, -1).contiguous()
    hist_a = torch.zeros(NUM_ENVS, imagination.history_length, action_dim, device=device)
    imagination.begin_rollout({"ref_cursor": recorded["cursor"]}, env_index, horizon=8)
    pol = imagination.policy_obs(recorded["state"], context, None, ref_index, hist_s, hist_a)
    cri = imagination.critic_obs(recorded["state"], context, ref_index, hist_a[:, -1])
    noisy = imagination.add_obs_noise(pol)
    print(f"\npolicy width imagined {pol.shape[-1]} / env {recorded['policy_width']}")
    print(f"critic width imagined {cri.shape[-1]} / env {recorded['critic_width']}")
    print(f"sensor noise re-added: max |noisy - clean| = {float((noisy - pol).abs().max()):.4f}")
    assert pol.shape[-1] == recorded["policy_width"], "policy width mismatch"
    assert cri.shape[-1] == recorded["critic_width"], "critic width mismatch"
    assert worst < 1e-4, f"seeded goal disagrees with the recorded goal by {worst}"
    print("\nOK: seeded goals match the recorded ones")
    sim.close()
