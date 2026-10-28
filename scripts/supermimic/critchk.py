"""Imagination against reality, from the same start states under the same policy.

Loads a SuperMimic checkpoint, rolls the *simulator* forward from a batch of states with the current
policy, and imagines the same rollout in the world model from the very same states, histories and
reference cursors. Both sides act deterministically and without sensor noise, so any difference is the
world model's. Reports:

* per imagined step: mean reward in reality vs in imagination, and how far the imagined state is from
  the real one (in units of each dimension's one-step change);
* survival: the fraction of real rollouts still alive at each step vs the imagined survival weight;
* the critic: its value at the start and at step H on real states vs on imagined states, against the
  discounted return the simulator actually delivered from the start state.

Usage (inside the container)::

    CKPT=<run>/model_N.pt python scripts/supermimic/critchk.py
"""

import os
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = int(os.environ.get("NUM_ENVS", "1024"))
CKPT = os.environ["CKPT"]
WARMUP = 40  # real steps before the comparison, for a populated history and replay
RETURN_STEPS = 200  # real steps used for the Monte-Carlo return from the start state

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "critchk"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, agent_cfg = cli.resolve_train_cfg(TASK, overrides)
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.observations.policy.enable_corruption = False  # both sides noise-free

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinct_rl.runners import OnPolicyRunner

    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import REWARD_TERMS
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    sim = gym.make(TASK, cfg=env_cfg)
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    alg, imag = runner.alg, runner.alg.imagination
    device = alg.device

    # the learners and normalisers only: the replay in the checkpoint has another env count
    ckpt = torch.load(CKPT, weights_only=True, map_location=device)
    alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
    alg.world_model.load_state_dict(ckpt["world_model"])
    for name, normalizer in runner.normalizers.items():
        normalizer.load_state_dict(ckpt[f"{name}_normalizer_state_dict"])
        normalizer.eval()
    alg.actor_critic.eval()
    alg.world_model.eval()
    imag.policy_noise_halfwidth = None
    gamma, horizon = alg.gamma, alg.horizon
    print(f"loaded {CKPT}: iteration {ckpt.get('iter')}, H = {horizon}")

    def pack_state(pack):
        split = imag.split_obs_pack(pack)
        state = torch.cat([split[n].reshape(NUM_ENVS, -1) for n in imag.state_format], dim=-1)
        context = torch.cat([split[n].reshape(NUM_ENVS, -1) for n in imag.context_format], dim=-1)
        return state, context

    def act(obs):
        return alg.actor_critic.act_inference(obs)

    def env_step(action):
        obs, _, dones, infos = env.step(action)
        extra = infos["observations"]
        extra.setdefault("policy", obs)
        terms = unwrapped.reward_manager._termwise_reward_buf["rewards"]
        # the manager's term buffer is weighted but not yet multiplied by dt; the imagination's reward is
        reward = torch.stack([terms[n] for n in REWARD_TERMS], dim=-1).sum(-1) * unwrapped.step_dt
        return runner._prepare_extra_obs(extra), extra, reward, dones.reshape(-1).bool().to(device)

    with torch.no_grad():
        obs, extras = env.get_observations()
        extra = extras["observations"]
        extra.setdefault("policy", obs)
        obs = runner._prepare_extra_obs(extra)
        hist_s, hist_a, hist_done = [], [], []
        h = imag.history_length
        episode_step = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        for _ in range(WARMUP):
            action = act(obs)
            obs, extra, _, done = env_step(action)
            s, _ = pack_state(extra)
            hist_s.append(s)
            hist_a.append(unwrapped.action_manager.action.clone())
            hist_done.append(done)
            episode_step = (episode_step + 1) * (~done).long()  # the replay's own bookkeeping
        # environments whose last ``h`` frames lie in one episode: their recorded window is what the
        # manager's history holds, so the imagined observation must reproduce the real one exactly
        intact_history = ~torch.stack(hist_done[-h:]).any(dim=0)
        # start states: the current one, with the history and cursor the algorithm would replay
        s0, ctx = pack_state(extra)
        start = {
            "state": s0,
            "context": ctx,
            "env_index": torch.arange(NUM_ENVS, device=device),
            "seeds": {"ref_cursor": extra["ref_cursor"].clone()},
            "history_state": None,
            "history_action": None,
        }
        # built exactly as the algorithm builds a replayed start
        start["history_state"], start["history_action"] = alg.history_since_episode_start(
            torch.stack(hist_s[-h:], dim=1), torch.stack(hist_a[-h:], dim=1), episode_step
        )
        obs0 = obs

        # ---- reality ------------------------------------------------------------------------------
        real_states, real_rewards, real_dones = [], [], []
        obs = obs0
        real_a0 = act(obs0)
        real_terms1 = None
        for step in range(RETURN_STEPS):
            obs, extra, reward, done = env_step(act(obs))
            if step == 0:
                terms = unwrapped.reward_manager._termwise_reward_buf["rewards"]
                real_terms1 = torch.stack([terms[n].clone() for n in REWARD_TERMS], dim=-1) * unwrapped.step_dt
            real_states.append(pack_state(extra)[0])
            real_rewards.append(reward)
            real_dones.append(done)
        real_states = torch.stack(real_states)  # s_1 ..
        real_rewards = torch.stack(real_rewards)  # r_1 ..
        real_dones = torch.stack(real_dones)
        alive = torch.cumprod((~real_dones).float(), dim=0)  # alive after step t
        # discounted return delivered by the simulator (truncated at RETURN_STEPS)
        disc = gamma ** torch.arange(RETURN_STEPS, device=device, dtype=torch.float32).unsqueeze(1)
        alive_before = torch.cat([torch.ones(1, NUM_ENVS, device=device), alive[:-1]], dim=0)
        mc_return = (disc * alive_before * real_rewards).sum(0)

        # ---- imagination from the same starts --------------------------------------------------------
        alg._bounds = alg._state_bounds()  # no replay: no clamping
        imagined = alg._imagine(start)

    # ---- compare ------------------------------------------------------------------------------------
    full = alive[horizon - 1] > 0  # rollouts the simulator kept alive over the whole horizon
    step_scale = (real_states[1:] - real_states[:-1])[:, full].reshape(-1, s0.shape[-1]).std(0).clamp_min(1e-4)
    print(f"\n{int(full.sum())}/{NUM_ENVS} real rollouts survive all {horizon} steps (compared below)")
    print(f"{'step':>5s} {'real r':>9s} {'imag r':>9s} {'state err':>10s} {'real alive':>11s} {'imag surv':>10s}")
    surv = torch.cumprod(1.0 - imagined["done"].squeeze(-1), dim=0)  # (H, B)
    for t in (1, 2, 4, 8, 12, 16, 24, 32):
        if t > horizon:
            break
        err = ((imagined["state"][t] - real_states[t - 1]).abs() / step_scale)[full].mean()
        print(
            f"{t:5d} {float(real_rewards[t - 1][full].mean()):9.3f} {float(imagined['reward'][t - 1].squeeze(-1)[full].mean()):9.3f}"
            f" {float(err):10.2f} {float(alive[t - 1].mean()):11.3f} {float(surv[t - 1].mean()):10.3f}"
        )
    # ---- step 1 in detail: is the difference in the action, the observation or the state? -------
    with torch.no_grad():
        imag.subset_reference_rows(torch.arange(NUM_ENVS, device=device))
        ref0 = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        imag_obs0 = imag.policy_obs(
            start["state"],
            start["context"],
            start["history_action"][:, -1],
            ref0,
            start["history_state"],
            start["history_action"],
        )
        imag_a0 = imagined["prev_action"][1]
        imag_terms1 = imag.reward(imagined["state"][1], imag_a0, start["history_action"][:, -1], ref0 + 1)
    # the runner normalises observation groups in place, so compare in the normalised space
    diff = (alg._normalize("policy", imag_obs0) - obs0).abs()
    print(f"\nstep 1 observation, {int(intact_history.sum())} envs with an uninterrupted history window:")
    offset = 0
    for name, dim in runner.env.get_obs_format()["policy"].items():
        width = 1
        for d in dim:
            width *= int(d)
        part = diff[intact_history, offset : offset + width]
        part_all = diff[:, offset : offset + width]
        print(
            f"    {name:18s} max {float(part.max()):9.4f}  mean {float(part.mean()):8.4f}   (all envs: max"
            f" {float(part_all.max()):9.4f})"
        )
        offset += width
    print(
        f"step 1: max |imagined obs - real obs| (normalised) = {float(diff.max()):.4e}"
        f"   mean |a_imag - a_real| = {float((imag_a0 - real_a0).abs().mean()):.4e}"
        f"   mean |a_0 - a_-1| real = {float((real_a0 - start['history_action'][:, -1]).abs().mean()):.4f}"
    )
    print("step 1 reward by term, mean over surviving rollouts:  real / imagined")
    for i, name in enumerate(REWARD_TERMS):
        print(f"    {name:22s} {float(real_terms1[full, i].mean()):9.4f} {float(imag_terms1[full, i].mean()):9.4f}")

    real_h = (disc[:horizon] * alive_before[:horizon] * real_rewards[:horizon]).sum(0)
    imag_h = (
        (gamma ** torch.arange(horizon, device=device)).view(-1, 1)
        * torch.cat([torch.ones(1, NUM_ENVS, device=device), surv[:-1]])
        * imagined["reward"].squeeze(-1)
    ).sum(0)
    print(f"\nH-step discounted reward:  real {float(real_h.mean()):.2f}   imagined {float(imag_h.mean()):.2f}")

    with torch.no_grad():
        prev_h = imagined["prev_action"][horizon]
        v0 = imagined["value"][0].squeeze(-1)
        vH_imag = imagined["value"][horizon].squeeze(-1)
        ref_h = torch.full((NUM_ENVS,), horizon, dtype=torch.long, device=device)
        imag.subset_reference_rows(torch.arange(NUM_ENVS, device=device))
        vH_real = alg._value(real_states[horizon - 1], ctx, ref_h, prev_h).squeeze(-1)
    print("\ncritic, mean over rollouts that survive the horizon:")
    print(f"    V(s_0)                     {float(v0[full].mean()):10.2f}")
    print(f"    simulator return from s_0  {float(mc_return[full].mean()):10.2f}   (truncated at {RETURN_STEPS} steps)")
    print(f"    V(real s_H)                {float(vH_real[full].mean()):10.2f}")
    print(f"    V(imagined s_H)            {float(vH_imag[full].mean()):10.2f}")
    print(
        f"    corr(V(s_0), return)       {float(torch.corrcoef(torch.stack([v0[full], mc_return[full]]))[0, 1]):10.3f}"
    )
    sim.close()
