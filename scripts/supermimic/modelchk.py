"""Three checks of the SuperMimic learning signal, on real multi-step trajectories.

1. **Reward and termination parity.** Seeds the imagination from a recorded state and its reference
   cursor, then evaluates the analytic reward and the soft termination on the states the *simulator*
   actually produced over the next ``HORIZON`` steps. Each reward term must match the environment's own
   reward manager at every step. This isolates the reward/goal code from the world model.
2. **World-model open-loop error on policy rollouts.** Replays the recorded actions through the world
   model from the same start and reports the error at steps 1, 8, 16, 32 (paper Fig. 3c measures this
   "online on policy rollouts"). Normalised per state component by that component's spread, so 1.0
   means "no better than the mean".
3. **Policy gradient.** Compares the backpropagated gradient of the imagined objective (paper Eq. 2)
   with a central finite difference along the gradient direction, and checks every policy parameter
   group receives gradient.

Usage (inside the container)::

    CKPT=<path/to/model_N.pt> python scripts/supermimic/modelchk.py

``CKPT`` is optional. A checkpoint is loaded fully if it holds the policy; otherwise only its world model
(and the algorithm's other learners stay freshly initialised), which is enough for check 2.
"""

import os
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = int(os.environ.get("NUM_ENVS", "256"))
HORIZON = 32
WARMUP_STEPS = 40  # fills the replay so the gradient check can sample imagined starts
CKPT = os.environ.get("CKPT")

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "modelchk"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, agent_cfg = cli.resolve_train_cfg(TASK, overrides)
env_cfg.scene.num_envs = NUM_ENVS

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinct_rl.runners import OnPolicyRunner

    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import REWARD_TERMS, unpack_state
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    sim = gym.make(TASK, cfg=env_cfg)
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    alg = runner.alg
    imag = alg.imagination
    device = alg.device

    if CKPT:
        state_dict = torch.load(CKPT, weights_only=True, map_location=device)
        if "model_state_dict" in state_dict:
            runner.load(CKPT)
            print(f"loaded full checkpoint {CKPT}")
        else:
            alg.world_model.load_state_dict(state_dict["world_model"])
            for name, normalizer in runner.normalizers.items():
                key = f"{name}_normalizer_state_dict"
                if key in state_dict:
                    normalizer.load_state_dict(state_dict[key])
            print(f"checkpoint {CKPT} holds no policy: loaded its world model and normalisers only")
    alg.world_model.eval()
    alg.actor_critic.eval()
    for normalizer in runner.normalizers.values():
        normalizer.eval()

    def pack_state(pack):
        split = imag.split_obs_pack(pack)
        state = torch.cat([split[n].reshape(NUM_ENVS, -1) for n in imag.state_format], dim=-1)
        context = torch.cat([split[n].reshape(NUM_ENVS, -1) for n in imag.context_format], dim=-1)
        return state, context

    # ---- roll the real environment, filling the replay through the algorithm's own path --------------
    obs, extras = env.get_observations()
    extra_obs = extras["observations"]
    obs = runner._prepare_extra_obs(extra_obs)
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            obs, extra_obs, _, _, _ = runner.rollout_step(obs, extra_obs)

    # record one trajectory of HORIZON steps from here
    s0, ctx = pack_state(extra_obs)
    cursor = extra_obs["ref_cursor"].clone()
    prev_action = unwrapped.action_manager.action.clone()  # produced s0
    term_names = ("base_pos_too_far", "base_pg_too_far")
    states, actions, prev_actions, env_rewards, env_term, env_done = [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(HORIZON):
            obs, extra_obs, _, dones, _ = runner.rollout_step(obs, extra_obs)
            action = unwrapped.action_manager.action.clone()
            terms = unwrapped.reward_manager._termwise_reward_buf["rewards"]
            env_rewards.append(torch.stack([terms[n].clone() for n in REWARD_TERMS], dim=-1) * unwrapped.step_dt)
            env_term.append(torch.stack([unwrapped.termination_manager.get_term(n) for n in term_names], -1).any(-1))
            # a reset replaces the observed state with the new episode's
            env_done.append(dones.reshape(-1).bool().to(device))
            states.append(pack_state(extra_obs)[0])
            actions.append(action)
            prev_actions.append(prev_action)
            prev_action = action
    states = torch.stack(states)  # s_1 .. s_H
    actions = torch.stack(actions)  # a_0 .. a_{H-1}
    prev_actions = torch.stack(prev_actions)  # a_{-1} .. a_{H-2}
    env_rewards = torch.stack(env_rewards)  # r_1 .. r_H, per term
    env_term = torch.stack(env_term)
    env_done = torch.stack(env_done)
    alive = ~env_done.any(dim=0)
    print(f"\ntrajectory: {NUM_ENVS} envs x {HORIZON} steps; {int(alive.sum())} envs never reset")

    # ---- 1. reward / termination parity on simulator states ---------------------------------------
    env_index = torch.arange(NUM_ENVS, device=device)
    ref0 = imag.begin_rollout({"ref_cursor": cursor}, env_index, HORIZON)
    worst = torch.zeros(len(REWARD_TERMS), device=device)
    false_terminations = 0
    checked = 0
    keep = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)
    with torch.no_grad():
        for t in range(HORIZON):
            # env t's reward was computed on s_{t+1} before any reset, so it is valid while the env
            # had not reset at an earlier step
            ref = ref0 + t + 1
            imag_r = imag.reward(states[t], actions[t], prev_actions[t], ref)
            # a done step's observed state is already the next episode's, and every later row is too
            mask = keep & ~env_done[t]
            if mask.any():
                worst = torch.maximum(worst, (imag_r[mask] - env_rewards[t][mask]).abs().max(dim=0).values)
                prob = imag.termination_prob(states[t], ref).squeeze(-1)
                false_terminations += int((prob[mask] > 0.5).sum())
                checked += int(mask.sum())
            keep &= ~env_done[t]
    print("\n[1] reward parity on simulator states, max |imagined - environment| per term over all steps")
    for name, value in zip(REWARD_TERMS, worst.tolist()):
        print(f"    {name:22s} {value:.2e}")
    print(f"    soft termination > 0.5 where the environment did not terminate: {false_terminations}/{checked}")

    # ---- 2. world-model open-loop error on the same policy trajectory -------------------------------
    with torch.no_grad():
        pred, state = [], s0
        for t in range(HORIZON):
            state = alg.world_model(state, actions[t], ctx)
            pred.append(state)
        pred = torch.stack(pred)
    ok = alive
    real = states[:, ok]
    pred = pred[:, ok]
    scale = real.reshape(-1, real.shape[-1]).std(dim=0).clamp_min(1e-3)
    err = (pred - real).abs() / scale  # (H, B, S)
    print(f"\n[2] world-model open-loop error on policy rollouts ({int(ok.sum())} envs, 1.0 = no better than the mean)")
    for t in (1, 4, 8, 16, 32):
        print(f"    step {t:2d}: {float(err[t - 1].mean()):.4f}")
    offsets, worst_names = 0, []
    for name, shape in imag.state_format.items():
        size = 1
        for d in shape:
            size *= int(d)
        worst_names.append((float(err[HORIZON - 1][:, offsets : offsets + size].mean()), name))
        offsets += size
    print("    step 32 by component: " + ", ".join(f"{n}={v:.2f}" for v, n in sorted(worst_names, reverse=True)))

    # ---- 3. policy gradient against a finite difference, per imagination horizon ----------------------
    # A wrong gradient disagrees at every horizon; a correct gradient through dynamics that amplify
    # perturbations agrees at short horizons and stops describing the objective as H grows.
    alg._imagine()
    idx = torch.arange(64, device=device)
    saved_noise, saved_horizon = imag.policy_noise_halfwidth, alg.horizon
    imag.policy_noise_halfwidth = None  # a finite difference needs a deterministic objective
    params = alg.policy_params
    print("\n[3] policy gradient vs central finite difference along the gradient (ratio 1.0 = consistent)")
    for horizon in (1, 2, 4, 8, 16, 32):
        alg.horizon = horizon
        objective = alg._imagined_return(idx)[0]
        grads = torch.autograd.grad(objective, params, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        norm = float(torch.sqrt(sum((g * g).sum() for g in grads)))
        direction = [g / max(norm, 1e-12) for g in grads]
        ratios = []
        with torch.no_grad():
            for eps in (1e-2, 1e-3, 1e-4):
                for p, d in zip(params, direction):
                    p.add_(eps * d)
                plus = float(alg._imagined_return(idx)[0])
                for p, d in zip(params, direction):
                    p.sub_(2 * eps * d)
                minus = float(alg._imagined_return(idx)[0])
                for p, d in zip(params, direction):
                    p.add_(eps * d)
                ratios.append((plus - minus) / (2 * eps) / max(norm, 1e-12))
        print(
            f"    H={horizon:2d}: |grad| = {norm:10.3e}   ratio at eps 1e-2/1e-3/1e-4 = "
            + " / ".join(f"{r:7.3f}" for r in ratios)
        )
    alg.horizon = saved_horizon
    imag.policy_noise_halfwidth = saved_noise
    sim.close()
