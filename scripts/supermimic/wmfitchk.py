"""Which world-model fit gives accurate long rollouts? An A/B on identical data.

Collects transitions from the task with a reference-replaying policy (the reference joint angles as
setpoints, plus the collection noise), so states look like what a tracking policy visits. The
reference-deviation terminations are disabled so trajectories are long enough to test 32 steps. Then
fits the paper-size world model three ways on the same training environments and measures open-loop
error on held-out environments:

* ``A old``         -- k independent one-step fits (each step restarted from the logged state), raw-unit
  loss, quaternions absolute: what the algorithm did before.
* ``B autoregress`` -- autoregressive k-step fit, but still the raw-unit loss.
* ``C paper``       -- autoregressive k-step fit with the loss in units of each dimension's one-step
  change and every component a delta: ``WorldModel.rollout_loss`` now.

Error is reported normalised by each component's one-step change (1.0 = off by a typical step), and
in metres for the tracked link positions.
"""

import os
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = int(os.environ.get("NUM_ENVS", "1024"))
STEPS = int(os.environ.get("STEPS", "300"))
FIT_STEPS = int(os.environ.get("FIT_STEPS", "3000"))
K = 8
EVAL_H = 32

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "wmfit"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, agent_cfg = cli.resolve_train_cfg(TASK, overrides)
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.terminations.base_pos_too_far = None
env_cfg.terminations.base_pg_too_far = None

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinct_rl.modules.world_model import WorldModel

    from instinctlab.tasks.shadowing.supermimic.supermimic_provider import build_imagination
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    sim = gym.make(TASK, cfg=env_cfg)
    sim.reset()
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    obs_format = env.get_obs_format()
    imag = build_imagination(env=env, obs_format=obs_format)
    device = unwrapped.device
    action_term = unwrapped.action_manager.get_term("joint_pos")
    goal_start = 0  # joint_pos_ref is the first policy term; its first column is the current frame
    num_joints = int(obs_format["policy"]["joint_pos_ref"][-1])
    noise_std = float(agent_cfg.algorithm.collection_noise_std)

    def split(pack):
        parts = imag.split_obs_pack(pack)
        state = torch.cat([parts[n].reshape(NUM_ENVS, -1) for n in imag.state_format], dim=-1)
        context = torch.cat([parts[n].reshape(NUM_ENVS, -1) for n in imag.context_format], dim=-1)
        return state, context

    # ---- collect ------------------------------------------------------------------------------------
    obs, extras = env.get_observations()
    pack = extras["observations"]
    states, contexts, actions, dones = [], [], [], []
    with torch.no_grad():
        s, c = split(pack)
        states.append(s)
        contexts.append(c)
        for _ in range(STEPS):
            q_ref_rel = pack["policy"][:, goal_start : goal_start + num_joints]  # q_ref - q0, policy order
            action = q_ref_rel / action_term._scale + noise_std * torch.randn(NUM_ENVS, num_joints, device=device)
            obs, rew, done, extras = env.step(action)
            pack = extras["observations"]
            s, c = split(pack)
            states.append(s)
            contexts.append(c)
            actions.append(action)
            dones.append(done.reshape(-1).bool())
    states = torch.stack(states)  # (T+1, E, S)
    contexts = torch.stack(contexts)  # (T+1, E, C)
    actions = torch.stack(actions)  # (T, E, A)
    dones = torch.stack(dones)  # (T, E): dones[t] means states[t+1] belongs to a new episode
    print(f"collected {STEPS} steps x {NUM_ENVS} envs; resets {int(dones.sum())}")

    # a window starting at t of length L is usable when no step t .. t+L-1 ended an episode
    def window_starts(length: int, envs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        broken = dones.float().unfold(0, length, 1).amax(dim=-1) > 0  # (T-L+1, E)
        broken = broken[:, envs]
        t, e = torch.nonzero(~broken, as_tuple=True)
        return t, envs[e]

    perm = torch.randperm(NUM_ENVS, device=device)
    train_envs, test_envs = perm[: int(0.8 * NUM_ENVS)], perm[int(0.8 * NUM_ENVS) :]
    train_t, train_e = window_starts(K, train_envs)
    test_t, test_e = window_starts(EVAL_H, test_envs)
    print(f"train windows (k={K}): {train_t.numel()}, test windows (H={EVAL_H}): {test_t.numel()}")

    def gather(t, e, length):
        offs = torch.arange(length + 1, device=device)
        s = states[t.unsqueeze(1) + offs, e.unsqueeze(1)]  # (B, L+1, S)
        a = actions[t.unsqueeze(1) + offs[:-1], e.unsqueeze(1)]  # (B, L, A)
        return s, a, contexts[t, e]

    # scale for reporting: each dimension's one-step change on the training data
    s_tr, _, _ = gather(train_t, train_e, 1)
    step_scale = (s_tr[:, 1] - s_tr[:, 0]).std(dim=0).clamp_min(1e-4)
    link = {}
    off = 0
    for name, shape in imag.state_format.items():
        n = 1
        for d in shape:
            n *= int(d)
        link[name] = slice(off, off + n)
        off += n

    def make_model(predict_delta):
        return WorldModel(
            state_segments=imag.state_format,
            action_size=actions.shape[-1],
            context_size=contexts.shape[-1],
            hidden_sizes=tuple(agent_cfg.algorithm.world_model_hidden_sizes),
            predict_delta=predict_delta,
        ).to(device)

    old_delta = (
        "root_pos_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "joint_pos",
        "joint_vel",
        "body_link_pos_w",
        "body_link_lin_vel_w",
        "body_link_ang_vel_w",
    )

    def loss_old(model, s, a, c):
        """k one-step fits from logged states, raw-unit MSE (the previous ``rollout_loss``)."""
        err = []
        for j in range(K):
            pred = model(s[:, j], a[:, j], c, update_normalizer=True)
            err.append(((pred - s[:, j + 1]) ** 2).mean())
        return torch.stack(err).mean()

    def loss_autoregressive_raw(model, s, a, c):
        """Autoregressive k-step fit, raw-unit MSE; normalisers updated from logged data."""
        model._update_normalizers(s[:, 0], a, c, s[:, 1:], None)
        state, err = s[:, 0], []
        for j in range(K):
            state = model(state, a[:, j], c)
            err.append(((state - s[:, j + 1]) ** 2).mean())
        return torch.stack(err).mean()

    def loss_paper(model, s, a, c):
        return model.rollout_loss(s[:, 0], a, c, s[:, 1:])["loss"]

    variants = {
        "A old": (old_delta, loss_old),
        "B autoregress": (old_delta, loss_autoregressive_raw),
        "C paper": ("all", loss_paper),
    }
    chosen = os.environ.get("VARIANTS")
    if chosen:
        variants = {k: v for k, v in variants.items() if k.split()[0] in chosen.split(",")}
    results = {}
    breakdown = {}
    for label, (predict_delta, loss_fn) in variants.items():
        torch.manual_seed(0)
        model = make_model(predict_delta)
        opt = torch.optim.Adam(model.parameters(), lr=float(agent_cfg.algorithm.world_model_learning_rate))
        for it in range(FIT_STEPS):
            pick = torch.randint(0, train_t.numel(), (int(agent_cfg.algorithm.wm_batch_size),), device=device)
            s, a, c = gather(train_t[pick], train_e[pick], K)
            loss = loss_fn(model, s, a, c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pick = torch.randperm(test_t.numel(), device=device)[:2048]
            s, a, c = gather(test_t[pick], test_e[pick], EVAL_H)
            state, err, link_err, per_dim = s[:, 0], [], [], []
            for j in range(EVAL_H):
                state = model(state, a[:, j], c)
                e = (state - s[:, j + 1]).abs() / step_scale  # (B, S)
                err.append(e.mean())
                per_dim.append(e.mean(dim=0))
                d = (state[:, link["body_link_pos_w"]] - s[:, j + 1, link["body_link_pos_w"]]).reshape(-1, 14, 3)
                link_err.append(d.norm(dim=-1).mean())
        results[label] = (torch.stack(err).tolist(), torch.stack(link_err).tolist())
        per_dim = torch.stack(per_dim)  # (H, S)
        breakdown[label] = {
            name: (float(per_dim[0, sl].mean()), float(per_dim[-1, sl].mean()), float(per_dim[0, sl].max()))
            for name, sl in link.items()
        }
        print(f"fitted {label} (final train loss {float(loss):.4f})", flush=True)

    print("\nopen-loop error on held-out environments")
    print(f"{'':16s}" + "".join(f"{'step ' + str(t):>12s}" for t in (1, 4, 8, 16, 32)))
    for label, (err, _) in results.items():
        print(f"{label:16s}" + "".join(f"{err[t - 1]:12.2f}" for t in (1, 4, 8, 16, 32)) + "   (x one-step change)")
    for label, (_, lerr) in results.items():
        print(
            f"{label:16s}"
            + "".join(f"{lerr[t - 1] * 100:11.1f}c" for t in (1, 4, 8, 16, 32))
            + "   (link position, cm)"
        )
    for label, comps in breakdown.items():
        print(f"\n{label}: error by component (x one-step change): step 1 mean / step 32 mean / step 1 worst dim")
        for name, (e1, e32, w1) in sorted(comps.items(), key=lambda kv: -kv[1][0]):
            print(f"    {name:22s} {e1:8.2f} {e32:8.2f} {w1:8.2f}")
    print("\none-step change scale used above (mean over dims, per component):")
    for name, sl in link.items():
        print(f"    {name:22s} {float(step_scale[sl].mean()):.5f}  (min {float(step_scale[sl].min()):.5f})")
    sim.close()
