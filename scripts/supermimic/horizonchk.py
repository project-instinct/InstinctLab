"""How far can the learned dynamics model be trusted?

The policy is trained by backpropagating through an H = 32-step open-loop rollout of the world model,
so the only question that matters for that design is how quickly the model's prediction diverges from
the simulator. This collects real transitions from the task, fits the model the way the algorithm does,
then compares an open-loop rollout against the simulator started from the same state with the same
actions.

Errors are normalised per state component by that component's spread over the real trajectory, so a
value of 1.0 means "as large as the natural variation of that quantity". Beyond that point the
prediction carries no more information than the mean.
"""

import numpy as np
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = 64
COLLECT_STEPS = 200
HORIZON = 32
FIT_STEPS = 3000
FIT_BATCH = 256
K = 8

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "hz"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, _ = cli.resolve_train_cfg(TASK, overrides)
env_cfg.scene.num_envs = NUM_ENVS

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinct_rl.modules.world_model import WorldModel
    from instinctlab.tasks.shadowing.supermimic.supermimic_provider import build_imagination
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    sim = gym.make(TASK, cfg=env_cfg)
    sim.reset()
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    device = torch.device("cuda")

    obs_format = env.get_obs_format()
    imagination = build_imagination(env=env, obs_format=obs_format)
    state_names = list(imagination.state_format)
    action_dim = unwrapped.action_manager.total_action_dim

    def pack_to_state(pack):
        split = imagination.split_obs_pack(pack)
        return torch.cat([split[n].reshape(split[n].shape[0], -1) for n in state_names], dim=-1)

    # ---- collect real transitions -------------------------------------------------------------
    obs, extras = env.get_observations()
    states = [pack_to_state(extras["observations"])]
    actions = []
    for _ in range(COLLECT_STEPS):
        # actions in the range the policy produces: unit-scale noise scaled by the action std
        action = torch.randn(NUM_ENVS, action_dim, device=device) * 0.5
        obs, reward, dones, extras = env.step(action)
        states.append(pack_to_state(extras["observations"]))
        actions.append(action)
    states = torch.stack(states)  # (T+1, B, S)
    actions = torch.stack(actions)  # (T, B, A)
    print(f"collected {states.shape[0]} steps x {NUM_ENVS} envs, state size {states.shape[-1]}")

    # ---- fit the world model the same way the algorithm does -----------------------------------
    model = WorldModel(
        state_segments=imagination.state_format,
        action_size=action_dim,
        hidden_sizes=(512, 512, 512),
        predict_delta=(
            "root_pos_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
        ),
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4)

    steps, batch = states.shape[0], states.shape[1]
    offsets_k = torch.arange(K, device=device)
    for it in range(FIT_STEPS):
        t0 = torch.randint(0, steps - K - 1, (FIT_BATCH,), device=device)
        e = torch.randint(0, batch, (t0.shape[0],), device=device)
        state0 = states[t0, e]
        target = states[t0[:, None] + 1 + offsets_k[None, :], e[:, None]]
        act = actions[t0[:, None] + offsets_k[None, :], e[:, None]]
        losses = model.rollout_loss(state0, act, None, target, update_normalizer=True)
        optimiser.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if it % 500 == 0:
            print(f"  fit {it:4d}  k-step loss {float(losses['loss'].detach()):.4f}")
    model.eval()

    # ---- open-loop rollout vs. reality --------------------------------------------------------
    # averaged over many start times so one unlucky sample does not set the conclusion
    scale = states.reshape(-1, states.shape[-1]).std(dim=0).clamp_min(1e-3)
    per_step = torch.zeros(HORIZON + 1, device=device)
    counts = torch.zeros(HORIZON + 1, device=device)
    per_component = torch.zeros(HORIZON + 1, states.shape[-1], device=device)
    for start in range(0, 40, 8):
        real = states[start : start + HORIZON + 1]
        act = actions[start : start + HORIZON]
        pred = [states[start]]
        with torch.no_grad():
            for t in range(HORIZON):
                pred.append(model(pred[-1], act[t], None))
        pred = torch.stack(pred)
        err = ((pred - real).abs() / scale).mean(dim=1)  # (H+1, S)
        per_step += err.mean(dim=-1)
        per_component += err
        counts += 1
    per_step /= counts[0]
    per_component /= counts.view(-1, 1)

    offsets, slices = 0, {}
    for name, shape in imagination.state_format.items():
        size = int(np.prod(shape))
        slices[name] = slice(offsets, offsets + size)
        offsets += size

    print("\nnormalised error by rollout step (1.0 = prediction no better than the mean):")
    for t in range(1, HORIZON + 1):
        print(f"   step {t:2d}: {float(per_step[t]):10.4f}")

    print("\nper-component error at selected steps:")
    print("   component            " + "".join(f"{t:>10d}" for t in (1, 4, 8, 16, 32)))
    for name, sl in slices.items():
        row = "".join(f"{float(per_component[t, sl].mean()):10.4f}" for t in (1, 4, 8, 16, 32))
        print(f"   {name:20s}{row}")

    sim.close()
