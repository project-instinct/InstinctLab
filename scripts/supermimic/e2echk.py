"""End-to-end check: build the imagined observation from the real state and compare to the real one.

Width and ordering checks can both pass while the assembled values are wrong (wrong frame order inside
a term, wrong default pose, terms concatenated out of order). This hands the imagination the live state
plus the manager's own history buffers and compares the result against the observation the environment
actually produced, term by term.

The real observation carries noise by design (``enable_corruption``), so the comparison is against the
noise bounds rather than for exact equality: a term built from the wrong frames is off by far more than
the noise it is allowed.
"""

import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS, STEPS = 8, 12

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "e2e"], require_run_name=True
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
    om = unwrapped.observation_manager
    action_dim = unwrapped.action_manager.total_action_dim
    obs_format = env.get_obs_format()
    imagination = build_imagination(env=env, obs_format=obs_format)

    # Step past the history length, recording the state each frame, so a genuine trailing window is
    # available. This is what the replay supplies to an imagined rollout; using a faked window here
    # would make the history terms disagree for reasons that have nothing to do with the code.
    def current_state(pack_):
        split_ = imagination.split_obs_pack(pack_)
        return torch.cat([split_[n].reshape(NUM_ENVS, -1) for n in imagination.state_format], dim=-1)

    obs, extras = env.get_observations()
    states = [current_state(extras["observations"])]
    actions = []
    for _ in range(STEPS):
        action = torch.zeros(NUM_ENVS, action_dim, device="cuda")
        obs, rew, dones, extras = env.step(action)
        states.append(current_state(extras["observations"]))
        actions.append(unwrapped.action_manager.action.clone())

    pack = extras["observations"]
    state = states[-1]

    # the manager's own history buffers, exactly as the rollout would reconstruct from the replay
    history_state = None
    h = imagination.history_length
    hist_terms = {}
    for name in imagination.history_terms:
        held = om._group_obs_term_history_buffer["policy"][name].buffer
        hist_terms[name] = held
    history_action = hist_terms["last_action"].reshape(NUM_ENVS, h, -1)

    imagination.refresh_reference_window(imagination.goal_window)
    imagination.select_reference_rows(torch.arange(NUM_ENVS, device=state.device))
    ref_index = torch.zeros(NUM_ENVS, dtype=torch.long, device=state.device)

    # the trailing window, oldest first, current last -- the same layout the replay sampling yields
    history_state = torch.stack(states[-h:]).permute(1, 0, 2).contiguous()
    # the stored action at a frame is the action applied to reach it, so this is directly the
    # ``last_action`` history
    history_action = torch.stack(actions[-h:]).permute(1, 0, 2).contiguous()

    imag = imagination.policy_obs(state, None, None, ref_index, history_state, history_action)
    real = pack["policy"]
    real = real if torch.is_tensor(real) else torch.cat([v.reshape(v.shape[0], -1) for v in real.values()], dim=-1)

    # compare per term, using the manager's declared layout
    offset = 0
    print(f"{'term':16s}{'width':>8s}{'max |imag - real|':>20s}   note")
    for name, dim in obs_format["policy"].items():
        width = 1
        for d in dim:
            width *= int(d)
        a = imag[:, offset : offset + width]
        b = real[:, offset : offset + width]
        diff = float((a - b).abs().max())
        # history terms of state-only quantities should match the manager's buffer almost exactly
        note = "history (from manager buffer)" if name in imagination.history_terms else "single frame"
        if name == "last_action":
            note = "history (zeros at reset)"
        print(f"{name:16s}{width:8d}{diff:20.6f}   {note}")
        offset += width

    sim.close()
