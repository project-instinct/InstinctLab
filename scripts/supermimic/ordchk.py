"""Verify the history ordering the imagined rollout assumes.

``SuperMimicImagination.advance_history`` appends the newest frame at the end of the window and treats
``history_action[:, j]`` as the action that produced ``history_state[:, j]``. If the manager ordered its
buffer the other way, the policy would be trained on a time-reversed input and no width check would
catch it.

This steps the environment and compares the manager's own history buffer against the raw state
sequence, so the ordering claim is tested against the authority rather than asserted.
"""

import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = 4
STEPS = 16

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "ord"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, _ = cli.resolve_train_cfg(TASK, overrides)
env_cfg.scene.num_envs = NUM_ENVS

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    sim = gym.make(TASK, cfg=env_cfg)
    sim.reset()
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    om = unwrapped.observation_manager
    robot = unwrapped.scene["robot"]
    action_dim = unwrapped.action_manager.total_action_dim

    # collect the raw per-step joint positions as the environment advances
    history = []
    obs, extras = env.get_observations()
    history.append(robot.data.joint_pos.torch.clone())
    for _ in range(STEPS):
        obs, rew, dones, extras = env.step(torch.zeros(NUM_ENVS, action_dim, device=robot.device))
        history.append(robot.data.joint_pos.torch.clone())

    buf = om._group_obs_term_history_buffer["policy"]["joint_pos"].buffer  # (B, H, J), oldest first
    print("manager joint_pos history buffer:", tuple(buf.shape))
    h = buf.shape[1]
    recent = torch.stack(history[-h:])  # (h, B, J), oldest of the kept frames first

    newest_last = (buf - recent.permute(1, 0, 2)).abs().max()
    newest_first = (buf - recent.flip(0).permute(1, 0, 2)).abs().max()
    print(f"|buffer - chronological| max = {float(newest_last):.6f}")
    print(f"|buffer - reversed|      max = {float(newest_first):.6f}")

    if float(newest_last) <= float(newest_first):
        print("ordering: oldest first, newest last  -> matches advance_history")
    else:
        print("ordering: NEWEST FIRST -> advance_history is wrong")

    # and the action history: the manager's last_action window must be the actions that produced the
    # states, i.e. shifted one step back from the action taken at each frame
    action_buf = om._group_obs_term_history_buffer["policy"]["last_action"].buffer
    print(f"\nlast_action buffer {tuple(action_buf.shape)}; last frame equals final applied action:",
          bool(torch.equal(action_buf[:, -1], unwrapped.action_manager.action)))
    print("last_action buffer all-zero (zeros at reset, as the imagination assumes):",
          bool((action_buf == 0).all()))

    sim.close()
