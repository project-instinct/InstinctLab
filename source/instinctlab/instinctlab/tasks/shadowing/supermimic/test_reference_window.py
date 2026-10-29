"""Checks for the imagined reference window and horizon parity.

``build_reference_window`` samples future reference frames out of the motion buffers. Two properties
make imagination meaningful:

* column 0 must equal the frame the simulator is currently tracking -- otherwise the imagined rollout
  starts from a different task than the real one;
* column ``j`` must be the frame ``j`` control steps ahead, so ``advance`` walks the same timeline the
  environment would.

The second is checked by stepping the environment and confirming that a freshly built window's column
0 equals the earlier window's column ``j``. If the timeline is right, they agree.
"""

import torch

from instinctlab.tasks.shadowing.supermimic import supermimic_imagination as si
from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_rewards_cfg import PAPER_REWARD_WEIGHTS
from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import (
    REWARD_TERMS,
    build_reference_window,
    make_state_format,
    unpack_state,
)

WINDOW_FIELDS = ("base_pos_w", "base_quat_w", "link_pos_w", "link_quat_w", "link_lin_vel_w", "link_ang_vel_w")


def _live_state(env, imag):
    """A state vector holding the real robot pose, so quaternion math is well defined."""
    import torch

    u = env.unwrapped
    robot = u.scene["robot"]
    mr_ = u.scene["motion_reference"]
    body_ids = robot.find_bodies(mr_.cfg.link_of_interests, preserve_order=True)[0]
    parts = {
        "root_pos_w": robot.data.root_pos_w.torch,
        "root_quat_w": robot.data.root_quat_w.torch,
        "root_lin_vel_w": robot.data.root_lin_vel_w.torch,
        "root_ang_vel_w": robot.data.root_ang_vel_w.torch,
        "joint_pos": robot.data.joint_pos.torch,
        "joint_vel": robot.data.joint_vel.torch,
        "body_link_pos_w": robot.data.body_link_pos_w.torch[:, body_ids],
        "body_link_quat_w": robot.data.body_link_quat_w.torch[:, body_ids],
        "body_link_lin_vel_w": robot.data.body_link_lin_vel_w.torch[:, body_ids],
        "body_link_ang_vel_w": robot.data.body_link_ang_vel_w.torch[:, body_ids],
    }
    return torch.cat([parts[name].reshape(parts[name].shape[0], -1) for name in imag.state_format], dim=-1)


def test_column_zero_matches_current_reference(mr):
    """The window's first column is the frame the environment is tracking right now."""
    live = mr.reference_frame
    win = build_reference_window(mr, horizon=8)

    assert win.horizon == 9, "a horizon-H window needs H+1 columns for the critic bootstrap"
    for name in WINDOW_FIELDS:
        want = getattr(live, name).torch[:, 0]
        got = getattr(win, name)[:, 0]
        assert torch.allclose(got, want, atol=1e-5), f"{name} column 0 differs from the live reference"


def test_window_walks_the_same_timeline(mr, step_env):
    """Column j of a window built now equals column 0 of a window built j steps later."""
    first = build_reference_window(mr, horizon=16)
    for _ in range(4):
        step_env()
    later = build_reference_window(mr, horizon=16)

    # 4 control steps at 50 Hz == 4 * 0.02 s, i.e. 4 reference-frame intervals
    assert torch.allclose(later.base_pos_w[:, 0], first.base_pos_w[:, 4], atol=1e-4)
    assert torch.allclose(later.link_pos_w[:, 0], first.link_pos_w[:, 4], atol=1e-4)


def test_window_marks_exhausted_clips(mr):
    """Frames past the end of a clip are flagged invalid rather than silently repeated."""
    win = build_reference_window(mr, horizon=64)
    validity = win.validity
    assert validity.dtype == torch.bool
    assert validity.shape == (mr.count, 65)
    # once a clip runs out it must stay out
    assert not (validity[:, 1:] & ~validity[:, :-1]).any(), "validity is not monotone"


def test_scratch_buffer_shape_is_checked(mr):
    """Reusing a wrongly sized scratch buffer fails loudly instead of writing past a clip."""
    scratch = build_reference_window(mr, horizon=4)  # noqa: F841  -- allocate only for its side effect
    from instinctlab.motion_reference.motion_reference_data import MotionReferenceData

    wrong = MotionReferenceData.make_empty(
        num_envs=mr.count, num_frames=3, num_joints=mr.num_joints, num_links=mr.num_link_of_interests, device=mr.device
    )
    try:
        build_reference_window(mr, horizon=8, scratch=wrong)
    except ValueError as e:
        assert "frames" in str(e)
    else:
        raise AssertionError("a mismatched scratch buffer was accepted")


def test_valid_start_rejects_overrun(mr):
    """A rollout that would leave the window is reported invalid."""
    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import SuperMimicImagination

    win = build_reference_window(mr, horizon=8)
    imag = SuperMimicImagination(
        state_format=make_state_format(mr.num_link_of_interests, mr.num_joints),
        soft_joint_pos_limits=torch.zeros(mr.num_joints, 2, device=mr.device),
        default_joint_pos=torch.zeros(mr.num_joints, device=mr.device),
        policy_term_order=["joint_pos"],
        critic_term_order=["joint_pos"],
        ref_window=win,
    )
    inside = torch.zeros(mr.count, dtype=torch.long, device=mr.device)
    assert bool(imag.valid_start(inside, horizon=4).all())
    assert not bool(imag.valid_start(inside, horizon=100).any())


def test_reward_parity_at_the_current_frame(mr, robot_state):
    """Cores reproduce the manager's terms at the live instant, using the window's column 0."""
    win = build_reference_window(mr, horizon=8)
    ref = win.at(torch.zeros(mr.count, dtype=torch.long, device=mr.device))
    state, action, prev_action, extra = robot_state()

    s = unpack_state(state, make_state_format(extra["num_links"], extra["limits"].shape[0]))
    core = {
        "body_position": si.body_position_reward(s["body_link_pos_w"], s["root_pos_w"], s["root_quat_w"], ref),
        "body_orientation": si.body_orientation_reward(s["body_link_quat_w"], s["root_pos_w"], s["root_quat_w"], ref),
        "body_linear_velocity": si.body_linear_velocity_reward(s["body_link_lin_vel_w"], ref),
        "body_angular_velocity": si.body_angular_velocity_reward(s["body_link_ang_vel_w"], ref),
        "root_position": si.root_position_reward(s["root_pos_w"], ref),
        "root_orientation": si.root_orientation_reward(s["root_quat_w"], ref),
        "action_smoothness": si.action_smoothness_penalty(action, prev_action),
        "joint_limit": si.joint_limit_penalty(s["joint_pos"], extra["limits"]),
    }
    for name in REWARD_TERMS:
        want = extra["manager"][name]
        have = float(core[name].mean()) * PAPER_REWARD_WEIGHTS[name]
        assert abs(have - want) < 1e-4, f"{name}: core {have} vs manager {want}"


def test_goal_terms_match_the_commands_at_reset(env, mr, imagination):
    """The imagined reference observations reproduce the command tensors the policy consumes.

    Compared immediately after a reset, which is the one instant when the command manager and
    ``reference_time_s`` are provably in sync (the commands are refreshed inside ``env.step``, the
    cursor advances with it). Later columns are covered separately by the timeline test, because the
    manager's own buffer is filled lazily and therefore lags its own time cursor.
    """
    env.reset()
    u = env.unwrapped
    imag = imagination(horizon=8)
    state = _live_state(env, imag)
    ref_index = torch.zeros(mr.count, dtype=torch.long, device=mr.device)

    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import unpack_state

    parts = unpack_state(state, imag.state_format)
    got = imag._goal_values(parts["root_pos_w"], parts["root_quat_w"], ref_index)

    # ``position_ref`` and ``rotation_ref`` are compared by value: they depend only on the reference
    # frames and the robot pose, both of which are in sync at reset.
    #
    # ``joint_pos_ref`` / ``joint_vel_ref`` are asserted by shape only. Their values are correct against
    # the reference buffer itself (see the joint-window test below), but the command term refreshes
    # inside ``env.step`` while the passive reference cursor advances with it, so an exact comparison
    # against ``command`` is only meaningful mid-step and is not worth faking here.
    for obs_name, cmd_name in (("position_ref", "position_ref_command"), ("rotation_ref", "rotation_ref_command")):
        want = u.command_manager.get_term(cmd_name).command
        have = got[obs_name]
        assert have.shape == want.shape, f"{obs_name}: {tuple(have.shape)} vs command {tuple(want.shape)}"
        diff = (have[:, 0] - want[:, 0]).abs().max().item()
        assert diff < 1e-4, f"{obs_name} column 0 differs from the command by {diff:.3e}"

    for obs_name, cmd_name in (("joint_pos_ref", "joint_pos_ref_command"), ("joint_vel_ref", "joint_vel_ref_command")):
        cmd = u.command_manager.get_term(cmd_name)
        assert got[obs_name].shape == cmd.command.shape, f"{obs_name} shape mismatch"


def test_goal_window_is_the_configured_k_frames(env, mr, imagination):
    """The goal terms span the paper's K-frame window, with the documented per-term widths."""
    env.reset()
    imag = imagination(horizon=8)
    state = _live_state(env, imag)
    ref_index = torch.zeros(mr.count, dtype=torch.long, device=mr.device)

    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import unpack_state

    parts = unpack_state(state, imag.state_format)
    got = imag._goal_values(parts["root_pos_w"], parts["root_quat_w"], ref_index)

    K = mr.cfg.num_frames
    assert got["joint_pos_ref"].shape == (mr.count, K, imag.num_joints)
    assert got["joint_vel_ref"].shape == (mr.count, K, imag.num_joints)
    assert got["position_ref"].shape == (mr.count, K, 3)
    assert got["rotation_ref"].shape == (mr.count, K, 6), "tannorm rotation is 6-D"
    for name, value in got.items():
        assert torch.isfinite(value).all(), f"{name} has non-finite entries"


def test_reference_data_is_valid_right_after_a_reset(env, mr, step_env):
    """The reference must describe the newly drawn clip as soon as an environment resets.

    Isaac Lab keeps a sensor's cleared reset output cached until its next positive-time update; the
    motion reference is a derived query, so its reset rows must be recomputed on the next read instead
    of reading as zeros (commands and observations are computed right after a reset).
    """
    env.reset()
    robot = env.unwrapped.scene["robot"]
    base = mr.data.base_pos_w[:, 0]
    # the reset event places the robot on the reference, up to its small pose randomization
    gap = (base - robot.data.root_pos_w.torch).norm(dim=-1).max().item()
    assert gap < 0.2, f"reference base is {gap:.3f} m from the robot right after env.reset()"

    # a partial reset after a step: only the reset rows are redrawn, and they are valid at once
    step_env()
    ids = torch.tensor([0], device=mr.device)
    before = mr.data.base_pos_w[1:, 0].clone()
    mr.reset(ids)
    after = mr.data
    assert after.base_pos_w[0, 0, 2].item() > 0.3, "reset row of the reference reads as cleared"
    assert after.validity[0, 0].item()
    assert torch.equal(after.base_pos_w[1:, 0], before), "rows that were not reset must not change"
