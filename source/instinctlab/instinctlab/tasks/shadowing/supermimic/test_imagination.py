"""Numeric and gradient checks for the SuperMimic differentiable reward cores.

The cores are a transcription of the manager reward terms, so the thing that matters is that they
compute the same numbers and that gradients flow. The parity reference here is the manager formula
re-implemented from ``instinctlab.envs.mdp.rewards`` on the same inputs, which catches transcription
errors (wrong sign, wrong frame, wrong reduction) without needing a simulator.

Run inside the container:
    docker compose --env-file .env.base run --rm --no-deps instinctlab -c \
        "cd /home/leo/workspace/instinctlab_ws/instinctlab && python -m pytest \
         source/instinctlab/instinctlab/tasks/shadowing/supermimic/test_imagination.py -x -q"
"""

import pytest
import torch

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, yaw_quat, quat_inv

from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import (
    REWARD_TERMS,
    ReferenceWindow,
    SuperMimicImagination,
    body_angular_velocity_reward,
    body_linear_velocity_reward,
    body_orientation_reward,
    body_position_reward,
    joint_limit_penalty,
    make_state_format,
    root_orientation_reward,
    root_position_reward,
    termination_prob,
    unpack_state,
)

B = 4
NUM_LINKS = 3
NUM_JOINTS = 5


def _rand_unit_quat(B, N=None, seed=None):
    g = torch.Generator().manual_seed(seed if seed is not None else 0)
    q = torch.randn((B, 4) if N is None else (B, N, 4), generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def _window(B, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (B, L)
    return ReferenceWindow(
        base_pos_w=torch.randn(B, L, 3, generator=g),
        base_quat_w=_rand_unit_quat(B, L, seed + 1),
        link_pos_w=torch.randn(B, L, NUM_LINKS, 3, generator=g),
        link_quat_w=_rand_unit_quat(B, L * NUM_LINKS, seed + 2).reshape(B, L, NUM_LINKS, 4),
        link_lin_vel_w=torch.randn(B, L, NUM_LINKS, 3, generator=g),
        link_ang_vel_w=torch.randn(B, L, NUM_LINKS, 3, generator=g),
        validity=torch.ones(B, L, dtype=torch.bool),
        joint_pos=torch.randn(B, L, NUM_JOINTS, generator=g),
        joint_vel=torch.randn(B, L, NUM_JOINTS, generator=g),
        joint_pos_mask=torch.ones(B, L, NUM_JOINTS, dtype=torch.bool),
        joint_vel_mask=torch.ones(B, L, NUM_JOINTS, dtype=torch.bool),
        base_pos_plane_mask=torch.ones(B, L, dtype=torch.bool),
        base_pos_height_mask=torch.ones(B, L, dtype=torch.bool),
    )


# --- parity against the manager formulas -----------------------------------------------------------


def test_body_position_matches_manager_formula():
    """``link_pos_imitation_gauss`` with in_relative_world_frame=True, combine mean_prod."""
    ref = _window(B, 1)
    root_pos = torch.randn(B, 3)
    root_quat = _rand_unit_quat(B)
    link_pos = torch.randn(B, NUM_LINKS, 3)
    std = 0.3

    # manager: base_pos = robot x-y with reference z; delta yaw between robot and reference base
    base_pos = root_pos.clone()
    base_pos[:, 2] = ref.base_pos_w[:, 0, 2]
    delta_quat = yaw_quat(quat_mul(root_quat, quat_inv(ref.base_quat_w[:, 0])))

    from isaaclab.utils.math import quat_apply

    ref_pos = base_pos.unsqueeze(1) + quat_apply(
        delta_quat.unsqueeze(1).expand(-1, NUM_LINKS, -1),
        ref.link_pos_w[:, 0] - ref.base_pos_w[:, 0].unsqueeze(1),
    )
    sq = torch.sum(torch.square(link_pos - ref_pos), dim=-1)
    expected = torch.exp(-torch.mean(sq, dim=-1) / (std * std))

    got = body_position_reward(link_pos, root_pos, root_quat, ref, std=std)
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)


def test_body_orientation_matches_manager_formula():
    ref = _window(B, 1)
    root_pos = torch.randn(B, 3)
    root_quat = _rand_unit_quat(B)
    link_quat = _rand_unit_quat(B, NUM_LINKS, seed=11)
    std = 0.4

    delta_quat = yaw_quat(quat_mul(root_quat, quat_inv(ref.base_quat_w[:, 0])))
    ref_quat = quat_mul(delta_quat.unsqueeze(1).expand(-1, NUM_LINKS, -1), ref.link_quat_w[:, 0])
    diff = quat_mul(ref_quat, quat_conjugate(link_quat))
    angle = axis_angle_from_quat(diff.reshape(-1, 4)).norm(dim=-1).reshape(B, NUM_LINKS)
    expected = torch.exp(-torch.mean(torch.square(angle), dim=-1) / (std * std))

    got = body_orientation_reward(link_quat, root_pos, root_quat, ref, std=std)
    assert torch.allclose(got, expected, atol=1e-5), (got, expected)


def test_velocity_terms_match_manager_formula():
    ref = _window(B, 1)
    v = torch.randn(B, NUM_LINKS, 3)
    w = torch.randn(B, NUM_LINKS, 3)

    sq_v = torch.sum(torch.square(v - ref.link_lin_vel_w[:, 0]), dim=-1)
    expected_v = torch.exp(-torch.mean(sq_v, dim=-1) / 1.0**2)
    assert torch.allclose(body_linear_velocity_reward(v, ref), expected_v, atol=1e-6)

    sq_w = torch.sum(torch.square(w - ref.link_ang_vel_w[:, 0]), dim=-1)
    expected_w = torch.exp(-torch.mean(sq_w, dim=-1) / 3.14**2)
    assert torch.allclose(body_angular_velocity_reward(w, ref), expected_w, atol=1e-6)


def test_root_terms_match_and_respect_validity():
    ref = _window(B, 1)
    ref.validity[1, 0] = False
    root_pos = torch.randn(B, 3)
    root_quat = _rand_unit_quat(B, seed=5)

    sq = torch.sum(torch.square(root_pos - ref.base_pos_w[:, 0]), dim=-1)
    expected_pos = torch.exp(-sq / 0.3**2) * ref.validity[:, 0].float()
    assert torch.allclose(root_position_reward(root_pos, ref), expected_pos, atol=1e-6)
    # a masked reference contributes no reward
    assert root_position_reward(root_pos, ref)[1].item() == pytest.approx(0.0)

    diff = quat_mul(ref.base_quat_w[:, 0], quat_conjugate(root_quat))
    angle = axis_angle_from_quat(diff).norm(dim=-1)
    expected_rot = torch.exp(-torch.square(angle) / 0.4**2)
    assert torch.allclose(root_orientation_reward(root_quat, ref), expected_rot, atol=1e-5)


def test_joint_limit_matches_manager_sign_convention():
    """Manager: -(q - lower).clip(max=0) + (q - upper).clip(min=0), summed over joints."""
    lower = torch.full((NUM_JOINTS,), -1.0)
    upper = torch.full((NUM_JOINTS,), 1.0)
    limits = torch.stack([lower, upper], dim=-1)
    q = torch.tensor([[0.0, 1.5, -1.5, 0.5, -0.5]])

    expected = -(q - lower).clip(max=0.0) + (q - upper).clip(min=0.0)
    got = joint_limit_penalty(q, limits)
    assert torch.allclose(got, expected.sum(dim=-1), atol=1e-6)
    # only the two violations count, 0.5 each
    assert got.item() == pytest.approx(1.0)


# --- gradient flow ---------------------------------------------------------------------------------


def test_state_gradients_reach_every_term():
    """BPTT is the whole point: d(reward)/d(state) must be non-zero and finite."""
    ref = _window(B, 1)
    fmt = make_state_format(NUM_LINKS, NUM_JOINTS)
    state = torch.randn(B, sum(int(torch.tensor(s).prod()) for s in fmt.values()), requires_grad=True)
    action = torch.randn(B, NUM_JOINTS, requires_grad=True)
    prev_action = torch.randn(B, NUM_JOINTS)

    imag = SuperMimicImagination(
        state_format=make_state_format(NUM_LINKS, NUM_JOINTS),
        soft_joint_pos_limits=torch.stack([torch.full((NUM_JOINTS,), -1.0), torch.full((NUM_JOINTS,), 1.0)], dim=-1),
        default_joint_pos=torch.zeros(NUM_JOINTS),
        policy_term_order=["joint_pos"],
        critic_term_order=["joint_pos"],
        history_length=10,
        ref_window=ref,
    )
    ref_index = torch.zeros(B, dtype=torch.long)
    r = imag.reward(state, action, prev_action, ref_index)
    assert r.shape == (B, len(REWARD_TERMS))

    r.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    assert action.grad is not None and action.grad.abs().sum() > 0


def test_termination_is_soft_and_orders_correctly():
    ref = _window(B, 1)
    root_quat = ref.base_quat_w[:, 0].clone()
    # on-reference root: no termination
    p_ok = termination_prob(ref.base_pos_w[:, 0], root_quat, ref)
    assert torch.all(p_ok < 0.5)

    # 1 m too high: terminate
    high = ref.base_pos_w[:, 0].clone()
    high[:, 2] += 1.0
    p_bad = termination_prob(high, root_quat, ref)
    assert torch.all(p_bad > 0.5)

    # soft: strictly inside (0, 1) and differentiable
    near = ref.base_pos_w[:, 0].clone()
    near[:, 2] += 0.255
    state = near.clone().requires_grad_(True)
    p = termination_prob(state, root_quat, ref)
    assert ((p > 0) & (p < 1)).all(), p
    p.sum().backward()
    assert state.grad.abs().sum() > 0


def test_unpack_state_roundtrip():
    fmt = make_state_format(NUM_LINKS, NUM_JOINTS)
    flat = torch.arange(sum(int(torch.tensor(s).prod()) for s in fmt.values()), dtype=torch.float).unsqueeze(0)
    parts = unpack_state(flat, fmt)
    rebuilt = torch.cat([parts[n].reshape(1, -1) for n in fmt], dim=-1)
    assert torch.equal(rebuilt, flat)
    assert parts["body_link_pos_w"].shape == (1, NUM_LINKS, 3)
    assert parts["joint_pos"].shape == (1, NUM_JOINTS)


def test_at_selects_the_right_frame():
    """Column offsets pick per-env frames; past the end the last column is clamped, not wrapped."""
    ref = _window(B, 4)
    idx = torch.tensor([0, 1, 2, 3])
    one = ref.at(idx)
    assert torch.allclose(one.base_pos_w[:, 0], ref.base_pos_w[torch.arange(B), idx], atol=1e-6)
    assert one.link_quat_w.shape[-1] == 4, "quaternion components must survive selection"
    past = ref.at(torch.full((B,), 99, dtype=torch.long))
    assert torch.allclose(past.base_pos_w[:, 0], ref.base_pos_w[:, -1], atol=1e-6)
