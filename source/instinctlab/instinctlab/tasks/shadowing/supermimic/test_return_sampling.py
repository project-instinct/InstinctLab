"""Return-based reset sampling (paper App. D.1) against a hand computation, on a stub environment."""

import torch
from types import SimpleNamespace

from instinctlab.envs.mdp.events.return_sampling import return_based_reset_sampling
from instinctlab.utils.torch import ConcatBatchTensor

GAMMA = 0.99


def _setup():
    # two clips with 2 and 3 one-second bins; env 0 starts in clip 0 bin 1, env 1 in clip 1 bin 2
    weights = ConcatBatchTensor(batch_sizes=[2, 3], data_shape=tuple(), dtype=torch.float, device="cpu")
    weights.fill_data(torch.tensor(0.2))
    buffer = SimpleNamespace(
        cfg=SimpleNamespace(env_starting_stub_sampling_strategy="concat_motion_bins", motion_bin_length_s=1.0),
        _motion_bin_weights=weights,
        _assigned_env_motion_selection=torch.tensor([0, 1]),
        _motion_buffer_start_time_s=torch.tensor([1.5, 2.2]),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"motion_reference": SimpleNamespace(motion_buffers={"m": buffer})},
        reward_buf=torch.zeros(2),
        reset_buf=torch.zeros(2, dtype=torch.bool),
        reset_time_outs=torch.zeros(2, dtype=torch.bool),
    )
    term = return_based_reset_sampling(SimpleNamespace(params={}), env)
    return env, buffer, term


def test_bin_returns_and_reset_distribution():
    env, buffer, term = _setup()
    rewards = [0.05, 0.07, 0.04]
    for step, r in enumerate(rewards):
        env.reward_buf = torch.tensor([r, 0.1])
        last = step == len(rewards) - 1
        # env 0 falls on the last step; env 1 runs on
        env.reset_buf = torch.tensor([last, False])
        env.reset_time_outs = torch.tensor([False, False])
        if last:
            # the reset inside the step has already drawn env 0's next start: clip 1, bin 0
            buffer._assigned_env_motion_selection = torch.tensor([1, 1])
            buffer._motion_buffer_start_time_s = torch.tensor([0.3, 2.2])
        term(env, None, gamma=GAMMA)

    fallen = sum(GAMMA**t * r for t, r in enumerate(rewards))  # terminated: no bootstrap
    expected = torch.zeros(5)
    expected[1] = 0.2 * fallen  # clip 0, bin 1
    assert torch.allclose(term.bin_return, expected, atol=1e-6)
    # env 0's new episode is attributed to its new start (global bin 2) from its first step on
    env.reward_buf = torch.tensor([0.0, 0.1])
    env.reset_buf = torch.tensor([False, False])
    term(env, None, gamma=GAMMA)
    assert term.start_bin.tolist() == [2, 4]

    # the distribution: 50% uniform, 40% softmax(-25 G / G_max), 10% running mean of the softmax
    probs = buffer._motion_bin_weights._concatenated_tensor
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert probs[1] < probs[0]  # the bin with a return is sampled less than the untried ones
    assert torch.allclose(probs[[0, 2, 3, 4]], probs[0].expand(4))


def test_truncated_episode_is_bootstrapped():
    env, buffer, term = _setup()
    env.reward_buf = torch.tensor([0.08, 0.08])
    env.reset_buf = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([True, False])
    term(env, None, gamma=GAMMA)
    bootstrap = GAMMA * 0.08 / (1.0 - GAMMA)
    assert torch.isclose(term.bin_return[1], torch.tensor(0.2 * (0.08 + bootstrap)), atol=1e-5)
