import math
import torch

import pytest

from instinctlab.tasks.shadowing.acp.hindsight_labeler import (
    HindsightContactLabeler,
    HindsightContactLabelerCfg,
    label_next_contacts,
)
from instinctlab.tasks.shadowing.acp.planner_handoff import linear_ramp

FOOTPRINT = ((-0.8, 0.8), (-0.5, 0.5))


def _trajectory(flags, body_x=None):
    """One slot, one body: contact flags (L,), body x position per step (default 0.1 * step), base at the origin."""
    num_steps = len(flags)
    contact = torch.tensor(flags, dtype=torch.bool).view(1, num_steps, 1)
    x = torch.arange(num_steps, dtype=torch.float) * 0.1 if body_x is None else torch.as_tensor(body_x)
    body_pos = torch.zeros(1, num_steps, 1, 3)
    body_pos[0, :, 0, 0] = x
    return contact, body_pos, torch.zeros(1, num_steps, 3), torch.zeros(1, num_steps)


def test_next_onset_drops_short_segments_and_respects_the_horizon():
    # segment of 2 steps at 2-3 (too short), segment of 4 steps at 6-9, segment of 3 at 12-14
    flags = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    contact, body_pos, root_pos, yaw = _trajectory(flags, body_x=torch.zeros(16))
    body_pos[0, 6, 0, 0], body_pos[0, 12, 0, 0] = 0.6, 0.7
    target, valid = label_next_contacts(contact, body_pos, root_pos, yaw, torch.tensor([16]), 5, 3, FOOTPRINT)
    # steps 1..6 see the onset at 6 (within 5 steps), step 0 does not; 7..11 see 12 (7 is 5 before 12)
    expected_valid = [False] + [True] * 6 + [True] * 6 + [False] * 3
    assert valid[0, :, 0].tolist() == expected_valid
    assert torch.allclose(target[0, 1:7, 0, 0], torch.full((6,), 0.6))
    assert torch.allclose(target[0, 7:13, 0, 0], torch.full((6,), 0.7))
    assert torch.all(target[0, ~valid[0, :, 0]] == 0)
    # with a minimum of 4 steps, the last segment (3 steps) is dropped too
    _, valid = label_next_contacts(contact, body_pos, root_pos, yaw, torch.tensor([16]), 5, 4, FOOTPRINT)
    assert valid[0, :, 0].tolist() == [False] + [True] * 6 + [False] * 9


def test_target_is_in_the_body_yaw_frame_of_each_step_and_inside_the_footprint():
    flags = [0, 0, 0, 1, 1, 1]
    contact, body_pos, root_pos, yaw = _trajectory(flags, body_x=torch.zeros(6))
    body_pos[0, 3, 0] = torch.tensor([1.0, 0.3, 0.05])
    root_pos[0, :, 0] = torch.tensor([0.0, 0.5, 0.8, 0.9, 0.9, 0.9])  # the base walks towards the contact
    yaw[0, 2] = math.pi / 2  # at step 2 the robot faces +y
    target, valid = label_next_contacts(contact, body_pos, root_pos, yaw, torch.tensor([6]), 10, 2, FOOTPRINT)
    # step 0: 1.0 m ahead (outside the 0.8 m footprint); step 1: (0.5, 0.3); step 2, facing +y: (0.3, -0.2)
    assert valid[0, :4, 0].tolist() == [False, True, True, True]
    assert torch.allclose(target[0, 1, 0], torch.tensor([0.5, 0.3, 0.05]))
    assert torch.allclose(target[0, 2, 0], torch.tensor([0.3, -0.2, 0.05]), atol=1e-6)
    # beyond the trajectory length nothing is valid
    _, valid = label_next_contacts(contact, body_pos, root_pos, yaw, torch.tensor([3]), 10, 2, FOOTPRINT)
    assert not valid.any()


def _labeler(num_envs=4, num_slots=2, max_steps=40, horizon_s=0.2, min_contact_steps=3):
    cfg = HindsightContactLabelerCfg(
        body_names=["a", "b"],
        num_slots=num_slots,
        max_recorded_steps=max_steps,
        horizon_s=horizon_s,
        min_contact_steps=min_contact_steps,
    )
    return HindsightContactLabeler(cfg, num_envs, obs_size=3, step_dt=0.02, device="cpu")


def _record_steps(labeler, num_steps, num_envs=4, force=None):
    """Record ``num_steps`` steps; body 0 of every env pushes 5 N from step 12 on (body 1 only 0.5 N)."""
    for step in range(num_steps):
        obs = torch.stack([torch.full((3,), float(step))] * num_envs) + torch.arange(num_envs).view(-1, 1) * 100.0
        force_z = torch.zeros(num_envs, 2)
        if step >= 12:
            force_z[:, 0], force_z[:, 1] = -5.0, 0.5
        body_pos = torch.zeros(num_envs, 2, 3)
        body_pos[:, :, 0] = 0.3
        labeler.record(obs, force_z, body_pos, torch.zeros(num_envs, 3), torch.zeros(num_envs))


def test_admission_force_threshold_and_samples():
    labeler = _labeler()
    envs = torch.arange(4)
    # env 2 starts at phase 0.5: never recorded; env 3 is eligible but there are only 2 slots
    labeler.start_episodes(envs, torch.tensor([0.0, 0.05, 0.5, 0.1]))
    assert labeler.num_recording == 2 and labeler.stats["unrecorded_episodes"] == 1
    _record_steps(labeler, 20)
    # env 0 reached 0.95 of its clip, env 1 only 0.5
    labeler.finish_episodes(torch.tensor([0, 1]), torch.tensor([20, 20]), torch.tensor([0.95, 0.5]))
    assert labeler.stats["admitted_episodes"] == 1 and labeler.stats["rejected_episodes"] == 1
    assert labeler.num_recording == 0
    samples = labeler.pop_samples()
    assert samples["planner_obs"].shape == (20, 3) and samples["target_pos"].shape == (20, 2, 3)
    assert torch.equal(samples["planner_obs"][:, 0], torch.arange(20.0))  # env 0's observations, in order
    # body 0: onset at step 12, horizon 10 steps -> steps 2..12 have a target; body 1 never exceeds 1 N
    assert samples["target_valid"][:, 0].tolist() == [False] * 2 + [True] * 11 + [False] * 7
    assert not samples["target_valid"][:, 1].any()
    assert torch.allclose(samples["target_pos"][2, 0], torch.tensor([0.3, 0.0, 0.0]))
    assert labeler.pop_samples()["planner_obs"].shape == (0, 3)


def test_truncated_record_drops_the_unlabelable_tail():
    labeler = _labeler(num_slots=1, max_steps=30)
    labeler.start_episodes(torch.tensor([0]), torch.tensor([0.0]))
    _record_steps(labeler, 45)  # the record stops at 30 steps
    labeler.finish_episodes(torch.tensor([0]), torch.tensor([45]), torch.tensor([1.0]))
    samples = labeler.pop_samples()
    # 30 recorded steps minus horizon (10) and minimum segment (3) plus one
    assert samples["planner_obs"].shape[0] == 30 - 10 - 3 + 1


def test_slot_is_reused_after_an_episode():
    labeler = _labeler(num_slots=1)
    labeler.start_episodes(torch.tensor([0]), torch.tensor([0.0]))
    _record_steps(labeler, 5)
    labeler.finish_episodes(torch.tensor([0]), torch.tensor([5]), torch.tensor([0.2]))
    labeler.start_episodes(torch.tensor([1]), torch.tensor([0.0]))
    assert labeler.slot_env.tolist() == [1] and int(labeler.slot_length[0]) == 0


def test_linear_ramp():
    assert [linear_ramp(u, 10, 20) for u in (0, 10, 20, 30, 40)] == [0.0, 0.0, 0.5, 1.0, 1.0]
    with pytest.raises(ValueError):
        linear_ramp(0, 0, 0)
