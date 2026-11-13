import numpy as np

from instinctlab.motion_reference.motion_files.terrain_contact_motion import next_onset_index


def test_next_onset_index():
    flag = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0]], dtype=bool)
    # body 0: onsets at frames 0 and 4; body 1: onset at frame 2; T = 6 where there is none
    expected = np.array([[0, 2], [4, 2], [4, 2], [4, 6], [4, 6], [6, 6]])
    np.testing.assert_array_equal(next_onset_index(flag), expected)


def test_redraw_dropout_keeps_the_draw_until_the_target_moves():
    import torch

    from instinctlab.tasks.shadowing.acp.contact_schedule import redraw_dropout

    torch.manual_seed(0)
    num_envs = 4000
    target = torch.zeros(num_envs, 2, 3)
    dropped = redraw_dropout(torch.zeros(num_envs, 2, dtype=torch.bool), target, torch.full_like(target, float("nan")), 0.25, 0.01)
    assert abs(dropped.float().mean().item() - 0.25) < 0.02  # first step: every target is drawn
    # the same targets (moved by less than 1 cm): the draw is kept
    assert torch.equal(redraw_dropout(dropped, target + 0.005, target, 0.25, 0.01), dropped)
    # body 0 gets new targets: redrawn there, kept for body 1
    moved = target.clone()
    moved[:, 0, 0] += 0.3
    again = redraw_dropout(dropped, moved, target, 0.25, 0.01)
    assert torch.equal(again[:, 1], dropped[:, 1])
    assert abs(again[:, 0].float().mean().item() - 0.25) < 0.02
    assert not torch.equal(again[:, 0], dropped[:, 0])
