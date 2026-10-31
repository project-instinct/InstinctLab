"""InstantTrack's reference-residual policy against the live SuperMimic environment."""

import torch


def test_reference_residual_commands_the_next_reference_pose(env):
    """With a zero decoder output the residual alone commands the reference pose one control step ahead: it
    reads frame 1 of ``joint_pos_ref`` (``q_ref - q_default`` over the goal window t, t+1, ...) and divides
    by the action term's own scale. The joint target is then ``q_ref(t+1)`` plus the action term's offset
    from the nominal default pose, which is the task's joint-calibration randomization."""
    from instinct_rl.modules import build_actor_critic

    from instinctlab.tasks.shadowing.supermimic.config.g1.agents.instanttrack_agent_cfg import G1InstantTrackRunnerCfg
    from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

    wrapped = InstinctRlVecEnvWrapper(env)
    u = env.unwrapped
    policy_cfg = G1InstantTrackRunnerCfg().to_dict()["policy"]
    model = build_actor_critic(
        policy_cfg.pop("class_name"),
        policy_cfg,
        wrapped.get_obs_format(),
        num_actions=wrapped.num_actions,
        num_rewards=1,
    ).to(u.device)
    obs, _ = wrapped.get_observations()

    action_term = u.action_manager.get_term("joint_pos")
    command = u.command_manager.get_term("joint_pos_ref_command")
    target = action_term._offset + action_term._scale * model.reference_action(obs)
    calibration = action_term._offset - command._default_joint_pos
    expected = command._default_joint_pos + command.command[:, 1] + calibration
    assert torch.allclose(target, expected, atol=1e-5), (target - expected).abs().max()
    # and the reference really is the next frame, not the current one
    assert not torch.allclose(command.command[:, 0], command.command[:, 1])
