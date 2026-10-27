"""Provider for the algorithm inputs the SuperMimic task supplies at runtime.

Referenced by name from the agent config::

    algorithm.inputs = {"imagination": "instinctlab.tasks.shadowing.supermimic.supermimic_provider:build_imagination"}

The indirection exists because ``SuperMimicImagination`` can only be built once the simulator is running
-- it needs the live motion-reference manager, the articulation's soft joint limits and default pose, and
the observation manager's term order. It therefore cannot be written into the YAML config that the runner
resolves before launch.

The layout is *not* specified here: it comes from the ``wm_state`` observation group in ``obs_format``, so
the term names are the world-model state component names and there is no second vocabulary to keep in
sync. This module only turns those into a differentiable view of the task's MDP.
"""

from __future__ import annotations

from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import SuperMimicImagination

#: Observation group carrying the world-model state.
WM_STATE_GROUP = "wm_state"
#: Observation group carrying the reference frame index. ``ref_index`` is that group's single term.
REF_INDEX_GROUP = "ref_index"
#: Extra replay key the algorithm stores alongside each transition.
REF_INDEX_KEY = "ref_index"


def build_imagination(env, obs_format) -> SuperMimicImagination:
    """Build the differentiable view of the task's MDP.

    Args:
        env: The vectorised environment the runner is driving.
        obs_format: ``{group: {term: shape}}`` for every observation group.

    Returns:
        An object satisfying ``instinct_rl.env.imagination.ImaginationInterface``.
    """
    if WM_STATE_GROUP not in obs_format:
        raise KeyError(
            f"the SuperMimic agent needs a {WM_STATE_GROUP!r} observation group; "
            f"available groups are {sorted(obs_format)}"
        )

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    motion_reference = unwrapped.scene["motion_reference"]
    observation_manager = unwrapped.observation_manager

    # The policy's reference commands select joints by name; reuse the command term's own selection so
    # the goal terms match what the policy was trained against.
    joint_pos_cmd = unwrapped.command_manager.get_term("joint_pos_ref_command")

    # No link re-selection is needed: the ``wm_state`` group selects the tracked bodies with
    # ``preserve_order=True`` using the same body-name list as the critic's link terms, so the state's
    # link axis is already in the order the critic observation expects. (The critic's ``body_ids`` are
    # *robot* body indices -- up to 29 -- and applying them again would index past a 14-wide axis.)

    return SuperMimicImagination(
        state_format=dict(obs_format[WM_STATE_GROUP]),
        soft_joint_pos_limits=robot.data.soft_joint_pos_limits.torch[0],
        # default pose in the policy's joint order, matching the ``wm_state`` group's joint order
        default_joint_pos=robot.data.default_joint_pos.torch[0][joint_pos_cmd._asset_joint_ids],
        policy_term_order=list(observation_manager.active_terms["policy"]),
        critic_term_order=list(observation_manager.active_terms["critic"]),
        # Two index spaces: the reference buffer is in the motion's joint order, the robot in the
        # policy's. The goal terms select the buffer with ``reference_joint_ids`` and the default pose
        # with ``policy_joint_ids``.
        reference_joint_ids=joint_pos_cmd._motion_reference_joint_ids,
        policy_joint_ids=joint_pos_cmd._asset_joint_ids,
        critic_link_ids=None,
        goal_window=motion_reference.cfg.num_frames,
        # Read the history layout off the observation terms rather than restating it: the manager is
        # the authority on both which terms carry history and how many frames, and the imagined
        # observation has to reproduce that arrangement exactly.
        history_terms=_history_terms(observation_manager, "policy"),
        history_length=_history_length(observation_manager, "policy"),
        motion_reference=motion_reference,
    )


def _term_cfgs(observation_manager, group: str):
    """The resolved ``ObservationTermCfg`` objects of a group, in ``active_terms`` order."""
    return observation_manager._group_obs_term_cfgs[group]


def _history_terms(observation_manager, group: str) -> tuple[str, ...]:
    """Names of the terms in *group* that keep a multi-frame history."""
    names = observation_manager.active_terms[group]
    cfgs = _term_cfgs(observation_manager, group)
    return tuple(name for name, cfg in zip(names, cfgs) if int(cfg.history_length or 0) > 0)


def _history_length(observation_manager, group: str) -> int:
    """The history length the group's terms share, or 0 when none carry history.

    Raises if terms disagree: the imagination builds one stacked window for all of them, so mixed
    lengths would silently reshape the observation.
    """
    names = observation_manager.active_terms[group]
    cfgs = _term_cfgs(observation_manager, group)
    lengths = {int(cfg.history_length or 0) for cfg in cfgs}
    lengths.discard(0)
    if not lengths:
        return 0
    if len(lengths) > 1:
        per_term = {
            name: int(cfg.history_length or 0) for name, cfg in zip(names, cfgs) if int(cfg.history_length or 0) > 0
        }
        raise ValueError(
            f"terms of the {group!r} group use different history lengths {per_term}; the imagined "
            f"observation stacks a single window, so they must agree"
        )
    return lengths.pop()
