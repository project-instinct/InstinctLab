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

import numpy as np
import torch

from isaaclab.utils.noise import UniformNoiseCfg

from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import SuperMimicImagination

#: Observation group carrying the world-model state.
WM_STATE_GROUP = "wm_state"
#: Observation group carrying the simulation context (randomized physics parameters).
SIM_CONTEXT_GROUP = "sim_context"


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
        # The *nominal* default pose, in the policy's joint order to match the ``wm_state`` group.
        # Deliberately not the live ``default_joint_pos``: that carries each environment's randomized
        # calibration offset, and the imagination builds one observation for a batch drawn from many
        # envs. Using a single env's calibrated pose would bias every other env's ``joint_pos``
        # observation by up to the offset. The offset is modelled explicitly instead, through the
        # ``sim_context`` group.
        default_joint_pos=_nominal_default_joint_pos(robot, joint_pos_cmd._asset_joint_ids),
        policy_term_order=list(observation_manager.active_terms["policy"]),
        critic_term_order=list(observation_manager.active_terms["critic"]),
        # Two index spaces: the reference buffer is in the motion's joint order, the robot in the
        # policy's. The goal terms select the buffer with ``reference_joint_ids`` and the default pose
        # with ``policy_joint_ids``.
        reference_joint_ids=joint_pos_cmd._motion_reference_joint_ids,
        policy_joint_ids=joint_pos_cmd._asset_joint_ids,
        critic_link_ids=None,
        context_format=_context_format(obs_format),
        goal_window=motion_reference.cfg.num_frames,
        # Read the history layout off the observation terms rather than restating it: the manager is
        # the authority on both which terms carry history and how many frames, and the imagined
        # observation has to reproduce that arrangement exactly.
        history_terms=_history_terms(observation_manager, "policy"),
        history_length=_history_length(observation_manager, "policy"),
        policy_noise_halfwidth=_noise_halfwidth(observation_manager, "policy"),
        motion_reference=motion_reference,
    )


def _noise_halfwidth(observation_manager, group: str) -> torch.Tensor | None:
    """Per-dimension half-width of the group's uniform sensor noise, in the flat observation layout.

    Read from the term configs so the imagined noise is the noise the environment applies. Only
    zero-mean additive uniform noise is reproduced; any other noise model raises, because imagining it
    wrongly would train the policy on a different input distribution than it is deployed with.
    """
    if not observation_manager.cfg.__dict__[group].enable_corruption:
        return None
    widths = []
    for cfg, dim in zip(observation_manager._group_obs_term_cfgs[group], observation_manager.group_obs_term_dim[group]):
        size = int(np.prod(dim))
        noise = cfg.noise
        if noise is None:
            widths.append(torch.zeros(size))
            continue
        if not isinstance(noise, UniformNoiseCfg) or noise.operation != "add":
            raise NotImplementedError(f"cannot imagine noise {type(noise).__name__} ({noise.operation})")
        n_min, n_max = float(noise.n_min), float(noise.n_max)
        if abs(n_min + n_max) > 1e-9:
            raise NotImplementedError(f"cannot imagine non-zero-mean noise U[{n_min}, {n_max}]")
        widths.append(torch.full((size,), n_max))
    return torch.cat(widths)


def _nominal_default_joint_pos(robot, joint_ids) -> torch.Tensor:
    """The un-randomized default joint pose, reindexed into the policy's joint order."""
    nominal = getattr(robot.data, "default_joint_pos_nominal", None)
    pose = robot.data.default_joint_pos.torch[0] if nominal is None else nominal
    return pose[joint_ids]


def _context_format(obs_format) -> dict:
    """The simulation-context layout, or an empty layout when the task supplies none.

    Read from the observation group rather than restated, so the algorithm's context vector and the
    group the env collects cannot drift apart.
    """
    return dict(obs_format.get(SIM_CONTEXT_GROUP, {}))


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
            "observation stacks a single window, so they must agree"
        )
    return lengths.pop()
