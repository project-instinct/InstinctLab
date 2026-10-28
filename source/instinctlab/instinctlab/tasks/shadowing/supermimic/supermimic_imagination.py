"""Differentiable reward/termination cores for SuperMimic, and the imagination interface.

SuperMimic's policy gradient is computed by backpropagating an imagined return through a learned
world model (arXiv:2610.28480v1, Sec. 3.3). That needs the task's MDP expressed as plain torch over
state tensors, with no ``env`` object, no articulation and no manager. This module provides that.

**Why the formulas live here and not in the shared reward module.** The manager-based terms in
``instinctlab.envs.mdp.rewards`` read ``env.scene[...]`` and the motion-reference manager directly, so
they physically cannot run during imagination. Rewriting them generically would mean re-expressing
every task's reward in a config DSL. Instead the eight terms SuperMimic actually uses are written here
as pure functions of ``(state, reference, action)``, and a parity test in the task's test module
asserts they agree with the manager terms to ~1e-5. That test is what prevents the two definitions
from drifting; the formulas are transcribed from the manager implementations line by line.

**State layout uses ``ArticulationData`` names.** Component names match the attributes the reward
terms already read (``root_pos_w``, ``body_link_pos_w``, ...) so there is no parallel vocabulary and
the existing ``get_obs_slice`` helpers apply. Frames are declared per component.

**The reference is looked up, never predicted.** Reference targets are stored per frame, so the
targets for steps t..t+H are buffer rows. The world model predicts robot state only. Note that under
``in_relative_world_frame`` the target is re-anchored to the *predicted* robot pose each step
(``_relative_world_anchor``), so all imagination error comes from state prediction, not the reference.

Only the eight terms of the paper's Table A.2 are implemented. BeyondMimic's self-collision penalty
is absent by design -- the paper removes it because deterministic imagined rollouts make its exploit
unprofitable (App. A), which is also what makes every term here a pure function of state.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import OrderedDict, Sequence

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    transform_points,
    yaw_quat,
)

from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_rewards_cfg import (
    PAPER_REWARD_STDS,
    PAPER_REWARD_WEIGHTS,
)
from instinctlab.utils.math import quat_to_tan_norm

# ---------------------------------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------------------------------

#: Per-link components, whose shape depends on the task's ``link_of_interests``.
LINK_COMPONENTS = ("body_link_pos_w", "body_link_quat_w", "body_link_lin_vel_w", "body_link_ang_vel_w")
LINK_COMPONENT_DIMS = {"body_link_pos_w": 3, "body_link_quat_w": 4, "body_link_lin_vel_w": 3, "body_link_ang_vel_w": 3}

#: Order of the reward vector produced by :meth:`SuperMimicImagination.reward`. Matches the reward
#: manager's term order so a parity test can compare term by term.
REWARD_TERMS = (
    "body_position",
    "body_orientation",
    "body_linear_velocity",
    "body_angular_velocity",
    "root_position",
    "root_orientation",
    "action_smoothness",
    "joint_limit",
)


def make_state_format(num_links: int, num_joints: int) -> OrderedDict[str, tuple]:
    """World-model state layout.

    Component names mirror ``ArticulationData`` attributes, so the reward formulas below read the same
    quantities the manager terms do and no parallel vocabulary is introduced. Joint and link counts
    come from the task's own configuration rather than being written in.
    """
    fmt = OrderedDict(
        [
            ("root_pos_w", (3,)),
            ("root_quat_w", (4,)),
            ("root_lin_vel_w", (3,)),
            ("root_ang_vel_w", (3,)),
            ("joint_pos", (num_joints,)),
            ("joint_vel", (num_joints,)),
        ]
    )
    for name in LINK_COMPONENTS:
        fmt[name] = (num_links, LINK_COMPONENT_DIMS[name])
    return fmt


def state_size(fmt: OrderedDict[str, tuple]) -> int:
    total = 0
    for shape in fmt.values():
        n = 1
        for d in shape:
            n *= int(d)
        total += n
    return total


def _seg(fmt: OrderedDict[str, tuple], name: str) -> slice:
    start = 0
    for key, shape in fmt.items():
        n = 1
        for d in shape:
            n *= int(d)
        if key == name:
            return slice(start, start + n)
        start += n
    raise KeyError(f"{name} not in state format; have {list(fmt)}")


def unpack_state(state: torch.Tensor, fmt: OrderedDict[str, tuple]) -> dict[str, torch.Tensor]:
    """Split a flat ``(B, state_size)`` state into named tensors."""
    return {name: state[..., _seg(fmt, name)].reshape(state.shape[0], *shape) for name, shape in fmt.items()}


# ---------------------------------------------------------------------------------------------------
# Reference window
# ---------------------------------------------------------------------------------------------------


@dataclass
class ReferenceWindow:
    """Reference rows for one rollout: column ``j`` is the reference at ``t + j * control_dt``.

    A snapshot rather than a live lookup: the motion-reference manager memoises against the simulator
    timestamp, which does not advance during imagination.
    """

    base_pos_w: torch.Tensor  # (B, L, 3)
    base_quat_w: torch.Tensor  # (B, L, 4)
    link_pos_w: torch.Tensor  # (B, L, num_links, 3)
    link_quat_w: torch.Tensor  # (B, L, num_links, 4)
    link_lin_vel_w: torch.Tensor  # (B, L, num_links, 3)
    link_ang_vel_w: torch.Tensor  # (B, L, num_links, 3)
    validity: torch.Tensor  # (B, L) bool
    joint_pos: torch.Tensor  # (B, L, num_joints)
    joint_vel: torch.Tensor  # (B, L, num_joints)
    joint_pos_mask: torch.Tensor  # (B, L, num_joints) bool
    joint_vel_mask: torch.Tensor  # (B, L, num_joints) bool
    base_pos_plane_mask: torch.Tensor  # (B, L) bool
    base_pos_height_mask: torch.Tensor  # (B, L) bool

    _FIELD_NAMES = (
        "base_pos_w",
        "base_quat_w",
        "link_pos_w",
        "link_quat_w",
        "link_lin_vel_w",
        "link_ang_vel_w",
        "validity",
        "joint_pos",
        "joint_vel",
        "joint_pos_mask",
        "joint_vel_mask",
        "base_pos_plane_mask",
        "base_pos_height_mask",
    )

    @property
    def horizon(self) -> int:
        return self.base_pos_w.shape[1]

    def at(self, ref_index: torch.Tensor | int) -> ReferenceWindow:
        """One step per env, where ``ref_index`` is a *column offset* into this window.

        Column ``j`` is the reference at ``t + j * control_dt``, so ``ref_index`` starts at 0 and
        :meth:`SuperMimicImagination.advance` adds one per imagined step. Past the end the last column
        is clamped; ``validity`` records that the clip ran out.

        An ``int`` is one column for every row -- the case of a batch of rollouts started together --
        and is a slice rather than a gather.
        """
        if isinstance(ref_index, int):
            return self.columns(ref_index, 1)
        offset = ref_index.clamp(0, self.horizon - 1)

        def pick(t: torch.Tensor) -> torch.Tensor:
            # one index per env, broadcast over the remaining (link, component) dims. Building the
            # index from each tensor's own shape keeps quaternion fields at 4 components.
            idx = offset.reshape(-1, 1, *([1] * (t.dim() - 2))).expand(-1, 1, *t.shape[2:])
            return torch.gather(t, 1, idx)

        return ReferenceWindow(
            base_pos_w=pick(self.base_pos_w),
            base_quat_w=pick(self.base_quat_w),
            link_pos_w=pick(self.link_pos_w),
            link_quat_w=pick(self.link_quat_w),
            link_lin_vel_w=pick(self.link_lin_vel_w),
            link_ang_vel_w=pick(self.link_ang_vel_w),
            validity=pick(self.validity),
            joint_pos=pick(self.joint_pos),
            joint_vel=pick(self.joint_vel),
            joint_pos_mask=pick(self.joint_pos_mask),
            joint_vel_mask=pick(self.joint_vel_mask),
            base_pos_plane_mask=pick(self.base_pos_plane_mask),
            base_pos_height_mask=pick(self.base_pos_height_mask),
        )

    def columns(self, start: int, length: int) -> ReferenceWindow:
        """Columns ``start .. start + length - 1`` for every row, the last column repeated past the end."""
        if start >= 0 and start + length <= self.horizon:
            return ReferenceWindow(
                **{name: getattr(self, name)[:, start : start + length] for name in self._FIELD_NAMES}
            )
        idx = torch.arange(start, start + length, device=self.base_pos_w.device).clamp(0, self.horizon - 1)
        return ReferenceWindow(**{name: getattr(self, name).index_select(1, idx) for name in self._FIELD_NAMES})

    def rows(self, index: torch.Tensor) -> ReferenceWindow:
        """Select one row per output sample, keeping every frame column."""
        return ReferenceWindow(**{name: getattr(self, name)[index] for name in self._FIELD_NAMES})

    def goal_window(self, ref_index: torch.Tensor, length: int) -> ReferenceWindow:
        """Columns ``ref_index .. ref_index+length-1``: the goal encoder's K-frame sequence.

        Unrolled into ``length`` single-frame windows rather than one tensor, so the reward cores and
        the observation assembly keep working on the same single-frame type they already handle.
        """
        return [self.at(ref_index + k) for k in range(length)]


def build_reference_window(manager, horizon: int, scratch=None) -> ReferenceWindow:
    """Sample reference frames ``t .. t+horizon`` from a motion-reference manager.

    The manager's ``reference_frame`` only exposes the current frame, and its passive properties are
    memoised against the simulator timestamp -- neither is usable inside an imagined rollout. This
    samples the underlying motion buffers directly at ``reference_time_s + j * frame_interval_s``,
    so column ``j`` is the reference the robot would be tracking ``j`` control steps from now. The
    targets are *looked up*, never predicted; only the robot dynamics are learned.

    Args:
        manager: The scene's ``MotionReferenceManagerBase``.
        horizon: Number of imagined steps ``H``. Returns ``H + 1`` columns, because the critic
            bootstrap reads frame ``H`` as well as the rewards at ``0 .. H-1``.
        scratch: Optional reusable ``MotionReferenceData`` buffer of ``num_frames == horizon + 1``.
            Allocating one per call costs ~90 MB at 4096 envs, so callers that build a window every
            iteration should pass the same buffer back in.

    Returns:
        A :class:`ReferenceWindow` whose column ``j`` is the frame at ``t + j * control_dt``.
    """
    from instinctlab.motion_reference.motion_reference_data import MotionReferenceData

    num_frames = horizon + 1
    if scratch is None:
        scratch = MotionReferenceData.make_empty(
            num_envs=manager.count,
            num_frames=num_frames,
            num_joints=manager.num_joints,
            num_links=manager.num_link_of_interests,
            device=manager.device,
        )
    elif scratch.validity.shape[1] != num_frames:
        raise ValueError(
            f"scratch buffer holds {scratch.validity.shape[1]} frames, need {num_frames} "
            f"(horizon {horizon} + 1); allocate a matching buffer instead of reusing this one"
        )

    # Column j is j control steps ahead. "one_frame_interval" offsets the whole window by one frame,
    # matching what the manager does for the policy's own window.
    start = 1.0 if manager.cfg.data_start_from == "one_frame_interval" else 0.0
    offsets = torch.arange(num_frames, device=manager.device, dtype=torch.float32) + start

    for buffer in manager.motion_buffers.values():
        env_slice = buffer.assigned_env_slice
        env_ids = manager.ALL_INDICES[env_slice]
        if env_ids.numel() == 0:
            continue
        sample_timestamp = manager.reference_time_s[env_ids].unsqueeze(-1) + (
            offsets.unsqueeze(0) * manager.frame_interval_s[env_ids].unsqueeze(-1)
        )
        buffer.fill_motion_data(env_ids, sample_timestamp, manager.env_origins, scratch)

    def _t(name: str) -> torch.Tensor:
        return getattr(scratch, name).torch

    return ReferenceWindow(
        base_pos_w=_t("base_pos_w"),
        base_quat_w=_t("base_quat_w"),
        link_pos_w=_t("link_pos_w"),
        link_quat_w=_t("link_quat_w"),
        link_lin_vel_w=_t("link_lin_vel_w"),
        link_ang_vel_w=_t("link_ang_vel_w"),
        validity=_t("validity"),
        joint_pos=_t("joint_pos"),
        joint_vel=_t("joint_vel"),
        joint_pos_mask=_t("joint_pos_mask"),
        joint_vel_mask=_t("joint_vel_mask"),
        base_pos_plane_mask=_t("base_pos_plane_mask"),
        base_pos_height_mask=_t("base_pos_height_mask"),
    )


# ---------------------------------------------------------------------------------------------------
# Reward cores -- transcribed from the manager terms, differentiable
# ---------------------------------------------------------------------------------------------------


def _kernel(sq_dist: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-sq_dist / (std * std))


def relative_world_anchor(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ref_base_pos_w: torch.Tensor,
    ref_base_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``in_relative_world_frame`` anchoring used by the links terms.

    Mirrors ``MotionReferenceManager.reference_relative_base_pos`` and ``..._delta_quat``: the
    reference is placed at the robot's x-y with the reference height, and yaw-aligned to the robot.
    """
    base_pos = root_pos_w.clone()
    base_pos[..., 2] = ref_base_pos_w[..., 2]
    delta_quat = yaw_quat(quat_mul(root_quat_w, quat_inv(ref_base_quat_w)))
    return base_pos, delta_quat


def anchor_positions(
    ref_pos_w: torch.Tensor,
    base_pos: torch.Tensor,
    delta_quat: torch.Tensor,
    ref_base_pos_w: torch.Tensor,
) -> torch.Tensor:
    """``base_pos + delta_quat * (ref_pos_w - ref_base_pos_w)``, broadcasting over links."""
    local = ref_pos_w - ref_base_pos_w.unsqueeze(-2)
    quat = delta_quat.unsqueeze(-2).expand(*local.shape[:-1], 4)
    return base_pos.unsqueeze(-2) + quat_apply(quat, local)


def anchor_quaternions(
    ref_quat_w: torch.Tensor,
    delta_quat: torch.Tensor,
) -> torch.Tensor:
    """``delta_quat * ref_quat_w``, broadcasting over links.

    Quaternions are rotated, not translated, so the base-position subtraction does not apply --
    contrast :func:`anchor_positions`.
    """
    quat = delta_quat.unsqueeze(-2).expand(*ref_quat_w.shape[:-1], 4)
    return quat_mul(quat, ref_quat_w)


def body_position_reward(
    link_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ref: ReferenceWindow,
    std: float = PAPER_REWARD_STDS["body_position"],
) -> torch.Tensor:
    """``exp(-mean_b ||p_b - p_b^des||^2 / std^2)``; the paper's Body Position term."""
    base_pos, delta_quat = relative_world_anchor(root_pos_w, root_quat_w, ref.base_pos_w[:, 0], ref.base_quat_w[:, 0])
    ref_pos = anchor_positions(ref.link_pos_w[:, 0], base_pos, delta_quat, ref.base_pos_w[:, 0])
    sq = torch.sum(torch.square(link_pos_w - ref_pos), dim=-1)
    return _kernel(torch.mean(sq, dim=-1), std)


def body_orientation_reward(
    link_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ref: ReferenceWindow,
    std: float = PAPER_REWARD_STDS["body_orientation"],
) -> torch.Tensor:
    """``exp(-mean_b ||log(R_b^des R_b^T)||^2 / std^2)``; the paper's Body Orientation term."""
    _, delta_quat = relative_world_anchor(root_pos_w, root_quat_w, ref.base_pos_w[:, 0], ref.base_quat_w[:, 0])
    ref_quat = anchor_quaternions(ref.link_quat_w[:, 0], delta_quat)
    diff = quat_mul(ref_quat, quat_conjugate(link_quat_w))
    angle = torch.norm(axis_angle_from_quat(diff.reshape(-1, 4)), dim=-1).reshape(link_quat_w.shape[:2])
    return _kernel(torch.mean(torch.square(angle), dim=-1), std)


def body_linear_velocity_reward(
    link_lin_vel_w: torch.Tensor,
    ref: ReferenceWindow,
    std: float = PAPER_REWARD_STDS["body_linear_velocity"],
) -> torch.Tensor:
    """``exp(-mean_b ||v_b^des - v_b||^2 / std^2)``; the paper's Body Linear velocity term."""
    sq = torch.sum(torch.square(link_lin_vel_w - ref.link_lin_vel_w[:, 0]), dim=-1)
    return _kernel(torch.mean(sq, dim=-1), std)


def body_angular_velocity_reward(
    link_ang_vel_w: torch.Tensor,
    ref: ReferenceWindow,
    std: float = PAPER_REWARD_STDS["body_angular_velocity"],
) -> torch.Tensor:
    """``exp(-mean_b ||w_b^des - w_b||^2 / std^2)``; the paper's Body Angular velocity term."""
    sq = torch.sum(torch.square(link_ang_vel_w - ref.link_ang_vel_w[:, 0]), dim=-1)
    return _kernel(torch.mean(sq, dim=-1), std)


def root_position_reward(
    root_pos_w: torch.Tensor,
    ref: ReferenceWindow,
    std: float = PAPER_REWARD_STDS["root_position"],
) -> torch.Tensor:
    """``exp(-||p_root^des - p_root||^2 / std^2)``, masked by reference validity."""
    sq = torch.sum(torch.square(root_pos_w - ref.base_pos_w[:, 0]), dim=-1)
    valid = ref.validity[:, 0].to(device=root_pos_w.device, dtype=root_pos_w.dtype)
    return _kernel(sq, std) * valid


def root_orientation_reward(
    root_quat_w: torch.Tensor,
    ref: ReferenceWindow,
    std: float = PAPER_REWARD_STDS["root_orientation"],
) -> torch.Tensor:
    """``exp(-||log(R_root^des R_root^T)||^2 / std^2)``; axis-angle, as BeyondMimic configures it."""
    diff = quat_mul(ref.base_quat_w[:, 0], quat_conjugate(root_quat_w))
    angle = torch.norm(axis_angle_from_quat(diff), dim=-1)
    return _kernel(torch.square(angle), std)


def action_smoothness_penalty(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """``||a_t - a_{t-1}||^2``, i.e. ``mdp.action_rate_l2``."""
    return torch.sum(torch.square(action - prev_action), dim=-1)


def joint_limit_penalty(
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,
) -> torch.Tensor:
    """``sum_j max(0, q_j - q_j^max) + max(0, q_j^min - q_j)``, as ``mdp.joint_pos_limits``.

    ``soft_limits`` is ``(num_joints, 2)`` taken from the articulation's ``soft_joint_pos_limits``;
    it is static, so it is supplied by the adapter rather than stored in the state.
    """
    out_of_limits = -(joint_pos - soft_limits[:, 0]).clip(max=0.0)
    out_of_limits = out_of_limits + (joint_pos - soft_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=-1)


# ---------------------------------------------------------------------------------------------------
# Termination core
# ---------------------------------------------------------------------------------------------------


def root_deviation(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ref: ReferenceWindow,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SuperMimic's two termination errors: root height error and tilt error.

    ``e_p,z`` is the signed height offset; ``e_tilt`` is the absolute difference between the
    z-components of the gravity direction in the reference and robot root frames (App. A:
    ``|e_p,z| > 0.25 m or e_tilt > 0.8``).
    """
    height_error = root_pos_w[..., 2] - ref.base_pos_w[:, 0, 2]
    tilt_error = torch.abs(_gravity_z_in_frame(root_quat_w) - _gravity_z_in_frame(ref.base_quat_w[:, 0]))
    return height_error, tilt_error


def _projected_gravity(quat_w: torch.Tensor) -> torch.Tensor:
    """World gravity ``(0, 0, -1)`` in the frame of ``quat_w`` (x, y, z, w), i.e. ``mdp.projected_gravity``,
    in closed form: the negated third row of the rotation matrix."""
    x, y, z, w = quat_w.unbind(-1)
    return torch.stack(
        (-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), 2.0 * (x * x + y * y) - 1.0),
        dim=-1,
    )


def _gravity_z_in_frame(quat_w: torch.Tensor) -> torch.Tensor:
    """z-component of the world gravity direction ``(0, 0, -1)`` in the frame of ``quat_w`` (x, y, z, w).

    ``quat_apply_inverse(q, (0, 0, -1))[..., 2]`` in closed form, ``2 (x^2 + y^2) - 1``, without
    building the constant vector on every call.
    """
    return 2.0 * (quat_w[..., 0] * quat_w[..., 0] + quat_w[..., 1] * quat_w[..., 1]) - 1.0


def termination_prob(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ref: ReferenceWindow,
    height_threshold: float = 0.25,
    tilt_threshold: float = 0.8,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Soft termination probability, differentiable w.r.t. state.

    The simulator terminates hard; imagination needs a probability in ``[0, 1]`` or the survival
    weight ``c_t`` in the paper's objective (Eq. 2) would be a step function with zero gradient. Each
    threshold is softened with a sigmoid whose temperature is small compared with the thresholds, so
    the value tracks the hard flag closely while remaining differentiable.
    """
    height_error, tilt_error = root_deviation(root_pos_w, root_quat_w, ref)
    p_height = torch.sigmoid((torch.abs(height_error) - height_threshold) / temperature)
    p_tilt = torch.sigmoid((tilt_error - tilt_threshold) / temperature)
    p = torch.maximum(p_height, p_tilt)
    # an invalid reference row terminates, matching the manager's dataset_exhausted term
    exhausted = (~ref.validity[:, 0]).to(device=p.device, dtype=p.dtype)
    p = torch.maximum(p, exhausted)
    return p.unsqueeze(-1)


# ---------------------------------------------------------------------------------------------------
# Imagination interface
# ---------------------------------------------------------------------------------------------------


class SuperMimicImagination:
    """Implements ``instinct_rl.env.imagination.ImaginationInterface`` for the G1 SuperMimic task.

    The class is a thin adapter: it unpacks the flat imagined state into named tensors and calls the
    pure cores above. It holds the reference snapshot for the current rollout, so ``reward`` and
    ``termination_prob`` can index future frames without touching the simulator.

    Observation assembly reads term *names* from the env's observation manager and builds them from
    state, so the imagined policy observation has the same layout as the real one by construction.
    Terms whose value is not derivable from state (e.g. a sensor reading) raise rather than silently
    returning zeros, because a silently wrong observation would corrupt the gradient.
    """

    #: Reward weights and the terms they apply to, from the paper's Table A.2.
    REWARD_WEIGHTS = PAPER_REWARD_WEIGHTS
    REWARD_STDS = PAPER_REWARD_STDS

    def __init__(
        self,
        *,
        state_format: OrderedDict[str, tuple],
        soft_joint_pos_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        policy_term_order: list[str],
        critic_term_order: list[str],
        reference_joint_ids: torch.Tensor | slice | None = None,
        policy_joint_ids: torch.Tensor | slice | None = None,
        critic_link_ids: torch.Tensor | slice | None = None,
        context_format: OrderedDict[str, tuple] | dict[str, tuple] | None = None,
        goal_window: int = 10,
        history_terms: Sequence[str] = (),
        history_length: int = 0,
        policy_noise_halfwidth: torch.Tensor | None = None,
        termination_temperature: float = 0.02,
        motion_reference=None,
        ref_window: ReferenceWindow | None = None,
        step_dt: float = 1.0,
    ) -> None:
        # The layout is supplied by the caller (normally the ``wm_state`` observation group), not
        # derived here: term names are the state component names, so there is one vocabulary.
        self.state_format = OrderedDict(state_format)
        self.state_size = state_size(self.state_format)
        self.num_joints = int(np.prod(self.state_format["joint_pos"])) if "joint_pos" in self.state_format else 0
        self.num_links = int(self.state_format["body_link_pos_w"][0]) if "body_link_pos_w" in self.state_format else 0
        # optional: turns a replayed absolute frame index into a window-relative column offset
        self.motion_reference = motion_reference
        self.soft_joint_pos_limits = soft_joint_pos_limits
        # Two joint orders are in play. The world-model state carries joints in the policy's order, so
        # ``joint_pos`` observations subtract ``_default_joint_pos`` directly. The reference buffer is
        # in the articulation's order, so the goal terms select it through ``joint_ids`` -- exactly what
        # the manager's reference commands do.
        self._default_joint_pos = default_joint_pos
        # ``joint_ids[j]`` is the policy-order position of reference joint ``j``. With no permutation the
        # two orders coincide and the default pose needs no reindexing.
        # Either a permutation (tensor of policy-order indices) or a pass-through ``slice`` when the
        # articulation's order already matches; both index the reference buffer the same way.
        # ``reference_joint_ids`` indexes the reference buffer (motion joint order -> policy order);
        # ``policy_joint_ids`` indexes the robot (articulation order -> policy order). They are different
        # spaces and must not be interchanged.
        self.goal_joint_ids = reference_joint_ids
        self._goal_default_joint_pos = default_joint_pos
        self.goal_window = int(goal_window)
        # The state's link axis is already in the critic observation's order, so no re-selection happens
        # by default. ``critic_link_ids`` exists only for tasks whose state and critic select differently.
        self._critic_link_ids: torch.Tensor | slice = slice(None) if critic_link_ids is None else critic_link_ids
        # The simulation context the world model is conditioned on (friction, CoM, joint calibration),
        # supplied by the caller from the env's ``sim_context`` group. Empty when a task has none.
        self.context_format: OrderedDict[str, tuple] = OrderedDict(context_format or {})
        self.policy_term_order = list(policy_term_order)
        self.critic_term_order = list(critic_term_order)
        # Paper Table A.1: the policy sees the last ``history_length`` proprioceptive frames. The
        # manager appends the current frame *before* returning, so the newest frame is last; the
        # imagination must reproduce both the set of terms and that ordering exactly, or the policy
        # would be trained on a differently-arranged input than it is deployed with.
        self.history_terms = tuple(history_terms)
        self.history_length = int(history_length)
        missing = [t for t in self.history_terms if t not in self.policy_term_order]
        if missing:
            raise ValueError(f"history terms {missing} are not in the policy observation group")
        self.termination_temperature = termination_temperature
        # The env's reward manager multiplies every weighted term by the control interval, so the reward
        # the simulator hands the agent is ``dt`` times the weighted sum. The imagined reward is scaled the
        # same way: the critic's targets, the gradient clipping and the paper's normalised return
        # (H-step reward / (H * 0.1), 0.1 = (4 * 1 + 2 * 0.5) * 0.02) all assume that scale.
        self.step_dt = float(step_dt)
        self.ref_window = ref_window
        self._sample_ref_window: ReferenceWindow | None = None
        # Paper Sec. 3.3: "the model emits the noise-free observation, we re-add sensor noise". Half-width
        # of the uniform noise for every dimension of the flat policy observation (zero where a term
        # carries none), or None to imagine noise-free observations.
        self.policy_noise_halfwidth = policy_noise_halfwidth

    #: Observation group carrying the world-model state.
    STATE_GROUP = "wm_state"
    #: Observation group carrying the simulation context.
    CONTEXT_GROUP = "sim_context"

    def _reward_scale(self, like: torch.Tensor) -> torch.Tensor:
        """``weight * step_dt`` per reward term, built once per device and dtype (a host-to-device copy
        on every imagined step would synchronise the stream)."""
        key = (like.device, like.dtype)
        cache = self.__dict__.setdefault("_reward_scale_cache", {})
        if key not in cache:
            weights = torch.tensor([self.REWARD_WEIGHTS[name] for name in REWARD_TERMS], dtype=torch.float64)
            cache[key] = (weights * self.step_dt).to(device=like.device, dtype=like.dtype)
        return cache[key]

    def world_model_parameterization(self):
        """How the world model reads this state and predicts its change: heading-invariant features and
        frame-local increments (``world_model_frames``), so the dynamics need not be learned separately
        for every environment origin and heading."""
        from instinctlab.tasks.shadowing.supermimic.world_model_frames import HeadingFrameParameterization

        return HeadingFrameParameterization(self.state_format)

    def split_obs_pack(self, pack: dict) -> dict[str, torch.Tensor]:
        """Slice the flat ``wm_state`` / ``sim_context`` groups into named components.

        The observation pipeline ships one concatenated tensor per group, while the algorithm works in
        named components (that is what the replay stores and what the reward cores index). The layouts
        live here, so this is the only place that knows the offsets.
        """
        out = self._split_group(pack, self.STATE_GROUP, self.state_format)
        out.update(self._split_group(pack, self.CONTEXT_GROUP, self.context_format))
        return out

    @staticmethod
    def _split_group(pack: dict, group_name: str, layout) -> dict[str, torch.Tensor]:
        """Slice one flat observation group according to ``layout`` (``{name: shape}``)."""
        group = pack.get(group_name)
        if group is None or not layout:
            return {}
        total = int(sum(int(np.prod(shape)) for shape in layout.values()))
        if group.shape[-1] != total:
            raise ValueError(f"{group_name} carries {group.shape[-1]} values but its layout describes {total}")
        out = {}
        start = 0
        for name, shape in layout.items():
            n = int(np.prod(shape))
            out[name] = group[..., start : start + n].reshape(group.shape[0], *shape)
            start += n
        return out

    # -- reference handling ------------------------------------------------------------------------

    def set_reference_window(self, window: ReferenceWindow | None) -> None:
        """Install the reference snapshot for the upcoming rollout."""
        self.ref_window = window

    #: Observation groups the replay must store per transition so a rollout can be seeded from it.
    SEED_GROUPS = ("ref_cursor",)
    #: Simulation-context term holding the per-joint calibration offset.
    JOINT_OFFSET_TERM = "joint_offset"

    def _joint_offset(self, context: torch.Tensor | None) -> torch.Tensor | None:
        """The calibration offset ``(B, J)`` carried in the context, or None when the task has none."""
        if context is None or self.JOINT_OFFSET_TERM not in self.context_format:
            return None
        return unpack_state(context, self.context_format)[self.JOINT_OFFSET_TERM]

    def begin_rollout(self, seeds: dict[str, torch.Tensor], env_index: torch.Tensor, horizon: int) -> torch.Tensor:
        """Build the reference window of a batch of rollouts seeded from replayed states.

        Paper Sec. 3.3: each imagined rollout starts from a replay-buffer state, and the goal is read
        from the reference motion. The reference must be the one *that state* was tracking, so the
        window is looked up from the cursor recorded with the state (its clip and the time inside it),
        not from wherever its environment is now. Using the environment's current position pairs a
        state with an unrelated stretch of motion, and the policy is then taught to track a goal its
        state does not correspond to.

        Args:
            seeds: ``{"ref_cursor": (B, 2)}`` as recorded by the ``reference_cursor`` term.
            env_index: ``(B,)`` environment each sample came from, for its world origin.
            horizon: Imagined steps ``H``.

        Returns:
            ``(B,)`` column index of the start frame, i.e. zeros.
        """
        if self.motion_reference is None:
            raise RuntimeError("this imagination was built without a motion reference manager")
        manager = self.motion_reference
        if len(manager.motion_buffers) != 1:
            raise NotImplementedError("reference cursors are only defined for a single motion buffer")
        buffer = next(iter(manager.motion_buffers.values()))

        cursor = seeds["ref_cursor"].to(manager.device)
        env_index = env_index.to(manager.device).long()
        # The window spans the rollout *plus* the K-frame goal look-ahead: at imagined step t the goal
        # terms read columns t .. t + K - 1, and the critic bootstrap reads column H.
        num_frames = int(horizon) + int(self.goal_window)
        start = 1.0 if manager.cfg.data_start_from == "one_frame_interval" else 0.0
        offsets = torch.arange(num_frames, device=manager.device, dtype=torch.float32) + start
        clip_time_s = cursor[:, 1:2] + offsets.unsqueeze(0) * manager.frame_interval_s[env_index].unsqueeze(-1)

        frames = buffer.gather_frames(cursor[:, 0].round().long(), clip_time_s, manager.env_origins[env_index])
        self._sample_ref_window = ReferenceWindow(**frames)
        self.ref_window = self._sample_ref_window
        return torch.zeros(cursor.shape[0], dtype=torch.long, device=manager.device)

    def rollout_start_valid(
        self, seeds: dict[str, torch.Tensor], env_index: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """``(B,)`` True where the clip still has a reference frame ``horizon`` steps after the start.

        A rollout that runs past its clip's end would end in the dataset-exhausted termination, a
        bookkeeping event rather than a failure, and its zero bootstrap would teach the critic that
        states near a clip end are worthless. Such starts are not imagined from.
        """
        manager = self.motion_reference
        buffer = next(iter(manager.motion_buffers.values()))
        seqs = buffer._all_motion_sequences
        cursor = seeds["ref_cursor"].to(manager.device)
        motion_ids = cursor[:, 0].round().long().to(buffer.buffer_device)
        start = 1.0 if manager.cfg.data_start_from == "one_frame_interval" else 0.0
        env_index = env_index.to(manager.device).long()
        t_end = cursor[:, 1] + (int(horizon) + start) * manager.frame_interval_s[env_index]
        frame = torch.round(t_end.to(buffer.buffer_device) * seqs.framerate[motion_ids])
        return (frame < seqs.buffer_length[motion_ids]).to(cursor.device)

    def subset_reference_rows(self, sample_index: torch.Tensor) -> None:
        """Narrow the rollout's reference window to a batch subset (e.g. one policy chunk)."""
        if self._sample_ref_window is None:
            raise RuntimeError("no rollout window; call begin_rollout first")
        keep = sample_index.to(self._sample_ref_window.base_pos_w.device).long()
        if int(keep.max()) >= self._sample_ref_window.base_pos_w.shape[0]:
            raise IndexError(
                f"sample index {int(keep.max())} is out of range for "
                f"{self._sample_ref_window.base_pos_w.shape[0]} window rows"
            )
        self.ref_window = self._sample_ref_window.rows(keep)

    def _ref(self, ref_index: torch.Tensor | None) -> ReferenceWindow | None:
        if self.ref_window is None or ref_index is None:
            return None
        return self.ref_window.at(ref_index)

    def advance(self, ref_index: torch.Tensor | None) -> torch.Tensor | None:
        return None if ref_index is None else ref_index + 1

    def valid_start(self, ref_index: torch.Tensor | None, horizon: int) -> torch.Tensor:
        """Whether a rollout of ``horizon`` steps starting at ``ref_index`` stays inside the window."""
        if self.ref_window is None or ref_index is None:
            return torch.ones(1, dtype=torch.bool)
        return (ref_index + horizon) < self.ref_window.horizon

    # -- reward ------------------------------------------------------------------------------------

    def reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        ref_index: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-term reward times ``step_dt``, ``(B, num_terms)``, differentiable w.r.t. state and action."""
        ref = self._ref(ref_index)
        if ref is None:
            raise ValueError("SuperMimic reward needs a reference window and ref_index")
        s = unpack_state(state, self.state_format)

        terms = [
            body_position_reward(s["body_link_pos_w"], s["root_pos_w"], s["root_quat_w"], ref),
            body_orientation_reward(s["body_link_quat_w"], s["root_pos_w"], s["root_quat_w"], ref),
            body_linear_velocity_reward(s["body_link_lin_vel_w"], ref),
            body_angular_velocity_reward(s["body_link_ang_vel_w"], ref),
            root_position_reward(s["root_pos_w"], ref),
            root_orientation_reward(s["root_quat_w"], ref),
            action_smoothness_penalty(action, prev_action),
            joint_limit_penalty(s["joint_pos"], self.soft_joint_pos_limits),
        ]
        return torch.stack(terms, dim=-1) * self._reward_scale(state)

    def termination_prob(self, state: torch.Tensor, ref_index: torch.Tensor | None) -> torch.Tensor:
        ref = self._ref(ref_index)
        if ref is None:
            raise ValueError("SuperMimic termination needs a reference window and ref_index")
        s = unpack_state(state, self.state_format)
        return termination_prob(
            s["root_pos_w"],
            s["root_quat_w"],
            ref,
            temperature=self.termination_temperature,
        )

    # -- observations ------------------------------------------------------------------------------

    def _term_value(
        self,
        name: str,
        s: dict[str, torch.Tensor],
        prev_action: torch.Tensor | None,
        joint_offset: torch.Tensor | None = None,
    ):
        """Value of a manager observation term, built from state.

        Only terms derivable from the imagined state are supported. Anything else raises: a wrong
        observation would corrupt the policy gradient silently, which is worse than failing.

        ``joint_offset`` is the per-sample calibration offset ``(B, J)`` from the simulation context. The
        environment observes ``q - q0`` with its *randomized* default pose ``q0 = nominal + offset``, so
        the offset must be subtracted too or every imagined joint observation is off by it.
        """
        if name == "joint_pos":
            rel = s["joint_pos"] - self._default_joint_pos
            if joint_offset is None:
                return rel
            # broadcast a (B, J) offset over the frame axis of a (B, H, J) history
            return rel - (joint_offset.unsqueeze(1) if rel.dim() == 3 else joint_offset)
        if name == "joint_vel":
            return s["joint_vel"]
        if name == "base_ang_vel":
            return quat_apply_inverse(s["root_quat_w"], s["root_ang_vel_w"])
        if name == "base_lin_vel":
            return quat_apply_inverse(s["root_quat_w"], s["root_lin_vel_w"])
        if name == "projected_gravity":
            return _projected_gravity(s["root_quat_w"])
        if name == "last_action":
            # At the first imagined step there is no previous action. The environment's own value at
            # reset is zeros, so use the same thing rather than failing.
            if prev_action is None:
                return torch.zeros(
                    s["joint_pos"].shape[0], int(np.prod(self.state_format["joint_pos"])), device=s["joint_pos"].device
                )
            return prev_action
        # The critic's privileged link observations are base-frame views of the same world-frame state
        # the world model predicts, so they are recomputed rather than transported.
        if name == "link_pos":
            body_ids = self._critic_link_ids
            return self._to_base_frame(s["body_link_pos_w"][:, body_ids], s["root_pos_w"], s["root_quat_w"])
        if name == "link_rot":
            body_ids = self._critic_link_ids
            link_quat_b = self._to_base_frame_quat(s["body_link_quat_w"][:, body_ids], s["root_quat_w"])
            return quat_to_tan_norm(link_quat_b)
        raise NotImplementedError(
            f"observation term {name!r} is not derivable from the world-model state; "
            "either add it to the state layout or exclude it from the imagined observation"
        )

    def _goal_values(
        self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, ref_index: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """The four reference terms of the policy group, over the K-frame goal window.

        Mirrors the manager's ``PositionRefCommand`` / ``RotationRefCommand`` /
        ``JointPosRefCommand`` / ``JointVelRefCommand``. ``position_ref`` is anchored to the *predicted*
        robot pose, which is what the ``anchor_frame="robot"`` configuration means: the goal is the
        reference's future path expressed in the robot's current frame.
        """
        if self.ref_window is None or ref_index is None:
            raise ValueError("reference observations need a reference window and ref_index")
        if isinstance(ref_index, int):
            # every row at the same column: the K goal frames are one slice of the window
            w = self.ref_window.columns(ref_index, self.goal_window)
            pos, quat, jpos, jvel = w.base_pos_w, w.base_quat_w, w.joint_pos, w.joint_vel
            jpos_mask, jvel_mask = w.joint_pos_mask, w.joint_vel_mask
            plane, height, valid = w.base_pos_plane_mask, w.base_pos_height_mask, w.validity
        else:
            frames = self.ref_window.goal_window(ref_index, self.goal_window)
            pos = torch.stack([f.base_pos_w[:, 0] for f in frames], dim=1)  # (B, K, 3)
            quat = torch.stack([f.base_quat_w[:, 0] for f in frames], dim=1)  # (B, K, 4)
            jpos = torch.stack([f.joint_pos[:, 0] for f in frames], dim=1)  # (B, K, J)
            jvel = torch.stack([f.joint_vel[:, 0] for f in frames], dim=1)
            jpos_mask = torch.stack([f.joint_pos_mask[:, 0] for f in frames], dim=1)
            jvel_mask = torch.stack([f.joint_vel_mask[:, 0] for f in frames], dim=1)
            plane = torch.stack([f.base_pos_plane_mask[:, 0] for f in frames], dim=1)  # (B, K)
            height = torch.stack([f.base_pos_height_mask[:, 0] for f in frames], dim=1)
            valid = torch.stack([f.validity[:, 0] for f in frames], dim=1)

        ids = self.goal_joint_ids if self.goal_joint_ids is not None else slice(None)
        # default pose is (num_joints,): broadcast along the joint axis of the (B, K, J) window
        joint_pos = (jpos[:, :, ids] - self._goal_default_joint_pos) * jpos_mask[:, :, ids]
        joint_pos = joint_pos * valid.unsqueeze(-1)
        joint_vel = jvel[:, :, ids] * jvel_mask[:, :, ids] * valid.unsqueeze(-1)

        # Anchor the reference path in the robot's current frame, using the same helper the manager's
        # ``PositionRefCommand`` calls: ``transform_points`` applies ``p -> R p + t`` with the *inverse*
        # transform returned by ``subtract_frame_transforms``.
        anchor_pos_inv, anchor_quat_inv = subtract_frame_transforms(root_pos_w, root_quat_w)
        # (B, K, 3) -> the helper wants points as (B, P, 3) with a per-batch (B, 3) transform
        rel_pos = transform_points(pos, anchor_pos_inv, anchor_quat_inv)
        rel_pos = rel_pos * torch.logical_or(plane, height).unsqueeze(-1) * valid.unsqueeze(-1)

        rel_quat = quat_mul(anchor_quat_inv.unsqueeze(1).expand(-1, quat.shape[1], 4), quat)
        rel_rot = quat_to_tan_norm(rel_quat) * valid.unsqueeze(-1)

        return {
            "joint_pos_ref": joint_pos,
            "joint_vel_ref": joint_vel,
            "position_ref": rel_pos,
            "rotation_ref": rel_rot,
        }

    @staticmethod
    def _to_base_frame(points: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
        """World points -> the root's local frame, matching ``instinct_mdp.link_pos_b``."""
        pos_inv, quat_inv_ = subtract_frame_transforms(root_pos_w, root_quat_w)
        return transform_points(points, pos_inv, quat_inv_)

    @staticmethod
    def _to_base_frame_quat(quat_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
        """World orientations -> the root's local frame, matching ``instinct_mdp.link_quat_b``."""
        inv = quat_inv(root_quat_w).unsqueeze(1).expand(-1, quat_w.shape[1], 4)
        return quat_mul(inv, quat_w)

    def _assemble(self, order: list[str], s, prev_action) -> torch.Tensor:
        parts = [self._term_value(n, s, prev_action) for n in order]
        return torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=-1)

    _GOAL_TERMS = ("joint_pos_ref", "joint_vel_ref", "position_ref", "rotation_ref")

    def policy_obs(
        self,
        state: torch.Tensor,
        context: torch.Tensor | None,
        prev_action: torch.Tensor | None,
        ref_index: torch.Tensor | None,
        history_state: torch.Tensor | None = None,
        history_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the policy observation group from an imagined state.

        History terms keep ``history_length`` frames. The current frame is the one last, matching the
        observation manager, which appends before returning; frames older than the rollout start come
        from the replayed window the rollout was seeded with.

        Args:
            state: ``(B, state_size)`` current imagined state.
            context: Unused by the current task; kept for interface parity.
            prev_action: ``(B, num_actions)`` action applied to reach ``state``.
            ref_index: Column offset into the reference window.
            history_state: ``(B, H, state_size)`` past states, oldest first, current last. Required when
                history terms are configured.
            history_action: ``(B, H, num_actions)`` past actions, oldest first. Shifting this forward by one
                turns it into the ``last_action`` history.
        """
        s = unpack_state(state, self.state_format)
        needs_goal = any(t in self.policy_term_order for t in self._GOAL_TERMS)
        goals = self._goal_values(s["root_pos_w"], s["root_quat_w"], ref_index) if needs_goal else {}

        joint_offset = self._joint_offset(context)
        past = self._history_terms(history_state, history_action, joint_offset) if self.history_terms else {}

        parts = []
        for name in self.policy_term_order:
            if name in goals:
                value = goals[name]
            elif name in past:
                value = past[name]
            else:
                value = self._term_value(name, s, prev_action, joint_offset)
            parts.append(value.reshape(value.shape[0], -1))
        return torch.cat(parts, dim=-1)

    def _history_terms(
        self,
        history_state: torch.Tensor | None,
        history_action: torch.Tensor | None,
        joint_offset: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-term ``(B, H, D)`` history over the trailing ``history_length`` frames.

        ``history_state[:, -1]`` is the frame the rollout starts from and ``history_state[:, 0]`` the
        oldest one kept. ``history_action`` is aligned to the same frames: ``history_action[:, j]`` is
        the action that *produced* ``history_state[:, j]``, so it is directly the ``last_action``
        observation. The caller supplies it shifted by one row, which it gets for free by sampling one
        extra row from the replay.
        """
        if history_state is None:
            raise ValueError(
                "this task configures observation history, so the imagined rollout needs the replayed "
                "state window; pass history_state"
            )
        h = int(self.history_length)
        if history_state.shape[1] != h:
            raise ValueError(f"history_state has {history_state.shape[1]} frames, expected {h}")
        batch = history_state.shape[0]
        # one unpack of the whole (B*H) window: rows are ordered (sample, frame), matching the tensor
        # layout, so reshaping back to (B, H, ...) restores the frame axis.
        flat = unpack_state(history_state.reshape(batch * h, -1), self.state_format)
        frames = {k: v.reshape(batch, h, *v.shape[1:]) for k, v in flat.items()}

        out = {}
        for name in self.history_terms:
            if name == "last_action":
                if history_action is None:
                    raise ValueError("the last_action history needs the replayed action window")
                out[name] = history_action.reshape(batch, h, -1)
                continue
            # a state-only term: evaluate it on every frame at once
            out[name] = self._term_value(name, frames, None, joint_offset)
        return out

    def advance_history(
        self,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll the two history windows forward by one imagined step.

        Appending the action *alongside* the state it produced keeps ``history_action[:, j]`` equal to
        ``last_action`` at ``history_state[:, j]``, which is the alignment the policy expects.
        """
        return (
            torch.cat([history_state[:, 1:], next_state.unsqueeze(1)], dim=1),
            torch.cat([history_action[:, 1:], action.unsqueeze(1)], dim=1),
        )

    def critic_obs(
        self,
        state: torch.Tensor,
        context: torch.Tensor | None,
        ref_index: torch.Tensor | None,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        """Assemble the critic observation group. Paper: the critic is V(s, g, xi), so the simulation
        context terms are filled from ``context`` rather than derived from state."""
        s = unpack_state(state, self.state_format)
        needs_goal = any(t in self.critic_term_order for t in self._GOAL_TERMS)
        goals = self._goal_values(s["root_pos_w"], s["root_quat_w"], ref_index) if needs_goal else {}
        ctx = unpack_state(context, self.context_format) if context is not None and self.context_format else {}

        parts = []
        for name in self.critic_term_order:
            if name in goals:
                value = goals[name]
            elif name in ctx:
                value = ctx[name]
            else:
                value = self._term_value(name, s, prev_action, ctx.get(self.JOINT_OFFSET_TERM))
            parts.append(value.reshape(value.shape[0], -1))
        return torch.cat(parts, dim=-1)

    def add_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Re-add the policy's sensor noise to a noise-free imagined observation (paper Sec. 3.3).

        Applied to the raw observation, before normalisation, exactly where the environment applies
        it. Uniform per dimension, matching the ``UniformNoiseCfg`` of each term.
        """
        if self.policy_noise_halfwidth is None:
            return obs
        width = self.policy_noise_halfwidth
        if width.device != obs.device or width.dtype != obs.dtype:
            # moved once: the tensor is kept on the observation's device from then on
            width = self.policy_noise_halfwidth = width.to(device=obs.device, dtype=obs.dtype)
        if width.shape[-1] != obs.shape[-1]:
            raise ValueError(f"noise layout has {width.shape[-1]} dims, observation has {obs.shape[-1]}")
        return obs + (2.0 * torch.rand_like(obs) - 1.0) * width
