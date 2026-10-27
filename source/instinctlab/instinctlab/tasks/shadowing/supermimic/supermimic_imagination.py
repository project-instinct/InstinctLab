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

from dataclasses import dataclass

import numpy as np
from typing import OrderedDict, Sequence

import torch

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

from instinctlab.utils.math import quat_to_tan_norm

from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_rewards_cfg import (
    PAPER_REWARD_STDS,
    PAPER_REWARD_WEIGHTS,
)

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


def make_state_format(num_links: int, num_joints: int) -> "OrderedDict[str, tuple]":
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


def state_size(fmt: "OrderedDict[str, tuple]") -> int:
    total = 0
    for shape in fmt.values():
        n = 1
        for d in shape:
            n *= int(d)
        total += n
    return total


def _seg(fmt: "OrderedDict[str, tuple]", name: str) -> slice:
    start = 0
    for key, shape in fmt.items():
        n = 1
        for d in shape:
            n *= int(d)
        if key == name:
            return slice(start, start + n)
        start += n
    raise KeyError(f"{name} not in state format; have {list(fmt)}")


def unpack_state(state: torch.Tensor, fmt: "OrderedDict[str, tuple]") -> dict[str, torch.Tensor]:
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

    def at(self, ref_index: torch.Tensor) -> "ReferenceWindow":
        """One step per env, where ``ref_index`` is a *column offset* into this window.

        Column ``j`` is the reference at ``t + j * control_dt``, so ``ref_index`` starts at 0 and
        :meth:`SuperMimicImagination.advance` adds one per imagined step. Past the end the last column
        is clamped; ``validity`` records that the clip ran out.
        """
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

    def rows(self, index: torch.Tensor) -> "ReferenceWindow":
        """Select one row per output sample, keeping every frame column."""
        return ReferenceWindow(
            **{name: getattr(self, name)[index] for name in self._FIELD_NAMES}
        )

    def goal_window(self, ref_index: torch.Tensor, length: int) -> "ReferenceWindow":
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
    base_pos, delta_quat = relative_world_anchor(
        root_pos_w, root_quat_w, ref.base_pos_w[:, 0], ref.base_quat_w[:, 0]
    )
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
    gravity = torch.tensor([0.0, 0.0, -1.0], device=root_pos_w.device, dtype=root_pos_w.dtype).expand(
        root_pos_w.shape[0], 3
    )
    grav_robot = quat_apply_inverse(root_quat_w, gravity)
    grav_ref = quat_apply_inverse(ref.base_quat_w[:, 0], gravity)
    tilt_error = torch.abs(grav_robot[:, 2] - grav_ref[:, 2])
    return height_error, tilt_error


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
        state_format: "OrderedDict[str, tuple]",
        soft_joint_pos_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        policy_term_order: list[str],
        critic_term_order: list[str],
        reference_joint_ids: torch.Tensor | slice | None = None,
        policy_joint_ids: torch.Tensor | slice | None = None,
        critic_link_ids: torch.Tensor | slice | None = None,
        goal_window: int = 10,
        history_terms: Sequence[str] = (),
        history_length: int = 0,
        termination_temperature: float = 0.02,
        motion_reference=None,
        ref_window: ReferenceWindow | None = None,
    ) -> None:
        # The layout is supplied by the caller (normally the ``wm_state`` observation group), not
        # derived here: term names are the state component names, so there is one vocabulary.
        self.state_format = OrderedDict(state_format)
        self.state_size = state_size(self.state_format)
        self.num_joints = int(np.prod(self.state_format["joint_pos"])) if "joint_pos" in self.state_format else 0
        self.num_links = (
            int(self.state_format["body_link_pos_w"][0]) if "body_link_pos_w" in self.state_format else 0
        )
        # optional: turns a replayed absolute frame index into a window-relative column offset
        self.motion_reference = motion_reference
        self._window_scratch = None
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
        # The protocol declares a context layout; this task feeds the world model no simulation context
        # (no friction / CoM / calibration terms), which is a documented deviation from the paper.
        self.context_format: "OrderedDict[str, tuple]" = OrderedDict()
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
        self.ref_window = ref_window
        self._env_ref_window: ReferenceWindow | None = None
        self._sample_ref_window: ReferenceWindow | None = None
        self._prev_action: torch.Tensor | None = None

    def split_obs_pack(self, pack: dict) -> dict[str, torch.Tensor]:
        """Slice the flat ``wm_state`` observation group into named state components.

        The observation pipeline ships one concatenated tensor per group, while the algorithm works in
        named components (that is what the replay stores and what the reward cores index). The layout
        lives here, so this is the only place that knows the offsets.
        """
        group = pack.get("wm_state")
        if group is None:
            return {}
        if group.shape[-1] != self.state_size:
            raise ValueError(
                f"wm_state carries {group.shape[-1]} values but the state layout describes {self.state_size}"
            )
        out = {}
        start = 0
        for name, shape in self.state_format.items():
            n = 1
            for d in shape:
                n *= int(d)
            out[name] = group[..., start : start + n].reshape(group.shape[0], *shape)
            start += n
        return out

    # -- reference handling ------------------------------------------------------------------------

    def set_reference_window(self, window: ReferenceWindow | None) -> None:
        """Install the reference snapshot for the upcoming rollout."""
        self.ref_window = window

    def refresh_reference_window(self, horizon: int) -> ReferenceWindow:
        """Rebuild the snapshot from the manager's current time, reusing one scratch buffer.

        Called once per update rather than per rollout step: the reference does not change while the
        world model is rolled forward.
        """
        if self.motion_reference is None:
            raise RuntimeError("this imagination was built without a motion reference manager")
        from instinctlab.motion_reference.motion_reference_data import MotionReferenceData

        # The window must span the rollout *plus* the K-frame goal look-ahead: at imagined step t the
        # goal terms read columns t .. t + K - 1, so ``horizon`` alone would clamp the goal to the last
        # column for most of the rollout and quietly feed the policy a frozen future.
        num_frames = int(horizon) + int(self.goal_window)
        if self._window_scratch is None or self._window_scratch.validity.shape[1] != num_frames:
            self._window_scratch = MotionReferenceData.make_empty(
                num_envs=self.motion_reference.count,
                num_frames=num_frames,
                num_joints=self.motion_reference.num_joints,
                num_links=self.motion_reference.num_link_of_interests,
                device=self.motion_reference.device,
            )
        self.ref_window = build_reference_window(
            self.motion_reference, horizon=num_frames - 1, scratch=self._window_scratch
        )
        self._env_ref_window = None
        self._sample_ref_window = None
        return self.ref_window

    def select_reference_rows(self, env_index: torch.Tensor) -> None:
        """Expand the per-environment window to one row per imagined sample.

        Rollouts are seeded from replay states drawn across environments, so a window with one row per
        env cannot be indexed by sample. ``env_index[i]`` says which env sample ``i`` came from.
        """
        if self.ref_window is None:
            raise RuntimeError("no reference window to reindex; call refresh_reference_window first")
        self._env_ref_window = self.ref_window
        self._sample_ref_window = self.ref_window.rows(env_index.to(self.ref_window.base_pos_w.device).long())
        self.ref_window = self._sample_ref_window

    def subset_reference_rows(self, sample_index: torch.Tensor) -> None:
        """Narrow the sample-level window to a batch subset (e.g. one policy chunk).

        Distinct from :meth:`select_reference_rows`: that one indexes *environments*, this one indexes
        the already-expanded sample rows.
        """
        if getattr(self, "_sample_ref_window", None) is None:
            raise RuntimeError("no sample-level window; call select_reference_rows first")
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
        """Per-term reward, ``(B, num_terms)``, differentiable w.r.t. ``state`` and ``action``."""
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
        weights = torch.tensor(
            [self.REWARD_WEIGHTS[name] for name in REWARD_TERMS],
            device=state.device,
            dtype=state.dtype,
        )
        return torch.stack(terms, dim=-1) * weights

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

    def _term_value(self, name: str, s: dict[str, torch.Tensor], prev_action: torch.Tensor | None):
        """Value of a manager observation term, built from state.

        Only terms derivable from the imagined state are supported. Anything else raises: a wrong
        observation would corrupt the policy gradient silently, which is worse than failing.
        """
        if name == "joint_pos":
            return s["joint_pos"] - self._default_joint_pos
        if name == "joint_vel":
            return s["joint_vel"]
        if name == "base_ang_vel":
            return quat_apply_inverse(s["root_quat_w"], s["root_ang_vel_w"])
        if name == "base_lin_vel":
            return quat_apply_inverse(s["root_quat_w"], s["root_lin_vel_w"])
        if name == "projected_gravity":
            gravity = torch.tensor([0.0, 0.0, -1.0], device=s["root_quat_w"].device, dtype=s["root_quat_w"].dtype)
            return quat_apply_inverse(s["root_quat_w"], gravity.expand(s["root_quat_w"].shape[0], 3))
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
            link_quat_b = self._to_base_frame_quat(
                s["body_link_quat_w"][:, body_ids], s["root_quat_w"]
            )
            return quat_to_tan_norm(link_quat_b)
        raise NotImplementedError(
            f"observation term {name!r} is not derivable from the world-model state; "
            f"either add it to the state layout or exclude it from the imagined observation"
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
        goals = (
            self._goal_values(s["root_pos_w"], s["root_quat_w"], ref_index) if needs_goal else {}
        )

        past = self._history_terms(history_state, history_action) if self.history_terms else {}

        parts = []
        for name in self.policy_term_order:
            if name in goals:
                value = goals[name]
            elif name in past:
                value = past[name]
            else:
                value = self._term_value(name, s, prev_action)
            parts.append(value.reshape(value.shape[0], -1))
        return torch.cat(parts, dim=-1)

    def _history_terms(
        self, history_state: torch.Tensor | None, history_action: torch.Tensor | None
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
            out[name] = self._term_value(name, frames, None)
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
    ) -> torch.Tensor:
        s = unpack_state(state, self.state_format)
        needs_goal = any(t in self.critic_term_order for t in self._GOAL_TERMS)
        goals = (
            self._goal_values(s["root_pos_w"], s["root_quat_w"], ref_index) if needs_goal else {}
        )

        parts = []
        for name in self.critic_term_order:
            value = goals[name] if name in goals else self._term_value(name, s, self._prev_action)
            parts.append(value.reshape(value.shape[0], -1))
        return torch.cat(parts, dim=-1)

    def add_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """No-op: the runner's normaliser handles scaling, and the paper's per-term noise is applied
        by the caller that knows the term boundaries."""
        return obs
