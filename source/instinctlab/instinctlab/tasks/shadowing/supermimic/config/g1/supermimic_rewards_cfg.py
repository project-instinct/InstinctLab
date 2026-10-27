"""Reward and termination sets for the SuperMimic task.

Two sets are available, selected by ``super_mimic_design`` on the env cfg:

* ``"paper"`` (default) -- SuperMimic's own instantiation (arXiv:2610.28480v1, App. A).
* ``"beyondmimic"`` -- the parent task's sets, for A/B comparison against the existing tracker.

**Why every value is written out here instead of inherited.** The paper is explicit that it reuses
BeyondMimic's *formulation* -- "We use BeyondMimic's motion-tracking reward (Table A.2), removing its
self-collision penalty" (App. A) -- so the reward functions are the same. But a reproduction must pin
its own numbers: inheriting ``BeyondMimicRewardsCfg`` would alias these weights to another task's
config, so retuning BeyondMimic later would silently change what SuperMimic reproduces. The values
below are transcribed from the paper's Table A.2 and are independent of the BeyondMimic task's config.

Table A.2, verbatim (``T`` is the set of tracked bodies):

    Task (tracking)
        Body Position           exp(-1/|T| sum_b ||p_b^des - p_b||^2 / 0.3^2)              1.0
        Body Orientation        exp(-1/|T| sum_b ||log(R_b^des R_b^T)||^2 / 0.4^2)         1.0
        Body Linear velocity    exp(-1/|T| sum_b ||v_b^des - v_b||^2 / 1.0^2)             1.0
        Body Angular velocity   exp(-1/|T| sum_b ||w_b^des - w_b||^2 / 3.14^2)            1.0
        Root Position           exp(-||p_root^des - p_root||^2 / 0.3^2)                   0.5
        Root Orientation        exp(-||log(R_root^des R_root^T)||^2 / 0.4^2)              0.5
    Regularization
        Action smoothness       ||a_t - a_{t-1}||^2                                      -0.1
        Joint limit             sum_j max(0, q_j - q_j^max) + max(0, q_j^min - q_j)       -2.0

Two deltas from BeyondMimic, both stated in App. A:

* the self-collision penalty is **removed** (not zero-weighted). It guards against reward hacking by
  a *stochastic* policy -- pressing two nearly-touching bodies together suppresses action noise and
  raises the velocity reward. SuperMimic's imagined rollouts are deterministic, so the exploit has no
  payoff and the term is unnecessary.
* the joint-limit weight is lowered from -10 to -2.

Because no term depends on contact, every reward here is a pure function of (articulation state,
reference frame index). That is what lets the differentiable cores in ``supermimic_imagination.py``
share the same formulas, and it is verified against this config by the parity test.
"""

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils.configclass import configclass

import instinctlab.envs.mdp as instinct_mdp

from instinctlab.managers import MultiRewardCfg

#: Paper Table A.2 weights. Module constants so ``supermimic_imagination.py`` applies the same numbers
#: and the parity test can assert them.
PAPER_REWARD_WEIGHTS = {
    "body_position": 1.0,
    "body_orientation": 1.0,
    "body_linear_velocity": 1.0,
    "body_angular_velocity": 1.0,
    "root_position": 0.5,
    "root_orientation": 0.5,
    "action_smoothness": -0.1,
    "joint_limit": -2.0,
}

#: Paper Table A.2 kernel widths, i.e. the denominators inside each Gaussian.
PAPER_REWARD_STDS = {
    "body_position": 0.3,
    "body_orientation": 0.4,
    "body_linear_velocity": 1.0,
    "body_angular_velocity": 3.14,
    "root_position": 0.3,
    "root_orientation": 0.4,
}

#: Paper App. A "Termination": ``|e_p,z| > 0.25 m or e_tilt > 0.8``.
PAPER_HEIGHT_THRESHOLD = 0.25
PAPER_TILT_THRESHOLD = 0.8


@configclass
class SuperMimicPaperRewardsCfg:
    """SuperMimic's reward, Table A.2, stated in full.

    The functions are InstinctLab's existing tracking terms -- the paper reuses BeyondMimic's
    formulation, so reimplementing the kernels would be the deviation. Only the weights, the kernel
    widths, and the absence of the contact term distinguish this from BeyondMimic.
    """

    # Task (tracking). BeyondMimic's links terms already operate in the robot-relative world frame
    # with mean-over-links reduction, which is what Table A.2's equations describe.
    body_position = RewTerm(
        func=instinct_mdp.link_pos_imitation_gauss,
        weight=PAPER_REWARD_WEIGHTS["body_position"],
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": PAPER_REWARD_STDS["body_position"],
        },
    )
    body_orientation = RewTerm(
        func=instinct_mdp.link_rot_imitation_gauss,
        weight=PAPER_REWARD_WEIGHTS["body_orientation"],
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": PAPER_REWARD_STDS["body_orientation"],
        },
    )
    body_linear_velocity = RewTerm(
        func=instinct_mdp.link_lin_vel_imitation_gauss,
        weight=PAPER_REWARD_WEIGHTS["body_linear_velocity"],
        params={"combine_method": "mean_prod", "std": PAPER_REWARD_STDS["body_linear_velocity"]},
    )
    body_angular_velocity = RewTerm(
        func=instinct_mdp.link_ang_vel_imitation_gauss,
        weight=PAPER_REWARD_WEIGHTS["body_angular_velocity"],
        params={"combine_method": "mean_prod", "std": PAPER_REWARD_STDS["body_angular_velocity"]},
    )
    root_position = RewTerm(
        func=instinct_mdp.base_position_imitation_gauss,
        weight=PAPER_REWARD_WEIGHTS["root_position"],
        params={"std": PAPER_REWARD_STDS["root_position"]},
    )
    root_orientation = RewTerm(
        func=instinct_mdp.base_rot_imitation_gauss,
        weight=PAPER_REWARD_WEIGHTS["root_orientation"],
        params={"std": PAPER_REWARD_STDS["root_orientation"], "difference_type": "axis_angle"},
    )

    # Regularization.
    action_smoothness = RewTerm(func=mdp.action_rate_l2, weight=PAPER_REWARD_WEIGHTS["action_smoothness"])
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=PAPER_REWARD_WEIGHTS["joint_limit"],
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # BeyondMimic's ``undesired_contacts`` term is deliberately absent -- see the module docstring.
    # The manager skips ``None`` terms, so leaving the attribute out drops it from the MDP entirely.


@configclass
class SuperMimicPaperTerminationsCfg:
    """SuperMimic's terminations, stated in full.

    App. A: "An episode terminates when the root body deviates from the reference beyond a height or
    tilt threshold (|e_p,z| > 0.25 m or e_tilt > 0.8) ... Unlike BeyondMimic, we do not terminate on
    end-effector deviation."

    ``time_out``, ``dataset_exhausted`` and ``out_of_border`` are environment bookkeeping rather than
    paper terms: the first bounds the episode length, the second ends a clip, the third keeps the
    robot inside the terrain. They are kept so the MDP is well-posed.
    """

    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)
    base_pos_too_far = DoneTermCfg(
        func=instinct_mdp.pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "distance_threshold": PAPER_HEIGHT_THRESHOLD,
            "check_at_keyframe_threshold": -1,
            "print_reason": False,
            "height_only": True,
        },
    )
    base_pg_too_far = DoneTermCfg(
        func=instinct_mdp.projected_gravity_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "projected_gravity_threshold": PAPER_TILT_THRESHOLD,
            "check_at_keyframe_threshold": -1,
            "print_reason": False,
        },
    )
    dataset_exhausted = DoneTermCfg(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={"reference_cfg": SceneEntityCfg("motion_reference"), "print_reason": False},
    )
    out_of_border = DoneTermCfg(
        func=instinct_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
    )

    # "Unlike BeyondMimic, we do not terminate on end-effector deviation." No ``link_pos_too_far``.


@configclass
class SuperMimicRewardsCfg(MultiRewardCfg):
    """Reward group wrapper.

    The environment routes to ``MultiRewardManager`` on ``isinstance(cfg.rewards, MultiRewardCfg)``,
    so this container is required by the framework, not by anything in the paper.
    """

    rewards: SuperMimicPaperRewardsCfg = SuperMimicPaperRewardsCfg()
