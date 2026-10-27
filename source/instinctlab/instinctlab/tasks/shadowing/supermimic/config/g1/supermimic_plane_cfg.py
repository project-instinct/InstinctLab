"""Configuration for the G1 robot in the SuperMimic task.

SuperMimic trains a motion-tracking policy by backpropagating through a learned world model, so the
task provides three things beyond a normal tracking task:

* the reference motion stack (LAFAN1, converted to InstinctLab npz),
* a **world-model state layout** (the ``wm_state`` and ``ref_index`` observation groups) whose term
  names *are* the world-model state component names, and
* a provider for the algorithm's differentiable view of the MDP
  (``supermimic_provider.py``), named by the agent config and resolved by the runner.

The observation groups are configured here; the provider is referenced by dotted path, because it can
only be built once the simulator is running.

Data: ``~/Datasets/LAFAN1_G1_retargetted_instinctnpz`` -- the 40 LAFAN1 clips converted with
``scripts/lafan1_csv_to_instinct.py`` (2.45 h, matching the paper's dataset size).

This file subclasses the BeyondMimic G1 task and swaps the motion buffer, so tracking rewards,
terminations and action scaling are shared with the existing implementation instead of duplicated.
The scene must be built at class-definition time (not in ``__post_init__``): the parent's
``BeyondMimicSceneCfg.__post_init__`` drops ``robot_reference`` when ``motion_reference`` is
``MISSING``, which happens if the scene is constructed before the motion cfg is assigned.
"""

import os

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import instinctlab.envs.mdp as instinct_mdp

import instinctlab.tasks.shadowing.beyondmimic.beyondmimic_env_cfg as beyondmimic_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_POLICY_JOINT_ORDER_V1,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.utils.config import set_cfg_joint_order
from instinctlab.tasks.shadowing.beyondmimic.config.g1.beyondmimic_plane_cfg import (
    G1BeyondMimicPlaneEnvCfg,
)
from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_rewards_cfg import (
    SuperMimicRewardsCfg,
    SuperMimicPaperTerminationsCfg,
)

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

LAFAN1_G1_PATH = os.path.expanduser("~/Datasets/LAFAN1_G1_retargetted_instinctnpz")
MOTION_NAME = "Lafan1"

#: Paper App. A: the goal encoder sees a K = 10 future reference sequence.
SUPER_MIMIC_GOAL_WINDOW = 10

#: Paper Table A.1: the policy sees the last 10 proprioceptive observations.
SUPER_MIMIC_OBS_HISTORY = 10

#: Terms of the policy group that carry observation history. Deliberately *not* the four reference
#: terms: those already span the K = 10 future window, so stacking 10 frames of history would square
#: the goal window instead of adding memory. These are the proprioceptive terms the paper describes.
SUPER_MIMIC_HISTORY_TERMS = ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "last_action")




@configclass
class SuperMimicAmassMotionCfg(AmassMotionCfgBase):
    """LAFAN1 motions for SuperMimic.

    ``filtered_motion_selection_filepath`` is left as ``None`` so every ``*retargetted.npz`` under
    ``path`` is used; the converted dataset holds exactly the paper's 40 clips.
    """

    path = LAFAN1_G1_PATH
    retargetting_func = None
    filtered_motion_selection_filepath = None
    motion_start_from_middle_range = [0.0, 0.8]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    motion_bin_length_s = 1.0  # reset-bin length used by the return-based reset curriculum


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_[^/]+/RobotReference",
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    symmetric_augmentation_joint_names=list(G1_29DOF_POLICY_JOINT_ORDER_V1),
    # SuperMimic's goal encoder is conditioned on a K = 10 future reference sequence rather than a
    # single frame (arXiv:2610.28480v1, App. A, "Goal"). ``data_start_from="current_time"`` makes the
    # first window column the current frame, so the columns are t, t+1, ..., t+9 at frame_interval_s.
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=SUPER_MIMIC_GOAL_WINDOW,
    data_start_from="current_time",
    visualizing_robot_offset=(0.0, 1.5, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        MOTION_NAME: SuperMimicAmassMotionCfg(),
    },
    mp_split_method="Even",
)


#: The tracked bodies, taken from the motion reference itself so the world-model state and the reference
#: buffer cannot drift apart. Order matters: the reward cores pair robot link ``i`` with reference link
#: ``i``.
TRACKED_BODIES = list(motion_reference_cfg.link_of_interests)


@configclass
class SuperMimicObsExtrasCfg:
    """Observation groups the SuperMimic algorithm consumes beyond ``policy`` / ``critic``.

    These are not policy inputs. They are the transported description of the task's state for the
    differentiable world model: the group's term names *are* the world-model state component names, so
    the algorithm's ``state_specs`` and the imagination's layout both come from here and there is no
    second vocabulary to keep in sync.

    ``concatenate_terms`` is False so each term keeps its own shape -- ``body_link_pos_w`` stays
    (num_links, 3) rather than flattening -- which is what the reward cores index. Corruption is off:
    these values train the dynamics model and must not carry sensor noise.
    """

    @configclass
    class WmStateCfg(ObsGroupCfg):
        """Every quantity the paper's reward compares (Table A.2)."""

        # world frame, to match ``body_link_pos_w`` and the reward/reference terms; IsaacLab's
        # ``mdp.root_pos_w`` would subtract the environment origin and mix two frames in one state.
        root_pos_w = ObsTerm(func=instinct_mdp.body_root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_quat_w = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        body_link_pos_w = ObsTerm(
            func=instinct_mdp.body_link_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TRACKED_BODIES, preserve_order=True)},
        )
        body_link_quat_w = ObsTerm(
            func=instinct_mdp.body_link_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TRACKED_BODIES, preserve_order=True)},
        )
        body_link_lin_vel_w = ObsTerm(
            func=instinct_mdp.body_link_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TRACKED_BODIES, preserve_order=True)},
        )
        body_link_ang_vel_w = ObsTerm(
            func=instinct_mdp.body_link_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TRACKED_BODIES, preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RefIndexCfg(ObsGroupCfg):
        """Reference frame index, stored per transition so a rollout can be re-seeded from replay."""

        ref_index = ObsTerm(
            func=instinct_mdp.reference_frame_index,
            params={"asset_cfg": SceneEntityCfg("motion_reference")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SimContextCfg(ObsGroupCfg):
        """The randomized simulation parameters the dynamics model is conditioned on.

        Paper Sec. 3.2 supplies the world model with a simulation context so it can represent that the
        same action produces different outcomes in different worlds. These read the values the
        randomization events actually realized rather than re-sampling them, which would disagree with
        the physics that is running.

        Corruption is off and terms stay unflattened for the same reasons as ``wm_state``.
        """

        friction = ObsTerm(
            func=instinct_mdp.rigid_body_friction,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_com = ObsTerm(
            # Must select the body whose CoM the parent task actually randomizes, which is the torso
            # (``base_com`` event, ``body_names="torso_link"``). Reading a body that is never
            # randomized yields a constant, so the world model would condition on a value that carries
            # no information and would not notice the bug.
            func=instinct_mdp.rigid_body_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",), preserve_order=True)},
        )
        joint_offset = ObsTerm(
            func=instinct_mdp.joint_calibration_offset,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    wm_state = WmStateCfg()
    ref_index = RefIndexCfg()
    sim_context = SimContextCfg()


SUPER_MIMIC_DESIGNS = ("paper", "beyondmimic")


@configclass
class G1SuperMimicEnvCfg(G1BeyondMimicPlaneEnvCfg):
    """G1 SuperMimic environment: the BeyondMimic tracking task on the LAFAN1 dataset.

    ``super_mimic_design`` selects the MDP instantiation:

    * ``"paper"`` (default) -- SuperMimic's own (arXiv:2610.28480v1, App. A): eight reward terms with
      no self-collision penalty, joint limit at -2, and no end-effector termination.
    * ``"beyondmimic"`` -- the parent task's reward and termination sets unchanged, for A/B comparison.

    The two sets differ only where the paper says so; everything else is inherited.
    """

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    super_mimic_design: str = "paper"

    observations_extra: SuperMimicObsExtrasCfg = SuperMimicObsExtrasCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.super_mimic_design not in SUPER_MIMIC_DESIGNS:
            raise ValueError(
                f"super_mimic_design must be one of {SUPER_MIMIC_DESIGNS}, got {self.super_mimic_design!r}"
            )
        # Assign both sets explicitly so the choice is idempotent: re-running __post_init__ must not
        # depend on which design happened to be applied first.
        # the observation manager iterates the groups of ``observations``, so the extra groups are
        # attached there rather than kept as a sibling config
        for group_name in ("wm_state", "ref_index", "sim_context"):
            setattr(self.observations, group_name, getattr(self.observations_extra, group_name))

        # The world-model state must be in the *policy* joint order: the imagination subtracts the
        # default pose from ``joint_pos`` to build the policy observation, which is only correct if both
        # are ordered alike. The parent already applies this order to the actions and to the
        # policy/critic joint terms; the ``wm_state`` joint terms need it too.
        joint_order = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
        for term_name in ("joint_pos", "joint_vel"):
            term = getattr(self.observations.wm_state, term_name)
            # a fresh selector per term: scene-entity resolution mutates the cfg it is given
            term.params["asset_cfg"] = set_cfg_joint_order(SceneEntityCfg("robot"), joint_order)

        # Paper Table A.1: proprioceptive history. ``flatten_history_dim`` (default) makes each term
        # report ``history * dim``, and the observation manager appends the current frame before
        # returning, so the newest frame is always last. The imagination reproduces exactly this.
        for term_name in SUPER_MIMIC_HISTORY_TERMS:
            term = getattr(self.observations.policy, term_name, None)
            if term is not None:
                term.history_length = SUPER_MIMIC_OBS_HISTORY

        # The LAFAN1 clips include acrobatic motions (falls, rolls, jumps) whose contact count exceeds
        # BeyondMimic's 224-constraint budget; the MJWarp solver aborts the run when it overflows
        # ("nefc overflow"). Raise the budgets for this task only.
        from instinctlab.tasks.utils.newton import newton_physics_cfg

        if self.sim.physics is not None and hasattr(self.sim.physics, "newton_mjwarp"):
            self.sim.physics.newton_mjwarp = newton_physics_cfg(njmax=1024, nconmax=256, margin=0.0, gap=0.01)

        if self.super_mimic_design == "paper":
            self.rewards = SuperMimicRewardsCfg()
            self.terminations = SuperMimicPaperTerminationsCfg()
        else:
            self.rewards = beyondmimic_cfg.BeyondMimicSingleRewardsCfg()
            self.terminations = beyondmimic_cfg.BeyondMimicTerminationCfg()
