"""ACP stage 2 for the G1: the contact planner, trained from hindsight contact labels of the tracker's own rollouts,
takes over the tracker's contact targets.

On top of the stage-1 tracker task (``acp_tracker_cfg``, same policy and critic observations, so a stage-1
checkpoint loads):

- observation group ``planner``: the 17 x 11 points of ``height_scanner`` as (x, y) and height in the body-yaw
  frame, the 5-step proprioception and the goal of the tracker (with the tracker's noise; heights N(0, 0.03^2));
- observation group ``contact_source`` (N, 1): 1 = the algorithm feeds the planner's targets instead of the
  reference's, with probability eta_obs of the training iteration (paper Eq. S1);
- reward ``track_planned_contact`` (weight 1.5 eta_rew) and the reference-contact rewards scaled by 1 - eta_rew;
- hindsight labeling of the episodes that start in the first 10 % of a clip and reach 90 % (``contact_labeler``);
- start phase U(0, 0.98) of the clip, no CSI.

The eta schedules' start iterations and ramp length are not in the paper (choices below, see the reproduction plan).
"""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptive_cfg
from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.managers import MultiRewardCfg
from instinctlab.tasks.shadowing.acp import planner_handoff
from instinctlab.tasks.shadowing.acp.hindsight_labeler import HindsightContactLabelerCfg
from instinctlab.utils.config import set_cfg_joint_order

from .acp_tracker_cfg import (
    CONTACT_BODIES,
    ACPObservationsCfg,
    ACPRewardsCfg,
    ACPTerrainContactMotionCfg,
    G1ACPTrackerEnvCfg,
)

# Hand-off schedules (paper Eq. S1, in PPO iterations of stage 2; u_rew > u_obs as the paper requires). The
# planner trains from the first iteration with enough replay samples; its targets reach the tracker from
# ETA_OBS_START on, and the reward moves to them from ETA_REW_START on.
ETA_OBS_START = 500
ETA_REW_START = 1000
ETA_RAMP_ITERATIONS = 2000
REFERENCE_CONTACT_REWARDS = ["contact_label", "contact_pos_base", "contact_pos_global"]


@configclass
class ACPPlannerObservationsCfg(ACPObservationsCfg):
    @configclass
    class PlannerObsCfg(ACPObservationsCfg.PolicyObsCfg):
        """The tracker's proprioception and goal terms (same noise), without its depth image and contact targets,
        plus the planner scan."""

        depth_image = None
        contact_targets = None
        terrain_xy = ObsTermCfg(
            func=planner_handoff.scan_points_xy_b, params={"sensor_cfg": SceneEntityCfg("height_scanner")}
        )
        terrain_height = ObsTermCfg(
            func=planner_handoff.scan_points_height_b,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.03),
            clip=(-2.0, 2.0),
        )

    @configclass
    class ContactSourceObsCfg(ObsGroupCfg):
        planner_gate = ObsTermCfg(
            func=planner_handoff.planner_gate,
            params={"start_iteration": ETA_OBS_START, "ramp_iterations": ETA_RAMP_ITERATIONS},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    planner: PlannerObsCfg = PlannerObsCfg()
    contact_source: ContactSourceObsCfg = ContactSourceObsCfg()


@configclass
class ACPPlannerRewardsCfg(ACPRewardsCfg):
    # paper: weight 1.5 eta_rew, sigma_c = 1 m
    track_planned_contact = RewTermCfg(
        func=planner_handoff.track_planned_contact, weight=1.5, params={"command_name": "contact_schedule", "std": 1.0}
    )


@configclass
class ACPPlannerRewardGroupsCfg(MultiRewardCfg):
    rewards = ACPPlannerRewardsCfg()


@configclass
class ACPPlannerCurriculumCfg:
    contact_reward_handoff = CurriculumTermCfg(
        func=planner_handoff.RewardWeightHandoff,
        params={
            "start_iteration": ETA_REW_START,
            "ramp_iterations": ETA_RAMP_ITERATIONS,
            "reference_terms": REFERENCE_CONTACT_REWARDS,
            "planner_terms": ["track_planned_contact"],
            "group_name": "rewards",
        },
    )


@configclass
class G1ACPPlannerEnvCfg(G1ACPTrackerEnvCfg):
    observations: ACPPlannerObservationsCfg = ACPPlannerObservationsCfg()
    rewards: ACPPlannerRewardGroupsCfg = ACPPlannerRewardGroupsCfg()
    curriculum: ACPPlannerCurriculumCfg = ACPPlannerCurriculumCfg()
    contact_labeler: HindsightContactLabelerCfg = HindsightContactLabelerCfg(body_names=CONTACT_BODIES)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2400  # paper Table S6, per worker
        # stage-2 initial phase U(0, 0.98) of the clip (Supp. Table S2), instead of stage 1's BeyondMimic bin
        # sampling (whose curriculum term this task's curriculum replaces, so its smoothing event goes too)
        motion = self.scene.motion_reference.motion_buffers["TerrainMotion"]
        motion.motion_bin_length_s = None
        motion.env_starting_stub_sampling_strategy = "independent"
        motion.motion_start_from_middle_range = [0.0, 0.98]
        self.events.bin_fail_counter_smoothing = None
        # the hindsight labeler counts contacts with the terrain only (its collider), not self-contacts
        self.scene.contact_forces.filter_prim_paths_expr = ["/World/ground/terrain/mesh"]
        joint_order = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
        self.observations.planner.joint_pos.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.planner.joint_vel.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.run_name = "g1ACPPlanner"


@configclass
class G1ACPPlannerEnvCfg_PLAY(G1ACPPlannerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.reset_robot.params["randomize_pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
        # play with scripts/acp/play_planner.py: it runs the planner and sends its targets as policy feedback


@configclass
class G1ACPPlannerSequentialHandoffEnvCfg(G1ACPPlannerEnvCfg):
    """Experiment variant: the reward moves to the planned contacts only after the planner's targets fully reach the
    tracker (eta_rew starts when the eta_obs ramp ends), so the two hand-offs do not overlap."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.contact_reward_handoff.params["start_iteration"] = ETA_OBS_START + ETA_RAMP_ITERATIONS


LATE_ETA_OBS_START = 2000
"""Late hand-off experiments: the planner trains on 2000 iterations of labels before its targets reach the tracker."""


@configclass
class G1ACPPlannerReferenceLabelsEnvCfg(G1ACPPlannerEnvCfg):
    """Experiment variant: the planner learns the reference motion's contacts (the schedule the stage-1 tracker
    follows) instead of the robot's own."""

    def __post_init__(self):
        super().__post_init__()
        self.contact_labeler.contact_source = "reference"


@configclass
class G1ACPPlannerLateHandoffEnvCfg(G1ACPPlannerEnvCfg):
    """Experiment variant: both hand-offs start LATE_ETA_OBS_START - ETA_OBS_START iterations later."""

    def __post_init__(self):
        super().__post_init__()
        delay = LATE_ETA_OBS_START - ETA_OBS_START
        self.observations.contact_source.planner_gate.params["start_iteration"] = ETA_OBS_START + delay
        self.curriculum.contact_reward_handoff.params["start_iteration"] = ETA_REW_START + delay


@configclass
class G1ACPPlannerReferenceLabelsLateHandoffEnvCfg(G1ACPPlannerLateHandoffEnvCfg):
    """Experiment variant: reference-contact labels and the late hand-off."""

    def __post_init__(self):
        super().__post_init__()
        self.contact_labeler.contact_source = "reference"


PARTIAL_HANDOFF_ETA_MAX = 0.5
"""Partial hand-off experiment: both ramps stop at this eta. Every step keeps the reference targets with
probability 1 - eta (planner_gate draws per env and step), so the tracker keeps crossing obstacles under reference
targets, which keeps the hindsight labels coming after the hand-off, and keeps half of the reference-contact reward."""


@configclass
class G1ACPPlannerPartialHandoffEnvCfg(G1ACPPlannerEnvCfg):
    """Experiment variant: the hand-off ramps to ``PARTIAL_HANDOFF_ETA_MAX`` instead of 1."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.contact_source.planner_gate.params["eta_max"] = PARTIAL_HANDOFF_ETA_MAX
        self.curriculum.contact_reward_handoff.params["eta_max"] = PARTIAL_HANDOFF_ETA_MAX


@configclass
class G1ACPPlannerLabelAllEnvCfg(G1ACPPlannerEnvCfg):
    """Experiment variant: hindsight labels from every episode, not only reference-following successes. Episodes of
    any start phase are recorded (one slot per env), and the ones that end before the admission end phase (mostly
    failures at the obstacles) are labeled up to their last step with a complete lookahead, so the planner also gets
    labels for the approaches the tracker fails on."""

    def __post_init__(self):
        super().__post_init__()
        self.contact_labeler.start_phase_max = 1.0
        self.contact_labeler.label_unadmitted = True
        self.contact_labeler.num_slots = self.scene.num_envs


@configclass
class ACPPlannerAdaptiveStartCurriculumCfg(ACPPlannerCurriculumCfg):
    beyond_adaptive_sampling = CurriculumTermCfg(func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting)


@configclass
class G1ACPPlannerAdaptiveStartEnvCfg(G1ACPPlannerEnvCfg):
    """Experiment variant: stage 1's BeyondMimic adaptive start sampling (1 s bins, failing bins sampled more) instead
    of the paper's U(0, 0.98) start phase."""

    curriculum: ACPPlannerAdaptiveStartCurriculumCfg = ACPPlannerAdaptiveStartCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        motion = self.scene.motion_reference.motion_buffers["TerrainMotion"]
        tracker_motion = ACPTerrainContactMotionCfg()
        motion.motion_bin_length_s = tracker_motion.motion_bin_length_s
        motion.env_starting_stub_sampling_strategy = tracker_motion.env_starting_stub_sampling_strategy
        motion.motion_start_from_middle_range = tracker_motion.motion_start_from_middle_range
        self.events.bin_fail_counter_smoothing = perceptive_cfg.EventsCfg().bin_fail_counter_smoothing


@configclass
class G1ACPPlannerReferenceLabelsAdaptiveStartEnvCfg(G1ACPPlannerAdaptiveStartEnvCfg):
    """Experiment variant: reference-contact labels and the adaptive start sampling."""

    def __post_init__(self):
        super().__post_init__()
        self.contact_labeler.contact_source = "reference"


@configclass
class G1ACPPlannerDepthEnvCfg(G1ACPPlannerEnvCfg):
    """Experiment variant: the planner observation group also holds the tracker's depth image (for a planner with an
    image encoder, ``G1ACPDepthPlannerPPORunnerCfg``)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.planner.depth_image = ACPObservationsCfg.PolicyObsCfg().depth_image
