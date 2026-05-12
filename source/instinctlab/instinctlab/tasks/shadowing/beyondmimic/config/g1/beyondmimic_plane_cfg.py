import os

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.beyondmimic.beyondmimic_env_cfg as beyondmimic_cfg

##
# Pre-defined configs
##
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import (
    ActuatorMonitorTerm,
    MonitorTermCfg,
    RewardSumMonitorTerm,
    ShadowingJointReferenceMonitorTerm,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.aistpp_motion_cfg import AistppMotionCfg as AistppMotionCfgBase
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# Motion list + weights: data/dataset_folder/beyondmimic.yaml (paths in yaml are relative to `path` below)
_PROJECT_INSTINCT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 9)
)
_DATASET_FOLDER = os.path.join(_PROJECT_INSTINCT_ROOT, "data", "dataset_folder")
_BEYOND_MIMIC_SELECTION_YAML = os.path.join(_DATASET_FOLDER, "beyondmimic.yaml")
MOTION_NAME = os.path.splitext(os.path.basename(_BEYOND_MIMIC_SELECTION_YAML))[0]


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    """AMASS motion configuration for BeyondMimic."""

    path = _DATASET_FOLDER
    retargetting_func = None
    filtered_motion_selection_filepath = _BEYOND_MIMIC_SELECTION_YAML
    motion_start_from_middle_range = [0.0, 0.8]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    motion_bin_length_s = 1.0


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
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
    frame_interval_s=0.0,
    update_period=0.02,
    num_frames=1,
    visualizing_robot_offset=(0.0, 1.5, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        MOTION_NAME: AmassMotionCfg(),
    },
    mp_split_method="Even",
)


@configclass
class G1BeyondMimicPlaneEnvCfg(beyondmimic_cfg.BeyondMimicEnvCfg):
    """G1 BeyondMimic plane environment configuration."""

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    def __post_init__(self):
        super().__post_init__()

        # add link_of_interests to the policy observation
        if self.observations.policy.__dict__.get("link_pos", None) is not None:
            self.observations.policy.link_pos.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        if self.observations.policy.__dict__.get("link_rot", None) is not None:
            self.observations.policy.link_rot.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        if hasattr(self.observations, "critic") and self.observations.critic is not None:
            if self.observations.critic.__dict__.get("link_pos", None) is not None:
                self.observations.critic.link_pos.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )
            if self.observations.critic.__dict__.get("link_rot", None) is not None:
                self.observations.critic.link_rot.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale

        assert (
            len(list(self.scene.motion_reference.motion_buffers.keys())) == 1
        ), "Only support single motion buffer for now"

        # self.events.push_robot = None
        # self.rewards.rewards.undesired_contacts = None

        self.run_name: str = "".join(
            [
                "G1BeyondMimic",
                (
                    "_linVelObs"
                    if getattr(self.observations.policy, "base_lin_vel", None) is not None
                    and self.observations.policy.base_lin_vel.scale != 0.0
                    else ""
                ),
                f"_{MOTION_NAME}",
                ("_noPush" if self.events.push_robot is None else ""),
                ("_noContactPenalty" if getattr(self.rewards.rewards, "undesired_contacts", None) is None else ""),
                "_GmrMotion",
            ]
        )


@configclass
class G1BeyondMimicPlaneEnvCfg_PLAY(G1BeyondMimicPlaneEnvCfg):
    """G1 BeyondMimic plane environment configuration for playing/evaluation."""

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(
            debug_vis=True,
        ),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=[4.0, 0.75, 1.0],
        lookat=[0.0, 0.75, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.curriculum.beyond_adaptive_sampling = None
        self.events.push_robot = None
        self.events.bin_fail_counter_smoothing = None

        # If you want to play the motion from start and till the end, uncomment the following lines
        self.episode_length_s = 6000.0
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_bin_length_s = None

        # enable print_reason option in the termination terms
        for term in self.terminations.__dict__.values():
            if term is None:
                continue
            if "print_reason" in term.params:
                term.params["print_reason"] = True

        # enable debug_vis option in commands
        for cmd in self.commands.__dict__.values():
            cmd.debug_vis = True

        # add PLAY-specific monitor term
        self.monitors.shoulder_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.waist_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.knee_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.reward_sum = MonitorTermCfg(
            func=RewardSumMonitorTerm,
        )
        self.monitors.reference_stat_case = MonitorTermCfg(
            func=ShadowingJointReferenceMonitorTerm,
            params=dict(
                reference_cfg=SceneEntityCfg(
                    "motion_reference",
                    joint_names=[
                        "left_hip_pitch.*",
                    ],
                ),
            ),
        )
