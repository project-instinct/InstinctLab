# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import yaml
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.mdp as shadowing_mdp
import instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg as shadowing_cfg

##
# Pre-defined configs
##
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.managers import MultiRewardCfg
from instinctlab.monitors import (
    ActuatorMonitorTerm,
    MonitorTermCfg,
    RewardSumMonitorTerm,
    ShadowingBasePosMonitorTerm,
    ShadowingJointReferenceMonitorTerm,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.aistpp_motion_cfg import AistppMotionCfg as AistppMotionCfgBase
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.utils.humanoid_ik import HumanoidSmplRotationalIK

combine_method = "prod"
G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# MOTION_NAME = "AccadRun" # success
# _hacked_selected_file_ = "ACCAD/Male2Running_c3d/C5 - walk to run_retargetted.npz"

# MOTION_NAME = "AccadMartialBounce"
# _hacked_selected_file_ = "ACCAD/MartialArtsWalksTurns_c3d/E8 - bounce_retargetted.npz"
# MOTION_NAME = "KitStomp"
# _hacked_selected_file_ = "KIT/3/stomp_left03_retargetted.npz"

# MOTION_NAME = "AccadMartialSpin"
# _hacked_selected_file_ = "ACCAD/Male2MartialArtsKicks_c3d/G20_-__reverse_spin_cresent_right_retargetted.npz"
# MOTION_NAME = "KitStretch"  # requires balancing
# _hacked_selected_file_ = "KIT/3/streching_leg01_retargetted.npz"

MOTION_NAME = "LafanKungfu1"
_hacked_selected_files_ = ["fightAndSports1_subject1_retargetted.npz"]
# MOTION_NAME = "LafanSprint1"
# _hacked_selected_files_ = ["sprint1_subject2_retargetted.npz"]

# MOTION_NAME = "test"
# _hacked_selected_files_ = ["CMU/90/90_26_retargetted.npz"]

MOTION_NAME = "LafanFight5Files"
_path_ = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
_hacked_selected_files_ = [
    "fight1_subject2_retargetted.npz",
    "fight1_subject3_retargetted.npz",
    "fight1_subject5_retargetted.npz",
    "fightAndSports1_subject1_retargetted.npz",
    "fightAndSports1_subject4_retargetted.npz",
]


MOTION_NAME = "LafanFiltered"
_path_ = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
_hacked_selected_files_ = [
    "aiming1_subject1_retargetted.npz",  # O
    "aiming1_subject4_retargetted.npz",  # O
    "aiming2_subject2_retargetted.npz",  # O
    "aiming2_subject3_retargetted.npz",  # O
    "aiming2_subject5_retargetted.npz",  # O
    "dance1_subject1_retargetted.npz",  # O
    "dance1_subject2_retargetted.npz",  # O
    "dance1_subject3_retargetted.npz",  # O
    "dance2_subject1_retargetted.npz",  # O
    "dance2_subject2_retargetted.npz",  # O
    "dance2_subject3_retargetted.npz",  # O
    "dance2_subject4_retargetted.npz",  # O
    "dance2_subject5_retargetted.npz",  # O
    "fallAndGetUp1_subject1_retargetted.npz",  # O
    "fallAndGetUp1_subject4_retargetted.npz",  # O
    "fallAndGetUp1_subject5_retargetted.npz",  # O
    "fallAndGetUp2_subject2_retargetted.npz",  # O
    "fallAndGetUp2_subject3_retargetted.npz",  # O
    "fallAndGetUp3_subject1_retargetted.npz",  # O
    "fight1_subject2_retargetted.npz",  # O
    "fight1_subject3_retargetted.npz",  # O
    "fight1_subject5_retargetted.npz",  # O
    "fightAndSports1_subject1_retargetted.npz",  # O
    "fightAndSports1_subject4_retargetted.npz",  # O
    "ground1_subject1_retargetted.npz",  # O
    "ground1_subject4_retargetted.npz",  # O
    "ground1_subject5_retargetted.npz",  # O
    "ground2_subject2_retargetted.npz",  # O
    "ground2_subject3_retargetted.npz",  # O
    "jumps1_subject1_retargetted.npz",  # O
    "jumps1_subject2_retargetted.npz",  # O
    "jumps1_subject5_retargetted.npz",  # O
    "multipleActions1_subject1_retargetted.npz",  # O
    "multipleActions1_subject2_retargetted.npz",  # O
    # "multipleActions1_subject3_retargetted.npz", # X
    # "multipleActions1_subject4_retargetted.npz", # - (some sitting pose, but seems torlerable)
    # "obstacles1_subject1_retargetted.npz", # X
    # "obstacles1_subject2_retargetted.npz", # X
    # "obstacles1_subject5_retargetted.npz", # X
    # "obstacles2_subject1_retargetted.npz", # X
    # "obstacles2_subject2_retargetted.npz", # X
    # "obstacles2_subject5_retargetted.npz", # disable all obstacles
    # "obstacles3_subject3_retargetted.npz", # disable all obstacles
    # "obstacles3_subject4_retargetted.npz", # disable all obstacles
    # "obstacles4_subject2_retargetted.npz", # disable all obstacles
    # "obstacles4_subject3_retargetted.npz", # X  # disable all obstacles
    # "obstacles4_subject4_retargetted.npz", # disable all obstacles
    # "obstacles5_subject2_retargetted.npz", # disable all obstacles
    # "obstacles5_subject3_retargetted.npz", # disable all obstacles
    # "obstacles5_subject4_retargetted.npz", # disable all obstacles
    # "obstacles6_subject1_retargetted.npz", # disable all obstacles
    # "obstacles6_subject4_retargetted.npz", # disable all obstacles
    # "obstacles6_subject5_retargetted.npz", # disable all obstacles
    "push1_subject2_retargetted.npz",  # O
    "pushAndFall1_subject1_retargetted.npz",  # O
    "pushAndFall1_subject4_retargetted.npz",  # O
    "pushAndStumble1_subject2_retargetted.npz",  # O
    "pushAndStumble1_subject3_retargetted.npz",  # O
    "pushAndStumble1_subject5_retargetted.npz",  # O
    "run1_subject2_retargetted.npz",
    "run1_subject5_retargetted.npz",
    "run2_subject1_retargetted.npz",
    "run2_subject4_retargetted.npz",
    "sprint1_subject2_retargetted.npz",
    "sprint1_subject4_retargetted.npz",
    "walk1_subject1_retargetted.npz",
    "walk1_subject2_retargetted.npz",
    "walk1_subject5_retargetted.npz",
    "walk2_subject1_retargetted.npz",
    "walk2_subject3_retargetted.npz",
    "walk2_subject4_retargetted.npz",
    "walk3_subject1_retargetted.npz",
    "walk3_subject2_retargetted.npz",
    "walk3_subject3_retargetted.npz",
    "walk3_subject4_retargetted.npz",
    "walk3_subject5_retargetted.npz",
    "walk4_subject1_retargetted.npz",  # O
]

# MOTION_NAME = "LafanGetup2S3"
# _path_ = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
# _hacked_selected_files_ = [
#     "fallAndGetUp2_subject3_retargetted.npz",
# ]

with open(f"/tmp/{MOTION_NAME}.yaml", "w") as f:
    yaml.dump(
        {
            "selected_files": _hacked_selected_files_,
        },
        f,
    )


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    # path = os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702")
    # path = os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_torsoBase_retargetted_20250825_instinctnpz")
    # path = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
    # path = os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_torsoBase_retargetted_20250901_instinctnpz")
    # path = _path_
    path = os.path.expanduser("~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single")
    retargetting_func = None
    filtered_motion_selection_filepath = None
    motion_start_from_middle_range = [0.0, 0.8]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    # env_starting_stub_sampling_strategy = "concat_motion_bins"
    env_starting_stub_sampling_strategy = "independent"
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    motion_bin_length_s = 1.0
    # _hacked_selected_files_ = _hacked_selected_files_


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
    # symmetric_augmentation_link_mapping=[
    #     0,
    #     1,
    #     3,
    #     2,
    #     5,
    #     4,
    #     7,
    #     6,
    #     9,
    #     8,
    #     11,
    #     10,
    #     13,
    #     12,
    # ],
    # symmetric_augmentation_joint_mapping=G1_29Dof_torsoBase_symmetric_augmentation_joint_mapping,
    # symmetric_augmentation_joint_reverse_buf=G1_29Dof_torsoBase_symmetric_augmentation_joint_reverse_buf,
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    visualizing_robot_offset=(0.0, 1.5, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        #     "CMU_KIT": AmassMotionCfg(
        #         path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_retargetted_20250702"),  # type: ignore
        #         filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
        #             "~/Datasets/AMASS_selections/CMU_KIT_weighted_retargetted_20250702.yaml",
        #         ),
        #     ),
        # "CMU_KIT_DanceDB_BioMotionLab": AmassMotionCfg(
        #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_DanceDB_BioMotionLab_retargetted_20250702"),  # type: ignore
        #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
        #         "~/Datasets/AMASS_selections/CMU_KIT_DanceDB_BioMotionLab_weighted_retargetted_20250702.yaml",
        #     ),
        # ),
        # "CMU_KIT_ACCAD_DanceDB_HumanEva": AmassMotionCfg(
        #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702"),  # type: ignore
        #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
        #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_retargetted_20250702.yaml",
        #         "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_moverange_20250724_retargetted_20250702.yaml",
        #     ),
        # ),
        # "UbisoftLAFAN1_GMR": AmassMotionCfg(
        #     path=os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz"),  # type: ignore
        #     filtered_motion_selection_filepath=None,
        # ),
        MOTION_NAME: AmassMotionCfg(),
    },
    mp_split_method="Even",
)


@configclass
class G1PlaneShadowingEnvCfg(shadowing_cfg.ShadowingEnvCfg):
    scene: shadowing_cfg.ShadowingSceneCfg = shadowing_cfg.ShadowingSceneCfg(
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
        # for actuator in self.scene.robot.actuators.values():
        #     actuator.min_delay = 0
        #     actuator.max_delay = 0
        self.actions.joint_pos.scale = beyondmimic_action_scale

        assert (
            len(list(self.scene.motion_reference.motion_buffers.keys())) == 1
        ), "Only support single motion buffer for now"
        motion_buffer = list(self.scene.motion_reference.motion_buffers.values())[0]
        if motion_buffer.motion_bin_length_s is not None:
            if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins":
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting,
                )
            elif motion_buffer.env_starting_stub_sampling_strategy == "independent":
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=instinct_mdp.BeyondMimicAdaptiveWeighting,
                )
            else:
                raise ValueError(
                    "Unsupported env starting stub sampling method:"
                    f" {motion_buffer.env_starting_stub_sampling_strategy}"
                )

        self.run_name: str = "".join(
            [
                "G1Shadowing",
                f"_{MOTION_NAME}",
                (
                    "_odomObs"
                    if ("base_lin_vel" in self.observations.policy.__dict__.keys())
                    and self.commands.position_b_ref_command.anchor_frame == "robot"
                    else ""
                ),
                # (
                #     "_" + "-".join(self.scene.motion_reference.motion_buffers.keys())
                #     if self.scene.motion_reference.motion_buffers
                #     else ""
                # ),
                # (
                #     f"_proprioHist{self.observations.policy.joint_pos.history_length}"
                #     if self.observations.policy.joint_pos.history_length > 0
                #     else ""
                # ),
                # (
                #     f"_futureRef{self.scene.motion_reference.num_frames}"
                #     if self.scene.motion_reference.num_frames > 1
                #     else ""
                # ),
                # f"_FrameStartFrom{self.scene.motion_reference.data_start_from}",
                # "_forLoopMotionWeights",
                # "_forLoopMotionSample",
                ("_pgTermXYalso" if not self.terminations.base_pg_too_far.params["z_only"] else ""),
                (
                    "_concatMotionBins"
                    if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
                "_fixFramerate_diveroll4",
            ]
        )


@configclass
class G1PlaneShadowingEnvCfg_PLAY(G1PlaneShadowingEnvCfg):
    scene: shadowing_cfg.ShadowingSceneCfg = shadowing_cfg.ShadowingSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(
            debug_vis=True,
            # motion_buffers={
            # "amass": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702"),  # type: ignore
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         "~/Datasets/AMASS_selections/amass_test_motion_files.yaml"
            #     ),
            # ),
            # "CMU_KIT_ACCAD_DanceDB_HumanEva": AmassMotionCfg(
            #     path=os.path.expanduser(  # type: ignore
            #         "~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702"
            #     ),
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_moverange_top50_20250724_retargetted_20250702.yaml",
            #         "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_moverange_top30-50_20250724_retargetted_20250702.yaml",
            #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_orient_height_20250725_top20_retargetted_20250702.yaml",
            #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702_noFastVel_filterOutGimbal_filterOutSit_20250825.yaml",
            #     ),
            # ),
            # "AMASS_GMR": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_retargetted_instinctnpz"),  # type: ignore
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         "~/Datasets/AMASS_selections/AMASS_SMPLX-NG_GMR_weighted_moverange_20250724_retargetted_instinctnpz.yaml"
            #     ),
            # ),
            # "instinct_testset": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/ProjectInstinct/tests"),  # type: ignore
            #     filtered_motion_selection_filepath=None,
            # ),
            # "UbisoftLAFAN1_GMR": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz_selected"),  # type: ignore
            #     filtered_motion_selection_filepath=None,
            # ),
            # },
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

        # enable print_reason option in the termination terms
        for term in self.terminations.__dict__.values():
            if term is None:
                continue
            if "print_reason" in term.params:
                term.params["print_reason"] = True
        # self.episode_length_s = 10.0
        # for term_name, term in self.terminations.__dict__.items():
        #     if (not term_name == "dataset_exhausted") and (not term_name == "time_out"):
        #         self.terminations.__dict__[term_name] = None

        # enable debug_vis option in commands
        for cmd in self.commands.__dict__.values():
            cmd.debug_vis = True

        # add PLAY-specific monitor term
        # self.monitors.shoulder_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.waist_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.knee_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.reward_sum = MonitorTermCfg(
        #     func=RewardSumMonitorTerm,
        # )
        # self.monitors.reference_stat_case = MonitorTermCfg(
        #     func=ShadowingJointReferenceMonitorTerm,
        #     params=dict(
        #         reference_cfg=SceneEntityCfg(
        #             "motion_reference",
        #             joint_names=[
        #                 "left_hip_pitch.*",
        #             ],
        #         ),
        #     ),
        # )
        # self.monitors.shadowing_base_pos = MonitorTermCfg(
        #     func=ShadowingBasePosMonitorTerm,
        #     params=dict(
        #         robot_cfg=SceneEntityCfg("robot"),
        #         motion_reference_cfg=SceneEntityCfg("motion_reference"),
        #     ),
        # )
