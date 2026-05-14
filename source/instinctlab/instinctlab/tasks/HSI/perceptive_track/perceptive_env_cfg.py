import math
import os
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
from instinctlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinctlab.managers import MultiRewardCfg
from instinctlab.monitors import (
    MonitorTermCfg,
    MotionReferenceMonitorTerm,
    ShadowingJointPosMonitorTerm,
    ShadowingJointVelMonitorTerm,
    ShadowingLinkPosMonitorTerm,
    ShadowingPositionMonitorTerm,
    ShadowingRotationMonitorTerm,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.sensors import GroupedRayCasterCfg, NoisyGroupedRayCasterCameraCfg
from instinctlab.tasks.HSI import mdp as shadowing_mdp
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinctlab.terrains.trimesh import MotionMatchedTerrainCfg
from instinctlab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthContourNoiseCfg,
    DepthNormalizationCfg,
    DepthSkyArtifactNoiseCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    StereoTooCloseNoiseCfg,
)

# PROPRIO_HISTORY_LENGTH = 0
PROPRIO_HISTORY_LENGTH = 5


@configclass
class PerceptiveShadowingSceneCfg(InteractiveSceneCfg):
    """Configuration for the BeyondMimic scene with necessary scene entities as motion reference."""

    env_spacing = 10

    # robots
    robot: ArticulationCfg = MISSING

    # robot reference articulation
    robot_reference: ArticulationCfg = None

    # motion reference
    motion_reference: MotionReferenceManagerCfg = MISSING  # type: ignore

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        # terrain_type="plane",
        # terrain_generator=None,
        terrain_type="hacked_generator",
        terrain_generator=FiledTerrainGeneratorCfg(
            size=(20, 20),
            border_width=0.0,
            border_height=0.0,
            num_rows=13,
            num_cols=13,
            sub_terrains={
                "motion_matched": MotionMatchedTerrainCfg(
                    proportion=1.0,
                ),
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=[
            "/World/ground",
            # NOTE: Don't forget to add the robot links in robot-specific configuration file.
        ],
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),
                0.015,
                0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016,
            ),
            rot=(
                math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),
                math.sin(math.radians(0.5) / 2),
                math.sin(math.radians(48) / 2),
                0.0,
            ),
            convention="world",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(87) / 2),  # fovx
            vertical_aperture=2 * math.tan(math.radians(58) / 2),  # fovy
            height=int(270 / 10),
            width=int(480 / 10),
        ),
        data_types=["distance_to_image_plane"],
        noise_pipeline={
            # "depth_contour_noise": DepthContourNoiseCfg(
            #     contour_threshold=1.8,  # in [m]
            #     maxpool_kernel_size=1,
            # ),
            # "depth_artifact_noise": DepthArtifactNoiseCfg(),
            # "reflection_artifact_noise": DepthArtifactNoiseCfg(noise_value=30.0),
            # "stereo_noise": RangeBasedGaussianNoiseCfg(
            #     max_value=1.2,
            #     min_value=0.12,
            #     noise_std=0.02,
            # ),
            # "sky_artifact_noise": DepthSkyArtifactNoiseCfg(),
            # "gaussian_blur_noise": GaussianBlurNoiseCfg(
            #     kernel_size=3,
            #     sigma=0.5,
            # ),
            # "stereo_too_close_noise": StereoTooCloseNoiseCfg(),
            # These last two noise model will affect the processing on the onboard device.
            "normalize": DepthNormalizationCfg(
                depth_range=(0.0, 2.0),
                normalize=True,
            ),
            "crop_and_resize": CropAndResizeCfg(
                crop_region=(2, 2, 2, 2),
                resize_shape=(18, 32),
            ),
        },
        # data_histories={"distance_to_image_plane": 5},
        update_period=1 / 60,
        debug_vis=False,
        depth_clipping_behavior="max",  # clip to the maximum value
        min_distance=0.05,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )

    def __post_init__(self):
        if type(self.motion_reference) is type(MISSING) or not self.motion_reference.debug_vis:
            delattr(self, "robot_reference")


@configclass
class CommandCfg:
    position_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="robot",
    )
    position_b_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="reference",
    )
    rotation_ref_command = instinct_mdp.RotationRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        in_base_frame=True,
        rotation_mode="tannorm",
    )
    joint_pos_ref_command = instinct_mdp.JointPosRefCommandCfg(current_state_command=False)
    joint_vel_ref_command = instinct_mdp.JointVelRefCommandCfg(current_state_command=False)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        # Currently, just a dummy observation
        joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
        joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
        position_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        )
        rotation_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )

        height_scan = ObsTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=[-20.0, 20.0],
        )
        depth_image = ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane"},
            # params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"},
        )

        # proprioception
        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        # base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTermCfg(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyObsCfg = PolicyObsCfg()

    @configclass
    class CriticObsCfg(ObsGroupCfg):
        """Critic observations for BeyondMimic."""

        # BeyondMimic specific reference observations
        joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
        joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
        position_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "position_ref_command"})

        # proprioception
        link_pos = ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
        )
        link_rot = ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
        )

        height_scan = ObsTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=[-20.0, 20.0],
        )

        base_lin_vel = ObsTermCfg(
            func=mdp.base_lin_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTermCfg(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    critic: CriticObsCfg = CriticObsCfg()


@configclass
class RewardsCfg:
    base_position_imitation_gauss = RewTermCfg(
        func=instinct_mdp.base_position_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.3,
        },
    )
    base_rot_imitation_gauss = RewTermCfg(
        func=instinct_mdp.base_rot_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.4,
            "difference_type": "axis_angle",
        },
    )
    link_pos_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_pos_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.3,
        },
    )
    link_rot_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_rot_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.4,
        },
    )
    link_lin_vel_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_lin_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 1.0,
        },
    )
    link_ang_vel_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_ang_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 3.14,
        },
    )
    action_rate_l2 = RewTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # undesired_contacts = RewTermCfg(
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[
    #                 r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    #             ],
    #         ),
    #         "threshold": 1.0,
    #     },
    # )
    applied_torque_limits_by_ratio = RewTermCfg(
        func=instinct_mdp.applied_torque_limits_by_ratio,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*ankle.*",
                    ".*wrist.*",
                ],
            )
        },
    )


@configclass
class RewardGroupsCfg(MultiRewardCfg):
    rewards = RewardsCfg()


@configclass
class EventsCfg:
    # domain rand
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.25, 2.0),
            "dynamic_friction_range": (1.2, 1.8),
            "restitution_range": (0.0, 0.5),
            "make_consistent": True,
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTermCfg(
        func=instinct_mdp.randomize_default_joint_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "offset_distribution_params": (-0.01, 0.01),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "com_range": {
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    randomize_ray_offsets = EventTermCfg(
        func=instinct_mdp.randomize_ray_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("camera"),
            "offset_pose_ranges": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-math.radians(2), math.radians(2)),
                "pitch": (-math.radians(10), math.radians(10)),
                "yaw": (-math.radians(2), math.radians(2)),
            },
            "distribution": "uniform",
        },
    )
    randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_rigid_body_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "torso_link",
                    "left_ankle.*",
                    "right_ankle.*",
                    "left_wrist.*",
                    "right_wrist.*",
                ],
            ),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    match_motion_ref_with_scene = EventTermCfg(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
        },
    )
    reset_robot = EventTermCfg(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("robot"),
            # reset with position offset to put the robot_reference in scene.
            "position_offset": [0.0, 0.0, 0.0],
            "dof_vel_ratio": 1.0,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            # Pose randomization (+-5cm position, +-6degrees rotation)
            "randomize_pose_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.0, 0.0),
            },
            # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
            "randomize_velocity_range": {},
            # Joint position randomization (+-0.1 rad)
            "randomize_joint_pos_range": (-0.1, 0.1),
        },
    )
    bin_fail_counter_smoothing = EventTermCfg(
        func=instinct_mdp.beyondmimic_bin_fail_counter_smoothing,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # every environment step
        params={
            "curriculum_name": "beyond_adaptive_sampling",
        },
    )
    # push_robot = EventTermCfg(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(0.5, 2.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.2, 0.2),
    #             "roll": (-0.52, 0.52),
    #             "pitch": (-0.52, 0.52),
    #             "yaw": (-0.78, 0.78),
    #         },
    #     },
    # )


@configclass
class CurriculumCfg:
    beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
        func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)
    # illegal_reset_contact = DoneTermCfg(
    #     func=instinct_mdp.illegal_reset_contact,
    #     time_out=True,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[
    #                 r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    #             ],
    #         ),
    #         "threshold": 500,
    #         "episode_length_threshold": 2,
    #     },
    # )
    base_pos_too_far = DoneTermCfg(
        func=instinct_mdp.pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "distance_threshold": 0.25,
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
            "projected_gravity_threshold": 0.8,
            "check_at_keyframe_threshold": -1,
            "z_only": False,
            "print_reason": False,
        },
    )
    link_pos_too_far = DoneTermCfg(
        func=instinct_mdp.link_pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg(
                "motion_reference",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                preserve_order=True,
            ),
            "distance_threshold": 0.25,
            "in_base_frame": False,
            "check_at_keyframe_threshold": -1,
            "height_only": True,
            "print_reason": False,
        },
    )

    dataset_exhausted = DoneTermCfg(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "print_reason": False,
        },
    )
    out_of_border = DoneTermCfg(
        func=instinct_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
    )


@configclass
class MonitorCfg:
    dataset = MonitorTermCfg(
        func=MotionReferenceMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("motion_reference"),
            sample_stat_interval=500,
            top_n_samples=5,
        ),
    )
    shadowing_position = MonitorTermCfg(
        func=ShadowingPositionMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            check_at_keyframe_threshold=0.03,
        ),
    )
    shadowing_rotation = MonitorTermCfg(
        func=ShadowingRotationMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_pos = MonitorTermCfg(
        func=ShadowingJointPosMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_vel = MonitorTermCfg(
        func=ShadowingJointVelMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_link_pos_b = MonitorTermCfg(
        func=ShadowingLinkPosMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            masking=True,
        ),
    )
    shadowing_link_pos_w = MonitorTermCfg(
        func=ShadowingLinkPosMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=False,
            masking=True,
        ),
    )


@configclass
class PerceptiveShadowingEnvCfg(InstinctLabRLEnvCfg):
    scene: PerceptiveShadowingSceneCfg = PerceptiveShadowingSceneCfg()
    commands: CommandCfg = CommandCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardGroupsCfg = RewardGroupsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1.0 / 50.0 / self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**24
