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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.mdp as shadowing_mdp
from instinctlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinctlab.managers import MultiRewardCfg
from instinctlab.monitors import MonitorTermCfg
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.sensors import HierarchicalContactSensorCfg


@configclass
class BeyondMimicSceneCfg(InteractiveSceneCfg):
    """Configuration for the BeyondMimic scene with necessary scene entities as motion reference."""

    env_spacing = 4.0

    # robots
    robot: ArticulationCfg = MISSING

    # full-robot reference visualization, unmanaged by the scene articulation lifecycle
    robot_reference: AssetBaseCfg | None = None

    # motion reference
    motion_reference: MotionReferenceManagerCfg = MISSING  # type: ignore

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
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
    contact_forces = HierarchicalContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )

    def __post_init__(self):
        if type(self.motion_reference) is type(MISSING) or not self.motion_reference.debug_vis:
            delattr(self, "robot_reference")


@configclass
class BeyondMimicCommandsCfg:
    """BeyondMimic command configuration following their approach."""

    position_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="robot",
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
class BeyondMimicActionsCfg:
    """Action specifications for the BeyondMimic MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
    )


@configclass
class BeyondMimicObservationsCfg:
    """BeyondMimic observation configuration following their approach."""

    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        """Policy observations for BeyondMimic."""

        # BeyondMimic specific reference observations
        joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
        joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
        position_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        )
        rotation_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )

        # proprioception
        base_lin_vel = ObsTermCfg(
            func=mdp.base_lin_vel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticObsCfg(ObsGroupCfg):
        """Critic observations for BeyondMimic."""

        # BeyondMimic specific reference observations
        joint_pos_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        )
        joint_vel_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        )
        position_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
        )
        rotation_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
        )

        # proprioception
        link_pos = ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        )
        link_rot = ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        )
        base_lin_vel = ObsTermCfg(
            func=mdp.base_lin_vel,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
        )
        last_action = ObsTermCfg(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = PolicyObsCfg()
    critic = CriticObsCfg()


@configclass
class BeyondMimicRewardsCfg:
    """BeyondMimic reward terms following their approach."""

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
    undesired_contacts = RewTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class BeyondMimicSingleRewardsCfg(MultiRewardCfg):
    """Single reward configuration for BeyondMimic."""

    rewards: BeyondMimicRewardsCfg = BeyondMimicRewardsCfg()


@configclass
class BeyondMimicEventsCfg:
    """BeyondMimic events config such as termination conditions."""

    # startup
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
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

    # interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )

    # for motion initialization and reset
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
            "position_offset": [0.0, 0.0, 0.0],
            "dof_vel_ratio": 1.0,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            # Pose randomization (+-5cm position, +-6degrees rotation)
            "randomize_pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
            "randomize_velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
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


@configclass
class BeyondMimicCurriculumCfg:
    """BeyondMimic curriculum terms for the MDP."""

    beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
        func=instinct_mdp.BeyondMimicAdaptiveWeighting,
    )


@configclass
class BeyondMimicTerminationCfg:
    """BeyondMimic termination terms for the MDP."""

    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)
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
            "projected_gravity_threshold": 0.8,  # distance on z-axis of projected gravity
            "check_at_keyframe_threshold": -1,
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
class BeyondMimicMonitorCfg:
    """BeyondMimic monitor configuration."""

    dataset = MonitorTermCfg(
        func="instinctlab.monitors.monitors:MotionReferenceMonitorTerm",
        params=dict(
            asset_cfg=SceneEntityCfg("motion_reference"),
            sample_stat_interval=500,
            top_n_samples=5,
        ),
    )
    shadowing_position = MonitorTermCfg(
        func="instinctlab.monitors.monitors:ShadowingPositionMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            check_at_keyframe_threshold=0.03,
        ),
    )
    shadowing_rotation = MonitorTermCfg(
        func="instinctlab.monitors.monitors:ShadowingRotationMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_pos = MonitorTermCfg(
        func="instinctlab.monitors.monitors:ShadowingJointPosMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_vel = MonitorTermCfg(
        func="instinctlab.monitors.monitors:ShadowingJointVelMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_link_pos = MonitorTermCfg(
        func="instinctlab.monitors.monitors:ShadowingLinkPosMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            masking=True,
        ),
    )


@configclass
class BeyondMimicEnvCfg(InstinctLabRLEnvCfg):
    """Configuration for the BeyondMimic environment."""

    scene: BeyondMimicSceneCfg = BeyondMimicSceneCfg(num_envs=4096)
    actions: BeyondMimicActionsCfg = BeyondMimicActionsCfg()
    observations: BeyondMimicObservationsCfg = BeyondMimicObservationsCfg()
    commands: BeyondMimicCommandsCfg = BeyondMimicCommandsCfg()
    rewards: BeyondMimicSingleRewardsCfg = BeyondMimicSingleRewardsCfg()
    events: BeyondMimicEventsCfg = BeyondMimicEventsCfg()
    curriculum: BeyondMimicCurriculumCfg = BeyondMimicCurriculumCfg()
    terminations: BeyondMimicTerminationCfg = BeyondMimicTerminationCfg()
    monitors: BeyondMimicMonitorCfg = BeyondMimicMonitorCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1.0 / 50.0 / self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.sim.physics is None:
            from isaaclab_physx.physics import PhysxCfg

            self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)

        self.run_name = "BeyondMimic"
