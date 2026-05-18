import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as parkour_mdp
import instinctlab.tasks.HSI.perceptive_downstream_gen.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import ActuatorMonitorTerm, MonitorTermCfg
from instinctlab.sensors import get_link_prim_targets

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG
# Must match frozen VAE bundle (dagger run name uses propHistory4_depthHist10Skip3).
PROPRIO_HISTORY_LENGTH = 4
TEACHER_PROPRIO_HISTORY_LENGTH = 8


@configclass
class CommandsCfg:
    """Velocity / pose-style commands (same family as parkour ``base_velocity``)."""

    base_velocity = parkour_mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05,
        ranges=parkour_mdp.PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
        random_velocity_terrain=None,
        # Keys must match ``sub_terrains`` in ``PerceptiveShadowingSceneCfg.terrain`` (perceptive_env_cfg.py).
        velocity_ranges={
            "boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            # "mesh_boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
        },
        only_positive_lin_vel_x=True,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
        target_dis_threshold=0.4,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        depth_image = ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised_history",
                "history_skip_frames": 2,
            },
        )

        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        velocity_commands = ObsTermCfg(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
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
        last_action = ObsTermCfg(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticObsCfg(ObsGroupCfg):
        height_scan = ObsTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=[-20.0, 20.0],
        )

        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        velocity_commands = ObsTermCfg(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTermCfg(func=mdp.last_action, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class G1PerceptiveVaePlayMonitorCfg:
    right_ankle_pitch_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
        ),
    )
    left_ankle_pitch_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
        ),
    )
    right_knee_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
        ),
    )
    left_knee_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
        ),
    )


@configclass
class G1PerceptiveVaeRewardsCfg(perceptual_cfg.RewardsCfg):
    """Parkour-aligned task rewards in addition to base downstream regularizers."""

    track_lin_vel_xy_exp = RewTermCfg(
        func=parkour_mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTermCfg(
        func=parkour_mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    heading_error = RewTermCfg(
        func=parkour_mdp.heading_error,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    dont_wait = RewTermCfg(
        func=parkour_mdp.dont_wait,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )
    stand_still = RewTermCfg(
        func=parkour_mdp.stand_still,
        weight=-0.3,
        params={"command_name": "base_velocity", "offset": 4.0},
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
class G1PerceptiveVaeRewardGroupsCfg(perceptual_cfg.RewardGroupsCfg):
    rewards: G1PerceptiveVaeRewardsCfg = G1PerceptiveVaeRewardsCfg()


@configclass
class G1PerceptiveVaeTerminationsCfg(perceptual_cfg.TerminationsCfg):
    """Extra safety terminations aligned with parkour MDP."""
    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTermCfg(func=parkour_mdp.bad_orientation, params={"limit_angle": 1.0})
    root_height = DoneTermCfg(
        func=parkour_mdp.root_height_below_env_origin_minimum,
        params={"minimum_height": 0.5},
    )

    out_of_border = DoneTermCfg(
        func=instinct_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
    )


@configclass
class G1PerceptiveVaeEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: G1PerceptiveVaeRewardGroupsCfg = G1PerceptiveVaeRewardGroupsCfg()
    terminations: G1PerceptiveVaeTerminationsCfg = G1PerceptiveVaeTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # PoseVelocityCommand requires flat patch sampling key ``target`` on terrain sub-terrains.
        _target_patch = FlatPatchSamplingCfg(
            num_patches=50,
            patch_radius=[0.05, 0.10, 0.15, 0.20],
            max_height_diff=0.05,
        )
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            new_sub = {}
            for name, sub_terrain in tg.sub_terrains.items():
                existing = getattr(sub_terrain, "flat_patch_sampling", None) or {}
                if "target" not in existing:
                    merged = dict(existing)
                    merged["target"] = _target_patch
                    new_sub[name] = sub_terrain.replace(flat_patch_sampling=merged)
                else:
                    new_sub[name] = sub_terrain
            self.scene.terrain.terrain_generator = tg.replace(sub_terrains=new_sub)

        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))
        self.scene.camera.data_histories["distance_to_image_plane_noised"] = 10
        self.observations.policy.depth_image.params["history_skip_frames"] = 3
        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale

        self.run_name = "g1PerceptiveVaeGen" + "".join(
            [
                f"_propHistory{PROPRIO_HISTORY_LENGTH}",
                f"_depthHist{self.scene.camera.data_histories['distance_to_image_plane_noised']}Skip{self.observations.policy.depth_image.params['history_skip_frames']}",
            ]
        )


@configclass
class G1PerceptiveVaeEnvCfg_PLAY(G1PerceptiveVaeEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=10,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    monitors: G1PerceptiveVaePlayMonitorCfg | None = G1PerceptiveVaePlayMonitorCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=[0.0, 2.0, 2.5],
        lookat=[0.0, 0.0, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator.num_rows = 13
        self.scene.terrain.terrain_generator.num_cols = 13

        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True

        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None

        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
