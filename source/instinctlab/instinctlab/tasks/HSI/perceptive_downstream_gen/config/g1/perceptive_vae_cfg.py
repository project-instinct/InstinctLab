import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as parkour_mdp  # terrain / termination utilities only
import instinctlab.tasks.HSI.perceptive_downstream_gen.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import ActuatorMonitorTerm, MonitorTermCfg
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.HSI.perceptive_downstream_gen.commands import RootPositionTrajectoryCommandCfg


def root_position_command_relative(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World command position minus robot root world position ``(num_envs, 3)``."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return cmd - asset.data.root_pos_w


def track_root_pos_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    err_xy = torch.sum(torch.square(asset.data.root_pos_w[:, :2] - cmd[:, :2]), dim=-1)
    return torch.exp(-err_xy / std**2)


def root_position_command_error_too_large(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_xy_error: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when planar root XY is farther than ``max_xy_error`` from the commanded world position."""
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    dist_xy = torch.norm(asset.data.root_pos_w[:, :2] - cmd[:, :2], dim=-1)
    return dist_xy > max_xy_error


G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG
# Must match frozen VAE bundle (dagger run name uses propHistory4_depthHist10Skip3).
PROPRIO_HISTORY_LENGTH = 4
TEACHER_PROPRIO_HISTORY_LENGTH = 8


@configclass
class CommandsCfg:
    """World-frame root XY trajectory over flat-patch waypoint chain (see RootPositionTrajectoryCommand)."""

    root_position = RootPositionTrajectoryCommandCfg(
        asset_name="robot",
        resampling_time_range=(30.0, 45.0),
        debug_vis=False,
        speed=0.2,
        num_waypoints=8,
        max_waypoints_stored=64,
        target_patch_key="target",
        arrival_tolerance=0.02,
        command_z_offset_from_root=0.0,
        pos_metrics_std=0.35,
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

        root_position_commands = ObsTermCfg(
            func=root_position_command_relative,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "root_position"},
            noise=None,
        )
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

        root_position_commands = ObsTermCfg(
            func=root_position_command_relative,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "root_position"},
            noise=None,
        )
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
    """Rewards aligned with ``RootPositionTrajectoryCommand`` (world XY position + planner velocity)."""

    track_root_pos_xy_exp = RewTermCfg(
        func=track_root_pos_xy_exp,
        weight=2.0,
        params={"command_name": "root_position", "std": 0.35},
    )
    is_alive = RewTermCfg(func=mdp.is_alive, weight=10.0)
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

    root_position_tracking_lost = DoneTermCfg(
        func=root_position_command_error_too_large,
        time_out=True,
        params={
            "command_name": "root_position",
            "max_xy_error": 2.0,
        },
    )
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

        # RootPositionTrajectoryCommand uses ``terrain.flat_patches["target"]``; ensure patch sampling exists.
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

        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1

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
