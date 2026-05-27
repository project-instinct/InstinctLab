import math
from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs import ViewerCfg
from isaaclab.managers import ManagerTermBase
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
import instinctlab.tasks.HSI.perceptive_downstream_climb.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_delayed_actuators,
)
from instinctlab.monitors import ActuatorMonitorTerm, MonitorTermCfg
from instinctlab.sensors import get_link_prim_targets

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG
# Must match frozen VAE bundle (dagger run name uses propHistory4_depthHist10Skip3).
PROPRIO_HISTORY_LENGTH = 4
TEACHER_PROPRIO_HISTORY_LENGTH = 8


class root_pos_termination(ManagerTermBase):
    """Terminate when root XY drifts too far from the reset-root speed-integrated reference."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._start_root_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self._goal_root_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self._lin_vel_x = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._capture_reference(env_ids)

    def __call__(
        self,
        env,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_xy_error: float = 1.0,
        print_reason: bool = False,
    ) -> torch.Tensor:
        missing_env_ids = torch.logical_not(self._initialized).nonzero(as_tuple=False).flatten()
        if len(missing_env_ids) > 0:
            self._capture_reference(missing_env_ids)

        asset = env.scene[asset_cfg.name]
        goal_vec_xy = self._goal_root_xy - self._start_root_xy
        goal_dist = torch.norm(goal_vec_xy, dim=-1)
        goal_dir_xy = goal_vec_xy / torch.clamp(goal_dist, min=1.0e-6).unsqueeze(-1)
        elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
        progress = torch.minimum(torch.clamp(self._lin_vel_x, min=0.0) * elapsed_s, goal_dist)
        target_root_xy = self._start_root_xy + goal_dir_xy * progress.unsqueeze(-1)

        error = torch.norm(asset.data.root_pos_w[:, :2] - target_root_xy, dim=-1)
        return_ = error > max_xy_error
        if print_reason and return_.any():
            print(f"root_pos_termination: {return_.sum()} envs")
        return return_

    def _capture_reference(self, env_ids: Sequence[int] | slice) -> None:
        asset = self._env.scene[self.cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        command_name = self.cfg.params.get("command_name", "base_velocity")
        command_term = self._env.command_manager.get_term(command_name)

        self._start_root_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]
        self._goal_root_xy[env_ids] = command_term.pos_command_w[env_ids, :2]

        lin_vel_x = command_term.max_command_b[:, 0]
        if hasattr(command_term, "random_velocity_indices"):
            lin_vel_x = torch.where(command_term.random_velocity_indices, command_term.random_lin_vel_x, lin_vel_x)
        if hasattr(command_term, "is_standing_env"):
            lin_vel_x = torch.where(command_term.is_standing_env, torch.zeros_like(lin_vel_x), lin_vel_x)
        self._lin_vel_x[env_ids] = lin_vel_x[env_ids]
        self._initialized[env_ids] = True


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
            lin_vel_x=(0.4, 0.8), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
        random_velocity_terrain=None,
        velocity_ranges={
            "specified_box": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
        },
        target_mode="fixed_goal",
        relative_target_pos=(10.0, 0.0, 0.0),
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
            func=instinct_mdp.delayed_visualizable_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised_history",
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
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
class G1PerceptiveVaeRewardsCfg:
    """Reward terms copied from ``parkour_env_cfg.G1Rewards`` (parkour locomotion MDP)."""

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
    heading_error = RewTermCfg(func=parkour_mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"})
    dont_wait = RewTermCfg(func=parkour_mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    is_alive = RewTermCfg(func=parkour_mdp.is_alive, weight=3.0)
    stand_still = RewTermCfg(
        func=parkour_mdp.stand_still, weight=-0.3, params={"command_name": "base_velocity", "offset": 4.0}
    )
    action_rate_l2 = RewTermCfg(func=parkour_mdp.action_rate_l2, weight=-0.005)
    energy = RewTermCfg(
        func=parkour_mdp.motors_power_square,
        weight=-5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            "normalize_by_stiffness": True,
        },
    )

    dof_pos_limits = RewTermCfg(
        func=parkour_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_limits = RewTermCfg(
        func=parkour_mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    torque_limits = RewTermCfg(
        func=parkour_mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )

@configclass
class G1PerceptiveVaeRewardGroupsCfg(perceptual_cfg.RewardGroupsCfg):
    rewards: G1PerceptiveVaeRewardsCfg = G1PerceptiveVaeRewardsCfg()


@configclass
class G1PerceptiveVaeTerminationsCfg(perceptual_cfg.TerminationsCfg):
    """Extra safety terminations aligned with parkour MDP."""
    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTermCfg(func=parkour_mdp.bad_orientation, params={"limit_angle": 1.5})
    # body_pos_default = DoneTermCfg(
    #     func=instinct_mdp.bad_global_body_pos_from_default,
    #     time_out=False,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "threshold": 0.5,
    #         "disable_flag": False,
    #     },
    # )
    root_height = DoneTermCfg(
        func=parkour_mdp.root_height_below_env_origin_minimum,
        params={"minimum_height": 0.5},
    )
    root_pos_termination = DoneTermCfg(
        func=root_pos_termination,
        time_out=False,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "max_xy_error": 0.5,
            "print_reason": False,
        },
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
            num_patches=5,
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
        self.scene.camera.data_histories["distance_to_image_plane_noised"] = 37
        self.observations.policy.depth_image.params["history_skip_frames"] = 5
        self.scene.robot.actuators = beyondmimic_g1_29dof_delayed_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale

        self.run_name = "g1PerceptiveVaeClimb" + "".join(
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

        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2

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
