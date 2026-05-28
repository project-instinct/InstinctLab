import math
import os
from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
import numpy as np
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
PROPRIO_HISTORY_LENGTH = 5
TEACHER_PROPRIO_HISTORY_LENGTH = 8
_WBCHSI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../.."))
_MANUAL_ROOT_TRAJECTORY_ENV = "INSTINCTLAB_MANUAL_ROOT_TRAJECTORY"
_DEFAULT_MANUAL_ROOT_TRAJECTORY_PATH = os.path.join(
    _WBCHSI_ROOT,
    "data",
    "root_pose_trajs",
    "Instinct-HSIDownstreamClimb-Perceptive-Vae-G1-Play-v0_20260528_140950",
    "trajectory.npz",
)


def _resolve_manual_root_trajectory_path(path: str | None = None) -> str:
    path = os.environ.get(_MANUAL_ROOT_TRAJECTORY_ENV, path or _DEFAULT_MANUAL_ROOT_TRAJECTORY_PATH)
    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isabs(path):
        cwd_path = os.path.abspath(path)
        wbchsi_path = os.path.abspath(os.path.join(_WBCHSI_ROOT, path))
        path = cwd_path if os.path.exists(cwd_path) else wbchsi_path
    return path


def _manual_root_trajectory_duration_s(path: str | None = None) -> float | None:
    path = _resolve_manual_root_trajectory_path(path)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        dt = float(np.asarray(data["dt"]).item()) if "dt" in data.files else 0.02
        return int(data["root_pos_w"].shape[0]) * dt


def _wrap_to_pi(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


class _ManualRootTrajectoryReferenceManager(ManagerTermBase):
    """Root pose reference loaded from an interactively collected ``trajectory.npz``."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._trajectory_path = _resolve_manual_root_trajectory_path(cfg.params.get("trajectory_path"))
        if not os.path.exists(self._trajectory_path):
            raise FileNotFoundError(
                f"Manual root trajectory not found: {self._trajectory_path}. "
                f"Set {_MANUAL_ROOT_TRAJECTORY_ENV}=/path/to/trajectory.npz to override."
            )

        with np.load(self._trajectory_path) as data:
            root_pos_w = np.asarray(data["root_pos_w"], dtype=np.float32)
            if "root_yaw_w" in data.files:
                root_yaw_w = np.asarray(data["root_yaw_w"], dtype=np.float32)
            else:
                root_quat_w = torch.as_tensor(np.asarray(data["root_quat_w"], dtype=np.float32))
                root_yaw_w = math_utils.euler_xyz_from_quat(root_quat_w)[2].cpu().numpy().astype(np.float32)
            env_origin_w = (
                np.asarray(data["env_origin_w"], dtype=np.float32)
                if "env_origin_w" in data.files
                else np.zeros(3, dtype=np.float32)
            )
            dt = float(np.asarray(data["dt"]).item()) if "dt" in data.files else float(env.step_dt)

        if root_pos_w.ndim != 2 or root_pos_w.shape[1] != 3:
            raise ValueError(f"Expected root_pos_w with shape [N, 3], got {root_pos_w.shape}.")
        if root_pos_w.shape[0] < 2:
            raise ValueError("Manual root trajectory must contain at least two frames.")

        self._local_root_pos = torch.as_tensor(root_pos_w - env_origin_w.reshape(1, 3), device=env.device)
        self._root_yaw = torch.as_tensor(root_yaw_w, device=env.device)
        self._dt = max(dt, 1.0e-6)
        self._num_frames = self._local_root_pos.shape[0]

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        pass

    def _sample_indices(self, env, lookahead_s: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt + lookahead_s
        frame_f = torch.clamp(elapsed_s / self._dt, min=0.0, max=float(self._num_frames - 1))
        idx0 = torch.floor(frame_f).to(dtype=torch.long)
        idx1 = torch.clamp(idx0 + 1, max=self._num_frames - 1)
        alpha = (frame_f - idx0.to(dtype=torch.float32)).unsqueeze(-1)
        return idx0, idx1, alpha

    def _target_root_state(self, env, lookahead_s: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        idx0, idx1, alpha = self._sample_indices(env, lookahead_s)
        local_pos = self._local_root_pos[idx0] * (1.0 - alpha) + self._local_root_pos[idx1] * alpha

        yaw0 = self._root_yaw[idx0]
        yaw1 = self._root_yaw[idx1]
        yaw = _wrap_to_pi(yaw0 + alpha.squeeze(-1) * _wrap_to_pi(yaw1 - yaw0))

        env_origins = getattr(env.scene, "env_origins", None)
        if env_origins is None:
            env_origins = torch.zeros_like(local_pos)
        return env_origins + local_pos, yaw

    def _root_pose_error(
        self,
        env,
        asset_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset = env.scene[asset_cfg.name]
        target_pos_w, target_yaw_w = self._target_root_state(env)
        pos_error = asset.data.root_pos_w[:, :3] - target_pos_w
        if hasattr(asset.data, "heading_w"):
            yaw_w = asset.data.heading_w
        else:
            yaw_w = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
        yaw_error = _wrap_to_pi(yaw_w - target_yaw_w)
        return pos_error, yaw_error, target_pos_w

    def _update_pose_velocity_command(
        self,
        env,
        command_name: str | None,
        command_lookahead_s: float,
        command_max_speed: float,
    ) -> None:
        if not command_name or not hasattr(env, "command_manager"):
            return
        try:
            command_term = env.command_manager.get_term(command_name)
        except Exception:
            return
        target_pos_w, _ = self._target_root_state(env, lookahead_s=command_lookahead_s)
        if hasattr(command_term, "pos_command_w"):
            command_term.pos_command_w[:] = target_pos_w
        if hasattr(command_term, "max_command_b"):
            command_term.max_command_b[:, 0] = command_max_speed
            command_term.max_command_b[:, 1] = command_max_speed
        if hasattr(command_term, "is_standing_env"):
            command_term.is_standing_env[:] = False

    def _trajectory_finished(self, env) -> torch.Tensor:
        return env.episode_length_buf >= (self._num_frames - 1)


class _RootPosReferenceManager(ManagerTermBase):
    """Speed-integrated root XY reference from reset pose toward ``PoseVelocityCommand`` goal."""

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

    def _ensure_reference(self) -> None:
        missing_env_ids = torch.logical_not(self._initialized).nonzero(as_tuple=False).flatten()
        if len(missing_env_ids) > 0:
            self._capture_reference(missing_env_ids)

    def _target_root_xy(self, env) -> torch.Tensor:
        self._ensure_reference()
        goal_vec_xy = self._goal_root_xy - self._start_root_xy
        goal_dist = torch.norm(goal_vec_xy, dim=-1)
        goal_dir_xy = goal_vec_xy / torch.clamp(goal_dist, min=1.0e-6).unsqueeze(-1)
        elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
        progress = torch.minimum(torch.clamp(self._lin_vel_x, min=0.0) * elapsed_s, goal_dist)
        return self._start_root_xy + goal_dir_xy * progress.unsqueeze(-1)

    def _xy_tracking_error(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        target_root_xy = self._target_root_xy(env)
        return torch.norm(asset.data.root_pos_w[:, :2] - target_root_xy, dim=-1)

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


class root_pos_termination(_RootPosReferenceManager):
    """Terminate when root XY drifts too far from the reset-root speed-integrated reference."""

    def __call__(
        self,
        env,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_xy_error: float = 1.0,
        print_reason: bool = False,
    ) -> torch.Tensor:
        error = self._xy_tracking_error(env, asset_cfg)
        return_ = error > max_xy_error
        if print_reason and return_.any():
            print(f"root_pos_termination: {return_.sum()} envs")
        return return_


class root_pos_track_xy_exp(_RootPosReferenceManager):
    """Reward tracking the speed-integrated root XY reference; closer root yields higher reward."""

    def __call__(
        self,
        env,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        std: float = 0.25,
    ) -> torch.Tensor:
        error = self._xy_tracking_error(env, asset_cfg)
        return torch.exp(-torch.square(error) / std**2)


class manual_root_trajectory_track_xyz_yaw_exp(_ManualRootTrajectoryReferenceManager):
    """Reward tracking an interactively collected root XYZ+yaw trajectory."""

    def __call__(
        self,
        env,
        trajectory_path: str | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        xy_std: float = 0.30,
        z_std: float = 0.12,
        yaw_std: float = 0.50,
        pos_weight: float = 0.7,
        yaw_weight: float = 0.3,
        command_name: str | None = "base_velocity",
        command_lookahead_s: float = 0.30,
        command_max_speed: float = 0.8,
    ) -> torch.Tensor:
        self._update_pose_velocity_command(env, command_name, command_lookahead_s, command_max_speed)
        pos_error, yaw_error, _ = self._root_pose_error(env, asset_cfg)
        xy_error_sq = torch.sum(torch.square(pos_error[:, :2]), dim=-1)
        z_error_sq = torch.square(pos_error[:, 2])
        pos_reward = torch.exp(-xy_error_sq / xy_std**2 - z_error_sq / z_std**2)
        yaw_reward = torch.exp(-torch.square(yaw_error) / yaw_std**2)
        return pos_weight * pos_reward + yaw_weight * yaw_reward


class manual_root_trajectory_termination(_ManualRootTrajectoryReferenceManager):
    """Terminate on trajectory completion or excessive root pose tracking error."""

    def __call__(
        self,
        env,
        trajectory_path: str | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_xy_error: float = 1.0,
        max_z_error: float = 0.4,
        max_yaw_error: float = math.pi,
        terminate_on_end: bool = True,
        print_reason: bool = False,
    ) -> torch.Tensor:
        pos_error, yaw_error, _ = self._root_pose_error(env, asset_cfg)
        return_ = torch.norm(pos_error[:, :2], dim=-1) > max_xy_error
        return_ |= torch.abs(pos_error[:, 2]) > max_z_error
        return_ |= torch.abs(yaw_error) > max_yaw_error
        if terminate_on_end:
            return_ |= self._trajectory_finished(env)
        if print_reason and return_.any():
            print(f"manual_root_trajectory_termination: {return_.sum()} envs")
        return return_


class manual_root_trajectory_commands(_ManualRootTrajectoryReferenceManager):
    """Command-like observation derived from the manual root trajectory.

    The output shape intentionally stays ``[num_envs, 3]`` to remain compatible with the existing
    ``velocity_commands`` observation slot: desired XY velocity in robot base frame plus yaw error.
    """

    def __call__(
        self,
        env,
        trajectory_path: str | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        lookahead_s: float = 0.30,
        max_speed: float = 0.8,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        target_pos_w, target_yaw_w = self._target_root_state(env, lookahead_s=lookahead_s)
        target_vec_w = target_pos_w - asset.data.root_pos_w[:, :3]
        target_vec_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), target_vec_w)
        desired_xy_b = target_vec_b[:, :2] / max(lookahead_s, 1.0e-6)
        speed = torch.norm(desired_xy_b, dim=-1, keepdim=True)
        desired_xy_b = desired_xy_b * torch.clamp(max_speed / torch.clamp(speed, min=1.0e-6), max=1.0)
        if hasattr(asset.data, "heading_w"):
            yaw_w = asset.data.heading_w
        else:
            yaw_w = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
        yaw_error = _wrap_to_pi(target_yaw_w - yaw_w).unsqueeze(-1)
        return torch.cat([desired_xy_b, yaw_error], dim=-1)


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
            lin_vel_x=(0.1, 0.1), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
        random_velocity_terrain=None,
        velocity_ranges={
            "specified_box": {"lin_vel_x": (0.1, 0.1), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
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
            func=manual_root_trajectory_commands,
            history_length=8,
            flatten_history_dim=True,
            params={
                "trajectory_path": None,
                "asset_cfg": SceneEntityCfg("robot"),
                "lookahead_s": 0.30,
                "max_speed": 0.8,
            },
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
            func=manual_root_trajectory_commands,
            history_length=8,
            flatten_history_dim=True,
            params={
                "trajectory_path": None,
                "asset_cfg": SceneEntityCfg("robot"),
                "lookahead_s": 0.30,
                "max_speed": 0.8,
            },
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

    track_manual_root_trajectory = RewTermCfg(
        func=manual_root_trajectory_track_xyz_yaw_exp,
        weight=2.0,
        params={
            "trajectory_path": None,
            "asset_cfg": SceneEntityCfg("robot"),
            "xy_std": 0.20,
            "z_std": 0.10,
            "yaw_std": 0.50,
            "pos_weight": 0.7,
            "yaw_weight": 0.3,
            "command_name": "base_velocity",
            "command_lookahead_s": 0.30,
            "command_max_speed": 0.8,
        },
    )

    # track_lin_vel_xy_exp = RewTermCfg(
    #     func=parkour_mdp.track_lin_vel_xy_exp,
    #     weight=2.0,
    #     params={"command_name": "base_velocity", "std": 0.5},
    # )
    # track_ang_vel_z_exp = RewTermCfg(
    #     func=parkour_mdp.track_ang_vel_z_exp,
    #     weight=2.0,
    #     params={"command_name": "base_velocity", "std": 0.5},
    # )
    # heading_error = RewTermCfg(func=parkour_mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"})
    # dont_wait = RewTermCfg(func=parkour_mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    # is_alive = RewTermCfg(func=parkour_mdp.is_alive, weight=3.0)
    # stand_still = RewTermCfg(
    #     func=parkour_mdp.stand_still, weight=-0.3, params={"command_name": "base_velocity", "offset": 4.0}
    # )
    # action_rate_l2 = RewTermCfg(func=parkour_mdp.action_rate_l2, weight=-0.005)
    # energy = RewTermCfg(
    #     func=parkour_mdp.motors_power_square,
    #     weight=-5e-5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
    #         "normalize_by_stiffness": True,
    #     },
    # )

    # dof_pos_limits = RewTermCfg(
    #     func=parkour_mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    # )
    # dof_vel_limits = RewTermCfg(
    #     func=parkour_mdp.joint_vel_limits,
    #     weight=-1.0,
    #     params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    # )
    # torque_limits = RewTermCfg(
    #     func=parkour_mdp.applied_torque_limits_by_ratio,
    #     weight=-0.01,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "limit_ratio": 0.8,
    #     },
    # )

@configclass
class G1PerceptiveVaeRewardGroupsCfg(perceptual_cfg.RewardGroupsCfg):
    rewards: G1PerceptiveVaeRewardsCfg = G1PerceptiveVaeRewardsCfg()


@configclass
class G1PerceptiveVaeTerminationsCfg(perceptual_cfg.TerminationsCfg):
    """Extra safety terminations aligned with parkour MDP."""
    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)

    # bad_orientation = DoneTermCfg(func=parkour_mdp.bad_orientation, params={"limit_angle": 1.5})
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
        params={"minimum_height": 0.6},
    )
    manual_root_trajectory_termination = DoneTermCfg(
        func=manual_root_trajectory_termination,
        time_out=False,
        params={
            "trajectory_path": None,
            "asset_cfg": SceneEntityCfg("robot"),
            "max_xy_error": 0.3,
            "max_z_error": 0.2,
            "max_yaw_error": math.pi,
            "terminate_on_end": True,
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
        manual_traj_duration_s = _manual_root_trajectory_duration_s()
        if manual_traj_duration_s is not None:
            self.episode_length_s = manual_traj_duration_s

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
