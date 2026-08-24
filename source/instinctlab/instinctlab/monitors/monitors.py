from __future__ import annotations

import csv
import os
import torch
from torch.distributions import Multinomial
from typing import TYPE_CHECKING, Sequence

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from instinctlab.motion_reference.utils import (
    get_base_position_distance,
    get_base_rotation_distance,
    get_base_velocity_difference,
    get_joint_position_difference,
    get_joint_velocity_difference,
    get_link_position_distance,
    matching_reference_timing,
)
from instinctlab.utils.prims import resolve_articulation_root_expression

from .monitor_manager import MonitorSensor, MonitorTerm

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.motion_reference import MotionReferenceManager

    from .monitor_cfg import MonitorTermCfg, TorqueMonitorSensorCfg


class TorqueMonitorSensor(MonitorSensor):
    """Monitor sensor for torque values at each simulator step.
    NOTE: If update of joint_acc every decimation, it significantly decreases the performance.
    """

    def __init__(self, cfg: TorqueMonitorSensorCfg):
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> dict:
        self._update_outdated_buffers()
        return self.get_log()

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # self._torque_buffer[env_ids] = 0.0
        resolved_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        resolved_ids = wp.to_torch(resolved_mask).nonzero(as_tuple=False).squeeze(-1)
        self._step_idx[resolved_ids] = 0
        super().reset(env_mask=resolved_mask)

    def update(self, dt: float, force_recompute: bool = False):
        self._step_idx += 1
        self._step_idx %= self.cfg.history_length
        super().update(dt, force_recompute)

    def get_log(self, is_episode=False) -> dict[str, float]:
        self._update_outdated_buffers()
        torques = torch.abs(self._torque_buffer)
        log = dict()
        log["mean"] = torques.mean(dim=1).mean(dim=1)
        log["max"] = torques.max(dim=1)[0].max(dim=1)[0]
        log["min"] = torques.min(dim=1)[0].min(dim=1)[0]
        return log

    """
    Implementation specific
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # set access to the articulation we want to monitor
        root_expr = resolve_articulation_root_expression(self.cfg.prim_path)
        self._view = self._physics_sim_view.create_articulation_view(root_expr.replace(".*", "*"))
        if self._view._backend is None:
            raise RuntimeError(f"Failed to create a PhysX articulation view at: {root_expr}")
        # set the buffer
        self._torque_buffer = torch.zeros(
            self._view.count, self.cfg.history_length, self._view.max_dofs, dtype=torch.float32, device=self.device
        )
        # set the buffer index
        self._step_idx = torch.zeros(self._view.count, dtype=torch.int32, device=self.device)

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        torques = wp.to_torch(self._view.get_dof_actuation_forces())
        self._torque_buffer[env_ids, self._step_idx[env_ids]] = torques[env_ids]

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._view = None


class JointStatMonitorTerm(MonitorTerm):
    """Measuring the performance of the joints. E.g. joint acc, joint vel, joint pos, action rate, etc."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._computed_torque_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_acc = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_acc_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_vel = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_vel_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_pos = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._action_rate = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._action_rate_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)

        self._last_joint_vel = 0.0

    """
    Operations
    """

    def update(self, dt: float):
        asset = self._env.scene[self.cfg.params["asset_cfg"].name]
        # NOTE: the *_max buffers accumulate a running maximum within each episode (zeroed in
        # reset_idx). Do not overwrite them with the per-step value, otherwise transient spikes
        # that settle before the last step of a logging interval become invisible in the logs.
        self._computed_torque_max[:] = torch.maximum(
            self._computed_torque_max,
            torch.abs(asset.data.computed_torque.torch[:, self.cfg.params["asset_cfg"].joint_ids]).max(dim=-1).values,
        )
        self._joint_acc[:] = (
            torch.abs(
                asset.data.joint_vel.torch[:, self.cfg.params["asset_cfg"].joint_ids] - self._last_joint_vel
            ).mean(dim=-1)
            / dt
        )
        self._joint_acc_max[:] = torch.maximum(
            self._joint_acc_max,
            torch.abs(asset.data.joint_vel.torch[:, self.cfg.params["asset_cfg"].joint_ids] - self._last_joint_vel)
            .max(dim=-1)
            .values
            / dt,
        )
        self._joint_vel[:] = torch.abs(asset.data.joint_vel.torch[:, self.cfg.params["asset_cfg"].joint_ids]).mean(
            dim=-1
        )
        self._joint_vel_max[:] = torch.maximum(
            self._joint_vel_max,
            torch.abs(asset.data.joint_vel.torch[:, self.cfg.params["asset_cfg"].joint_ids]).max(dim=-1).values,
        )
        self._joint_pos[:] = torch.abs(
            asset.data.joint_pos.torch[:, self.cfg.params["asset_cfg"].joint_ids]
            - asset.data.default_joint_pos.torch[:, self.cfg.params["asset_cfg"].joint_ids]
        ).mean(dim=-1)
        self._action_rate[:] = torch.abs(self._env.action_manager.action - self._env.action_manager.prev_action).mean(
            dim=-1
        )
        action_diff = torch.abs(self._env.action_manager.action - self._env.action_manager.prev_action)
        try:
            self._action_rate_max[:] = torch.maximum(
                self._action_rate_max, action_diff[:, self.cfg.params["asset_cfg"].joint_ids].max(dim=-1).values
            )
        except IndexError:
            # If the action is not a joint action, we just count all actions
            self._action_rate_max[:] = torch.maximum(self._action_rate_max, action_diff.max(dim=-1).values)

        self._last_joint_vel = asset.data.joint_vel.torch[:, self.cfg.params["asset_cfg"].joint_ids].detach().clone()

    def reset_idx(self, env_ids: Sequence[int] | slice):
        """Nothing to do here."""
        self._joint_acc_max[env_ids] = 0.0
        self._joint_vel_max[env_ids] = 0.0
        self._action_rate_max[env_ids] = 0.0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {}
        else:
            return {
                "computed_torque_max": self._computed_torque_max.mean(),
                "joint_acc": self._joint_acc.mean(),
                "joint_acc_max": self._joint_acc_max.max(),
                "joint_vel": self._joint_vel.mean(),
                "joint_vel_max": self._joint_vel_max.max(),
                "joint_pos": self._joint_pos.mean(),
                "action_rate": self._action_rate.mean(),
                "action_rate_max": self._action_rate_max.max(),
            }


class RewardSumMonitorTerm(MonitorTerm):
    """The sum of the reward term in each timestep"""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reward_buf = env.reward_manager._reward_buf

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {}
        elif isinstance(self._reward_buf, dict):
            return {k: v.mean() for k, v in self._reward_buf.items()}
        else:
            return {"rewards": self._reward_buf.mean()}


class ActuatorMonitorTerm(MonitorTerm):
    """The monitor term that helps to monitor any single actuator in the environment. If multiple actuators are'
    selected, the mean value of the actuator is computed.
    """

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not (self._env.num_envs == 1):
            print("\033[93mWarning: ActuatorMonitorTerm is designed to work with a single environment. \033[0m")
        self.asset: Articulation = self._env.scene[self.cfg.params["asset_cfg"].name]
        self.asset_joint_ids = self.cfg.params["asset_cfg"].joint_ids
        self.torque_plot_scale = self.cfg.params.get(
            "torque_plot_scale", 1.0
        )  # to make the plot more readable with joint_pos in it.
        self.joint_vel_plot_scale = self.cfg.params.get(
            "joint_vel_plot_scale", 1.0
        )  # to make the plot more readable with joint_vel in it.
        self.joint_power_plot_scale = self.cfg.params.get(
            "joint_power_plot_scale", 1.0
        )  # to make the plot more readable with joint_power in it.
        self._joint_pos_cmd = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._applied_torque = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._computed_torque = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        # joint_pos as in the robot model (w.r.t urdf zero-position)
        self._joint_pos_skeleton = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        # joint_vel as in the robot model (w.r.t motion reference zero-position)
        self._joint_vel = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        # joint_power = applied_torque * joint_vel (could be negative)
        self._joint_power = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)

    def update(self, dt: float):
        """Update the monitor term."""
        self._joint_pos_cmd[:] = self.asset.data.joint_pos_target.torch[:, self.asset_joint_ids].mean(dim=-1)
        self._applied_torque[:] = self.asset.data.applied_torque.torch[:, self.asset_joint_ids].mean(dim=-1)
        self._computed_torque[:] = self.asset.data.computed_torque.torch[:, self.asset_joint_ids].mean(dim=-1)
        self._joint_pos_skeleton[:] = self.asset.data.joint_pos.torch[:, self.asset_joint_ids].mean(dim=-1)
        self._joint_vel[:] = self.asset.data.joint_vel.torch[:, self.asset_joint_ids].mean(dim=-1)
        self._joint_power[:] = (self._applied_torque * self._joint_vel).mean(dim=-1)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        """Reset the monitor term for the given environment ids."""
        self._joint_pos_cmd[env_ids] = 0.0
        self._applied_torque[env_ids] = 0.0
        self._computed_torque[env_ids] = 0.0
        self._joint_pos_skeleton[env_ids] = 0.0
        self._joint_vel[env_ids] = 0.0
        self._joint_power[env_ids] = 0.0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {}
        else:
            return {
                "zeros": 0.0,
                "joint_pos_cmd": self._joint_pos_cmd.mean(),
                "applied_torque": self._applied_torque.mean() * self.torque_plot_scale,
                "computed_torque": self._computed_torque.mean() * self.torque_plot_scale,
                "joint_pos_skeleton": self._joint_pos_skeleton.mean(),
                "joint_vel": self._joint_vel.mean() * self.joint_vel_plot_scale,
                "joint_power": self._joint_power.mean() * self.joint_power_plot_scale,
            }


class ContactForceMonitorTerm(MonitorTerm):
    """Monitors net contact forces on selected sensor bodies. Logs mean and max force norms per step."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from isaaclab.sensors import ContactSensor

        self._contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self._body_ids = cfg.params["sensor_cfg"].body_ids
        self._force_mean = torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
        self._force_max = torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)

    def update(self, dt: float):
        net_forces = self._contact_sensor.data.net_forces_w_history.torch
        # net_forces shape: (num_envs, history_length, num_bodies, 3)
        force_norms = torch.norm(net_forces[:, :, self._body_ids], dim=-1)
        # max over history, then stats over selected bodies
        force_norms_max_hist = force_norms.max(dim=1).values  # (num_envs, num_selected_bodies)
        self._force_mean[:] = force_norms_max_hist.mean(dim=-1)
        self._force_max[:] = force_norms_max_hist.max(dim=-1).values

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._force_mean[env_ids] = 0.0
        self._force_max[env_ids] = 0.0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {}
        return {
            "force_mean": self._force_mean.mean(),
            "force_max": self._force_max.max(),
        }


class BodyStatMonitorTerm(MonitorTerm):
    """Measuring the performance of the body. E.g. body acc, body vel, body pos, action rate, etc."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._body_acc = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._body_acc_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._body_vel = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._body_vel_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)

        self._last_body_vel = 0.0

    """
    Operations
    """

    def update(self, dt: float):
        asset = self._env.scene[self.cfg.params["asset_cfg"].name]
        self._body_acc[:] = (
            torch.norm(
                asset.data.body_vel_w.torch[:, self.cfg.params["asset_cfg"].body_ids] - self._last_body_vel, dim=-1
            ).mean(dim=-1)
            / dt
        )
        self._body_acc_max[:] = torch.maximum(
            self._body_acc_max,
            torch.norm(
                asset.data.body_vel_w.torch[:, self.cfg.params["asset_cfg"].body_ids] - self._last_body_vel, dim=-1
            )
            .max(dim=-1)
            .values
            / dt,
        )
        self._body_vel[:] = torch.norm(
            asset.data.body_vel_w.torch[:, self.cfg.params["asset_cfg"].body_ids], dim=-1
        ).mean(dim=-1)
        self._body_vel_max[:] = torch.maximum(
            self._body_vel_max,
            torch.norm(asset.data.body_vel_w.torch[:, self.cfg.params["asset_cfg"].body_ids], dim=-1)
            .max(dim=-1)
            .values,
        )

        self._last_body_vel = asset.data.body_vel_w.torch[:, self.cfg.params["asset_cfg"].body_ids].detach().clone()

    def reset_idx(self, env_ids: Sequence[int] | slice):
        """Nothing to do here."""
        self._body_acc_max[env_ids] = 0.0
        self._body_vel_max[env_ids] = 0.0

    def get_log(self, is_episode=False):
        if is_episode:
            return {}
        else:
            return {
                "body_acc": self._body_acc.mean(),
                "body_acc_max": self._body_acc_max.max(),
                "body_vel": self._body_vel.mean(),
                "body_vel_max": self._body_vel_max.max(),
            }


class MotionReferenceMonitorTerm(MonitorTerm):
    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.num_env_reset = 0
        self.num_motion_reference_exhausted = 0.0  # (conditioned on the envs marked as reset)

    """
    Operations
    """

    def update(self, dt: float):
        pass

    @torch.no_grad()
    def reset_idx(self, env_ids: Sequence[int]):
        motion_reference = self._env.scene[self.cfg.params["asset_cfg"].name]
        self.num_env_reset = len(env_ids)
        self.num_motion_reference_exhausted = torch.sum(
            torch.logical_not(motion_reference.data.validity[env_ids].any(dim=-1)).to(torch.int)
        )

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "exhausted_count": self.num_motion_reference_exhausted,
                "exhausted_ratio": (
                    self.num_motion_reference_exhausted / self.num_env_reset if self.num_env_reset > 0 else float("nan")
                ),
            }
        else:
            should_compute_sample_stat = (
                self._env._sim_step_counter // self._env.cfg.decimation
            ) % self.cfg.params.get(
                "sample_stat_interval", 10
            ) == 0  # type: ignore
            stat = {}
            motion_reference = self._env.scene[self.cfg.params["asset_cfg"].name]
            if should_compute_sample_stat and hasattr(motion_reference, "_buffer_sample_weights"):
                distribution = Multinomial(probs=motion_reference._buffer_sample_weights)
                stat["sample_entropy"] = distribution.entropy()
                if self.cfg.params.get("top_n_samples", 0) > 0:  # type: ignore
                    top_n_idx = torch.topk(
                        motion_reference._buffer_sample_weights, self.cfg.params["top_n_samples"]  # type: ignore
                    ).indices
                    stat["top_n_samples"] = ",".join(
                        [
                            os.path.basename(
                                motion_reference._all_motion_files[motion_reference._buffer_motion_selection[i]]
                            )
                            for i in top_n_idx
                        ]
                    )
            return stat


class ShadowingPositionMonitorTerm(MonitorTerm):
    """Measuring the closeness of the shadowing effect, using the mean position error of the links and the DOFs across
    the trajectory.
    """

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not "robot_cfg" in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")
        if not "motion_reference_cfg" in self.cfg.params:
            self.cfg.params["motion_reference_cfg"] = SceneEntityCfg("motion_reference")

        self._base_pos_error = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_pos_error_currently = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_pos_error_xy = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_pos_error_z = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_pos_error_z_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._num_frames_should_reach = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self.device)
        self._all_ones = torch.ones(self._env.num_envs, dtype=torch.float32, device=self.device)

    """
    Operations
    """

    def update(self, dt: float):
        in_frame_mask = matching_reference_timing(
            self._env,
            self._all_ones,
            self._env.scene[self.cfg.params["motion_reference_cfg"].name],
            check_at_keyframe_threshold=self.cfg.params.get("check_at_keyframe_threshold", -1),  # type: ignore
            multiply_by_frame_interval=False,
        )

        root_diff = get_base_position_distance(
            self._env,
            self.cfg.params["robot_cfg"],
            self.cfg.params["motion_reference_cfg"],
            masking=True,
            squared=False,
            return_diff=True,
        )
        self._base_pos_error += torch.norm(root_diff, dim=-1) * in_frame_mask
        self._base_pos_error_currently = torch.norm(root_diff, dim=-1) * in_frame_mask
        self._base_pos_error_xy += torch.norm(root_diff[:, :2], dim=-1) * in_frame_mask
        self._base_pos_error_z += torch.abs(root_diff[:, 2]) * in_frame_mask
        self._base_pos_error_z_max = torch.maximum(
            self._base_pos_error_z_max, torch.abs(root_diff[:, 2]) * in_frame_mask
        )

        self._num_frames_should_reach += in_frame_mask.to(torch.int)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._episodic_base_pos_error = self._base_pos_error[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_base_pos_error_xy = self._base_pos_error_xy[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_base_pos_error_z = self._base_pos_error_z[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_base_pos_error_z_max = self._base_pos_error_z_max[env_ids].clone()

        self._base_pos_error[env_ids] = 0.0
        self._base_pos_error_xy[env_ids] = 0.0
        self._base_pos_error_z[env_ids] = 0.0
        self._base_pos_error_z_max[env_ids] = 0.0
        self._num_frames_should_reach[env_ids] = 0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "base_pos_error": self._episodic_base_pos_error.nanmean().item(),
                "base_pos_error_xy": self._episodic_base_pos_error_xy.nanmean().item(),
                "base_pos_error_z": self._episodic_base_pos_error_z.nanmean().item(),
                "base_pos_error_z_max": self._episodic_base_pos_error_z_max.max().item(),
            }
        else:
            return {
                "base_pos_error": (self._base_pos_error / self._num_frames_should_reach).nanmean().item(),
                "base_pos_error_currently": self._base_pos_error_currently.nanmean().item(),
                "base_pos_error_xy": (self._base_pos_error_xy / self._num_frames_should_reach).nanmean().item(),
                "base_pos_error_z": (self._base_pos_error_z / self._num_frames_should_reach).nanmean().item(),
                "base_pos_error_z_max": self._base_pos_error_z_max.max().item(),
            }


class ShadowingRotationMonitorTerm(MonitorTerm):
    """Measuring the performance of the robot's rotation during shadowing."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not "robot_cfg" in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")
        if not "motion_reference_cfg" in self.cfg.params:
            self.cfg.params["motion_reference_cfg"] = SceneEntityCfg("motion_reference")

        self._base_rot_error = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_rot_error_currently = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_rot_error_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._num_frames_should_reach = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self.device)
        self._all_ones = torch.ones(self._env.num_envs, dtype=torch.float32, device=self.device)

    """
    Operations
    """

    def update(self, dt: float):
        in_frame_mask = matching_reference_timing(
            self._env,
            self._all_ones,
            self._env.scene[self.cfg.params["motion_reference_cfg"].name],
            check_at_keyframe_threshold=self.cfg.params.get("check_at_keyframe_threshold", -1),  # type: ignore
            multiply_by_frame_interval=False,
        )

        _base_rot_error = get_base_rotation_distance(
            self._env,
            self.cfg.params["robot_cfg"],
            self.cfg.params["motion_reference_cfg"],
            masking=self.cfg.params.get("masking", True),  # type: ignore
        )
        self._base_rot_error += _base_rot_error * in_frame_mask
        self._base_rot_error_currently = _base_rot_error * in_frame_mask

        self._base_rot_error_max = torch.maximum(self._base_rot_error_max, _base_rot_error * in_frame_mask)

        self._num_frames_should_reach += in_frame_mask.to(torch.int)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._episodic_base_rot_error = self._base_rot_error[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_base_rot_error_max = self._base_rot_error_max[env_ids].clone()

        self._base_rot_error[env_ids] = 0.0
        self._base_rot_error_max[env_ids] = 0.0
        self._num_frames_should_reach[env_ids] = 0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "base_rot_error": self._episodic_base_rot_error.nanmean().item(),
                "base_rot_error_max": self._episodic_base_rot_error_max.max().item(),
            }
        else:
            return {
                "base_rot_error": (self._base_rot_error / self._num_frames_should_reach).nanmean().item(),
                "base_rot_error_currently": self._base_rot_error_currently.nanmean().item(),
                "base_rot_error_max": self._base_rot_error_max.max().item(),
            }


class ShadowingVelocityMonitorTerm(MonitorTerm):
    """Measuring the velocity difference between the robot and the motion reference."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not "robot_cfg" in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")
        if not "motion_reference_cfg" in self.cfg.params:
            self.cfg.params["motion_reference_cfg"] = SceneEntityCfg("motion_reference")

        self._base_vel_error = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_vel_error_currently = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._base_vel_error_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._num_frames_should_reach = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self.device)
        self._all_ones = torch.ones(self._env.num_envs, dtype=torch.float32, device=self.device)

    """
    Operations
    """

    def update(self, dt: float):
        in_frame_mask = matching_reference_timing(
            self._env,
            self._all_ones,
            self._env.scene[self.cfg.params["motion_reference_cfg"].name],
            check_at_keyframe_threshold=self.cfg.params.get("check_at_keyframe_threshold", -1),  # type: ignore
            multiply_by_frame_interval=False,
        )

        _base_vel_error = get_base_velocity_difference(
            self._env,
            self.cfg.params["robot_cfg"],
            self.cfg.params["motion_reference_cfg"],
            masking=self.cfg.params.get("masking", True),  # type: ignore
            anchor_frame=self.cfg.params.get("anchor_frame", "world"),  # type: ignore
        )
        self._base_vel_error += _base_vel_error * in_frame_mask
        self._base_vel_error_currently = _base_vel_error * in_frame_mask

        self._base_vel_error_max = torch.maximum(self._base_vel_error_max, _base_vel_error * in_frame_mask)

        self._num_frames_should_reach += in_frame_mask.to(torch.int)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._episodic_base_vel_error = self._base_vel_error[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_base_vel_error_max = self._base_vel_error_max[env_ids].clone()

        self._base_vel_error[env_ids] = 0.0
        self._base_vel_error_max[env_ids] = 0.0
        self._num_frames_should_reach[env_ids] = 0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "base_vel_error": self._episodic_base_vel_error.nanmean().item(),
                "base_vel_error_max": self._episodic_base_vel_error_max.max().item(),
            }
        else:
            return {
                "base_vel_error": (self._base_vel_error / self._num_frames_should_reach).nanmean().item(),
                "base_vel_error_currently": self._base_vel_error_currently.nanmean().item(),
                "base_vel_error_max": self._base_vel_error_max.max().item(),
            }


class ShadowingJointPosMonitorTerm(MonitorTerm):
    """Measuring the performance of the robot's joint positions during shadowing."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not "robot_cfg" in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")
        if not "motion_reference_cfg" in self.cfg.params:
            self.cfg.params["motion_reference_cfg"] = SceneEntityCfg("motion_reference")

        self._joint_pos_error = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_pos_error_currently = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_pos_error_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._num_frames_should_reach = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self.device)
        self._all_ones = torch.ones(self._env.num_envs, dtype=torch.float32, device=self.device)

    """
    Operations
    """

    def update(self, dt: float):
        in_frame_mask = matching_reference_timing(
            self._env,
            self._all_ones,
            self._env.scene[self.cfg.params["motion_reference_cfg"].name],
            check_at_keyframe_threshold=self.cfg.params.get("check_at_keyframe_threshold", -1),  # type: ignore
            multiply_by_frame_interval=False,
        )

        joint_diff = get_joint_position_difference(
            self._env,
            self.cfg.params["robot_cfg"],
            self.cfg.params["motion_reference_cfg"],
            masking=True,
        )
        # NOTE: If joint_names is assigned, we only compute the error for those joints.
        if not isinstance(self.cfg.params["motion_reference_cfg"].joint_ids, slice):
            joint_diff = joint_diff[:, self.cfg.params["motion_reference_cfg"].joint_ids]  # type: ignore
        if not isinstance(self.cfg.params["robot_cfg"].joint_ids, slice):
            joint_diff = joint_diff[:, self.cfg.params["robot_cfg"].joint_ids]
        self._joint_pos_error += torch.abs(joint_diff).mean(dim=-1) * in_frame_mask
        self._joint_pos_error_currently = torch.abs(joint_diff).mean(dim=-1) * in_frame_mask
        self._joint_pos_error_max = torch.maximum(
            self._joint_pos_error_max, torch.abs(joint_diff).max(dim=-1).values * in_frame_mask
        )

        self._num_frames_should_reach += in_frame_mask.to(torch.int)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._episodic_joint_pos_error = self._joint_pos_error[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_joint_pos_error_max = self._joint_pos_error_max[env_ids].clone()

        self._joint_pos_error[env_ids] = 0.0
        self._joint_pos_error_max[env_ids] = 0.0
        self._num_frames_should_reach[env_ids] = 0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "joint_pos_error": self._episodic_joint_pos_error.nanmean().item(),
                "joint_pos_error_max": self._episodic_joint_pos_error_max.max().item(),
            }
        else:
            return {
                "joint_pos_error": (self._joint_pos_error / self._num_frames_should_reach).nanmean().item(),
                "joint_pos_error_currently": self._joint_pos_error_currently.nanmean().item(),
                "joint_pos_error_max": self._joint_pos_error_max.max().item(),
            }


class ShadowingJointVelMonitorTerm(MonitorTerm):
    """Measuring the performance of the robot's joint velocities during shadowing."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not "robot_cfg" in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")
        if not "motion_reference_cfg" in self.cfg.params:
            self.cfg.params["motion_reference_cfg"] = SceneEntityCfg("motion_reference")

        self._joint_vel_error = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_vel_error_currently = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._joint_vel_error_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._num_frames_should_reach = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self.device)
        self._all_ones = torch.ones(self._env.num_envs, dtype=torch.float32, device=self.device)

    """
    Operations
    """

    def update(self, dt: float):
        in_frame_mask = matching_reference_timing(
            self._env,
            self._all_ones,
            self._env.scene[self.cfg.params["motion_reference_cfg"].name],
            check_at_keyframe_threshold=self.cfg.params.get("check_at_keyframe_threshold", -1),  # type: ignore
            multiply_by_frame_interval=False,
        )

        joint_diff = get_joint_velocity_difference(
            self._env,
            self.cfg.params["robot_cfg"],
            self.cfg.params["motion_reference_cfg"],
            masking=True,
        )
        # NOTE: If joint_names is assigned, we only compute the error for those joints.
        if not isinstance(self.cfg.params["motion_reference_cfg"].joint_ids, slice):
            joint_diff = joint_diff[:, self.cfg.params["motion_reference_cfg"].joint_ids]
        if not isinstance(self.cfg.params["robot_cfg"].joint_ids, slice):
            joint_diff = joint_diff[:, self.cfg.params["robot_cfg"].joint_ids]
        self._joint_vel_error += torch.abs(joint_diff).mean(dim=-1) * in_frame_mask
        self._joint_vel_error_currently = torch.abs(joint_diff).mean(dim=-1) * in_frame_mask
        self._joint_vel_error_max = torch.maximum(
            self._joint_vel_error_max, torch.abs(joint_diff).max(dim=-1).values * in_frame_mask
        )
        self._num_frames_should_reach += in_frame_mask.to(torch.int)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._episodic_joint_vel_error = self._joint_vel_error[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_joint_vel_error_max = self._joint_vel_error_max[env_ids].clone()

        self._joint_vel_error[env_ids] = 0.0
        self._joint_vel_error_max[env_ids] = 0.0
        self._num_frames_should_reach[env_ids] = 0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "joint_vel_error": self._episodic_joint_vel_error.nanmean().item(),
                "joint_vel_error_max": self._episodic_joint_vel_error_max.max().item(),
            }
        else:
            return {
                "joint_vel_error": (self._joint_vel_error / self._num_frames_should_reach).nanmean().item(),
                "joint_vel_error_currently": self._joint_vel_error_currently.nanmean().item(),
                "joint_vel_error_max": self._joint_vel_error_max.max().item(),
            }


class ShadowingLinkPosMonitorTerm(MonitorTerm):
    """Measuring the performance of the robot's link positions during shadowing."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not "robot_cfg" in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")
        if not "motion_reference_cfg" in self.cfg.params:
            self.cfg.params["motion_reference_cfg"] = SceneEntityCfg("motion_reference")

        self._link_pos_error = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._link_pos_error_currently = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._link_pos_error_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._num_frames_should_reach = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self.device)
        self._all_ones = torch.ones(self._env.num_envs, dtype=torch.float32, device=self.device)

    """
    Operations
    """

    def update(self, dt: float):
        in_frame_mask = matching_reference_timing(
            self._env,
            self._all_ones,
            self._env.scene[self.cfg.params["motion_reference_cfg"].name],
            check_at_keyframe_threshold=self.cfg.params.get("check_at_keyframe_threshold", -1),  # type: ignore
            multiply_by_frame_interval=False,
        )

        link_distance = get_link_position_distance(
            self._env,
            self.cfg.params["robot_cfg"],
            self.cfg.params["motion_reference_cfg"],
            in_base_frame=self.cfg.params.get("in_base_frame", False),  # type: ignore
            masking=True,
        )
        # NOTE: If link_names is assigned, we only compute the error for those links.
        # NOTE: Because the links are usually only a subset of the robot's, in the motion reference, we only accept the
        # body_names from the motion reference. DO Remember to set `preserve_order` to True.
        if not isinstance(self.cfg.params["motion_reference_cfg"].body_ids, slice):
            link_distance = link_distance[:, self.cfg.params["motion_reference_cfg"].body_ids]

        self._link_pos_error += link_distance.mean(dim=-1) * in_frame_mask
        self._link_pos_error_currently = link_distance.mean(dim=-1) * in_frame_mask
        self._link_pos_error_max = torch.maximum(
            self._link_pos_error_max, link_distance.max(dim=-1).values * in_frame_mask
        )
        self._num_frames_should_reach += in_frame_mask.to(torch.int)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._episodic_link_pos_error = self._link_pos_error[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_link_pos_error_max = self._link_pos_error_max[env_ids].clone()

        self._link_pos_error[env_ids] = 0.0
        self._link_pos_error_max[env_ids] = 0.0
        self._num_frames_should_reach[env_ids] = 0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "link_pos_error": self._episodic_link_pos_error.nanmean().item(),
                "link_pos_error_max": self._episodic_link_pos_error_max.max().item(),
            }
        else:
            return {
                "link_pos_error": (self._link_pos_error / self._num_frames_should_reach).nanmean().item(),
                "link_pos_error_currently": self._link_pos_error_currently.nanmean().item(),
                "link_pos_error_max": self._link_pos_error_max.max().item(),
            }


class ShadowingProgressMonitorTerm(MonitorTerm):
    """Monitor term to write the motion identifier to a specified file."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._motion_reference: MotionReferenceManager = self._env.scene[
            self.cfg.params.get("reference_cfg", SceneEntityCfg("motion_reference")).name
        ]
        self._output_file: str = self.cfg.params.get("output_file", None)  # type: ignore

        self._is_success = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._is_failed = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._progress = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)

        self._output_file_writer: csv.DictWriter | None = None
        if self._output_file is not None:
            self._output_file_writer = csv.DictWriter(
                open(self._output_file, "w", newline=""),
                fieldnames=["total_step", "motion_identifier", "is_success", "is_failed", "progress"],
            )
            self._output_file_writer.writeheader()

    """
    Operations
    """

    def reset_idx(self, env_ids: Sequence[int] | slice):
        """Reset the success ratio for the given environment IDs."""
        if isinstance(env_ids, slice):
            env_ids = list(range(self._env.num_envs))[env_ids]

        total_system_steps = self._env.common_step_counter
        motion_identifiers = self._motion_reference.get_current_motion_identifiers(env_ids)
        self._sampled_motion_count = len(set(motion_identifiers))
        self._sampled_motion_vs_len_envs = self._sampled_motion_count / len(env_ids)

        progress = (self._env.episode_length_buf * self._env.step_dt)[
            env_ids
        ] / self._motion_reference.assigned_motion_lengths[env_ids]
        is_success = progress >= self.cfg.params.get("success_threshold", 0.9)
        is_failed = progress < self.cfg.params.get("success_threshold", 0.9)

        self._is_success[env_ids] = is_success.float()
        self._is_failed[env_ids] = is_failed.float()
        self._progress[env_ids] = progress

        if (
            total_system_steps % self.cfg.params.get("write_interval", 100)  # type: ignore
            < self.cfg.params.get("write_interval_offset", 0)
            and total_system_steps > 0
            and self._output_file_writer is not None
        ):
            motion_weights = self._motion_reference.get_current_motion_weights()
            self._motion_entropy = torch.distributions.Categorical(motion_weights).entropy().item()

            if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != 0:
                return

            # write to file
            for idx, env_id in enumerate(env_ids):  # type: ignore
                self._output_file_writer.writerow(
                    {
                        "total_step": total_system_steps,
                        "motion_identifier": motion_identifiers[idx],
                        "is_success": self._is_success[env_id].item(),
                        "is_failed": self._is_failed[env_id].item(),
                        "progress": progress[idx].item(),
                    }
                )

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if not is_episode:
            return {
                "progress_mean": self._progress.mean().item() if self._progress.numel() > 0 else 0.0,
                "progress_std": self._progress.std().item() if self._progress.numel() > 1 else 0.0,
                "progress_max": self._progress.max().item() if self._progress.numel() > 0 else 0.0,
            }
        else:
            is_success = self._is_success[self._is_success > 0.0]
            is_failed = self._is_failed[self._is_failed > 0.0]
            progress = self._progress[self._progress > 0.0]
            return_ = {
                "is_success": is_success.mean().item() if is_success.numel() > 0 else 0.0,
                "is_failed": is_failed.mean().item() if is_failed.numel() > 0 else 0.0,
                "progress_mean": progress.mean().item() if progress.numel() > 0 else 0.0,
                "progress_std": progress.std().item() if progress.numel() > 1 else 0.0,
                "progress_max": progress.max().item() if progress.numel() > 0 else 0.0,
                "progress_min": progress.min().item() if progress.numel() > 0 else 0.0,
                "motion_entropy": getattr(self, "_motion_entropy", 0.0),
                "sampled_motion_count": getattr(self, "_sampled_motion_count", 0),
                "sampled_motion_vs_len_envs": getattr(self, "_sampled_motion_vs_len_envs", 0.0),
            }
            self._is_success[:] = 0.0
            self._is_failed[:] = 0.0
            self._progress[:] = 0.0
            return return_


class ShadowingJointReferenceMonitorTerm(MonitorTerm):
    """Monitoring the current motion reference case with the reference data."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not (self._env.num_envs == 1):
            print(
                "\033[93mWarning: ShadowingJointReferenceMonitorTerm is designed to work with a single environment."
                " \033[0m"
            )
        self._motion_reference: MotionReferenceManager = self._env.scene[
            self.cfg.params.get("reference_cfg", SceneEntityCfg("motion_reference")).name
        ]
        self._reference_joint_ids = self.cfg.params["reference_cfg"].joint_ids
        self._joint_pos_ref = torch.zeros(
            self._env.num_envs, len(self._reference_joint_ids), dtype=torch.float32, device=self.device
        )
        self._joint_vel_ref = torch.zeros(
            self._env.num_envs, len(self._reference_joint_ids), dtype=torch.float32, device=self.device
        )

    def update(self, dt: float):
        self._joint_pos_ref[:] = self._motion_reference.reference_frame.joint_pos[:, 0, self._reference_joint_ids]
        self._joint_vel_ref[:] = self._motion_reference.reference_frame.joint_vel[:, 0, self._reference_joint_ids]

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        return {
            "joint_pos_ref": self._joint_pos_ref.mean().item(),
            "joint_vel_ref": self._joint_vel_ref.mean().item(),
        }


class ShadowingBasePosMonitorTerm(MonitorTerm):
    """Monitoring the current motion reference case with the reference data."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not (self._env.num_envs == 1):
            print("\033[93mWarning: ShadowingBasePosMonitorTerm is designed to work with a single environment. \033[0m")
        self._motion_reference: MotionReferenceManager = self._env.scene[
            self.cfg.params.get("reference_cfg", SceneEntityCfg("motion_reference")).name
        ]
        self._robot: Articulation = self._env.scene[self.cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
        self._robot_base_pos = torch.zeros(self._env.num_envs, 3, dtype=torch.float32, device=self.device)
        self._reference_base_pos = torch.zeros(self._env.num_envs, 3, dtype=torch.float32, device=self.device)

    def update(self, dt: float):
        self._robot_base_pos[:] = self._robot.data.root_pos_w.torch
        self._reference_base_pos[:] = self._motion_reference.reference_frame.base_pos_w[:, 0]

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        return {
            "robot_base_pos_x": self._robot_base_pos[:, 0].mean().item(),
            "robot_base_pos_y": self._robot_base_pos[:, 1].mean().item(),
            "robot_base_pos_z": self._robot_base_pos[:, 2].mean().item(),
            "reference_base_pos_x": self._reference_base_pos[:, 0].mean().item(),
            "reference_base_pos_y": self._reference_base_pos[:, 1].mean().item(),
            "reference_base_pos_z": self._reference_base_pos[:, 2].mean().item(),
        }


class ShadowingGravityMonitorTerm(MonitorTerm):
    """Monitors the projected gravity distance between the robot and the motion reference."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if "robot_cfg" not in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")
        if "motion_reference_cfg" not in self.cfg.params:
            self.cfg.params["motion_reference_cfg"] = SceneEntityCfg("motion_reference")

        self._pg_diff = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._pg_diff_currently = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._pg_diff_max = torch.zeros(self._env.num_envs, dtype=torch.float32, device=self.device)
        self._num_frames_should_reach = torch.zeros(self._env.num_envs, dtype=torch.int32, device=self.device)
        self._all_ones = torch.ones(self._env.num_envs, dtype=torch.float32, device=self.device)

    """
    Operations
    """

    def update(self, dt: float):
        motion_reference: MotionReferenceManager = self._env.scene[self.cfg.params["motion_reference_cfg"].name]

        in_frame_mask = matching_reference_timing(
            self._env,
            self._all_ones,
            motion_reference,
            check_at_keyframe_threshold=self.cfg.params.get("check_at_keyframe_threshold", -1),  # type: ignore
            multiply_by_frame_interval=False,
        )

        robot: Articulation = self._env.scene[self.cfg.params["robot_cfg"].name]
        rot = robot.data.root_state_w.torch[:, 3:7]
        ref_rot = motion_reference.data.base_quat_w
        ref_rot = ref_rot[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

        pg = math_utils.quat_apply_inverse(rot, robot.data.GRAVITY_VEC_W.torch)
        ref_pg = math_utils.quat_apply_inverse(ref_rot, robot.data.GRAVITY_VEC_W.torch)

        if self.cfg.params.get("z_only", False):
            diff = (pg[:, 2] - ref_pg[:, 2]).abs()
        else:
            diff = torch.norm(pg - ref_pg, dim=-1)

        self._pg_diff += diff * in_frame_mask
        self._pg_diff_currently = diff * in_frame_mask
        self._pg_diff_max = torch.maximum(self._pg_diff_max, diff * in_frame_mask)
        self._num_frames_should_reach += in_frame_mask.to(torch.int)

    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._episodic_pg_diff = self._pg_diff[env_ids] / self._num_frames_should_reach[env_ids]
        self._episodic_pg_diff_max = self._pg_diff_max[env_ids].clone()

        self._pg_diff[env_ids] = 0.0
        self._pg_diff_max[env_ids] = 0.0
        self._num_frames_should_reach[env_ids] = 0

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {
                "pg_dist_mean": self._episodic_pg_diff.nanmean().item(),
                "pg_dist_max": self._episodic_pg_diff_max.max().item(),
            }
        else:
            return {
                "pg_dist_mean": (self._pg_diff / self._num_frames_should_reach).nanmean().item(),
                "pg_dist_currently": self._pg_diff_currently.nanmean().item(),
                "pg_dist_max": self._pg_diff_max.max().item(),
            }


class BaseQuaternionMonitorTerm(MonitorTerm):
    """Logs the robot's base quaternion (x, y, z, w) mean across environments."""

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if "robot_cfg" not in self.cfg.params:
            self.cfg.params["robot_cfg"] = SceneEntityCfg("robot")

    def get_log(self, is_episode=False) -> dict[str, float | torch.Tensor]:
        if is_episode:
            return {}
        asset: Articulation = self._env.scene[self.cfg.params["robot_cfg"].name]
        quat = asset.data.root_quat_w.torch
        return {
            "x": quat[:, 0].mean().item(),
            "y": quat[:, 1].mean().item(),
            "z": quat[:, 2].mean().item(),
            "w": quat[:, 3].mean().item(),
        }
