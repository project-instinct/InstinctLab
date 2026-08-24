from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

import instinctlab.utils.math as instinct_math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.motion_reference import MotionReferenceManager

    from .commands_cfg import (
        BaseHeightRefCommandCfg,
        BaseLinVelRefCommandCfg,
        HeadingErrorRefCommandCfg,
        HeadingRefCommandCfg,
        JointPosErrRefCommandCfg,
        JointPosRefCommandCfg,
        JointVelRefCommandCfg,
        LinkPosErrRefCommandCfg,
        LinkPosRefCommandCfg,
        LinkRefCommandCfg,
        LinkRotErrRefCommandCfg,
        LinkRotRefCommandCfg,
        PoseRefCommandCfg,
        PositionRefCommandCfg,
        ProjectedGravityRefCommandCfg,
        RotationRefCommandCfg,
        ShadowingCommandBaseCfg,
        TimeToTargetCommandCfg,
    )


class ShadowingCommandBase(CommandTerm):
    """ """

    cfg: ShadowingCommandBaseCfg

    def __init__(self, cfg: ShadowingCommandBaseCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # self.robot: Articulation = env.scene[cfg.asset_name]
        self._motion_reference: MotionReferenceManager = env.scene[cfg.motion_reference.name]
        cfg.asset_cfg.resolve(env.scene)
        robot = env.scene[cfg.asset_cfg.name]
        self._asset_joint_ids = cfg.asset_cfg.joint_ids
        if cfg.asset_cfg.joint_names is None:
            self._motion_reference_joint_ids = slice(None)
        else:
            resolved_joint_names = [robot.joint_names[index] for index in self._asset_joint_ids]
            self._motion_reference_joint_ids, reference_joint_names = self._motion_reference.find_joints(
                resolved_joint_names, preserve_order=True
            )
            if resolved_joint_names != reference_joint_names:
                raise ValueError("Robot and motion-reference joint selections do not match.")
        self._command: torch.Tensor = None  # override in the child class # type: ignore
        self._mask: torch.Tensor = None  # override in the child class # type: ignore

    @property
    def command(self) -> torch.Tensor:
        """Return the command tensor (not flattened)."""
        if self.cfg.current_state_command:
            return torch.cat([self._command, self._current_state], dim=1)
        return self._command  # type: ignore

    @property
    def mask(self) -> torch.Tensor:
        """Return the mask for the command tensor.
        ## NOTE:
            The mask should return an additional ones in time dimension if the current_state_command is True.
        """
        raise NotImplementedError(
            "The mask property is not implemented in the base class.",
        )

    @property
    def ALL_INDICES(self):
        """A proxy property to access ALL_INDICES to read the target motion/command at current timestep."""
        return self._motion_reference.ALL_INDICES

    @property
    def aiming_frame_idx(self):
        """A proxy property to access the aiming frame index of the motion reference.
        ## NOTE:
            a fun feature, when aiming_frame_idx becomes -1 (invalid), command_term.command[ALL_INDICES, aiming_frame_idx]
            will return the last frame of the motion reference (aka current state).
        """
        return self._motion_reference.aiming_frame_idx

    @property
    def link_of_interests(self) -> Sequence[str] | None:
        """Return the link of interest in the motion reference."""
        return self._motion_reference.cfg.link_of_interests

    @property
    def time_to_aiming_frame(self) -> torch.Tensor:
        """A proxy property to access the time to the aiming frame of the motion reference."""
        return self._motion_reference.time_to_aiming_frame

    @property
    def _motion_reference_updated_env_ids(self) -> Sequence[int] | torch.Tensor:
        """Return the environment indices that need to be updated, considering the motion reference
        (as a sensor) might updates during simulation step, but each command term is updated once in
        each env step.
        """
        return torch.where(self._motion_reference.time_passed_from_update < (self._env.step_dt - 1e-6))[0]

    def _update_metrics(self):
        """Update the metrics based on the motion reference and time."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Does not depend on the command term. self._command always depend on motion reference."""
        pass

    def _update_command(self):
        """Update the command based on the motion reference and time."""
        self._motion_reference.data  # trigger the update of the motion reference data

        # update the real-time mode for the position
        if getattr(self.cfg, "realtime_mode", False):
            env_ids = self._motion_reference.ALL_INDICES
        else:
            env_ids = self._motion_reference_updated_env_ids
        if len(env_ids) > 0:
            self._update_command_by_env_ids(env_ids)
        if getattr(self, "_visualizer", None) is not None:
            self._compute_debug_vis_data()

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the command term for the given environment indices."""
        self._update_command_by_env_ids(env_ids)
        return {}  # no metrics to return, override if needed.

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """Update the command based on the motion reference and time, for the given environment indices."""
        raise NotImplementedError

    def _debug_vis_callback(self, event):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis and (getattr(self.cfg, "visualizer_cfg", None) is not None):
            if not hasattr(self, "_visualizer"):
                self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self._visualizer.set_visibility(True)
        else:
            if hasattr(self, "_visualizer"):
                self._visualizer.set_visibility(False)

    def _compute_debug_vis_data(self):
        """Compute and update the data for visualization."""
        pass


class PoseRefCommand(ShadowingCommandBase):
    """ """

    cfg: PoseRefCommandCfg

    def __init__(self, cfg: PoseRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # update self.cfg.realtime_mode for backward compatibility
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # generate the command tensor buffer
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (3 + 4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3 + 3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (3 + 6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )  # (n_envs, num_frames, 3 + (4 or 3))
        self._mask = torch.ones(
            (
                self.num_envs,
                self._motion_reference.num_frames,
                4,
            ),  # (plane mask, height mask, orientation mask, heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 4),  # (plane mask, height mask, orientation mask, heading mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_pos_plane_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_pos_height_mask)
        mask[..., 2] = torch.logical_and(mask[..., 2], self._motion_reference.data.base_orientation_mask)
        mask[..., 3] = torch.logical_and(mask[..., 3], self._motion_reference.data.base_heading_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command(self):
        """Update the command based on the motion reference and time."""
        self._motion_reference.data  # trigger the update of the motion reference data

        # update the real-time mode for the position
        if self.cfg.realtime_mode == 1:
            env_ids = self._motion_reference.ALL_INDICES
        else:
            env_ids = self._motion_reference_updated_env_ids
        if len(env_ids) > 0:
            self._update_command_by_env_ids(env_ids)
        if getattr(self, "_visualizer", None) is not None:
            self._compute_debug_vis_data()

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        if self.cfg.anchor_frame == "robot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids],
            )
        elif self.cfg.anchor_frame == "reference":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.data.base_pos_w[env_ids, 0],
                self._motion_reference.data.base_quat_w[env_ids, 0],
            )
        self._command[env_ids, :, :3] = math_utils.transform_points(
            self._motion_reference.data.base_pos_w[env_ids],
            anchor_pos_w_inv,
            anchor_quat_w_inv,
        )

        # update the real-time mode for the rotation
        if self.cfg.realtime_mode == -1:
            env_ids = self._motion_reference.ALL_INDICES
        base_quat_b = math_utils.quat_mul(
            anchor_quat_w_inv.unsqueeze(1).expand(-1, self._motion_reference.num_frames, -1),
            self._motion_reference.data.base_quat_w[env_ids],
        )
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids, :, 3:7] = base_quat_b
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids, :, 3:6] = math_utils.axis_angle_from_quat(base_quat_b)
        elif self.cfg.rotation_mode == "euler":
            base_quat_b_ = base_quat_b.flatten(0, 1)
            euler_b_ = torch.stack(math_utils.euler_xyz_from_quat(base_quat_b_), dim=-1)
            self._command[env_ids, :, 3:6] = euler_b_.view(self._command[env_ids, :, 3:6].shape)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids, :, 3:9] = instinct_math_utils.quat_to_tan_norm(base_quat_b)

        self._command[env_ids, :, :3] *= torch.logical_or(
            self._motion_reference.data.base_pos_height_mask[env_ids],
            self._motion_reference.data.base_pos_plane_mask[env_ids],
        ).unsqueeze(-1) * self._mask[env_ids, :, 0:2].any(dim=-1, keepdim=True)
        self._command[env_ids, :, 3:] *= torch.logical_or(
            self._motion_reference.data.base_orientation_mask[env_ids],
            self._motion_reference.data.base_heading_mask[env_ids],
        ).unsqueeze(-1) * self._mask[env_ids, :, 2:4].any(dim=-1, keepdim=True)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)

        if self.cfg.current_state_command:
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, 6] = 1.0
            elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
                pass
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, 3:9] = 0
                self._current_state[env_ids, :, 3] = 1
                self._current_state[env_ids, :, 8] = 1

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_pose_ref_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_pose_ref_pos_w,
            orientations=self._vis_pose_ref_quat_w,
        )

    def _compute_debug_vis_data(self):
        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch

        # Since this command is defined in robot's local frame, we need to transform
        # the command data to the world frame. Even if it is not refreshed timely.
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        pose_ref_pos_w = math_utils.transform_points(
            keyframe_command[:, :3].unsqueeze(1),
            root_pos_w,
            root_quat_w,
        ).squeeze(1)
        if self.cfg.rotation_mode == "quaternion":
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                keyframe_command[:, 3:7],
            )
        elif self.cfg.rotation_mode == "axis_angle":
            pose_ref_quat_b = math_utils.quat_from_angle_axis(
                torch.norm(keyframe_command[:, 3:6], dim=-1),
                keyframe_command[:, 3:6],
            )
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                pose_ref_quat_b,
            )
        elif self.cfg.rotation_mode == "euler":
            pose_ref_quat_b = math_utils.quat_from_euler_xyz(
                keyframe_command[:, 3],
                keyframe_command[:, 4],
                keyframe_command[:, 5],
            )
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                pose_ref_quat_b,
            )
        elif self.cfg.rotation_mode == "tannorm":
            pose_ref_quat_b = instinct_math_utils.tan_norm_to_quat(keyframe_command[:, 3:9])
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                pose_ref_quat_b,
            )
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._vis_pose_ref_pos_w = pose_ref_pos_w
        self._vis_pose_ref_quat_w = pose_ref_quat_w


class PositionRefCommand(ShadowingCommandBase):
    cfg: PositionRefCommandCfg

    def __init__(self, cfg: PositionRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # update self.cfg.realtime_mode for backward compatibility
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # generate the command tensor buffer
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 3),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 2),  # (plane mask, height mask)
            device=self.device,
            dtype=torch.bool,
        )
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 3),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 2),  # (plane mask, height mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_pos_plane_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_pos_height_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        if self.cfg.anchor_frame == "robot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids],
            )
        elif self.cfg.anchor_frame == "reference":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.reference_frame.base_pos_w[env_ids, 0],
                self._motion_reference.reference_frame.base_quat_w[env_ids, 0],
            )
        elif self.cfg.anchor_frame == "ref_rot_robot_pos":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids],
                self._motion_reference.data.base_quat_w[env_ids, 0],
            )
        elif self.cfg.anchor_frame == "ref_pos_robot_rot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.data.base_pos_w[env_ids, 0],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids],
            )
        self._command[env_ids] = (
            math_utils.transform_points(
                self._motion_reference.data.base_pos_w[env_ids],
                anchor_pos_w_inv,
                anchor_quat_w_inv,
            )
            * torch.logical_or(
                self._motion_reference.data.base_pos_height_mask[env_ids],
                self._motion_reference.data.base_pos_plane_mask[env_ids],
            ).unsqueeze(-1)
            * self._mask[env_ids].any(dim=-1, keepdim=True)
        )

    def _debug_vis_callback(self, event):
        robot_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch
        _vis_point_w: torch.tensor = math_utils.transform_points(
            self._command,  # (num_envs, num_frames, 3)
            robot_pos_w,
            robot_quat_w,
        )  # (num_envs, num_frames, 3)
        _scales = torch.ones_like(_vis_point_w)
        _scales[:, 1:] *= 0.5
        self._visualizer.visualize(
            translations=_vis_point_w.flatten(0, 1),
            scales=_scales.flatten(0, 1),
        )


class RotationRefCommand(ShadowingCommandBase):
    """The sequential command that only contains the rotation part of the pose command."""

    cfg: RotationRefCommandCfg

    def __init__(self, cfg: RotationRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class."""
        super().__init__(cfg, env)
        # update self.cfg.realtime_mode for backward compatibility
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # generate the command tensor buffer
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )  # (n_envs, num_frames, 3 + (4 or 3))
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 2),  # (orientation mask, heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 2),  # (orientation mask, heading mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_orientation_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_heading_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        _, robot_quat_w_inv = math_utils.subtract_frame_transforms(
            self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids],
            self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids],
        )

        if self.cfg.in_base_frame:
            quat_reference = math_utils.quat_mul(
                robot_quat_w_inv.unsqueeze(1).expand(-1, self._motion_reference.num_frames, -1),
                self._motion_reference.data.base_quat_w[env_ids],
            )
        else:
            quat_reference = self._motion_reference.data.base_quat_w[env_ids]

        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids] = quat_reference
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids] = math_utils.axis_angle_from_quat(quat_reference)
        elif self.cfg.rotation_mode == "euler":
            quat_reference_ = quat_reference.flatten(0, 1)
            euler_ = torch.stack(math_utils.euler_xyz_from_quat(quat_reference_), dim=-1)
            self._command[env_ids] = euler_.view(self._command[env_ids].shape)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids] = instinct_math_utils.quat_to_tan_norm(quat_reference)

        self._command[env_ids] *= torch.logical_or(
            self._motion_reference.data.base_orientation_mask[env_ids],
            self._motion_reference.data.base_heading_mask[env_ids],
        ).unsqueeze(-1) * self._mask[env_ids].any(dim=-1, keepdim=True)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)

        if self.cfg.current_state_command and self.cfg.in_base_frame:
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, 3] = 1.0
            elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
                pass
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :] = 0
                self._current_state[env_ids, :, 0] = 1
                self._current_state[env_ids, :, 5] = 1
        if self.cfg.current_state_command and not self.cfg.in_base_frame:
            robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids]
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, :] = robot_quat_w.unsqueeze(1)
            elif self.cfg.rotation_mode in "axis_angle":
                self._current_state[env_ids, :, :] = math_utils.axis_angle_from_quat(robot_quat_w).unsqueeze(1)
            elif self.cfg.rotation_mode == "euler":
                self._current_state[env_ids, :, :] = math_utils.euler_xyz_from_quat(robot_quat_w).unsqueeze(1)
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :] = instinct_math_utils.quat_to_tan_norm(robot_quat_w).unsqueeze(1)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_quat_reference_w"):
            return
        self._visualizer.visualize(
            translations=self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch,
            orientations=self._vis_quat_reference_w,
        )

    def _compute_debug_vis_data(self):
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch

        # Since this command is defined in robot's local frame, we need to transform
        # the command data to the world frame. Even if it is not refreshed timely.
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        if self.cfg.rotation_mode == "quaternion":
            quat_reference = keyframe_command[:]
        elif self.cfg.rotation_mode == "axis_angle":
            quat_reference = math_utils.quat_from_angle_axis(
                torch.norm(keyframe_command, dim=-1),
                keyframe_command,
            )
        elif self.cfg.rotation_mode == "euler":
            quat_reference = math_utils.quat_from_euler_xyz(
                keyframe_command[..., 0],
                keyframe_command[..., 1],
                keyframe_command[..., 2],
            )
        elif self.cfg.rotation_mode == "tannorm":
            quat_reference = instinct_math_utils.tan_norm_to_quat(keyframe_command)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        if self.cfg.in_base_frame:
            quat_reference = math_utils.quat_mul(
                root_quat_w,
                quat_reference,
            )
        self._vis_quat_reference_w = quat_reference


class ProjectedGravityRefCommand(ShadowingCommandBase):
    """Command that projects the gravity vector onto the base frame of the reference robot."""

    cfg: ProjectedGravityRefCommandCfg

    def __init__(self, cfg: ProjectedGravityRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        gravity_dir = torch.tensor(env.cfg.sim.gravity, device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        self.GRAVITY_VEC_W = gravity_dir.unsqueeze(0).expand(self.num_envs, -1)
        # generate the command tensor buffer
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 3),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (orientation mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 3),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (orientation mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_orientation_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """Update the command for the given environment indices."""
        # get the base frame orientation
        base_quat_w = self._motion_reference.data.base_quat_w[env_ids]
        # project the gravity vector onto the base frame
        self._command[env_ids] = math_utils.quat_apply_inverse(base_quat_w, self.GRAVITY_VEC_W[env_ids].unsqueeze(1))
        # apply the mask
        self._command[env_ids] *= self._motion_reference.data.base_orientation_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                self._env.scene[self.cfg.asset_cfg.name].data.projected_gravity_b.torch[env_ids].unsqueeze(1)
            )
        if hasattr(self, "_visualizer"):
            _quat_ref = self._motion_reference.data.base_quat_w[
                env_ids, self._motion_reference.aiming_frame_idx[env_ids]
            ]
            self.compute_rollpitch_ref(env_ids, _quat_ref)

    def compute_rollpitch_ref(self, env_ids, _quat_ref):
        _roll_ref, _pitch_ref = math_utils.euler_xyz_from_quat(_quat_ref)[:2]
        _roll_ref = math_utils.wrap_to_pi(_roll_ref)
        _pitch_ref = math_utils.wrap_to_pi(_pitch_ref)
        if not hasattr(self, "_roll_ref"):
            self._roll_ref = torch.zeros(self._command.shape[0], device=self.device)
        if not hasattr(self, "_pitch_ref"):
            self._pitch_ref = torch.zeros(self._command.shape[0], device=self.device)
        self._roll_ref[env_ids] = _roll_ref
        self._pitch_ref[env_ids] = _pitch_ref

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_roll_ref") or not hasattr(self, "_pitch_ref"):
            return
        marker_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch.clone()
        marker_pos_w[:, 2] += 1.2  # raise the position for visualization
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        marker_orientations = math_utils.quat_from_euler_xyz(
            self._roll_ref,  # roll
            self._pitch_ref,  # pitch
            robot_heading_w,  # yaw
        )
        scales = torch.ones_like(marker_pos_w[:, :3])  # scale for visualization
        scales[:, 0] = 0.5  # set the scale for the x-axis
        scales[:, 1] = 0.5  # set the scale for the y-axis
        scales[:, 2] = 0.1  # set the scale for the z-axis
        self._visualizer.visualize(
            translations=marker_pos_w,
            orientations=marker_orientations,
            scales=scales,
        )


class HeadingRefCommand(ShadowingCommandBase):
    """Command that only contains the heading of the reference, which is the yaw component of the reference rotation."""

    cfg: HeadingRefCommandCfg

    def __init__(self, cfg: HeadingRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 1),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 1),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (heading mask)
                device=self.device,
                dtype=torch.bool,
            )
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_heading_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_metrics(self):
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._command[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        heading_error = torch.abs(math_utils.wrap_to_pi(heading_command_w - robot_heading_w))
        self.metrics["error_heading"] = heading_error

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """Update the command for the given environment indices."""
        # get the base frame orientation
        base_quat_w = self._motion_reference.data.base_quat_w[env_ids]
        # extract the heading (yaw) component from the quaternion
        self._command[env_ids] = math_utils.wrap_to_pi(
            math_utils.euler_xyz_from_quat(base_quat_w.reshape(-1, 4))[2]
        ).reshape(-1, self._motion_reference.num_frames, 1)
        # apply the mask
        self._command[env_ids] *= self._motion_reference.data.base_heading_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                math_utils.wrap_to_pi(
                    math_utils.euler_xyz_from_quat(
                        self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids]
                    )[2]
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_heading_ref"):
            return
        base_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch.clone()
        base_pos_w[:, 2] += 0.8
        scales = torch.ones_like(base_pos_w)  # scale for visualization
        scales[:, 0] = 4.0  # set the scale for the heading arrow
        scales[:, 1] = 1.0  # set the scale for the heading arrow
        scales[:, 2] = 0.1  # set the scale for the heading arrow
        # visualize the heading reference
        self._visualizer.visualize(translations=base_pos_w, orientations=self._vis_heading_ref, scales=scales)

    def _compute_debug_vis_data(self):
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._command[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        # get the quaternion to represent the heading
        heading_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(heading_command_w),  # pitch
            torch.zeros_like(heading_command_w),  # roll
            heading_command_w,  # yaw
        )
        heading_quat = math_utils.normalize(heading_quat)  # normalize the quaternion
        self._vis_heading_ref = heading_quat


class HeadingErrorRefCommand(ShadowingCommandBase):
    """Command that only contains the heading error of the reference, which is the yaw component of the reference rotation."""

    cfg: HeadingErrorRefCommandCfg

    def __init__(self, cfg: HeadingErrorRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 1),
            device=self.device,
        )
        self._target_heading = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 1),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 1),
                device=self.device,
            )
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self._update_command()

    def _update_metrics(self):
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._target_heading[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        heading_error = torch.abs(math_utils.wrap_to_pi(heading_command_w - robot_heading_w))
        self.metrics["error_heading"] = heading_error

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """Update the command for the given environment indices."""
        if len(env_ids) == 0:
            return
        # get the base frame orientation
        base_quat_w = self._motion_reference.data.base_quat_w[env_ids]
        # extract the heading (yaw) component from the quaternion
        self._target_heading[env_ids] = math_utils.wrap_to_pi(
            math_utils.euler_xyz_from_quat(base_quat_w.reshape(-1, 4))[2]
        ).reshape(-1, self._motion_reference.num_frames, 1)
        # get the robot's current heading
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids]
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        # compute the heading error
        self._command[env_ids] = math_utils.wrap_to_pi(
            self._target_heading[env_ids] - robot_heading_w.unsqueeze(1).unsqueeze(1)
        )
        # apply the mask
        self._command[env_ids] *= self._motion_reference.data.base_heading_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_heading_ref"):
            return
        base_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch.clone()
        base_pos_w[:, 2] += 0.8
        scales = torch.ones_like(base_pos_w)  # scale for visualization
        scales[:, 0] = 4.0  # set the scale for the heading arrow
        scales[:, 1] = 1.0  # set the scale for the heading arrow
        scales[:, 2] = 0.1  # set the scale for the heading arrow
        # visualize the heading reference
        self._visualizer.visualize(translations=base_pos_w, orientations=self._vis_heading_ref, scales=scales)

    def _compute_debug_vis_data(self):
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._target_heading[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        # get the quaternion to represent the heading
        heading_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(heading_command_w),  # pitch
            torch.zeros_like(heading_command_w),  # roll
            heading_command_w,  # yaw
        )
        heading_quat = math_utils.normalize(heading_quat)  # normalize the quaternion
        self._vis_heading_ref = heading_quat


class BaseHeightRefCommand(ShadowingCommandBase):
    """Command that only contains the height of the base.
    NOTE: Current implementation only refer to the motion reference directly. May lead to error if the terrain is not zero-flat-ground.
    """

    cfg: BaseHeightRefCommandCfg

    def __init__(self, cfg: BaseHeightRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        data_dims = (1,)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (height mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (height mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_pos_height_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :] = self._motion_reference.data.base_pos_w[env_ids][:, :, 2:3]
        self._command[env_ids] *= (
            self._motion_reference.data.base_pos_height_mask[env_ids].unsqueeze(-1)
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1)
            * self._mask[env_ids]
        )
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids, 2:3].unsqueeze(1)
            )

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_marker_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_marker_pos_w,
        )

    def _compute_debug_vis_data(self):
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        marker_translation = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch.clone()
        marker_translation[:, 2] = self._command[ALL_INDICES, aiming_frame_idx, 0]
        self._vis_marker_pos_w = marker_translation


class BaseLinVelRefCommand(ShadowingCommandBase):
    """Command that only contains the linear velocity of the base."""

    cfg: BaseLinVelRefCommandCfg

    def __init__(self, cfg: BaseLinVelRefCommandCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        super().__init__(cfg, env)
        # generate the command tensor buffer
        data_dims = (3,)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (velocity mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (velocity mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = self._mask  # No mask from motion reference data currently.
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids] = math_utils.quat_apply_inverse(
            self._motion_reference.data.base_quat_w[env_ids], self._motion_reference.data.base_lin_vel_w[env_ids]
        )
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                self._env.scene[self.cfg.asset_cfg.name].data.root_lin_vel_b.torch[env_ids].unsqueeze(1)
            )


class JointPosRefCommand(ShadowingCommandBase):
    """ """

    cfg: JointPosRefCommandCfg

    def __init__(self, cfg: JointPosRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        reference_joint_pos = self._motion_reference.data.joint_pos[..., self._motion_reference_joint_ids]
        data_dims = (reference_joint_pos.shape[2],)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, reference_joint_pos.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # get and copy the default joint position
        # shape (num_envs, num_joints)
        self._default_joint_pos = (
            self._env.scene[cfg.asset_cfg.name].data.default_joint_pos.torch[:, self._asset_joint_ids].clone()
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, reference_joint_pos.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_joints)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(
            mask,
            self._motion_reference.data.joint_pos_mask[..., self._motion_reference_joint_ids],
        )
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :] = self._motion_reference.data.joint_pos[env_ids][
            :, :, self._motion_reference_joint_ids
        ] - self._default_joint_pos[env_ids].unsqueeze(1)
        self._command[env_ids, :] *= self._motion_reference.data.joint_pos_mask[env_ids][
            :, :, self._motion_reference_joint_ids
        ]
        self._command[env_ids, :] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids, :] *= self._mask[env_ids]
        if self.cfg.current_state_command:
            self._current_state[env_ids] = self._env.scene[self.cfg.asset_cfg.name].data.joint_pos.torch[env_ids][
                :, self._asset_joint_ids
            ].unsqueeze(1) - self._default_joint_pos[env_ids].unsqueeze(1)


class JointPosErrRefCommand(ShadowingCommandBase):
    """ """

    cfg: JointPosErrRefCommandCfg

    def __init__(self, cfg: JointPosErrRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        reference_joint_pos = self._motion_reference.data.joint_pos[..., self._motion_reference_joint_ids]
        data_dims = (reference_joint_pos.shape[2],)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, reference_joint_pos.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, reference_joint_pos.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(
            mask,
            self._motion_reference.data.joint_pos_mask[..., self._motion_reference_joint_ids],
        )
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :] = self._motion_reference.data.joint_pos[env_ids][
            :, :, self._motion_reference_joint_ids
        ] - self._env.scene[self.cfg.asset_cfg.name].data.joint_pos.torch[env_ids][:, self._asset_joint_ids].unsqueeze(
            1
        )
        self._command[env_ids] *= (
            self._motion_reference.data.joint_pos_mask[env_ids][:, :, self._motion_reference_joint_ids]
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1)
            * self._mask[env_ids]
        )


class JointVelRefCommand(ShadowingCommandBase):
    """ """

    cfg: JointVelRefCommandCfg

    def __init__(self, cfg: JointVelRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        reference_joint_vel = self._motion_reference.data.joint_vel[..., self._motion_reference_joint_ids]
        data_dims = (reference_joint_vel.shape[2],)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, reference_joint_vel.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # get and copy the default joint velocity
        # shape (num_envs, num_joints)
        self._default_joint_vel = (
            self._env.scene[cfg.asset_cfg.name].data.default_joint_vel.torch[:, self._asset_joint_ids].clone()
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, reference_joint_vel.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_joints)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(
            mask,
            self._motion_reference.data.joint_vel_mask[..., self._motion_reference_joint_ids],
        )
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :] = self._motion_reference.data.joint_vel[env_ids][
            :, :, self._motion_reference_joint_ids
        ] - self._default_joint_vel[env_ids].unsqueeze(1)
        self._command[env_ids, :] *= (
            self._motion_reference.data.joint_vel_mask[env_ids][:, :, self._motion_reference_joint_ids]
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1)
            * self._mask[env_ids]
        )
        if self.cfg.current_state_command:
            self._current_state[env_ids] = self._env.scene[self.cfg.asset_cfg.name].data.joint_vel.torch[env_ids][
                :, self._asset_joint_ids
            ].unsqueeze(1) - self._default_joint_vel[env_ids].unsqueeze(1)


class LinkRefCommand(ShadowingCommandBase):
    """ """

    cfg: LinkRefCommandCfg

    def __init__(self, cfg: LinkRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (3 + 4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3 + 3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (3 + 6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (
                self.num_envs,
                self._motion_reference.num_frames,
                self._motion_reference.num_link_to_ref,
                2,  # (position mask, rotation mask)
            ),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_links, 2) # (position mask, rotation mask)
        self._body_ids = self._env.scene[cfg.asset_cfg.name].find_bodies(
            self._motion_reference.cfg.link_of_interests, preserve_order=True
        )[0]
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, 2),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_links, 2) # (position mask, rotation mask)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.link_pos_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.link_rot_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :, :, :3] = (
            self._motion_reference.data.link_pos_b[env_ids]
            * self._motion_reference.data.link_pos_mask[env_ids].unsqueeze(-1)
            * self._mask[env_ids, :, :, 0:1]
        )
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids, :, :, 3:7] = (
                self._motion_reference.data.link_quat_b[env_ids]
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids, :, :, 3:6] = (
                math_utils.axis_angle_from_quat(
                    self._motion_reference.data.link_quat_b[env_ids],
                )
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        elif self.cfg.rotation_mode == "euler":
            euler_ = torch.stack(
                math_utils.euler_xyz_from_quat(
                    self._motion_reference.data.link_quat_b[env_ids].reshape(-1, 4),
                ),
                dim=-1,
            )
            self._command[env_ids, :, :, 3:6] = (
                euler_.reshape(-1, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, 3)
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids, :, :, 3:9] = (
                instinct_math_utils.quat_to_tan_norm(self._motion_reference.data.link_quat_b[env_ids])
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)

        if self.cfg.current_state_command:
            root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch
            root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch
            root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(
                root_pos_w[env_ids], root_quat_w[env_ids]
            )
            self._current_state[env_ids, :, :, :3] = math_utils.transform_points(
                self._env.scene[self.cfg.asset_cfg.name].data.body_link_pos_w.torch[env_ids][:, self._body_ids],
                root_pos_w_inv,
                root_quat_w_inv,
            ).unsqueeze(1)
            current_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.body_link_quat_w.torch[env_ids][
                :, self._body_ids
            ]
            current_quat_b = math_utils.quat_mul(
                root_quat_w_inv.unsqueeze(1).expand(-1, self._motion_reference.num_link_to_ref, -1),
                current_quat_w,
            )

            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, :, 3:7] = current_quat_b.unsqueeze(1)
            elif self.cfg.rotation_mode == "axis_angle":
                self._current_state[env_ids, :, :, 3:6] = math_utils.axis_angle_from_quat(current_quat_b).unsqueeze(1)
            elif self.cfg.rotation_mode == "euler":
                self._current_state[env_ids, :, :, 3:6] = torch.stack(
                    math_utils.euler_xyz_from_quat(current_quat_b),
                    dim=1,
                )
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :, 3:9] = instinct_math_utils.quat_to_tan_norm(
                    current_quat_b
                ).unsqueeze(1)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_link_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_link_pos_w,
            orientations=self._vis_link_quat_w,
        )

    def _compute_debug_vis_data(self):
        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch

        # Since this command is defined in robot's local frame, we need to transform
        # the command data to the world frame. Even if it is not refreshed timely.
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        num_links = keyframe_command.shape[1]
        link_pos_w = math_utils.transform_points(
            keyframe_command[:, :, :3],
            root_pos_w,
            root_quat_w,
        )
        if self.cfg.rotation_mode == "quaternion":
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                keyframe_command[:, :, 3:7],
            )
        elif self.cfg.rotation_mode == "axis_angle":
            link_rot_b = keyframe_command[:, :, 3:6].reshape(-1, 3)
            link_quat_b = math_utils.quat_from_angle_axis(torch.norm(link_rot_b, dim=-1), link_rot_b)
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                link_quat_b.reshape(-1, num_links, 4),
            )
        elif self.cfg.rotation_mode == "euler":
            link_rot_b = keyframe_command[:, :, 3:6].reshape(-1, 3)
            link_quat_b = math_utils.quat_from_euler_xyz(
                link_rot_b[..., 0],  # pitch
                link_rot_b[..., 1],  # roll
                link_rot_b[..., 2],  # yaw
            )
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                link_quat_b.reshape(-1, num_links, 4),
            )
        elif self.cfg.rotation_mode == "tannorm":
            link_rot_b = keyframe_command[:, :, 3:9].reshape(-1, 6)
            link_quat_b = instinct_math_utils.tan_norm_to_quat(link_rot_b)
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                link_quat_b.reshape(-1, num_links, 4),
            )
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")

        self._vis_link_pos_w = link_pos_w.reshape(-1, 3)
        self._vis_link_quat_w = link_quat_w.reshape(-1, 4)


class LinkPosRefCommand(ShadowingCommandBase):
    """ """

    cfg: LinkPosRefCommandCfg

    def __init__(self, cfg: LinkPosRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        data_dims = (3,)
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_links) # (position mask)
        self._body_ids = self._env.scene[cfg.asset_cfg.name].find_bodies(
            self._motion_reference.cfg.link_of_interests, preserve_order=True
        )[0]
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_links, 1) # (position mask)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.link_pos_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :, :, :] = (
            self._motion_reference.data.link_pos_b[env_ids]
            * self._motion_reference.data.link_pos_mask[env_ids].unsqueeze(-1)
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
            * self._mask[env_ids].unsqueeze(-1)
        )

        if self.cfg.current_state_command:
            root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids]
            root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids]
            root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
            self._current_state[env_ids] = math_utils.transform_points(
                self._env.scene[self.cfg.asset_cfg.name].data.body_pos_w.torch[env_ids][:, self._body_ids],
                root_pos_w_inv,
                root_quat_w_inv,
            ).unsqueeze(1)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_link_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_link_pos_w,
        )

    def _compute_debug_vis_data(self):
        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch

        # Since this command is defined in robot's local frame, we need to transform
        # the command data to the world frame. Even if it is not refreshed timely.
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        link_pos_w = math_utils.transform_points(
            keyframe_command[:, :, :3],
            root_pos_w,
            root_quat_w,
        )
        self._vis_link_pos_w = link_pos_w.reshape(-1, 3)


class LinkRotRefCommand(ShadowingCommandBase):
    """ """

    cfg: LinkRotRefCommandCfg

    def __init__(self, cfg: LinkRotRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_links, 1) # (rotation mask)
        self._body_ids = self._env.scene[cfg.asset_cfg.name].find_bodies(
            self._motion_reference.cfg.link_of_interests, preserve_order=True
        )[0]
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_links, 1) # (rotation mask)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.link_rot_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        link_quat_ref_b = self._motion_reference.data.link_quat_b[env_ids]
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids] = link_quat_ref_b
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids] = math_utils.axis_angle_from_quat(link_quat_ref_b)
        elif self.cfg.rotation_mode == "euler":
            link_quat_ref_b_ = link_quat_ref_b.reshape(-1, 4)
            self._command[env_ids] = torch.stack(
                math_utils.euler_xyz_from_quat(link_quat_ref_b_),
                dim=-1,
            ).reshape(-1, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, 3)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids] = instinct_math_utils.quat_to_tan_norm(link_quat_ref_b)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._command[env_ids] *= self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids].unsqueeze(-1)

        if self.cfg.current_state_command:
            current_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.body_quat_w.torch[env_ids]
            current_quat_w = current_quat_w[:, self._body_ids]
            current_quat_b = math_utils.quat_mul(
                self._env.scene[self.cfg.asset_cfg.name]
                .data.root_quat_w.torch[env_ids]
                .unsqueeze(1)
                .expand(-1, self._motion_reference.num_link_to_ref, -1),
                current_quat_w,
            )
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, 0] = current_quat_b
            elif self.cfg.rotation_mode == "axis_angle":
                self._current_state[env_ids, 0] = math_utils.axis_angle_from_quat(current_quat_b)
            elif self.cfg.rotation_mode == "euler":
                self._current_state[env_ids, 0] = math_utils.euler_xyz_from_quat(current_quat_b)
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, 0] = instinct_math_utils.quat_to_tan_norm(current_quat_b)


class LinkPosErrRefCommand(LinkPosRefCommand):
    """ """

    cfg: LinkPosErrRefCommandCfg

    def __init__(self, cfg: LinkPosErrRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        link_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.body_pos_w.torch[:, self._body_ids]
        root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(
            self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids],
            self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids],
        )
        link_pos_b = math_utils.transform_points(
            link_pos_w[env_ids],
            root_pos_w_inv,
            root_quat_w_inv,
        )
        self._command[env_ids] = self._motion_reference.data.link_pos_b[env_ids] - link_pos_b.unsqueeze(1)
        self._command[env_ids] *= self._motion_reference.data.link_pos_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids].unsqueeze(-1)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_err_pos"):
            return
        # visualize the error direction
        self._visualizer.visualize(
            translations=self._vis_err_pos,
            orientations=self._vis_err_quat,
            scales=self._vis_err_scale,
        )

    def _compute_debug_vis_data(self):
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch
        link_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.body_pos_w.torch[:, self._body_ids]

        # get the error direction in world frame
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]  # (n_envs, n_links, 3)
        num_links = keyframe_command.shape[1]
        link_pos_err_w = math_utils.quat_apply(
            root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
            keyframe_command,
        ).reshape(-1, 3)
        err_scale, err_quat, err_pos = self._resolve_direction_to_arrow(link_pos_err_w, link_pos_w.reshape(-1, 3))

        self._vis_err_pos = err_pos
        self._vis_err_quat = err_quat
        self._vis_err_scale = err_scale

    def _resolve_direction_to_arrow(
        self,
        direction: torch.Tensor,
        start_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the arrow direction and scale.
        Args:
            direction: The direction of the arrow in shape (N, 3).
            start_point: The starting point of the arrow in shape (N, 3)
        """
        # obtain default scale of the marker
        default_scale = self._visualizer.cfg.markers["arrow"].scale
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(direction.shape[0], 1)
        arrow_scale[:, 0] *= torch.norm(direction, dim=-1) * 10
        # arrow-direction
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
        arrow_quat = math_utils.quat_from_angle_axis(
            angle,
            axis,
        )
        # arrow-position
        # NOTE: Due to the usd file of the arrow, the origin of the arrow file is neither at the start nor the middle.
        # It is at the 0.25 of the arrow length from the start to the end!!!!
        arrow_pos = start_point + 0.25 * direction

        return arrow_scale, arrow_quat, arrow_pos


class LinkRotErrRefCommand(LinkRotRefCommand):
    """ """

    cfg: LinkRotErrRefCommandCfg

    def __init__(self, cfg: LinkRotErrRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        link_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.body_quat_w.torch[env_ids]
        link_quat_w = link_quat_w[:, self._body_ids]  # (n_envs, n_links, 4)
        if self.cfg.in_base_frame:
            # transform the link quaternion to the base frame
            root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids]
            link_quat_ref_ = self._motion_reference.data.link_quat_b[env_ids]
            link_quat_ = math_utils.quat_mul(
                math_utils.quat_inv(root_quat_w.unsqueeze(1).expand(-1, self._motion_reference.num_link_to_ref, -1)),
                link_quat_w,
            )
        else:
            # transform the link quaternion to the world frame
            link_quat_ref_ = self._motion_reference.data.link_quat_w[env_ids]
            link_quat_ = link_quat_w
        # compute the error quaternion, aka. reference quaternion under the frame of current link quaternion
        link_quat_err = math_utils.quat_mul(
            math_utils.quat_inv(link_quat_).unsqueeze(1).expand(-1, self._motion_reference.num_frames, -1, -1),
            link_quat_ref_,
        )
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids] = link_quat_err
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids] = math_utils.axis_angle_from_quat(link_quat_err)
        elif self.cfg.rotation_mode == "euler":
            link_quat_err_ = link_quat_err.reshape(-1, 4)
            self._command[env_ids] = torch.stack(
                math_utils.euler_xyz_from_quat(link_quat_err_),
                dim=-1,
            ).reshape(-1, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, 3)
            # (n_envs, n_frames, n_links, 3)
            # euler angles in xyz order
            # (pitch, roll, yaw)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids] = instinct_math_utils.quat_to_tan_norm(link_quat_err)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._command[env_ids] *= self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids].unsqueeze(-1)

        if self.cfg.current_state_command:
            # all zero error
            self._current_state[env_ids] = 0.0
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, :, -1] = 1
            elif self.cfg.rotation_mode in ("axis_angle", "euler"):
                pass  # already set to zero
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :, 0] = 1
                self._current_state[env_ids, :, :, 5] = 1
            else:
                raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")


class TimeToTargetCommand(ShadowingCommandBase):
    """ """

    cfg: TimeToTargetCommandCfg

    def __init__(self, cfg: TimeToTargetCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        ## NOTE
            There is no mask for this command. The invalid value should be set to -1.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        data_dims = (1,)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
        self._update_command()

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :, 0] = self._motion_reference.data.time_to_target_frame[
            env_ids
        ] - self._motion_reference.time_passed_from_update[env_ids].unsqueeze(-1)
