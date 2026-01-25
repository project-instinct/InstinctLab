from __future__ import annotations

import os
from collections.abc import Sequence
from copy import copy
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.physics.tensors.impl.api as physx

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import SensorBase
from isaaclab.utils.buffers import TimestampedBuffer

from instinctlab.utils.prims import get_articulation_view

from .motion_reference_data import MotionReferenceData, MotionReferenceState

if TYPE_CHECKING:
    from .motion_reference_cfg import MotionReferenceManagerCfg
    from .motion_buffer import MotionBuffer

import numpy as np
import torch
import torch.distributed as dist

import pytorch_kinematics as pk


class MotionReferenceManager(SensorBase):
    """The manager to handle all motion references. There are multiple types of motion references
    should be supported:
        1. PlaneDataset: as from pre-collected dataset. The robot's base pose trajectory is in the
            world frame.
        2. InteractiveDataset: as from the pre-collected dataset. The robot's base pose trajectory is in
            the world frame. Also, the dataset is labeled with objects to interact with and can tell
            us how to generate the interaction objects at simulation initialization.
        3. TerrainDataset: as from the pre-collected dataset. The robot's base pose trajectory is in
            the world frame. Also, the dataset is labeled with terrain information and can tell us
            how to generate the terrain at simulation initialization.
        4. GenerativeMotion: for example locomotion command can be directly generated from the robot's
            current state. But this type of motion reference only contains the robot's base pose
            trajectory. And this type of motion reference does not need motion_reference_buffer.
    """

    cfg: MotionReferenceManagerCfg

    def __init__(
        self,
        cfg: MotionReferenceManagerCfg,
    ):
        """Initialize the motion reference manager."""
        super().__init__(cfg)
        self.cfg: MotionReferenceManagerCfg

    def __str__(self):
        """Get the tabular information of the motion reference managed buffer."""
        msg = f"<MotionReferenceManager> contains {len(self._motion_buffers)} motion buffers to use.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Motion Buffers"
        table.field_names = ["Name", "Num Trajectories"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Num Trajectories"] = "r"
        # add info on each term
        for name, buffer in self._motion_buffers.items():
            table.add_row([name, buffer.num_trajectories])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties
    """

    @property
    def data(self) -> MotionReferenceData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def reference_frame(self) -> MotionReferenceData:
        """The reference state of all the articulations in the scene at current time step.
        (Acquired passively)
        """
        outdated_mask = torch.logical_or(
            (self._reference_frame_timestamp - self._timestamp).abs() > 1e-6, self._reference_frame_timestamp < 1e-6
        )
        if outdated_mask.any():
            env_ids_to_update = self._ALL_INDICES[outdated_mask]
            for name, buffer in self._motion_buffers.items():
                env_ids_assignment = self._motion_buffer_assignment[name]
                assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
                env_ids_this_buffer_mask = torch.logical_and(
                    env_ids_to_update >= env_ids_assignment.start, env_ids_to_update < env_ids_assignment.stop
                )
                env_ids_this_buffer = env_ids_to_update[env_ids_this_buffer_mask]
                buffer.fill_motion_data(
                    env_ids_this_buffer,
                    self._timestamp[env_ids_this_buffer].unsqueeze(-1),  # (N, 1)
                    self.env_origins,
                    self._reference_frame,
                )
                # symmetric augment init reference state if needed
                if self._symmetric_augmentation_conditions_met:
                    env_ids_to_augment = env_ids_this_buffer[self._env_symmetric_augmentation_mask[env_ids_this_buffer]]
                    self._symmetric_augment_reference_data(self._reference_frame, env_ids_to_augment)
            self._reference_frame_timestamp[outdated_mask] = self._timestamp[outdated_mask]
        return self._reference_frame

    @property
    def reference_relative_base_pos(self) -> torch.Tensor:
        """To compute the reference link pos/orientation in the world frame (but relative to current robot's position)
        we compute a reference base position matching robot's x-y position but no z position.
        Shape: (num_envs, 3)
        """
        if ((self._reference_relative_base_pos.timestamp - self._timestamp).abs() > 1e-6).any():
            self._reference_relative_base_pos.data = self._view.get_root_transforms().clone()[:, :3]
            self._reference_relative_base_pos.data[:, 2] = self.reference_frame.base_pos_w[:, 0, 2]
            self._reference_relative_base_pos.timestamp = self._timestamp.clone()
        return self._reference_relative_base_pos.data

    @property
    def reference_relative_delta_quat(self) -> torch.Tensor:
        """To rotate the link's pos_w/quat_w and match the robot's rotation, in order to compare with
        the robot's actual link pos_w/quat_w, we need to correct its yaw.
        Any world-frame rotation in the motion reference left-multiply by this quat will get the robot's relative frame position.
        Shape: (num_envs, 4)
        """
        if ((self._reference_relative_delta_quat.timestamp - self._timestamp).abs() > 1e-6).any():
            _view_quat = math_utils.convert_quat(self._view.get_root_transforms().clone()[:, 3:7], to="wxyz")
            self._reference_relative_delta_quat.data = math_utils.yaw_quat(
                math_utils.quat_mul(
                    _view_quat,
                    math_utils.quat_inv(self.reference_frame.base_quat_w[:, 0]),
                )
            )
            self._reference_relative_delta_quat.timestamp = self._timestamp.clone()
        return self._reference_relative_delta_quat.data

    @property
    def reference_link_pos_relative_w(self) -> torch.Tensor:
        """Assuming the motion's base frame is at the robot's x-y position, heading;
        joint_pos / roll-pitch remains the same, the links' position in the world frame.
        Shape: (num_envs, num_links, 3)
        """
        if ((self._reference_link_pos_relative_w.timestamp - self._timestamp).abs() > 1e-6).any():
            delta_quat = self.reference_relative_delta_quat.unsqueeze(1).expand(
                -1,
                self.num_link_of_interests,
                -1,
            )
            base_pos = self.reference_relative_base_pos.unsqueeze(1).expand(
                -1,
                self.num_link_of_interests,
                -1,
            )
            self._reference_link_pos_relative_w.data = base_pos + math_utils.quat_apply(
                delta_quat,
                self.reference_frame.link_pos_w[:, 0]
                - self.reference_frame.base_pos_w.expand(
                    -1,
                    self.num_link_of_interests,
                    -1,
                ),
            )
            self._reference_link_pos_relative_w.timestamp = self._timestamp.clone()
        return self._reference_link_pos_relative_w.data

    @property
    def reference_link_quat_relative_w(self) -> torch.Tensor:
        """Assuming the motion's base frame is at the robot's x-y position, heading;
        joint_pos / roll-pitch remains the same, the links' quaternion in the world frame.
        (w, x, y, z) format.
        Shape: (num_envs, num_links, 4)
        """
        if ((self._reference_link_quat_relative_w.timestamp - self._timestamp).abs() > 1e-6).any():
            self._reference_link_quat_relative_w.data = math_utils.quat_mul(
                self.reference_relative_delta_quat.unsqueeze(1).expand(
                    -1,
                    self.num_link_of_interests,
                    -1,
                ),
                self.reference_frame.link_quat_w[:, 0],
            )
            self._reference_link_quat_relative_w.timestamp = self._timestamp.clone()
        return self._reference_link_quat_relative_w.data

    @property
    def num_frames(self) -> int:
        return self.cfg.num_frames

    @property
    def num_link_of_interests(self) -> int:
        return len(self.cfg.link_of_interests)

    @property
    def num_joints(self):
        return self._view.max_dofs

    @property
    def joint_names(self) -> list[str]:
        """Get the joint names of the articulation."""
        return self.isaac_joint_names

    @property
    def num_bodies(self):
        return len(self.cfg.link_of_interests)

    @property
    def body_names(self) -> list[str]:
        """Get the link names of the articulation."""
        return self.cfg.link_of_interests

    @property
    def num_link_to_ref(self) -> int:
        if self.cfg.robot_model_path is None or self.cfg.link_of_interests is None:
            return 0
        else:
            return len(self.cfg.link_of_interests)

    @property
    def frame_interval_s(self) -> torch.Tensor:
        """Shape: (num_envs,)"""
        return self._frame_interval_s

    @property
    def ALL_INDICES(self) -> torch.Tensor:
        return self._ALL_INDICES

    @property
    def time_passed_from_update(self) -> torch.Tensor:
        return self._timestamp - self._timestamp_last_update

    @property
    def is_at_keyframe(self) -> torch.Tensor:
        """Whether the passed time is at the multiple of frame_interval_s
        NOTE: currently assuming all frame interval are the same across frames in a given env.
        """
        return torch.where(
            self._timestamp > self._timestamp_last_update,
            (torch.clip(self.time_passed_from_update, min=0.0) % self.frame_interval_s) < 1e-6,
            torch.zeros_like(self._timestamp, dtype=torch.bool),
        )

    @property
    def aiming_frame_idx(self) -> torch.Tensor:
        # The frame index which currently the robot should be aiming at, w.r.t reference data.
        # e.g. The 0-th frame is the frame at `_timestamp_last_update + frame_interval_s`
        # When `_timestamp - _timestamp_last_update` == frame_interval_s,
        # the aiming frame should still be 0.
        if ((self._aiming_frame_idx.timestamp - self._timestamp).abs() > 1e-6).any():
            time_passed_from_update = self.time_passed_from_update  # (N,)
            time_to_target_frame = self.data.time_to_target_frame  # (N, num_frames)
            aiming_frame_idx = torch.sum(
                torch.logical_and(
                    time_passed_from_update.unsqueeze(-1) > time_to_target_frame,
                    time_to_target_frame > 0.0,
                ),
                dim=-1,
            )  # (N,)
            aiming_frame_idx[aiming_frame_idx >= self.num_frames] = -1
            self._aiming_frame_idx.data = aiming_frame_idx
            self._aiming_frame_idx.timestamp = self._timestamp.clone()
        return self._aiming_frame_idx.data

    @property
    def time_to_aiming_frame(self) -> torch.Tensor:
        # The time left to reach the aiming frame
        return (self.aiming_frame_idx + 1) * self.frame_interval_s - (self.time_passed_from_update)

    @property
    def env_origins(self) -> torch.Tensor:
        return (
            self._env_origins if hasattr(self, "_env_origins") else torch.zeros(self._view.count, 3, device=self.device)
        )

    @property
    def motion_buffers(self) -> dict[str, MotionBuffer]:
        """Get the active motion buffers."""
        return self._motion_buffers

    @property
    def _motion_buffer_num_trajectories(self) -> dict[str, int]:
        """Get the number of trajectories for each motion buffer as (a private, online, class property)."""
        return {name: buffer.num_trajectories for name, buffer in self._motion_buffers.items()}

    @property
    def complete_motion_lengths(self) -> torch.Tensor:
        """The motion reference length for ALL_INDICES in seconds.
        ## NOTE: This property returns the total trajectory length for all envs, EVEN if the env is started from the middle
        of the motion file. The lengths are computed based on the motion buffer's complete_motion_lengths.
        """
        lengths = torch.ones(self._view.count, device=self.device)
        for buffer_name, motion_buffer in self.motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[buffer_name]
            env_ids_this_buffer = self.ALL_INDICES[env_ids_assignment]
            lengths[env_ids_this_buffer] = motion_buffer.complete_motion_lengths

        return lengths

    @property
    def assigned_motion_lengths(self) -> torch.Tensor:
        """The motion reference length for ALL_INDICES in seconds, but only for the assigned env_ids.
        ## NOTE: This property returns the trajectory length for all envs. If the env is started from the middle of a motion
        file, the length is computed as the starting point to the end of the motion file.
        """
        lengths = torch.ones(self._view.count, device=self.device)
        for buffer_name, motion_buffer in self.motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[buffer_name]
            env_ids_this_buffer = self.ALL_INDICES[env_ids_assignment]
            lengths[env_ids_this_buffer] = motion_buffer.assigned_motion_lengths

        return lengths

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset the motion reference manager as a sensor, also reset the motion reference components."""
        assert getattr(
            self, "__motion_reference_manager_initialized", False
        ), "Motion reference manager is not initialized successfully. Please check the error message above."
        super().reset(env_ids)
        if env_ids is None:
            env_ids = self.ALL_INDICES

        # Reset motion buffer selection and resample selection
        self._reset_motion_buffers(env_ids)
        self._resample_buffer_collate_params(env_ids)
        self._resample_update_period(env_ids)

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Return the joint_ids of the given joint names. To meet the same interface with the isaaclab SceneEntity.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self.isaac_joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Return the link_ids of the given link names. To meet the same interface with the isaaclab SceneEntity.
        NOTE: This ids are selections across self.cfg.link_of_interests. Not the entire link ids in the articulation.
        """
        # find bodies
        return string_utils.resolve_matching_names(name_keys, self.cfg.link_of_interests, preserve_order)

    def get_init_reference_state(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> MotionReferenceState:
        """Get the initial reference state for the given env_ids."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            env_ids_this_buffer = (
                self.ALL_INDICES[env_ids_assignment] if env_ids is None else env_ids[env_ids_this_buffer_mask]
            )
            buffer.fill_init_reference_state(
                env_ids_this_buffer,
                self.env_origins,
                self._init_reference_state,
            )
            # symmetric augment init reference state if needed
            if self._symmetric_augmentation_conditions_met:
                env_ids_to_augment = env_ids_this_buffer[self._env_symmetric_augmentation_mask[env_ids_this_buffer]]
                self._init_reference_state.joint_pos[env_ids_to_augment] = self._symmetric_augment_joint_buffer(
                    self._init_reference_state.joint_pos[env_ids_to_augment],
                )
                self._init_reference_state.joint_vel[env_ids_to_augment] = self._symmetric_augment_joint_buffer(
                    self._init_reference_state.joint_vel[env_ids_to_augment],
                )
                self._init_reference_state.base_pos_w[env_ids_to_augment, 1] *= -1
                self._init_reference_state.base_quat_w[env_ids_to_augment] = self._symmetric_augment_quat_buffer(
                    self._init_reference_state.base_quat_w[env_ids_to_augment],
                )
                self._init_reference_state.base_lin_vel_w[env_ids_to_augment, 1] *= -1
                self._init_reference_state.base_ang_vel_w[
                    env_ids_to_augment, 1
                ] *= -1  # NOTE: validity provided by chatgpt
        return self._init_reference_state[env_ids]

    def target_link_pose_forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Considering the interested link of the motion reference is known from config, the forward kinematics output
        can be fixed to a series of link poses. This function is used to compute the target link poses from the joint
        positions (in isaacSim order).
        NOTE: This method serve as a callable object for the motion buffer to compute the target link poses. It should
        be optimized for faster computation later.
        ## Args:
            - joint_pos: (N, num_dofs) tensor, the joint positions in isaacSim order.
        ## Returns:
            The target link poses in base_link frame. (N, num_link_to_ref, 7) tensor, in the order or `self.cfg.link_of_interests`.
        """
        input_device = joint_pos.device
        joint_pos = joint_pos.to(self.device)
        all_link_poses = self._robot_kinematics_chain.forward_kinematics(joint_pos[:, self._joint_order_isaac_to_pk])
        link_pos_quat_b = torch.zeros(joint_pos.shape[0], self.num_link_to_ref, 7, device=self.device)
        for link_idx, link_name in enumerate(self.cfg.link_of_interests):
            pose_mat = all_link_poses[link_name].get_matrix().reshape(-1, 4, 4)
            link_pos_quat_b[:, link_idx, :3] = pose_mat[:, :3, 3]
            link_pos_quat_b[:, link_idx, 3:] = math_utils.quat_from_matrix(pose_mat[:, :3, :3])
        return link_pos_quat_b.to(input_device)

    def get_current_motion_identifiers(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> list[str]:
        """Get the current motion identifiers for the given env_ids."""
        if env_ids is None:
            env_ids = self.ALL_INDICES
        motion_identifiers = []
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            motion_identifiers.extend(buffer.get_current_motion_identifiers(env_ids[env_ids_this_buffer_mask]))  # type: ignore
        return motion_identifiers

    def get_current_motion_weights(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """Get the current motion weights for the given env_ids."""
        if env_ids is None:
            env_ids = self.ALL_INDICES
        motion_weights = torch.empty(len(env_ids), device=self.device, dtype=torch.float32)
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            motion_weights[env_ids_this_buffer_mask] = buffer.get_current_motion_weights(
                env_ids[env_ids_this_buffer_mask]
            )  # type: ignore
        return motion_weights

    def update_motion_weights(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        weight_ratio: float | torch.Tensor = 1.0,
    ):
        """Update the motion weights for the given env_ids, to update the motion sample weights when reset.
        ## NOTE:
            - multiple motion_buffers is not recommended.
            - multi processing is not recommended.
        """
        if isinstance(weight_ratio, torch.Tensor):
            assert len(weight_ratio) == len(
                env_ids
            ), "weight_ratio must be a scalar or a tensor with the same length as env_ids."
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            env_ids_this_buffer = env_ids[env_ids_this_buffer_mask]  # type: ignore
            buffer.update_motion_weights(
                env_ids_this_buffer,
                weight_ratio=(
                    weight_ratio[env_ids_this_buffer_mask] if isinstance(weight_ratio, torch.Tensor) else weight_ratio
                ),
            )

    """
    Operations to connect motion_reference with other components in the scene. Must be triggered through events
    ('reset' mode or 'startup' mode).
    """

    def match_scene(self, scene: InteractiveScene):
        """Match the motion reference with the scene, including objects and terrain. Considering some of the motion
        reference need to utilize un-flatten terrain, this information must be provided to the motion reference buffer
        and motion reference manager.
        """
        # store the env origins as the exact python object managed by the scene. (NOTE: pass-by-reference, not copy)
        self._env_origins = scene.env_origins
        for buffer in self.motion_buffers.values():
            # match the scene with the motion buffer.
            buffer.match_scene(scene)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        self._initialize_data()
        if self.cfg.robot_model_path is not None:
            # if robot model (urdf) if provided, initialize kinematics chain and joint order mapping
            self._initialize_robot_kinematics()
        self._initialize_motion_buffers()
        self._resample_buffer_collate_params()
        self._resample_update_period()
        print(self)  # print the tabular information of the motion reference managed buffer.
        # Due to isaaclab's _initialize_impl exception will not lead to complete program shutdown, we need to set a flag
        # to indicate the motion reference manager is initialized successfully.
        self.__motion_reference_manager_initialized = True

    def _update_buffers_impl(self, env_ids: Sequence[int] | torch.Tensor):
        """Update the motion reference buffers for the given env_ids."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        # compute the time left to reach the specific frame
        time_to_target_frame = torch.arange(self.cfg.num_frames, device=self.device, dtype=torch.float32)
        if self.cfg.data_start_from == "one_frame_interval":
            time_to_target_frame += 1
        time_to_target_frame = time_to_target_frame.unsqueeze(0).repeat(len(env_ids), 1)
        time_to_target_frame *= self.frame_interval_s[env_ids].unsqueeze(-1)

        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            env_ids_this_buffer = env_ids[env_ids_this_buffer_mask]
            buffer.fill_motion_data(
                env_ids_this_buffer,
                self._timestamp[env_ids_this_buffer].unsqueeze(-1) + time_to_target_frame[env_ids_this_buffer_mask],
                self.env_origins,
                self._data,
            )
            # symmetric augment init reference state if needed
            if self._symmetric_augmentation_conditions_met:
                env_ids_to_augment = env_ids_this_buffer[self._env_symmetric_augmentation_mask[env_ids_this_buffer]]
                self._symmetric_augment_reference_data(self._data, env_ids_to_augment)
        self._data.time_to_target_frame[env_ids] = time_to_target_frame

        # set invalid timestamp data to invalid value.
        self._data.time_to_target_frame[self._data.validity == 0] = -1.0

    """
    Manager's internal operations.
    """

    def _initialize_data(self):
        """Initialize _data for the 'sensor' output"""
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        self._view: physx.ArticulationView = get_articulation_view(
            self.cfg.prim_path,
            self._physics_sim_view,
        )
        self._ALL_INDICES = torch.arange(self._view.count, device=self.device)
        self.isaac_joint_names = self._view.shared_metatype.dof_names

        self._data = self.cfg.data_class_type.make_empty(
            self._view.count,
            self.cfg.num_frames,
            self._view.max_dofs,
            self.num_link_to_ref,
            device=self.device,
        )

        self._reference_frame = self.cfg.data_class_type.make_empty(
            self._view.count,
            1,  # num_frames
            self._view.max_dofs,
            self.num_link_to_ref,
            device=self.device,
        )
        # Add an additional timestamp to the reference frame data, which is used for lazy update of the reference frame.
        self._reference_frame_timestamp = torch.zeros(self._view.count, device=self.device)

        self._init_reference_state = MotionReferenceState.make_empty(
            self._view.count,
            self._view.max_dofs,
            device=self.device,
        )

        self._aiming_frame_idx = TimestampedBuffer()

        self._reference_relative_base_pos = TimestampedBuffer()
        self._reference_relative_delta_quat = TimestampedBuffer()
        self._reference_link_pos_relative_w = TimestampedBuffer()
        self._reference_link_quat_relative_w = TimestampedBuffer()

    def _initialize_motion_buffers(self):
        """Initialize all the motion reference buffers by computing specific data range of this process
        (in case of multi-workers).
        """
        self._motion_buffers: dict[str, MotionBuffer] = dict()
        assert len(self.cfg.motion_buffers) > 0, "At least one motion buffer should be specified."
        for motion_buffer_name, motion_buffer_cfg in self.cfg.motion_buffers.items():
            motion_buffer_cls: type[MotionBuffer] = motion_buffer_cfg.class_type
            self._motion_buffers[motion_buffer_name] = motion_buffer_cls(
                motion_buffer_cfg,
                articulation_view=self._view,
                link_of_interests=self.cfg.link_of_interests,
                forward_kinematics_func=self.target_link_pose_forward_kinematics,
                device=self.device,
            )
        # Step: compute motion trajectory assignment for current process based on rank_id/world_size (if multi-processing)
        # referring to how pytorch distributed launch is designed at https://pytorch.org/docs/stable/elastic/run.html
        if dist.is_initialized():
            # multi-processing, compute the trajectory assignment
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            local_rank = 0
            world_size = None
        self._assign_motion_trajectories_for_this_process(local_rank, world_size)
        self._assign_motion_buffer_for_envs()
        # TODO: Step: store interaction information (with terrain or rigidobjects) if needed.
        # wait for events to match with the entities in the scene.

    def _initialize_robot_kinematics(self):
        with open(self.cfg.robot_model_path) as f:
            self._robot_kinematics_chain = pk.build_chain_from_urdf(f.read()).to(dtype=torch.float, device=self.device)
        # joint_pos_pk = joint_pos_isaac[_joint_order_isaac_to_pk]
        self._joint_order_isaac_to_pk = torch.ones(self._view.max_dofs, device=self.device, dtype=torch.long) * -1
        for joint_i, joint_name in enumerate(self._robot_kinematics_chain.get_joint_parameter_names()):
            if not joint_name in self.isaac_joint_names:
                raise RuntimeError(
                    f"Joint name {joint_name} in the robot kinematics chain is not found in the physics simulation"
                    " view."
                )

            joint_idx_isaac = self.isaac_joint_names.index(joint_name)
            self._joint_order_isaac_to_pk[joint_i] = joint_idx_isaac
        # TODO: Considering tracking link is already known from config,
        # forward kinematics can be accelerated by ONNX or JIT.

    def _reset_motion_buffers(self, env_ids: Sequence[int] | torch.Tensor):
        """Reset each motion buffers if is selected by the given env_ids."""
        for name in self._motion_buffers.keys():
            env_ids_assignment = self._motion_buffer_assignment[name]
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            self._motion_buffers[name].reset(
                env_ids=env_ids[env_ids_this_buffer_mask],  # type: ignore
                symmetric_augmentation_mask_buffer=self._env_symmetric_augmentation_mask,
            )

    def _resample_buffer_collate_params(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Resample values for motion collate behaviors for the given env_ids. E.g. frame_interval_s,
        symmetric_augmentation_mask.
        """
        # resample the frame interval
        if env_ids is None and isinstance(self.cfg.frame_interval_s, Sequence):
            self._frame_interval_s = (
                torch.rand(self._view.count, device=self.device)
                * (self.cfg.frame_interval_s[1] - self.cfg.frame_interval_s[0])
                + self.cfg.frame_interval_s[0]
            )
        elif isinstance(self.cfg.frame_interval_s, Sequence):
            self._frame_interval_s[env_ids] = (
                torch.rand(len(env_ids), device=self.device)  # type: ignore
                * (self.cfg.frame_interval_s[1] - self.cfg.frame_interval_s[0])
                + self.cfg.frame_interval_s[0]
            )
        elif (not hasattr(self, "_frame_interval_s")) and (not isinstance(self.cfg.frame_interval_s, Sequence)):
            self._frame_interval_s = torch.ones(self._view.count, device=self.device) * self.cfg.frame_interval_s
        elif hasattr(self, "_frame_interval_s") and (not isinstance(self.cfg.frame_interval_s, Sequence)):
            pass
        else:
            raise ValueError("Invalid frame_interval_s configuration.")

        # resample the symmetric augmentation mask
        if env_ids is None or not hasattr(self, "_env_symmetric_augmentation_mask"):
            # Specifying whether the symmetric augmentation should be applied to the given env_ids.
            # The values in self._data should be already symmetrically augmented.
            self._env_symmetric_augmentation_mask = torch.zeros(self._view.count, device=self.device, dtype=torch.bool)

    def _resample_update_period(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Resample the update period for the given env_ids."""
        if env_ids is None and isinstance(self.cfg.update_period, (list, tuple)):
            # initialize the update period so that the sensor manager will update the sensor at the given period
            self.cfg = copy(self.cfg)
            self.cfg.update_period_range = copy(self.cfg.update_period)
            self.cfg.update_period = (
                torch.ones(
                    self._view.count,
                    device=self.device,
                )
                * self.cfg.update_period[0]
            )
            env_ids = self._ALL_INDICES  # type: ignore

        if isinstance(self.cfg.update_period, torch.Tensor) and self.cfg.update_period_sample_strategy is not None:
            if self.cfg.update_period_sample_strategy == "uniform":
                self.cfg.update_period[env_ids] = (
                    torch.rand(
                        len(env_ids),  # type: ignore
                        device=self.device,
                    )
                    * (self.cfg.update_period_range[1] - self.cfg.update_period_range[0])
                    + self.cfg.update_period_range[0]
                )  # type: ignore
            elif self.cfg.update_period_sample_strategy == "uniform_frame_limits":
                # Currently, assuming sample frame indices are still linearly increasing.
                max_motion_ref_length_s = self.frame_interval_s * self.cfg.num_frames  # (len(env_ids),)
                self.cfg.update_period[env_ids] = (
                    torch.rand(
                        len(env_ids),  # type: ignore
                        device=self.device,
                    )
                    * (max_motion_ref_length_s[env_ids] - self.frame_interval_s[env_ids])
                    + self.frame_interval_s[env_ids]
                )
            else:
                raise ValueError(f"Unknown update_period_sample_strategy: {self.cfg.update_period_sample_strategy}")

    def _assign_motion_trajectories_for_this_process(self, local_rank: int = 0, world_size: int | None = None):
        """Assign the motion trajectories for the current process based on the rank_id/world_size.
        `self._motion_buffer_num_trajectories` will be updated. As for the current process, only a subset of
        trajectories will be enabled. Not-used motion buffer (python object) will be deleted.
        """
        if world_size is None or self.cfg.mp_split_method == "None":
            # single process, enable all trajectories for all motion buffers.
            # no split method, enable all trajectories for all motion buffers.
            for buffer in self._motion_buffers.values():
                buffer.enable_trajectories()
        elif self.cfg.mp_split_method == "Even":
            # split the trajectories evenly (round-robin) for each process.
            # |----motion1-----|----motion2-----|----motion3-----|----motion4-----|----motion5-----|
            # [123412341234123412341234123412341234123412341234123412341234123412341234123412341234]
            idxs = torch.arange(
                sum(self._motion_buffer_num_trajectories.values()),
                device=self.device,
                dtype=torch.long,
            )
            if len(idxs) % world_size != 0:
                idxs = idxs[: -(len(idxs) % world_size)]
            idxs = idxs.reshape(-1, world_size)
            idxs = idxs[:, local_rank]
            buffer_traj_start = 0
            for buffer_name in list(self._motion_buffers.keys()):
                buffer = self._motion_buffers[buffer_name]
                buffer_traj_end = buffer.num_trajectories + buffer_traj_start
                idxs_this_buffer = idxs[(idxs >= buffer_traj_start) & (idxs < buffer_traj_end)] - buffer_traj_start
                if len(idxs_this_buffer) > 0:
                    buffer.enable_trajectories(idxs_this_buffer)
                else:
                    buffer = self._motion_buffers.pop(buffer_name)
                    del buffer
                buffer_traj_start = buffer_traj_end
        elif self.cfg.mp_split_method == "Segment":
            # split the trajectories based on the total number of trajectories.
            # Assuming all trajectories are put in a single sequence, each process will be assigned a segment.
            # |----motion1-----|----motion2-----|----motion3-----|----motion4-----|----motion5-----|
            # |-------------------process1-------------------|-------------------process2----------|
            total_traj_global = sum(self._motion_buffer_num_trajectories.values())
            num_traj_per_process = total_traj_global // world_size
            process_traj_start = num_traj_per_process * local_rank
            process_traj_end = (
                num_traj_per_process * (local_rank + 1) if local_rank < world_size - 1 else total_traj_global
            )
            buffer_traj_start = 0
            for buffer_name in list(self._motion_buffers.keys()):
                buffer = self._motion_buffers[buffer_name]
                buffer_traj_end = buffer.num_trajectories + buffer_traj_start
                buffer_slice_start = np.clip(process_traj_start, buffer_traj_start, buffer_traj_end) - buffer_traj_start
                buffer_slice_end = np.clip(process_traj_end, buffer_traj_start, buffer_traj_end) - buffer_traj_start
                if buffer_slice_start < buffer_slice_end:
                    buffer.enable_trajectories(slice(buffer_slice_start, buffer_slice_end))
                else:
                    buffer = self._motion_buffers.pop(buffer_name)
                    del buffer
                buffer_traj_start = buffer_traj_end
        else:
            raise ValueError(f"Unknown mp_split_method: {self.cfg.mp_split_method}")

    def _assign_motion_buffer_for_envs(self):
        """Assign which motion buffer is used for each envs. But defining this at startup stage leads to more stable
        buffer data behavior. It is just like the fixed terrain type in curriculum-based locomotion training setting.
        """
        # compute the ratio of trajectories for each motion buffer and how many envs should be assigned to each buffer
        total_traj = sum(self._motion_buffer_num_trajectories.values())
        traj_ratio_d = {name: num_traj / total_traj for name, num_traj in self._motion_buffer_num_trajectories.items()}
        num_env_assignment_d = {name: int(ratio * self._view.count) for name, ratio in traj_ratio_d.items()}

        # check if the assignment matches all envs, otherwise, slightly add/subtract from the largest/smallest
        if sum(num_env_assignment_d.values()) > self._view.count:
            overflow_num = sum(num_env_assignment_d.values()) - self._view.count
            max_num_env_name = max(num_env_assignment_d.keys(), key=num_env_assignment_d.get)  # type: ignore
            num_env_assignment_d[max_num_env_name] -= overflow_num
        elif sum(num_env_assignment_d.values()) < self._view.count:
            underflow_num = self._view.count - sum(num_env_assignment_d.values())
            min_num_env_name = min(num_env_assignment_d.keys(), key=num_env_assignment_d.get)  # type: ignore
            num_env_assignment_d[min_num_env_name] += underflow_num

        # assign the motion buffer for each envs
        self._motion_buffer_assignment: dict[str, slice] = dict()
        start_idx = 0
        for name, num_env in num_env_assignment_d.items():
            self._motion_buffer_assignment[name] = slice(start_idx, start_idx + num_env)
            start_idx += num_env

        # inform the motion buffer so that they can prepare their buffers.
        for name, buffer in self._motion_buffers.items():
            buffer.set_env_ids_assignments(self._motion_buffer_assignment[name])

    """
    Data Augmentation implementations.
    """

    @property
    def _symmetric_augmentation_conditions_met(self) -> bool | torch.Tensor:
        return (
            (self.cfg.symmetric_augmentation_joint_mapping is not None)
            and (self.cfg.symmetric_augmentation_joint_reverse_buf is not None)
            and (self.num_link_to_ref == 0 or self.cfg.symmetric_augmentation_link_mapping is not None)
            and self._env_symmetric_augmentation_mask.any()
        )

    def _symmetric_augment_joint_buffer(self, joint_pos_buf: torch.Tensor):
        """Symmetrically augment the joint position/velocity buffer, values changed in place.
        Assuming the last dimension is the joint dimension.
        """
        num_dims = len(joint_pos_buf.shape)
        joint_reverse_buf = torch.tensor(self.cfg.symmetric_augmentation_joint_reverse_buf, device=self.device)
        if num_dims > 1:
            for _ in range(num_dims - 1):
                joint_reverse_buf = joint_reverse_buf.unsqueeze(0)
        joint_pos_buf[..., :] = joint_pos_buf[..., self.cfg.symmetric_augmentation_joint_mapping] * joint_reverse_buf
        return joint_pos_buf

    def _symmetric_augment_ang_vel_buffer(self, ang_vel_buf: torch.Tensor):
        """Symmetrically augment the angular velocity buffer, values changed in place.
        Assuming the last dimension is the angular velocity dimension.
        By mirror w.r.t x-z plane, the y component and z component of angular velocity should be negated.
        """
        ang_vel_buf[..., 0] *= -1
        ang_vel_buf[..., 2] *= -1
        return ang_vel_buf

    def _symmetric_augment_link_pos_buffer(self, link_pos_buf: torch.Tensor):
        """Symmetrically augment the link position/rotation buffer w.r.t x-z plane, values changed in place.
        Assuming the last dimension is the position dimension, the second last dimension is the link dimension.
        """
        link_pos_buf[:] = link_pos_buf[..., self.cfg.symmetric_augmentation_link_mapping, :]
        link_pos_buf[..., 1] *= -1
        return link_pos_buf

    def _symmetric_augment_link_quat_buffer(self, link_quat_buf: torch.Tensor):
        """Symmetrically augment the link quaternion buffer w.r.t x-z plane, values changed in place.
        Assuming the last dimension is the quaternion (w, x, y, z) dimension, the second last dimension is the link dimension.
        """
        # mirror the quaternion w.r.t x-z plane
        # from https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
        link_quat_buf[:] = link_quat_buf[..., self.cfg.symmetric_augmentation_link_mapping, :]
        link_quat_buf[..., 1] *= -1
        link_quat_buf[..., 3] *= -1
        return link_quat_buf

    def _symmetric_augment_quat_buffer(self, quat_buf: torch.Tensor):
        """Symmetrically augment the quaternion buffer w.r.t x-z plane, values changed in place.
        Assuming the last dimension is the quaternion (w, x, y, z) dimension.
        """
        # from https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
        quat_buf[..., 1] *= -1
        quat_buf[..., 3] *= -1
        return quat_buf

    def _symmetric_augment_reference_data(self, data_buf: MotionReferenceData, env_ids: torch.Tensor | slice):
        """Symmetrically augment all parts of the reference data buffer w.r.t x-z plane, values changed in place."""
        data_buf.joint_pos[env_ids] = self._symmetric_augment_joint_buffer(
            data_buf.joint_pos[env_ids],
        )
        data_buf.joint_vel[env_ids] = self._symmetric_augment_joint_buffer(
            data_buf.joint_vel[env_ids],
        )
        data_buf.base_pos_w[env_ids, :, 1] *= -1
        data_buf.base_quat_w[env_ids] = self._symmetric_augment_quat_buffer(
            data_buf.base_quat_w[env_ids],
        )
        data_buf.base_lin_vel_w[env_ids, :, 1] *= -1
        data_buf.base_ang_vel_w[env_ids] = self._symmetric_augment_ang_vel_buffer(
            data_buf.base_ang_vel_w[env_ids],
        )
        data_buf.link_pos_w[env_ids] = self._symmetric_augment_link_pos_buffer(
            data_buf.link_pos_w[env_ids],
        )
        data_buf.link_quat_w[env_ids] = self._symmetric_augment_link_quat_buffer(
            data_buf.link_quat_w[env_ids],
        )
        data_buf.link_pos_b[env_ids] = self._symmetric_augment_link_pos_buffer(
            data_buf.link_pos_b[env_ids],
        )
        data_buf.link_quat_b[env_ids] = self._symmetric_augment_link_quat_buffer(
            data_buf.link_quat_b[env_ids],
        )
        data_buf.link_lin_vel_b[env_ids, :, 1] *= -1
        data_buf.link_ang_vel_b[env_ids] = self._symmetric_augment_ang_vel_buffer(
            data_buf.link_ang_vel_b[env_ids],
        )
        data_buf.link_lin_vel_w[env_ids, :, 1] *= -1
        data_buf.link_ang_vel_w[env_ids] = self._symmetric_augment_ang_vel_buffer(
            data_buf.link_ang_vel_w[env_ids],
        )

    """
    Visualization.
    """

    def _find_reference_view(self):
        """Find the ArticulationView to serve as a motion reference visualization."""
        if (
            hasattr(self, "_backend")
            and hasattr(self, "_physics_sim_view")
            and (not self.cfg.reference_prim_path is None)
        ):
            self._reference_view: physx.ArticulationView = get_articulation_view(
                self.cfg.reference_prim_path, self._physics_sim_view
            )
            if self._reference_view._backend is None:
                delattr(self, "_reference_view")
                raise RuntimeError("Failed to find the reference view.")

    def _set_reference_view_state(self):
        """Set the articulation view to the reference state for motion visualization."""
        if self.cfg.visualizing_robot_from == "aiming_frame":
            aiming_frame_idx = self.aiming_frame_idx
            robot_pos_w = self.data.base_pos_w[self.ALL_INDICES, aiming_frame_idx].clone()
            robot_quat_w_ = self.data.base_quat_w[self.ALL_INDICES, aiming_frame_idx]
            robot_quat_w = math_utils.convert_quat(robot_quat_w_, to="xyzw")
            robot_joint_pos = self.data.joint_pos[self.ALL_INDICES, aiming_frame_idx]
        elif self.cfg.visualizing_robot_from == "reference_frame":
            robot_pos_w = self.reference_frame.base_pos_w[self.ALL_INDICES, 0].clone()
            robot_quat_w_ = self.reference_frame.base_quat_w[self.ALL_INDICES, 0]
            robot_quat_w = math_utils.convert_quat(robot_quat_w_, to="xyzw")
            robot_joint_pos = self.data.joint_pos[self.ALL_INDICES, 0]
        else:
            raise ValueError(f"Unsupported cfg.visualizing_robot_from: {self.cfg.visualizing_robot_from}")

        # add position offset in case the reference is overlapping with the real robot.
        robot_pos_w[:, 0] += self.cfg.visualizing_robot_offset[0]
        robot_pos_w[:, 1] += self.cfg.visualizing_robot_offset[1]
        robot_pos_w[:, 2] += self.cfg.visualizing_robot_offset[2]

        # set the root transform
        root_pose_w = torch.cat([robot_pos_w, robot_quat_w], dim=-1)
        self._reference_view.set_root_transforms(
            root_pose_w,
            indices=self.ALL_INDICES,
        )
        self._reference_view.set_root_velocities(
            torch.zeros_like(root_pose_w[..., :6]),
            indices=self.ALL_INDICES,
        )

        # set the joint positions
        self._reference_view.set_dof_positions(
            robot_joint_pos,
            indices=self.ALL_INDICES,
        )
        self._reference_view.set_dof_velocities(
            torch.zeros_like(robot_joint_pos),
            indices=self.ALL_INDICES,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis and self.cfg.visualizing_marker_types:
            if not hasattr(self, "_visualizer"):
                self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self._visualizer.set_visibility(True)
        else:
            if hasattr(self, "_visualizer"):
                self._visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.cfg.visualizing_marker_types:
            if not hasattr(self, "_view") or not hasattr(self, "_visualizer"):
                return
            aiming_frame_idx = self.aiming_frame_idx

            pos_list = []
            quat_list = []
            index_list = []

            if "root" in self.cfg.visualizing_marker_types:
                root_pos_w = self.data.base_pos_w[self.ALL_INDICES, aiming_frame_idx]
                root_quat_w = self.data.base_quat_w[self.ALL_INDICES, aiming_frame_idx]
                root_indices = torch.ones(self._view.count, device=self.device, dtype=torch.long) * 0

                pos_list.append(root_pos_w.reshape(-1, 3))
                quat_list.append(root_quat_w.reshape(-1, 4))
                index_list.append(root_indices.reshape(-1))

            if "links" in self.cfg.visualizing_marker_types:
                link_pos_w = self.data.link_pos_w[self.ALL_INDICES, aiming_frame_idx]
                link_quat_w = self.data.link_quat_w[self.ALL_INDICES, aiming_frame_idx]
                link_indices = torch.ones(link_pos_w.shape[:2], device=self.device, dtype=torch.long) * 1

                pos_list.append(link_pos_w.reshape(-1, 3))
                quat_list.append(link_quat_w.reshape(-1, 4))
                index_list.append(link_indices.reshape(-1))

            if "relative_links" in self.cfg.visualizing_marker_types:
                link_pos_w = self.reference_link_pos_relative_w
                link_quat_w = self.reference_link_quat_relative_w
                link_indices = torch.ones(link_pos_w.shape[:2], device=self.device, dtype=torch.long) * 2

                pos_list.append(link_pos_w.reshape(-1, 3))
                quat_list.append(link_quat_w.reshape(-1, 4))
                index_list.append(link_indices.reshape(-1))

            if pos_list:
                pos_w = torch.cat(pos_list, dim=0)
                quat_w = torch.cat(quat_list, dim=0)
                indices = torch.cat(index_list, dim=0)
                self._visualizer.visualize(
                    translations=pos_w,
                    orientations=quat_w,
                    marker_indices=indices,
                )

        if not hasattr(self, "_reference_view"):
            self._find_reference_view()
        if hasattr(self, "_reference_view"):
            self._set_reference_view_state()

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        delattr(self, "_physics_sim_view")
        delattr(self, "_view")
        if hasattr(self, "_reference_view"):
            delattr(self, "_reference_view")
