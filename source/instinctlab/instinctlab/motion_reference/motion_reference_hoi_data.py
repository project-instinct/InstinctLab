from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass, fields

import warp as wp

from isaaclab.utils.warp import ProxyArray

import instinctlab.utils.torch as torch_utils

from .motion_reference_data import MotionReferenceData, MotionReferenceState, MotionSequence


@dataclass
class HoiMotionSequence(MotionSequence):
    """A motion reference buffer that supports human-object interaction."""

    object_pos_w: torch.Tensor = None  # type: ignore
    """ The positions of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 3], N is the batch size.
    """

    object_quat_w: torch.Tensor = None  # type: ignore
    """ The quaternions of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 4], in (x, y, z, w), N is the batch size.
    """

    object_lin_vel_w: torch.Tensor = None  # type: ignore
    """ The linear velocities of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 3], N is the batch size.
    """

    object_ang_vel_w: torch.Tensor = None  # type: ignore
    """ The angular velocities of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 3], N is the batch size.
    """

    object_validity: torch.Tensor = None  # type: ignore
    """ A boolean tensor indicating whether the object data is valid or not.

    Shape: [N, num_frames, num_objects], N is the batch size.
    """

    scene_object_names: list[str] = None
    """ The names of the objects in the scene. """

    attrs_with_frame_dim: tuple = (
        "joint_pos",
        "joint_vel",
        "base_pos_w",
        "base_lin_vel_w",
        "base_quat_w",
        "base_ang_vel_w",
        "link_pos_b",
        "link_quat_b",
        "link_pos_w",
        "link_quat_w",
        "link_lin_vel_b",
        "link_ang_vel_b",
        "link_lin_vel_w",
        "link_ang_vel_w",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
        "object_ang_vel_w",
        "object_validity",
    )

    @staticmethod
    def make_empty(
        batch_size: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        num_objects: int,
        device=torch.device("cpu"),
    ) -> HoiMotionSequence:
        return_ = HoiMotionSequence(
            joint_pos=torch.zeros(batch_size, num_frames, num_joints, device=device),
            joint_vel=torch.zeros(batch_size, num_frames, num_joints, device=device),
            base_pos_w=torch.zeros(batch_size, num_frames, 3, device=device),
            base_lin_vel_w=torch.zeros(batch_size, num_frames, 3, device=device),
            base_quat_w=torch.zeros(batch_size, num_frames, 4, device=device),
            base_ang_vel_w=torch.zeros(batch_size, num_frames, 3, device=device),
            framerate=torch.zeros(batch_size, dtype=torch.int, device=device),
            buffer_length=torch.zeros(batch_size, dtype=torch.int, device=device),
            link_pos_b=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_quat_b=torch.zeros(batch_size, num_frames, num_links, 4, device=device),
            link_pos_w=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_quat_w=torch.zeros(batch_size, num_frames, num_links, 4, device=device),
            link_lin_vel_b=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_ang_vel_b=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_lin_vel_w=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_ang_vel_w=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            object_pos_w=torch.zeros(batch_size, num_frames, num_objects, 3, device=device),
            object_quat_w=torch.zeros(batch_size, num_frames, num_objects, 4, device=device),
            object_lin_vel_w=torch.zeros(batch_size, num_frames, num_objects, 3, device=device),
            object_ang_vel_w=torch.zeros(batch_size, num_frames, num_objects, 3, device=device),
            object_validity=torch.zeros(batch_size, num_frames, num_objects, dtype=torch.bool, device=device),
        )
        return_.base_quat_w[:, :, 3] = 1.0
        return_.link_quat_b[:, :, :, 3] = 1.0
        return_.link_quat_w[:, :, :, 3] = 1.0
        return_.object_quat_w[:, :, :, 3] = 1.0
        return return_

    @staticmethod
    def make_emtpy_concat_batch(
        buffer_lengths: Sequence[int],
        num_joints: int,
        num_links: int,
        num_objects: int,
        device=torch.device("cpu"),
    ) -> HoiMotionSequence:
        return_ = HoiMotionSequence(
            joint_pos=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_joints,), device=device
            ),  # type: ignore
            joint_vel=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_joints,), device=device
            ),  # type: ignore
            base_pos_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(3,), device=device),  # type: ignore
            base_lin_vel_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(3,), device=device),  # type: ignore
            base_quat_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(4,), device=device),  # type: ignore
            base_ang_vel_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(3,), device=device),  # type: ignore
            framerate=torch.zeros((len(buffer_lengths),), device=device),
            buffer_length=torch.tensor(buffer_lengths, device=device),
            link_pos_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_quat_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 4), device=device
            ),  # type: ignore
            link_pos_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_quat_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 4), device=device
            ),  # type: ignore
            link_lin_vel_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_ang_vel_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_lin_vel_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_ang_vel_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            object_pos_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_objects, 3), device=device
            ),  # type: ignore
            object_quat_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_objects, 4), device=device
            ),  # type: ignore
            object_lin_vel_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_objects, 3), device=device
            ),  # type: ignore
            object_ang_vel_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_objects, 3), device=device
            ),  # type: ignore
            object_validity=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_objects,), device=device, dtype=torch.bool
            ),  # type: ignore
        )
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        return_.base_quat_w.fill_data(identity_quat)  # type: ignore
        return_.link_quat_b.fill_data(identity_quat.unsqueeze(0).expand(num_links, -1))  # type: ignore
        return_.link_quat_w.fill_data(identity_quat.unsqueeze(0).expand(num_links, -1))  # type: ignore
        return_.object_quat_w.fill_data(identity_quat.unsqueeze(0).expand(num_objects, -1))  # type: ignore
        return return_


@dataclass
class HoiMotionReferenceData(MotionReferenceData):
    """A motion reference data that supports human-object interaction."""

    object_pos_w: torch.Tensor = None  # type: ignore
    """ The positions of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 3], N is the batch size.
    """

    object_quat_w: torch.Tensor = None  # type: ignore
    """ The quaternions of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 4], in (x, y, z, w), N is the batch size.
    """

    object_lin_vel_w: torch.Tensor = None  # type: ignore
    """ The linear velocities of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 3], N is the batch size.
    """

    object_ang_vel_w: torch.Tensor = None  # type: ignore
    """ The angular velocities of the objects (w.r.t the world).

    Shape: [N, num_frames, num_objects, 3], N is the batch size.
    """

    object_validity: torch.Tensor = None  # type: ignore
    """ A boolean tensor indicating whether the object data is valid or not.

    Shape: [N, num_frames, num_objects], N is the batch size.
    """

    scene_object_names: list[str] = None
    """ The names of the objects in the scene. """

    @staticmethod
    def make_empty(
        num_envs: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        num_objects: int,
        device=torch.device("cpu"),
        scene_object_names: list[str] = None,
    ) -> HoiMotionReferenceData:
        return_ = HoiMotionReferenceData(
            joint_pos=ProxyArray(wp.zeros((num_envs, num_frames, num_joints), device=str(device))),
            joint_vel=ProxyArray(wp.zeros((num_envs, num_frames, num_joints), device=str(device))),
            joint_pos_mask=ProxyArray(wp.ones((num_envs, num_frames, num_joints), dtype=wp.bool, device=str(device))),
            joint_vel_mask=ProxyArray(wp.ones((num_envs, num_frames, num_joints), dtype=wp.bool, device=str(device))),
            base_pos_w=ProxyArray(wp.zeros((num_envs, num_frames, 3), device=str(device))),
            base_lin_vel_w=ProxyArray(wp.zeros((num_envs, num_frames, 3), device=str(device))),
            base_quat_w=ProxyArray(wp.zeros((num_envs, num_frames, 4), device=str(device))),
            base_ang_vel_w=ProxyArray(wp.zeros((num_envs, num_frames, 3), device=str(device))),
            base_pos_plane_mask=ProxyArray(wp.ones((num_envs, num_frames), dtype=wp.bool, device=str(device))),
            base_pos_height_mask=ProxyArray(wp.ones((num_envs, num_frames), dtype=wp.bool, device=str(device))),
            base_orientation_mask=ProxyArray(wp.ones((num_envs, num_frames), dtype=wp.bool, device=str(device))),
            base_heading_mask=ProxyArray(wp.ones((num_envs, num_frames), dtype=wp.bool, device=str(device))),
            link_pos_b=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 3), device=str(device))),
            link_quat_b=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 4), device=str(device))),
            link_pos_w=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 3), device=str(device))),
            link_quat_w=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 4), device=str(device))),
            link_lin_vel_b=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 3), device=str(device))),
            link_ang_vel_b=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 3), device=str(device))),
            link_lin_vel_w=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 3), device=str(device))),
            link_ang_vel_w=ProxyArray(wp.zeros((num_envs, num_frames, num_links, 3), device=str(device))),
            link_pos_mask=ProxyArray(wp.ones((num_envs, num_frames, num_links), dtype=wp.bool, device=str(device))),
            link_rot_mask=ProxyArray(wp.ones((num_envs, num_frames, num_links), dtype=wp.bool, device=str(device))),
            time_to_target_frame=ProxyArray(wp.zeros((num_envs, num_frames), device=str(device))),
            validity=ProxyArray(wp.zeros((num_envs, num_frames), dtype=wp.bool, device=str(device))),
            object_pos_w=ProxyArray(wp.zeros((num_envs, num_frames, num_objects, 3), device=str(device))),
            object_quat_w=ProxyArray(wp.zeros((num_envs, num_frames, num_objects, 4), device=str(device))),
            object_lin_vel_w=ProxyArray(wp.zeros((num_envs, num_frames, num_objects, 3), device=str(device))),
            object_ang_vel_w=ProxyArray(wp.zeros((num_envs, num_frames, num_objects, 3), device=str(device))),
            object_validity=ProxyArray(
                wp.zeros((num_envs, num_frames, num_objects), dtype=wp.bool, device=str(device))
            ),
            scene_object_names=scene_object_names,
        )
        return_.base_quat_w[:, :, 3] = 1.0
        return_.link_quat_b[:, :, :, 3] = 1.0
        return_.link_quat_w[:, :, :, 3] = 1.0
        return_.object_quat_w[:, :, :, 3] = 1.0
        return return_

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Reset the motion reference data for the given env_ids. Zeros tensors and marks as invalid."""
        super().reset(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.validity.torch.device)
        parent_field_names = {f.name for f in fields(MotionReferenceData)}
        for field in fields(self):
            if field.name in parent_field_names:
                continue
            tensor = getattr(self, field.name)
            if not isinstance(tensor, ProxyArray) or tensor.shape[0] != self.validity.shape[0]:
                continue

            if field.name == "object_validity":
                tensor[env_ids] = False
            else:
                tensor[env_ids] = 0
                if "quat" in field.name:
                    tensor[env_ids, ..., 3] = 1.0


@dataclass
class HoiMotionReferenceState(MotionReferenceState):
    """The robot and object state at the beginning of the motion reference. For the purpose of reset."""

    object_pos_w: torch.Tensor = None  # type: ignore
    object_quat_w: torch.Tensor = None  # type: ignore
    object_lin_vel_w: torch.Tensor = None  # type: ignore
    object_ang_vel_w: torch.Tensor = None  # type: ignore
    object_validity: torch.Tensor = None  # type: ignore
    scene_object_names: list[str] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.object_pos_w is None:
            assert (
                self.object_pos_w.shape[0] == self.base_pos_w.shape[0]
            ), "The first dimension of object_pos_w should be the same as base_pos_w."

    @staticmethod
    def make_empty(
        num_envs: int,
        num_joints: int,
        num_objects: int,
        device=torch.device("cpu"),
        scene_object_names: list[str] = None,
        **kwargs,
    ) -> HoiMotionReferenceState:
        return_ = HoiMotionReferenceState(
            joint_pos=ProxyArray(wp.zeros((num_envs, num_joints), device=str(device))),
            joint_vel=ProxyArray(wp.zeros((num_envs, num_joints), device=str(device))),
            base_pos_w=ProxyArray(wp.zeros((num_envs, 3), device=str(device))),
            base_quat_w=ProxyArray(wp.zeros((num_envs, 4), device=str(device))),
            base_lin_vel_w=ProxyArray(wp.zeros((num_envs, 3), device=str(device))),
            base_ang_vel_w=ProxyArray(wp.zeros((num_envs, 3), device=str(device))),
            object_pos_w=ProxyArray(wp.zeros((num_envs, num_objects, 3), device=str(device))),
            object_quat_w=ProxyArray(wp.zeros((num_envs, num_objects, 4), device=str(device))),
            object_lin_vel_w=ProxyArray(wp.zeros((num_envs, num_objects, 3), device=str(device))),
            object_ang_vel_w=ProxyArray(wp.zeros((num_envs, num_objects, 3), device=str(device))),
            object_validity=ProxyArray(wp.zeros((num_envs, num_objects), dtype=wp.bool, device=str(device))),
            scene_object_names=scene_object_names,
        )
        return_.base_quat_w[:, 3] = 1.0
        return_.object_quat_w[:, :, 3] = 1.0
        return return_

    def __getitem__(self, idx: Sequence[int] | torch.Tensor | None):
        if idx is None:
            return self

        return HoiMotionReferenceState(
            joint_pos=self.joint_pos[idx],
            joint_vel=self.joint_vel[idx],
            base_pos_w=self.base_pos_w[idx],
            base_quat_w=self.base_quat_w[idx],
            base_lin_vel_w=self.base_lin_vel_w[idx],
            base_ang_vel_w=self.base_ang_vel_w[idx],
            object_pos_w=self.object_pos_w[idx],
            object_quat_w=self.object_quat_w[idx],
            object_lin_vel_w=self.object_lin_vel_w[idx],
            object_ang_vel_w=self.object_ang_vel_w[idx],
            object_validity=self.object_validity[idx],
            scene_object_names=self.scene_object_names,
        )
