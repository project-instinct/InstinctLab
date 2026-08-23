from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass, fields

import warp as wp

from isaaclab.utils.warp import ProxyArray

import instinctlab.utils.torch as torch_utils


@dataclass
class MotionSequence:
    """A standard motion reference buffer that can be used to load into memory. Additional link poses are precomputed
    for faster access during simulation.
    Each motion can have leading dims as [N, ...] or [...] with no batch dimension.
    """

    joint_pos: torch.Tensor = None  # type: ignore
    """ Positions of the joints of the given robot model.

    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    joint_vel: torch.Tensor = None  # type: ignore
    """ Velocities of the joints of the given robot model.
    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    base_pos_w: torch.Tensor = None  # type: ignore
    """ Positions of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    base_lin_vel_w: torch.Tensor = None  # type: ignore
    """ Linear velocities of the base of the robot model.
    Shape: [N, num_frames, 3], N is the batch size.
    """

    base_quat_w: torch.Tensor = None  # type: ignore
    """ Quaternions of the base of the robot model.

    Shape: [N, num_frames, 4], in (x, y, z, w), N is the batch size.
    """

    base_ang_vel_w: torch.Tensor = None  # type: ignore
    """ Angular velocities of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    framerate: torch.Tensor = None  # type: ignore
    """ The framerate of the motion data.

    Shape: [N,], N is the batch size.
    """

    buffer_length: torch.Tensor = None  # type: ignore
    """ The number of frames of each motion data across batch

    Shape: [N,], N is the batch size.
    """

    link_pos_b: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the base frame of the motion reference).
    NOTE: This is a passive parameter computed from joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_quat_b: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the base).
    NOTE: This is a passive parameter computed from joint_poss.

    Shape: [N, num_frames, num_links, 4], in (x, y, z, w), N is the batch size.
    """

    link_pos_w: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_quat_w: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the world).

    Shape: [N, num_frames, num_links, 4], in (x, y, z, w), N is the batch size.
    """

    link_lin_vel_b: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the base frame of the motion reference).
    NOTE: This is a passive parameter computed from joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_ang_vel_b: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the base frame of the motion reference).
    NOTE: This is a passive parameter computed from joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_lin_vel_w: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_ang_vel_w: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

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
    )
    attrs_only_batch_dim: tuple = ("framerate", "buffer_length")

    @staticmethod
    def make_empty(
        batch_size: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        device=torch.device("cpu"),
    ) -> MotionSequence:
        return_ = MotionSequence(
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
        )
        return_.base_quat_w[:, :, 3] = 1.0
        return_.link_quat_b[:, :, :, 3] = 1.0
        return_.link_quat_w[:, :, :, 3] = 1.0
        return return_

    @staticmethod
    def make_emtpy_concat_batch(
        buffer_lengths: Sequence[int],
        num_joints: int,
        num_links: int,
        device=torch.device("cpu"),
    ) -> MotionSequence:
        return_ = MotionSequence(
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
        )
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        return_.base_quat_w.fill_data(identity_quat)  # type: ignore
        return_.link_quat_b.fill_data(identity_quat.unsqueeze(0).expand(num_links, -1))  # type: ignore
        return_.link_quat_w.fill_data(identity_quat.unsqueeze(0).expand(num_links, -1))  # type: ignore
        return return_

    def __post_init__(self):
        if not self.joint_pos is None:
            # all attributes should have the same leading dimensions
            assert all(
                getattr(self, attr).shape[0] == self.joint_pos.shape[0] for attr in self.attrs_with_frame_dim
            ), "All attributes should have the same leading dimension."
        else:
            # all attributes should be None
            assert all(
                getattr(self, attr) is None for attr in (self.attrs_with_frame_dim + self.attrs_only_batch_dim)
            ), "All attributes should be None or not None."


@dataclass
class MotionReferenceData:
    """Each motion can have leading dims as [N, ...] or [...] with no batch dimension."""

    joint_pos: torch.Tensor = None  # type: ignore
    """ Positions of the joints in urdf coordinate of the given robot model.

    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    joint_vel: torch.Tensor = None  # type: ignore
    """ Velocities of the joints of the given robot model.

    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    joint_pos_mask: torch.Tensor = None  # type: ignore
    """
    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    joint_vel_mask: torch.Tensor = None  # type: ignore
    """
    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    base_pos_w: torch.Tensor = None  # type: ignore
    """ Positions of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    base_lin_vel_w: torch.Tensor = None  # type: ignore
    """ Linear velocities of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    base_quat_w: torch.Tensor = None  # type: ignore
    """ Quaternions of the base of the robot model.

    Shape: [N, num_frames, 4], in (x, y, z, w), N is the batch size.
    """

    base_ang_vel_w: torch.Tensor = None  # type: ignore
    """ Angular velocities of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    base_pos_plane_mask: torch.Tensor = None  # type: ignore
    """ Shape: [N, num_frames], N is the batch size for each term. """
    base_pos_height_mask: torch.Tensor = None  # type: ignore
    """ Shape: [N, num_frames], N is the batch size for each term. """
    base_orientation_mask: torch.Tensor = None  # type: ignore
    """ Shape: [N, num_frames], N is the batch size for each term. """
    base_heading_mask: torch.Tensor = None  # type: ignore
    """ The masks for the base position and orientation.

    Shape: [N, num_frames], N is the batch size for each term.
    """

    link_pos_b: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_quat_b: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 4], in (x, y, z, w), N is the batch size.
    """

    link_pos_w: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_quat_w: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the world).

    Shape: [N, num_frames, num_links, 4], in (x, y, z, w), N is the batch size.
    """

    link_lin_vel_b: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_ang_vel_b: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_lin_vel_w: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_ang_vel_w: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    link_pos_mask: torch.Tensor = None  # type: ignore
    """ The mask for the link positions.

    Shape: [N, num_frames, num_links], N is the batch size.
    """

    link_rot_mask: torch.Tensor = None  # type: ignore
    """ The mask for the link rotations.

    Shape: [N, num_frames, num_links], N is the batch size.
    """

    time_to_target_frame: torch.Tensor = None  # type: ignore
    """ The time expected to reach the specific frame.

    Shape: [N, num_frames], N is the batch size.
    """

    validity: torch.Tensor = None  # type: ignore
    """ A boolean tensor indicating whether the current data is valid or not.
    Usually indicating the data buffer is exhausted or not.

    Shape: [N, num_frames], N is the batch size.
    """

    @staticmethod
    def make_empty(
        num_envs: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        device=torch.device("cpu"),
        **kwargs,
    ) -> MotionReferenceData:
        return_ = MotionReferenceData(
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
        )
        return_.base_quat_w[:, :, 3] = 1.0
        return_.link_quat_b[:, :, :, 3] = 1.0
        return_.link_quat_w[:, :, :, 3] = 1.0
        return return_

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Reset the motion reference data for the given env_ids. Zeros tensors and marks as invalid."""
        env_ids = torch.as_tensor(env_ids, device=self.validity.torch.device)
        for field in fields(self):
            tensor = getattr(self, field.name)
            if not isinstance(tensor, ProxyArray) or tensor.shape[0] != self.validity.shape[0]:
                continue
            if field.name == "validity":
                tensor[env_ids] = False
            elif field.name == "time_to_target_frame":
                tensor[env_ids] = -1.0
            elif field.name.endswith("_mask"):
                tensor[env_ids] = True
            else:
                tensor[env_ids] = 0
                if "quat" in field.name:
                    tensor[env_ids, ..., 3] = 1.0


@dataclass
class MotionReferenceState:
    """The robot state at the beginning of the motion reference. For the purpose of reset."""

    joint_pos: torch.Tensor = None  # type: ignore
    joint_vel: torch.Tensor = None  # type: ignore
    base_pos_w: torch.Tensor = None  # type: ignore
    base_quat_w: torch.Tensor = None  # type: ignore
    base_lin_vel_w: torch.Tensor = None  # type: ignore
    base_ang_vel_w: torch.Tensor = None  # type: ignore

    # Since link positions can be acquired directly from forward kinematics, it is not necessary to store them here.

    def __post_init__(self):
        if not self.joint_pos is None:
            assert (
                self.joint_pos.shape == self.joint_vel.shape
            ), "The shape of joint_pos and joint_vel should be the same."
            assert (
                self.joint_pos.shape[0] == self.base_pos_w.shape[0] == self.base_quat_w.shape[0]
            ), "The first dimension of joint_pos, base_pos, base_quat, and base_lin_vel should be the same."
        else:
            assert (
                self.joint_pos is None
                and self.joint_vel is None
                and self.base_pos_w is None
                and self.base_quat_w is None
                and self.base_lin_vel_w is None
                and self.base_ang_vel_w is None
            ), "All attributes should be None or not None."

    @staticmethod
    def make_empty(
        num_envs: int,
        num_joints: int,
        device=torch.device("cpu"),
        **kwargs,
    ) -> MotionReferenceState:
        return_ = MotionReferenceState(
            joint_pos=ProxyArray(wp.zeros((num_envs, num_joints), device=str(device))),
            joint_vel=ProxyArray(wp.zeros((num_envs, num_joints), device=str(device))),
            base_pos_w=ProxyArray(wp.zeros((num_envs, 3), device=str(device))),
            base_quat_w=ProxyArray(wp.zeros((num_envs, 4), device=str(device))),
            base_lin_vel_w=ProxyArray(wp.zeros((num_envs, 3), device=str(device))),
            base_ang_vel_w=ProxyArray(wp.zeros((num_envs, 3), device=str(device))),
        )
        return_.base_quat_w[:, 3] = 1.0
        return return_

    def __getitem__(self, idx: Sequence[int] | torch.Tensor | None):
        if idx is None:
            return self

        return MotionReferenceState(
            joint_pos=self.joint_pos[idx],
            joint_vel=self.joint_vel[idx],
            base_pos_w=self.base_pos_w[idx],
            base_quat_w=self.base_quat_w[idx],
            base_lin_vel_w=self.base_lin_vel_w[idx],
            base_ang_vel_w=self.base_ang_vel_w[idx],
        )
