from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass, fields

import warp as wp

from isaaclab.utils.warp import ProxyArray

import instinctlab.utils.torch as torch_utils

from .motion_reference_data import MotionReferenceData, MotionSequence

CONTACT_FIELDS_WITH_FRAME_DIM = ("contact_flag", "contact_pos_w", "next_contact_onset_time", "next_contact_pos_w")


@dataclass
class ContactMotionSequence(MotionSequence):
    """A motion reference buffer with the contact schedule of selected bodies.

    Contact positions are points on the terrain, in the clip frame (the world frame once the motion origin is added).
    """

    contact_flag: torch.Tensor = None  # type: ignore
    """ Whether each contact body is in contact. Shape: [N, num_frames, num_contact_bodies], bool. """

    contact_pos_w: torch.Tensor = None  # type: ignore
    """ Contact position of each body, 0 where it is not in contact. Shape: [N, num_frames, num_contact_bodies, 3]. """

    next_contact_onset_time: torch.Tensor = None  # type: ignore
    """ Seconds until each body's next contact onset at or after the frame (0 if the onset is this frame), -1 if there
    is none within the onset horizon. Shape: [N, num_frames, num_contact_bodies]. """

    next_contact_pos_w: torch.Tensor = None  # type: ignore
    """ Position of that onset, 0 where there is none. Shape: [N, num_frames, num_contact_bodies, 3]. """

    attrs_with_frame_dim: tuple = MotionSequence.attrs_with_frame_dim + CONTACT_FIELDS_WITH_FRAME_DIM

    @staticmethod
    def make_emtpy_concat_batch(
        buffer_lengths: Sequence[int],
        num_joints: int,
        num_links: int,
        num_contact_bodies: int,
        device=torch.device("cpu"),
    ) -> ContactMotionSequence:
        base = MotionSequence.make_emtpy_concat_batch(buffer_lengths, num_joints, num_links, device=device)

        def concat(data_shape, dtype=torch.float32):
            return torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=data_shape, device=device, dtype=dtype
            )

        return_ = ContactMotionSequence(
            **{f.name: getattr(base, f.name) for f in fields(MotionSequence) if f.name != "attrs_with_frame_dim"},
            contact_flag=concat((num_contact_bodies,), torch.bool),  # type: ignore
            contact_pos_w=concat((num_contact_bodies, 3)),  # type: ignore
            next_contact_onset_time=concat((num_contact_bodies,)),  # type: ignore
            next_contact_pos_w=concat((num_contact_bodies, 3)),  # type: ignore
        )
        return return_


@dataclass
class ContactMotionReferenceData(MotionReferenceData):
    """Motion reference data with the contact schedule of ``contact_body_names`` (fields of
    :class:`ContactMotionSequence`) and a goal."""

    contact_flag: torch.Tensor = None  # type: ignore
    contact_pos_w: torch.Tensor = None  # type: ignore
    next_contact_onset_time: torch.Tensor = None  # type: ignore
    next_contact_pos_w: torch.Tensor = None  # type: ignore

    goal_pos_w: torch.Tensor = None  # type: ignore
    """ Reference base position a lookahead after the frame, clamped to the clip end. [N, num_frames, 3]. """

    goal_quat_w: torch.Tensor = None  # type: ignore
    """ Reference base orientation at the same time, (x, y, z, w). Shape: [N, num_frames, 4]. """

    contact_body_names: list[str] = None  # type: ignore
    """ Names of the contact bodies, in the order of the last dimension of the contact fields. """

    @staticmethod
    def make_empty(
        num_envs: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        contact_body_names: Sequence[str],
        device=torch.device("cpu"),
        **kwargs,
    ) -> ContactMotionReferenceData:
        base = MotionReferenceData.make_empty(num_envs, num_frames, num_joints, num_links, device=device)
        num_bodies = len(contact_body_names)
        device = str(device)
        return_ = ContactMotionReferenceData(
            **{f.name: getattr(base, f.name) for f in fields(MotionReferenceData)},
            contact_flag=ProxyArray(wp.zeros((num_envs, num_frames, num_bodies), dtype=wp.bool, device=device)),
            contact_pos_w=ProxyArray(wp.zeros((num_envs, num_frames, num_bodies, 3), device=device)),
            next_contact_onset_time=ProxyArray(wp.zeros((num_envs, num_frames, num_bodies), device=device)),
            next_contact_pos_w=ProxyArray(wp.zeros((num_envs, num_frames, num_bodies, 3), device=device)),
            goal_pos_w=ProxyArray(wp.zeros((num_envs, num_frames, 3), device=device)),
            goal_quat_w=ProxyArray(wp.zeros((num_envs, num_frames, 4), device=device)),
            contact_body_names=list(contact_body_names),
        )
        return_.goal_quat_w[:, :, 3] = 1.0
        return_.next_contact_onset_time[:, :, :] = -1.0
        return return_

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        """Base reset (flags False, positions 0, quaternions identity), and no next contact onset (time -1)."""
        super().reset(env_ids)
        self.next_contact_onset_time[env_ids] = -1.0
