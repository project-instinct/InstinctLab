from __future__ import annotations

import numpy as np
import torch
from dataclasses import fields
from typing import TYPE_CHECKING

from instinctlab.motion_reference.motion_reference_contact_data import (
    CONTACT_FIELDS_WITH_FRAME_DIM,
    ContactMotionReferenceData,
    ContactMotionSequence,
)
from instinctlab.motion_reference.motion_reference_data import MotionSequence

from .terrain_motion import TerrainMotion

if TYPE_CHECKING:
    from .terrain_contact_motion_cfg import TerrainContactMotionCfg


def next_onset_index(flag: np.ndarray) -> np.ndarray:
    """``(T, B)`` index of the first contact onset at or after each frame, ``T`` where there is none.

    An onset is a frame in contact whose previous frame is not (frame 0 counts when it is in contact).
    """
    num_frames = flag.shape[0]
    onset = flag & ~np.concatenate([np.zeros_like(flag[:1]), flag[:-1]], axis=0)
    index = np.where(onset, np.arange(num_frames)[:, None], num_frames)
    return np.minimum.accumulate(index[::-1], axis=0)[::-1]


class TerrainContactMotion(TerrainMotion):
    """Terrain motions whose files carry per-frame contact labels (``contact_body_names``, ``contact_flag (T, B)``,
    ``contact_pos_w (T, B, 3)``, see ``scripts/annotate_motion_contacts.py``).

    At load time each clip gets its contact schedule at the buffer's framerate: the contact flag and the next contact
    onset (at or after the frame, within ``contact_onset_horizon_s``; zero and invalid if none). The goal is the
    reference base pose a lookahead ahead, capped at the clip end; the lookahead is sampled per episode from
    ``goal_lookahead_range_s``. The data buffer must be a :class:`ContactMotionReferenceData` with the same
    ``contact_body_names``.
    """

    cfg: TerrainContactMotionCfg

    def _read_retargetted_motion_file(self, filepath: str) -> ContactMotionSequence:
        sequence = super()._read_retargetted_motion_file(filepath)
        raw_data = np.load(filepath, allow_pickle=True)
        for key in ("contact_body_names", "contact_flag", "contact_pos_w"):
            if key not in raw_data.files:
                raise KeyError(f"{filepath} has no '{key}'; annotate it with scripts/annotate_motion_contacts.py")
        names = [str(n) for n in raw_data["contact_body_names"]]
        missing = [n for n in self.cfg.contact_body_names if n not in names]
        if missing:
            raise KeyError(f"{filepath} has no contact labels for {missing}")
        order = [names.index(n) for n in self.cfg.contact_body_names]
        flag = raw_data["contact_flag"][:, order].astype(bool)
        pos = raw_data["contact_pos_w"][:, order].astype(np.float32)
        source_framerate = float(np.asarray(raw_data["framerate"]).item())
        num_source = flag.shape[0]

        onset = next_onset_index(flag)
        valid = (onset < num_source) & (
            onset - np.arange(num_source)[:, None] <= self.cfg.contact_onset_horizon_s * source_framerate
        )
        next_pos = np.where(valid[..., None], pos[np.minimum(onset, num_source - 1), np.arange(len(order))], 0.0)

        # nearest source frame for every frame of the (possibly resampled) sequence
        framerate = float(sequence.framerate)
        num_frames = int(sequence.buffer_length)
        frames = np.arange(num_frames)
        source = np.clip(np.round(frames * source_framerate / framerate).astype(int), 0, num_source - 1)

        def tensor(array, dtype=torch.float):
            return torch.as_tensor(array, dtype=dtype, device=self.buffer_device)

        return ContactMotionSequence(
            **{f.name: getattr(sequence, f.name) for f in fields(MotionSequence) if f.name != "attrs_with_frame_dim"},
            contact_flag=tensor(flag[source], torch.bool),
            next_contact_pos_w=tensor(next_pos[source]),
            next_contact_valid=tensor(valid[source], torch.bool),
        )

    def _load_motion_sequences(self):
        """Load all clips into one concatenated :class:`ContactMotionSequence`."""
        print(f"[TerrainContactMotion] Loading motion files, should be {len(self._all_motion_files)} in total...")
        all_motion_sequences = [self._read_motion_file(i) for i in range(len(self._all_motion_files))]
        self._all_motion_sequences = ContactMotionSequence.make_emtpy_concat_batch(
            buffer_lengths=[int(motion.buffer_length) for motion in all_motion_sequences],
            num_joints=self.articulation_access.max_dofs,
            num_links=self.num_link_to_ref,
            num_contact_bodies=len(self.cfg.contact_body_names),
            device=self.buffer_device,
        )
        for i, motion in enumerate(all_motion_sequences):
            for attr in self._all_motion_sequences.attrs_with_frame_dim:
                getattr(self._all_motion_sequences, attr)[i, : motion.buffer_length] = getattr(motion, attr)
            for attr in self._all_motion_sequences.attrs_only_batch_dim:
                getattr(self._all_motion_sequences, attr)[i] = getattr(motion, attr)
        print(f"[TerrainContactMotion] All {len(all_motion_sequences)} motion files loaded.")

    def _sample_assigned_env_starting_stub(self, env_ids: torch.Tensor | None = None) -> None:
        """Also sample each episode's goal lookahead."""
        super()._sample_assigned_env_starting_stub(env_ids)
        if not hasattr(self, "_assigned_env_goal_lookahead_s"):
            self._assigned_env_goal_lookahead_s = torch.zeros(self.num_assigned_envs, device=self.buffer_device)
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        low, high = self.cfg.goal_lookahead_range_s
        self._assigned_env_goal_lookahead_s[assigned_ids] = low + (high - low) * torch.rand(
            len(assigned_ids), device=self.buffer_device
        )

    def fill_motion_data(
        self,
        env_ids: torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: ContactMotionReferenceData,
    ) -> None:
        super().fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)
        if list(data_buffer.contact_body_names) != list(self.cfg.contact_body_names):
            raise ValueError(
                f"data buffer contact bodies {data_buffer.contact_body_names} differ from the motion buffer's"
                f" {list(self.cfg.contact_body_names)}"
            )
        if len(env_ids) == 0:
            return
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        sequences = self._all_motion_sequences
        # same frame selection as AmassMotion.fill_motion_data; frames past the clip end read its last frame
        frames = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp.to(self.buffer_device))
            * sequences.framerate[motion_ids].unsqueeze(-1)
        )
        last = sequences.buffer_length[motion_ids].unsqueeze(-1) - 1
        frames = torch.clamp(torch.minimum(frames, last), min=0).long()
        num_frames = frames.shape[1]
        flat_motion = motion_ids.unsqueeze(-1).expand(-1, num_frames).flatten()
        flat_frame = frames.flatten()
        origins = self._get_motion_based_origin(env_origins, env_ids).to(self.output_device)
        validity = data_buffer.validity[env_ids]

        for attr in CONTACT_FIELDS_WITH_FRAME_DIM:
            data = getattr(sequences, attr)[flat_motion, flat_frame]
            data = data.reshape(len(env_ids), num_frames, *data.shape[1:]).to(self.output_device)
            if attr == "next_contact_pos_w":
                data = data + origins[:, None, None]
            if data.dtype == torch.bool:
                data = data & validity.reshape(validity.shape + (1,) * (data.dim() - 2))
            getattr(data_buffer, attr)[env_ids] = data

        goal_frames = (
            frames
            + torch.round(
                self._assigned_env_goal_lookahead_s[assigned_ids].unsqueeze(-1)
                * sequences.framerate[motion_ids].unsqueeze(-1)
            ).long()
        )
        goal_frames = torch.minimum(goal_frames, last).flatten()
        goal_pos = sequences.base_pos_w[flat_motion, goal_frames].reshape(len(env_ids), num_frames, 3)
        goal_quat = sequences.base_quat_w[flat_motion, goal_frames].reshape(len(env_ids), num_frames, 4)
        data_buffer.goal_pos_w[env_ids] = goal_pos.to(self.output_device) + origins[:, None]
        data_buffer.goal_quat_w[env_ids] = goal_quat.to(self.output_device)
