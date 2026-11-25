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

from .amass_motion import AmassMotion

if TYPE_CHECKING:
    from .contact_motion_cfg import ContactMotionCfg


def next_onset_index(flag: np.ndarray) -> np.ndarray:
    """``(T, B)`` index of the first contact onset at or after each frame, ``T`` where there is none.

    An onset is a frame in contact whose previous frame is not (frame 0 counts when it is in contact).
    """
    num_frames = flag.shape[0]
    onset = flag & ~np.concatenate([np.zeros_like(flag[:1]), flag[:-1]], axis=0)
    index = np.where(onset, np.arange(num_frames)[:, None], num_frames)
    return np.minimum.accumulate(index[::-1], axis=0)[::-1]


class ContactMotion(AmassMotion):
    """Retargeted motions whose files carry per-frame contact labels (``contact_body_names``, ``contact_flag (T, B)``,
    ``contact_pos_w (T, B, 3)``, 0 where not in contact).

    At load time each clip gets its contact schedule at the buffer's framerate: the contact flag and position, and the
    next contact onset at or after the frame within ``contact_onset_horizon_s`` (its time and position; time -1 and
    position 0 if there is none). The goal is the reference base pose a lookahead ahead, capped at the clip end; the
    lookahead is sampled from ``goal_lookahead_range_s`` at reset and every ``goal_lookahead_resample_interval_s``. The
    data buffer must be a :class:`ContactMotionReferenceData` with the same ``contact_body_names``.
    """

    cfg: ContactMotionCfg

    def _read_retargetted_motion_file(self, filepath: str) -> ContactMotionSequence:
        sequence = super()._read_retargetted_motion_file(filepath)
        raw_data = np.load(filepath, allow_pickle=True)
        num_bodies = len(self.cfg.contact_body_names)
        num_source = int(raw_data["joint_pos"].shape[0])
        keys = ("contact_body_names", "contact_flag", "contact_pos_w")
        if all(key in raw_data.files for key in keys):
            names = [str(n) for n in raw_data["contact_body_names"]]
            missing = [n for n in self.cfg.contact_body_names if n not in names]
            if missing:
                raise KeyError(f"{filepath} has no contact labels for {missing}")
            order = [names.index(n) for n in self.cfg.contact_body_names]
            flag = raw_data["contact_flag"][:, order].astype(bool)
            pos = raw_data["contact_pos_w"][:, order].astype(np.float32)
        elif self.cfg.contact_keys_required:
            missing_keys = [key for key in keys if key not in raw_data.files]
            raise KeyError(f"{filepath} has no {missing_keys}; annotate it with scripts/annotate_motion_contacts.py")
        else:
            flag = np.zeros((num_source, num_bodies), dtype=bool)
            pos = np.zeros((num_source, num_bodies, 3), dtype=np.float32)
        source_framerate = float(np.asarray(raw_data["framerate"]).item())

        onset = next_onset_index(flag)
        frames_to_onset = onset - np.arange(num_source)[:, None]
        valid = (onset < num_source) & (frames_to_onset <= self.cfg.contact_onset_horizon_s * source_framerate)
        onset_time = np.where(valid, frames_to_onset / source_framerate, -1.0).astype(np.float32)
        next_pos = np.where(valid[..., None], pos[np.minimum(onset, num_source - 1), np.arange(num_bodies)], 0.0)
        pos = np.where(flag[..., None], pos, 0.0)

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
            contact_pos_w=tensor(pos[source]),
            next_contact_onset_time=tensor(onset_time[source]),
            next_contact_pos_w=tensor(next_pos[source]),
        )

    def _load_motion_sequences(self):
        """Load all clips into one concatenated :class:`ContactMotionSequence`."""
        print(f"[ContactMotion] Loading motion files, should be {len(self._all_motion_files)} in total...")
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
        print(f"[ContactMotion] All {len(all_motion_sequences)} motion files loaded.")

    def _sample_assigned_env_starting_stub(self, env_ids: torch.Tensor | None = None) -> None:
        """Also sample each episode's goal lookahead."""
        super()._sample_assigned_env_starting_stub(env_ids)
        if not hasattr(self, "_assigned_env_goal_lookahead_s"):
            self._assigned_env_goal_lookahead_s = torch.zeros(self.num_assigned_envs, device=self.buffer_device)
            self._assigned_env_goal_resample_at_s = torch.zeros(self.num_assigned_envs, device=self.buffer_device)
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        self._sample_goal_lookahead(assigned_ids)
        self._assigned_env_goal_resample_at_s[assigned_ids] = self.cfg.goal_lookahead_resample_interval_s

    def _sample_goal_lookahead(self, assigned_ids: torch.Tensor) -> None:
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

        # resample the goal lookahead of episodes that passed their resampling time
        episode_time_s = sample_timestamp[:, 0].to(self.buffer_device)
        due = episode_time_s >= self._assigned_env_goal_resample_at_s[assigned_ids]
        if due.any():
            due_ids = assigned_ids[due]
            interval = self.cfg.goal_lookahead_resample_interval_s
            self._sample_goal_lookahead(due_ids)
            next_resample_s = (torch.floor(episode_time_s[due] / interval) + 1) * interval
            self._assigned_env_goal_resample_at_s[due_ids] = next_resample_s

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

        contact = {}
        for attr in CONTACT_FIELDS_WITH_FRAME_DIM:
            data = getattr(sequences, attr)[flat_motion, flat_frame]
            contact[attr] = data.reshape(len(env_ids), num_frames, *data.shape[1:]).to(self.output_device)
        # past the clip end there is no contact schedule
        flag = contact["contact_flag"] & validity.unsqueeze(-1)
        onset_time = torch.where(validity.unsqueeze(-1), contact["next_contact_onset_time"], -1.0)
        next_valid = onset_time >= 0.0
        data_buffer.contact_flag[env_ids] = flag
        data_buffer.contact_pos_w[env_ids] = (contact["contact_pos_w"] + origins[:, None, None]) * flag.unsqueeze(-1)
        data_buffer.next_contact_onset_time[env_ids] = onset_time
        data_buffer.next_contact_pos_w[env_ids] = (
            contact["next_contact_pos_w"] + origins[:, None, None]
        ) * next_valid.unsqueeze(-1)

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
