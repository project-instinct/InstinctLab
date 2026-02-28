from __future__ import annotations

import inspect
import os
import pickle as pkl
import yaml
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from instinctlab.motion_reference import MotionReferenceData, MotionReferenceState, MotionSequence
from instinctlab.motion_reference.motion_buffer import MotionBuffer
from instinctlab.motion_reference.utils import estimate_angular_velocity, estimate_velocity
from instinctlab.utils.torch import ConcatBatchTensor

if TYPE_CHECKING:
    from .amass_motion_cfg import AmassMotionCfg

import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Sequence

import pytorch_kinematics as pk
import warp as wp


@wp.kernel
def sample_bin_kernel(
    motion_ids: wp.array(dtype=wp.int32),  # type: ignore
    batch_starts: wp.array(dtype=wp.int32),  # type: ignore
    batch_sizes: wp.array(dtype=wp.int32),  # type: ignore
    cdf: wp.array(dtype=wp.float32),  # type: ignore
    output: wp.array(dtype=wp.float32),  # type: ignore
    bin_length_s: wp.float32,
    seed: wp.int32,
):
    tid = wp.tid()
    state = wp.rand_init(seed + tid)
    motion = motion_ids[tid]
    start = batch_starts[motion]
    size = batch_sizes[motion]

    total = cdf[start + size - 1]
    if total == wp.float(0.0):
        output[tid] = wp.float(0.0)
        return
    u = wp.randf(state, wp.float(0.0), total)

    low = int(0)
    high = size - 1
    while low < high:
        mid = (low + high) // 2
        if cdf[start + mid] < u:
            low = mid + 1
        else:
            high = mid
    selected = low

    bin_idx = wp.float(selected) + wp.randf(state, wp.float(0.0), wp.float(1.0))
    output[tid] = bin_idx * bin_length_s


def sample_start_times(
    motion_ids: torch.Tensor,
    motion_bin_weights: ConcatBatchTensor,
    buffer_device: torch.device,
    motion_bin_length_s: float | None,
) -> torch.Tensor:
    if motion_bin_length_s is None:
        raise ValueError("motion_bin_length_s must be provided")

    num_assign = len(motion_ids)
    temp_starts = torch.empty(num_assign, dtype=torch.float32, device=buffer_device)

    global_cum = torch.cumsum(motion_bin_weights._concatenated_tensor, dim=0)
    prev_ends = motion_bin_weights._batch_starts[1:] - 1
    segment_offsets_values = torch.cat([torch.tensor([0.0], device=global_cum.device), global_cum[prev_ends]])
    segment_lengths = motion_bin_weights._batch_sizes
    offsets = torch.repeat_interleave(segment_offsets_values, segment_lengths)
    flat_cdf = global_cum - offsets

    wp.launch(
        kernel=sample_bin_kernel,
        dim=num_assign,
        inputs=[
            wp.from_torch(motion_ids.to(torch.int32), dtype=wp.int32),  # type: ignore
            wp.from_torch(motion_bin_weights._batch_starts.to(torch.int32), dtype=wp.int32),  # type: ignore
            wp.from_torch(motion_bin_weights._batch_sizes.to(torch.int32), dtype=wp.int32),  # type: ignore
            wp.from_torch(flat_cdf, dtype=wp.float32),  # type: ignore
            wp.from_torch(temp_starts, dtype=wp.float32),
            motion_bin_length_s,
            torch.randint(0, (1 << 31) - 1, (1,), device=buffer_device).item(),
        ],
        device=buffer_device,
    )

    return temp_starts


class AmassMotion(MotionBuffer):
    """Processing AMASS formatted file structure.
    NOTE: the indexing logic is as follows:
        self._env_buffer_selection: corresponding _motion_buffer index of each env
        self._buffer_motion_selection: the indices of the motion files to use in each motion buffer
    """

    cfg: AmassMotionCfg

    def __init__(
        self,
        cfg: AmassMotionCfg,
        *args,
        **kwargs,
    ):
        """Initialize the motion dataset.

        Args:
            cfg: The configuration parameters for the motion dataset.
            robot_chain: The pytorch_kinematics.Chain for computing positions for link_of_interests.
        """
        # initialize base class
        super().__init__(cfg, *args, **kwargs)
        self.output_device = self.device
        if self.cfg.buffer_device == "output_device":
            self.buffer_device = self.output_device
        else:
            self.buffer_device = torch.device("cpu")
        self._joint_limits = self._joint_limits.to(self.buffer_device)
        self._refresh_motion_file_list()
        self._prepare_retargetting_func()

    """
    Properties.
    """

    @property
    def num_trajectories(self) -> int:
        return len(self._all_motion_files)

    @property
    def complete_motion_lengths(self) -> torch.Tensor:
        return (
            self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[:]]
            / self._all_motion_sequences.framerate[self._assigned_env_motion_selection[:]]
        )

    @property
    def assigned_motion_lengths(self) -> torch.Tensor:
        return (
            (
                self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection]
                / self._all_motion_sequences.framerate[self._assigned_env_motion_selection]
            )
            - self._motion_buffer_start_time_s
        ).to(self.output_device)

    """
    Operations.
    """

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        if not traj_ids is None:
            if isinstance(traj_ids, torch.Tensor):
                self._all_motion_files = [self._all_motion_files[i] for i in traj_ids]
            elif isinstance(traj_ids, slice):
                self._all_motion_files = self._all_motion_files[traj_ids]
            self._motion_weights = self._motion_weights[traj_ids].to(self.buffer_device)
        self._load_motion_sequences()
        if self.cfg.motion_bin_length_s is not None:
            self._init_motion_bin_weights()

    def set_env_ids_assignments(self, env_ids):
        return_ = super().set_env_ids_assignments(env_ids)

        # set env_motion selection buffer
        self._sample_assigned_env_starting_stub()

        return return_

    def reset(self, env_ids: Sequence[int] | torch.Tensor, symmetric_augmentation_mask_buffer: torch.Tensor) -> None:
        self._sample_assigned_env_starting_stub(env_ids)
        # sample symmetric augmentation mask. In AMASS, all motions are able to be augmented.
        symmetric_augmentation_mask_buffer[env_ids] = torch.randint(
            0,
            2,
            (len(env_ids),),
            device=self.output_device,
            dtype=torch.bool,
        )

    def fill_init_reference_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
        state_buffer: MotionReferenceState,
    ) -> None:
        assert self.env_ids_is_assigned(env_ids).all(), "The env_ids should be assigned to this motion buffer."
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        frame_selection = torch.floor(
            self._motion_buffer_start_time_s[assigned_ids]
            * self._all_motion_sequences.framerate[self._assigned_env_motion_selection[assigned_ids]]
        ).to(torch.long)

        base_pos_w = self._all_motion_sequences.base_pos_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]
        base_quat_w = self._all_motion_sequences.base_quat_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        base_lin_vel_w = self._all_motion_sequences.base_lin_vel_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]
        base_ang_vel_w = self._all_motion_sequences.base_ang_vel_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        joint_pos = self._all_motion_sequences.joint_pos[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        joint_vel = self._all_motion_sequences.joint_vel[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        base_pos_w += self._get_motion_based_origin(env_origins, env_ids)

        # check if any link is below the ground
        if self.cfg.ensure_link_below_zero_ground:
            link_pos_w = self._all_motion_sequences.link_pos_w[
                self._assigned_env_motion_selection[assigned_ids], frame_selection
            ].to(self.output_device)
            min_link_height = link_pos_w[..., 2].min(dim=-1).values
            if (min_link_height < 0).any():
                # if any link is below the ground, raise it
                base_pos_w[..., 2] += -min_link_height
        base_pos_w[..., 2] += self.cfg.motion_start_height_offset

        state_buffer.joint_pos[env_ids] = joint_pos.to(self.output_device)
        state_buffer.joint_vel[env_ids] = joint_vel.to(self.output_device)
        state_buffer.base_pos_w[env_ids] = base_pos_w.to(self.output_device)
        state_buffer.base_quat_w[env_ids] = base_quat_w.to(self.output_device)
        state_buffer.base_lin_vel_w[env_ids] = base_lin_vel_w.to(self.output_device)
        state_buffer.base_ang_vel_w[env_ids] = base_ang_vel_w.to(self.output_device)

    def fill_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        # compute the frame selection for each env
        frame_selections = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp.to(self.buffer_device))
            * self._all_motion_sequences.framerate[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
        )  # shape [len(env_ids), num_frames]

        # check if the sample_frame_idx is out of range
        data_buffer.validity[env_ids] = (
            (
                frame_selections
                < self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(
                    -1
                )
            )
            .to(torch.bool)
            .to(self.output_device)
        )
        frame_selections = torch.where(
            data_buffer.validity[env_ids].to(self.buffer_device),
            frame_selections,
            self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
            - 1,
        ).to(torch.long)

        # flattened indexing to get the data from all motion sequences
        assigned_ids_across_frame = assigned_ids.unsqueeze(-1).repeat(
            1, frame_selections.shape[1]
        )  # shape [len(env_ids), num_frames]
        assigned_ids_across_frame_ = assigned_ids_across_frame.flatten()
        frame_selections_ = frame_selections.flatten()

        # joint
        data_buffer.joint_pos[env_ids] = (
            self._all_motion_sequences.joint_pos[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.joint_pos.shape[1:])
            .to(self.output_device)
        )
        data_buffer.joint_vel[env_ids] = (
            self._all_motion_sequences.joint_vel[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.joint_vel.shape[1:])
            .to(self.output_device)
        )
        # link_pos_b and link_quat_b
        data_buffer.link_pos_b[env_ids] = (
            self._all_motion_sequences.link_pos_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_pos_b.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_quat_b[env_ids] = (
            self._all_motion_sequences.link_quat_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_quat_b.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_pos_w[env_ids] = (
            self._all_motion_sequences.link_pos_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_pos_w.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_quat_w[env_ids] = (
            self._all_motion_sequences.link_quat_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_quat_w.shape[1:])
            .to(self.output_device)
        )
        # link_lin_vel_b and link_ang_vel_b
        data_buffer.link_lin_vel_b[env_ids] = (
            self._all_motion_sequences.link_lin_vel_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_lin_vel_b.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_ang_vel_b[env_ids] = (
            self._all_motion_sequences.link_ang_vel_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_ang_vel_b.shape[1:])
            .to(self.output_device)
        )
        # link_lin_vel_w and link_ang_vel_w
        data_buffer.link_lin_vel_w[env_ids] = (
            self._all_motion_sequences.link_lin_vel_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_lin_vel_w.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_ang_vel_w[env_ids] = (
            self._all_motion_sequences.link_ang_vel_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_ang_vel_w.shape[1:])
            .to(self.output_device)
        )

        # base_pos and base_quat
        base_pos_w = self._all_motion_sequences.base_pos_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 3)
        base_lin_vel_w = self._all_motion_sequences.base_lin_vel_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 3)
        base_quat_w = self._all_motion_sequences.base_quat_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 4)
        base_ang_vel_w = self._all_motion_sequences.base_ang_vel_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 3)
        data_buffer.base_pos_w[env_ids] = base_pos_w.to(self.output_device)
        data_buffer.base_lin_vel_w[env_ids] = base_lin_vel_w.to(self.output_device)
        data_buffer.base_quat_w[env_ids] = base_quat_w.to(self.output_device)
        data_buffer.base_ang_vel_w[env_ids] = base_ang_vel_w.to(self.output_device)

        # update the base_pos_w and link_pos_w with the env_origins
        data_buffer.base_pos_w[env_ids] += self._get_motion_based_origin(env_origins, env_ids).unsqueeze(
            1
        )  # avoiding inplace operation
        data_buffer.link_pos_w[env_ids] += (
            self._get_motion_based_origin(env_origins, env_ids).unsqueeze(1).unsqueeze(1)
        )  # avoiding inplace operation

    def get_current_motion_identifiers(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> list[str]:
        """Get the identifiers of the motion files for each env.
        Args:
            env_ids: The env IDs to get the motion identifiers for.
        Returns:
            A list of motion identifiers for each env.
        """
        if env_ids is None:
            return [self._all_motion_files[i] for i in self._assigned_env_motion_selection]
        else:
            assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
            return [self._all_motion_files[i] for i in self._assigned_env_motion_selection[assigned_ids]]

    def get_current_motion_weights(self, env_ids=None):
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device) if env_ids is not None else None
        if not hasattr(self, "_motion_weights"):
            return super().get_current_motion_weights(env_ids)
        else:
            return self._motion_weights[self._assigned_env_motion_selection[assigned_ids]]

    def update_motion_weights(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        weight_ratio: float | torch.Tensor = 1.0,
    ) -> None:
        """Update the motion weights for sampling.
        ## Args:
            - env_ids: The environment IDs that should be updated. If None, update all envs.
        """
        if not hasattr(self, "_motion_weights"):
            return super().update_motion_weights(env_ids, weight_ratio)
        if env_ids is not None:
            env_ids = self.env_ids_to_assigned_ids(env_ids)
        self._motion_weights[self._assigned_env_motion_selection[env_ids]] *= weight_ratio
        # normalize the motion weights
        self._motion_weights = torch.clamp(self._motion_weights, min=0.0)
        self._motion_weights /= self._motion_weights.sum()

    """
    Internal helper functions (for loading motion files and calling retargetting functions).
    """

    def _init_motion_weights(self, motion_weights: torch.Tensor | None = None) -> None:
        """Initialize sampling weights for each motion file and their temporal segments."""
        # motion weights
        if motion_weights is not None:
            self._motion_weights = torch.tensor(
                motion_weights,
                device=self.buffer_device,
                dtype=torch.float,
            )
        else:
            self._motion_weights = torch.ones(
                len(self._all_motion_files),
                device=self.buffer_device,
                dtype=torch.float,
            )

        # safety check
        assert len(self._all_motion_files) == len(self._motion_weights), (
            "The number of motion files and motion weights should be the same."
            f"Got {len(self._all_motion_files)} motion files and {len(self._motion_weights)} motion weights."
        )

    def _init_motion_bin_weights(self) -> None:
        print("Initializing motion bin weights motion_start_from_middle_range is disabled.")
        num_motion_bins = torch.floor(
            (self._all_motion_sequences.buffer_length / self._all_motion_sequences.framerate).to(self.buffer_device)
            / self.cfg.motion_bin_length_s
        ).to(torch.long)
        self._motion_bin_weights = ConcatBatchTensor(
            batch_sizes=num_motion_bins,  # type: ignore
            data_shape=tuple(),
            device=self.buffer_device,
        )
        for motion_idx in range(len(self._all_motion_files)):
            motion_bin_probability = torch.ones(
                num_motion_bins[motion_idx],  # type: ignore
                device=self.buffer_device,
                dtype=torch.float,
            )
            motion_bin_probability /= motion_bin_probability.sum()
            self._motion_bin_weights[motion_idx] = motion_bin_probability

    def _refresh_motion_file_list(self):
        """Refresh the motion files list and motion weights from the file system."""
        # search through all the files in the directory and record the files ending with "_poses.npz"
        if self.cfg.filtered_motion_selection_filepath is not None:
            if self.cfg.subset_selection is not None:
                print("Warning: subset_selection is ignored when filtered_motion_files is provided.")
            with open(self.cfg.filtered_motion_selection_filepath) as f:
                yaml_data = yaml.safe_load(f)
            all_motion_files = yaml_data["selected_files"]
            for i, motion_file in enumerate(all_motion_files):
                all_motion_files[i] = os.path.join(self.cfg.path, motion_file)
        elif self.cfg.subset_selection is not None:
            all_motion_files = []
            for subset, subjects in self.cfg.subset_selection.items():
                if subjects is None:
                    # all subjects in the subset
                    subjects = os.listdir(os.path.join(self.cfg.path, subset))
                for subject in subjects:
                    for root, _, files in os.walk(os.path.join(self.cfg.path, subset, subject)):
                        for file in files:
                            for endings in self.cfg.supported_file_endings:
                                if file.endswith(endings):
                                    all_motion_files.append(os.path.join(root, file))
        else:
            all_motion_files = []
            for root, _, files in os.walk(self.cfg.path, followlinks=True):
                for file in files:
                    for endings in self.cfg.supported_file_endings:
                        if file.endswith(endings):
                            all_motion_files.append(os.path.join(root, file))
        self._all_motion_files: list[str] = all_motion_files

        # hack to override motion file list for debugging
        # NOTE: mind the _motion_weights value.
        # self._all_motion_files = [
        #     # "/home/leo/Datasets/AMASS/CMU/140/140_04_poses.npz",
        #     # "/home/leo/Datasets/AMASS/CMU/140/140_07_poses.npz",
        #     # "/home/leo/Datasets/AMASS/CMU/140/140_08_poses.npz",
        #     # "/home/leo/Datasets/AMASS/KIT/1721/RecoveryStepping_30_45_02_poses.npz",
        #     "/home/leo/Datasets/AMASS/BioMotionLab_NTroje/rub072/0014_knocking2_poses.npz",
        #
        print(f"[{self.__class__.__name__} Motion] {len(all_motion_files)} files found")
        assert len(self._all_motion_files) > 0, (
            "No motion files found. Please check the path."
            "The supported file are shown in the source code of this function."
        )

        self._init_motion_weights(yaml_data.get("motion_weights", None) if "yaml_data" in locals() else None)

    def _read_motion_file(self, motion_file_idx: int) -> MotionSequence:
        filepath = self._all_motion_files[motion_file_idx]
        if filepath.endswith("poses.npz") or filepath.endswith("stageii.npz"):
            return self._read_amass_motion_file(filepath)
        elif filepath.endswith("retargetted.npz") or filepath.endswith("retargeted.npz"):
            return self._read_retargetted_motion_file(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def _read_amass_motion_file(self, filepath: str) -> MotionSequence:
        """Read the motion file and return the retargeted data. (involving IO)"""
        raw_data = np.load(filepath, mmap_mode="r")
        try:
            framerate = raw_data["mocap_framerate"]
        except KeyError:
            framerate = raw_data["mocap_frame_rate"]
        poses = raw_data["poses"]
        poses = torch.from_numpy(poses).to(dtype=torch.float, device=self.buffer_device)
        root_trans = torch.from_numpy(raw_data["trans"]).to(dtype=torch.float, device=self.buffer_device)
        framerate = framerate.item()
        return self._build_motion_sequence_from_smpl(poses, root_trans, framerate)

    def _read_retargetted_motion_file(self, filepath: str) -> MotionSequence:
        raw_data = np.load(filepath, mmap_mode="r", allow_pickle=True)
        framerate = raw_data["framerate"].item()
        joint_names = (
            raw_data["joint_names"] if isinstance(raw_data["joint_names"], list) else raw_data["joint_names"].tolist()
        )
        joint_pos = torch.as_tensor(raw_data["joint_pos"], device=self.buffer_device, dtype=torch.float)
        root_trans = torch.as_tensor(raw_data["base_pos_w"], device=self.buffer_device, dtype=torch.float)
        root_quat = torch.as_tensor(raw_data["base_quat_w"], device=self.buffer_device, dtype=torch.float)

        # qpos_isaac = qpos[retargetted_joints_to_output_joints_ids]
        retargetted_joints_to_output_joints_ids = [joint_names.index(j_name) for j_name in self.isaac_joint_names]
        joint_pos = joint_pos[:, retargetted_joints_to_output_joints_ids]

        return self._pack_retargetted_motion_sequence(
            root_trans,
            root_quat,
            joint_pos,
            framerate,
        )

    def _build_motion_sequence_from_smpl(
        self, poses: torch.Tensor, root_trans: torch.Tensor, framerate: float
    ) -> MotionSequence:
        # skip frames to reduce the data size
        if self.cfg.skip_frames >= 1:
            nframes = poses.shape[0]
            sampling_frame_idxs = np.arange(0, nframes, int(self.cfg.skip_frames + 1)).astype(np.int32)
            framerate /= self.cfg.skip_frames + 1
            poses = poses[sampling_frame_idxs]
            root_trans = root_trans[sampling_frame_idxs]

        # retargetting to joints if needed
        assert self.retargetting_func is not None, "Retargetting function is not provided."
        # by SMPL-H definition, only previous 22 joints are used, but SMPL-X uses 24 joints
        smpl_poses = poses[:, : 24 * 3].reshape(-1, 24, 3)

        joint_pos, root_poses = self.retargetting_func(
            smpl_poses,
            root_trans,
            joint_names=self.isaac_joint_names,
            **self.cfg.retargetting_func_kwargs,
        )
        root_trans, root_quat = root_poses[..., :3], root_poses[..., 3:]  # shape [num_frames, 7]

        return self._pack_retargetted_motion_sequence(
            root_trans,
            root_quat,
            joint_pos,
            framerate,
        )

    def _pack_retargetted_motion_sequence(
        self, root_trans: torch.Tensor, root_quat: torch.Tensor, joint_pos: torch.Tensor, framerate: float | int
    ):
        """Pack the retargetted motion sequence into a MotionSequence object.
        Args:
            - root_trans: shape [num_frames, 3]
            - root_quat: shape [num_frames, 4]
            - joint_pos: shape [num_frames, num_joints]
            - framerate: the framerate of the motion sequence
        """

        if self.cfg.motion_interpolate_func:
            # Apply motion interpolation to match the target framerate
            root_trans, root_quat, joint_pos = self.cfg.motion_interpolate_func(
                root_trans, root_quat, joint_pos, framerate, self.cfg.motion_target_framerate
            )
            framerate = self.cfg.motion_target_framerate

        # estimate joint_vel, base_lin_vel_w, base_ang_vel_w
        if self.cfg.velocity_estimation_method is not None:
            joint_vel = estimate_velocity(
                joint_pos.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
            base_lin_vel_w = estimate_velocity(
                root_trans.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
            base_ang_vel_w = estimate_angular_velocity(
                root_quat.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
        else:
            joint_vel = torch.zeros_like(joint_pos)
            base_lin_vel_w = torch.zeros_like(root_trans)
            base_ang_vel_w = torch.zeros_like(root_trans)

        # TODO/NOTE: May lead to bugs if the links of interest is empty.
        link_pos_quat_b = self.forward_kinematics_func(joint_pos)  # [num_frames, num_links, 7]
        link_pos_b = link_pos_quat_b[..., :3]
        link_quat_b = link_pos_quat_b[..., 3:]
        link_pos_w, link_quat_w = math_utils.combine_frame_transforms(
            root_trans.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            root_quat.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            link_pos_b,
            link_quat_b,
        )

        if self.cfg.velocity_estimation_method is not None:
            # estimate link linear velocity in world frame
            link_lin_vel_b = estimate_velocity(
                link_pos_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_ang_vel_b = estimate_angular_velocity(
                link_quat_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            # estimate the link linear velocity to world frame
            link_lin_vel_w = estimate_velocity(
                link_pos_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_ang_vel_w = estimate_angular_velocity(
                link_quat_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
        else:
            link_lin_vel_b = torch.zeros_like(link_pos_b)
            link_ang_vel_b = torch.zeros_like(link_quat_b)
            link_lin_vel_w = torch.zeros_like(link_pos_w)
            link_ang_vel_w = torch.zeros_like(link_quat_w)

        return MotionSequence(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_pos_w=root_trans,
            base_lin_vel_w=base_lin_vel_w,
            base_quat_w=root_quat,
            base_ang_vel_w=base_ang_vel_w,
            link_pos_b=link_pos_b,
            link_quat_b=link_quat_b,
            link_lin_vel_b=link_lin_vel_b,
            link_ang_vel_b=link_ang_vel_b,
            link_lin_vel_w=link_lin_vel_w,
            link_ang_vel_w=link_ang_vel_w,
            link_pos_w=link_pos_w,
            link_quat_w=link_quat_w,
            framerate=torch.as_tensor(framerate),
            buffer_length=torch.as_tensor(joint_pos.shape[0]),
        )

    def _prepare_retargetting_func(self):
        # print(f"[AMASS Motion] Retargetting to {self.isaac_joint_names}")
        if inspect.isclass(self.cfg.retargetting_func):
            self.retargetting_func = self.cfg.retargetting_func(
                joint_names=self.isaac_joint_names,
                device=self.buffer_device,
                **self.cfg.retargetting_func_kwargs,
            )
        else:
            self.retargetting_func = self.cfg.retargetting_func

    def _load_motion_sequences(self):
        """Load all motion sequence files into memory in a retargetted form."""
        # load the motions for the first time all together
        # TODO: use multiprocessing to load the motion files
        print(f"[AMASS Motion] Loading motion files, should be {len(self._all_motion_files)} in total...")
        all_motion_sequences = list(map(self._read_motion_file, range(len(self._all_motion_files))))
        print(f"[AMASS Motion] All {len(all_motion_sequences)} motion files loaded.")

        # fill the motion sequences into a single pytorch tensor buffer, to accelerate the access
        # max_buffer_length = max([int(motion.buffer_length) for motion in all_motion_sequences])
        # self._all_motion_sequences = MotionSequence.make_empty(
        #     len(all_motion_sequences),
        #     max_buffer_length,
        #     num_joints=self.articulation_view.max_dofs,
        #     num_links=self.num_link_to_ref,
        #     device=self.buffer_device,
        # )
        print(
            "[AMASS Motion] buffer lengths statistics:"
            f" mean: {np.array([motion.buffer_length for motion in all_motion_sequences]).mean()},"
            f" max: {np.array([motion.buffer_length for motion in all_motion_sequences]).max()},"
            f" min: {np.array([motion.buffer_length for motion in all_motion_sequences]).min()},"
        )
        self._all_motion_sequences = MotionSequence.make_emtpy_concat_batch(
            buffer_lengths=[int(motion.buffer_length) for motion in all_motion_sequences],
            num_joints=self.articulation_view.max_dofs,
            num_links=self.num_link_to_ref,
            device=self.buffer_device,
        )
        for i, motion in enumerate(all_motion_sequences):
            for attr in self._all_motion_sequences.attrs_with_frame_dim:
                getattr(self._all_motion_sequences, attr)[i, : motion.buffer_length] = getattr(motion, attr)
            for attr in self._all_motion_sequences.attrs_only_batch_dim:
                getattr(self._all_motion_sequences, attr)[i] = getattr(motion, attr)

    """
    Internal helper functions (for sampling motion files).
    """

    def _sample_assigned_env_starting_stub(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Sample motion file id and start time for the assigned envs.
        By stub, it means the necessary information to sample the motion data for the assigned envs.
        """

        # build necessary stubs if not built.
        if not hasattr(self, "_assigned_env_motion_selection"):
            self._assigned_env_motion_selection = torch.zeros(
                self.num_assigned_envs,
                dtype=torch.long,
                device=self.buffer_device,
            )
        if not hasattr(self, "_motion_buffer_start_time_s"):
            self._motion_buffer_start_time_s = torch.zeros(
                self.num_assigned_envs,
                dtype=torch.float,
                device=self.buffer_device,
            )

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        if self.cfg.env_starting_stub_sampling_strategy == "independent":
            self._sample_assigned_env_motion_selection(assigned_ids)
            self._sample_env_motion_start_time(assigned_ids)
        elif self.cfg.env_starting_stub_sampling_strategy == "concat_motion_bins":
            self._sample_as_concat_motion_bins(assigned_ids)
        else:
            raise ValueError(
                f"Unsupported env starting stub sampling method: {self.cfg.env_starting_stub_sampling_strategy}"
            )

    def _sample_assigned_env_motion_selection(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Resample the motion selection for the buffer, build the motion selection if not built.
        This samples the env_motion_selection independent from the start time.
        """
        if len(assigned_ids) == 0:
            return
        self._assigned_env_motion_selection[assigned_ids] = torch.multinomial(
            self._motion_weights,
            len(assigned_ids),
            replacement=True,
        ).to(self.buffer_device)

    def _sample_env_motion_start_time(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Sample the start time for the assigned envs. (at reset)
        This samples the start time for the assigned envs, which is independent from the motion selection.
        """
        assert assigned_ids is not None  # type: ignore

        if self.cfg.motion_start_from_middle_range[1] > 0.0:
            assert (
                self.cfg.motion_start_from_middle_range[0] >= 0
                and self.cfg.motion_start_from_middle_range[1] >= self.cfg.motion_start_from_middle_range[0]
            ), (
                "motion_start_from_middle_range should be non-negative and the second value should be larger than the"
                " first one."
            )
            random_start_time = (
                torch.rand(
                    len(assigned_ids),
                    device=self.buffer_device,
                )
                * (self.cfg.motion_start_from_middle_range[1] - self.cfg.motion_start_from_middle_range[0])
                + self.cfg.motion_start_from_middle_range[0]
            )
            random_start_time *= self._all_motion_sequences.buffer_length[
                self._assigned_env_motion_selection[assigned_ids]
            ].to(torch.float)
            random_start_time /= self._all_motion_sequences.framerate[
                self._assigned_env_motion_selection[assigned_ids]
            ].to(torch.float)
            self._motion_buffer_start_time_s[assigned_ids] = random_start_time

        if hasattr(self, "_motion_bin_weights"):
            motion_ids = self._assigned_env_motion_selection[assigned_ids]
            self._motion_buffer_start_time_s[assigned_ids] = sample_start_times(
                motion_ids, self._motion_bin_weights, self.buffer_device, self.cfg.motion_bin_length_s
            )

    def _sample_as_concat_motion_bins(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Sample the motion file id and start time for the assigned envs as if all motions are concatenated into a single motion."""
        flattened_bin_ids = torch.multinomial(
            self._motion_bin_weights._concatenated_tensor,
            len(assigned_ids),
            replacement=True,
        ).to(self.buffer_device)
        # identify the motion_id and the actual bin id
        motion_ids, bin_ids = self._motion_bin_weights.unwarp_flattened_idx(flattened_bin_ids)
        self._assigned_env_motion_selection[assigned_ids] = motion_ids
        self._motion_buffer_start_time_s[assigned_ids] = (
            bin_ids * self.cfg.motion_bin_length_s
            + torch.rand(len(assigned_ids), device=self.buffer_device) * self.cfg.motion_bin_length_s
        )

    """
    Internal helper functions (Others).
    """

    def _get_motion_based_origin(self, env_origins: torch.Tensor, env_ids: Sequence[int] | torch.Tensor):
        """Update the base position with origin, or use the env_origins from caller directly."""
        return env_origins[env_ids]
