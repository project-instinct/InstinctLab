from __future__ import annotations

import numpy as np
import os
import torch
from typing import TYPE_CHECKING, Sequence, cast

import isaaclab.utils.math as math_utils

from instinctlab.motion_reference import HoiMotionReferenceData, HoiMotionReferenceState, HoiMotionSequence
from instinctlab.motion_reference.utils import estimate_angular_velocity, estimate_velocity, pose_interpolate_bilinear
from instinctlab.utils.math import quat_slerp_batch

from .amass_motion import AmassMotion

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .omomo_motion_cfg import OmomoMotionCfg


class OmomoMotion(AmassMotion):
    """Processing OMOMO formatted file structure with human-object interaction support."""

    cfg: OmomoMotionCfg

    def _resolve_object_indices(self, scene_object_names: list[str], motion_ids: torch.Tensor) -> torch.Tensor:
        """Map each motion file's object to its slot index in the scene buffer.

        Returns:
            Tensor of shape ``[len(motion_ids)]`` with the scene slot index for
            each motion, or ``-1`` when the object is absent from *scene_object_names*.
        """
        name_to_idx = {name: i for i, name in enumerate(scene_object_names)}
        return torch.tensor(
            [name_to_idx.get(self._motion_object_names_list[mid], -1) for mid in motion_ids.tolist()],
            dtype=torch.long,
            device=motion_ids.device,
        )

    def _read_retargetted_motion_file(self, filepath: str) -> HoiMotionSequence:
        raw_data = np.load(filepath, mmap_mode="r", allow_pickle=True)
        framerate = raw_data["framerate"].item()
        joint_names = (
            raw_data["joint_names"] if isinstance(raw_data["joint_names"], list) else raw_data["joint_names"].tolist()
        )
        joint_pos = torch.as_tensor(raw_data["joint_pos"], device=self.buffer_device, dtype=torch.float)
        root_trans = torch.as_tensor(raw_data["base_pos_w"], device=self.buffer_device, dtype=torch.float)
        root_quat_wxyz = torch.as_tensor(raw_data["base_quat_w"], device=self.buffer_device, dtype=torch.float)
        root_quat = math_utils.convert_quat(root_quat_wxyz, to="xyzw")

        # Retargeting joints
        retargetted_joints_to_output_joints_ids = [joint_names.index(j_name) for j_name in self.isaac_joint_names]
        joint_pos = joint_pos[:, retargetted_joints_to_output_joints_ids]

        # --- Object Data Processing ---
        # Retargeted NPZ files store object_quat_w as WXYZ. Convert at this file boundary.
        # Extract object name from filename: subXX_OBJECTNAME_XXX_retargeted.npz
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        object_name = parts[1]
        scene_object_names = [object_name]

        # (T, 3) -> (T, 1, 3)
        object_pos_w = torch.as_tensor(
            raw_data["object_pos_w"], device=self.buffer_device, dtype=torch.float
        ).unsqueeze(1)
        # (T, 4) -> (T, 1, 4)
        object_quat_wxyz = torch.as_tensor(raw_data["object_quat_w"], device=self.buffer_device, dtype=torch.float)
        object_quat_w = math_utils.convert_quat(object_quat_wxyz, to="xyzw").unsqueeze(1)

        object_validity = torch.ones((object_pos_w.shape[0], 1), device=self.buffer_device, dtype=torch.bool)

        # ------------------------------

        return self._pack_retargetted_motion_sequence(
            root_trans,
            root_quat,
            joint_pos,
            framerate,
            object_pos_w,
            object_quat_w,
            object_validity,
            scene_object_names,
        )

    def _pack_retargetted_motion_sequence(
        self,
        root_trans: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        framerate: float | int,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        object_validity: torch.Tensor,
        scene_object_names: list[str],
    ) -> HoiMotionSequence:

        if self.cfg.motion_interpolate_func:
            # Note: We might need to interpolate object data too.
            # For now, assuming standard interpolation for robot, and we'll do simple interpolation for object if needed.
            # But AmassMotionCfg.motion_interpolate_func signature is fixed.
            # If we strictly need interpolation, we should update the cfg or handle it here manually for objects.
            # Calling the super/cfg function for robot parts:
            root_trans, root_quat, joint_pos = self.cfg.motion_interpolate_func(
                root_trans, root_quat, joint_pos, framerate, self.cfg.motion_target_framerate
            )

            # Interpolate object data
            if framerate != self.cfg.motion_target_framerate:
                interpolate_func = self.cfg.object_interpolate_func
                if interpolate_func is None:
                    interpolate_func = pose_interpolate_bilinear

                object_pos_w, object_quat_w, object_validity = interpolate_func(
                    object_pos_w,
                    object_quat_w,
                    object_validity,
                    framerate,
                    self.cfg.motion_target_framerate,
                )

            framerate = self.cfg.motion_target_framerate

        # Robot velocities
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

        # Object velocities
        if self.cfg.velocity_estimation_method is not None:
            # object_pos_w is [T, N, 3]
            object_lin_vel_w = estimate_velocity(
                object_pos_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            object_ang_vel_w = estimate_angular_velocity(
                object_quat_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
        else:
            object_lin_vel_w = torch.zeros_like(object_pos_w)
            object_ang_vel_w = torch.zeros_like(object_quat_w)[..., :3]

        # Link poses (Robot)
        link_pos_quat_b = self.forward_kinematics_func(joint_pos)
        link_pos_b = link_pos_quat_b[..., :3]
        link_quat_b = link_pos_quat_b[..., 3:]

        link_pos_w, link_quat_w = math_utils.combine_frame_transforms(
            root_trans.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            root_quat.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            link_pos_b,
            link_quat_b,
        )

        if self.cfg.velocity_estimation_method is not None:
            link_lin_vel_b = estimate_velocity(
                link_pos_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_ang_vel_b = estimate_angular_velocity(
                link_quat_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_lin_vel_w = estimate_velocity(
                link_pos_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            link_ang_vel_w = estimate_angular_velocity(
                link_quat_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
        else:
            link_lin_vel_b = torch.zeros_like(link_pos_b)
            link_ang_vel_b = torch.zeros_like(link_quat_b)[..., :3]
            link_lin_vel_w = torch.zeros_like(link_pos_w)
            link_ang_vel_w = torch.zeros_like(link_quat_w)[..., :3]

        return HoiMotionSequence(
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
            object_pos_w=object_pos_w,
            object_quat_w=object_quat_w,
            object_lin_vel_w=object_lin_vel_w,
            object_ang_vel_w=object_ang_vel_w,
            object_validity=object_validity,
            scene_object_names=scene_object_names,
        )

    def _load_motion_sequences(self):
        """Load all motion sequence files into memory in a retargetted form."""
        print(f"[OMOMO Motion] Loading motion files, should be {len(self._all_motion_files)} in total...")
        all_motion_sequences = list(map(self._read_motion_file, range(len(self._all_motion_files))))
        for m in all_motion_sequences:
            assert isinstance(m, HoiMotionSequence), "OmomoMotion yields HoiMotionSequence"
        all_motion_sequences = cast(list[HoiMotionSequence], all_motion_sequences)
        print(f"[OMOMO Motion] All {len(all_motion_sequences)} motion files loaded.")

        print(
            "[OMOMO Motion] buffer lengths statistics:"
            f" mean: {np.array([motion.buffer_length for motion in all_motion_sequences]).mean()},"
            f" max: {np.array([motion.buffer_length for motion in all_motion_sequences]).max()},"
            f" min: {np.array([motion.buffer_length for motion in all_motion_sequences]).min()},"
        )

        # Determine max number of objects
        max_num_objects = 0
        if len(all_motion_sequences) > 0:
            # For OMOMO dataset, it is typically 1 for each trajectory file.
            max_num_objects = max([motion.object_pos_w.shape[1] for motion in all_motion_sequences])

        self._all_motion_sequences = HoiMotionSequence.make_emtpy_concat_batch(
            buffer_lengths=[int(motion.buffer_length) for motion in all_motion_sequences],
            num_joints=self.articulation_access.max_dofs,
            num_links=self.num_link_to_ref,
            num_objects=max_num_objects,
            device=self.buffer_device,
        )

        self._motion_object_names_list = [motion.scene_object_names[0] for motion in all_motion_sequences]

        for i, motion in enumerate(all_motion_sequences):
            for attr in self._all_motion_sequences.attrs_with_frame_dim:
                # Handle object padding if necessary
                data = getattr(motion, attr)
                if "object" in attr:
                    # motion data might have fewer objects than max_num_objects
                    current_objects = data.shape[1]
                    if current_objects < max_num_objects:
                        # Pad with zeros or some default
                        if attr == "object_validity":
                            padding = torch.zeros(
                                (data.shape[0], max_num_objects - current_objects), device=data.device, dtype=torch.bool
                            )
                        else:
                            padding = torch.zeros(
                                (data.shape[0], max_num_objects - current_objects, *data.shape[2:]),
                                device=data.device,
                                dtype=data.dtype,
                            )
                            if "quat" in attr:
                                padding[..., 3] = 1.0
                        data = torch.cat([data, padding], dim=1)

                getattr(self._all_motion_sequences, attr)[i, : motion.buffer_length] = data

            for attr in self._all_motion_sequences.attrs_only_batch_dim:
                getattr(self._all_motion_sequences, attr)[i] = getattr(motion, attr)

    def fill_init_reference_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
        state_buffer: HoiMotionReferenceState,
    ) -> None:
        super().fill_init_reference_state(env_ids, env_origins, state_buffer)

        if not all(
            hasattr(state_buffer, attr)
            for attr in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "object_validity")
        ):
            return

        assert self.env_ids_is_assigned(env_ids).all(), "The env_ids should be assigned to this motion buffer."
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        frame_selection = torch.floor(
            self._motion_buffer_start_time_s[assigned_ids] * self._all_motion_sequences.framerate[motion_ids]
        ).to(torch.long)
        max_frames = self._all_motion_sequences.buffer_length[motion_ids] - 1
        frame_selection = torch.clamp(frame_selection, torch.zeros_like(max_frames), max_frames)

        target_obj_idxs = self._resolve_object_indices(state_buffer.scene_object_names, motion_ids)

        # Fetch from source object slot 0 (each OMOMO file has exactly one object)
        # ConcatBatchTensor only accepts (batch_idx, data_idx); object index 0 applied after
        object_pos_w = self._all_motion_sequences.object_pos_w[motion_ids, frame_selection][..., 0, :]
        object_quat_w = self._all_motion_sequences.object_quat_w[motion_ids, frame_selection][..., 0, :]
        object_lin_vel_w = self._all_motion_sequences.object_lin_vel_w[motion_ids, frame_selection][..., 0, :]
        object_ang_vel_w = self._all_motion_sequences.object_ang_vel_w[motion_ids, frame_selection][..., 0, :]
        object_validity = self._all_motion_sequences.object_validity[motion_ids, frame_selection][..., 0]

        object_pos_w += self._get_motion_based_origin(env_origins, env_ids)

        # Reset all object slots for these envs
        state_buffer.object_pos_w[env_ids] = 0.0
        state_buffer.object_quat_w[env_ids] = 0.0
        state_buffer.object_quat_w[env_ids, :, 3] = 1.0
        state_buffer.object_lin_vel_w[env_ids] = 0.0
        state_buffer.object_ang_vel_w[env_ids] = 0.0
        state_buffer.object_validity[env_ids] = False

        # Scatter into the correct object slot per env
        env_ids_dev = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.output_device)
        target_obj_idxs_dev = target_obj_idxs.to(self.output_device)

        valid_mask = target_obj_idxs_dev != -1
        if not valid_mask.any():
            return

        v_env = env_ids_dev[valid_mask]
        v_obj = target_obj_idxs_dev[valid_mask]
        v_src = valid_mask.to(object_pos_w.device)

        state_buffer.object_pos_w[v_env, v_obj] = object_pos_w[v_src].to(self.output_device)
        state_buffer.object_quat_w[v_env, v_obj] = object_quat_w[v_src].to(self.output_device)
        state_buffer.object_lin_vel_w[v_env, v_obj] = object_lin_vel_w[v_src].to(self.output_device)
        state_buffer.object_ang_vel_w[v_env, v_obj] = object_ang_vel_w[v_src].to(self.output_device)
        state_buffer.object_validity[v_env, v_obj] = object_validity[v_src].to(self.output_device)

    def fill_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: HoiMotionReferenceData,
    ) -> None:
        super().fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)

        if not all(
            hasattr(data_buffer, attr)
            for attr in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "object_validity")
        ):
            return

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        motion_ids = self._assigned_env_motion_selection[assigned_ids]

        frame_selections = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp.to(self.buffer_device))
            * self._all_motion_sequences.framerate[motion_ids].unsqueeze(-1)
        )
        frame_selections = torch.where(
            data_buffer.validity[env_ids].to(self.buffer_device),
            frame_selections,
            self._all_motion_sequences.buffer_length[motion_ids].unsqueeze(-1) - 1,
        ).to(torch.long)

        num_frames = frame_selections.shape[1]
        assigned_ids_across_frame = assigned_ids.unsqueeze(-1).expand(-1, num_frames).flatten()
        frame_selections_flat = frame_selections.flatten()

        if data_buffer.object_validity.shape[2] <= 0:
            return

        target_obj_idxs = self._resolve_object_indices(data_buffer.scene_object_names, motion_ids)

        env_ids_dev = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.output_device)
        target_obj_idxs_dev = target_obj_idxs.to(self.output_device)

        valid_mask = target_obj_idxs_dev != -1
        if not valid_mask.any():
            return

        v_env = env_ids_dev[valid_mask]
        v_obj = target_obj_idxs_dev[valid_mask]

        # Invalidate object slots that this motion buffer does not provide data for
        num_objects = data_buffer.object_validity.shape[2]
        for obj_idx in range(num_objects):
            not_target = v_obj != obj_idx
            if not_target.any():
                data_buffer.object_validity[v_env[not_target], :, obj_idx] = False

        # Filter flattened indices to valid envs only
        valid_expanded = valid_mask.unsqueeze(-1).expand(-1, num_frames).flatten().to(assigned_ids_across_frame.device)
        assigned_ids_across_frame = assigned_ids_across_frame[valid_expanded]
        frame_selections_flat = frame_selections_flat[valid_expanded]

        object_attrs = ["object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"]
        for attr in object_attrs:
            # ConcatBatchTensor only accepts (batch_idx, data_idx); object index 0 applied after
            source_data = getattr(self._all_motion_sequences, attr)[
                self._assigned_env_motion_selection[assigned_ids_across_frame],
                frame_selections_flat,
            ][..., 0, :].reshape(len(v_env), num_frames, *getattr(data_buffer, attr).shape[3:])

            if attr == "object_pos_w":
                source_data += self._get_motion_based_origin(env_origins, v_env).unsqueeze(1).to(self.buffer_device)

            getattr(data_buffer, attr)[v_env, :, v_obj] = source_data.to(self.output_device)

        # ConcatBatchTensor only accepts (batch_idx, data_idx); object index 0 applied after
        source_validity = self._all_motion_sequences.object_validity[
            self._assigned_env_motion_selection[assigned_ids_across_frame],
            frame_selections_flat,
        ][..., 0].reshape(len(v_env), num_frames)

        data_buffer.object_validity[v_env, :, v_obj] = source_validity.to(self.output_device)
        data_buffer.object_validity[v_env, :, v_obj] &= data_buffer.validity[v_env]
