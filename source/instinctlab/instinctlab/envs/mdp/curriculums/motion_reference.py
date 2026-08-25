from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import warp as wp

from isaaclab.managers.manager_base import ManagerTermBase

from instinctlab.utils.torch import ConcatBatchTensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg
    from isaaclab.motion_reference.motion_files.amass_motion import AmassMotion

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


class BeyondMimicAdaptiveWeighting(ManagerTermBase):
    """The adaptive weighting strategy from BeyondMimic source code.
    The replicated implementation might be in-accurate due to the code structure difference.
    ## References
    - https://github.com/HybridRobotics/whole_body_tracking/blob/dcecabd8c24c68f59d143fdf8e3a670f420c972d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L207
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # A note of needed parameters
        self.adaptive_uniform_ratio = cfg.params.get("adaptive_uniform_ratio", 0.1)
        self.adaptive_kernel_size = cfg.params.get("adaptive_kernel_size", 3)
        self.adaptive_alpha = cfg.params.get("adaptive_alpha", 0.001)
        self.adaptive_lambda = cfg.params.get("adaptive_lambda", 0.8)

        self.motion_reference: MotionReferenceManagerBase = env.scene[
            cfg.params.get("reference_name", "motion_reference")
        ]

        # NOTE: For simplicity, only support one motion buffer for now.
        assert len(self.motion_reference.motion_buffers.keys()) == 1, "Only support one motion buffer for now."
        self.motion_buffer_name = list(self.motion_reference.motion_buffers.keys())[0]
        self.motion_buffer: AmassMotion = self.motion_reference.motion_buffers[self.motion_buffer_name]  # type: ignore
        self.__motion_bin_length_s = self.motion_buffer.cfg.motion_bin_length_s

        self.motion_bin_nums = self.motion_buffer._motion_bin_weights._batch_sizes
        self.motion_bin_fail_counter = ConcatBatchTensor(
            batch_sizes=self.motion_bin_nums,  # type: ignore
            data_shape=tuple(),
            dtype=torch.float,
            device=self.motion_buffer.buffer_device,
        )
        self.motion_bin_fail_counter.fill_data(torch.as_tensor([0.0], device=self.motion_buffer.buffer_device)[0])
        self.current_motion_bin_fail_counter = ConcatBatchTensor(
            batch_sizes=self.motion_bin_nums,  # type: ignore
            data_shape=tuple(),
            dtype=torch.float,
            device=self.motion_buffer.buffer_device,
        )
        self.current_motion_bin_fail_counter.fill_data(
            torch.as_tensor([0.0], device=self.motion_buffer.buffer_device)[0]
        )

        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.motion_buffer.buffer_device,
        )
        self.kernel = self.kernel / self.kernel.sum()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        adaptive_uniform_ratio: float = 0.1,
        adaptive_kernel_size: int = 3,
        adaptive_alpha: float = 0.001,
        adaptive_lambda: float = 0.8,
    ) -> dict[str, float] | None:
        if len(env_ids) == 0:
            raise ValueError("No env_ids")

        self._update_current_motion_bin_fail_counter(env_ids)

        # Compute the new motion bin probability
        new_prob_concat = self._compute_smoothed_probs()

        # Update the motion bin probability
        num_motions = self.motion_bin_nums.shape[0]
        motion_sums = torch.zeros(num_motions, dtype=torch.float32, device=self.device)
        motion_id_per_bin = torch.repeat_interleave(
            torch.arange(num_motions, device=self.device, dtype=torch.int32), self.motion_bin_nums
        )
        wp.launch(
            kernel=compute_sums,
            dim=self.motion_bin_fail_counter.shape[0],
            inputs=[
                wp.from_torch(motion_id_per_bin, dtype=wp.int32),  # type: ignore
                wp.from_torch(new_prob_concat, dtype=wp.float32),  # type: ignore
                wp.from_torch(motion_sums, dtype=wp.float32),  # type: ignore
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)

        repeat_sums = torch.repeat_interleave(motion_sums, self.motion_bin_nums)
        new_prob_concat /= repeat_sums.clamp(min=1e-10)  # Avoid division by zero

        self.motion_buffer._motion_bin_weights._concatenated_tensor.copy_(new_prob_concat)

        # Metrics to log
        sampling_entropy = []
        sampling_top1_probs = []
        sampling_top1_bins = []
        for motion_idx in range(min(self.motion_bin_nums.shape[0], 4)):
            w = self.motion_buffer._motion_bin_weights[motion_idx]
            sampling_entropy.append(
                -torch.sum(w * torch.log(w + 1e-12)).item() / self.motion_bin_nums[motion_idx].log().item()
            )
            sampling_top1_probs.append(w.max().item())
            sampling_top1_bins.append(w.argmax().item() / self.motion_bin_nums[motion_idx].item())

        return_ = {}
        for i, motion_idx in enumerate(range(min(self.motion_bin_nums.shape[0], 4))):
            # Log the first 4 motion buffer stats in case of too many motions
            return_[f"motion_{i}_sampling_entropy"] = sampling_entropy[i]
            return_[f"motion_{i}_sampling_top1_prob"] = sampling_top1_probs[i]
            return_[f"motion_{i}_sampling_top1_bin"] = sampling_top1_bins[i]
        return return_

    def _update_current_motion_bin_fail_counter(self, env_ids):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if episode_failed.any():
            # NOTE: only one motion buffer, so for each motion_buffer, env_ids === assigned_ids
            current_bin_idx = torch.clamp(
                (
                    self._env.episode_length_buf[env_ids] * self._env.step_dt
                    + self.motion_buffer._motion_buffer_start_time_s[env_ids]
                )
                / self.__motion_bin_length_s,
                max=self.motion_bin_nums[self.motion_buffer._assigned_env_motion_selection[env_ids]] - 1,
            ).to(torch.long)
            failed_bin_idx = current_bin_idx[episode_failed].to(torch.long)
            failed_motion_ids = self.motion_buffer._assigned_env_motion_selection[env_ids][episode_failed]
        else:
            failed_bin_idx = torch.tensor([], device=self.motion_buffer.buffer_device, dtype=torch.int)
            failed_motion_ids = torch.tensor([], device=self.motion_buffer.buffer_device, dtype=torch.int)

        self.current_motion_bin_fail_counter._concatenated_tensor.fill_(0)

        # Update the current motion bin fail counter
        if episode_failed.any():
            wp.launch(
                kernel=update_current_fail_counters,
                dim=failed_bin_idx.shape[0],
                inputs=[
                    wp.from_torch(failed_motion_ids.to(torch.int32), dtype=wp.int32),  # type: ignore
                    wp.from_torch(failed_bin_idx.to(torch.int32), dtype=wp.int32),  # type: ignore
                    wp.from_torch(self.current_motion_bin_fail_counter._batch_starts.to(torch.int32), dtype=wp.int32),  # type: ignore
                    wp.from_torch(self.current_motion_bin_fail_counter._concatenated_tensor, dtype=wp.float32),  # type: ignore
                ],
                device=self.device,
            )

    def _compute_smoothed_probs(self):
        num_motions = self.motion_bin_nums.shape[0]
        total_bins = self.motion_bin_fail_counter.shape[0]

        uniforms = self.adaptive_uniform_ratio / self.motion_bin_nums.float()
        uniform_per_bin = torch.repeat_interleave(uniforms, self.motion_bin_nums)
        probability_concat = self.motion_bin_fail_counter._concatenated_tensor + uniform_per_bin

        new_prob_concat = torch.empty_like(probability_concat)

        motion_id_per_bin = torch.repeat_interleave(
            torch.arange(num_motions, device=self.device, dtype=torch.int32), self.motion_bin_nums
        )

        wp.launch(
            kernel=compute_smoothed_probs,
            dim=total_bins,
            inputs=[
                wp.from_torch(self.motion_bin_fail_counter._batch_starts.to(torch.int32), dtype=wp.int32),  # type: ignore
                wp.from_torch(self.motion_bin_nums.to(torch.int32), dtype=wp.int32),  # type: ignore
                wp.from_torch(motion_id_per_bin, dtype=wp.int32),  # type: ignore
                wp.from_torch(self.kernel, dtype=wp.float32),  # type: ignore
                wp.from_torch(probability_concat, dtype=wp.float32),  # type: ignore
                wp.from_torch(new_prob_concat, dtype=wp.float32),  # type: ignore
                self.adaptive_kernel_size,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)

        return new_prob_concat


class BeyondConcatMotionAdaptiveWeighting(BeyondMimicAdaptiveWeighting):
    """The adaptive weighting strategy inspired by BeyondMimic. The difference is that, this curriculum compute the
    motion_bin_weights as if all bins are from a single motion files.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        adaptive_uniform_ratio: float = 0.1,
        adaptive_kernel_size: int = 3,
        adaptive_alpha: float = 0.001,
        adaptive_lambda: float = 0.8,
    ) -> dict[str, float] | None:
        self._update_current_motion_bin_fail_counter(env_ids)

        # Compute the new motion bin probability
        new_prob_concat = self._compute_smoothed_probs()

        # normalize the probability directly as if they are from a single motion file
        new_prob_concat /= new_prob_concat.sum()
        self.motion_buffer._motion_bin_weights._concatenated_tensor.copy_(new_prob_concat)

        # compute the sampling stats, as if they are from a single motion file
        w = self.motion_buffer._motion_bin_weights._concatenated_tensor
        N = self.motion_bin_nums.sum()
        assert N == w.shape[0], "N should be the same as the number of bins"
        sampling_entropy = -torch.sum(w * torch.log(w + 1e-12)).item() / N.log().item()
        sampling_top1_probs = w.max().item()
        sampling_top1_bins = w.argmax().item() / N.item()
        return {
            "sampling_entropy": sampling_entropy,
            "sampling_top1_prob": sampling_top1_probs,
            "sampling_top1_bin": sampling_top1_bins,
        }


@wp.kernel
def update_current_fail_counters(
    failed_motion_ids: wp.array(dtype=wp.int32),  # type: ignore
    failed_bin_idx: wp.array(dtype=wp.int32),  # type: ignore
    batch_starts: wp.array(dtype=wp.int32),  # type: ignore
    current_fail_counter: wp.array(dtype=wp.float32),  # type: ignore
):
    tid = wp.tid()
    motion_id = failed_motion_ids[tid]
    bin_idx = failed_bin_idx[tid]
    pos = batch_starts[motion_id] + bin_idx
    wp.atomic_add(current_fail_counter, pos, 1.0)


@wp.kernel
def compute_smoothed_probs(
    batch_starts: wp.array(dtype=wp.int32),  # type: ignore
    batch_sizes: wp.array(dtype=wp.int32),  # type: ignore
    motion_id_per_bin: wp.array(dtype=wp.int32),  # type: ignore
    kernel: wp.array(dtype=wp.float32),  # type: ignore
    prob: wp.array(dtype=wp.float32),  # type: ignore
    output: wp.array(dtype=wp.float32),  # type: ignore
    kernel_size: int,
):
    tid = wp.tid()
    motion = motion_id_per_bin[tid]
    num_bins = batch_sizes[motion]
    local_j = tid - batch_starts[motion]
    if local_j >= num_bins:
        return
    smoothed = float(0.0)
    for k in range(kernel_size):
        local_idx = wp.min(local_j + k, num_bins - 1)
        prob_val = prob[batch_starts[motion] + local_idx]
        smoothed += kernel[k] * prob_val
    output[tid] = smoothed


@wp.kernel
def compute_sums(
    motion_id_per_bin: wp.array(dtype=wp.int32),  # type: ignore
    values: wp.array(dtype=wp.float32),  # type: ignore
    sums: wp.array(dtype=wp.float32),  # type: ignore
):
    tid = wp.tid()
    motion = motion_id_per_bin[tid]
    wp.atomic_add(sums, motion, values[tid])
