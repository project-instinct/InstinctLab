from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.managers.manager_base import ManagerTermBase

from instinctlab.utils.torch import ConcatBatchTensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg

    from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
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
        env_ids: torch.Tensor | slice,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_kernel_size: int = 3,
        adaptive_alpha: float = 0.001,
        adaptive_lambda: float = 0.8,
    ) -> dict[str, float] | None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
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
        env_ids: torch.Tensor | slice,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_kernel_size: int = 3,
        adaptive_alpha: float = 0.001,
        adaptive_lambda: float = 0.8,
    ) -> dict[str, float] | None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
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


class CurriculumStateInitialization(ManagerTermBase):
    """Curriculum State Initialization (CSI) of ACP (He et al., arXiv:2610.31608, stage 1).

    Each clip keeps an initialization phase (fraction of its duration), starting at ``initial_phase``. After an
    episode on the clip the phase decreases by ``phase_step`` if the episode reached ``completion_phase`` and increases
    by ``phase_step`` otherwise, clipped to ``phase_range``. Episodes that end in the same step update their clip by
    the sum of their steps, clipped once.

    The phase reaches the motion buffer through its start-time sampling: every bin weight of a clip is zero except the
    bin at the phase. Use a bin of one frame (``motion_bin_length_s`` = the control step) for an exact phase, and the
    ``"independent"`` start sampling so clips keep their motion weights.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.motion_reference: MotionReferenceManagerBase = env.scene[
            cfg.params.get("reference_name", "motion_reference")
        ]
        if len(self.motion_reference.motion_buffers) != 1:
            raise ValueError("CurriculumStateInitialization supports one motion buffer.")
        self.motion_buffer: AmassMotion = next(iter(self.motion_reference.motion_buffers.values()))  # type: ignore
        buffer_cfg = self.motion_buffer.cfg
        if buffer_cfg.motion_bin_length_s is None or buffer_cfg.env_starting_stub_sampling_strategy != "independent":
            raise ValueError(
                "CurriculumStateInitialization needs motion_bin_length_s set and the 'independent' start sampling."
            )
        self.bin_length_s = buffer_cfg.motion_bin_length_s
        self.num_bins = self.motion_buffer._motion_bin_weights._batch_sizes
        self.bin_starts = self.motion_buffer._motion_bin_weights._batch_starts
        device = self.motion_buffer.buffer_device
        self.phase = torch.full(
            (self.motion_buffer.num_trajectories,), cfg.params.get("initial_phase", 0.9), device=device
        )
        self._write_bin_weights()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        reference_name: str = "motion_reference",
        initial_phase: float = 0.9,
        completion_phase: float = 0.99,
        phase_step: float = 0.05,
        phase_range: tuple[float, float] = (0.0, 0.99),
    ) -> dict[str, float]:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        buffer = self.motion_buffer
        # environments reset before any step (e.g. at startup) did not play an episode
        env_ids = env_ids[env.episode_length_buf[env_ids] > 0]
        if len(env_ids) > 0:
            # NOTE: one motion buffer, so assigned ids are env ids
            ids = env_ids.to(buffer.buffer_device)
            motion_ids = buffer._assigned_env_motion_selection[ids]
            end_time_s = (
                buffer._motion_buffer_start_time_s[ids] + env.episode_length_buf[env_ids].to(ids.device) * env.step_dt
            )
            clip_length_s = (
                buffer._all_motion_sequences.buffer_length[motion_ids]
                / buffer._all_motion_sequences.framerate[motion_ids]
            )
            completed = end_time_s / clip_length_s >= completion_phase
            step = torch.where(completed, -phase_step, phase_step)
            delta = torch.zeros_like(self.phase).index_add_(0, motion_ids, step)
            touched = torch.zeros_like(self.phase, dtype=torch.bool)
            touched[motion_ids] = True
            self.phase = torch.where(touched, torch.clamp(self.phase + delta, *phase_range), self.phase)
            self._write_bin_weights()
        return {
            "phase_mean": self.phase.mean().item(),
            "phase_min": self.phase.min().item(),
            "phase_at_start_fraction": (self.phase <= phase_range[0] + 1e-6).float().mean().item(),
        }

    def _write_bin_weights(self):
        weights = self.motion_buffer._motion_bin_weights._concatenated_tensor
        bins = torch.minimum((self.phase * self.num_bins).long(), self.num_bins - 1)
        weights.zero_()
        weights[self.bin_starts + bins] = 1.0


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
