"""Return-based reset sampling over motion bins (SuperMimic, arXiv:2610.28480v1, App. D.1).

"The motion timeline is divided into 1 s bins, and each bin b keeps a running estimate G_b of the
discounted return from it, updated on the policy's own rollouts. Resets mix three distributions: 50%
uniform over bins, 40% from softmax_b(-25 G_b / G_max), where G_max is the largest attainable
discounted return, and 10% in proportion to each bin's past softmax weight, so a hard motion keeps
some practice after it is learned."

``failure_share`` adds InstantTrack's failure-focused resets (github.com/Huy-Quang-Dao/instantTrack, ``--fail_reset``):
each bin keeps a running termination rate of the episodes that end in it, and that share of the reset
distribution starts episodes one bin before the recent failure points, so a clip that is learned and then
forgotten is practised again before it drops out of the evaluation.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

    from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion


class return_based_reset_sampling(ManagerTermBase):
    """Keeps the per-bin return estimates and writes the reset distribution into the motion buffer.

    Runs as an ``interval`` event every control step with ``is_global_time=True``. Interval events run
    after the step's resets, so for an environment that ended an episode this step ``reward_buf`` holds
    that episode's last reward and the motion buffer already holds the next episode's start. The motion
    buffer must sample starts with ``env_starting_stub_sampling_strategy="concat_motion_bins"``, which
    draws (motion, bin) pairs from the weights written here.

    Params:
        reference_name: Scene entity of the motion reference manager.
        gamma: Discount of the return estimate (the agent's).
        max_step_reward: Largest per-step reward the env hands out (after the ``dt`` scaling); with it
            ``G_max = max_step_reward / (1 - gamma)``.
        temperature: The ``25`` of ``softmax(-25 G_b / G_max)``.
        mix: Weights of the uniform, softmax and past-softmax parts.
        estimate_rate: Weight of a new episode's return in the running estimate ``G_b``.
        past_rate: Rate of the running average of the softmax weights.
        failure_share: Share of the distribution drawn in proportion to the termination rate of the next
            bin of the same clip (0 disables it). InstantTrack uses 0.3.
        failure_rate: Weight of an ending episode in its end bin's termination rate.
        failure_decay: Per-step decay of every termination rate, so bins no episode ends in any more fade out.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        manager = env.scene[params.get("reference_name", "motion_reference")]
        if len(manager.motion_buffers) != 1:
            raise NotImplementedError("return-based reset sampling supports a single motion buffer")
        self.buffer: AmassMotion = next(iter(manager.motion_buffers.values()))
        if self.buffer.cfg.env_starting_stub_sampling_strategy != "concat_motion_bins":
            raise ValueError(
                "return-based reset sampling needs env_starting_stub_sampling_strategy='concat_motion_bins'"
            )
        weights = self.buffer._motion_bin_weights
        self.bin_starts = weights._batch_starts.to(self.device)
        self.bin_counts = weights._batch_sizes.to(self.device)
        self.bin_length_s = float(self.buffer.cfg.motion_bin_length_s)
        num_bins = int(self.bin_counts.sum())

        self.bin_return = torch.zeros(num_bins, device=self.device)
        self.past_softmax = torch.full((num_bins,), 1.0 / num_bins, device=self.device)
        self.episode_return = torch.zeros(env.num_envs, device=self.device)  # discounted
        self.episode_reward = torch.zeros(env.num_envs, device=self.device)  # undiscounted
        self.episode_steps = torch.zeros(env.num_envs, device=self.device)
        self.start_bin = self._current_start_bins()
        # failure-focused resets: where each episode started within its clip, and per-bin termination rates
        self.start_motion = self.buffer._assigned_env_motion_selection.to(self.device).long()
        self.start_time_s = self.buffer._motion_buffer_start_time_s.to(self.device).float()
        self.bin_failure = torch.zeros(num_bins, device=self.device)
        # the bin one before each bin of the same clip; a clip's first bin keeps itself
        bins = torch.arange(num_bins, device=self.device)
        first = torch.zeros(num_bins, dtype=torch.bool, device=self.device)
        first[self.bin_starts] = True
        self.previous_bin = torch.where(first, bins, bins - 1)

    def _current_start_bins(self) -> torch.Tensor:
        """Global bin index of every environment's current episode start."""
        motion = self.buffer._assigned_env_motion_selection.to(self.device).long()
        start_s = self.buffer._motion_buffer_start_time_s.to(self.device)
        local = torch.floor(start_s / self.bin_length_s).long()
        local = torch.minimum(local.clamp(min=0), self.bin_counts[motion] - 1)
        return self.bin_starts[motion] + local

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        reference_name: str = "motion_reference",
        gamma: float = 0.99,
        max_step_reward: float = 0.1,
        temperature: float = 25.0,
        mix: tuple[float, float, float] = (0.5, 0.4, 0.1),
        estimate_rate: float = 0.2,
        past_rate: float = 0.001,
        failure_share: float = 0.0,
        failure_rate: float = 0.1,
        failure_decay: float = 0.999,
    ) -> None:
        # an episode's start bin is read on its first step: by then the reset that began it has drawn it
        fresh = self.episode_steps == 0
        if bool(fresh.any()):
            self.start_bin = torch.where(fresh, self._current_start_bins(), self.start_bin)
            if failure_share > 0.0:
                motion = self.buffer._assigned_env_motion_selection.to(self.device).long()
                start_s = self.buffer._motion_buffer_start_time_s.to(self.device).float()
                self.start_motion = torch.where(fresh, motion, self.start_motion)
                self.start_time_s = torch.where(fresh, start_s, self.start_time_s)
        reward_buf = env.reward_buf
        if isinstance(reward_buf, dict):
            # a multi-reward manager keeps one buffer per reward group; the return is over their sum
            reward_buf = sum(reward_buf.values())
        reward = reward_buf.reshape(env.num_envs, -1).sum(-1).to(self.device)
        self.episode_return += gamma**self.episode_steps * reward
        self.episode_reward += reward
        self.episode_steps += 1

        done = env.reset_buf.to(self.device).bool()
        if bool(done.any()):
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            steps = self.episode_steps[ids]
            # a truncated episode (time limit, clip end) is bootstrapped with its mean reward
            truncated = env.reset_time_outs.to(self.device)[ids].bool()
            bootstrap = gamma**steps * (self.episode_reward[ids] / steps) / (1.0 - gamma)
            episode_return = self.episode_return[ids] + torch.where(truncated, bootstrap, 0.0)
            bins = self.start_bin[ids]
            self.bin_return[bins] = (1.0 - estimate_rate) * self.bin_return[bins] + estimate_rate * episode_return
            if failure_share > 0.0:
                # the bin the episode ended in: its start time plus the steps it ran, within its clip
                motion = self.start_motion[ids]
                end_s = self.start_time_s[ids] + steps * env.step_dt
                local = torch.floor(end_s / self.bin_length_s).long()
                end_bins = self.bin_starts[motion] + torch.minimum(local.clamp(min=0), self.bin_counts[motion] - 1)
                terminated = (~truncated).float()
                self.bin_failure.mul_(failure_decay)
                self.bin_failure[end_bins] = (1.0 - failure_rate) * self.bin_failure[
                    end_bins
                ] + failure_rate * terminated
            self.episode_return[ids] = 0.0
            self.episode_reward[ids] = 0.0
            self.episode_steps[ids] = 0.0

        g_max = max_step_reward / (1.0 - gamma)
        softmax = torch.softmax(-temperature * self.bin_return / g_max, dim=0)
        self.past_softmax.mul_(1.0 - past_rate).add_(past_rate * softmax)
        uniform = torch.full_like(softmax, 1.0 / softmax.numel())
        probs = mix[0] * uniform + mix[1] * softmax + mix[2] * self.past_softmax / self.past_softmax.sum()
        probs = probs / probs.sum()
        if failure_share > 0.0:
            # start one bin before where episodes recently failed
            failure = torch.zeros_like(self.bin_failure).index_add_(0, self.previous_bin, self.bin_failure)
            total = failure.sum()
            failure = torch.where(total > 0, failure / total.clamp(min=1e-12), probs)
            probs = (1.0 - failure_share) * probs + failure_share * failure
        self.buffer._motion_bin_weights._concatenated_tensor.copy_(probs / probs.sum())
