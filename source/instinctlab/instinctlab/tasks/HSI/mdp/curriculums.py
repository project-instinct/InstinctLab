from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg
    from isaaclab.motion_reference.motion_files.amass_motion import AmassMotion

    from instinctlab.motion_reference import MotionReferenceManager


def update_motion_reference_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reference_name: str = "motion_reference",
    success_weight_ratio: float = 0.6,
) -> dict[str, float] | None:
    """Update the motion reference weights based on the timeouts and terminated states."""
    motion_reference: MotionReferenceManager = env.scene[reference_name]
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return None
    timeout_env_ids = env_ids[env.reset_time_outs[env_ids]]
    terminated_env_ids = env_ids[env.reset_terminated[env_ids]]
    motion_reference.update_motion_weights(
        timeout_env_ids,
        weight_ratio=success_weight_ratio,
    )
    motion_reference.update_motion_weights(
        terminated_env_ids,
        weight_ratio=1.0 / success_weight_ratio,
    )
    return {
        "timeout_ratio": len(timeout_env_ids) / len(env_ids) if len(env_ids) > 0 else 0.0,
        "terminated_ratio": len(terminated_env_ids) / len(env_ids) if len(env_ids) > 0 else 0.0,
    }


def update_motion_reference_weights_by_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reference_name: str = "motion_reference",
) -> dict[str, float] | None:
    """Update the motion reference weights based on (1-progress) of the environment."""
    num_envs = len(env_ids)
    if num_envs == 0:
        return None
    motion_reference: MotionReferenceManager = env.scene[reference_name]
    current_weights = motion_reference.get_current_motion_weights(env_ids)
    weight_sum = current_weights.sum()
    progress = (env.episode_length_buf * env.step_dt)[env_ids] / motion_reference.assigned_motion_lengths[env_ids]

    # normalize the ratio to make the weight sum remain the same
    # make the env with less progress have higher weight
    weight_ratio = 1.0 - progress
    weight_ratio /= weight_ratio.sum()
    weight_ratio *= weight_sum
    motion_reference.update_motion_weights(
        env_ids,
        weight_ratio=weight_ratio,
    )
    return {
        "weight_sum": weight_sum,
        "weight_ratio_sum": weight_ratio.sum(),
        "progress_mean": progress.mean(),
        "progress_std": progress.std(),
    }


class update_motion_reference_weights_by_delayed_stats(ManagerTermBase):
    """A manager term that updates the motion reference weights based on the delayed statistics of the environment,
    such as progress, shadowing error ...
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delayed_progress = cfg.params.get("init_delayed_progress", 0.0)
        self.experience_length = cfg.params.get("init_experience_length", 0.0)
        self.motion_reference: MotionReferenceManager = env.scene[cfg.params.get("reference_name", "motion_reference")]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reference_name: str = "motion_reference",
        progress_refresh_epsilon: float = 0.01,
        length_refresh_epsilon: float = 0.01,
        ratio_option: Literal["div_mean", "div_sum"] = "div_mean",
        init_delayed_progress: float = 0.0,
        init_experience_length: float = 0.02,
    ) -> dict[str, float] | None:
        """Update the motion reference weights based on the delayed statistics of the environment."""
        if len(env_ids) == 0 or env.common_step_counter < 1:
            # if no env_ids or the environment has not stepped yet, return None
            return None

        experience_length = (env.episode_length_buf * env.step_dt)[env_ids]

        self.experience_length = (
            1 - length_refresh_epsilon
        ) * self.experience_length + length_refresh_epsilon * experience_length.mean()
        progress = experience_length / self.motion_reference.assigned_motion_lengths[env_ids]
        self.delayed_progress = (
            1 - progress_refresh_epsilon
        ) * self.delayed_progress + progress_refresh_epsilon * progress.mean()

        # compute and normalize the weight ratio
        weight_ratio = (1.0 - progress) / torch.sqrt(torch.clamp(experience_length, min=1e-6))
        if ratio_option == "div_mean":
            weight_ratio /= (1 - self.delayed_progress) / torch.sqrt(self.experience_length)
        elif ratio_option == "div_sum":
            weight_ratio /= weight_ratio.sum()
        else:
            raise ValueError(f"Unknown ratio option: {ratio_option}")

        # update the motion reference weights
        assert len(env_ids) == len(weight_ratio), "env_ids and weight_ratio must have the same length"
        self.motion_reference.update_motion_weights(
            env_ids,
            weight_ratio=weight_ratio,
        )
        return {
            "delayed_progress": self.delayed_progress,
            "delayed_experience_length": self.experience_length,
            "weight_ratio_sum": weight_ratio.sum(),
            "progress_mean": progress.mean(),
            "progress_std": progress.std(),
        }


def update_motion_reference_weights_by_experience(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reference_name: str = "motion_reference",
    success_weight_ratio: float = 1.0,
    full_trajectory_weight_ratio: float = 0.6,
    full_trajectory_threshold: float = 0.8,
    sampled_weight_ratio: float = 0.99,
):
    """Update the motion reference weights based on the experience of the environment.
    If the motion is successful or sampled, the weight of this motion shall be decreased.

    Args:
        - success_weight_ratio: The weight ratio for succeeded motion env.
        - full_trajectory_weight_ratio: The weight ratio the env that experience the most of the trajectory and succeeded.
        - sampled_weight_ratio: The weight ratio for the motion that is sampled once.

    """
    motion_reference: MotionReferenceManager = env.scene[reference_name]
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return None
    experience_length = (env.episode_length_buf * env.step_dt)[env_ids]
    full_trajectory_env_ids_mask = (
        experience_length / motion_reference.complete_motion_lengths[env_ids] > full_trajectory_threshold
    )

    timeout_env_ids = env_ids[env.reset_time_outs[env_ids]]

    full_trajectory_env_ids = env_ids[full_trajectory_env_ids_mask & env.reset_time_outs[env_ids]]

    if len(timeout_env_ids) > 0 and success_weight_ratio < 1.0:
        motion_reference.update_motion_weights(
            timeout_env_ids,
            weight_ratio=success_weight_ratio,
        )
    if len(full_trajectory_env_ids) > 0 and sampled_weight_ratio < 1.0:
        motion_reference.update_motion_weights(
            env_ids,
            weight_ratio=sampled_weight_ratio,
        )
    if len(full_trajectory_env_ids) > 0 and full_trajectory_weight_ratio < 1.0:
        motion_reference.update_motion_weights(
            full_trajectory_env_ids,
            weight_ratio=full_trajectory_weight_ratio,
        )

    return {
        "timeout_ratio": len(timeout_env_ids) / len(env_ids) if len(env_ids) > 0 else 0.0,
        "sampled_ratio": (len(env_ids) - len(timeout_env_ids)) / len(env_ids) if len(env_ids) > 0 else 0.0,
    }
