"""Curriculum terms for reset-to-fall difficulty scaling.

The event term resolved by ``event_name`` must be a
:class:`~gilab.envs.mdp.events.ResetToFallOrDefault` (or compatible subclass).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gilab.envs.mdp.events.reset_to_fall import ResetToFallOrDefault

    from isaaclab.envs import ManagerBasedRLEnv


def update_reset_fall_difficulty(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str = "reset_robot",
    difficulty_step_size: float = 0.1,
) -> torch.Tensor:
    """Increase / decrease the fall-reset difficulty based on episode outcome.

    Only adjusts environments currently in fall mode. Environments that timed out
    get harder; environments that terminated early get easier.
    """
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return torch.tensor(0.0)

    time_outs = env.reset_time_outs[env_ids]
    terminates = env.reset_terminated[env_ids]

    reset_event_term: ResetToFallOrDefault = env.event_manager.get_term_cfg(event_name).func
    reset_fall_mode = reset_event_term.reset_as_fall_mode[env_ids]

    env_ids_fall_mode = env_ids[reset_fall_mode]
    env_ids_down = env_ids[reset_fall_mode & terminates]
    env_ids_up = env_ids[reset_fall_mode & time_outs]

    reset_event_term.reset_difficulty[env_ids_down] -= difficulty_step_size
    reset_event_term.reset_difficulty[env_ids_down] = torch.clip(
        reset_event_term.reset_difficulty[env_ids_down], 0.0, 1.0
    )
    reset_event_term.reset_difficulty[env_ids_up] += difficulty_step_size
    reset_event_term.reset_difficulty[env_ids_up] = torch.clip(reset_event_term.reset_difficulty[env_ids_up], 0.0, 1.0)

    return torch.mean(reset_event_term.reset_difficulty[env_ids_fall_mode])


def update_reset_fall_difficulty_by_termination_terms(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str = "reset_robot",
    difficulty_step_size: float = 0.1,
    prevent_increase_timeout_terms: list[str] = [],
    prevent_decrease_termination_terms: list[str] = [],
) -> torch.Tensor:
    """Like :func:`update_reset_fall_difficulty` but filters individual termination terms.

    Some timeout terms (e.g. ``illegal_reset_contact``) fire as time_out but should not
    increase difficulty. Some termination terms fire as terminated but should not decrease
    difficulty. This function recomputes the effective timeout/terminated signals by
    excluding the listed terms before adjusting difficulty.
    """
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return torch.tensor(0.0)

    term_mgr = env.termination_manager

    effective_time_outs = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    effective_terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name, term_cfg in zip(term_mgr._term_names, term_mgr._term_cfgs):
        term_done = term_mgr.get_term(name)
        if term_cfg.time_out:
            if name not in prevent_increase_timeout_terms:
                effective_time_outs |= term_done
        else:
            if name not in prevent_decrease_termination_terms:
                effective_terminated |= term_done

    time_outs = effective_time_outs[env_ids]
    terminates = effective_terminated[env_ids]

    reset_event_term: ResetToFallOrDefault = env.event_manager.get_term_cfg(event_name).func
    reset_fall_mode = reset_event_term.reset_as_fall_mode[env_ids]

    env_ids_fall_mode = env_ids[reset_fall_mode]
    env_ids_down = env_ids[reset_fall_mode & terminates]
    env_ids_up = env_ids[reset_fall_mode & time_outs]

    reset_event_term.reset_difficulty[env_ids_down] -= difficulty_step_size
    reset_event_term.reset_difficulty[env_ids_down] = torch.clip(
        reset_event_term.reset_difficulty[env_ids_down], 0.0, 1.0
    )
    reset_event_term.reset_difficulty[env_ids_up] += difficulty_step_size
    reset_event_term.reset_difficulty[env_ids_up] = torch.clip(reset_event_term.reset_difficulty[env_ids_up], 0.0, 1.0)

    return torch.mean(reset_event_term.reset_difficulty[env_ids_fall_mode])
