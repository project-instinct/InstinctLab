"""Grace-period termination wrappers for fall-reset environments.

The event term resolved by ``reset_event_name`` must be a
:class:`~instinctlab.envs.mdp.events.ResetToFallOrDefault` (or compatible subclass).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Callable, Sequence

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg

    from instinctlab.envs.mdp.events import ResetToFallOrDefault


class FallGraceTermination(ManagerTermBase):
    """Wraps a termination function and suppresses it for fall-reset envs during a grace period.

    When ``ResetToFallOrDefault`` resets an env in fall mode, its initial state will
    likely violate termination conditions immediately (e.g. contact forces, position
    deviation). This wrapper gives those envs a configurable number of steps before the
    inner termination is allowed to fire.

    Config params:
        inner_func: The termination function to wrap.
        inner_params: kwargs forwarded to ``inner_func``.
        reset_event_name: Name of the ``ResetToFallOrDefault`` event term (default: ``"reset_robot"``).
        grace_steps: Number of steps after a fall-reset during which the inner termination is suppressed.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._inner_func: Callable = cfg.params["inner_func"]
        self._inner_params: dict = cfg.params.get("inner_params", {})
        self._reset_event_name: str = cfg.params.get("reset_event_name", "reset_robot")
        self._grace_steps: int = cfg.params.get("grace_steps", 25)

        self._is_fall_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        inner_is_class = isinstance(self._inner_func, type) and issubclass(self._inner_func, ManagerTermBase)
        if inner_is_class:
            inner_cfg_copy = cfg.__class__(
                func=self._inner_func,
                params=self._inner_params,
                time_out=cfg.time_out,
            )
            self._inner_instance = self._inner_func(inner_cfg_copy, env)
        else:
            self._inner_instance = None

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        reset_event: ResetToFallOrDefault = self._env.event_manager.get_term_cfg(self._reset_event_name).func
        self._is_fall_reset[env_ids] = reset_event.reset_as_fall_mode[env_ids]

        if self._inner_instance is not None:
            self._inner_instance.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        inner_func: Callable | None = None,
        inner_params: dict | None = None,
        reset_event_name: str = "reset_robot",
        grace_steps: int = 5,
    ) -> torch.Tensor:
        if self._inner_instance is not None:
            result = self._inner_instance(env, **self._inner_params)
        else:
            result = self._inner_func(env, **self._inner_params)

        in_grace = self._is_fall_reset & (env.episode_length_buf <= self._grace_steps)
        result[in_grace] = False
        return result


class ConsecutiveFallGraceTermination(ManagerTermBase):
    """Wraps a termination function and only fires after the inner condition is true for
    consecutive steps exceeding a grace threshold.

    Fall-reset envs and default-reset envs have independent grace thresholds. The inner
    termination must be true for ``grace_steps`` (fall) or ``grace_steps_ref`` (default)
    consecutive steps before the termination actually triggers. If the inner condition
    becomes false at any step, the counter resets to zero.

    Config params:
        inner_func: The termination function to wrap.
        inner_params: kwargs forwarded to ``inner_func``.
        reset_event_name: Name of the ``ResetToFallOrDefault`` event term (default: ``"reset_robot"``).
        grace_steps: Consecutive true steps required before terminating a fall-reset env.
        grace_steps_ref: Consecutive true steps required before terminating a default-reset env.
            Defaults to 0 (terminate immediately).
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._inner_func: Callable = cfg.params["inner_func"]
        self._inner_params: dict = cfg.params.get("inner_params", {})
        self._reset_event_name: str = cfg.params.get("reset_event_name", "reset_robot")
        self._grace_steps: int = cfg.params.get("grace_steps", 25)
        self._grace_steps_ref: int = cfg.params.get("grace_steps_ref", 0)

        self._is_fall_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._consecutive_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        inner_is_class = isinstance(self._inner_func, type) and issubclass(self._inner_func, ManagerTermBase)
        if inner_is_class:
            inner_cfg_copy = cfg.__class__(
                func=self._inner_func,
                params=self._inner_params,
                time_out=cfg.time_out,
            )
            self._inner_instance = self._inner_func(inner_cfg_copy, env)
        else:
            self._inner_instance = None

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        reset_event: ResetToFallOrDefault = self._env.event_manager.get_term_cfg(self._reset_event_name).func
        self._is_fall_reset[env_ids] = reset_event.reset_as_fall_mode[env_ids]
        self._consecutive_count[env_ids] = 0

        if self._inner_instance is not None:
            self._inner_instance.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        inner_func: Callable | None = None,
        inner_params: dict | None = None,
        reset_event_name: str = "reset_robot",
        grace_steps: int = 5,
        grace_steps_ref: int = 0,
    ) -> torch.Tensor:
        if self._inner_instance is not None:
            result = self._inner_instance(env, **self._inner_params)
        else:
            result = self._inner_func(env, **self._inner_params)

        self._consecutive_count[result] += 1
        self._consecutive_count[~result] = 0

        grace_threshold = torch.where(self._is_fall_reset, self._grace_steps, self._grace_steps_ref)
        return self._consecutive_count > grace_threshold
