from __future__ import annotations

import math
import torch
from abc import abstractmethod
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerBase, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import SensorBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MonitorSensor(SensorBase):
    """Base class for monitoring the environment in scene sensors.
    Need implementation of `_update_buffers_impl` to acquire data in each sim step.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        # Import the backend only after launch_simulation has initialized Kit.
        from isaaclab_physx.physics import PhysxManager

        self._physics_sim_view = PhysxManager.get_physics_sim_view()

    # def _update_buffers_impl(self, env_ids: Sequence[int]):

    @abstractmethod
    def get_log(self, is_episode=False) -> dict[str, float]:
        pass

    def _invalidate_initialize_callback(self, event):
        """Invalidate the initialize callback."""
        super()._invalidate_initialize_callback(event)
        self._physics_sim_view = None


class MonitorTerm(ManagerTermBase):
    """Base class for monitoring the environment in scene managers. Typically gets data
    in each env step.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env: ManagerBasedRLEnv = env

    def update(self, *args, **kwargs):
        """Called after each environment step to acquire data from the environment."""
        pass

    def reset_idx(self, env_ids: Sequence[int] | slice):
        # placeholder
        pass

    @abstractmethod
    def get_log(self, is_episode=False) -> dict[str, float]:
        pass


class MonitorManager(ManagerBase):
    _env: ManagerBasedRLEnv

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self._terms: dict[str, MonitorTerm] = dict()
        super().__init__(cfg, env)

        self._monitor_data = dict()

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<MonitorManager> contains {len(self._terms)} active groups.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Monitor Terms"
        table.field_names = ["Index", "Name", "Func"]
        # set alignment of table columns
        table.align["Group"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        index = 0
        for term_name in self._terms.keys():
            term = self._terms[term_name]
            if isinstance(term, MonitorTerm):
                func = term.__class__.__name__
            else:
                func = term.__name__
            table.add_row([index, term_name, func])
            index += 1
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Names of the active monitor terms (standard ``ManagerBase`` contract)."""
        return list(self._terms.keys())

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        return True

    """
    Operations
    """

    def update(self, *args, dt, **kwargs):
        monitor_infos = dict()
        for term_name, term in self._terms.items():
            if isinstance(term, MonitorTerm):
                term.update(*args, dt=dt, **kwargs)
            # MonitorSensor update happens when sensor_manager updates
            log = term.get_log(is_episode=False)
            for key, value in log.items():
                monitor_infos[f"Step_Monitor/{term_name}_{key}"] = value
        return monitor_infos

    def reset(self, env_ids: Sequence[int], is_episode=False):
        monitor_infos = dict()
        for term_name, term in self._terms.items():
            if isinstance(term, MonitorTerm):
                term.reset_idx(env_ids)
            log = term.get_log(is_episode=is_episode)
            for key, value in log.items():
                monitor_infos[f"Episode_Monitor/{term_name}_{key}"] = value
        return monitor_infos

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Get the active iterable terms for the given environment index."""
        iterable_terms = []
        for term_name, term in self._terms.items():
            if isinstance(term, MonitorTerm):
                term_data = term.get_log(is_episode=False)
                term_data_list = []
                for data in term_data.values():
                    if isinstance(data, torch.Tensor):
                        data = data.cpu().item()
                    if not (math.isnan(data) or math.isinf(data)):
                        term_data_list.append(data)
                    else:
                        term_data_list.append(0.0)
                if term_data_list:
                    iterable_terms.append((term_name, term_data_list))
        return iterable_terms

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if isinstance(term_cfg, SceneEntityCfg):
                term = self._env.scene[term_cfg.name]
            else:
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                term = term_cfg.func
            self._terms[term_name] = term
