"""Tests for :mod:`instinctlab.tasks.utils.backends`: backend presets and the per-backend task ids.

The cfgs are composed as the train/play scripts do (``resolve_task_config``); no simulator is launched.
"""

from __future__ import annotations

import gymnasium as gym

import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.utils import to_dict
from isaaclab_tasks.utils import resolve_task_config

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.utils.backends import BACKENDS, physics_preset
from instinctlab.utils.instinct_rl_cli import AGENT_ENTRY_POINT

PHYSX_TASKS = sorted(
    task
    for task in gym.registry
    if task.startswith("Instinct-") and not any(f"{suffix}-" in task for suffix, _ in BACKENDS.values() if suffix)
)
PHYSICS_CLASSES = {"physx": PhysxCfg, "newton": NewtonCfg, "ovphysx": OvPhysxCfg}


def _task_id(physx_task: str, backend: str) -> str:
    suffix = BACKENDS[backend][0]
    if physx_task.endswith("-Play-v0"):
        return physx_task.replace("-Play-v0", f"{suffix}-Play-v0")
    return physx_task.replace("-v0", f"{suffix}-v0")


def _env_cfg(task: str, overrides=()):
    return resolve_task_config(task, AGENT_ENTRY_POINT, play_mode="-Play-" in task, overrides=list(overrides))[0]


def test_every_task_has_every_backend():
    assert len(PHYSX_TASKS) == 14
    for task in PHYSX_TASKS:
        for backend in BACKENDS:
            assert _task_id(task, backend) in gym.registry


@pytest.mark.parametrize("task", PHYSX_TASKS)
@pytest.mark.parametrize("backend", ["newton", "ovphysx"])
def test_backend_id_equals_the_physics_selector(task, backend):
    by_id = _env_cfg(_task_id(task, backend))
    by_selector = _env_cfg(task, [f"physics={BACKENDS[backend][1]}"])
    assert type(by_id.sim.physics) is PHYSICS_CLASSES[backend]
    assert repr(to_dict(by_id)) == repr(to_dict(by_selector))


@pytest.mark.parametrize("task", [task for task in PHYSX_TASKS if "-Play-" not in task])
def test_ovphysx_keeps_the_physx_buffers(task):
    physx, ovphysx = _env_cfg(task).sim.physics, _env_cfg(_task_id(task, "ovphysx")).sim.physics
    for name in OvPhysxCfg.__dataclass_fields__:
        if name.startswith("gpu_"):
            assert getattr(ovphysx, name) == getattr(physx, name), name


def test_ovphysx_play_tasks_default_to_viser():
    for task in (t for t in PHYSX_TASKS if "-Play-" in t):
        viewers = _env_cfg(_task_id(task, "ovphysx")).sim.visualizer_cfgs
        assert [viewer.backend.visualizer_type for viewer in viewers] == ["viser"], task


def test_a_backend_id_rejects_another_backend():
    with pytest.raises(ValueError, match="physics=ovphysx"):
        _env_cfg("Instinct-Locomotion-Flat-G1-Newton-v0", ["physics=ovphysx"])


def test_physics_preset_rejects_settings_ovphysx_cannot_take():
    with pytest.raises(ValueError, match="bounce_threshold_velocity"):
        physics_preset(PhysxCfg(bounce_threshold_velocity=0.2), newton=NewtonCfg())
