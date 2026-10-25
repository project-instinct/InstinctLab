"""Tests for the Newton native actuator gain reader in :mod:`instinctlab.envs.mdp.events.randomization`.

Newton controllers index backend DOFs; the reader must return the gains in the articulation's public joint order,
which ``ArticulationCfg.joint_ordering`` can make differ from the backend order. A fake asset stands in for Newton.
"""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from instinctlab.envs.mdp.events.randomization import _read_native_actuator_gain


class _WarpLike:
    def __init__(self, values):
        self._values = np.asarray(values)

    def numpy(self):
        return self._values


def _asset(public, backend, kp_by_backend_index, num_envs=2):
    indices = [env * len(backend) + i for env in range(num_envs) for i in range(len(backend))]
    controller = SimpleNamespace(kp=_WarpLike([kp_by_backend_index[i % len(backend)] for i in indices]))
    actuator = SimpleNamespace(controller=controller, indices=_WarpLike(indices))
    return SimpleNamespace(
        newton_actuator_adapter=SimpleNamespace(actuators=[actuator], num_joints=len(backend)),
        num_instances=num_envs,
        num_joints=len(public),
        joint_names=list(public),
        backend_joint_names=list(backend),
        device="cpu",
    )


def test_gains_come_back_in_public_order():
    backend = ["a", "b", "c"]
    public = ["c", "a", "b"]
    gains, covered = _read_native_actuator_gain(_asset(public, backend, [10.0, 20.0, 30.0]), "kp")
    assert covered.all()
    torch.testing.assert_close(gains, torch.tensor([[30.0, 10.0, 20.0]] * 2))


def test_matching_orders_are_unchanged():
    names = ["a", "b", "c"]
    gains, _ = _read_native_actuator_gain(_asset(names, names, [1.0, 2.0, 3.0]), "kp")
    torch.testing.assert_close(gains, torch.tensor([[1.0, 2.0, 3.0]] * 2))
