"""Tests for :mod:`instinctlab.utils.backend_dispatch`.

The tests never launch a simulator: ``SimulationContext.instance()`` is replaced by a fake so that the
dispatch rule (exact backend names) is pinned without Kit or USD.
"""

from __future__ import annotations

import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

import isaaclab.sim as sim_utils

from instinctlab.utils import backend_dispatch


def _fake_simulation(monkeypatch, backend_name: str):
    """Install a simulation context whose backend manager reports *backend_name*."""
    fake_sim = SimpleNamespace(backend_manager=SimpleNamespace(backend_name=backend_name))
    monkeypatch.setattr(sim_utils.SimulationContext, "instance", classmethod(lambda cls: fake_sim))
    return fake_sim


def _component_module(monkeypatch, module_name: str) -> types.ModuleType:
    """Register an importable module that records the cfg it was constructed with."""
    module = types.ModuleType(module_name)

    class FakeComponent:
        def __init__(self, cfg):
            self.cfg = cfg

    FakeComponent.__module__ = module_name
    module.FakeComponent = FakeComponent
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


@pytest.mark.parametrize("backend_name", ["physx", "newton", "ovphysx"])
def test_supported_backends_are_returned_verbatim(monkeypatch, backend_name):
    _fake_simulation(monkeypatch, backend_name)
    assert backend_dispatch.active_physics_backend() == backend_name


def test_unknown_backend_names_the_backend_and_supported_set(monkeypatch):
    _fake_simulation(monkeypatch, "mujoco")
    with pytest.raises(NotImplementedError, match="mujoco.*physx, newton, ovphysx"):
        backend_dispatch.active_physics_backend()


def test_missing_simulation_context_raises(monkeypatch):
    monkeypatch.setattr(sim_utils.SimulationContext, "instance", classmethod(lambda cls: None))
    with pytest.raises(RuntimeError, match="simulation context"):
        backend_dispatch.active_physics_backend()


@pytest.mark.parametrize("backend_name", ["physx", "newton", "ovphysx"])
def test_component_dispatch_uses_the_active_backend(monkeypatch, backend_name):
    _fake_simulation(monkeypatch, backend_name)
    for name in ("physx", "newton", "ovphysx"):
        _component_module(monkeypatch, f"instinctlab_test_{name}_component")

    cfg = object()
    component = backend_dispatch.create_backend_component(
        cfg,
        {
            "physx": "instinctlab_test_physx_component:FakeComponent",
            "newton": "instinctlab_test_newton_component:FakeComponent",
            "ovphysx": "instinctlab_test_ovphysx_component:FakeComponent",
        },
    )
    assert component.cfg is cfg
    assert type(component).__module__ == f"instinctlab_test_{backend_name}_component"


def test_component_without_the_backend_names_itself_and_its_supported_set(monkeypatch):
    _fake_simulation(monkeypatch, "ovphysx")
    with pytest.raises(NotImplementedError, match="FakeComponent .*'ovphysx'.*Supported backends: physx"):
        backend_dispatch.create_backend_component(object(), {"physx": "instinctlab_test_physx_component:FakeComponent"})


def test_module_import_does_not_load_the_simulator():
    """Importing the dispatch module must not import ``isaaclab.sim`` (the snapshot harness hooks its import)."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('dispatch_probe', {backend_dispatch.__file__!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert not hasattr(module, 'SimulationContext'), 'SimulationContext is bound at module level'\n"
        "print('isaaclab.sim' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
