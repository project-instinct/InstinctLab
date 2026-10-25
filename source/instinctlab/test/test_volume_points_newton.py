"""Simulator-free tests for the Newton volume-points backend.

``NewtonVolumePoints`` reads the Newton model, its state and the resolved site map from the simulation's
physics-manager instance. On Isaac Lab main the class-level ``NewtonManager._cl_site_index_map`` is the empty
declaration dict, so a class-level read either raises a misleading "site was not registered" error or, worse,
silently resolves stale indices. These tests pin the instance read with a fake manager and never launch a
simulator.
"""

from __future__ import annotations

import torch
from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager

from instinctlab.sensors.volume_points.newton import NewtonVolumePoints
from instinctlab.sensors.volume_points.volume_points_data import VolumePointsData

wp.init()

SITE_LABEL = "ft_7"
ROBOT_PRIM_PATH = "/World/envs/env_.*/Robot"
BODY_LABELS = ["Robot/left_ankle_roll_link", "Robot/right_ankle_roll_link"]
SITE_OFFSETS = [(0.1, 0.0, 0.0), (0.0, 0.2, 0.0)]
BODY_POSITIONS = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]


class _FakeNewtonManager:
    """Stand-in for ``sim.backend_manager`` that exposes only the instance API."""

    def __init__(self, site_map, model, state):
        self._cl_site_index_map = site_map
        self._model = model
        self._state_0 = state

    def get_model(self):
        return self._model

    def get_state_0(self):
        return self._state_0


def _make_model() -> SimpleNamespace:
    return SimpleNamespace(
        shape_body=wp.array([0, 1], dtype=wp.int32, device="cpu"),
        shape_transform=wp.array(
            [wp.transform(wp.vec3(*offset), wp.quat(0.0, 0.0, 0.0, 1.0)) for offset in SITE_OFFSETS],
            dtype=wp.transform,
            device="cpu",
        ),
        body_com=wp.array([wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu"),
        body_label=list(BODY_LABELS),
    )


def _make_state() -> SimpleNamespace:
    return SimpleNamespace(
        body_q=wp.array(
            [wp.transform(wp.vec3(*pos), wp.quat(0.0, 0.0, 0.0, 1.0)) for pos in BODY_POSITIONS],
            dtype=wp.transform,
            device="cpu",
        ),
        body_qd=wp.zeros(2, dtype=wp.spatial_vector, device="cpu"),
    )


def _make_sensor(site_map) -> NewtonVolumePoints:
    """Build a sensor whose initialization must resolve its sites through the manager instance."""
    manager = _FakeNewtonManager(site_map, _make_model(), _make_state())
    sensor = object.__new__(NewtonVolumePoints)
    # Callback handles normally set by SensorBase.__init__; SensorBase.__del__ clears them.
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None
    sensor._debug_vis_handle = None
    sensor._backend_manager = manager
    sensor.cfg = SimpleNamespace(prim_path=ROBOT_PRIM_PATH)
    sensor._num_envs = 1
    sensor._device = "cpu"
    sensor._site_label = SITE_LABEL
    sensor._body_pattern = "Robot/.*_ankle_roll_link"
    sensor._site_indices = None
    sensor._newton_model = None
    sensor._data = VolumePointsData.make_zero(num_envs=1, num_bodies=2, point_num_each_body=2, device="cpu")
    return sensor


def test_sites_resolve_through_the_manager_instance(monkeypatch):
    # Decoy: the class-level declaration dict must never be the source of the resolved indices.
    monkeypatch.setattr(NewtonManager, "_cl_site_index_map", {SITE_LABEL: (None, [[1, 0]])})

    sensor = _make_sensor({SITE_LABEL: (None, [[0, 1]])})

    sensor._initialize_backend_impl()

    assert sensor.num_bodies == 2
    assert sensor.body_names == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert wp.to_torch(sensor._site_indices).tolist() == [0, 1]
    assert sensor._body_state_is_valid()


def test_refresh_reads_the_manager_state(monkeypatch):
    monkeypatch.setattr(NewtonManager, "_cl_site_index_map", {SITE_LABEL: (None, [[1, 0]])})

    sensor = _make_sensor({SITE_LABEL: (None, [[0, 1]])})
    sensor._initialize_backend_impl()
    sensor._refresh_body_state(wp.ones(1, dtype=wp.bool, device="cpu"), torch.tensor([0]))

    expected_pos = torch.tensor(
        [[[x + dx, y + dy, z] for (dx, dy, _), (x, y, z) in zip(SITE_OFFSETS, BODY_POSITIONS)]],
        dtype=torch.float32,
    )
    assert torch.allclose(sensor._data.pos_w.torch, expected_pos, atol=1e-6)
    assert torch.allclose(sensor._data.quat_w.torch[..., 3], torch.ones(1, 2), atol=1e-6)
    assert torch.equal(sensor._data.vel_w.torch, torch.zeros(1, 2, 3))


def test_missing_site_raises_for_the_registered_label(monkeypatch):
    monkeypatch.setattr(NewtonManager, "_cl_site_index_map", {SITE_LABEL: (None, [[1, 0]])})

    with pytest.raises(ValueError, match=SITE_LABEL):
        _make_sensor({})._initialize_backend_impl()


def test_bodies_follow_the_robot_body_ordering(monkeypatch):
    import isaaclab.sim as sim_utils

    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path=ROBOT_PRIM_PATH, body_ordering=("pelvis", "right_ankle_roll_link", "left_ankle_roll_link")
        )
    )
    fake_sim = SimpleNamespace(_interactive_scene=SimpleNamespace(articulations={"robot": robot}))
    monkeypatch.setattr(sim_utils.SimulationContext, "instance", classmethod(lambda cls: fake_sim))
    monkeypatch.setattr(NewtonManager, "_cl_site_index_map", {})
    sensor = _make_sensor({SITE_LABEL: (None, [[0, 1]])})

    sensor._initialize_backend_impl()

    assert sensor.body_names == ["right_ankle_roll_link", "left_ankle_roll_link"]
    assert wp.to_torch(sensor._site_indices).tolist() == [1, 0]
