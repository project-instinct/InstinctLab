"""Tests for the InstinctLab visualizer configurations.

The Newton GL visualizer lifecycle is exercised with a fake backend and viewer: opening a real
window needs a display, which the test environment does not provide.
"""

from __future__ import annotations

import importlib
import numpy as np
import warnings

import pytest

from isaaclab.utils.string import string_to_callable
from isaaclab.visualizers import VisualizerCfg
from isaaclab.visualizers.visualizer_cfg import migrate_visualizer_cfgs

from instinctlab.tasks.utils.backends import select_backend
from instinctlab.tasks.utils.newton import InstinctNewtonBackendCfg, InstinctNewtonVisualizerCfg
from instinctlab.tasks.utils.newton_visualizer import InstinctNewtonGLVisualizer


class _FakeBodyQ:
    """Stand-in for a warp array carrying body poses."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float64)

    def numpy(self):
        return self._values


class _FakeCamera:
    def __init__(self):
        self.pos = None
        self.pitch = 0.0
        self.yaw = 0.0
        self.target = None

    def look_at(self, target):
        self.target = tuple(target)

    def get_view_matrix(self):
        return np.eye(4, dtype=np.float32)


class _FakeViewer:
    def __init__(self):
        self.camera = _FakeCamera()
        self.show_visual = True
        self.set_camera_calls = []

    def set_camera(self, pos, pitch, yaw):
        self.set_camera_calls.append((pos, pitch, yaw))


class _FakeBackend:
    def __init__(self, labels, body_q):
        self.model = type("Model", (), {"body_label": labels})()
        self.state_0 = type("State", (), {"body_q": _FakeBodyQ(body_q)})()


LABELS = ["world/envs/env_0/Robot/pelvis", "world/envs/env_0/Robot/torso_link"]
BODY_Q = [
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
]


@pytest.fixture
def visualizer(monkeypatch):
    """Build an InstinctNewtonGLVisualizer with the simulator-backed base methods replaced."""
    from isaaclab_visualizers.newton import NewtonGLVisualizer

    backend = _FakeBackend(list(LABELS), BODY_Q)
    viewer = _FakeViewer()
    steps = []

    def fake_initialize(self, sim, *, cameras):
        self._viewer = viewer
        self.backend = backend
        self._is_initialized = True

    def fake_step(self, dt):
        steps.append(dt)

    monkeypatch.setattr(NewtonGLVisualizer, "initialize", fake_initialize)
    monkeypatch.setattr(NewtonGLVisualizer, "step", fake_step)

    def build(backend_cfg):
        instance = InstinctNewtonGLVisualizer(InstinctNewtonVisualizerCfg(backend=backend_cfg))
        instance.initialize(object(), cameras=[])
        return instance

    return build, viewer, steps


def test_newton_visualizer_cfg_uses_non_deprecated_form():
    cfg = InstinctNewtonVisualizerCfg(
        backend=InstinctNewtonBackendCfg(show_contacts=True, show_visual=False, follow_body=True)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrated = migrate_visualizer_cfgs(cfg)

    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert migrated == [cfg]
    assert migrated[0] is cfg
    assert cfg.backend.visualizer_type == "newton_gl"
    assert string_to_callable(str(cfg.backend.class_type)) is InstinctNewtonGLVisualizer


def test_show_visual_overrides_viewer(visualizer):
    build, viewer, _ = visualizer
    build(InstinctNewtonBackendCfg(show_visual=False))
    assert viewer.show_visual is False


def test_show_visual_none_keeps_viewer_default(visualizer):
    build, viewer, _ = visualizer
    build(InstinctNewtonBackendCfg())
    assert viewer.show_visual is True


def test_follow_body_recenters_camera_each_step(visualizer):
    build, viewer, steps = visualizer
    instance = build(InstinctNewtonBackendCfg(follow_body=True))

    expected_lookat = (1.0, 2.0, 3.4)
    expected_eye = (2.5, 2.0, 4.5)

    # initialize() already places the camera on the torso.
    assert instance.cfg.lookat == pytest.approx(expected_lookat)
    assert instance.cfg.eye == pytest.approx(expected_eye)
    assert tuple(viewer.camera.target) == pytest.approx(expected_lookat)

    instance.cfg.eye = (0.0, 0.0, 0.0)
    instance.step(0.02)

    assert steps == [0.02]
    assert instance.cfg.lookat == pytest.approx(expected_lookat)
    assert instance.cfg.eye == pytest.approx(expected_eye)
    assert tuple(viewer.camera.pos) == pytest.approx(expected_eye)
    assert tuple(viewer.camera.target) == pytest.approx(expected_lookat)


def test_follow_body_falls_back_to_pelvis(visualizer, monkeypatch):
    from isaaclab_visualizers.newton import NewtonGLVisualizer

    backend = _FakeBackend([LABELS[0]], [BODY_Q[0]])

    def fake_initialize(self, sim, *, cameras):
        self._viewer = _FakeViewer()
        self.backend = backend
        self._is_initialized = True

    monkeypatch.setattr(NewtonGLVisualizer, "initialize", fake_initialize)
    instance = InstinctNewtonGLVisualizer(
        InstinctNewtonVisualizerCfg(backend=InstinctNewtonBackendCfg(follow_body=True))
    )
    instance.initialize(object(), cameras=[])

    assert instance.cfg.lookat == pytest.approx((0.0, 0.0, 0.4))
    assert instance.cfg.eye == pytest.approx((1.5, 0.0, 1.5))


def test_follow_body_disabled_leaves_camera(visualizer):
    build, _, steps = visualizer
    instance = build(InstinctNewtonBackendCfg())
    instance.cfg.eye = (7.0, 8.0, 9.0)
    instance.step(0.02)

    assert steps == [0.02]
    assert instance.cfg.eye == (7.0, 8.0, 9.0)


APP_CFGS = [
    (
        "instinctlab.tasks.locomotion.config.g1.flat_env_cfg",
        "G1FlatEnvCfg",
        (2.0, 2.0, 0.5),
        (0.0, 0.0, 0.0),
    ),
    (
        "instinctlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg",
        "G1PlaneShadowingEnvCfg_PLAY",
        (4.0, 0.75, 1.0),
        (0.0, 0.75, 0.0),
    ),
    (
        "instinctlab.tasks.shadowing.beyondmimic.config.g1.beyondmimic_plane_cfg",
        "G1BeyondMimicPlaneEnvCfg_PLAY",
        (4.0, 0.75, 1.0),
        (0.0, 0.75, 0.0),
    ),
    (
        "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg",
        "G1PerceptiveShadowingEnvCfg_PLAY",
        (1.5, 0.0, 1.5),
        (0.0, 0.0, 0.0),
    ),
    (
        "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_vae_cfg",
        "G1PerceptiveVaeEnvCfg_PLAY",
        (0.0, 2.0, 2.5),
        (0.0, 0.0, 0.0),
    ),
    (
        "instinctlab.tasks.shadowing.perceptive_hoi.config.g1.perceptive_shadowing_cfg",
        "G1PerceptiveHoiShadowingEnvCfg_PLAY",
        (1.5, 0.0, 1.5),
        (0.0, 0.0, 0.0),
    ),
    (
        "instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg",
        "G1ParkourEnvCfg_PLAY",
        (4.0, 0.75, 1.0),
        (0.0, 0.75, 0.0),
    ),
]


@pytest.mark.parametrize("module_name, cfg_name, eye, lookat", APP_CFGS)
def test_task_cfgs_configure_kit_visualizer(module_name, cfg_name, eye, lookat):
    cfg = select_backend(getattr(importlib.import_module(module_name), cfg_name)(), "physx")

    assert cfg.sim.default_visualizer_cfg is None
    assert len(cfg.sim.visualizer_cfgs) == 1
    visualizer_cfg = cfg.sim.visualizer_cfgs[0]
    assert isinstance(visualizer_cfg, VisualizerCfg)
    assert visualizer_cfg.backend.visualizer_type == "kit"
    assert visualizer_cfg.backend.origin_type == "asset"
    assert visualizer_cfg.backend.origin_track_path == "robot"
    assert visualizer_cfg.eye == eye
    assert visualizer_cfg.lookat == lookat
    assert visualizer_cfg.window.size == (1280, 720)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrate_visualizer_cfgs(cfg.sim.visualizer_cfgs, cfg.sim.default_visualizer_cfg)
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize(
    "module_name, cfg_name",
    [
        (
            "instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg",
            "G1PerceptiveShadowingEnvCfg_PLAY",
        ),
        ("instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg", "G1ParkourEnvCfg_PLAY"),
    ],
)
def test_newton_task_cfgs_configure_instinct_visualizer(module_name, cfg_name):
    cfg = select_backend(getattr(importlib.import_module(module_name), cfg_name)(), "newton")

    assert cfg.sim.default_visualizer_cfg is None
    visualizer_cfg = cfg.sim.visualizer_cfgs[0]
    assert isinstance(visualizer_cfg, InstinctNewtonVisualizerCfg)
    backend = visualizer_cfg.backend
    assert isinstance(backend, InstinctNewtonBackendCfg)
    assert backend.visualizer_type == "newton_gl"
    assert backend.follow_body is True
    assert string_to_callable(str(backend.class_type)) is InstinctNewtonGLVisualizer
