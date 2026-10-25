"""Real-backend tests for InstinctLab's OvPhysX components (volume points, motion-reference articulation access).

OvPhysX runs kitless, so these tests build a small G1 scene directly with :class:`SimulationContext`. The G1 URDF
converts through the Kit-free conversion on a cold asset cache. The grouped ray casters and cameras reuse Isaac Lab's
OvPhysX ray-caster classes and are exercised by the task snapshots and checkpoint plays instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import torch  # noqa: E402

import warp as wp  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG  # noqa: E402
from instinctlab.motion_reference.ovphysx import OvPhysxMotionReferenceManager  # noqa: E402
from instinctlab.sensors import Grid3dPointsGeneratorCfg, VolumePointsCfg  # noqa: E402
from instinctlab.sensors.volume_points.ovphysx import OvPhysxVolumePoints  # noqa: E402

NUM_ENVS = 4
POINTS = Grid3dPointsGeneratorCfg(
    x_min=-0.02, x_max=0.02, x_num=2, y_min=-0.02, y_max=0.02, y_num=2, z_min=0.0, z_max=0.0, z_num=1
)


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = G1_29DOF_TORSOBASE_POPSICLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    feet = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot", body_names_expr=".*_ankle_roll_link", points_generator=POINTS
    )
    knee = VolumePointsCfg(prim_path="{ENV_REGEX_NS}/Robot", body_names_expr="left_knee_link", points_generator=POINTS)


@pytest.fixture(scope="module")
def scene():
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device="cuda:0", dt=0.005)
    with build_simulation_context(device="cuda:0", sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_SceneCfg(num_envs=NUM_ENVS, env_spacing=2.5))
        sim.reset()
        for _ in range(5):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        yield sim, scene


def _step(sim, scene, steps=2):
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def _expected(scene, names):
    robot = scene["robot"]
    ids = [robot.body_names.index(name) for name in names]
    return robot.data.body_link_pos_w.torch[:, ids], robot.data.body_link_quat_w.torch[:, ids]


def test_volume_points_follow_their_own_links(scene):
    sim, scene = scene
    assert isinstance(scene["feet"], OvPhysxVolumePoints)
    _step(sim, scene)
    for name, bodies in (("feet", ["left_ankle_roll_link", "right_ankle_roll_link"]), ("knee", ["left_knee_link"])):
        sensor = scene[name]
        assert sensor.body_names == bodies
        pos, quat = _expected(scene, bodies)
        torch.testing.assert_close(sensor.data.pos_w.torch, pos)
        torch.testing.assert_close(sensor.data.quat_w.torch, quat)


@pytest.mark.parametrize(
    "env_ids",
    [
        slice(None),
        slice(1, 4, 2),
        torch.tensor([0, 2], dtype=torch.int32, device="cuda:0"),
        torch.tensor([1, 3], dtype=torch.int64, device="cuda:0"),
    ],
    ids=["all", "slice", "int32", "int64"],
)
def test_volume_points_reset_and_update(scene, env_ids):
    sim, scene = scene
    for name in ("feet", "knee"):
        scene[name].reset(env_ids)
    _step(sim, scene)
    pos, _ = _expected(scene, ["left_ankle_roll_link", "right_ankle_roll_link"])
    torch.testing.assert_close(scene["feet"].data.pos_w.torch, pos)
    assert scene["feet"].data.points_pos_w.torch.shape[:2] == (NUM_ENVS, 2)


def test_motion_reference_articulation_access(scene):
    sim, scene = scene
    robot = scene["robot"]
    # Only the articulation access is under test, so the manager skips its sensor setup (motion data, callbacks).
    manager = object.__new__(OvPhysxMotionReferenceManager)
    manager._initialize_handle = manager._invalidate_initialize_handle = manager._prim_deletion_handle = None
    manager._sim = sim
    manager._backend_manager = sim.backend_manager
    manager._view = manager._create_articulation_view("/World/envs/env_.*/Robot")

    assert manager.count == NUM_ENVS
    assert manager.max_dofs == robot.num_joints
    assert manager.joint_dof_names == robot.joint_names
    root = manager.get_root_transforms()
    torch.testing.assert_close(root[:, :3], robot.data.root_link_pos_w.torch)
    torch.testing.assert_close(root[:, 3:], robot.data.root_link_quat_w.torch)
    torch.testing.assert_close(manager.get_dof_positions(), robot.data.joint_pos.torch)
    limits = manager.get_dof_limits()
    assert limits.shape == (NUM_ENVS, robot.num_joints, 2)
    torch.testing.assert_close(limits.to(robot.device), robot.data.joint_pos_limits.torch)

    # The reference write path: state lands in the simulation for every environment.
    manager._reference_view = manager._view
    target_root = root.clone()
    target_root[:, 2] += 0.5
    target_joints = torch.zeros_like(manager.get_dof_positions()) + 0.1
    manager._write_reference_view_state(target_root, target_joints)
    torch.testing.assert_close(manager.get_root_transforms(), target_root)
    torch.testing.assert_close(manager.get_dof_positions(), target_joints)
    torch.testing.assert_close(manager.get_dof_velocities(), torch.zeros_like(target_joints))
    wp.synchronize()
