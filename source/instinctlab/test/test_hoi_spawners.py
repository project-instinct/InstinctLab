"""Tests for the perceptive HOI object spawners and their saved-config handling.

The HOI objects spawn as ``isaaclab.sim.MeshFileCfg`` rigid objects from in-memory triangle data, so the tests check
the spawned configuration, that the env cfg still validates, and that ``params/env.yaml`` masking keeps the config
loadable. They instantiate configclasses and never launch a simulator.
"""

from __future__ import annotations

import numpy as np
import yaml

import pytest

from isaaclab.sim import MeshFileCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsMeshCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils.io import dump_yaml

from instinctlab.tasks.shadowing.perceptive_hoi.config.g1.perceptive_shadowing_cfg import (
    MESH_FILE_PATHS,
    MESH_FILE_SCALES,
    G1PerceptiveHoiShadowingEnvCfg,
)
from instinctlab.utils import instinct_rl_cli as cli

OBJECT_NAMES = list(MESH_FILE_PATHS)


@pytest.fixture(scope="module")
def env_cfg():
    return G1PerceptiveHoiShadowingEnvCfg()


@pytest.mark.parametrize("object_name", OBJECT_NAMES)
def test_objects_spawn_as_in_memory_mesh_rigid_objects(env_cfg, object_name):
    cfg = getattr(env_cfg.scene, object_name)
    spawn = cfg.spawn
    assert isinstance(spawn, MeshFileCfg)
    assert isinstance(spawn.mesh, MeshFileCfg.TriangleMeshCfg)
    assert spawn.mesh.vertices.shape[1] == 3
    assert spawn.mesh.faces.shape[1] == 3
    assert spawn.scale == MESH_FILE_SCALES[object_name]
    assert isinstance(spawn.mass_props, MassCfg) and spawn.mass_props.mass == 1.0
    assert isinstance(spawn.collision_props, UsdPhysicsCollisionCfg)
    assert spawn.rigid_props == [UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]
    # ``none`` keeps the authored triangles as the collider, like the removed InstinctLab spawner did.
    assert isinstance(spawn.mesh_collision_props, UsdPhysicsMeshCollisionCfg)
    assert spawn.mesh_collision_props.mesh_approximation_name == "none"
    assert cfg.prim_path == f"/World/envs/env_[^/]+/{object_name}"


def test_env_cfg_validates(env_cfg):
    """The mesh payload must be traversable: ``ManagerBasedEnv.__init__`` validates the whole cfg tree."""
    assert env_cfg.validate() == []


def test_depth_camera_targets_the_object_prims(env_cfg):
    targets = [
        target if isinstance(target, str) else target.prim_expr for target in env_cfg.scene.camera.mesh_prim_paths
    ]
    for object_name in OBJECT_NAMES:
        assert "/World/envs/env_[^/]+/" + object_name in targets


def test_spawned_cfg_has_no_deprecated_schema_cfgs(env_cfg):
    """The ported HOI task uses main's non-deprecated schema cfgs."""
    from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG, G1_REFERENCE_CFG

    for spawn in (
        env_cfg.scene.robot.spawn,
        G1_29DOF_TORSOBASE_POPSICLE_CFG.spawn,
        G1_REFERENCE_CFG.spawn,
        env_cfg.scene.terrain.physics_material,
    ):
        dump = yaml.dump(cli.env_cfg_for_dump({"spawn": spawn}))
        for legacy in (
            "JointDriveBaseCfg",
            "RigidBodyPropertiesCfg",
            "ArticulationRootPropertiesCfg",
            "CollisionBaseCfg",
            "CollisionPropertiesCfg",
            "MassPropertiesCfg",
            "RigidBodyMaterialCfg",
            "MeshCollisionBaseCfg",
        ):
            assert legacy not in dump


def test_env_yaml_masks_and_restores_the_mesh_payload(env_cfg, tmp_path):
    path = tmp_path / "env.yaml"
    dump_yaml(str(path), cli.env_cfg_for_dump(env_cfg))
    assert cli.IN_MEMORY_MESH_PLACEHOLDER in path.read_text()
    # the live config keeps its triangles, so the spawner still has a mesh after the dump
    assert np.asarray(env_cfg.scene.floorlamp.spawn.mesh.vertices).shape[1] == 3
    # and replaying the dump drops the placeholder instead of overwriting the task's mesh
    restored = cli._drop_in_memory_mesh_placeholders(cli.load_saved_yaml(str(path)))
    assert cli.IN_MEMORY_MESH_PLACEHOLDER not in yaml.dump(restored)
    restored_mesh = restored["scene"]["floorlamp"]["spawn"]["mesh"]
    assert "vertices" not in restored_mesh and "faces" not in restored_mesh


def test_hydra_arguments_are_rejected_for_in_memory_meshes(env_cfg):
    cli._reject_in_memory_meshes_with_hydra("T", ["env.scene.num_envs=8"], env_cfg)
    with pytest.raises(ValueError, match="in-memory meshes"):
        cli._reject_in_memory_meshes_with_hydra("T", ["seed=0"], env_cfg)
