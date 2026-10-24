"""Tests for :mod:`instinctlab.sim.spawners.from_files.asset_cache`.

The cache is keyed by the URDF, its meshes, the converter config, the postprocessor code and the Isaac Lab/Isaac Sim
versions, so a URDF importer change rebuilds it. The tests write a tiny URDF and never launch a simulator, so they
cover the cold and warm paths without the URDF converter.
"""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from instinctlab.sim.spawners.from_files import asset_cache
from instinctlab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg

URDF = """<robot name="r">
  <link name="base">
    <inertial><mass value="1.0"/><inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/></inertial>
    <collision><geometry><mesh filename="base.stl"/></geometry></collision>
  </link>
</robot>
"""


@pytest.fixture
def urdf_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTINCTLAB_ASSET_CACHE", str(tmp_path / "cache"))
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "robot.urdf").write_text(URDF)
    (tmp_path / "assets" / "base.stl").write_text("solid base\nendsolid base\n")
    return UrdfFileCfg(asset_path=str(tmp_path / "assets" / "robot.urdf"), fix_base=True)


def _publish(cfg, tmp_path):
    # ``spawn_from_urdf`` publishes the flattened URDF USD, which carries the source URDF's name.
    flat = tmp_path / f"{os.path.basename(cfg.asset_path).split('.')[0]}.usda"
    flat.write_text("#usda 1.0\n")
    return asset_cache.publish_asset_cache(cfg, flat)


def test_cache_lives_under_the_configured_root(urdf_cfg, tmp_path):
    assert asset_cache.asset_cache_root() == (tmp_path / "cache").resolve()
    assert asset_cache.asset_cache_dir(urdf_cfg).is_relative_to(asset_cache.asset_cache_root())


def test_digest_is_stable_and_tracks_the_inputs(urdf_cfg, tmp_path):
    digest = asset_cache.asset_cache_digest(urdf_cfg)
    assert digest == asset_cache.asset_cache_digest(urdf_cfg)
    # The spawner's own postprocessing (capsule replacement) changes the derived USD, so it changes the cache key.
    assert (
        asset_cache.asset_cache_digest(
            urdf_cfg.replace(replace_cylinders_with_capsules=not urdf_cfg.replace_cylinders_with_capsules)
        )
        != digest
    )
    flipped = urdf_cfg.replace(merge_fixed_joints=not urdf_cfg.merge_fixed_joints)
    assert asset_cache.asset_cache_digest(flipped) != digest
    # Touching the mesh invalidates the cache as well.
    (tmp_path / "assets" / "base.stl").write_text("solid base\nfacet vertex 0 0 0\nendsolid base\n")
    assert asset_cache.asset_cache_digest(urdf_cfg) != digest


def test_cold_then_warm(urdf_cfg, tmp_path, monkeypatch):
    # Without a reachable standalone importer, a cold cache forces Kit.
    monkeypatch.setattr("isaaclab.utils.version.standalone_importers_available", lambda: False)
    assert not asset_cache.has_valid_asset_cache(urdf_cfg)
    assert asset_cache.ensure_asset_cache_kit_args(
        SimpleNamespace(robot=SimpleNamespace(spawn=urdf_cfg)), SimpleNamespace(kit_args="", require_kit=False)
    )
    cached = _publish(urdf_cfg, tmp_path)
    assert cached.is_file()
    assert asset_cache.has_valid_asset_cache(urdf_cfg)
    assert asset_cache.asset_cached_usd_path(urdf_cfg) == cached
    # Warm: no rebuild, no forced Kit, and the manifest records the digest and the input hashes.
    assert not asset_cache.ensure_asset_cache_kit_args(
        SimpleNamespace(robot=SimpleNamespace(spawn=urdf_cfg)), SimpleNamespace(kit_args="", require_kit=False)
    )
    metadata = json.loads((cached.parent / "instinctlab_asset.json").read_text())
    assert metadata["asset_digest"] == asset_cache.asset_cache_digest(urdf_cfg)
    assert metadata["inputs"]["files"]["urdf"]
    assert metadata["inputs"]["files"]["meshes"] == {"base.stl": metadata["inputs"]["files"]["meshes"]["base.stl"]}


def test_edited_cache_is_rejected(urdf_cfg, tmp_path):
    cached = _publish(urdf_cfg, tmp_path)
    cached.write_text("#usda 1.0\n# edited\n")
    assert not asset_cache.has_valid_asset_cache(urdf_cfg)
    (cached.parent / "instinctlab_asset.json").unlink()
    assert not asset_cache.has_valid_asset_cache(urdf_cfg)


def test_uncached_cfgs_are_found_recursively(urdf_cfg):
    env_cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=urdf_cfg)), extras={})
    assert asset_cache.find_uncached_asset_cfgs(env_cfg) == [urdf_cfg]
    urdf_cfg.asset_cache_enabled = False
    assert asset_cache.find_uncached_asset_cfgs(env_cfg) == []


def test_cache_key_uses_the_installed_versions(urdf_cfg):
    inputs = asset_cache._asset_cache_inputs(urdf_cfg)
    assert inputs["isaaclab_version"] == asset_cache._package_version("isaaclab")
    assert inputs["isaacsim_version"] == asset_cache._package_version("isaacsim")
    assert inputs["postprocessor_version"] == asset_cache.ASSET_CACHE_POSTPROCESSOR_VERSION
    assert os.path.basename(inputs["asset_path"]) == "robot.urdf"


def test_cold_cache_stays_kitless_with_the_standalone_importer(urdf_cfg, monkeypatch):
    monkeypatch.setattr("isaaclab.utils.version.standalone_importers_available", lambda: True)
    monkeypatch.setitem(sys.modules, "isaacsim.asset.importer.utils.impl.physx_types", ModuleType("physx_types"))
    launcher_args = SimpleNamespace(kit_args="", require_kit=False)
    assert not asset_cache.has_valid_asset_cache(urdf_cfg)
    assert not asset_cache.ensure_asset_cache_kit_args(
        SimpleNamespace(robot=SimpleNamespace(spawn=urdf_cfg)), launcher_args
    )
    assert launcher_args == SimpleNamespace(kit_args="", require_kit=False)
