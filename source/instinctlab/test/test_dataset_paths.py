"""Tests for the ``~`` expansion at the dataset path consumers.

The motion buffers and the mesh terrains read their dataset paths when they build a file list or a terrain, not in
``__post_init__``, so an override applied after construction (``env.<path>=`` on the CLI, a Hydra override) still
resolves. These tests drive those consumers with stub buffers and never launch a simulator.
"""

from __future__ import annotations

import numpy as np
import trimesh
import yaml
from types import SimpleNamespace

import pytest

from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
from instinctlab.motion_reference.motion_files.terrain_motion import TerrainMotion
from instinctlab.terrains.trimesh import mesh_terrains
from instinctlab.terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg


class _AmassStub(AmassMotion):
    """``AmassMotion`` without the simulation-side buffers the file scan does not touch."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._weights = None

    def _init_motion_weights(self, motion_weights=None):
        self._weights = motion_weights


class _TerrainStub(TerrainMotion):
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer_device = "cpu"
        self.output_device = "cpu"


def _amass_cfg(tmp_path, **kwargs):
    cfg = SimpleNamespace(
        path=str(tmp_path / "motions"),
        filtered_motion_selection_filepath=None,
        subset_selection=None,
        supported_file_endings=["poses.npz"],
        motion_bin_length_s=None,
        motion_start_from_middle_range=(0.0, 0.0),
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_amass_walks_an_expanded_dataset_path(home):
    motion = home / "motions"
    motion.mkdir()
    (motion / "walk_poses.npz").write_bytes(b"")
    stub = _AmassStub(_amass_cfg(home, path="~/motions"))
    stub._refresh_motion_file_list()
    assert stub._all_motion_files == [str(motion / "walk_poses.npz")]


def test_amass_expands_the_selection_file_and_joins_the_dataset_path(home):
    (home / "motions").mkdir()
    selection = home / "selection.yaml"
    selection.write_text(yaml.safe_dump({"selected_files": ["walk_poses.npz"], "motion_weights": [2.0]}))
    stub = _AmassStub(_amass_cfg(home, filtered_motion_selection_filepath="~/selection.yaml"))
    stub._refresh_motion_file_list()
    assert stub._all_motion_files == [str(home / "motions" / "walk_poses.npz")]
    assert stub._weights == [2.0]


def test_terrain_motion_expands_metadata_and_dataset_path(home):
    (home / "motions").mkdir()
    metadata = home / "metadata.yaml"
    metadata.write_text(
        yaml.safe_dump(
            {
                "terrains": [{"terrain_id": 0, "terrain_file": "terrain.stl"}],
                "motion_files": [{"terrain_id": 0, "motion_file": "walk_poses.npz", "weight": 1.0}],
            }
        )
    )
    cfg = _amass_cfg(home, metadata_yaml="~/metadata.yaml", max_origins_per_motion=4)
    stub = _TerrainStub(cfg)
    stub._refresh_motion_file_list()
    assert stub._all_motion_files == [str(home / "motions" / "walk_poses.npz")]
    assert stub.yaml_data["terrains"][0]["terrain_id"] == 0


def _terrain_cfg(home) -> MotionMatchedTerrainCfg:
    cfg = MotionMatchedTerrainCfg()
    cfg.path = "~/dataset"
    cfg.metadata_yaml = "~/metadata.yaml"
    cfg.size = (2.0, 2.0)
    return cfg


def test_mesh_terrain_expands_metadata_and_dataset_path(home):
    dataset = home / "dataset"
    dataset.mkdir()
    terrain = trimesh.creation.box(extents=(2.0, 2.0, 0.2))
    terrain.export(str(dataset / "terrain.stl"))
    metadata = home / "metadata.yaml"
    metadata.write_text(
        yaml.safe_dump(
            {
                "terrains": [{"terrain_id": 0, "terrain_file": "terrain.stl"}],
                "motion_files": [{"terrain_id": 0, "motion_file": "walk_poses.npz", "weight": 1.0}],
            }
        )
    )
    cfg = _terrain_cfg(home)
    mesh, origin = mesh_terrains.motion_matched_terrain(difficulty=0.0, cfg=cfg)
    assert isinstance(mesh, trimesh.Trimesh)
    assert np.asarray(origin).shape == (3,)
