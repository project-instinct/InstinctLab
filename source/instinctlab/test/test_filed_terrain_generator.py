"""Tests for :class:`instinctlab.terrains.terrain_generator.FiledTerrainGenerator`.

``subterrain_specific_cfgs[row * num_cols + col]`` must be the config of the sub-terrain at (row, col) in both
generation modes: Isaac Lab builds curriculum terrains column by column and random terrains row by row. Flat
sub-terrains keep the generation cheap; no simulator is launched.
"""

from __future__ import annotations

import pytest

from isaaclab.terrains import MeshPlaneTerrainCfg

from instinctlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg


@pytest.mark.parametrize("curriculum", [True, False])
def test_subterrain_cfgs_are_indexed_by_grid_position(curriculum):
    num_rows, num_cols = 5, 3
    cfg = FiledTerrainGeneratorCfg(
        size=(2.0, 2.0),
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum=curriculum,
        use_cache=False,
        sub_terrains={"plane": MeshPlaneTerrainCfg(proportion=1.0)},
    )
    generator = FiledTerrainGenerator(cfg)
    cfgs = generator.subterrain_specific_cfgs
    assert len(cfgs) == num_rows * num_cols
    for row in range(num_rows):
        for col in range(num_cols):
            difficulty = cfgs[row * num_cols + col].difficulty
            assert generator.get_subterrain_cfg(row, col) is cfgs[row * num_cols + col]
            if curriculum:
                # curriculum difficulty of row r lies in [r, r + 1) / num_rows
                assert row / num_rows <= difficulty < (row + 1) / num_rows
