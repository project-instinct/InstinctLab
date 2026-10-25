from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):

        # Access the i-th row, j-th column subterrain config by
        # self._subterrain_specific_cfgs[i*num_cols + j]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        super().__init__(cfg, device)

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        """This function intercept the terrain mesh generation process and records the specific config
        for each subterrain.
        """
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        # >>> NOTE: This code snippet is copied from the super implementation because they copied the cfg
        # but we need to store the modified cfg for each subterrain.
        cfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # <<< NOTE
        self._subterrain_specific_cfgs.append(cfg)  # since in super function, cfg is a copy of the original config.

        return mesh, origin

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all subterrains."""
        return self._subterrain_specific_cfgs.copy()  # Return a copy to avoid external modification.

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a subterrain by its row and column index."""
        num_cols = self.cfg.num_cols
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy().tolist()  # Convert to list if it's a tensor.
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, int):
            return self._subterrain_specific_cfgs[idx] if 0 <= idx < len(self._subterrain_specific_cfgs) else None
