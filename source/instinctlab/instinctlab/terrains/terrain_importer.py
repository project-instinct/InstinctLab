from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator
from isaaclab.terrains import TerrainImporter as TerrainImporterBase
from isaaclab.utils.timer import Timer

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg
    from .virtual_obstacle import VirtualObstacleBase


class TerrainImporter(TerrainImporterBase):
    def __init__(self, cfg: TerrainImporterCfg):
        self._virtual_obstacles = {}
        for name, virtual_obstacle_cfg in cfg.virtual_obstacles.items():
            if virtual_obstacle_cfg is None:
                continue
            virtual_obstacle = virtual_obstacle_cfg.class_type(virtual_obstacle_cfg)
            self._virtual_obstacles[name] = virtual_obstacle

        if cfg.terrain_type == "hacked_generator":
            self._hacked_terrain_type = "hacked_generator"
            cfg.terrain_type = "plane"
        super().__init__(cfg)

    @property
    def virtual_obstacles(self) -> dict[str, VirtualObstacleBase]:
        """Get the virtual obstacles representing the edges.
        TODO: Make the returned value more general.
        """
        # still pointing the same VirtualObstacleBase objects but the dict is a copy.
        return self._virtual_obstacles.copy()

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg] | None:
        """Get the specific configurations for all subterrains."""
        # This is a placeholder. The actual implementation should return the specific configurations.
        return (
            self.terrain_generator.subterrain_specific_cfgs
            if hasattr(self, "terrain_generator") and hasattr(self.terrain_generator, "subterrain_specific_cfgs")
            else None
        )

    """
    Operations - Import.
    """

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Import a mesh into the simulation.
        NOTE: By designing this interface, IsaacLab's terrain importer calls import_mesh only once when startup.

        Args:
            name: The name of the mesh.
            mesh: The trimesh object to import.
        """
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())  # remove duplicate faces
        mesh.remove_unreferenced_vertices()
        # Generate virtual obstacles based on the imported mesh.
        # NOTE: generate virtual obstacle first because it might modify the mesh.
        for name, virtual_obstacle in self._virtual_obstacles.items():
            with Timer(f"Generate virtual obstacle {name}"):
                virtual_obstacle.generate(mesh, device=self.device)

        super().import_mesh(name, mesh)

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """
        ## NOTE
        This is a hack to fit the self-defined tasks and guide the control flow to a self-defined implementation.
        In this case, we won't need to change the code of IsaacLab.
        """
        if getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            self.terrain_generator = getattr(
                self.cfg.terrain_generator,
                "class_type",
                TerrainGenerator,
            )(cfg=self.cfg.terrain_generator, device=self.device)
            self.import_mesh("terrain", self.terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(self.terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = self.terrain_generator.flat_patches
        else:
            # Call the base class implementation.
            super().import_ground_plane(name, size)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization flag.

        Args:
            vis: True to enable debug visualization, False to disable.
        """
        results = super().set_debug_vis(debug_vis)

        for name, virtual_obstacle in self._virtual_obstacles.items():
            if debug_vis:
                virtual_obstacle.visualize()
            else:
                virtual_obstacle.disable_visualizer()

        return results

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the environment origins.

        Args:
            origins: The origins of the environments. Shape is (num_envs, 3).
        """
        if origins is None and getattr(self, "_hacked_terrain_type", None) == "hacked_generator":
            # In case of None override, we don't need to do anything
            pass
        else:
            return super().configure_env_origins(origins)
