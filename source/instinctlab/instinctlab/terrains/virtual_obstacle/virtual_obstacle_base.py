from __future__ import annotations

import torch
import trimesh
from abc import ABC, abstractmethod
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.warp import ProxyArray


@configclass
class VirtualObstacleCfg:
    """Configuration for a virtual obstacle."""

    class_type: type = MISSING
    """The class to use for the virtual obstacle."""

    visualizer: VisualizationMarkersCfg = MISSING
    """The visualizer configuration for the virtual obstacle."""


class VirtualObstacleBase(ABC):
    def __init__(self, cfg: VirtualObstacleCfg):
        self.cfg = cfg

    @abstractmethod
    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        """Generate the virtual obstacle mesh based on the provided terrain mesh.
        NOTE: This interface might be updated in the future to support more complex generation logic.

        Args:
            mesh (trimesh.Trimesh): The terrain mesh to generate the virtual obstacle from.

        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def disable_visualizer(self) -> None:
        """Disable the visualizer for the virtual obstacle if there is one."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    """
    Operations only after being generated.
    If called before generation, it should skip and print a warning.
    """

    @abstractmethod
    def visualize(self):
        """Visualize the virtual obstacle."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def get_points_penetration_offset(self, points: ProxyArray) -> ProxyArray:
        """Get the penetration offset for the given points.

        Args:
            points: Warp-first points with ``wp.vec3f`` elements.

        Returns:
            Warp-first penetration offsets with the same shape as ``points``.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
