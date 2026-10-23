# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Sub-module for spawning assets from mesh and URDF files."""

from .from_files_cfg import UrdfFileCfg

__all__ = ["UrdfFileCfg"]

# MeshFileCfg was removed: use ``isaaclab.sim.MeshFileCfg``.
