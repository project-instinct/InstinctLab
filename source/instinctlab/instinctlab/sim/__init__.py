# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""InstinctLab extensions to Isaac Lab sim utilities."""

from .spawners import UrdfFileCfg

__all__ = ["UrdfFileCfg"]

# MeshFileCfg was removed: use ``isaaclab.sim.MeshFileCfg``.
