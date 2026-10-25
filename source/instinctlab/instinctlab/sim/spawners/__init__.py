# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""InstinctLab extensions to Isaac Lab spawners."""

from .from_files import UrdfFileCfg

__all__ = ["UrdfFileCfg"]

# MeshFileCfg was removed: use ``isaaclab.sim.MeshFileCfg``.
