# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Configuration for spawning assets from URDF files."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg as IsaacLabUrdfFileCfg
from isaaclab.utils.configclass import configclass


@configclass
class UrdfFileCfg(IsaacLabUrdfFileCfg):
    """URDF file to spawn a flattened InstinctLab articulation from.

    Isaac Sim's URDF importer 3.0 wraps the rigid-link tree in a ``Geometry``
    scope.  Isaac Lab's built-in contact and ray-caster sensors address rigid
    bodies directly below the articulation prim, so InstinctLab post-processes
    the generated USD and moves that link tree one level up.
    """

    func: Callable | str = "{DIR}.from_files:spawn_from_urdf"

    # Disable the importer's layered asset transformation.  The transformed
    # output duplicates every link as ``Geometry/<link>/<link>``; the raw
    # importer output is a single kinematic tree and is what we flatten below.
    run_asset_transformer: bool = False

    # The multi-physics pass adds MuJoCo actuator prims targeted at the
    # importer's original ``/g1/Physics/<joint>`` paths.  We already relocate
    # those joints while flattening the articulation, so avoid creating the
    # extra target bookkeeping unless a task explicitly opts in.
    run_multi_physics_conversion: bool = False

    asset_cache_enabled: bool = True
    """Cache the flattened USD keyed by the URDF, meshes, converter config, and postprocessor code.

    This makes generated USD a derived immutable artifact while ``UrdfFileCfg`` remains the public
    source interface for task configs.
    """


# MeshFileCfg was removed: mesh files and in-memory meshes spawn through ``isaaclab.sim.MeshFileCfg``.
