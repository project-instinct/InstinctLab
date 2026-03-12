# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Configuration for spawning assets from mesh files."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.utils import configclass

from . import from_files


@configclass
class MeshFileCfg(converters.MeshConverterCfg, FileCfg):
    """Mesh file (OBJ, STL, FBX) to spawn asset from.

    It uses the :class:`MeshConverter` class to create a USD file from the mesh and spawns the
    imported USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified
    by specifying the respective properties in the configuration class.

    See :meth:`spawn_from_mesh` for more information.

    TODO: Typical ___FileCfg inherit FileCfg before converters.___ConverterCfg, which I don't think through here.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

    **Collision and instancing:**

    The MeshConverter bakes :attr:`collision_props` into the USD during conversion. By default,
    :attr:`apply_collision_props_at_spawn` is False, so spawn-time modification is skipped (avoids
    warning when geometry is instanceable). Set it to True when you need per-spawn collision overrides;
    this requires :attr:`make_instanceable` to be False (auto-resolved in :meth:`__post_init__`).
    """

    func: Callable = from_files.spawn_from_mesh

    apply_collision_props_at_spawn: bool = False
    """Whether to call modify_collision_properties at spawn time. Defaults to False.

    When False, collision is only applied during MeshConverter (baked into USD). Spawn-time modify
    is skipped, avoiding the warning when geometry is instanceable. When True, collision props
    are applied again at spawn, allowing per-instance override. Requires make_instanceable=False.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.apply_collision_props_at_spawn and self.make_instanceable:
            object.__setattr__(self, "make_instanceable", False)
