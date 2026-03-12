# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Spawn functions for mesh files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd

from isaaclab.sim import converters

# Import the private helper from IsaacLab - we cannot add spawn_from_mesh to IsaacLab
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone

if TYPE_CHECKING:
    from . import from_files_cfg


@clone
def spawn_from_mesh(
    prim_path: str,
    cfg: from_files_cfg.MeshFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a mesh file (OBJ, STL, FBX) and override the settings with the given config.

    It uses the :class:`MeshConverter` class to create a USD file from the mesh. This file is then
    imported at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new
    prim or throw an error that the prim already exists. Instead, it just takes the existing prim
    and overrides the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single config and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex
            pattern, then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim.
            Defaults to None, in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the mesh file does not exist at the given path.
    """
    mesh_converter = converters.MeshConverter(cfg)
    spawn_cfg = cfg if cfg.apply_collision_props_at_spawn else cfg.replace(collision_props=None)
    return _spawn_from_usd_file(prim_path, mesh_converter.usd_path, spawn_cfg, translation, orientation, **kwargs)
