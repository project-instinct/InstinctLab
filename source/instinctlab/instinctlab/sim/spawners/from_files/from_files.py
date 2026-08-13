# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Spawn functions for standalone mesh files."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone

if TYPE_CHECKING:
    from . import from_files_cfg


def _flatten_urdf_geometry(usd_path: str) -> str:
    """Flatten the raw URDF importer's rigid-link tree.

    The flattened USD is exported beside the converter output so the original
    importer cache remains untouched.
    """
    stage = Usd.Stage.Open(usd_path)
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError(f"Generated URDF USD has no default prim: {usd_path}")

    geometry_prim = stage.GetPrimAtPath(root_prim.GetPath().AppendChild("Geometry"))
    if not geometry_prim:
        return usd_path

    root_path = root_prim.GetPath()
    geometry_path = geometry_prim.GetPath()
    layer = stage.GetRootLayer()

    rigid_prims = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetPath().HasPrefix(geometry_path)
    ]
    if not rigid_prims:
        raise RuntimeError(f"No rigid bodies found below '{geometry_path}'")

    # Record each rigid body's world transform before changing parents.  The
    # importer writes link transforms as parent-relative xformOps, so moving a
    # link to the asset root requires baking that accumulated transform into
    # the moved prim.
    world_transforms: dict[str, Gf.Transform] = {}
    for prim in rigid_prims:
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        transform = Gf.Transform()
        transform.SetMatrix(matrix)
        world_transforms[str(prim.GetPath())] = transform

    path_map = {old_path: str(root_path.AppendChild(Sdf.Path(old_path).name)) for old_path in world_transforms}

    # Move deepest links first so a parent is still at its original path when
    # its child is reparented.
    for prim in sorted(rigid_prims, key=lambda prim: len(str(prim.GetPath())), reverse=True):
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), root_path, 0))
        if not layer.Apply(edit):
            raise RuntimeError(f"Failed to reparent rigid body '{prim.GetPath()}' to '{root_path}'")

    stage.RemovePrim(geometry_path)

    for old_path, transform in world_transforms.items():
        prim = stage.GetPrimAtPath(path_map[old_path])
        if not prim:
            raise RuntimeError(f"Flattened rigid body not found at '{path_map[old_path]}'")
        xformable = UsdGeom.Xformable(prim)
        if not xformable.GetOrderedXformOps():
            # The articulation root often has no xformOps and is already
            # identity under the asset prim.
            continue
        prim.GetAttribute("xformOp:translate").Set(transform.GetTranslation())
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(transform.GetRotation().GetQuat()))
        prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*transform.GetScale()))

    # Sdf namespace edits do not rewrite relationship targets in the same
    # layer.  Joints still point at the original nested link paths, so remap
    # every target using the old-to-new rigid-body path table.
    ordered_old_paths = sorted(path_map, key=len, reverse=True)

    def remap_target(target: str) -> Sdf.Path:
        for old_path in ordered_old_paths:
            if target == old_path:
                return Sdf.Path(path_map[old_path])
            if target.startswith(f"{old_path}/"):
                return Sdf.Path(f"{path_map[old_path]}{target[len(old_path):]}")
        return Sdf.Path(target)

    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            targets = [str(target) for target in relationship.GetTargets()]
            if not any(
                any(target == old_path or target.startswith(f"{old_path}/") for old_path in ordered_old_paths)
                for target in targets
            ):
                continue
            relationship.SetTargets([remap_target(target) for target in targets])

    flat_dir = os.path.join(os.path.dirname(usd_path), "flattened")
    os.makedirs(flat_dir, exist_ok=True)
    flat_usd_path = os.path.join(flat_dir, os.path.basename(usd_path))
    layer.Export(flat_usd_path)
    return flat_usd_path


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an articulation from a URDF with a flattened rigid-link tree."""
    urdf_loader = converters.UrdfConverter(cfg)
    flat_usd_path = _flatten_urdf_geometry(urdf_loader.usd_path)
    return _spawn_from_usd_file(prim_path, flat_usd_path, cfg, translation, orientation, **kwargs)


@clone
def spawn_from_mesh(
    prim_path: str,
    cfg: from_files_cfg.MeshFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a rigid object from a standalone mesh file."""
    mesh_converter = converters.MeshConverter(cfg)
    spawn_cfg = cfg if cfg.apply_collision_props_at_spawn else cfg.replace(collision_props=None)
    return _spawn_from_usd_file(prim_path, mesh_converter.usd_path, spawn_cfg, translation, orientation, **kwargs)
