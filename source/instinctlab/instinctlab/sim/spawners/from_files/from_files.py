# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Spawn functions for URDF files."""

from __future__ import annotations

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file
from isaaclab.sim.utils import clone

from . import asset_cache

if TYPE_CHECKING:
    from . import from_files_cfg


def _capsule_extent(radius: float, height: float, axis: str) -> list[Gf.Vec3f]:
    """Return the USD extent for a capsule whose cylinder section has the given length."""
    half_length = 0.5 * height + radius
    half_extents = {
        "X": (half_length, radius, radius),
        "Y": (radius, half_length, radius),
        "Z": (radius, radius, half_length),
    }
    try:
        x, y, z = half_extents[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported capsule axis: {axis}") from exc
    return [Gf.Vec3f(-x, -y, -z), Gf.Vec3f(x, y, z)]


def _replace_collision_cylinders_with_capsules(stage: Usd.Stage) -> int:
    """Convert importer-generated collision cylinders to capsules in-place."""
    cylinders = [
        prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Cylinder) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    for prim in cylinders:
        cylinder = UsdGeom.Cylinder(prim)
        radius = float(cylinder.GetRadiusAttr().Get())
        height = float(cylinder.GetHeightAttr().Get())
        axis = str(cylinder.GetAxisAttr().Get())

        # Retyping keeps applied schemas and authored relationships on the
        # prim while changing only the concrete geometry schema.
        if not prim.SetTypeName("Capsule"):
            raise RuntimeError(f"Failed to change cylinder type at '{prim.GetPath()}'.")

        capsule = UsdGeom.Capsule(prim)
        capsule.GetRadiusAttr().Set(radius)
        capsule.GetHeightAttr().Set(height)
        capsule.GetAxisAttr().Set(axis)
        capsule.GetExtentAttr().Set(_capsule_extent(radius, height, axis))

    return len(cylinders)


def _flatten_urdf_geometry(usd_path: str, replace_cylinders_with_capsules: bool = False) -> str:
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

    joint_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]

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

    # The importer also keeps every joint in a separate Physics scope.  Move
    # each joint below its flattened body0 link, matching the official Newton
    # USD layout, and record the old-to-new joint path for relationship remaps.
    joint_path_map: dict[str, str] = {}
    for prim in joint_prims:
        old_joint_path = str(prim.GetPath())
        body0_relationship = prim.GetRelationship("physics:body0")
        body0_targets = [str(target) for target in body0_relationship.GetTargets()]
        if len(body0_targets) != 1:
            raise RuntimeError(f"Joint '{old_joint_path}' must have exactly one physics:body0 target.")
        body0_target = body0_targets[0]
        body0_target = path_map.get(body0_target, body0_target)
        new_parent = Sdf.Path(body0_target)
        new_joint_path = str(new_parent.AppendChild(Sdf.Path(old_joint_path).name))
        joint_path_map[old_joint_path] = new_joint_path

    # Sdf namespace edits do not rewrite relationship targets in the same
    # layer.  Remap both rigid-body and joint targets in every relationship,
    # including ``isaac:physics:robotJoints`` and any actuator target.
    target_path_map = {**path_map, **joint_path_map}
    ordered_old_paths = sorted(target_path_map, key=len, reverse=True)

    def remap_target(target: str) -> Sdf.Path:
        for old_path in ordered_old_paths:
            if target == old_path:
                return Sdf.Path(target_path_map[old_path])
            if target.startswith(f"{old_path}/"):
                return Sdf.Path(f"{target_path_map[old_path]}{target[len(old_path):]}")
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

    # Reparent deepest joints first, in case the converter ever nests joints.
    for old_joint_path, new_joint_path in sorted(joint_path_map.items(), key=lambda item: len(item[0]), reverse=True):
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(
            Sdf.NamespaceEdit.Reparent(
                Sdf.Path(old_joint_path),
                Sdf.Path(new_joint_path).GetParentPath(),
                0,
            )
        )
        if not layer.Apply(edit):
            raise RuntimeError(f"Failed to reparent joint '{old_joint_path}' to '{new_joint_path}'")

    physics_prim = stage.GetPrimAtPath(root_path.AppendChild("Physics"))
    if physics_prim and not physics_prim.GetChildren():
        stage.RemovePrim(physics_prim.GetPath())

    if replace_cylinders_with_capsules:
        _replace_collision_cylinders_with_capsules(stage)

    flat_dir = os.path.join(os.path.dirname(usd_path), "flattened")
    os.makedirs(flat_dir, exist_ok=True)
    flat_usd_path = os.path.join(flat_dir, os.path.basename(usd_path))
    layer.Export(flat_usd_path)
    return flat_usd_path


def _check_urdf_inertia(usd_path: str, urdf_path: str, rtol: float = 1e-4) -> None:
    """Raise if a converted link's inertia differs from the URDF tensor.

    The simulator rebuilds the tensor as ``R @ diag(diagonalInertia) @ R.T`` with ``R`` from ``physics:principalAxes``.
    The Isaac Sim 6.0.1 importer wrote ``R.T`` instead, which silently rotated every link inertia. Links that absorb
    fixed-joint children are skipped, since the importer merges the children's inertia into them.
    """
    root = ET.parse(urdf_path).getroot()
    fixed_parent = {
        joint.find("child").get("link"): joint.find("parent").get("link")
        for joint in root.iter("joint")
        if joint.get("type") == "fixed"
    }
    urdf_inertia: dict[str, np.ndarray] = {}
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None or inertial.find("inertia") is None:
            continue
        a = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
        tensor = np.array(
            [
                [a["ixx"], a["ixy"], a["ixz"]],
                [a["ixy"], a["iyy"], a["iyz"]],
                [a["ixz"], a["iyz"], a["izz"]],
            ]
        )
        origin = inertial.find("origin")
        roll, pitch, yaw = (float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split())
        # URDF rpy is the fixed-axis rotation Rz(yaw) @ Ry(pitch) @ Rx(roll) of the inertial frame in the link frame
        cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        rpy = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rpy = rpy @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        urdf_inertia[link.get("name")] = rpy @ tensor @ rpy.T
    merged = set()
    for child in fixed_parent:
        if child in urdf_inertia:
            link = child
            while link in fixed_parent:
                link = fixed_parent[link]
            merged.add(link)

    mismatches = []
    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        name = prim.GetName()
        if not prim.HasAPI(UsdPhysics.MassAPI) or name not in urdf_inertia or name in merged:
            continue
        axes = prim.GetAttribute("physics:principalAxes").Get()
        moments = np.array(prim.GetAttribute("physics:diagonalInertia").Get(), dtype=float)
        w, x, y, z = np.array([axes.GetReal(), *axes.GetImaginary()]) / axes.GetLength()
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        rebuilt = rotation @ np.diag(moments) @ rotation.T
        error = np.abs(rebuilt - urdf_inertia[name]).max() / max(np.abs(moments).max(), 1e-12)
        if error > rtol:
            mismatches.append(f"{name} (relative error {error:.2e})")
    if mismatches:
        raise RuntimeError(
            f"Converted inertia of '{usd_path}' does not match '{urdf_path}' for: {', '.join(mismatches)}."
            " Check physics:principalAxes written by the URDF importer."
        )


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an articulation from a URDF with a flattened rigid-link tree."""
    if getattr(cfg, "asset_cache_enabled", True) and asset_cache.has_valid_asset_cache(cfg):
        cached_usd_path = asset_cache.asset_cached_usd_path(cfg)
        return spawn_from_usd_file(prim_path, str(cached_usd_path), cfg, translation, orientation, **kwargs)

    # Isaac Sim's importer 3.0 removed this option. Keep the upstream converter
    # cache and warning path clean, then apply the legacy geometry mapping in
    # our post-processing step.
    converter_cfg = cfg.replace(replace_cylinders_with_capsules=False)
    urdf_loader = converters.UrdfConverter(converter_cfg)
    flat_usd_path = _flatten_urdf_geometry(urdf_loader.usd_path, cfg.replace_cylinders_with_capsules)
    _check_urdf_inertia(flat_usd_path, cfg.asset_path)
    if getattr(cfg, "asset_cache_enabled", True):
        flat_usd_path = str(asset_cache.publish_asset_cache(cfg, flat_usd_path))
    return spawn_from_usd_file(prim_path, flat_usd_path, cfg, translation, orientation, **kwargs)


# spawn_from_mesh was removed: ``isaaclab.sim.spawners.from_files.spawn_from_mesh`` spawns mesh files and
# in-memory (trimesh) meshes.
