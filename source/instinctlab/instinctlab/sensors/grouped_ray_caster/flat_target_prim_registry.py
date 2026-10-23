"""Target resolution for the grouped ray-caster sensors.

Grouping is provided by Isaac Lab's per-environment multi-mesh tables: each configured target expression
contributes one row of mesh slots per environment, tracked targets are bound to their body through the
scene-data provider, and padded slots never produce a hit. This module only adapts target resolution where
InstinctLab's scenes differ from upstream's assumptions.
"""

from __future__ import annotations

import logging
import numpy as np
import re
import trimesh
from typing import TYPE_CHECKING

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import convert_to_warp_mesh

from instinctlab.utils.math import matrix_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg


logger = logging.getLogger(__name__)


class FlatTargetPrimRegistryMixin:
    """Resolve grouped ray-cast targets per environment, keeping articulation roots as one body each."""

    def _build_mesh_records(self, target_cfg, plan: cloner.ClonePlan | None):
        """Resolve one target expression and reject targets whose prim count differs per environment."""
        records_per_env = None
        if plan is not None and target_cfg.track_mesh_transforms:
            records_per_env = self._build_articulation_root_records(target_cfg, plan)
        if records_per_env is None:
            records_per_env = super()._build_mesh_records(target_cfg, plan)
        self._check_uniform_record_count(target_cfg, records_per_env)
        return records_per_env

    def _build_articulation_root_records(self, target_cfg, plan: cloner.ClonePlan):
        """Keep an articulation-root link as one tracked mesh target.

        The upstream multi-mesh builder interprets a prim with ``ArticulationRootAPI`` as a request for
        every rigid-body descendant. Importer 3 authors that API on the G1 torso link, but InstinctLab
        lists every robot link explicitly. Expanding the torso would therefore add the whole robot once
        and the other 29 links a second time.

        Returns:
            Per-environment mesh records, or None when the target matches no articulation root.
        """
        records_per_env = [[] for _ in range(self._num_envs)]
        found_articulation_root = False
        for source_root, destination_template, source_prims, env_ids in self._iter_plan_targets(target_cfg, plan):
            articulation_root_prims = [prim for prim in source_prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if not articulation_root_prims:
                continue
            if len(articulation_root_prims) != len(source_prims):
                raise RuntimeError(
                    f"Ray-cast target '{target_cfg.prim_expr}' mixes articulation-root and non-root prims."
                )

            found_articulation_root = True
            for source_prim in articulation_root_prims:
                if not source_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(
                        f"Articulation-root ray-cast target '{source_prim.GetPath()}' is not a rigid body."
                    )
                mesh_id = self._load_target_prim_warp_mesh(source_prim, target_cfg, reference_prim=source_prim)
                owner_suffix = cloner.path.relative_to(str(source_prim.GetPath()), source_root)
                if owner_suffix is None:
                    raise RuntimeError(
                        f"Tracked target owner '{source_prim.GetPath()}' is not under ClonePlan source root"
                        f" '{source_root}'."
                    )
                for env_id in env_ids:
                    body_path = destination_template.format(env_id) + owner_suffix
                    records_per_env[env_id].append((mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0), body_path))

        return records_per_env if found_articulation_root else None

    def _iter_plan_targets(self, target_cfg, plan: cloner.ClonePlan):
        """Yield the ClonePlan rows that own ``target_cfg.prim_expr``.

        This mirrors upstream's ClonePlan target resolution: only the shortest matched suffix is used, and
        each row reports the destination environment IDs whose instance text the expression selects.
        """
        sources = cloner.path.get_asset_prototype_paths(plan)
        templates, starts, worlds, world_starts = cloner.path.get_world_prototype_asset_templates(
            plan, include_world_indices=True
        )
        matches = []
        # Match populated world prototypes, excluding the shared-world slice.
        for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
            for index in range(*starts[group : group + 2]):
                if (matched := cloner.path.match(target_cfg.prim_expr, templates[index])) is not None:
                    matches.append((group, index, matched))
        suffix = min((matched.suffix for _, _, matched in matches), key=len, default=None)
        for group, index, matched in matches:
            if matched.suffix != suffix:
                continue
            env_ids = worlds[world_starts[group] : world_starts[group + 1]]
            env_ids = env_ids[
                resolve_matching_names(matched.instance, env_ids.astype(str), raise_when_no_match=False)[0]
            ]
            if not len(env_ids):
                continue
            source_root = sources[plan.topology.world_prototypes[index]]
            source_pattern = re.compile(source_root + suffix)
            source_prims = sim_utils.get_all_matching_child_prims(
                source_root, lambda prim: source_pattern.fullmatch(prim.GetPath().pathString) is not None
            )
            yield source_root, templates[index], source_prims, env_ids

    @staticmethod
    def _check_uniform_record_count(target_cfg, records_per_env: list[list[tuple]]) -> None:
        """Reject a target that resolves a different number of prims in different environments.

        An environment without any record is accepted: a static target may exist in only some prototypes.
        Ragged non-empty rows cannot be expressed by the per-environment mesh tables, which pad each target
        to its widest row, so they would silently stop hitting in the narrower environments.
        """
        counts = [len(records) for records in records_per_env]
        populated = {count for count in counts if count}
        if len(populated) > 1:
            raise RuntimeError(
                f"Ray-cast target '{target_cfg.prim_expr}' resolves a different number of prims per"
                f" environment: {counts}."
            )

    def _initialize_rays_impl(self) -> None:
        """Initialize the ray buffers and state the grouped sensor's mount rule.

        Upstream rejects a view count that differs from the environment count in ``_initialize_impl``; the
        grouped sensors keep their own check because it names the public contract they promise: one mounted
        body per environment, so a target group never aliases another environment's frame.
        """
        super()._initialize_rays_impl()
        if self._view_count != self._num_envs:
            raise RuntimeError(
                f"Grouped ray caster at '{self.cfg.prim_path}' has {self._view_count} ray batches for"
                f" {self._num_envs} environments; each instance must mount on exactly one body per"
                " environment."
            )

    def _load_target_prim_warp_mesh(
        self,
        target_prim: Usd.Prim,
        target_cfg: MultiMeshRayCasterCfg.RaycastTargetCfg,
        reference_prim: Usd.Prim | None = None,
    ) -> int:
        if "/Geometry/" not in target_prim.GetPath().pathString:
            return super()._load_target_prim_warp_mesh(target_prim, target_cfg, reference_prim)

        reference_prim = target_prim if reference_prim is None else reference_prim
        prim_key = (f"{target_prim.GetPath()}@{reference_prim.GetPath()}", self._device)
        if prim_key in BaseMultiMeshRayCaster.meshes:
            return BaseMultiMeshRayCaster.meshes[prim_key].id

        # The importer nests child rigid links below their parent. Traverse instance
        # geometry, but stop before entering a descendant IsaacLinkAPI prim.
        mesh_prims: list[Usd.Prim] = []
        queue = list(target_prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        while queue:
            prim = queue.pop(0)
            is_importer_link = any(
                schema_name.split(":", maxsplit=1)[0] == "IsaacLinkAPI" for schema_name in prim.GetAppliedSchemas()
            )
            if is_importer_link or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if prim.GetTypeName() in PRIMITIVE_MESH_TYPES + ["Mesh"] and not prim.HasAPI(UsdPhysics.CollisionAPI):
                mesh_prims.append(prim)
            queue.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))

        if not mesh_prims:
            raise RuntimeError(f"No visual mesh prims found for imported link prim: {target_prim.GetPath()}")

        trimesh_meshes = []
        for mesh_prim in mesh_prims:
            mesh = (
                create_trimesh_from_geom_mesh(mesh_prim)
                if mesh_prim.GetTypeName() == "Mesh"
                else create_trimesh_from_geom_shape(mesh_prim)
            )
            mesh.apply_scale(sim_utils.resolve_prim_scale(mesh_prim))
            relative_pos, relative_quat = sim_utils.resolve_prim_pose(mesh_prim, reference_prim)
            transform = np.eye(4)
            transform[:3, :3] = matrix_from_quat_xyzw(np.asarray(relative_quat, dtype=np.float64))
            transform[:3, 3] = np.asarray(relative_pos, dtype=np.float64)
            mesh.apply_transform(transform)
            trimesh_meshes.append(mesh)

        if len(trimesh_meshes) == 1:
            trimesh_mesh = trimesh_meshes[0]
        elif target_cfg.merge_prim_meshes:
            trimesh_mesh = trimesh.util.concatenate(trimesh_meshes)
        else:
            raise RuntimeError(
                f"Multiple visual meshes found for imported link prim '{target_prim.GetPath()}', but merging is"
                " disabled."
            )

        wp_mesh = convert_to_warp_mesh(trimesh_mesh.vertices, trimesh_mesh.faces, device=self._device)
        BaseMultiMeshRayCaster.meshes[prim_key] = wp_mesh
        logger.info(
            "Read %d visual meshes for imported link prim '%s' with %d vertices and %d faces.",
            len(mesh_prims),
            target_prim.GetPath(),
            len(trimesh_mesh.vertices),
            len(trimesh_mesh.faces),
        )
        return wp_mesh.id
