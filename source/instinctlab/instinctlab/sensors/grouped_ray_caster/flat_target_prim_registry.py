from __future__ import annotations

import logging
import numpy as np
import re
import trimesh
from typing import TYPE_CHECKING

import warp as wp
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.warp import convert_to_warp_mesh

from instinctlab.utils.math import matrix_from_quat_xyzw
from instinctlab.utils.warp.kernels import copy_flat_mesh_transforms_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg


logger = logging.getLogger(__name__)


class FlatTargetPrimRegistryMixin:
    """Build flat target-prim records and fixed world membership for grouped ray casting."""

    def _build_mesh_records(self, target_cfg, plan, dummy_mesh_id):
        """Keep an articulation-root link as one tracked mesh target.

        The upstream multi-mesh builder interprets a prim with
        ``ArticulationRootAPI`` as a request for every rigid-body descendant.
        Importer 3 authors that API on the G1 torso link, but InstinctLab lists
        every robot link explicitly. Expanding the torso would therefore add
        the whole robot once and the other 29 links a second time.
        """
        if plan is None or not target_cfg.track_mesh_transforms:
            return super()._build_mesh_records(target_cfg, plan, dummy_mesh_id)

        records_per_env = [[] for _ in range(self._num_envs)]
        tracked_target_exprs: list[str] = []
        found_articulation_root = False
        for source_root, destination_template, source_path, env_ids in iter_clone_plan_matches(
            plan, target_cfg.prim_expr
        ):
            source_prims = sim_utils.find_matching_prims(source_path)
            articulation_root_prims = [prim for prim in source_prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if not articulation_root_prims:
                continue
            if len(articulation_root_prims) != len(source_prims):
                raise RuntimeError(
                    f"Ray-cast target '{target_cfg.prim_expr}' mixes articulation-root and non-root prims."
                )

            found_articulation_root = True
            mesh_ids = []
            for source_prim in articulation_root_prims:
                if not source_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(
                        f"Articulation-root ray-cast target '{source_prim.GetPath()}' is not a rigid body."
                    )
                mesh_id = self._load_target_prim_warp_mesh(source_prim, target_cfg, reference_prim=source_prim)
                dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
                mesh_ids.append(mesh_id)

                source_prim_path = str(source_prim.GetPath())
                if source_prim_path == source_root:
                    owner_suffix = ""
                elif source_prim_path.startswith(source_root + "/"):
                    owner_suffix = source_prim_path[len(source_root) :]
                else:
                    raise RuntimeError(
                        f"Tracked target owner '{source_prim_path}' is not under ClonePlan source root '{source_root}'."
                    )
                tracked_target_exprs.append(destination_template.format(".*") + owner_suffix)

            for env_id in env_ids:
                for mesh_id in mesh_ids:
                    records_per_env[env_id].append((mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0)))

        if found_articulation_root:
            if not tracked_target_exprs:
                raise RuntimeError(f"No tracked body expression resolved for target '{target_cfg.prim_expr}'.")
            return records_per_env, dummy_mesh_id, tracked_target_exprs
        return super()._build_mesh_records(target_cfg, plan, dummy_mesh_id)

    @staticmethod
    def _mesh_record_key(record) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
        """Return the exact identity used to share one static entity across worlds."""
        mesh_id, position, orientation = record
        return int(mesh_id), tuple(float(value) for value in position), tuple(float(value) for value in orientation)

    @staticmethod
    def _view_world_ids(view) -> list[int] | None:
        """Extract concrete world IDs from a tracked view when paths are available."""
        prim_paths = getattr(view, "prim_paths", None)
        if prim_paths is None or len(prim_paths) != view.count:
            return None

        world_ids = []
        for prim_path in prim_paths:
            match = re.search(r"/env_(\d+)(?:/|$)", str(prim_path))
            if match is None:
                return None
            world_ids.append(int(match.group(1)))
        return world_ids

    def _initialize_warp_meshes(self) -> None:
        """Build flat entity records and fixed world-to-entity membership.

        A global static mesh is one entity referenced by every applicable world.
        Dynamic per-world bodies remain distinct entities even when their Warp
        geometry ID is shared. The membership is precomputed once for the sensor
        lifetime and checked before it is copied to the device.
        """
        sim = SimulationContext.instance()
        plan = sim.get_clone_plan() if sim is not None else None

        self._num_meshes_per_env.clear()
        self._mesh_views = []
        self._tracked_view_entity_indices: list[wp.array | None] = []

        flat_mesh_ids: list[int] = []
        flat_mesh_positions: list[tuple[float, ...]] = []
        flat_mesh_orientations: list[tuple[float, ...]] = []
        world_entity_indices: list[list[int]] = [[] for _ in range(self._num_envs)]
        world_entity_sets: list[set[int]] = [set() for _ in range(self._num_envs)]
        static_entities: dict[tuple[int, tuple[float, ...], tuple[float, ...]], int] = {}
        dummy_mesh_id: int | None = None

        def append_entity(record) -> int:
            mesh_id, position, orientation = record
            entity_index = len(flat_mesh_ids)
            flat_mesh_ids.append(int(mesh_id))
            flat_mesh_positions.append(tuple(float(value) for value in position))
            flat_mesh_orientations.append(tuple(float(value) for value in orientation))
            return entity_index

        def add_world_membership(world_id: int, entity_index: int) -> None:
            if world_id < 0 or world_id >= self._num_envs:
                raise RuntimeError(f"Ray-cast world ID {world_id} is outside [0, {self._num_envs}).")
            if entity_index not in world_entity_sets[world_id]:
                world_entity_sets[world_id].add(entity_index)
                world_entity_indices[world_id].append(entity_index)

        for target_cfg in self._raycast_targets_cfg:
            records_per_world, dummy_mesh_id, tracked_target_exprs = self._build_mesh_records(
                target_cfg, plan, dummy_mesh_id
            )
            if len(records_per_world) != self._num_envs:
                raise RuntimeError(
                    f"Ray-cast target '{target_cfg.prim_expr}' returned {len(records_per_world)} world rows; "
                    f"expected {self._num_envs}."
                )

            record_counts = [len(records) for records in records_per_world]
            self._num_meshes_per_env[target_cfg.prim_expr] = max(record_counts, default=0)
            view = self._create_tracked_target_view(tracked_target_exprs) if target_cfg.track_mesh_transforms else None
            self._mesh_views.append(view)

            if not target_cfg.track_mesh_transforms:
                for world_id, records in enumerate(records_per_world):
                    for record in records:
                        key = self._mesh_record_key(record)
                        entity_index = static_entities.get(key)
                        if entity_index is None:
                            entity_index = append_entity(record)
                            static_entities[key] = entity_index
                        add_world_membership(world_id, entity_index)
                self._tracked_view_entity_indices.append(None)
                continue

            if view is None:
                raise RuntimeError(f"Tracked ray-cast target '{target_cfg.prim_expr}' did not create a physics view.")

            view_count = view.shape[0] if isinstance(view, wp.array) else int(view.count)
            populated_worlds = [world_id for world_id, count in enumerate(record_counts) if count > 0]
            if view_count == 1:
                if any(record_counts[world_id] != 1 for world_id in populated_worlds):
                    raise RuntimeError(
                        f"Single-body ray-cast target '{target_cfg.prim_expr}' has per-world record counts "
                        f"{record_counts}."
                    )
                first_record = records_per_world[populated_worlds[0]][0] if populated_worlds else None
                if first_record is None:
                    raise RuntimeError(f"Tracked ray-cast target '{target_cfg.prim_expr}' has no mesh record.")
                if any(int(records_per_world[world_id][0][0]) != int(first_record[0]) for world_id in populated_worlds):
                    raise RuntimeError(
                        f"Single-body ray-cast target '{target_cfg.prim_expr}' resolves different geometry per world."
                    )
                entity_index = append_entity(first_record)
                for world_id in populated_worlds:
                    add_world_membership(world_id, entity_index)
                view_entity_indices = [entity_index]
            else:
                if view_count != sum(record_counts):
                    raise RuntimeError(
                        f"Tracked ray-cast target '{target_cfg.prim_expr}' has {view_count} physics bodies but "
                        f"{sum(record_counts)} flat mesh records across worlds {record_counts}."
                    )

                target_entities_per_world: list[list[int]] = [[] for _ in range(self._num_envs)]
                for world_id, records in enumerate(records_per_world):
                    for record in records:
                        entity_index = append_entity(record)
                        target_entities_per_world[world_id].append(entity_index)
                        add_world_membership(world_id, entity_index)

                concrete_view_world_ids = self._view_world_ids(view)
                if concrete_view_world_ids is None:
                    if len(set(record_counts)) != 1:
                        raise RuntimeError(
                            f"Cannot prove tracked-view ordering for ragged target '{target_cfg.prim_expr}': "
                            f"{record_counts}."
                        )
                    view_entity_indices = [
                        entity_index for entities in target_entities_per_world for entity_index in entities
                    ]
                else:
                    per_world_view_count = [0] * self._num_envs
                    view_entity_indices = []
                    for view_index, world_id in enumerate(concrete_view_world_ids):
                        if world_id < 0 or world_id >= self._num_envs:
                            raise RuntimeError(
                                f"Tracked view index {view_index} for '{target_cfg.prim_expr}' belongs to invalid "
                                f"world {world_id}."
                            )
                        mesh_slot = per_world_view_count[world_id]
                        if mesh_slot >= len(target_entities_per_world[world_id]):
                            raise RuntimeError(
                                f"Tracked view index {view_index} exceeds world {world_id}'s mesh records for "
                                f"'{target_cfg.prim_expr}'."
                            )
                        view_entity_indices.append(target_entities_per_world[world_id][mesh_slot])
                        per_world_view_count[world_id] += 1
                    if per_world_view_count != record_counts:
                        raise RuntimeError(
                            f"Tracked view world counts {per_world_view_count} do not match mesh record counts "
                            f"{record_counts} for '{target_cfg.prim_expr}'."
                        )

            if len(view_entity_indices) != view_count:
                raise RuntimeError(
                    f"Tracked target '{target_cfg.prim_expr}' produced {len(view_entity_indices)} view mappings; "
                    f"expected {view_count}."
                )
            self._tracked_view_entity_indices.append(wp.array(view_entity_indices, dtype=wp.int32, device=self._device))

        if dummy_mesh_id is None or not flat_mesh_ids:
            raise RuntimeError(f"No meshes found for ray-casting. Check mesh prim paths: {self.cfg.mesh_prim_paths}")

        num_entities = len(flat_mesh_ids)
        world_mesh_offsets = [0]
        flat_world_mesh_indices: list[int] = []
        for world_id, entity_indices in enumerate(world_entity_indices):
            for mesh_index in entity_indices:
                if mesh_index < 0 or mesh_index >= num_entities:
                    raise RuntimeError(
                        f"World {world_id} references flat mesh index {mesh_index}, but there are"
                        f" {num_entities} entities."
                    )
            flat_world_mesh_indices.extend(entity_indices)
            world_mesh_offsets.append(len(flat_world_mesh_indices))

        self._flat_mesh_ids_wp = wp.array(flat_mesh_ids, dtype=wp.uint64, device=self._device)
        self._flat_mesh_positions_w = wp.array(flat_mesh_positions, dtype=wp.vec3f, device=self._device)
        self._flat_mesh_orientations_w = wp.array(flat_mesh_orientations, dtype=wp.quatf, device=self._device)
        self._world_mesh_indices_wp = wp.array(flat_world_mesh_indices, dtype=wp.int32, device=self._device)
        self._world_mesh_offsets_wp = wp.array(world_mesh_offsets, dtype=wp.int32, device=self._device)
        self._num_flat_mesh_entities = num_entities
        self._num_world_mesh_indices = len(flat_world_mesh_indices)

        logger.info(
            "Built %d flat ray-cast entities and %d fixed memberships across %d worlds.",
            self._num_flat_mesh_entities,
            self._num_world_mesh_indices,
            self._num_envs,
        )

    def _initialize_rays_impl(self) -> None:
        """Initialize upstream ray buffers and the fixed world ID of every ray."""
        super()._initialize_rays_impl()
        if self._view_count != self._num_envs:
            raise RuntimeError(
                f"Grouped ray caster has {self._view_count} ray batches for {self._num_envs} worlds; "
                "the ray-to-world mapping is not one-to-one."
            )
        ray_world_ids = np.broadcast_to(
            np.arange(self._num_envs, dtype=np.int32)[:, None], (self._num_envs, self.num_rays)
        ).copy()
        if ray_world_ids.shape != (self._view_count, self.num_rays):
            raise RuntimeError(
                f"Ray world-ID table has shape {ray_world_ids.shape}; expected ({self._view_count}, {self.num_rays})."
            )
        self._ray_world_ids_wp = wp.array2d(ray_world_ids, dtype=wp.int32, device=self._device)

        if self.cfg.update_mesh_ids and self._num_flat_mesh_entities > np.iinfo(np.int16).max:
            raise RuntimeError(
                f"Cannot report {self._num_flat_mesh_entities} flat mesh IDs through the int16 sensor data contract."
            )

    def _update_mesh_transforms(self) -> None:
        """Scatter live physics transforms into their checked flat entity indices."""
        for view, entity_indices in zip(self._mesh_views, self._tracked_view_entity_indices):
            if view is None:
                if entity_indices is not None:
                    raise RuntimeError("Static ray-cast target unexpectedly has tracked entity indices.")
                continue
            is_newton_view = isinstance(view, wp.array)
            view_count = view.shape[0] if is_newton_view else int(view.count)
            if entity_indices is None or entity_indices.shape[0] != view_count:
                raise RuntimeError(
                    f"Tracked ray-cast view has {view_count} transforms but "
                    f"{0 if entity_indices is None else entity_indices.shape[0]} flat entity indices."
                )

            if is_newton_view:
                transforms_wp = wp.empty(view_count, dtype=wp.transformf, device=self._device)
                self._update_newton_site_transforms(
                    view,
                    transforms_wp,
                    wp.empty(view_count, dtype=wp.vec3f, device=self._device),
                    wp.empty(view_count, dtype=wp.quatf, device=self._device),
                )
            else:
                transforms = view.get_transforms()
                transforms_wp = (
                    transforms.view(wp.transformf)
                    if isinstance(transforms, wp.array)
                    else wp.from_torch(transforms.contiguous()).view(wp.transformf)
                )
            wp.launch(
                copy_flat_mesh_transforms_kernel,
                dim=view_count,
                inputs=[
                    transforms_wp,
                    entity_indices,
                    int(self._num_flat_mesh_entities),
                    self._flat_mesh_positions_w,
                    self._flat_mesh_orientations_w,
                ],
                device=self._device,
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
