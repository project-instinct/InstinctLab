from __future__ import annotations

import re
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors.ray_caster import MultiMeshRayCaster as NewtonMultiMeshRayCaster
from isaaclab_physx.sensors.ray_caster import MultiMeshRayCaster

from isaaclab.sensors.ray_caster.kernels import fill_ray_hits_distance_inf_kernel

from instinctlab.utils.warp.kernels import raycast_flat_mesh_groups_min_distance_kernel

from .flat_target_prim_registry import FlatTargetPrimRegistryMixin

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg


class GroupedRayCaster(FlatTargetPrimRegistryMixin, MultiMeshRayCaster):
    """PhysX ray caster over flat mesh entities grouped by fixed world IDs."""

    cfg: GroupedRayCasterCfg

    def _update_buffers_impl(self, env_mask: wp.array):
        self._update_ray_infos(env_mask)
        self._update_mesh_transforms()

        wp.launch(
            fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, False],
            outputs=[self._data._ray_hits_w, self._ray_distance_wp, self._dummy_normal_wp],
            device=self._device,
        )

        wp.launch(
            raycast_flat_mesh_groups_min_distance_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._ray_world_ids_wp,
                self._world_mesh_indices_wp,
                self._world_mesh_offsets_wp,
                self._flat_mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self._data._ray_hits_w,
                self._ray_distance_wp,
                self._dummy_normal_wp,
                self._dummy_face_id_wp,
                self._data.ray_mesh_ids.warp if self.cfg.update_mesh_ids else self._ray_mesh_id_wp,
                self._flat_mesh_positions_w,
                self._flat_mesh_orientations_w,
                float(self.cfg.max_distance),
                float(self.cfg.min_distance),
                int(self._num_envs),
                int(self._num_flat_mesh_entities),
                int(self._num_world_mesh_indices),
                int(self.num_rays),
                int(False),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )


class NewtonGroupedRayCaster(FlatTargetPrimRegistryMixin, NewtonMultiMeshRayCaster):
    """Newton ray caster over flat mesh entities grouped by fixed world IDs."""

    cfg: GroupedRayCasterCfg

    def _register_sites_for_expr(self, prim_expr: str) -> list[str]:
        attach_expr = prim_expr
        if prim_expr.rsplit("/", 1)[-1].lower() in ("camera", "raycaster"):
            attach_expr = prim_expr.rsplit("/", 1)[0]
        body_pattern = re.sub(r"env_\.\*", "env_0", attach_expr)
        if body_pattern.startswith("/World/envs/env_0/"):
            identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
            return [NewtonManager.cl_register_site(body_pattern, identity)]
        return super()._register_sites_for_expr(prim_expr)

    def _register_target_sites_for_exprs(self, owner_exprs: list[str]) -> list[str]:
        identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
        patterns = [re.sub(r"env_(?:\.\*|\*)", "env_0", owner_expr) for owner_expr in owner_exprs]
        return [NewtonManager.cl_register_site(pattern, identity) for pattern in patterns]

    def _create_tracked_target_view(self, target_prim_path: str | list[str]):
        target_exprs = target_prim_path if isinstance(target_prim_path, list) else [target_prim_path]
        lookup_key = tuple(re.sub(r"env_\.\*", "env_*", expr) for expr in target_exprs)
        labels = self._tracked_site_labels_by_target[lookup_key]
        site_indices = self._resolve_site_indices(labels, str(target_prim_path), self._num_envs)
        return wp.array(site_indices, dtype=wp.int32, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array):
        self._update_ray_infos(env_mask)
        self._update_mesh_transforms()

        wp.launch(
            fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, False],
            outputs=[self._data._ray_hits_w, self._ray_distance_wp, self._dummy_normal_wp],
            device=self._device,
        )

        wp.launch(
            raycast_flat_mesh_groups_min_distance_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._ray_world_ids_wp,
                self._world_mesh_indices_wp,
                self._world_mesh_offsets_wp,
                self._flat_mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self._data._ray_hits_w,
                self._ray_distance_wp,
                self._dummy_normal_wp,
                self._dummy_face_id_wp,
                self._data.ray_mesh_ids.warp if self.cfg.update_mesh_ids else self._ray_mesh_id_wp,
                self._flat_mesh_positions_w,
                self._flat_mesh_orientations_w,
                float(self.cfg.max_distance),
                float(self.cfg.min_distance),
                int(self._num_envs),
                int(self._num_flat_mesh_entities),
                int(self._num_world_mesh_indices),
                int(self.num_rays),
                int(False),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )
