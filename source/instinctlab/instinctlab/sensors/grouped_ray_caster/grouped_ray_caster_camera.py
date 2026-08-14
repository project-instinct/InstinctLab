from __future__ import annotations

import re
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_physx.sensors.ray_caster import MultiMeshRayCasterCamera
from isaaclab_newton.sensors.ray_caster import MultiMeshRayCasterCamera as NewtonMultiMeshRayCasterCamera

from isaaclab.sensors.ray_caster import kernels as ray_caster_kernels
from isaaclab_newton.physics import NewtonManager

from instinctlab.utils.warp.kernels import raycast_flat_mesh_groups_min_distance_kernel

from .grouped_ray_caster import FlatTargetPrimRegistryMixin

if TYPE_CHECKING:
    from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg


class GroupedRayCasterCamera(FlatTargetPrimRegistryMixin, MultiMeshRayCasterCamera):
    """PhysX multi-mesh ray-caster camera with an ignored near-hit interval."""

    cfg: GroupedRayCasterCameraCfg

    def _update_buffers_impl(self, env_mask: wp.array):
        self._update_ray_infos(env_mask)
        self._update_frame(env_mask, frame_op=1)
        self._update_mesh_transforms()

        return_normal = "normals" in self.cfg.data_types
        wp.launch(
            ray_caster_kernels.fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, return_normal],
            outputs=[self.ray_hits_w.warp, self._ray_distance_cam_wp, self._ray_normal_w],
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
                self.ray_hits_w.warp,
                self._ray_distance_cam_wp,
                self._ray_normal_w,
                self._ray_face_id_wp,
                self._ray_mesh_id_wp,
                self._flat_mesh_positions_w,
                self._flat_mesh_orientations_w,
                float(ray_caster_kernels.CAMERA_RAYCAST_MAX_DIST),
                float(self.cfg.min_distance),
                int(self._num_envs),
                int(self._num_flat_mesh_entities),
                int(self._num_world_mesh_indices),
                int(self.num_rays),
                int(return_normal),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )

        if "distance_to_image_plane" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.compute_distance_to_image_plane_to_image_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._data.quat_w_world.warp,
                    self._ray_distance_cam_wp,
                    self._ray_directions_w,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[self._data.output["distance_to_image_plane"].warp],
                device=self._device,
            )

        if "distance_to_camera" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.copy_float2d_to_image1_depth_clipped_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._ray_distance_cam_wp,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[self._data.output["distance_to_camera"].warp],
                device=self._device,
            )

        if return_normal:
            wp.launch(
                ray_caster_kernels.copy_vec3_2d_to_image3_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_normal_w, int(self.image_shape[1]), self._data.output["normals"].warp],
                device=self._device,
            )

        if self.cfg.update_mesh_ids:
            wp.launch(
                ray_caster_kernels.copy_int16_2d_to_image1_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_mesh_id_wp, int(self.image_shape[1]), self._data.image_mesh_ids.warp],
                device=self._device,
            )


class NewtonGroupedRayCasterCamera(FlatTargetPrimRegistryMixin, NewtonMultiMeshRayCasterCamera):
    """Newton multi-mesh ray-caster camera with the grouped-world flat-mesh update path."""

    cfg: GroupedRayCasterCameraCfg

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
        self._update_frame(env_mask, frame_op=1)
        self._update_mesh_transforms()

        return_normal = "normals" in self.cfg.data_types
        wp.launch(
            ray_caster_kernels.fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, return_normal],
            outputs=[self.ray_hits_w.warp, self._ray_distance_cam_wp, self._ray_normal_w],
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
                self.ray_hits_w.warp,
                self._ray_distance_cam_wp,
                self._ray_normal_w,
                self._ray_face_id_wp,
                self._ray_mesh_id_wp,
                self._flat_mesh_positions_w,
                self._flat_mesh_orientations_w,
                float(ray_caster_kernels.CAMERA_RAYCAST_MAX_DIST),
                float(self.cfg.min_distance),
                int(self._num_envs),
                int(self._num_flat_mesh_entities),
                int(self._num_world_mesh_indices),
                int(self.num_rays),
                int(return_normal),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )

        if "distance_to_image_plane" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.compute_distance_to_image_plane_to_image_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._data.quat_w_world.warp,
                    self._ray_distance_cam_wp,
                    self._ray_directions_w,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[self._data.output["distance_to_image_plane"].warp],
                device=self._device,
            )

        if "distance_to_camera" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.copy_float2d_to_image1_depth_clipped_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._ray_distance_cam_wp,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[self._data.output["distance_to_camera"].warp],
                device=self._device,
            )

        if return_normal:
            wp.launch(
                ray_caster_kernels.copy_vec3_2d_to_image3_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_normal_w, int(self.image_shape[1]), self._data.output["normals"].warp],
                device=self._device,
            )

        if self.cfg.update_mesh_ids:
            wp.launch(
                ray_caster_kernels.copy_int16_2d_to_image1_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_mesh_id_wp, int(self.image_shape[1]), self._data.image_mesh_ids.warp],
                device=self._device,
            )
