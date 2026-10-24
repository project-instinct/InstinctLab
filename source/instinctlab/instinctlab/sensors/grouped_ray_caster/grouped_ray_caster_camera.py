from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.sensors.ray_caster import kernels as ray_caster_kernels

from instinctlab.sensors.grouped_ray_caster.grouped_kernels import raycast_mesh_table_min_distance_kernel
from instinctlab.utils.backend_dispatch import create_backend_component

if TYPE_CHECKING:
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg


class GroupedRayCasterCameraKernelMixin:
    """Backend-neutral grouped ray-cast camera update."""

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
            raycast_mesh_table_min_distance_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(ray_caster_kernels.CAMERA_RAYCAST_MAX_DIST),
                float(self.cfg.min_distance),
                int(return_normal),
                int(False),  # return_face_id
                int(self.cfg.update_mesh_ids),
            ],
            outputs=[
                self.ray_hits_w.warp,
                self._ray_distance_cam_wp,
                self._ray_normal_w,
                self._ray_face_id_wp,
                self._ray_mesh_id_wp,
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


class GroupedRayCasterCamera:
    """Construct the grouped ray-caster camera for the active physics backend."""

    def __new__(cls, cfg: GroupedRayCasterCameraCfg):
        return create_backend_component(
            cfg,
            {
                "physx": "instinctlab.sensors.grouped_ray_caster.physx:PhysxGroupedRayCasterCamera",
                "newton": "instinctlab.sensors.grouped_ray_caster.newton:NewtonGroupedRayCasterCamera",
            },
        )
