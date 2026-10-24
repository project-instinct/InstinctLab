from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.sensors.ray_caster.kernels import fill_ray_hits_distance_inf_kernel

from instinctlab.sensors.grouped_ray_caster.grouped_kernels import raycast_mesh_table_min_distance_kernel
from instinctlab.utils.backend_dispatch import create_backend_component

if TYPE_CHECKING:
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_cfg import GroupedRayCasterCfg


class GroupedRayCasterKernelMixin:
    """Backend-neutral grouped ray-cast update."""

    cfg: GroupedRayCasterCfg

    def _update_buffers_impl(self, env_mask: wp.array):
        self._update_ray_infos(env_mask)
        self._update_mesh_transforms()

        # Fill output and distance buffers with inf for masked environments
        wp.launch(
            fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, False],
            outputs=[self._data._ray_hits_w, self._ray_distance_wp, self._dummy_normal_wp],
            device=self._device,
        )

        # Ray-cast against all meshes; one thread per ray keeps the closest hit.
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
                float(self.cfg.max_distance),
                float(self.cfg.min_distance),
                int(False),  # return_normal
                int(False),  # return_face_id
                int(self.cfg.update_mesh_ids),
            ],
            outputs=[
                self._data._ray_hits_w,
                self._ray_distance_wp,
                self._dummy_normal_wp,
                self._dummy_face_id_wp,
                self._data.ray_mesh_ids.warp if self.cfg.update_mesh_ids else self._ray_mesh_id_wp,
            ],
            device=self._device,
        )


class GroupedRayCaster:
    """Construct the grouped ray caster for the active physics backend."""

    def __new__(cls, cfg: GroupedRayCasterCfg):
        return create_backend_component(
            cfg,
            {
                "physx": "instinctlab.sensors.grouped_ray_caster.physx:PhysxGroupedRayCaster",
                "newton": "instinctlab.sensors.grouped_ray_caster.newton:NewtonGroupedRayCaster",
            },
        )
