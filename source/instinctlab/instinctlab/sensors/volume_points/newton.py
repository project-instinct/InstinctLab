from __future__ import annotations

import re
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import NewtonManager

from instinctlab.sensors.volume_points.volume_points import VolumePointsBase

if TYPE_CHECKING:
    import torch

    from instinctlab.sensors.volume_points.volume_points_cfg import VolumePointsCfg


@wp.kernel
def _update_newton_body_state_kernel(
    env_mask: wp.array(dtype=wp.bool),
    site_indices: wp.array(dtype=wp.int32),
    num_bodies: int,
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    out_pos_w: wp.array2d(dtype=wp.vec3f),
    out_quat_w: wp.array2d(dtype=wp.quatf),
    out_vel_w: wp.array2d(dtype=wp.vec3f),
    out_ang_vel_w: wp.array2d(dtype=wp.vec3f),
):
    env_id, local_body_id = wp.tid()
    if not env_mask[env_id]:
        return

    site_idx = site_indices[env_id * num_bodies + local_body_id]
    body_idx = shape_body[site_idx]
    site_xform = shape_transform[site_idx]
    body_xform = body_q[body_idx]
    body_quat = wp.transform_get_rotation(body_xform)

    out_pos_w[env_id, local_body_id] = wp.transform_get_translation(body_xform) + wp.quat_rotate(
        body_quat, site_xform.p
    )
    out_quat_w[env_id, local_body_id] = body_quat * site_xform.q

    ang_vel_w = wp.spatial_bottom(body_qd[body_idx])
    body_origin_from_com_w = wp.quat_rotate(body_quat, site_xform.p - body_com[body_idx])
    out_vel_w[env_id, local_body_id] = wp.spatial_top(body_qd[body_idx]) + wp.cross(ang_vel_w, body_origin_from_com_w)
    out_ang_vel_w[env_id, local_body_id] = ang_vel_w


class NewtonVolumePoints(VolumePointsBase):
    """Newton rigid-body state access for the volume-points sensor."""

    cfg: VolumePointsCfg

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__(cfg)
        root_pattern = re.sub(r"env_\.\*", "env_0", self.cfg.prim_path).rstrip("/")
        name_exprs = self.cfg.body_names_expr
        if isinstance(name_exprs, str):
            name_exprs = [name_exprs]
        self._body_pattern = "|".join(f"(?:{root_pattern}/{expr})" for expr in name_exprs)
        self._register_body_sites()
        self._site_indices = None
        self._newton_model = None

    def _register_body_sites(self) -> None:
        identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
        self._site_label = NewtonManager.cl_register_site(self._body_pattern, identity)

    def _initialize_backend_impl(self) -> None:
        site_map = NewtonManager._cl_site_index_map
        if self._site_label not in site_map:
            raise ValueError(
                f"VolumePoints body site '{self._site_label}' for '{self._body_pattern}' was not registered."
            )

        global_idx, per_world = site_map[self._site_label]
        if global_idx is not None or per_world is None:
            raise RuntimeError("VolumePoints requires body-attached Newton sites in every environment.")
        if len(per_world) != self._num_envs:
            raise RuntimeError(f"VolumePoints resolved {len(per_world)} Newton worlds, expected {self._num_envs}.")

        body_counts = {len(world_sites) for world_sites in per_world}
        if len(body_counts) != 1 or not body_counts or next(iter(body_counts)) == 0:
            raise RuntimeError(
                f"VolumePoints requires the same non-zero body count in every Newton world; got {sorted(body_counts)}."
            )
        self._num_bodies = next(iter(body_counts))
        site_indices = [site_idx for world_sites in per_world for site_idx in world_sites]
        self._site_indices = wp.array(site_indices, dtype=wp.int32, device=self._device)
        self._newton_model = NewtonManager._model
        if self._newton_model is None:
            raise RuntimeError("Newton simulation model is not initialized.")

        shape_body = wp.to_torch(self._newton_model.shape_body)
        first_world_body_ids = shape_body[per_world[0]].cpu().tolist()
        self._body_names = [self._newton_model.body_label[body_id].split("/")[-1] for body_id in first_world_body_ids]

    def _refresh_body_state(self, env_mask: wp.array, env_ids: torch.Tensor) -> None:
        state = NewtonManager._state_0
        if self._newton_model is None or state is None:
            raise RuntimeError("Newton volume-points state is not initialized.")

        wp.launch(
            _update_newton_body_state_kernel,
            dim=(self._num_envs, self._num_bodies),
            inputs=[
                env_mask,
                self._site_indices,
                self._num_bodies,
                self._newton_model.shape_body,
                self._newton_model.shape_transform,
                self._newton_model.body_com,
                state.body_q,
                state.body_qd,
            ],
            outputs=[
                self._data.pos_w.warp,
                self._data.quat_w.warp,
                self._data.vel_w.warp,
                self._data.ang_vel_w.warp,
            ],
            device=self._device,
        )

    def _body_state_is_valid(self) -> bool:
        return self._newton_model is not None and self._site_indices is not None

    def _invalidate_backend_impl(self) -> None:
        self._site_indices = None
        self._newton_model = None
        self._register_body_sites()
