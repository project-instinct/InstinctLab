from __future__ import annotations

import re
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_physx.physics import PhysxManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils

from instinctlab.sensors.volume_points.volume_points import VolumePointsBase

if TYPE_CHECKING:
    import torch

    import omni.physics.tensors.api as physx

    from instinctlab.sensors.volume_points.volume_points_cfg import VolumePointsCfg


class PhysxVolumePoints(VolumePointsBase):
    """PhysX rigid-body state access for the volume-points sensor."""

    cfg: VolumePointsCfg

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        """PhysX rigid-body view captured by the sensor."""
        return self._body_physx_view

    def _initialize_backend_impl(self) -> None:
        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        if self._physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")

        root_matches = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path)
        if not root_matches:
            raise RuntimeError(f"Sensor root at path '{self.cfg.prim_path}' could not be resolved.")
        template_root, destination_root_expr = root_matches[0]

        name_exprs = self.cfg.body_names_expr
        if isinstance(name_exprs, str):
            name_exprs = [name_exprs]
        name_patterns = [re.compile(f"^{expr}$") for expr in name_exprs]
        body_prims = sim_utils.get_all_matching_child_prims(
            template_root.GetPath(),
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and any(pattern.match(prim.GetName()) is not None for pattern in name_patterns),
            traverse_instance_prims=False,
        )
        if not body_prims:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find rigid bodies matching {name_exprs}."
            )

        template_root_path = template_root.GetPath().pathString
        body_paths_glob = [
            (destination_root_expr + prim.GetPath().pathString[len(template_root_path) :]).replace(".*", "*")
            for prim in body_prims
        ]
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_paths_glob)
        self._num_bodies = self.body_physx_view.count // self._num_envs
        if self._num_bodies != len(body_prims):
            raise RuntimeError(
                "Failed to initialize volume points sensor for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_paths_glob}"
            )
        self._body_names = [path.split("/")[-1] for path in self.body_physx_view.prim_paths[: self.num_bodies]]

    def _refresh_body_state(self, env_mask: wp.array, env_ids: torch.Tensor) -> None:
        body_poses = wp.to_torch(self.body_physx_view.get_transforms()).view(-1, self.num_bodies, 7)[env_ids]
        body_vels = wp.to_torch(self.body_physx_view.get_velocities()).view(-1, self.num_bodies, 6)[env_ids]
        self._data.pos_w.torch[env_ids] = body_poses[..., :3]
        self._data.quat_w.torch[env_ids] = body_poses[..., 3:]
        self._data.vel_w.torch[env_ids] = body_vels[..., :3]
        self._data.ang_vel_w.torch[env_ids] = body_vels[..., 3:]

    def _body_state_is_valid(self) -> bool:
        return self._body_physx_view is not None

    def _invalidate_backend_impl(self) -> None:
        self._physics_sim_view = None
        self._body_physx_view = None
