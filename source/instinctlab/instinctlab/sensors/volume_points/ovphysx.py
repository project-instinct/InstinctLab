from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

import isaaclab_ov.tensor_types as TT
import warp as wp
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView
from pxr import UsdPhysics

import isaaclab.sim as sim_utils

from instinctlab.sensors.volume_points.volume_points import VolumePointsBase, env_body_rows
from instinctlab.utils.prims import sort_by_articulation_order

if TYPE_CHECKING:
    from instinctlab.sensors.volume_points.volume_points_cfg import VolumePointsCfg


class OvPhysxVolumePoints(VolumePointsBase):
    """OvPhysX rigid-body state access for the volume-points sensor."""

    cfg: VolumePointsCfg

    def _initialize_backend_impl(self) -> None:
        physx_instance = self._backend_manager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")

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

        # Bodies in the order of the robot articulation (ArticulationCfg.body_ordering), whatever the backend.
        order = sort_by_articulation_order(self.cfg.prim_path, "body", [prim.GetName() for prim in body_prims])
        body_prims = sorted(body_prims, key=lambda prim: order.index(prim.GetName()))

        template_root_path = template_root.GetPath().pathString
        body_suffixes = [prim.GetPath().pathString[len(template_root_path) :] for prim in body_prims]
        body_globs = [sim_utils.path_expr_to_glob(destination_root_expr + suffix) for suffix in body_suffixes]
        self._view = OvPhysxView(physx_instance, prim_paths=body_globs, device=self._device)
        self._view.binding_for(TT.RIGID_BODY_POSE)
        self._view.binding_for(TT.RIGID_BODY_VELOCITY)

        self._num_bodies = len(body_prims)
        if self._view.count != self._num_envs * self._num_bodies:
            raise RuntimeError(
                "Failed to initialize volume points sensor for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_globs}"
                f"\n\tMatched bodies     : {self._view.count}, expected {self._num_envs * self._num_bodies}"
            )
        self._row_order = env_body_rows(self._view.prim_paths, body_suffixes, self._device)
        self._body_names = [prim.GetName() for prim in body_prims]

    def _refresh_body_state(self, env_mask: wp.array, env_ids: torch.Tensor) -> None:
        poses = wp.to_torch(self._view.get_attribute(TT.RIGID_BODY_POSE)).reshape(-1, 7)[self._row_order]
        vels = wp.to_torch(self._view.get_attribute(TT.RIGID_BODY_VELOCITY)).reshape(-1, 6)[self._row_order]
        body_poses = poses.view(-1, self._num_bodies, 7)[env_ids]
        body_vels = vels.view(-1, self._num_bodies, 6)[env_ids]
        self._data.pos_w.torch[env_ids] = body_poses[..., :3]
        self._data.quat_w.torch[env_ids] = body_poses[..., 3:]
        self._data.vel_w.torch[env_ids] = body_vels[..., :3]
        self._data.ang_vel_w.torch[env_ids] = body_vels[..., 3:]

    def _body_state_is_valid(self) -> bool:
        return getattr(self, "_view", None) is not None

    def _invalidate_backend_impl(self) -> None:
        self._view = None
