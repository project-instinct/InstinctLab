"""Helpers for constructing PhysX tensor views from Isaac Lab prim expressions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source
from isaaclab.utils.warp.math_ops import transform_to_vec_quat


class NewtonArticulationViewAdapter:
    """Adapt a Newton articulation view to the PhysX view API used by InstinctLab."""

    def __init__(self, view):
        self._view = view
        self.count = view.count
        self.max_dofs = view.joint_dof_count
        self.shared_metatype = SimpleNamespace(dof_names=view.joint_dof_names)
        self._backend = "newton"

    def _state_0(self):
        from isaaclab_newton.physics import NewtonManager

        return NewtonManager.get_state_0()

    def get_root_transforms(self):
        transforms = self._view.get_root_transforms(self._state_0())[:, 0]
        positions, quaternions = transform_to_vec_quat(transforms)
        return torch.cat([wp.to_torch(positions), wp.to_torch(quaternions)], dim=-1)

    def get_dof_positions(self):
        return wp.to_torch(self._view.get_dof_positions(self._state_0())[:, 0])

    def get_dof_velocities(self):
        return wp.to_torch(self._view.get_dof_velocities(self._state_0())[:, 0])

    def _mask_from_indices(self, indices, device):
        mask = torch.ones(self.count, dtype=torch.bool, device=device)
        warp_device = "cuda" if device.type == "cuda" else "cpu"
        if indices is None:
            return wp.from_numpy(mask.cpu().numpy(), dtype=wp.bool, device=warp_device)
        indices = torch.as_tensor(indices, device=device).long()
        mask[:] = False
        mask[indices] = True
        return wp.from_numpy(mask.cpu().numpy(), dtype=wp.bool, device=warp_device)

    def _to_warp(self, values, dtype):
        warp_device = "cuda" if values.device.type == "cuda" else "cpu"
        data = np.ascontiguousarray(values.cpu().numpy())
        return wp.array(data, dtype=dtype, device=warp_device)

    def set_root_transforms(self, transforms, indices=None):
        transforms = torch.as_tensor(transforms)
        device = transforms.device
        mask = self._mask_from_indices(indices, device)
        values = transforms[indices] if indices is not None and len(indices) != self.count else transforms
        values_wp = self._to_warp(values, wp.transformf)
        self._view.set_root_transforms(self._state_0(), values_wp, mask=mask)

    def set_root_velocities(self, velocities, indices=None):
        velocities = torch.as_tensor(velocities)
        device = velocities.device
        mask = self._mask_from_indices(indices, device)
        values = velocities[indices] if indices is not None and len(indices) != self.count else velocities
        values_wp = self._to_warp(values, wp.spatial_vectorf)
        self._view.set_root_velocities(self._state_0(), values_wp, mask=mask)

    def set_dof_positions(self, positions, indices=None):
        positions = torch.as_tensor(positions)
        device = positions.device
        mask = self._mask_from_indices(indices, device)
        values = positions[indices] if indices is not None and len(indices) != self.count else positions
        if values.ndim == 2:
            values = values.unsqueeze(1)
        values_wp = self._to_warp(values, wp.float32)
        self._view.set_dof_positions(self._state_0(), values_wp, mask=mask)

    def set_dof_velocities(self, velocities, indices=None):
        velocities = torch.as_tensor(velocities)
        device = velocities.device
        mask = self._mask_from_indices(indices, device)
        values = velocities[indices] if indices is not None and len(indices) != self.count else velocities
        if values.ndim == 2:
            values = values.unsqueeze(1)
        values_wp = self._to_warp(values, wp.float32)
        self._view.set_dof_velocities(self._state_0(), values_wp, mask=mask)

    def get_dof_limits(self):
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        lower = wp.to_torch(self._view.get_attribute("joint_limit_lower", model)[:, 0])[0]
        upper = wp.to_torch(self._view.get_attribute("joint_limit_upper", model)[:, 0])[0]
        return torch.stack([lower, upper], dim=-1).unsqueeze(0)


def get_articulation_view(
    prim_path: str,
    physics_sim_view: Any | None = None,
) -> Any:
    """Create the backend articulation view for a single articulation below an asset expression."""
    from pxr import UsdPhysics

    matches = resolve_matching_prims_from_source(prim_path)
    if not matches:
        raise RuntimeError(f"No asset prim found at path expression: {prim_path}")

    asset_prim, asset_expr = matches[0]
    asset_path = asset_prim.GetPath().pathString
    articulation_roots = get_all_matching_child_prims(
        asset_path,
        predicate=lambda prim: bool(prim.HasAPI(UsdPhysics.ArticulationRootAPI)),
        traverse_instance_prims=False,
    )
    if len(articulation_roots) != 1:
        matched_paths = [prim.GetPath().pathString for prim in articulation_roots]
        raise RuntimeError(
            f"Expected exactly one ArticulationRootAPI prim below '{asset_path}' "
            f"(resolved from '{prim_path}'), found {len(articulation_roots)}: {matched_paths}."
        )

    root_path = articulation_roots[0].GetPath().pathString
    root_expr = asset_expr + root_path[len(asset_path) :]

    sim = SimulationContext.instance()
    manager = sim.physics_manager if sim is not None else None
    manager_name = manager.__name__ if isinstance(manager, type) else str(manager)
    if manager is not None and "newton" in manager_name.lower():
        from isaaclab_newton.physics import NewtonManager

        from newton import JointType
        from newton.selection import ArticulationView

        articulation_view = ArticulationView(
            NewtonManager.get_model(),
            root_expr.replace(".*", "*"),
            verbose=False,
            exclude_joint_types=[JointType.FREE, JointType.FIXED],
        )
        NewtonManager.get_physics_sim_view().append(articulation_view)
        return NewtonArticulationViewAdapter(articulation_view)

    if physics_sim_view is None:
        from isaaclab_physx.physics import PhysxManager

        physics_sim_view = PhysxManager.get_physics_sim_view()
    if physics_sim_view is None:
        raise RuntimeError(f"Failed to create a PhysX articulation view at: {root_expr}")

    articulation_view = physics_sim_view.create_articulation_view(root_expr.replace(".*", "*"))
    if articulation_view._backend is None:
        raise RuntimeError(f"Failed to create a PhysX articulation view at: {root_expr}")
    return articulation_view
