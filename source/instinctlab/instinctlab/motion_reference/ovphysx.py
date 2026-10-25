from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab_ov.tensor_types as TT
import warp as wp
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView, _expand_env_pattern

from isaaclab.sim.utils import path_expr_to_glob

from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase
from instinctlab.utils.prims import resolve_articulation_root_expression

if TYPE_CHECKING:
    from instinctlab.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg


def _dense(values: torch.Tensor) -> torch.Tensor:
    """A copy with row-major strides: OvPhysX bindings reject strided buffers, and Warp also checks the strides of
    size-1 dimensions, which ``torch.Tensor.contiguous`` leaves as they are (e.g. one environment sliced from a wider
    tensor)."""
    return values.clone(memory_format=torch.contiguous_format)


class OvPhysxMotionReferenceManager(MotionReferenceManagerBase):
    """Motion-reference articulation access implemented with OvPhysX tensor bindings."""

    cfg: MotionReferenceManagerCfg

    def _create_articulation_view(self, prim_path: str) -> OvPhysxView:
        physx_instance = self._backend_manager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        root_expr = resolve_articulation_root_expression(prim_path)
        # One binding over the per-environment roots, in environment order (as Isaac Lab's OvPhysX articulation).
        paths = _expand_env_pattern(path_expr_to_glob(root_expr), self._sim.get_clone_plan())
        view = OvPhysxView(physx_instance, prim_paths=paths, device=self._backend_manager.get_device())
        if view.try_binding_for(TT.ROOT_POSE) is None:
            raise RuntimeError(f"Failed to create an OvPhysX articulation view at: {root_expr}")
        return view

    @property
    def count(self) -> int:
        return self._view.count

    @property
    def max_dofs(self) -> int:
        return self._view.dof_count

    def _view_joint_dof_names(self, view) -> list[str]:
        return list(view.dof_names)

    def get_root_transforms(self) -> torch.Tensor:
        return wp.to_torch(self._view.get_attribute(TT.ROOT_POSE)).reshape(self.count, 7)

    def _get_backend_dof_positions(self) -> torch.Tensor:
        return wp.to_torch(self._view.get_attribute(TT.DOF_POSITION))

    def _get_backend_dof_velocities(self) -> torch.Tensor:
        return wp.to_torch(self._view.get_attribute(TT.DOF_VELOCITY))

    def _get_backend_dof_limits(self) -> torch.Tensor:
        # CPU-only binding, as PhysX's ``get_dof_limits``.
        return wp.to_torch(self._view.get_attribute(TT.DOF_LIMIT)).reshape(self.count, self.max_dofs, 2)

    def _write_backend_reference_view_state(self, root_pose_w: torch.Tensor, joint_pos: torch.Tensor) -> None:
        view = self._reference_view
        root_velocity = torch.zeros(root_pose_w.shape[0], 6, dtype=root_pose_w.dtype, device=root_pose_w.device)
        view.set_attribute(TT.ROOT_POSE, wp.from_torch(_dense(root_pose_w)))
        view.set_attribute(TT.ROOT_VELOCITY, wp.from_torch(root_velocity))
        view.set_attribute(TT.DOF_POSITION, wp.from_torch(_dense(joint_pos)))
        view.set_attribute(TT.DOF_VELOCITY, wp.from_torch(torch.zeros_like(_dense(joint_pos))))

    def _invalidate_backend_impl(self) -> None:
        # OvPhysX closes the bindings of every live view when its runtime is released.
        pass
