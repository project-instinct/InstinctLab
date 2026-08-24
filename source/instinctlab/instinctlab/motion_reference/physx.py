from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.physics.tensors.api as physx

from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase
from instinctlab.utils.prims import resolve_articulation_root_expression

if TYPE_CHECKING:
    from instinctlab.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg


class PhysxMotionReferenceManager(MotionReferenceManagerBase):
    """Motion-reference articulation access implemented directly with PhysX views."""

    cfg: MotionReferenceManagerCfg

    def _create_articulation_view(self, prim_path: str):
        if not hasattr(self, "_physics_sim_view"):
            self._physics_sim_view = physx.create_simulation_view(self._backend)
            self._physics_sim_view.set_subspace_roots("/")

        root_expr = resolve_articulation_root_expression(prim_path)
        articulation_view = self._physics_sim_view.create_articulation_view(root_expr.replace(".*", "*"))
        if articulation_view._backend is None:
            raise RuntimeError(f"Failed to create a PhysX articulation view at: {root_expr}")
        return articulation_view

    @property
    def count(self) -> int:
        return self._view.count

    @property
    def max_dofs(self) -> int:
        return self._view.max_dofs

    @property
    def joint_dof_names(self) -> list[str]:
        return list(self._view.shared_metatype.dof_names)

    def get_root_transforms(self) -> torch.Tensor:
        return self._view.get_root_transforms()

    def get_dof_positions(self) -> torch.Tensor:
        return self._view.get_dof_positions()

    def get_dof_velocities(self) -> torch.Tensor:
        return self._view.get_dof_velocities()

    def get_dof_limits(self) -> torch.Tensor:
        return self._view.get_dof_limits()

    def _write_reference_view_state(self, root_pose_w: torch.Tensor, joint_pos: torch.Tensor) -> None:
        self._reference_view.set_root_transforms(root_pose_w, indices=self.ALL_INDICES)
        self._reference_view.set_root_velocities(
            torch.zeros_like(root_pose_w[..., :6]),
            indices=self.ALL_INDICES,
        )
        self._reference_view.set_dof_positions(joint_pos, indices=self.ALL_INDICES)
        self._reference_view.set_dof_velocities(
            torch.zeros_like(joint_pos),
            indices=self.ALL_INDICES,
        )

    def _invalidate_backend_impl(self) -> None:
        if hasattr(self, "_physics_sim_view"):
            delattr(self, "_physics_sim_view")
