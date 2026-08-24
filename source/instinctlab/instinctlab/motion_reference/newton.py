from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import JointType
from newton.selection import ArticulationView

from isaaclab.utils.warp.math_ops import transform_to_vec_quat

from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase
from instinctlab.utils.prims import resolve_articulation_root_expression

if TYPE_CHECKING:
    from instinctlab.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg


class NewtonMotionReferenceManager(MotionReferenceManagerBase):
    """Motion-reference articulation access implemented directly with Newton views."""

    cfg: MotionReferenceManagerCfg

    def _create_articulation_view(self, prim_path: str):
        root_expr = resolve_articulation_root_expression(prim_path)
        articulation_view = ArticulationView(
            NewtonManager.get_model(),
            root_expr.replace(".*", "*"),
            verbose=False,
            exclude_joint_types=[JointType.FREE, JointType.FIXED],
        )
        NewtonManager.get_physics_sim_view().append(articulation_view)
        return articulation_view

    @property
    def count(self) -> int:
        return self._view.count

    @property
    def max_dofs(self) -> int:
        return self._view.joint_dof_count

    @property
    def joint_dof_names(self) -> list[str]:
        return list(self._view.joint_dof_names)

    def get_root_transforms(self) -> torch.Tensor:
        transforms = self._view.get_root_transforms(NewtonManager.get_state_0())[:, 0]
        positions, quaternions = transform_to_vec_quat(transforms)
        return torch.cat([wp.to_torch(positions), wp.to_torch(quaternions)], dim=-1)

    def get_dof_positions(self) -> torch.Tensor:
        return wp.to_torch(self._view.get_dof_positions(NewtonManager.get_state_0())[:, 0])

    def get_dof_velocities(self) -> torch.Tensor:
        return wp.to_torch(self._view.get_dof_velocities(NewtonManager.get_state_0())[:, 0])

    def get_dof_limits(self) -> torch.Tensor:
        model = NewtonManager.get_model()
        lower = wp.to_torch(self._view.get_attribute("joint_limit_lower", model)[:, 0])[0]
        upper = wp.to_torch(self._view.get_attribute("joint_limit_upper", model)[:, 0])[0]
        return torch.stack([lower, upper], dim=-1).unsqueeze(0)

    def _write_reference_view_state(self, root_pose_w: torch.Tensor, joint_pos: torch.Tensor) -> None:
        state = NewtonManager.get_state_0()
        mask = self._all_env_mask(root_pose_w.device)
        self._reference_view.set_root_transforms(state, self._to_warp(root_pose_w, wp.transformf), mask=mask)
        self._reference_view.set_root_velocities(
            state,
            self._to_warp(torch.zeros_like(root_pose_w[..., :6]), wp.spatial_vectorf),
            mask=mask,
        )
        self._reference_view.set_dof_positions(
            state,
            self._to_warp(joint_pos.unsqueeze(1), wp.float32),
            mask=mask,
        )
        self._reference_view.set_dof_velocities(
            state,
            self._to_warp(torch.zeros_like(joint_pos).unsqueeze(1), wp.float32),
            mask=mask,
        )

    def _all_env_mask(self, device: torch.device) -> wp.array:
        warp_device = "cuda" if device.type == "cuda" else "cpu"
        mask = np.ones(self.count, dtype=np.bool_)
        return wp.from_numpy(mask, dtype=wp.bool, device=warp_device)

    @staticmethod
    def _to_warp(values: torch.Tensor, dtype) -> wp.array:
        warp_device = "cuda" if values.device.type == "cuda" else "cpu"
        data = np.ascontiguousarray(values.detach().cpu().numpy())
        return wp.array(data, dtype=dtype, device=warp_device)

    def _invalidate_backend_impl(self) -> None:
        pass
