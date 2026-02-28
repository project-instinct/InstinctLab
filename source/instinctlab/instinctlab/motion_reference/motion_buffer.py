from __future__ import annotations

import os
import yaml
from collections.abc import Sequence
from copy import copy
from typing import TYPE_CHECKING, Callable

import omni.log as omni_log
import omni.physics.tensors.impl.api as physx

import isaaclab.utils.math as math_utils

from .motion_reference_data import MotionReferenceData, MotionReferenceState

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from .motion_reference_cfg import MotionBufferCfg

import torch


class MotionBuffer:
    """This is the base class for motion buffers, which serve similar to Dataset in PyTorch.
    But some of the motion buffer may return the motion reference by generating it on the fly.
    """

    generative: bool = False

    def __init__(
        self,
        cfg: MotionBufferCfg,
        articulation_view: physx.ArticulationView,
        link_of_interests: Sequence[str],
        forward_kinematics_func: Callable[..., torch.Tensor],
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.articulation_view = articulation_view
        """The articulation view to access the robot's state, ie. robot root pose."""
        self.isaac_joint_names = self.articulation_view.shared_metatype.dof_names
        """The joint names in the Isaac Sim physics simulation view."""
        self.link_of_interests = link_of_interests
        self.forward_kinematics_func = forward_kinematics_func
        """The pre-built function to compute the interested link poses.
        - input: joint_pos [N, num_joints],
        - output: link_poses [N, num_links, 7].
        """
        self.device = device

        _joint_limits = self.articulation_view.get_dof_limits()[0]  # remove the env dimension
        self._joint_limits = _joint_limits.to(self.device)

    """
    Properties.
    """

    @property
    def num_trajectories(self) -> int:
        """The number of trajectories in the motion buffer. For generative motion buffer, just fake
        a number.
        """
        raise NotImplementedError

    @property
    def num_link_to_ref(self) -> int:
        """The number of links to reference in the motion buffer."""
        return len(self.link_of_interests)

    @property
    def num_assigned_envs(self) -> int:
        """The number of envs assigned to this motion buffer."""
        return self.assigned_env_slice.stop - self.assigned_env_slice.start

    @property
    def complete_motion_lengths(self) -> torch.Tensor:
        """The motion reference lengths for each assigned env. the unit is in seconds.
        ## NOTE:
        This is typically the length of the motion reference data, which does not depend on the starting time of the env.
        """
        # This is a placeholder. The actual implementation should return the correct motion lengths.
        return torch.ones(self.num_assigned_envs, device=self.device) * torch.inf

    @property
    def assigned_motion_lengths(self) -> torch.Tensor:
        """The motion reference lengths for each assigned env. the unit is in seconds.
        ## NOTE:
        This is the time length from the starting time of the env to the end of the motion reference data.
        It should be shorter than `self.complete_motion_lengths` if the env starts at a later time.
        """
        # This is a placeholder. The actual implementation should return the correct motion lengths.
        return self.complete_motion_lengths

    """
    Operations for startup.
    """

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        """Assuming all motion reference trajectories are stored in a sequential way, the motion manager will call this
        function to enable only the selected trajectories, depending on how multi-processing is implemented.
        By "enabled", it means the motion buffer will generate the motion data for only the enabled trajectories.
        ## Args:
            - traj_ids: The trajectory IDs to enable. start from 0 to `self.num_trajectories - 1` in this motion buffer.
                If None, enable all trajectories.
                If values are out of range, just ignore them. (In other words, no motion enabled if all traj_ids are out of range.)
        """
        pass

    def set_env_ids_assignments(self, env_ids: slice) -> None:
        """Set the environment IDs assignments for the motion buffer. The motion buffer is expected only fill the
        motion reference data for these envs.
        NOTE: Currently only support continuous env IDs. (for simplicity)
        """
        assert env_ids.step is None, "Currently only support continuous env IDs."
        self.assigned_env_slice = env_ids

    """
    Operations for rollouts. (motion update and reset)
    """

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | None, symmetric_augmentation_mask_buffer: torch.Tensor
    ) -> None:
        """Reset the motion buffer for the specified env_ids.
        ## Args:
            - env_ids: The environment IDs to reset the motion buffer, which is among the entire num_envs.
                If None, reset all buffers.
            - symmetric_augmentation_mask_buffer: The buffer to store the symmetric augmentation mask for each env. The
                motion buffer is expected to sample and fill the symmetric augmentation mask for each env. Because some
                of the motions do or do not support symmetric augmentation.
                NOTE: the motion buffer is not responsible for the symmetric augmentation, but only to update the mask for now.
        """
        raise NotImplementedError

    def fill_init_reference_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
        state_buffer: MotionReferenceState,
    ) -> None:
        """Fill the initial reference state to the MotionReferenceState (buffer), but only for the specified env_ids.
        ## Args:
            - env_ids: The environment IDs to fill the initial reference state.
            - env_origins: The origins for each robot/env, which is also affecting the motion reference data, shape (num_envs, 3).
            - state_buffer: The MotionReferenceState to fill the initial reference state, which is a buffer for all envs,
                shape (num_envs, ...)
        """
        raise NotImplementedError

    def fill_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        """Fill the motion data to the MotionReferenceData (buffer), but only for the specified env_ids. The motion_buffer
        should not modify any other data in the MotionReferenceData.
        ## NOTE:
            'time_to_target_frame' are not filled.
            num_env_ids is the length of env_ids, which is the number of envs to deal with in this motion buffer.
            num_envs is the total of envs (robots) during the simulation.
        ## Args:
            - env_ids: The environment IDs to fill the motion data.
            - sample_timestamp: The time (s) to sample from the start of each env, shape (num_env_ids, num_frames).
            - env_origins: The origins for each robot/env, which is also affecting the motion reference data, shape (num_envs, 3).
            - data_buffer: The MotionReferenceData to fill the motion data, which is a buffer for all envs,
                shape (num_envs, num_frames, ...)
        """
        raise NotImplementedError

    """
    Helper functions.
    """

    def env_ids_to_assigned_ids(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """Since the motion buffer only fills the motion reference data for the assigned envs, this function maps the
        env_ids to the assigned_ids. env_ids is the selection among all envs, while assigned_ids is the selection among
        the envs assigned to this motion buffer.
        NOTE: Currently only support continuous assigned env IDs. (for simplicity)
        """
        if env_ids is None:
            return torch.arange(self.num_assigned_envs, device=self.device)
        else:
            return env_ids - self.assigned_env_slice.start

    def env_ids_is_assigned(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Check if the env_ids are assigned to this motion buffer."""
        return (env_ids >= self.assigned_env_slice.start) & (env_ids < self.assigned_env_slice.stop)

    def get_current_motion_identifiers(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> list[str]:
        """Get the motion identifiers for the specified env_ids. The identifier must be unique for each motion reference
        even between different experiment runs. Typically, the identifier is the motion file path or name.
        If the motion buffer is generative, the identifier should be a motion_buffer-specific parameter combination.
        """
        raise NotImplementedError

    def get_current_motion_weights(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """Get the current motion weights for the specified env_ids. The motion weights are used to determine the
        probability of selecting a motion reference for each env.
        If the motion buffer is generative, the weights should be based on the current state of the motion buffer.
        """
        return (
            torch.ones(len(env_ids), device=self.device, dtype=torch.float32)
            if env_ids is not None
            else torch.ones(self.num_assigned_envs, device=self.device, dtype=torch.float32)
        )

    def update_motion_weights(
        self,
        env_ids: torch.Tensor | None = None,
        weight_ratio: float | torch.Tensor = 1.0,
    ) -> None:
        """Update the motion weights for the motion buffer. This is typically used to update the motion weights based on
        the success or failure of the envs.
        ## Args:
            - env_ids: The environment IDs that should be updated. If None, update all envs.
        """
        pass  # Nothing happens by default, but can be overridden by subclasses.

    def match_scene(self, scene: InteractiveScene):
        """Let the motion buffer match the scene.
        For example a scene-interactive motion may need to match the motion starting point to the terrain.
        Recommended to call `scene.terrain.terrain_origins` and `scene.terrain.subterrain_specific_cfgs`
        to get some structured terrain information.
        """
        pass
