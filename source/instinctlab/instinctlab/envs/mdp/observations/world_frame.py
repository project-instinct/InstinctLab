"""World-frame link observations.

The manager's existing link terms return base-frame quantities (``link_pos_b``, ``link_quat_b``), which
are the right inputs for a policy but the wrong ones for a differentiable dynamics model: the reward
compares link poses and velocities in the *world* frame. These four terms expose the raw
``ArticulationData`` world-frame fields so a world-model state group can be assembled from observation
terms, with no second vocabulary for the state layout.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def body_link_pos_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Positions of the selected bodies in the world frame.

    Returns:
        (num_envs, num_bodies, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]


def body_link_quat_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Orientations of the selected bodies in the world frame, (x, y, z, w).

    Returns:
        (num_envs, num_bodies, 4)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]


def body_link_lin_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Linear velocities of the selected bodies in the world frame.

    Returns:
        (num_envs, num_bodies, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]


def body_link_ang_vel_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angular velocities of the selected bodies in the world frame.

    Returns:
        (num_envs, num_bodies, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids]

def body_root_pos_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root position in the *world* frame, i.e. without subtracting the environment origin.

    IsaacLab's own :func:`~isaaclab.envs.mdp.root_pos_w` returns the environment frame, which is the
    right input for a policy but the wrong one next to ``body_link_pos_w`` and every reward term:
    those are all world frame. Mixing the two puts a constant per-environment offset (up to tens of
    metres across a large grid) inside the world-model state, where it silently breaks anything that
    compares a predicted root position against a reference trajectory.

    Returns:
        (num_envs, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch
