"""Observation terms exposing randomized simulation parameters per environment.

A dynamics model that only sees state and action cannot represent the fact that the *same* action
produces different outcomes when the friction, the centre of mass or a joint zero-offset differ. The
SuperMimic paper therefore conditions the model on a simulation context alongside state and action.

These terms read the values the randomization events actually realized in the simulator. That is
deliberately different from sampling them again: the events may bucket, clip or apply them with an
operation, and a re-sample would disagree with what physics is running. Read the realized value or
nothing.

The quantities are absolute physical parameters, not offsets from a nominal. Absolute values need no
baseline captured before randomization, which keeps these terms free of initialization-ordering
assumptions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


class rigid_body_friction(ManagerTermBase):
    """Mean friction coefficient of the asset's collision shapes, one value per environment.

    Newton stores a single ``mu`` per shape and the randomization event samples it per shape, so the
    underlying parameter is high dimensional. It is reduced to a mean here because a dynamics model
    only needs to tell "grippier" from "slipperier" worlds; carrying every shape would spend most of
    the context width on near-identical numbers.

    The binding is resolved once: friction is randomized at startup and does not change during an
    episode, so re-reading it every step would buy nothing and cost a backend round trip.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._reported_unavailable = False
        self._binding = self._resolve_binding(env, cfg.params.get("asset_cfg", SceneEntityCfg("robot")))

    @staticmethod
    def _resolve_binding(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg):
        """Warp binding for the per-shape friction, or None if this backend does not expose it."""
        asset = env.scene[asset_cfg.name]
        root_view = getattr(asset, "_root_view", None)
        if root_view is None or not hasattr(root_view, "get_attribute"):
            return None
        try:
            model = env.sim.backend_manager.get_model()
            return root_view.get_attribute("shape_material_mu", model)[:, 0]
        except Exception:
            return None

    def __call__(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        """Returns:
        (num_envs, 1).

        Zeros when the backend does not expose the material buffer. That keeps the term inert rather
        than wrong: a missing context entry trains a slightly less informed model, a fabricated one
        trains it on false information.
        """
        if self._binding is None:
            return torch.zeros(env.num_envs, 1, device=env.device)
        import warp as wp

        mu = wp.to_torch(self._binding)  # (num_envs, num_shapes), a view -- no copy
        return mu.to(device=env.device, dtype=torch.float32).mean(dim=-1, keepdim=True)


def rigid_body_com(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Centre of mass of the selected bodies, in each body's own frame.

    The paper randomizes the base centre of mass; the selection is a parameter so a task can choose a
    different body without this module changing.

    Returns:
        (num_envs, num_bodies * 3), ordered by ``asset_cfg.body_ids``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com = asset.data.body_com_pos_b.torch[:, asset_cfg.body_ids]
    return com.reshape(com.shape[0], -1)


def joint_calibration_offset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-joint calibration offset: the realized default pose minus the nominal one.

    ``randomize_default_joint_pos`` adds this offset to the default pose (and to the action offset) to
    model joint zero errors, and records the nominal pose first, so the difference recovers exactly the
    error the simulator is using.

    Returns:
        (num_envs, num_joints), zero when the event did not run.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos.torch
    nominal = getattr(asset.data, "default_joint_pos_nominal", None)
    if nominal is None:
        return torch.zeros_like(default)
    return (default - nominal)[:, asset_cfg.joint_ids]
