from typing import TypeVar

from isaaclab.envs.mdp.actions import JointActionCfg
from isaaclab.managers import SceneEntityCfg

JointOrderCfgT = TypeVar("JointOrderCfgT", SceneEntityCfg, JointActionCfg)


def set_cfg_joint_order(cfg: JointOrderCfgT, joint_order: list[str]) -> JointOrderCfgT:
    """Set the joint order shared by scene-entity and joint-action configurations."""
    setattr(cfg, "joint_names", joint_order.copy())
    setattr(cfg, "preserve_order", True)
    return cfg
