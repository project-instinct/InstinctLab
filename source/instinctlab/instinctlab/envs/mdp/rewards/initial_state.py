from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def _resolve_body_ids(asset: Articulation, asset_cfg: SceneEntityCfg):
    if asset_cfg.body_ids == slice(None):
        return slice(None)
    return torch.as_tensor(asset_cfg.body_ids, device=asset.device, dtype=torch.long)


class _InitialStateRewardBase(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = self._env.scene[self.asset_cfg.name]

    def _new_episode_mask(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env.episode_length_buf <= 1

    def _update_buffer(self, buffer: torch.Tensor | None, values: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
        if buffer is None or buffer.shape != values.shape:
            buffer = torch.zeros_like(values)
        mask = self._new_episode_mask(env)
        if torch.any(mask):
            buffer[mask] = values[mask]
        return buffer


class base_position_imitation_gauss_from_initial(_InitialStateRewardBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._base_pos_init_w = None

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1):
        base_pos = self.asset.data.root_pos_w
        self._base_pos_init_w = self._update_buffer(self._base_pos_init_w, base_pos, env)
        base_pos_diff = base_pos - self._base_pos_init_w
        return torch.exp(-torch.sum(torch.square(base_pos_diff), dim=-1) / (std * std))


class base_rot_imitation_gauss_from_initial(_InitialStateRewardBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._base_quat_init_w = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        difference_type: Literal["axis_angle", "box_minus"] = "axis_angle",
        std: float = 0.3,
    ):
        quat = self.asset.data.root_quat_w
        self._base_quat_init_w = self._update_buffer(self._base_quat_init_w, quat, env)
        quat_ref = self._base_quat_init_w
        if difference_type == "axis_angle":
            quat_diff = math_utils.quat_mul(quat_ref, math_utils.quat_conjugate(quat))
            rot_error = torch.norm(math_utils.axis_angle_from_quat(quat_diff), dim=-1)
        elif difference_type == "box_minus":
            rot_error = torch.norm(math_utils.quat_box_minus(quat_ref, quat), dim=-1)
        else:
            raise ValueError(f"Unsupported difference method: {difference_type}.")
        return torch.exp(-torch.square(rot_error) / (std * std))


class link_pos_imitation_gauss_from_initial(_InitialStateRewardBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._link_pos_init = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        in_base_frame: bool = True,
        in_relative_world_frame: bool = False,
        std: float = 0.1,
        masked: bool = False,
        combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
    ):
        body_ids = _resolve_body_ids(self.asset, asset_cfg)
        links_pos_w = self.asset.data.body_link_pos_w[:, body_ids]
        if in_base_frame:
            root_pos_w = self.asset.data.root_pos_w
            root_quat_w = self.asset.data.root_quat_w
            root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
            links_pos = math_utils.transform_points(links_pos_w, root_pos_w_inv, root_quat_w_inv)
        elif in_relative_world_frame:
            links_pos = links_pos_w - self.asset.data.root_pos_w.unsqueeze(1)
        else:
            links_pos = links_pos_w

        self._link_pos_init = self._update_buffer(self._link_pos_init, links_pos, env)
        link_pos_square = torch.sum(torch.square(links_pos - self._link_pos_init), dim=-1)

        if masked:
            link_mask = torch.ones_like(link_pos_square)
        else:
            link_mask = 1

        if combine_method == "prod":
            link_pos_square = torch.sum(link_pos_square * link_mask, dim=-1)
        if combine_method == "mean_prod":
            link_pos_square = torch.mean(link_pos_square * link_mask, dim=-1)

        rewards = torch.exp(-link_pos_square / (std * std))
        if combine_method == "sum":
            rewards = torch.sum(rewards * link_mask, dim=-1)
        return rewards


class link_rot_imitation_gauss_from_initial(_InitialStateRewardBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._link_rot_init = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        in_base_frame: bool = True,
        in_relative_world_frame: bool = False,
        std: float = 0.4,
        masked: bool = False,
        combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
    ):
        body_ids = _resolve_body_ids(self.asset, asset_cfg)
        links_rot_w = self.asset.data.body_link_quat_w[:, body_ids]
        if in_base_frame or in_relative_world_frame:
            root_quat_w = self.asset.data.root_quat_w
            root_quat_w_inv = math_utils.quat_inv(root_quat_w)
            link_rot = math_utils.quat_mul(root_quat_w_inv.unsqueeze(1).expand(-1, links_rot_w.shape[1], -1), links_rot_w)
        else:
            link_rot = links_rot_w

        self._link_rot_init = self._update_buffer(self._link_rot_init, link_rot, env)
        link_rot_error_magnitude = math_utils.quat_error_magnitude(
            link_rot.reshape(-1, 4), self._link_rot_init.reshape(-1, 4)
        ).reshape(link_rot.shape[0], link_rot.shape[1])
        link_rot_error_square = torch.square(link_rot_error_magnitude)

        if masked:
            link_rot_mask = torch.ones_like(link_rot_error_square)
        else:
            link_rot_mask = 1

        if combine_method == "prod":
            link_rot_error_square = torch.sum(link_rot_error_square * link_rot_mask, dim=-1)
        if combine_method == "mean_prod":
            link_rot_error_square = torch.mean(link_rot_error_square * link_rot_mask, dim=-1)

        rewards = torch.exp(-link_rot_error_square / (std * std))
        if combine_method == "sum":
            rewards = torch.sum(rewards * link_rot_mask, dim=-1)
        return rewards


class link_lin_vel_imitation_gauss_from_initial(_InitialStateRewardBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._link_lin_vel_init = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        in_base_frame: bool = False,
        std: float = 0.4,
        masked: bool = False,
        combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
    ):
        body_ids = _resolve_body_ids(self.asset, asset_cfg)
        links_lin_vel_w = self.asset.data.body_link_lin_vel_w[:, body_ids]
        if in_base_frame:
            root_pos_w = self.asset.data.root_pos_w
            root_quat_w = self.asset.data.root_quat_w
            root_lin_vel_w = self.asset.data.root_lin_vel_w
            root_ang_vel_w = self.asset.data.root_ang_vel_w
            links_pos_w = self.asset.data.body_link_pos_w[:, body_ids]
            link_pos_offset_w = links_pos_w - root_pos_w.unsqueeze(1)
            link_lin_vel = math_utils.quat_apply_inverse(
                root_quat_w.unsqueeze(1).expand(-1, links_lin_vel_w.shape[1], -1),
                links_lin_vel_w - root_lin_vel_w.unsqueeze(1) - torch.cross(root_ang_vel_w.unsqueeze(1), link_pos_offset_w, dim=-1),
            )
        else:
            link_lin_vel = links_lin_vel_w

        self._link_lin_vel_init = self._update_buffer(self._link_lin_vel_init, link_lin_vel, env)
        link_lin_vel_square = torch.sum(torch.square(link_lin_vel - self._link_lin_vel_init), dim=-1)

        if masked:
            link_lin_vel_mask = torch.ones_like(link_lin_vel_square)
        else:
            link_lin_vel_mask = 1

        if combine_method == "prod":
            link_lin_vel_square = torch.sum(link_lin_vel_square * link_lin_vel_mask, dim=-1)
        if combine_method == "mean_prod":
            link_lin_vel_square = torch.mean(link_lin_vel_square * link_lin_vel_mask, dim=-1)

        rewards = torch.exp(-link_lin_vel_square / (std * std))
        if combine_method == "sum":
            rewards = torch.sum(rewards * link_lin_vel_mask, dim=-1)
        return rewards


class link_ang_vel_imitation_gauss_from_initial(_InitialStateRewardBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._link_ang_vel_init = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        in_base_frame: bool = False,
        std: float = 0.4,
        masked: bool = False,
        combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
    ):
        body_ids = _resolve_body_ids(self.asset, asset_cfg)
        links_ang_vel_w = self.asset.data.body_link_ang_vel_w[:, body_ids]
        if in_base_frame:
            root_quat_w = self.asset.data.root_quat_w
            root_ang_vel_w = self.asset.data.root_ang_vel_w
            link_ang_vel = math_utils.quat_apply_inverse(
                root_quat_w.unsqueeze(1).expand(-1, links_ang_vel_w.shape[1], -1),
                links_ang_vel_w - root_ang_vel_w.unsqueeze(1),
            )
        else:
            link_ang_vel = links_ang_vel_w

        self._link_ang_vel_init = self._update_buffer(self._link_ang_vel_init, link_ang_vel, env)
        link_ang_vel_square = torch.sum(torch.square(link_ang_vel - self._link_ang_vel_init), dim=-1)

        if masked:
            link_ang_vel_mask = torch.ones_like(link_ang_vel_square)
        else:
            link_ang_vel_mask = 1

        if combine_method == "prod":
            link_ang_vel_square = torch.sum(link_ang_vel_square * link_ang_vel_mask, dim=-1)
        if combine_method == "mean_prod":
            link_ang_vel_square = torch.mean(link_ang_vel_square * link_ang_vel_mask, dim=-1)

        rewards = torch.exp(-link_ang_vel_square / (std * std))
        if combine_method == "sum":
            rewards = torch.sum(rewards * link_ang_vel_mask, dim=-1)
        return rewards
