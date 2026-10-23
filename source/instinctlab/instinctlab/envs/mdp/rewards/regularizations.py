from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor


class constant_reward(ManagerTermBase):
    """Constant reward term, used for debugging and testing. You may hack and override this term whenever you want to."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.reward = torch.ones(asset.num_instances, device=env.device, dtype=torch.float) * cfg.params.get(
            "reward", 0.0
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        return self.reward


def motors_power_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    power_j = asset.data.applied_torque.torch * asset.data.joint_vel.torch  # (batch_size, num_joints)
    if normalize_by_stiffness:
        for _, actuator in asset.actuators.items():
            power_j[:, actuator.joint_indices] /= actuator.stiffness
    power_j = power_j[:, asset_cfg.joint_ids]  # (batch_size, num_selected_joints)
    power = torch.sum(torch.square(power_j), dim=-1)  # (batch_size,)
    if normalize_by_num_joints:
        power /= power_j.shape[-1]
    return power


def body_lin_acc_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_num_bodies: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    bodies_acc = torch.norm(
        asset.data.body_lin_acc_w.torch[:, asset_cfg.body_ids, :], dim=-1
    )  # (batch_size, num_bodies)
    body_lin_acc_err = torch.square(bodies_acc)  # (batch_size, num_bodies)
    body_lin_acc_err = torch.sum(body_lin_acc_err, dim=-1)  # (batch_size,)

    if normalize_by_num_bodies:
        body_lin_acc_err /= bodies_acc.shape[-1]

    return body_lin_acc_err


def body_lin_acc_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_num_bodies: bool = False,
    torlerance: float = 10.0,
    sigma: float = 50.0,
    combine_method: str = "prod",
):
    asset: Articulation = env.scene[asset_cfg.name]
    bodies_acc = torch.norm(
        asset.data.body_lin_acc_w.torch[:, asset_cfg.body_ids, :], dim=-1
    )  # (batch_size, num_bodies)
    if torlerance > 0:
        bodies_acc = torch.clamp(bodies_acc - torlerance, min=0.0)
    body_lin_acc_err = torch.square(bodies_acc)  # (batch_size, num_bodies)

    if combine_method == "prod":
        body_lin_acc_err = torch.sum(body_lin_acc_err, dim=-1)  # (batch_size,)
        if normalize_by_num_bodies:
            body_lin_acc_err /= bodies_acc.shape[-1]

    body_lin_acc_err = torch.exp(-body_lin_acc_err / (sigma * sigma))

    if combine_method == "sum":
        body_lin_acc_err = torch.sum(body_lin_acc_err, dim=-1)  # (batch_size,)
        if normalize_by_num_bodies:
            body_lin_acc_err /= bodies_acc.shape[-1]

    return body_lin_acc_err


def action_rate_gauss(
    env: ManagerBasedRLEnv,
    normalize_by_num_actions: bool = False,
    torlerance: float = 0,
    sigma: float = 2.0,
    combine_method: str = "prod",
):
    action_diff_abs = torch.abs(env.action_manager.action - env.action_manager.prev_action)  # (batch_size, num_actions)
    if torlerance > 0:
        action_diff_abs = torch.clamp(action_diff_abs - torlerance, min=0.0)
    action_diff_err = torch.square(action_diff_abs)  # (batch_size, num_actions)

    if combine_method == "prod":
        action_diff_err = torch.sum(action_diff_err, dim=-1)  # (batch_size,)
        if normalize_by_num_actions:
            action_diff_err /= action_diff_abs.shape[-1]

    action_diff_err = torch.exp(-action_diff_err / (sigma * sigma))

    if combine_method == "sum":
        action_diff_err = torch.sum(action_diff_err, dim=-1)  # (batch_size,)
        if normalize_by_num_actions:
            action_diff_err /= action_diff_abs.shape[-1]

    return action_diff_err


def action_rate_order(
    env: ManagerBasedRLEnv,
    order: int = 2,
    action_ids: Sequence[int] | None = None,
):
    action_diff_abs = torch.abs(env.action_manager.action - env.action_manager.prev_action)  # (batch_size, num_actions)
    action_diff_err = torch.pow(action_diff_abs, order)  # (batch_size, num_actions)
    if action_ids is not None:
        action_diff_err = action_diff_err[:, action_ids]

    return torch.sum(action_diff_err, dim=-1)


class action_rate_direction_switch(ManagerTermBase):
    """Reward term for counting the action changes when the moving direction changes.
    Hoping to reduce the action jittering.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_action = torch.zeros_like(env.action_manager.action)
        self._last_action_direction = torch.zeros_like(env.action_manager.action)  # +1 or -1
        self._action_direction_switch_count = torch.zeros_like(env.action_manager.action, dtype=torch.float)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._last_action[env_ids] = 0.0
        self._last_action_direction[env_ids] = 0.0
        self._action_direction_switch_count[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, eps: float = 0.8, mask: bool = False):
        """Compute the action rate directional error.
        Args:
            mask: If True, the error will be masked by the current direction switch mask
        """
        action = env.action_manager.action
        action_direction = torch.sign(action - self._last_action)
        direction_switch = action_direction != self._last_action_direction  # (batch_size, num_actions)
        self._action_direction_switch_count[env.episode_length_buf <= 1] = 0.0  # reset the count at the beginning
        self._action_direction_switch_count *= eps  # decay the count
        self._action_direction_switch_count += direction_switch.float()  # accumulate the count

        action_err = torch.square(action - self._last_action)  # (batch_size, num_actions)
        if mask:
            action_err *= direction_switch.float()
        action_directional_err = torch.sum(action_err * self._action_direction_switch_count, dim=-1)  # (batch_size,)
        if torch.any(action_directional_err < 0):
            # If the action directional error is negative, it means that the action has not changed in the direction.
            # This should not happen, so we set it to zero.
            action_directional_err = torch.clamp(action_directional_err, min=0.0)

        self._last_action[:] = action
        self._last_action_direction[:] = action_direction

        return action_directional_err


class action_rate_direction_consistent(ManagerTermBase):
    """Reward term for encouraging the action rate direction to be consistent with the previous action direction.
    Positive weight encourages longer action rate consistency.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_action = torch.zeros_like(env.action_manager.action)
        self._last_action_direction = torch.zeros_like(env.action_manager.action)  # +1 or -1
        self._action_rate_direction_consistency_count = torch.zeros_like(env.action_manager.action, dtype=torch.float)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._last_action[env_ids] = 0.0
        self._last_action_direction[env_ids] = 0.0
        self._action_rate_direction_consistency_count[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_ids: Sequence[int] | None = None,
        threshold: float = 0.5,  # seconds
        non_positive: bool = False,
        order: int = 1,  # order of the action rate consistency
    ):
        """Compute the action rate directional consistency.
        The reward is positive if the action direction is consistent with the previous action direction.
        Args:
            action_ids: If provided, only compute the reward for the specified action values.
                Due to the action manager design, user should call action_ids rather than some names.
            threshold: The minimum time duration for the action direction to be consistent before starting to reward.
            non_positive: If True, the reward will be non-positive, i.e., the reward will be zero if the action direction is consistent.
        """
        action = env.action_manager.action
        action_direction = torch.sign(action - self._last_action)
        direction_switched = action_direction != self._last_action_direction

        # compute the reward before resetting the count
        action_rate_consistency_time = (
            (self._action_rate_direction_consistency_count - threshold) * direction_switched.float()
        ).reshape(env.num_envs, -1)
        action_rate_consistency_time = (
            action_rate_consistency_time[:, action_ids] if action_ids is not None else action_rate_consistency_time
        )
        rewards = torch.sum(
            torch.pow(torch.abs(action_rate_consistency_time), order) * torch.sign(action_rate_consistency_time), dim=-1
        )  # (num_envs,)
        if non_positive:
            rewards = torch.clamp(rewards, max=0.0)
        # Advance the count
        self._action_rate_direction_consistency_count += env.step_dt  # increment the count

        # If the action direction has switched, we reset the count
        self._action_rate_direction_consistency_count[direction_switched] = 0.0
        self._action_rate_direction_consistency_count[env.episode_length_buf <= 1] = (
            0.0  # reset the count at the beginning
        )

        # Update the last action and direction
        self._last_action[:] = action
        self._last_action_direction[:] = action_direction

        return rewards


def joint_deviation_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_deviation = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(joint_deviation), dim=-1)  # (batch_size,)


class joint_torque_sign_switch(ManagerTermBase):
    """Since policy will seriously jitter and lead to torque sign switch, we need to penalize this behavior."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[self.asset_cfg.name]
        self._last_joint_torque = torch.zeros_like(self.asset.data.applied_torque.torch[:, self.asset_cfg.joint_ids])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        normalize_by_num_joints: bool = False,
    ):
        joint_torque = self.asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
        joint_torque_sign_switch = torch.clip(torch.sign(joint_torque) * torch.sign(-self._last_joint_torque), min=0.0)

        joint_torque_sign_switch_err = torch.sum(joint_torque_sign_switch, dim=-1)  # (batch_size,)

        if normalize_by_num_joints:
            joint_torque_sign_switch_err /= joint_torque.shape[-1]

        # only update the last joint torque for the joints that are not zero
        self._last_joint_torque[joint_torque != 0] = joint_torque[joint_torque != 0]

        return joint_torque_sign_switch_err


def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = False,
    normalize_by_num_joints: bool = False,
):
    """Similar implementation as isaaclab.envs.mdp.rewards.joint_torques_l2, with more options. The default behavior is the same."""
    asset: Articulation = env.scene[asset_cfg.name]
    torques = torch.abs(asset.data.applied_torque.torch)

    if normalize_by_stiffness:
        for _, actuator in asset.actuators.items():
            torques[:, actuator.joint_indices] /= actuator.stiffness
    torques = torques[:, asset_cfg.joint_ids]

    torques = torch.sum(torch.square(torques), dim=-1)

    if normalize_by_num_joints:
        torques /= torques.shape[-1]

    return torques


def joint_torques_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torlerance: float = 0.0,
    sigma: float = 0.4,
    combine_method: str = "prod",
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    torques = torch.abs(asset.data.applied_torque.torch)

    if normalize_by_stiffness:
        for _, actuator in asset.actuators.items():
            torques[:, actuator.joint_indices] /= actuator.stiffness
    torques = torques[:, asset_cfg.joint_ids]

    if torlerance > 0:
        torques = torch.clamp(torques - torlerance, min=0.0)
    torques = torch.square(torques)  # (batch_size, num_joints)

    if combine_method == "prod":
        torques = torch.sum(torques, dim=-1)
        if normalize_by_num_joints:
            torques /= torques.shape[-1]

    torques = torch.exp(-torques / (sigma * sigma))

    if combine_method == "sum":
        torques = torch.sum(torques, dim=-1)
        if normalize_by_num_joints:
            torques /= torques.shape[-1]

    return torques


class joint_torques_direction_switch(ManagerTermBase):
    """Reward term for counting the joint torques changes when the torque direction changes."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[self.asset_cfg.name]
        self._last_torque_direction = torch.sign(self.asset.data.applied_torque.torch[:, self.asset_cfg.joint_ids])
        self._torque_direction_switch_count = torch.zeros_like(
            self.asset.data.applied_torque.torch[:, self.asset_cfg.joint_ids], dtype=torch.float
        )  # (batch_size, num_joints)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._last_torque_direction[env_ids] = 0.0
        self._torque_direction_switch_count[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        eps: float = 0.8,
        mask: bool = False,
    ):
        """Compute the joint torque directional error.
        Args:
            mask: If True, the error will be masked by the current direction switch mask
        """
        joint_torque = self.asset.data.applied_torque.torch[:, self.asset_cfg.joint_ids]
        joint_torque_direction = torch.sign(joint_torque)
        direction_switch = joint_torque_direction != self._last_torque_direction

        self._torque_direction_switch_count[env.episode_length_buf <= 1] = 0.0  # reset the count at the beginning
        self._torque_direction_switch_count *= eps  # decay the count
        self._torque_direction_switch_count += direction_switch.float()  # accumulate the count

        joint_torque_err = torch.square(joint_torque)  # (batch_size, num_joints)
        if mask:
            joint_torque_err *= direction_switch.float()
        joint_torque_directional_err = torch.sum(
            joint_torque_err * self._torque_direction_switch_count, dim=-1
        )  # (batch_size,)
        if torch.any(joint_torque_directional_err < 0):
            # If the joint torque directional error is negative, it means that the joint torque has not changed in the direction.
            # This should not happen, so we set it to zero.
            joint_torque_directional_err = torch.clamp(joint_torque_directional_err, min=0.0)
        self._last_torque_direction[:] = joint_torque_direction

        return joint_torque_directional_err


class joint_acc_l2_step(ManagerTermBase):
    """Reward term for compute joint acceleration based on each environment step, since the joint acc is updated in each
    simulation step and that variable is usually un-stable.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self._last_joint_vel = torch.zeros_like(self.asset.data.joint_vel.torch)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        normalize_by_num_joints: bool = False,
    ):
        step_dt = env.step_dt
        joint_acc = (self.asset.data.joint_vel.torch - self._last_joint_vel) / step_dt
        joint_acc_err = torch.square(joint_acc[:, asset_cfg.joint_ids])  # (batch_size, num_joints)

        if normalize_by_num_joints:
            joint_acc_err = torch.sum(joint_acc_err, dim=-1) / joint_acc.shape[-1]
        else:
            joint_acc_err = torch.sum(joint_acc_err, dim=-1)

        self._last_joint_vel[:] = self.asset.data.joint_vel.torch
        return joint_acc_err

    def reset(self, env_ids: torch.Tensor | slice):
        self._last_joint_vel[env_ids] = 0.0


def joint_acc_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_num_joints: bool = False,
    torlerance: float = 10.0,
    sigma: float = 150,
    combine_method: str = "prod",
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = torch.abs(asset.data.joint_acc.torch[:, asset_cfg.joint_ids])
    if torlerance > 0:
        joint_acc = torch.clamp(joint_acc - torlerance, min=0.0)
    joint_acc_err = torch.square(joint_acc)  # (batch_size, num_joints)

    if combine_method == "prod":
        joint_acc_err = torch.sum(joint_acc_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            joint_acc_err /= joint_acc.shape[-1]

    joint_acc_err = torch.exp(-joint_acc_err / (sigma * sigma))

    if combine_method == "sum":
        joint_acc_err = torch.sum(joint_acc_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            joint_acc_err /= joint_acc.shape[-1]

    return joint_acc_err


class joint_acc_gauss_step(ManagerTermBase):
    """Reward term for compute joint acceleration based on each environment step, since the joint acc is updated in each
    simulation step and that variable is usually un-stable.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self._last_joint_vel = torch.zeros_like(self.asset.data.joint_vel.torch)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        normalize_by_num_joints: bool = False,
        torlerance: float = 10.0,
        sigma: float = 150,
        combine_method: str = "prod",
    ):
        step_dt = env.step_dt
        joint_acc = (self.asset.data.joint_vel.torch - self._last_joint_vel) / step_dt

        joint_acc = torch.abs(joint_acc[:, asset_cfg.joint_ids])
        if torlerance > 0:
            joint_acc = torch.clamp(joint_acc - torlerance, min=0.0)
        joint_acc_err = torch.square(joint_acc)  # (batch_size, num_joints)

        if combine_method == "prod":
            joint_acc_err = torch.sum(joint_acc_err, dim=-1)
            if normalize_by_num_joints:
                joint_acc_err /= joint_acc.shape[-1]

        joint_acc_err = torch.exp(-joint_acc_err / (sigma * sigma))

        if combine_method == "sum":
            joint_acc_err = torch.sum(joint_acc_err, dim=-1)
            if normalize_by_num_joints:
                joint_acc_err /= joint_acc.shape[-1]

        self._last_joint_vel[:] = self.asset.data.joint_vel.torch
        return joint_acc_err

    def reset(self, env_ids: torch.Tensor | slice):
        self._last_joint_vel[env_ids] = 0.0


class joint_acc_direction_switch(ManagerTermBase):
    """Reward term for counting the joint acceleration changes when the moving direction changes.
    Hoping to reduce the joint acceleration jittering.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._last_vel = torch.zeros_like(self.asset.data.joint_vel.torch[:, self.asset_cfg.joint_ids])
        self._last_acc_direction = torch.zeros_like(
            self.asset.data.joint_vel.torch[:, self.asset_cfg.joint_ids]
        )  # +1 or -1
        self._joint_acc_direction_switch_count = torch.zeros_like(
            self.asset.data.joint_vel.torch[:, self.asset_cfg.joint_ids], dtype=torch.float
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._last_vel[env_ids] = 0.0
        self._last_acc_direction[env_ids] = 0.0
        self._joint_acc_direction_switch_count[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        eps: float = 0.8,
        mask: bool = False,
    ):
        """
        Args:
            mask: If True, the error will be masked by the current direction switch mask
        """
        joint_vel = self.asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
        joint_acc_direction = torch.sign(joint_vel - self._last_vel)
        direction_switch = joint_acc_direction != self._last_acc_direction  # (batch_size, num_joints)
        self._joint_acc_direction_switch_count[env.episode_length_buf <= 1] = 0.0  # reset the count at the beginning
        self._joint_acc_direction_switch_count *= eps  # decay the count
        self._joint_acc_direction_switch_count += direction_switch.float()  # accumulate the count

        joint_acc = torch.square(joint_vel - self._last_vel)  # (batch_size, num_joints)
        if mask:
            joint_acc *= direction_switch.float()
        joint_acc_directional = torch.sum(joint_acc * self._joint_acc_direction_switch_count, dim=-1)  # (batch_size,)
        if torch.any(joint_acc_directional < 0):
            # If the joint acceleration directional error is negative, it means that the joint acceleration has not changed in the direction.
            # This should not happen, so we set it to zero.
            joint_acc_directional = torch.clamp(joint_acc_directional, min=0.0)
        self._last_vel[:] = joint_vel
        self._last_acc_direction[:] = joint_acc_direction
        return joint_acc_directional


def joint_vel_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_num_joints: bool = False,
    torlerance: float = 1.0,
    sigma: float = 50.0,
    combine_method: str = "prod",
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
    if torlerance > 0:
        joint_vel = torch.clamp(joint_vel - torlerance, min=0.0)
    joint_vel_err = torch.square(joint_vel)  # (batch_size, num_joints)

    if combine_method == "prod":
        joint_vel_err = torch.sum(joint_vel_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            joint_vel_err /= joint_vel.shape[-1]

    joint_vel_err = torch.exp(-joint_vel_err / (sigma * sigma))

    if combine_method == "sum":
        joint_vel_err = torch.sum(joint_vel_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            joint_vel_err /= joint_vel.shape[-1]

    return joint_vel_err


class joint_vel_direction_switch(ManagerTermBase):
    """Reward term for counting the joint velocity changes when the moving direction changes.
    Hoping to reduce the joint velocity jittering.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._last_vel_direction = torch.sign(self.asset.data.joint_vel.torch[:, self.asset_cfg.joint_ids])
        self._joint_vel_direction_switch_count = torch.zeros_like(
            self.asset.data.joint_vel.torch[:, self.asset_cfg.joint_ids], dtype=torch.float
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._last_vel_direction[env_ids] = 0.0
        self._joint_vel_direction_switch_count[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        eps: float = 0.8,
        mask: bool = False,
    ):
        """Compute the joint velocity directional error.
        Args:
            mask: If True, the error will be masked by the current direction switch mask
        """
        joint_vel = self.asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
        joint_vel_direction = torch.sign(joint_vel)
        direction_switch = joint_vel_direction != self._last_vel_direction
        self._joint_vel_direction_switch_count[env.episode_length_buf <= 1] = 0.0  # reset the count at the beginning
        self._joint_vel_direction_switch_count *= eps  # decay the count
        self._joint_vel_direction_switch_count += direction_switch.float()  # accumulate the count

        joint_vel_err = torch.square(joint_vel)  # (batch_size, num_joints)
        if mask:
            joint_vel_err *= direction_switch.float()
        joint_vel_directional_err = torch.sum(
            joint_vel_err * self._joint_vel_direction_switch_count, dim=-1
        )  # (batch_size,)
        if torch.any(joint_vel_directional_err < 0):
            # If the joint velocity directional error is negative, it means that the joint velocity has not changed in the direction.
            # This should not happen, so we set it to zero.
            joint_vel_directional_err = torch.clamp(joint_vel_directional_err, min=0.0)
        self._last_vel_direction[:] = joint_vel_direction
        return joint_vel_directional_err


def joint_err_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_num_joints: bool = False,
    torlerance: float = 0.57,
    sigma: float = 9.0,
    combine_method: str = "prod",
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_diff = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    joint_err = torch.abs(joint_diff)
    if torlerance > 0:
        joint_err = torch.clamp(joint_err - torlerance, min=0.0)
    joint_err_ = torch.square(joint_err)  # (batch_size, num_joints)

    if combine_method == "prod":
        joint_err_ = torch.sum(joint_err_, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            joint_err_ /= joint_diff.shape[-1]

    joint_err_ = torch.exp(-joint_err_ / (sigma * sigma))

    if combine_method == "sum":
        joint_err_ = torch.sum(joint_err_, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            joint_err_ /= joint_diff.shape[-1]

    return joint_err_


"""
Common Soft limits
"""


def joint_pos_limits_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_num_joints: bool = False,
    normalize_by_limits: bool = False,  # if True, normalize by the joint limits range
    sigma: float = 0.1,
    combine_method: str = "prod",
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    out_of_limits_err = torch.square(out_of_limits)

    if normalize_by_limits:
        out_of_limits_err /= torch.square(
            asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 1]
            - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 0]
        )

    if combine_method == "prod":
        out_of_limits_err = torch.sum(out_of_limits_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            out_of_limits_err /= out_of_limits.shape[-1]

    out_of_limits_err = torch.exp(-out_of_limits_err / (sigma * sigma))

    if combine_method == "sum":
        out_of_limits_err = torch.sum(out_of_limits_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            out_of_limits_err /= out_of_limits.shape[-1]

    return out_of_limits_err


def applied_torque_limits_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,  # whether to normalize by the joint stiffness
    normalize_by_num_joints: bool = False,
    sigma: float = 0.3,
    combine_method: str = "prod",
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(asset.data.applied_torque.torch - asset.data.computed_torque.torch)
    if normalize_by_stiffness:
        for _, actuator in asset.actuators.items():
            out_of_limits[:, actuator.joint_indices] /= actuator.stiffness
    out_of_limits = out_of_limits[:, asset_cfg.joint_ids]  # (batch_size, num_selected_joints)

    out_of_limits_err = torch.square(out_of_limits)

    if combine_method == "prod":
        out_of_limits_err = torch.sum(out_of_limits_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            out_of_limits_err /= out_of_limits.shape[-1]

    out_of_limits_err = torch.exp(-out_of_limits_err / (sigma * sigma))

    if combine_method == "sum":
        out_of_limits_err = torch.sum(out_of_limits_err, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            out_of_limits_err /= out_of_limits.shape[-1]

    return out_of_limits_err


def applied_torque_limits_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,  # whether to normalize by the joint stiffness
    normalize_by_num_joints: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = torch.abs(asset.data.applied_torque.torch - asset.data.computed_torque.torch)
    if normalize_by_stiffness:
        for _, actuator in asset.actuators.items():
            out_of_limits[:, actuator.joint_indices] /= actuator.stiffness
    out_of_limits = out_of_limits[:, asset_cfg.joint_ids]  # (batch_size, num_selected_joints)

    out_of_limits_err = torch.sum(torch.square(out_of_limits), dim=-1)  # (batch_size,)

    if normalize_by_num_joints:
        out_of_limits_err /= out_of_limits.shape[-1]

    return out_of_limits_err


def applied_torque_limits_by_ratio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
):
    """Penalize when the applied torque excceed certain ratio of the joint torque limit."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_effort_limits = asset.data.joint_effort_limits.torch  # (num_envs, num_joints)
    joint_effort_limits = joint_effort_limits[:, asset_cfg.joint_ids]

    applied_torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    applied_torque = torch.abs(applied_torque)

    out_of_limits = (applied_torque - joint_effort_limits * limit_ratio).clip(min=0)
    out_of_limits_err = torch.sum(torch.square(out_of_limits), dim=-1)  # (num_envs,)

    return out_of_limits_err


def contact_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ang_vel_penalty: bool = False,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize body sliding.

    This function penalizes the agent for sliding its body on the ground. The reward is computed as the
    norm of the linear velocity of the body multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the body are in contact with the ground.
    """
    # Penalize body sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > threshold
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    body_ang_vel = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    if ang_vel_penalty:
        reward = reward + torch.sum(body_ang_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def contact_rotate(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize body rotation when in contact with a static object (e.g. the ground).

    This function penalizes the agent for rotating its body when in contact. The reward is computed as the
    norm of the angular velocity of the body multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the body are in contact with a static object.
    """
    # Penalize body rotation
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > threshold
    )
    asset = env.scene[asset_cfg.name]

    body_ang_vel = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_ang_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def contact_switch(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, count_contact: bool = True, count_air: bool = True
) -> torch.Tensor:
    """Reward (penalize) when the sensor detects a contact or a detach."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the contact and detach status
    # is_contact: (batch_size, num_bodies)
    # is_first_contact: (batch_size, num_bodies)
    # is_first_detached: (batch_size, num_bodies)
    is_first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    is_first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    # last_contact_time = contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]

    # If not counting contacts or detachments, set the corresponding tensors to zero
    if not count_contact:
        is_first_contact[:] = 0
    if not count_air:
        is_first_air[:] = 0

    reward = torch.sum(is_first_contact.int() + is_first_air.int(), dim=-1)  # (batch_size,)
    return reward


def contact_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Reward the time the feet are in the air."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # Get the air time
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    # Compute the reward
    reward = torch.sum((last_air_time - threshold) * is_first_contact, dim=-1)  # (batch_size,)
    return reward


def contact_stay_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Reward the time the feet are in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    # Get the contact time
    last_contact_time = contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]
    # Compute the reward
    reward = torch.sum((last_contact_time - threshold) * is_first_air, dim=-1)  # (batch_size,)
    return reward
