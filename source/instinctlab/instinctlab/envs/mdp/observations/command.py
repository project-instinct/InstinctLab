from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from instinctlab.envs.mdp import ShadowingCommandBase


def generated_commands_slice(
    env: ManagerBasedEnv,
    command_name: str,
    ref_length: int | None = None,
):
    """Return command tensor from the command manager, optionally truncated on the time (frame) axis.

    Args:
        env: The environment instance.
        command_name: Name of the command term in the command manager.
        ref_length: If set, keep only the first ``ref_length`` frames along dim=1. If ``None``, return the full tensor.
    """
    command = env.command_manager.get_command(command_name)
    if ref_length is None:
        return command
    if ref_length > command.shape[1]:
        raise ValueError(
            f"ref_length ({ref_length}) exceeds command time dimension ({command.shape[1]}) "
            f"for command '{command_name}'. Increase motion_reference.num_frames or lower ref_length."
        )
    return command[:, :ref_length]


def command_mask(
    env: ManagerBasedEnv,
    command_name: str,
):
    """
    Args:
        command_name: the name of the command in the env.
    """
    command: ShadowingCommandBase = env.command_manager.get_term(command_name)
    return command.mask.to(torch.float32)
