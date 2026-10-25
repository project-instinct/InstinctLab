from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal, Sequence

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

import torch

from instinctlab.motion_reference.motion_reference_cfg import MotionBufferCfg

from .amass_motion import AmassMotion


@configclass
class AmassMotionCfg(MotionBufferCfg):
    """Configuration for the AMAS formatted motion data"""

    class_type: type | str = "{DIR}.amass_motion:AmassMotion"

    path: str = MISSING  # type: ignore
    """ the path to the motion dataset.
    A leading `~` is expanded when the file list is built (not in `__post_init__`), so CLI overrides resolve.
    `filtered_motion_selection_filepath` is expanded the same way, and its entries plus `subset_selection` are
    relative to `path`.
    """

    supported_file_endings: Sequence[str] = [
        "poses.npz",
        "stageii.npz",
        "retargetted.npz",
        "retargeted.npz",
        "soma.csv",
    ]
    """ At initialization stage, AmassMotion will walk through `cfg.path` and collect all files ending with
    `supported_file_endings`
    """

    skip_frames: int = 0
    """ The number of frames to skip in the motion data. The data frequency loaded will be
    (1 + skip_frames) * motion_data_frequency.
    """

    motion_interpolate_func: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    """ Considering the possible framerate mismatch between the motion data and RL environment, the function will interpolate the motion data to
    the desired framerate.
    Args:
        - root_trans: the root translation of the motion data. Shape is (N, 3)
        - root_quat: the root rotation of the motion data. Shape is (N, 4)
        - joint_pos: the joint rotation of the motion data. Shape is (N, num_joints)
        - source_framerate: the framerate of the motion data. float scalar
        - target_framerate: the framerate of the RL environment. float scalar
    Returns:
        - root_trans: the root translation of the motion data. Shape is (N, 3)
        - root_quat: the root rotation of the motion data. Shape is (N, 4)
        - joint_pos: the joint rotation of the motion data. Shape is (N, num_joints)
    """

    motion_target_framerate: float = 50.0
    """ The target framerate of the motion data. The motion data will be interpolated to this framerate if the interpolate func is provided.
    """

    assumed_file_framerate: float = 120.0
    """ The assumed framerate to use for motion files that do not store framerate metadata.
    For example, `soma.csv` files use this value directly.
    """

    velocity_estimation_method: Literal["frontward", "backward", "frontbackward", None] = "frontward"
    """ The method to estimate the velocity of the motion data.
    - "frontward": use the frontward difference to estimate the velocity.
    - "backward": use the backward difference to estimate the velocity.
    - "frontbackward": use both frontward and backward difference to estimate the velocity.
    """

    retargetting_func: Callable[..., tuple[torch.Tensor, torch.Tensor]] | type | None = None
    """ the function for the robot model to retarget the motion data to. (from SMPL)
        If None, assuming the motion data is already in the robot model's form,
        and must be a .npz file with "dof_pos", "poses", "mocap_framerate" keys at least.
    """

    retargetting_func_kwargs: dict[str, object] = {}
    """ the kwargs for the retargetting function. """

    filtered_motion_selection_filepath: str | None = None
    """ the path to the filtered motion file list (.yaml file). (Exclusive with `subset_selection`)
    The YAML file should be in the following structure:
    selected_files:
        - motion2_retargeted.npz
        - motion4_retargeted.npz
    weights:
        - 1.0
        - 1.0
    The weights are optional.
        If provided, the number of weights should be the same as the number of selected files.
        If not provided, all motions will be assigned equal weights.
    """

    subset_selection: dict[str, list[str]] | None = None
    """ the subset of the subset selection.
        For example a subset of CMU dataset among the whole AMASS dataset.
        (Exclusive with `filtered_motion_file_list`)
    """

    motion_start_from_middle_range: tuple[float, float] | list[float] = (0.0, 0.0)
    """ The temporal range to start the motion data from the middle of the motion reference buffer.
    """

    env_starting_stub_sampling_strategy: Literal["independent", "concat_motion_bins"] = "independent"
    """ The method to sample the starting stub for the assigned envs.
    - "independent": sample the motion file id and start time independently.
    - "concat_motion_bins": suppose we have the concatenated motion bins for all motions in this buffer, we sample which
        bin to use as if all motions are concatenated into a single motion.
    """

    motion_start_height_offset: float = 0.0
    """ The height offset for `fill_init_reference_state` to prevent penetration when robot reset.
    """

    buffer_device: Literal["cpu", "output_device"] = "output_device"
    """ the device for the motion buffer. """

    motion_bin_length_s: float | None = None
    """ The length of each bin (segment) in each of the motion files, in seconds. To manipulate the sampled starting point
    when resetting the env. If assigned, the motion_start_from_middle_range will be equal to (0.0, 1.0).
    """

    ensure_link_below_zero_ground: bool = True
    """ Whether to ensure no link is below zero ground by raising the base position if necessary. """
