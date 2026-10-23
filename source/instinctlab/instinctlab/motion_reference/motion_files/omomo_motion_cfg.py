from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    import torch

from isaaclab.utils.configclass import configclass

from .amass_motion_cfg import AmassMotionCfg
from .omomo_motion import OmomoMotion


@configclass
class OmomoMotionCfg(AmassMotionCfg):
    """Configuration for the OMOMO formatted motion data"""

    class_type: type | str = "{DIR}.omomo_motion:OmomoMotion"

    supported_file_endings: Sequence[str] = ["retargeted.npz", "retargetted.npz"]
    """ At initialization stage, OmomoMotion will walk through `cfg.path` and collect all files ending with
    `supported_file_endings`. `path` is the dataset root (the OMOMO default is `~/Datasets/OMOMO/retargeted`) and is
    expanded when the file list is built, so overriding it after construction still resolves.
    """

    object_interpolate_func: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] | None = None
    """ The function to interpolate the object data to the desired framerate.
    Args:
        - pos: the position of the object data. Shape is (T, N, 3)
        - quat: the rotation of the object data. Shape is (T, N, 4)
        - validity: the validity of the object data. Shape is (T, N)
        - source_framerate: the framerate of the motion data. float scalar
        - target_framerate: the framerate of the RL environment. float scalar
    Returns:
        - interpolated_pos: the interpolated position of the object data. Shape is (T_new, N, 3)
        - interpolated_quat: the interpolated rotation of the object data. Shape is (T_new, N, 4)
        - interpolated_validity: the interpolated validity of the object data. Shape is (T_new, N)
    """
