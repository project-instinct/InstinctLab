from __future__ import annotations

from typing import Literal

from isaaclab.utils.configclass import configclass

from .amass_motion_cfg import AmassMotionCfg
from .height_scanned_motion import HeightScannedMotion


@configclass
class HeightScannedMotionCfg(AmassMotionCfg):
    """Configuration for a height-scanned motion buffer.

    This motion buffer assumes the source motion is authored on the z=0 ground
    plane. At simulation time it samples the terrain height from a ray-caster
    height scanner and lifts the motion reference to follow the measured terrain.
    """

    class_type: type | str = "{DIR}.height_scanned_motion:HeightScannedMotion"

    height_scanner_name: str = "height_scanner"
    """Name of the ray-caster sensor in the scene used to sample terrain height."""

    height_sampling_mode: Literal["mean", "nearest"] = "mean"
    """How terrain height is sampled along the motion.

    - ``"mean"``: use the mean valid scanned height for each environment.
      This produces a constant mean-height trajectory offset.
    - ``"nearest"``: nearest-neighbor interpolate scanned heights at each
      reference-frame XY and optionally smooth along the motion.
    """

    height_smoothing_window: int = 1
    """Moving-average window used only in ``"nearest"`` mode."""

    height_offset: float = 0.0
    """Additional vertical offset added on top of the scanned terrain height."""

    debug_height_offset: bool = False
    """Print the computed height offset values during reference filling."""
