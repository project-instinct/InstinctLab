from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .root_position_trajectory_command import RootPositionTrajectoryCommand


@configclass
class RootPositionTrajectoryCommandCfg(CommandTermCfg):
    """World-frame root XY reference along flat-patch waypoints (axis-aligned integration)."""

    class_type: type = RootPositionTrajectoryCommand

    asset_name: str = MISSING
    """Scene articulation name receiving the command."""

    resampling_time_range: tuple[float, float] = (30.0, 45.0)
    """Episode resampling interval — long enough to finish most waypoint tours before reset."""

    speed: float = 0.2
    """Planner speed for world XY axes (m/s)."""

    num_waypoints: int = 8
    """Greedy nearest-neighbor patch count (capped by available flat patches and storage)."""

    max_waypoints_stored: int = 64
    """Row capacity in the waypoint buffer `(num_envs, max_waypoints_stored, 3)`."""

    target_patch_key: str = "target"
    """Must exist in ``TerrainImporter.flat_patches`` (see ``FlatPatchSamplingCfg`` injection)."""

    arrival_tolerance: float = 0.02
    """Meters: treat axis target as reached inside this band."""

    command_z_offset_from_root: float = 0.0
    """Adds to current robot root z when writing the command ``z`` component."""

    pos_metrics_std: float = 0.35
    """Std (m) used in ``track_pos_xy_exp`` logging metric."""
