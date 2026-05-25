from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .pose_velocity_command import PoseVelocityCommand


@configclass
class PoseVelocityCommandCfg(CommandTermCfg):
    """Configuration for the position command generator."""

    class_type: type = PoseVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    velocity_control_stiffness: float = 1.0
    """Scale factor to convert the position error to linear velocity command. Defaults to 1.0."""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    only_positive_lin_vel_x: bool = False
    """Whether to only sample positive linear x velocity commands. Defaults to False."""

    target_mode: str = "flat_patch"
    """Target sampling mode. Use ``flat_patch`` for terrain flat-patch sampling or ``fixed_goal`` for a fixed
    target offset from each environment origin."""

    relative_target_pos: tuple[float, float, float] = (10.0, 0.0, 0.0)
    """Target position offset from each environment origin when ``target_mode`` is ``fixed_goal``.

    The z component is accepted for configuration completeness, but the command target z is pinned to the current
    robot root z when resampling in ``fixed_goal`` mode.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands. Only used in random_velocity_terrains."""

    random_velocity_terrain: list[str] = None
    """List of terrain types for which the velocity commands should be randomized."""

    velocity_ranges: dict = None
    """Dictionary containing velocity ranges for different terrains. If not None, the velocity ranges will be set based on the terrain type."""

    lin_vel_threshold: float = 0.15
    """Minimal threshold for the linear velocity command (in m/s)."""

    ang_vel_threshold: float = 0.15
    """Minimal threshold for the angular velocity command (in rad/s)."""

    lin_vel_metrics_std: float = 0.5
    """Standard deviation for the linear velocity metrics."""

    ang_vel_metrics_std: float = 0.5
    """Standard deviation for the angular velocity metrics."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    target_dis_threshold: float = 0.2
    """The distance threshold to the target position below which the command is set to zero. Defaults to 0.2."""

    flat_patch_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TerrainFlatPatches",
        markers={
            "Goal": sim_utils.CylinderCfg(
                radius=0.15,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "Patches": sim_utils.CylinderCfg(
                radius=0.15,
                height=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """The configuration for the goal pose visualization marker."""

    patch_vis = False
    """Whether to visualize the flat patches."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
