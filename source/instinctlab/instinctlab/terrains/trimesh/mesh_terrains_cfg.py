from dataclasses import MISSING
from typing import List

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from ..height_field import PerlinPlaneTerrainCfg
from . import mesh_terrains


class WallTerrainCfgMixin:
    wall_prob: List[float] = [0.0, 0.0, 0.0, 0.0]  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls


@configclass
class MotionMatchedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for motion-matched terrain generation.

    ## Terrain Mesh Requirements
    - All terrain meshes must have the a border at the bottom.
    - The terrain origin (0, 0, 0) is at the surface of the terrain center, which means that the point should
        be above the terrain at (0, 0, t) given any t > 0 and below the terrain at (0, 0, t) given any t < 0.
    - The USER should ensure that the non-flat part of the terrain is within the size of the terrain.
    """

    function = mesh_terrains.motion_matched_terrain

    path: str = MISSING
    """Directory containing both terrains and the motions, so that these can be matched together.
    """

    metadata_yaml: str = MISSING
    """YAML file containing the motion matching configuration.
    This file should specify the motion matching parameters, such as the motion files to be used,
    the matching criteria, and any other relevant settings.

    You may use the `scripts/motion_matched_metadata_generator.py` to generate the metadata.yaml file if you arrange your
    dataset in the structure as described in `scripts/motion_matched_metadata_generator.py`.

    ## Typical yaml file structure

    ```yaml
    terrains:
        - terrain_id: "jumpbox1" # can be any string.
          terrain_file: "path/to/terrain.stl" # path to the terrain mesh file, relative to the datasetdir.
        - terrain_id: "jumpbox2"
          terrain_file: "path/to/another_terrain.stl"
    motion_files:
        - terrain_id: "jumpbox1" # should match the terrain_id above.
          motion_file: "path/to/motion1_poses.npz" # path to the motion file, relative to the datasetdir.
          weight: (optional) 1.0
        - terrain_id: "jumpbox2"
          motion_file: "path/to/motion2_retargetted.npz"
          weight: (optional) 1.0
    ```

    """

    add_base_plane: bool = True
    """If True, append a full sub-terrain plane (same convention as ``MeshPlaneTerrainCfg``) under the imported mesh."""

    randomize_boxes: bool = False
    """If True, apply per-component size randomization (absolute meter deltas on each axis)."""

    box_size_delta_range_x: tuple[float, float] = (-0.2, 0.2)
    box_size_delta_range_y: tuple[float, float] = (-0.2, 0.2)
    box_size_delta_range_z: tuple[float, float] = (-0.0, 0.0)
    """Absolute size perturbation range (meters) along x/y/z per mesh component.

    For each axis, a value ``delta`` is sampled uniformly from the tuple ``(low, high)``
    and applied as ``size_axis + delta`` before converting to a uniform scale factor.
    """

    box_randomize_prob: float = 1.0
    """Probability to randomize a component. 1.0 means always randomize."""


@configclass
class PerlinMeshFloatingBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a floating box mesh terrain."""

    function = mesh_terrains.floating_box_terrain
    floating_height: tuple[float, float] | float = MISSING
    """The height of the box above the ground. Could be a fixed value or a range (min, max)."""
    box_length: tuple[float, float] | float = MISSING
    """The length of the box along the y-axis. Could be a fixed value or a range (min, max)."""
    box_width: float | None = None
    """The width of the box along the x-axis. If None, it will be equal to the width of the terrain."""
    box_height: tuple[float, float] | float = MISSING
    """The height of the box along the z-axis."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    # values used for perlin noise generation
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = True
    """If True, no perlin noise will be generated exactly below the box."""


@configclass
class SpecifiedBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Single axis-aligned box with fixed size and horizontal placement.

    - ``box_center_offset_xy``: horizontal offset ``(dx, dy)`` of the **box center** from the sub-tile geometric
      center. After IsaacLab's per-tile ``-size/2`` recenter, that is the offset you see relative to env spawn
      (which uses the sub-tile ``origin``, not the box position).
    - The generator returns ``origin = (size[0]/2, size[1]/2, 0)`` (tile center on the ground plane), matching
      ``random_multi_box_terrain``. Do **not** set origin to the box center or ``env_origins`` will follow the box
      and the offset will appear cancelled when the viewer tracks the robot.
    - Mesh construction uses corner-frame center ``(size[0]/2 + dx, size[1]/2 + dy)`` before recentering.

    - ``box_width`` / ``box_length`` / ``box_height``: extents along patch x / y / z in the generator mesh frame.
    - ``floating_height``: clearance from nominal ground plane (``z = 0``) to the bottom face of the box.
    """

    function = mesh_terrains.specified_box_terrain

    box_width: float = MISSING
    """Extent along patch x."""

    box_length: float = MISSING
    """Extent along patch y."""

    box_height: float = MISSING
    """Extent along patch z (up)."""

    box_center_offset_xy: tuple[float, float] = (0.0, 0.0)
    """Box center offset from sub-tile center (meters). Env spawn stays at tile center via fixed ``origin``."""

    floating_height: float = 0.0
    """Vertical gap from ground plane to the bottom of the box."""

    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = True
    """If True, zero perlin under the box footprint (axis-aligned bounding square in UV)."""


@configclass
class PerlinMeshRandomMultiBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a sub terrain with multiple random boxes with perlin noise."""

    function = mesh_terrains.random_multi_box_terrain
    box_height_mean: tuple[float, float] | float = MISSING
    box_height_range: float = MISSING
    box_length_mean: tuple[float, float] | float = MISSING
    box_length_range: float = MISSING
    box_width_mean: tuple[float, float] | float = MISSING
    box_width_range: float = MISSING
    platform_width: float = MISSING

    generation_ratio: float = MISSING

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = False
    box_perlin_cfg: PerlinPlaneTerrainCfg | None = None
    """Used only when perlin_cfg is not None"""
