from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

from instinctlab.utils.perlin import generate_fractal_noise_2d

if TYPE_CHECKING:
    from . import hf_terrains_cfg

from . import hf_terrains_cfg
from .utils import generate_wall


def generate_perlin_noise(difficulty: float, cfg: hf_terrains_cfg.PerlinPlaneTerrainCfg) -> np.ndarray:
    noise_scale = (
        cfg.noise_scale
        if not isinstance(cfg.noise_scale, (list, tuple))
        else interpolate.interp1d([0, 1], cfg.noise_scale, kind="linear")(difficulty)
    )
    return (
        generate_fractal_noise_2d(
            xSize=cfg.size[0],
            ySize=cfg.size[1],
            xSamples=int(cfg.size[0] / cfg.horizontal_scale),
            ySamples=int(cfg.size[1] / cfg.horizontal_scale),
            frequency=cfg.noise_frequency,
            fractalOctaves=cfg.fractal_octaves,
            fractalLacunarity=cfg.fractal_lacunarity,
            fractalGain=cfg.fractal_gain,
            zScale=noise_scale,
            centering=cfg.centering,
        )
        / cfg.vertical_scale
    ).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_plane_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPlaneTerrainCfg) -> np.ndarray:
    return generate_perlin_noise(
        difficulty=difficulty,
        cfg=cfg,
    )  # type: ignore[return-value, arg-type]


@generate_wall
@height_field_to_mesh
def perlin_pyramid_sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPyramidSlopedTerrainCfg) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_sloped_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_sloped_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # create a meshgrid of the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # offset the meshgrid to the center of the terrain
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    # reshape the meshgrid to be 2D
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    # create a sloped surface
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_pyramid_stairs_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPyramidStairsTerrainCfg) -> np.ndarray:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_stairs_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_stairs_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stairs
    step_width = round(cfg.step_width / cfg.horizontal_scale)
    step_height = round(step_height / cfg.vertical_scale)
    # -- platform
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the steps
    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        # increment position
        # -- x
        start_x += step_width
        stop_x -= step_width
        # -- y
        start_y += step_width
        stop_y -= step_width
        # increment height
        current_step_height += step_height
        # add the step
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_discrete_obstacles_terrain(
    difficulty: float, cfg: hf_terrains_cfg.PerlinDiscreteObstaclesTerrainCfg
) -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. For ``obstacle_height_mode``: ``fixed`` uses one height from difficulty; ``choice`` randomizes
    sign/fraction of that height; ``random`` samples each obstacle's height uniformly in
    ``obstacle_height_range`` (meters), independently of ``difficulty``.
    Obstacles are placed randomly; the central square of width :obj:`cfg.platform_width` is cleared (flat platform).

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # ``fixed`` / ``choice``: one reference height from difficulty; ``random``: per-obstacle uniform in range.
    base_height_pix = round(
        (
            cfg.obstacle_height_range[0]
            + difficulty * (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0])
        )
        / cfg.vertical_scale
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles (planar size in pixels; height sampled in the loop)
    obs_width_min = round(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = round(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(1, width_pixels, 4)
    obs_y_range = np.arange(1, length_pixels, 4)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # generate the obstacles
    for _ in range(cfg.num_obstacles):
        # sample size
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice(
                [-base_height_pix, -base_height_pix // 2, base_height_pix // 2, base_height_pix]
            )
        elif cfg.obstacle_height_mode == "fixed":
            height = base_height_pix
        elif cfg.obstacle_height_mode == "random":
            h_m = float(np.random.uniform(cfg.obstacle_height_range[0], cfg.obstacle_height_range[1]))
            height = round(h_m / cfg.vertical_scale)
        else:
            raise ValueError(
                f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. "
                "Must be 'choice', 'fixed', or 'random'."
            )
        width = round(np.random.choice(obs_width_range))
        length = round(np.random.choice(obs_length_range))
        # sample position
        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))
        # clip start position to the terrain
        if x_start + width > width_pixels:
            x_start = width_pixels - width - 1
        if y_start + length > length_pixels:
            y_start = length_pixels - length - 1
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_wave_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinWaveTerrainCfg) -> np.ndarray:
    r"""Generate a terrain with a wave pattern.

    The terrain is a flat platform at the center of the terrain with a wave pattern. The wave pattern
    is generated by adding sinusoidal waves based on the number of waves and the amplitude of the waves.

    The height of the terrain at a point :math:`(x, y)` is given by:

    .. math::

        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right) + \cos\left(\frac{2 \pi y}{\lambda}\right) \right)

    where :math:`A` is the amplitude of the waves, :math:`\lambda` is the wavelength of the waves.

    .. image:: ../../_static/terrains/height_field/wave_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the number of waves is non-positive.
    """
    # check number of waves
    if cfg.num_waves < 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    # resolve terrain configuration
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # compute the wave number: nu = 2 * pi / lambda
    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length
    # create meshgrid for the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the waves
    hf_raw += amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
    # round off the heights to the nearest vertical step

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_stepping_stones_terrain(
    difficulty: float, cfg: hf_terrains_cfg.PerlinSteppingStonesTerrainCfg
) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)
    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            # ensure that stone stops along y-axis
            stop_y = min(length_pixels, start_y + stone_width)
            # randomly sample x-position
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            # fill first stone
            hf_raw[0:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
            # fill row with stones
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_x += stone_width + stone_distance
            # update y-position
            start_y += stone_width + stone_distance
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            # ensure that stone stops along x-axis
            stop_x = min(width_pixels, start_x + stone_width)
            # randomly sample y-position
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            # fill first stone
            hf_raw[start_x:stop_x, 0:stop_y] = np.random.choice(stone_height_range)
            # fill column with stones
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_y += stone_width + stone_distance
            # update x-position
            start_x += stone_width + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    return np.rint(hf_raw).astype(np.int16)


# -- Newly added terrains for parkour -- #


@generate_wall
@height_field_to_mesh
def perlin_parapet_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinParapetTerrainCfg) -> np.ndarray:
    """Generate a terrain with a parapet pattern.

    The terrain is a parapets pattern standing out from the ground.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """

    # resolve terrain configuration

    if isinstance(cfg.parapet_height, (list, tuple)):
        parapet_height = cfg.parapet_height[0] + difficulty * (cfg.parapet_height[1] - cfg.parapet_height[0])
    else:
        parapet_height = cfg.parapet_height

    if isinstance(cfg.parapet_length, (list, tuple)):
        parapet_length = cfg.parapet_length[0] + difficulty * (cfg.parapet_length[1] - cfg.parapet_length[0])
    else:
        parapet_length = cfg.parapet_length

    if cfg.parapet_width is None:
        parapet_width = cfg.size[0]
    else:
        parapet_width = cfg.parapet_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- parapets
    parapet_height = int(parapet_height / cfg.vertical_scale)
    parapet_width = int(parapet_width / cfg.horizontal_scale)
    parapet_length = int(parapet_length / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x = width_pixels // 2 - parapet_width // 2
    start_y = length_pixels // 2 - parapet_length // 2

    end_x = start_x + parapet_width
    end_y = start_y + parapet_length

    hf_raw[start_x:end_x, start_y:end_y] = parapet_height

    if cfg.curved_top_rate is not None and cfg.curved_top_rate > 0.0 and cfg.curved_top_rate > np.random.uniform():
        parapet_height = np.ones(parapet_length, dtype=np.float32) * parapet_height
        parapet_height = parapet_height * (1 - np.square(np.linspace(-1, 1, parapet_length)) * 0.5)
        parapet_height = parapet_height.reshape(-1, 1)

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_gutter_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinGutterTerrainCfg):
    """Generate a terrain with a gutter pattern.

    The terrain is a gutter pattern sinking down from the ground.

    """
    # resolve terrain configuration
    if isinstance(cfg.gutter_length, (list, tuple)):
        gutter_length = cfg.gutter_length[0] + difficulty * (cfg.gutter_length[1] - cfg.gutter_length[0])
    else:
        gutter_length = cfg.gutter_length

    if isinstance(cfg.gutter_depth, (list, tuple)):
        gutter_depth = np.random.uniform(cfg.gutter_depth[0], cfg.gutter_depth[1])
    else:
        gutter_depth = cfg.gutter_depth

    if cfg.gutter_width is None:
        gutter_width = cfg.size[0]
    else:
        gutter_width = cfg.gutter_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- gutters
    gutter_depth = int(gutter_depth / cfg.vertical_scale)
    gutter_width = int(gutter_width / cfg.horizontal_scale)
    gutter_length = int(gutter_length / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))

    start_x = width_pixels // 2 - gutter_width // 2
    start_y = length_pixels // 2 - gutter_length // 2

    end_x = start_x + gutter_width
    end_y = start_y + gutter_length

    hf_raw[start_x:end_x, start_y:end_y] = -gutter_depth

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_stairs_up_down_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinStairsUpDownTerrainCfg):
    """Generate a terrain with stairs going up and down."""
    # resolve terrain configuration
    if isinstance(cfg.per_step_height, (list, tuple)):
        per_step_height = cfg.per_step_height[0] + difficulty * (cfg.per_step_height[1] - cfg.per_step_height[0])
    else:
        per_step_height = cfg.per_step_height
    if isinstance(cfg.per_step_length, (list, tuple)):
        per_step_length = cfg.per_step_length[0] + difficulty * (cfg.per_step_length[1] - cfg.per_step_length[0])
    else:
        per_step_length = cfg.per_step_length
    if isinstance(cfg.num_steps, (list, tuple)):
        num_steps = cfg.num_steps[0] + difficulty * (cfg.num_steps[1] - cfg.num_steps[0])
    else:
        num_steps = cfg.num_steps
    platform_length = cfg.platform_length
    if cfg.per_step_width is None:
        per_step_width = cfg.size[0]
    else:
        per_step_width = cfg.per_step_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- steps
    per_step_height = int(per_step_height / cfg.vertical_scale)
    per_step_width = int(per_step_width / cfg.horizontal_scale)
    per_step_length = int(per_step_length / cfg.horizontal_scale)
    platform_length = int(platform_length / cfg.horizontal_scale)
    num_steps = int(num_steps)
    num_steps = min(num_steps, (length_pixels - platform_length) // (2 * per_step_length))

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    middle_x = width_pixels // 2
    middle_y = length_pixels // 2
    start_x = middle_x - per_step_width // 2
    end_x = start_x + per_step_width
    start_y_up = middle_y - platform_length // 2
    start_y_down = start_y_up + platform_length
    for i in range(num_steps):
        # going up
        start_y = start_y_up - i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_x:end_x, start_y:end_y] = (num_steps - i) * per_step_height

        # going down
        start_y = start_y_down + i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_x:end_x, start_y:end_y] = (num_steps - i) * per_step_height

    # add the platform in the center
    hf_raw[start_x:end_x, start_y_up:start_y_down] = num_steps * per_step_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_stairs_down_up_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinStairsDownUpTerrainCfg):
    """Generate a terrain with stairs going down and up."""
    # resolve terrain configuration
    if isinstance(cfg.per_step_height, (list, tuple)):
        per_step_height = cfg.per_step_height[0] + difficulty * (cfg.per_step_height[1] - cfg.per_step_height[0])
    else:
        per_step_height = cfg.per_step_height
    if isinstance(cfg.per_step_length, (list, tuple)):
        per_step_length = cfg.per_step_length[0] + difficulty * (cfg.per_step_length[1] - cfg.per_step_length[0])
    else:
        per_step_length = cfg.per_step_length
    if isinstance(cfg.num_steps, (list, tuple)):
        num_steps = cfg.num_steps[0] + difficulty * (cfg.num_steps[1] - cfg.num_steps[0])
    else:
        num_steps = cfg.num_steps

    platform_length = cfg.platform_length
    if cfg.per_step_width is None:
        per_step_width = cfg.size[0]
    else:
        per_step_width = cfg.per_step_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- steps
    per_step_height = int(per_step_height / cfg.vertical_scale)
    per_step_width = int(per_step_width / cfg.horizontal_scale)
    per_step_length = int(per_step_length / cfg.horizontal_scale)
    platform_length = int(platform_length / cfg.horizontal_scale)
    num_steps = int(num_steps)
    num_steps = min(num_steps, (length_pixels - platform_length) // (2 * per_step_length))

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    middle_x = width_pixels // 2
    middle_y = length_pixels // 2
    start_x = middle_x - per_step_width // 2
    end_x = start_x + per_step_width
    start_y_up = middle_y - platform_length // 2
    start_y_down = start_y_up + platform_length
    for i in range(num_steps):
        # going up
        start_y = start_y_up - i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_x:end_x, start_y:end_y] = -(num_steps - i) * per_step_height

        # going down
        start_y = start_y_down + i * per_step_length
        end_y = start_y + per_step_length
        hf_raw[start_x:end_x, start_y:end_y] = -(num_steps - i) * per_step_height

    # add the platform in the center
    hf_raw[start_x:end_x, start_y_up:start_y_down] = -num_steps * per_step_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_tilt_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinTiltTerrainCfg):
    """Generate a terrain with a tilt pattern.

    The terrain is a tilted plane with a slope that increases with difficulty.
    """

    # resolve terrain configuration
    if isinstance(cfg.wall_length, (list, tuple)):
        wall_length = np.random.uniform(*cfg.wall_length)
    else:
        wall_length = cfg.wall_length

    if isinstance(cfg.wall_opening_width, (list, tuple)):
        wall_opening_width = cfg.wall_opening_width[1] - difficulty * (
            cfg.wall_opening_width[1] - cfg.wall_opening_width[0]
        )
    else:
        wall_opening_width = cfg.wall_opening_width

    if isinstance(cfg.wall_opening_angle, (list, tuple)):
        wall_opening_angle = np.random.uniform(*cfg.wall_opening_angle)
    else:
        wall_opening_angle = cfg.wall_opening_angle

    if isinstance(cfg.wall_height, (list, tuple)):
        wall_height = np.random.uniform(*cfg.wall_height)
    else:
        wall_height = cfg.wall_height

    if cfg.wall_width is None:
        wall_width = cfg.size[0]
    else:
        wall_width = cfg.wall_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # -- walls
    wall_length = int(wall_length / cfg.horizontal_scale)
    wall_opening_width = int(wall_opening_width / cfg.horizontal_scale)
    wall_height = int(wall_height / cfg.vertical_scale)
    wall_width = int(wall_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x = width_pixels // 2 - wall_width // 2
    end_x = start_x + wall_width
    start_y = length_pixels // 2 - wall_length // 2
    end_y = start_y + wall_length
    middle_x = width_pixels // 2

    # generate the wall
    hf_raw[start_x:end_x, start_y:end_y] = wall_height

    # generate the wall opening
    if wall_opening_angle > 0:
        for length_px in range(wall_length + 1):
            current_y = end_y - length_px
            open_width = int(length_px * np.tan(np.deg2rad(wall_opening_angle)))
            open_width_start = middle_x - wall_opening_width // 2 - open_width
            open_width_end = open_width_start + wall_opening_width + 2 * open_width

            open_width_start = max(open_width_start, 0)
            open_width_end = min(open_width_end, width_pixels)
            current_y = max(current_y, 0)

            hf_raw[open_width_start:open_width_end, current_y] = 0
    else:
        open_width_start = middle_x - wall_opening_width // 2
        open_width_end = open_width_start + wall_opening_width
        open_width_start = max(open_width_start, 0)
        open_width_end = min(open_width_end, width_pixels)
        hf_raw[open_width_start:open_width_end, start_y:end_y] = 0

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_tilted_ramp_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinTiltedRampTerrainCfg):
    """Generate a terrain with a tilted ramp pattern."""

    # resolve terrain configuration
    if isinstance(cfg.tilt_angle, (list, tuple)):
        tilt_angle = (1 - difficulty) * cfg.tilt_angle[0] + difficulty * cfg.tilt_angle[1]
    else:
        tilt_angle = cfg.tilt_angle

    if isinstance(cfg.tilt_height, (list, tuple)):
        tilt_height = np.random.uniform(cfg.tilt_height[0], cfg.tilt_height[1])
    else:
        tilt_height = cfg.tilt_height

    if isinstance(cfg.tilt_width, (list, tuple)):
        tilt_width = np.random.uniform(cfg.tilt_width[0], cfg.tilt_width[1])
    else:
        tilt_width = cfg.tilt_width

    if isinstance(cfg.tilt_length, (list, tuple)):
        tilt_length = (1 - difficulty) * cfg.tilt_length[0] + difficulty * cfg.tilt_length[1]
    else:
        tilt_length = cfg.tilt_length

    if isinstance(cfg.switch_spacing, (list, tuple)):
        if cfg.spacing_curriculum:
            switch_spacing = (1 - difficulty) * cfg.switch_spacing[0] + difficulty * cfg.switch_spacing[1]
        else:
            switch_spacing = np.random.uniform(*cfg.switch_spacing)
    else:
        switch_spacing = cfg.switch_spacing

    # switch parameters to discrete units
    # --terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # --ramp
    tilt_height = int(tilt_height / cfg.vertical_scale)
    tilt_width = int(tilt_width / cfg.horizontal_scale)
    tilt_length = int(tilt_length / cfg.horizontal_scale)
    width_empty_space = int((width_pixels - tilt_width) / 2)

    hf_raw = np.zeros((width_pixels, length_pixels))
    tilt_angle = np.deg2rad(tilt_angle)

    if switch_spacing > 0.0:
        switch_spacing_px = int(switch_spacing / cfg.horizontal_scale)
        assert cfg.overlap_size is not None, "No overlap_size specified."
        overlap_size_px = int(cfg.overlap_size / cfg.horizontal_scale)
        start_left_px = int(width_pixels / 2 + overlap_size_px / 2)
        start_right_px = int(width_pixels / 2 - overlap_size_px / 2)
        tilt_left = True if np.random.uniform() < 0.5 else False
        start_y_px = 1
        end_y_px = start_y_px + switch_spacing_px
        while end_y_px < length_pixels and end_y_px <= tilt_length:
            if tilt_left:
                tilt_left_px = start_left_px - width_empty_space
                tilt_profile = (
                    np.linspace(tilt_left_px, 0, tilt_left_px)
                    * np.tan(tilt_angle)
                    * cfg.horizontal_scale
                    / cfg.vertical_scale
                    + tilt_height
                )
                y_size = end_y_px - start_y_px
                tilt_profile_2d = np.tile(tilt_profile[:, np.newaxis], (1, y_size))
                hf_raw[width_empty_space:start_left_px, start_y_px:end_y_px] += tilt_profile_2d
            else:
                tilt_right_px = width_pixels - start_right_px - width_empty_space
                tilt_profile = (
                    np.linspace(0, tilt_right_px, tilt_right_px)
                    * np.tan(tilt_angle)
                    * cfg.horizontal_scale
                    / cfg.vertical_scale
                    + tilt_height
                )
                y_size = end_y_px - start_y_px
                tilt_profile_2d = np.tile(tilt_profile[:, np.newaxis], (1, y_size))
                hf_raw[start_right_px:-width_empty_space, start_y_px:end_y_px] += tilt_profile_2d
            start_y_px = end_y_px
            end_y_px = start_y_px + switch_spacing_px
            tilt_left = not tilt_left
    else:
        # two-side tilted ramp
        start_px = int(width_pixels / 2)
        tilt_left_px = start_px - width_empty_space
        tilt_right_px = width_pixels - start_px - width_empty_space
        start_y_px = int((length_pixels - tilt_length) / 2)
        end_y_px = start_y_px + tilt_length
        tilt_profile = (
            np.linspace(tilt_left_px, 0, tilt_left_px) * np.tan(tilt_angle) * cfg.horizontal_scale / cfg.vertical_scale
            + tilt_height
        )
        tilt_Profile_2d = np.tile(tilt_profile[:, np.newaxis], (1, end_y_px - start_y_px))
        hf_raw[width_empty_space:start_px, start_y_px:end_y_px] += tilt_Profile_2d
        right_tilt_profile = (
            np.linspace(0, tilt_right_px, tilt_right_px)
            * np.tan(tilt_angle)
            * cfg.horizontal_scale
            / cfg.vertical_scale
            + tilt_height
        )
        right_tilt_profile_2d = np.tile(right_tilt_profile[:, np.newaxis], (1, end_y_px - start_y_px))
        hf_raw[start_px : width_pixels - width_empty_space, start_y_px:end_y_px] += right_tilt_profile_2d

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_slope_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSlopeUpDownTerrainCfg) -> np.ndarray:
    """Generate a terrain with a slope going up and down."""
    # resolve terrain configuration
    if isinstance(cfg.slope_angle, (list, tuple)):
        slope_angle = cfg.slope_angle[0] + difficulty * (cfg.slope_angle[1] - cfg.slope_angle[0])
    else:
        slope_angle = cfg.slope_angle
    if isinstance(cfg.per_slope_length, (list, tuple)):
        slope_length = cfg.per_slope_length[0] + difficulty * (cfg.per_slope_length[1] - cfg.per_slope_length[0])
    else:
        slope_length = cfg.per_slope_length
    platform_length = cfg.platform_length
    if cfg.slope_width is None:
        slope_width = cfg.size[0]
    else:
        slope_width = cfg.slope_width

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- slopes
    slope_angle = np.deg2rad(slope_angle)
    slope_width = int(slope_width / cfg.horizontal_scale)
    slope_length = int(slope_length / cfg.horizontal_scale)
    platform_length = int(platform_length / cfg.horizontal_scale)

    slope_height = int(slope_length * np.tan(slope_angle) * cfg.horizontal_scale / cfg.vertical_scale)

    if cfg.up_down == False:
        slope_height = -slope_height
    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    middle_x = width_pixels // 2
    middle_y = length_pixels // 2
    start_x = middle_x - slope_width // 2
    end_x = start_x + slope_width
    end_y_up = middle_y - platform_length // 2
    start_y_up = end_y_up - slope_length
    start_y_down = end_y_up + platform_length
    end_y_down = start_y_down + slope_length

    up_slope_profile = np.linspace(0, slope_height, slope_length)
    down_slope_profile = np.linspace(slope_height, 0, slope_length)

    # going up
    hf_raw[start_x:end_x, start_y_up:end_y_up] = np.tile(up_slope_profile, (slope_width, 1))
    # going down
    hf_raw[start_x:end_x, start_y_down:end_y_down] = np.tile(down_slope_profile, (slope_width, 1))
    # add the platform in the center
    hf_raw[start_x:end_x, end_y_up:start_y_down] = slope_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_cross_stone_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinCrossStoneTerrainCfg) -> np.ndarray:
    """Generate a terrain with a cross stone pattern."""
    # resolve terrain configuration
    stone_size = cfg.stone_size
    if isinstance(cfg.stone_height, (list, tuple)):
        stone_height = cfg.stone_height[0] + difficulty * (cfg.stone_height[1] - cfg.stone_height[0])
    else:
        stone_height = cfg.stone_height
    if isinstance(cfg.stone_spacing, (list, tuple)):
        stone_spacing = cfg.stone_spacing[0] + difficulty * (cfg.stone_spacing[1] - cfg.stone_spacing[0])
    else:
        stone_spacing = cfg.stone_spacing
    ground_depth = cfg.ground_depth
    platform_width = cfg.platform_width
    xy_random_ratio = cfg.xy_random_ratio

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stone_width = int(stone_size[0] / cfg.horizontal_scale)
    stone_length = int(stone_size[1] / cfg.horizontal_scale)
    stone_height = int(stone_height / cfg.vertical_scale)
    stone_spacing = int(stone_spacing / cfg.horizontal_scale)
    ground_depth = int(ground_depth / cfg.vertical_scale)
    platform_width = int(platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw += ground_depth

    # create platform (square in the center)
    platform_start_x = (width_pixels - platform_width) // 2
    platform_end_x = platform_start_x + platform_width
    platform_start_y = (length_pixels - platform_width) // 2
    platform_end_y = platform_start_y + platform_width
    hf_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = 0

    # generate stone positions
    # Get platform center coordinates
    platform_center_x = width_pixels // 2
    platform_center_y = length_pixels // 2

    # Create cross pattern from platform center extending in four directions

    # From platform extending left (moving away from center)
    x = platform_start_x - stone_spacing - stone_width
    while x >= 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(x + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height
        x -= stone_width + stone_spacing

    # Add one more stone in the remaining space if there's room
    if x + stone_width + stone_spacing > 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(0 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height

    # From platform extending right (moving away from center)
    x = platform_end_x + stone_spacing
    while x + stone_width <= width_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(x + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height
        x += stone_width + stone_spacing

    # Add one more stone in the remaining space if there's room
    if x - stone_width - stone_spacing < width_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(width_pixels - stone_width + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(platform_center_y - stone_length // 2 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height

    # From platform extending up (moving away from center)
    y = platform_start_y - stone_spacing - stone_length
    while y >= 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(y + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height
        y -= stone_length + stone_spacing

    # Add one more stone in the remaining space if there's room
    if y + stone_length + stone_spacing > 0:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(0 + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height

    # From platform extending down (moving away from center)
    y = platform_end_y + stone_spacing
    while y + stone_length <= length_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(y + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height
        y += stone_length + stone_spacing

    # Add one more stone in the remaining space if there's room
    if y - stone_length - stone_spacing < length_pixels:
        if xy_random_ratio > 0.0:
            rand_x = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
            rand_y = int((np.random.uniform() - 0.5) * stone_spacing * xy_random_ratio)
        else:
            rand_x, rand_y = 0, 0
        stone_x1 = np.clip(platform_center_x - stone_width // 2 + rand_x, 0, width_pixels - stone_width)
        stone_x2 = np.clip(stone_x1 + stone_width, 0, width_pixels)
        stone_y1 = np.clip(length_pixels - stone_length + rand_y, 0, length_pixels - stone_length)
        stone_y2 = np.clip(stone_y1 + stone_length, 0, length_pixels)
        if stone_x2 > stone_x1 and stone_y2 > stone_y1:
            hf_raw[stone_x1:stone_x2, stone_y1:stone_y2] = stone_height

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@generate_wall
@height_field_to_mesh
def perlin_square_gap_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSquareGapTerrainCfg) -> np.ndarray:
    """Generate a terrain with a square gap pattern."""
    # resolve terrain configuration
    gap_distance = cfg.gap_distance_range[0] + difficulty * (cfg.gap_distance_range[1] - cfg.gap_distance_range[0])

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- platform
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the steps
    # compute discrete gap size
    gap_distance_px = round(gap_distance / cfg.horizontal_scale)

    # platform coordinates (centered square)
    platform_start_x = (width_pixels - platform_width) // 2
    platform_end_x = platform_start_x + platform_width
    platform_start_y = (length_pixels - platform_width) // 2
    platform_end_y = platform_start_y + platform_width

    # ensure platform inside bounds
    platform_start_x = max(0, platform_start_x)
    platform_start_y = max(0, platform_start_y)
    platform_end_x = min(width_pixels, platform_end_x)
    platform_end_y = min(length_pixels, platform_end_y)

    # create central platform
    hf_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = 0

    # groove depth in pixels (use cfg.gap_depth if available, else default to one vertical unit)
    gap_depth = np.random.uniform(cfg.gap_depth[0], cfg.gap_depth[1])
    gap_value = -round(gap_depth / cfg.vertical_scale)

    # draw square "spiral" grooves (concentric square borders) outward from the platform
    # Only draw one ring of gap around the platform
    sx = platform_start_x - gap_distance_px
    ex = platform_end_x + gap_distance_px
    sy = platform_start_y - gap_distance_px
    ey = platform_end_y + gap_distance_px

    # clamp to terrain bounds
    sx_clamped = max(0, sx)
    ex_clamped = min(width_pixels, ex)
    sy_clamped = max(0, sy)
    ey_clamped = min(length_pixels, ey)

    t = gap_distance_px  # thickness of the groove

    # Draw the groove border (top, bottom, left, right)
    # Top edge
    hf_raw[sx_clamped:ex_clamped, sy_clamped : sy_clamped + t] = gap_value
    # Bottom edge
    hf_raw[sx_clamped:ex_clamped, max(sy_clamped, ey_clamped - t) : ey_clamped] = gap_value
    # Left edge
    hf_raw[sx_clamped : sx_clamped + t, sy_clamped:ey_clamped] = gap_value
    # Right edge
    hf_raw[max(sx_clamped, ex_clamped - t) : ex_clamped, sy_clamped:ey_clamped] = gap_value
    ey_clamped = min(length_pixels, ey)

    t = gap_distance_px  # thickness of the groove

    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        # add perlin noise to the terrain
        hf_raw += perlin_noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)
