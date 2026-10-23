# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import numpy as np
import trimesh
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import to_discrete_units

if TYPE_CHECKING:
    from isaaclab.terrains.height_field import HfTerrainBaseCfg


def terrain_origin(height_field: np.ndarray, cfg: HfTerrainBaseCfg) -> np.ndarray:
    """Origin [m] of a generated height field in the local frame of the un-padded field.

    The isaaclab ``height_field_to_mesh`` wrapper pads the generated field with a zero border, and the origin it
    expects is expressed in that padded field. It then shifts the returned origin by the border width. This helper
    reconstructs the padded field and its origin, and removes the border shift again, so that the final origin after
    the wrapper is the EA one: half of the padded terrain size for x and y, and the maximum height over the central
    2 m patch (border zeros included) for z.

    Args:
        height_field: The generated height field, without the border.
        cfg: The configuration of the terrain, with the sub-terrain size set by the wrapper.

    Returns:
        The origin [m] of the terrain, relative to the un-padded height field.
    """
    # the wrapper counts pixels with to_discrete_units() and always adds one pixel per side
    border_pixels = to_discrete_units(cfg.border_width, cfg.horizontal_scale) + 1
    # rebuild the padded field the wrapper converts to a mesh
    padded = np.zeros(
        (height_field.shape[0] + 2 * border_pixels, height_field.shape[1] + 2 * border_pixels), dtype=np.int16
    )
    padded[border_pixels:-border_pixels, border_pixels:-border_pixels] = height_field
    # terrain size as the wrapper derived it, i.e. from the padded pixel count
    size_x = (padded.shape[0] - 1) * cfg.horizontal_scale
    size_y = (padded.shape[1] - 1) * cfg.horizontal_scale
    # central 2 m patch of the padded field
    x1 = int((size_x * 0.5 - 1) / cfg.horizontal_scale)
    x2 = int((size_x * 0.5 + 1) / cfg.horizontal_scale)
    y1 = int((size_y * 0.5 - 1) / cfg.horizontal_scale)
    y2 = int((size_y * 0.5 + 1) / cfg.horizontal_scale)
    origin = np.array([0.5 * size_x, 0.5 * size_y, np.max(padded[x1:x2, y1:y2]) * cfg.vertical_scale])
    # the wrapper adds the border shift back, so express the origin in the un-padded field
    origin[:2] -= border_pixels * cfg.horizontal_scale
    return origin


def generate_wall(func: Callable) -> Callable:
    """Wrapper to add walls to the generated terrain mesh."""

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        meshes, origin = func(difficulty, cfg)
        if cfg is None or not hasattr(cfg, "wall_prob"):
            return meshes, origin

        mesh = meshes[0]
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        result_meshes = [mesh]

        # Get mesh bounds
        bounds = mesh.bounds
        min_bound, max_bound = bounds[0], bounds[1]

        # Left wall
        if np.random.uniform() < cfg.wall_prob[0]:
            left_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [min_bound[0] - wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(left_wall)

        # Right wall
        if np.random.uniform() < cfg.wall_prob[1]:
            right_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [max_bound[0] + wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(right_wall)

        # Front wall
        if np.random.uniform() < cfg.wall_prob[2]:
            front_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, min_bound[1] - wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(front_wall)

        # Back wall
        if np.random.uniform() < cfg.wall_prob[3]:
            back_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, max_bound[1] + wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(back_wall)

        return result_meshes, origin

    return wrapper
