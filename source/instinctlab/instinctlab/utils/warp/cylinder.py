import numpy as np
import torch
from collections import defaultdict

import warp as wp

from isaaclab.utils.warp import ProxyArray

from instinctlab.utils.warp.kernels import points_penetrate_cylinder_kernel


class CylinderSpatialGrid:
    """This is a class handling cylinders and sort them into a spatial grids.
    Then it can be used to check if points penetrate any of the cylinders.
    """

    def __init__(
        self,
        cylinders: torch.Tensor | np.ndarray,
        num_grid_cells: int = 64**3,
        device: str | torch.device = "cuda",
    ):
        """
        ## Args:
            cylinders: A tensor or numpy array of shape (M, 7) where M is the number of cylinders.
                Each cylinder is defined by its start point (x1, y1, z1), end point (x2, y2, z2), and radius (r).
            num_grid_cells: The number of grid cells to use for spatial partitioning.
                Usually the power of 2, e.g., 64^3 = 262144.
        """
        self.cylinders_np = cylinders if isinstance(cylinders, np.ndarray) else cylinders.cpu().numpy()
        assert (
            self.cylinders_np.shape[1] == 7
        ), "Cylinders should be of shape (M, 7) where M is the number of cylinders."
        self.num_grid_cells = num_grid_cells
        self.device = device

        self.num_cylinders = self.cylinders_np.shape[0]

        self._compute_bounding_box()
        self._create_grid()

    def _compute_bounding_box(self):
        xyz = np.concatenate([self.cylinders_np[:, :3], self.cylinders_np[:, 3:6]], axis=0)
        r_max = self.cylinders_np[:, 6].max()
        bbox_min = xyz.min(axis=0) - r_max  # (3,)
        bbox_max = xyz.max(axis=0) + r_max  # (3,)
        extent = bbox_max - bbox_min  # (3,)

        # Adaptive grid resolution per axis
        scale = extent / extent.max()
        res_f = scale * (self.num_grid_cells ** (1 / 3))
        grid_res = np.maximum(np.round(res_f), 1).astype(int)  # (3,)

        self.grid_res = grid_res
        self.total_num_cells = np.prod(grid_res)
        self.cell_size = extent / grid_res  # (3,)
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def get_flat_grid_idx(self, ix, iy, iz):
        """Get the flat index for the grid cell at (ix, iy, iz).
        Args:
            ix: x index of the grid cell.
            iy: y index of the grid cell.
            iz: z index of the grid cell.
        Returns:
            The flat index of the grid cell.
        """
        if ix < 0 or ix >= self.grid_res[0] or iy < 0 or iy >= self.grid_res[1] or iz < 0 or iz >= self.grid_res[2]:
            return -1
        if self.grid_res[0] <= 0 or self.grid_res[1] <= 0 or self.grid_res[2] <= 0:
            return -1
        # Flatten the 3D grid index to a 1D index
        if self.grid_res[0] * self.grid_res[1] * self.grid_res[2] <= 0:
            return -1
        return ix * self.grid_res[1] * self.grid_res[2] + iy * self.grid_res[2] + iz

    def _create_grid(self):
        grid = defaultdict(list)
        for idx, cyl in enumerate(self.cylinders_np):
            r = cyl[6]
            start = cyl[:3]
            end = cyl[3:6]
            # Compute the bounding box of the cylinder
            bbox_min = np.minimum(start, end) - r
            bbox_max = np.maximum(start, end) + r
            # Get the grid cell indices for the bounding box
            min_cell = np.floor((bbox_min - self.bbox_min) / self.cell_size).astype(int)
            max_cell = np.floor((bbox_max - self.bbox_min) / self.cell_size).astype(int)

            for ix in range(min_cell[0], max_cell[0] + 1):
                for iy in range(min_cell[1], max_cell[1] + 1):
                    for iz in range(min_cell[2], max_cell[2] + 1):
                        flat_grid_idx = self.get_flat_grid_idx(ix, iy, iz)
                        if flat_grid_idx < 0 or flat_grid_idx >= self.total_num_cells:
                            continue
                        grid[flat_grid_idx].append(idx)

        # Convert the grid to a sorted list of indices as in CSR format.
        # `grid` is a dict of lists (with variable length),
        # we need to convert it into a fixed-size array for fast GPU lookups from grid index to cylinder index.
        # For example `cell_indices` is a flat list of all cylinder indices in each grid cell,
        # `cell_indices` is an array of cylinder indices [1,4,5,3,2,5,1,9,7,6,8, ...]
        # `cell_offsets` is an array of offsets for each grid cell, e.g., [0, 3, 5, 7, 11, ...]
        # Then in
        #   cell 0, we have cylinders with indices [1, 4, 5]
        #   cell 1, we have cylinders with indices [3, 2]
        #   cell 2, we have cylinders with indices [5, 1]
        #   cell 3, we have cylinders with indices [9, 7, 6, 8]
        self.cell_offsets = np.zeros(self.total_num_cells + 1, dtype=np.int32)
        self.cell_indices = []
        for i in range(self.total_num_cells):
            self.cell_offsets[i] = len(self.cell_indices)
            self.cell_indices.extend(grid[i])
        self.cell_offsets[-1] = len(self.cell_indices)
        self.cell_indices = np.array(self.cell_indices, dtype=np.int32)

        self.cell_offsets_wp = wp.array(self.cell_offsets, dtype=wp.int32, device=str(self.device))
        self.cell_indices_wp = wp.array(self.cell_indices, dtype=wp.int32, device=str(self.device))
        self.cell_size_wp = wp.vec3(*self.cell_size)
        self.bbox_min_wp = wp.vec3(*self.bbox_min)
        self.grid_res_wp = wp.vec3i(*self.grid_res)
        self.cylinder_start_wp = wp.array(self.cylinders_np[:, :3], dtype=wp.vec3, device=str(self.device))
        self.cylinder_end_wp = wp.array(self.cylinders_np[:, 3:6], dtype=wp.vec3, device=str(self.device))
        self.cylinder_thickness_wp = wp.array(self.cylinders_np[:, 6], dtype=wp.float32, device=str(self.device))

    def get_points_penetration_offset(self, points: ProxyArray) -> ProxyArray:
        """Compute the penetration depth of points into cylinders in the grid.

        ## Args:
            points: A ProxyArray containing ``wp.vec3f`` points. Any array shape is accepted.

        ## Returns:
            A ProxyArray containing the maximum penetration offset for each point.
        """
        if not isinstance(points, ProxyArray):
            raise TypeError(f"Expected points to be a ProxyArray, got {type(points).__name__}.")
        if points.warp.dtype is not wp.vec3f:
            raise TypeError(f"Expected points with wp.vec3f elements, got {points.warp.dtype}.")

        points_wp = points.warp.flatten()
        penetration_offset_wp = wp.zeros(
            points.warp.shape,
            dtype=wp.vec3f,
            device=points.warp.device,
        )

        wp.launch(
            points_penetrate_cylinder_kernel,
            dim=points_wp.shape[0],
            inputs=[
                points_wp,
                self.cylinder_start_wp,
                self.cylinder_end_wp,
                self.cylinder_thickness_wp,
                self.cell_offsets_wp,
                self.cell_indices_wp,
                self.grid_res_wp,
                self.bbox_min_wp,
                self.cell_size_wp,
                penetration_offset_wp.flatten(),
            ],
            device=points.warp.device,
        )

        return ProxyArray(penetration_offset_wp)
