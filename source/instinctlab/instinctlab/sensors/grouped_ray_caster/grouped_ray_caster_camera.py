from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import isaacsim.core.utils.stage as stage_utils
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import CameraData
from isaaclab.sensors.ray_caster import RayCasterCamera
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view

from instinctlab.utils.warp.raycast import raycast_mesh_grouped

from . import GroupedRayCaster

if TYPE_CHECKING:
    from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg

# import logger
logger = logging.getLogger(__name__)


class GroupedRayCasterCamera(RayCasterCamera, GroupedRayCaster):
    """Grouped ray-cast camera sensor."""

    cfg: GroupedRayCasterCameraCfg

    """The configuration parameters."""
    UNSUPPORTED_TYPES: ClassVar[set[str]] = {
        "rgb",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "semantic_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: GroupedRayCasterCameraCfg):
        """Initializes the camera object.

        Args:
            cfg: The configuration parameters.

        Raises:
            ValueError: If the provided data types are not supported by the grouped-ray-caster camera.
        """
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        # create empty variables for storing output data
        self._data = CameraData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Grouped-Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Implementations.
    """

    def _initialize_warp_meshes(self):
        GroupedRayCaster._initialize_warp_meshes(self)

    def _initialize_rays_impl(self):
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
        )
        self.num_rays = self.ray_directions.shape[1]
        # create buffer to store ray hits
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        # set offsets
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

        self._data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self.device)

        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._create_ray_collision_groups()

    def _update_ray_infos(self, env_ids: Sequence[int]):
        """Updates the ray information buffers."""

        # compute poses from current view
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        # update the data
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        self._data.quat_w_ros[env_ids] = quat_w

        # note: full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_ids)
        self._update_mesh_transforms(env_ids)

        mesh_transforms, mesh_inv_transforms = self._get_mesh_transforms_and_inv_transforms()

        mesh_wp = [i for i in GroupedRayCaster.meshes.values()][0]
        self.ray_hits_w, ray_depth, ray_normal, _, _ = raycast_mesh_grouped(
            mesh_wp_device=mesh_wp.device,
            mesh_wp_ids=self._mesh_wp_ids,
            mesh_transforms=mesh_transforms,
            mesh_inv_transforms=mesh_inv_transforms,
            ray_group_ids=self._ray_collision_groups[env_ids],
            mesh_idxs_for_group=self._mesh_idxs_for_group,
            meah_idxs_slice_for_group=self._meah_idxs_slice_for_group,
            ray_starts=self._ray_starts_w[env_ids],
            ray_directions=self._ray_directions_w[env_ids],
            max_dist=self.cfg.max_distance * 2,  # in case of distance_to_image_plane and ray distance ambiguity
            min_dist=self.cfg.min_distance,
            return_distance=True,
            return_normal=True,
        )
        assert ray_depth is not None
        assert ray_normal is not None

        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # note: data is in camera frame so we only take the first component (z-axis of camera frame)
            distance_to_image_plane = (
                math_utils.quat_apply(
                    math_utils.quat_inv(self._data.quat_w_world[env_ids]).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * self._ray_directions_w[env_ids]),
                )
            )[:, :, 0]
            # apply the maximum distance after the transformation
            if self.cfg.depth_clipping_behavior == "max":
                distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
            elif self.cfg.depth_clipping_behavior == "zero":
                distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
                -1, *self.image_shape, 1
            )

        if "distance_to_camera" in self.cfg.data_types:
            if self.cfg.depth_clipping_behavior == "max":
                ray_depth = torch.clip(ray_depth, max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                ray_depth[ray_depth > self.cfg.max_distance] = 0.0
            self._data.output["distance_to_camera"][env_ids] = ray_depth.view(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            self._data.output["normals"][env_ids] = ray_normal.view(-1, *self.image_shape, 3)
