from __future__ import annotations

import re
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_physx.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCamera

from instinctlab.sensors.grouped_ray_caster.flat_target_prim_registry import FlatTargetPrimRegistryMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster import GroupedRayCasterKernelMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera import GroupedRayCasterCameraKernelMixin

if TYPE_CHECKING:
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_cfg import GroupedRayCasterCfg


class PhysxGroupedRayCasterBackendMixin:
    """PhysX access to tracked grouped-ray-caster target transforms."""

    @staticmethod
    def _tracked_target_count(view) -> int:
        return int(view.count)

    @staticmethod
    def _tracked_target_world_ids(view) -> list[int] | None:
        prim_paths = getattr(view, "prim_paths", None)
        if prim_paths is None or len(prim_paths) != view.count:
            return None

        world_ids = []
        for prim_path in prim_paths:
            match = re.search(r"/env_(\d+)(?:/|$)", str(prim_path))
            if match is None:
                return None
            world_ids.append(int(match.group(1)))
        return world_ids

    @staticmethod
    def _tracked_target_transforms_wp(view) -> wp.array:
        transforms = view.get_transforms()
        if isinstance(transforms, wp.array):
            return transforms.view(wp.transformf)
        return wp.from_torch(transforms.contiguous()).view(wp.transformf)


class PhysxGroupedRayCaster(
    GroupedRayCasterKernelMixin,
    FlatTargetPrimRegistryMixin,
    PhysxGroupedRayCasterBackendMixin,
    MultiMeshRayCaster,
):
    """PhysX ray caster over flat mesh entities grouped by fixed world IDs."""

    cfg: GroupedRayCasterCfg


class PhysxGroupedRayCasterCamera(
    GroupedRayCasterCameraKernelMixin,
    FlatTargetPrimRegistryMixin,
    PhysxGroupedRayCasterBackendMixin,
    MultiMeshRayCasterCamera,
):
    """PhysX grouped ray-caster camera."""

    cfg: GroupedRayCasterCameraCfg
