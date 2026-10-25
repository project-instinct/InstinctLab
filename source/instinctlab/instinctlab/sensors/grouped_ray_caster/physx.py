from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_physx.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCamera

from instinctlab.sensors.grouped_ray_caster.flat_target_prim_registry import FlatTargetPrimRegistryMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster import GroupedRayCasterKernelMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera import GroupedRayCasterCameraKernelMixin

if TYPE_CHECKING:
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_cfg import GroupedRayCasterCfg


class PhysxGroupedRayCaster(GroupedRayCasterKernelMixin, FlatTargetPrimRegistryMixin, MultiMeshRayCaster):
    """PhysX ray caster over per-environment target groups."""

    cfg: GroupedRayCasterCfg


class PhysxGroupedRayCasterCamera(
    GroupedRayCasterCameraKernelMixin,
    FlatTargetPrimRegistryMixin,
    MultiMeshRayCasterCamera,
):
    """PhysX grouped ray-caster camera."""

    cfg: GroupedRayCasterCameraCfg
