from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_ov.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCamera

from instinctlab.sensors.grouped_ray_caster.flat_target_prim_registry import FlatTargetPrimRegistryMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster import GroupedRayCasterKernelMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera import GroupedRayCasterCameraKernelMixin

if TYPE_CHECKING:
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_cfg import GroupedRayCasterCfg


class OvPhysxGroupedRayCaster(GroupedRayCasterKernelMixin, FlatTargetPrimRegistryMixin, MultiMeshRayCaster):
    """OvPhysX ray caster over per-environment target groups."""

    cfg: GroupedRayCasterCfg


class OvPhysxGroupedRayCasterCamera(
    GroupedRayCasterCameraKernelMixin,
    FlatTargetPrimRegistryMixin,
    MultiMeshRayCasterCamera,
):
    """OvPhysX grouped ray-caster camera."""

    cfg: GroupedRayCasterCameraCfg
