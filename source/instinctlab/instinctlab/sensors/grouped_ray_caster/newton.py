from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.sensors.ray_caster import LegacyMultiMeshRayCaster, LegacyMultiMeshRayCasterCamera

from instinctlab.sensors.grouped_ray_caster.flat_target_prim_registry import FlatTargetPrimRegistryMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster import GroupedRayCasterKernelMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera import GroupedRayCasterCameraKernelMixin

if TYPE_CHECKING:
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_cfg import GroupedRayCasterCfg


# The sensor body is registered through ``self._backend_manager.cl_register_site`` by Newton's own
# ``_NewtonRayCasterPoseMixin``: identical body patterns share one site, and each instance resolves its
# own site indices from the backend manager. Tracked ray-cast targets bind to Newton bodies through the
# scene-data provider, the same as on PhysX, so no Newton-specific target registration is needed.


class NewtonGroupedRayCaster(
    GroupedRayCasterKernelMixin,
    FlatTargetPrimRegistryMixin,
    LegacyMultiMeshRayCaster,
):
    """Newton ray caster over per-environment target groups."""

    cfg: GroupedRayCasterCfg


class NewtonGroupedRayCasterCamera(
    GroupedRayCasterCameraKernelMixin,
    FlatTargetPrimRegistryMixin,
    LegacyMultiMeshRayCasterCamera,
):
    """Newton grouped ray-caster camera."""

    cfg: GroupedRayCasterCameraCfg
