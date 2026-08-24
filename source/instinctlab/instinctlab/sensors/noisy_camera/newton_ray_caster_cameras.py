from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.sensors.ray_caster import MultiMeshRayCasterCamera, RayCasterCamera

from instinctlab.sensors.grouped_ray_caster.newton import NewtonGroupedRayCasterCamera
from instinctlab.sensors.noisy_camera.noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from instinctlab.sensors.noisy_camera.noisy_grouped_raycaster_camera_cfg import NoisyGroupedRayCasterCameraCfg
    from instinctlab.sensors.noisy_camera.noisy_multi_mesh_ray_caster_camera_cfg import NoisyMultiMeshRayCasterCameraCfg
    from instinctlab.sensors.noisy_camera.noisy_raycaster_camera_cfg import NoisyRayCasterCameraCfg


class NewtonNoisyRayCasterCamera(NoisyCameraMixin, RayCasterCamera):
    """Newton noisy ray-caster camera."""

    cfg: NoisyRayCasterCameraCfg


class NewtonNoisyMultiMeshRayCasterCamera(NoisyCameraMixin, MultiMeshRayCasterCamera):
    """Newton noisy multi-mesh ray-caster camera."""

    cfg: NoisyMultiMeshRayCasterCameraCfg


class NewtonNoisyGroupedRayCasterCamera(NoisyCameraMixin, NewtonGroupedRayCasterCamera):
    """Newton noisy grouped ray-caster camera."""

    cfg: NoisyGroupedRayCasterCameraCfg
