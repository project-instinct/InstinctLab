from __future__ import annotations

from typing import TYPE_CHECKING

from instinctlab.utils.backend_dispatch import create_backend_component

if TYPE_CHECKING:
    from instinctlab.sensors.noisy_camera.noisy_raycaster_camera_cfg import NoisyRayCasterCameraCfg


class NoisyRayCasterCamera:
    """Construct the noisy ray-caster camera for the active physics backend."""

    def __new__(cls, cfg: NoisyRayCasterCameraCfg):
        return create_backend_component(
            cfg,
            {
                "physx": "instinctlab.sensors.noisy_camera.physx_ray_caster_cameras:PhysxNoisyRayCasterCamera",
                "newton": "instinctlab.sensors.noisy_camera.newton_ray_caster_cameras:NewtonNoisyRayCasterCamera",
            },
        )
