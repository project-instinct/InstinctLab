from __future__ import annotations

from typing import TYPE_CHECKING

from instinctlab.utils.backend_dispatch import create_backend_component

if TYPE_CHECKING:
    from instinctlab.sensors.noisy_camera.noisy_grouped_raycaster_camera_cfg import NoisyGroupedRayCasterCameraCfg


class NoisyGroupedRayCasterCamera:
    """Construct the noisy grouped ray-caster camera for the active physics backend."""

    def __new__(cls, cfg: NoisyGroupedRayCasterCameraCfg):
        return create_backend_component(
            cfg,
            {
                "physx": "instinctlab.sensors.noisy_camera.physx_ray_caster_cameras:PhysxNoisyGroupedRayCasterCamera",
                "newton": (
                    "instinctlab.sensors.noisy_camera.newton_ray_caster_cameras:NewtonNoisyGroupedRayCasterCamera"
                ),
            },
        )
