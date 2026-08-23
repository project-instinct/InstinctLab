from __future__ import annotations

from typing import TYPE_CHECKING

from ..grouped_ray_caster.newton_grouped_ray_caster_camera import NewtonGroupedRayCasterCamera
from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_grouped_raycaster_camera_cfg import NoisyGroupedRayCasterCameraCfg


class NewtonNoisyGroupedRayCasterCamera(NoisyCameraMixin, NewtonGroupedRayCasterCamera):
    cfg: NoisyGroupedRayCasterCameraCfg
