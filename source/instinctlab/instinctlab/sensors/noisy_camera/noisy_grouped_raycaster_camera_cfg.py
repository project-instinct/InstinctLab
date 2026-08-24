from isaaclab.utils.configclass import configclass

from instinctlab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
from instinctlab.sensors.noisy_camera.noisy_camera_cfg import NoisyCameraCfgMixin


@configclass
class NoisyGroupedRayCasterCameraCfg(NoisyCameraCfgMixin, GroupedRayCasterCameraCfg):
    """
    Configuration class for the NoisyGroupedRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type | str = "{DIR}.noisy_grouped_raycaster_camera:NoisyGroupedRayCasterCamera"
