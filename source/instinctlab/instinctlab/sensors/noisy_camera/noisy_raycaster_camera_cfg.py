from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.utils.configclass import configclass

from instinctlab.sensors.noisy_camera.noisy_camera_cfg import NoisyCameraCfgMixin


@configclass
class NoisyRayCasterCameraCfg(NoisyCameraCfgMixin, RayCasterCameraCfg):
    """
    Configuration class for the NoisyRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type | str = "{DIR}.noisy_raycaster_camera:NoisyRayCasterCamera"
