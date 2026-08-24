from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg
from isaaclab.utils.configclass import configclass

from instinctlab.sensors.noisy_camera.noisy_camera_cfg import NoisyCameraCfgMixin


@configclass
class NoisyMultiMeshRayCasterCameraCfg(NoisyCameraCfgMixin, MultiMeshRayCasterCameraCfg):
    """
    Configuration class for the NoisyMultiMeshRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type | str = "{DIR}.noisy_multi_mesh_ray_caster_camera:NoisyMultiMeshRayCasterCamera"
