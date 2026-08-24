from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.configclass import configclass

from .points_generator_cfg import PointsGeneratorCfg

VOLUME_POINTS_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/volumePoints",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "sphere_penetrated": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)


@configclass
class VolumePointsCfg(SensorBaseCfg):
    """Backend-neutral configuration for the volume-points sensor."""

    class_type: type | str = "{DIR}.volume_points:VolumePoints"

    body_names_expr: str | list[str] = ".*"
    """Body-name expressions selected recursively below :attr:`prim_path`."""

    points_generator: PointsGeneratorCfg = MISSING
    """ The points generator configuration. The generator function should be callable and accept only its cfg.
    """

    visualizer_cfg: VisualizationMarkersCfg = VOLUME_POINTS_VISUALIZER_CFG
