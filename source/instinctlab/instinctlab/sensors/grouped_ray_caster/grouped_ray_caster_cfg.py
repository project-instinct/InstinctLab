from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils.configclass import configclass

from instinctlab.utils.urdf import urdf_importer_link_prim_path


@configclass
class GroupedRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for the GroupedRayCaster sensor."""

    class_type: type | str = "{DIR}.grouped_ray_caster:GroupedRayCaster"

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""


def get_link_prim_targets(
    links: list[str],
    urdf_path: str,
    asset_prim_path: str = "/World/envs/env_[^/]+/Robot",
    is_shared=True,  # whether the target prim is assumed to be the same mesh across all environments.
    **kwargs: dict,
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Build ray-cast targets for link geometry imported by URDF importer 3.0.

    The custom caster reads visual geometry owned by each selected rigid link while
    excluding collision geometry and descendant links.
    """
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=urdf_importer_link_prim_path(urdf_path, link, asset_prim_path),
            is_shared=is_shared,
            **kwargs,
        )
        for link in links
    ]
