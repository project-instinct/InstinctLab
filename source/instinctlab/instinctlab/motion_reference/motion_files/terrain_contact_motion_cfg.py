from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .contact_motion_cfg import ContactMotionCfg
from .terrain_contact_motion import TerrainContactMotion
from .terrain_motion_cfg import TerrainMotionCfg


@configclass
class TerrainContactMotionCfg(TerrainMotionCfg, ContactMotionCfg):
    """Terrain motions with contact labels (see :class:`ContactMotionCfg`)."""

    class_type: type | str = "{DIR}.terrain_contact_motion:TerrainContactMotion"
