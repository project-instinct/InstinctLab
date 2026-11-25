from __future__ import annotations

from typing import TYPE_CHECKING

from .contact_motion import ContactMotion
from .terrain_motion import TerrainMotion

if TYPE_CHECKING:
    from .terrain_contact_motion_cfg import TerrainContactMotionCfg


class TerrainContactMotion(TerrainMotion, ContactMotion):
    """Terrain motions with contact labels: the scene matching of :class:`TerrainMotion` and the contact schedule and
    goal of :class:`ContactMotion`."""

    cfg: TerrainContactMotionCfg
