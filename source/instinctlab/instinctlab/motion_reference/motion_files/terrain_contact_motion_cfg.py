from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .terrain_contact_motion import TerrainContactMotion
from .terrain_motion_cfg import TerrainMotionCfg


@configclass
class TerrainContactMotionCfg(TerrainMotionCfg):
    """Terrain motions with contact labels. Pair with ``MotionReferenceManagerCfg(data_class_type=
    ContactMotionReferenceData, contact_body_names=<the same names>)``."""

    class_type: type | str = "{DIR}.terrain_contact_motion:TerrainContactMotion"

    contact_body_names: Sequence[str] = MISSING
    """ Contact bodies, in the order of the contact fields; names as in the files' ``contact_body_names``. """

    contact_onset_horizon_s: float = 1.5
    """ A next contact onset further ahead than this is not a target. """

    goal_lookahead_range_s: tuple[float, float] = (0.5, 1.5)
    """ The goal is the reference base pose this far ahead of each frame, capped at the clip end; sampled uniformly
    per episode (ACP, Supp. "Observation encoding and curricula"). """
