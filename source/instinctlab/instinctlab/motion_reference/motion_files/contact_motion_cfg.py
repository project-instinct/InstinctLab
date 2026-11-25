from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .amass_motion_cfg import AmassMotionCfg
from .contact_motion import ContactMotion


@configclass
class ContactMotionCfg(AmassMotionCfg):
    """Retargeted motions with contact labels. Pair with ``MotionReferenceManagerCfg(data_class_type=
    ContactMotionReferenceData, contact_body_names=<the same names>)``."""

    class_type: type | str = "{DIR}.contact_motion:ContactMotion"

    contact_body_names: Sequence[str] = MISSING
    """ Contact bodies, in the order of the contact fields; names as in the files' ``contact_body_names``. """

    contact_onset_horizon_s: float = 1.5
    """ A next contact onset further ahead than this is not a target. """

    contact_keys_required: bool = True
    """ If True, a clip without contact keys raises; if False, such a clip has no contacts. """

    goal_lookahead_range_s: tuple[float, float] = (0.5, 1.5)
    """ The goal is the reference base pose this far ahead of each frame, capped at the clip end; sampled uniformly
    (ACP, Supp. "Observation encoding and curricula"). """

    goal_lookahead_resample_interval_s: float = 10.0
    """ The lookahead is sampled at reset and again every this many seconds of the episode; ``float("inf")`` keeps one
    lookahead per episode. """
