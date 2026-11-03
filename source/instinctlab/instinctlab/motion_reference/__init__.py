"""The implementation for reading and returning motion reference data.

FAQ:
    - How to use this module in AMP/WASABI implementation?
        * Use the motion reference to build discriminator observation and feed to the algorithm
        * Use the discriminator observation and discriminator (in algo) to label the discrimintaor
        reward.
"""

from .motion_reference_cfg import MotionReferenceManagerCfg
from .motion_reference_contact_data import ContactMotionReferenceData, ContactMotionSequence
from .motion_reference_data import MotionReferenceData, MotionReferenceState, MotionSequence
from .motion_reference_hoi_data import HoiMotionReferenceData, HoiMotionReferenceState, HoiMotionSequence

__all__ = [
    "ContactMotionReferenceData",
    "ContactMotionSequence",
    "HoiMotionReferenceData",
    "HoiMotionReferenceState",
    "HoiMotionSequence",
    "MotionReferenceData",
    "MotionReferenceManagerCfg",
    "MotionReferenceState",
    "MotionSequence",
]
