"""InstantTrack's environment change on the G1 SuperMimic task (github.com/Huy-Quang-Dao/instantTrack, v1.1).

The task, reward, termination, domain randomization and evaluation are SuperMimic's. The only change is
the reset distribution: 30% of resets start one bin before recent failure points (``--fail_reset 0.3``).
"""

from isaaclab.utils.configclass import configclass

from .supermimic_plane_cfg import G1SuperMimicEnvCfg


@configclass
class G1InstantTrackEnvCfg(G1SuperMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.return_based_reset_sampling.params["failure_share"] = 0.3
