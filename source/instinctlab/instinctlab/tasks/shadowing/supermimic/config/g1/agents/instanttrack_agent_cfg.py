"""InstantTrack's learner changes on SuperMimic (github.com/Huy-Quang-Dao/instantTrack, v1.1 ``instanttrack``
preset), ported onto this SuperMimic implementation.

Ported, each as a setting of the existing SuperMimic code:

* ``--pi_ema 0.995``: a Polyak average of the policy weights is what checkpoints, evaluation and export use.
* ``--lr 4e-4 --wm_batches 5``: every learner at 4e-4, 5 world-model batches per update instead of 10.
* ``--wm_hidden 1024,1024,1024``: a 1024^3 world model instead of 2048^3.
* ``--ref_residual 1.0``: the policy outputs a correction on top of the next reference joint pose.
* ``--fused_adam``: fused Adam kernels.
* ``--amp bf16`` (InstantTrack's default): bfloat16 network passes.
* ``--fail_reset 0.3``: in the environment config (``instanttrack_plane_cfg.py``).

Not ported: the contact reward (``--contact_reward``, a training-only reward term that costs normalized
return in InstantTrack's own ablation), and the systems changes tied to its code base (``--no_sync``,
``--fused_rot``, ``--compile_more``); this implementation replays its update passes as CUDA graphs instead.
"""

import re

from isaaclab.utils.configclass import configclass

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1, beyondmimic_action_scale

from .supermimic_agent_cfg import G1SuperMimicRunnerCfg, SuperMimicAlgorithmCfg, SuperMimicPolicyCfg


def _action_scale(joint_name: str) -> float:
    """The action scale of a joint: ``beyondmimic_action_scale`` is keyed by joint-name patterns."""
    matches = [scale for pattern, scale in beyondmimic_action_scale.items() if re.fullmatch(pattern, joint_name)]
    if len(matches) != 1:
        raise KeyError(f"{joint_name!r} matches {len(matches)} patterns of beyondmimic_action_scale")
    return matches[0]


@configclass
class InstantTrackPolicyCfg(SuperMimicPolicyCfg):
    # The action is ``q_default + alpha * a`` and ``joint_pos_ref`` holds ``q_ref - q_default`` for the goal
    # window t, t+1, ..., so adding ``joint_pos_ref[1] / alpha`` makes the decoder a correction on top of
    # the next reference pose. Both are in the policy joint order.
    reference_residual: dict = {
        "component": "joint_pos_ref",
        "frame": 1,
        "scale": [1.0 / _action_scale(name) for name in G1_29DOF_POLICY_JOINT_ORDER_V1],
    }


@configclass
class InstantTrackAlgorithmCfg(SuperMimicAlgorithmCfg):
    world_model_learning_rate: float = 4e-4
    policy_learning_rate: float = 4e-4
    critic_learning_rate: float = 4e-4
    learning_rate: float = 4e-4
    wm_batches_per_update: int = 5
    world_model_hidden_sizes: tuple = (1024, 1024, 1024)
    policy_ema_decay: float | None = 0.995
    fused_optimizers: bool = True
    mixed_precision: str | None = "bfloat16"


@configclass
class G1InstantTrackRunnerCfg(G1SuperMimicRunnerCfg):
    policy: InstantTrackPolicyCfg = InstantTrackPolicyCfg()
    algorithm: InstantTrackAlgorithmCfg = InstantTrackAlgorithmCfg()
    experiment_name = "g1_instanttrack"
