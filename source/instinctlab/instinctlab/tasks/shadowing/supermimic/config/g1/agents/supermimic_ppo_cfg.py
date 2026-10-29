"""PPO-Sim: the paper's PPO baseline on the SuperMimic task (arXiv:2610.28480v1, Sec. 4.1, Fig. 4a).

"PPO-Sim shares SuperMimic's motions, observations, reward, termination, domain randomization, actor and
critic network sizes, and evaluation protocol. It uses BeyondMimic's PPO hyperparameters and 8192
parallel environments per GPU, twice SuperMimic's."

So the environment is the SuperMimic task unchanged, the networks are SuperMimic's (motion encoder,
decoder, critic of Table A.4), and the algorithm and rollout length are BeyondMimic's PPO settings. The
policy is a Gaussian around the decoder's output, as PPO needs; evaluation plays its mean.
"""

import os

from isaaclab.utils.configclass import configclass

from instinctlab.tasks.shadowing.beyondmimic.config.g1.agents.beyondmimic_ppo_cfg import BeyondMimicAlgorithmCfg
from instinctlab.tasks.shadowing.supermimic.config.g1.agents.supermimic_agent_cfg import (
    SuperMimicNormalizersCfg,
    SuperMimicPolicyCfg,
)
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg


@configclass
class G1SuperMimicPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    """PPO in the simulator with SuperMimic's networks and BeyondMimic's PPO hyperparameters."""

    policy: SuperMimicPolicyCfg = SuperMimicPolicyCfg()
    algorithm: BeyondMimicAlgorithmCfg = BeyondMimicAlgorithmCfg()
    normalizers: SuperMimicNormalizersCfg = SuperMimicNormalizersCfg()

    num_steps_per_env = 24  # BeyondMimic
    max_iterations = 30000
    save_interval = 250
    log_interval = 10
    experiment_name = "g1_supermimic_ppo"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
