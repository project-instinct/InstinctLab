import os

from isaaclab.utils.configclass import configclass

from instinctlab.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_ppo_cfg import Conv2dHeadEncoderCfg
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlAME2ImageEncoderCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    """Tracker MLP {512, 256, 128}, ELU (paper, Supp. network architectures), on top of the depth-image CNN of the
    perceptive shadowing task."""

    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    encoder_configs = Conv2dHeadEncoderCfg()
    critic_encoder_configs = None


@configclass
class AME2EncoderCfg:
    """AME-2 attention encoder on the depth image; its query embeds all other actor inputs, so the actor MLP sees
    only the encoder output (attention output 96, proprioception embedding 128, global features 64)."""

    depth_image = InstinctRlAME2ImageEncoderCfg(
        component_names=[
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "last_action",
            "depth_image",
            "contact_targets",
            "goal",
        ],
        image_component="depth_image",
        output_size=288,
    )


@configclass
class AME2PolicyCfg(PolicyCfg):
    encoder_configs = AME2EncoderCfg()


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    """Stage-1 PPO settings of the paper (Table S6)."""

    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
    num_learning_epochs = 5
    num_mini_batches = 8
    learning_rate = 1e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.02
    max_grad_norm = 1.0


@configclass
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1ACPTrackerPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_acp_tracker"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None
        self.run_name = f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else ""


@configclass
class G1ACPTrackerAME2PPORunnerCfg(G1ACPTrackerPPORunnerCfg):
    policy: AME2PolicyCfg = AME2PolicyCfg()
