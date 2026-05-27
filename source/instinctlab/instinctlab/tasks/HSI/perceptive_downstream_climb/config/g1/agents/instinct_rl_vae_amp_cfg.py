import os

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg, InstinctRlPpoAlgorithmCfg

from .instinct_rl_vae_cfg import NormalizersCfg, VaePolicyCfg


@configclass
class AmpAlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiVaeFrozenPriorPPO"

    discriminator_kwargs = {
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
    }
    discriminator_reward_coef = 0.25
    discriminator_reward_type = "quad"
    discriminator_loss_func = "MSELoss"
    discriminator_gradient_penalty_coef = 5.0
    discriminator_optimizer_class_name = "AdamW"
    discriminator_weight_decay_coef = 3e-4
    discriminator_logit_weight_decay_coef = 0.04
    discriminator_optimizer_kwargs = {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
    }

    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    frozen_vae_bundle = None
    freeze_prior = True
    freeze_decoder = True
    freeze_depth_encoder = True


@configclass
class G1PerceptiveVaeAmpPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: VaePolicyCfg = VaePolicyCfg()
    algorithm: AmpAlgorithmCfg = AmpAlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()
    partial_policy_normalizer_from_bundle = True
    partial_policy_normalizer_components = [
        "depth_image",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
    ]
    partial_policy_normalizer_freeze_components = True

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_hsidownstream_climb_perceptive_vae_amp"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
