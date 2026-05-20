import os

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlEncoderVaeActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class Conv2dHeadEncoderCfg:
    @configclass
    class DepthImageEncoderCfg(InstinctRlConv2dHeadCfg):
        channels = [32, 32]
        kernel_sizes = [3, 3]
        strides = [1, 1]
        paddings = [1, 1]
        hidden_sizes = [
            32,
        ]
        nonlinearity = "ReLU"
        use_maxpool = False
        output_size = 32
        component_names = ["depth_image"]
        takeout_input_components = True

    depth_image = DepthImageEncoderCfg()


@configclass
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    encoder_configs = Conv2dHeadEncoderCfg()
    critic_encoder_configs = None


@configclass
class VaePolicyCfg(InstinctRlEncoderVaeActorCriticCfg):
    """VAE policy; frozen bundle loads decoder/prior only (see ``VaeFrozenPriorPPO``).

    Downstream adds ``root_position_commands`` to ``vae_input_subobs_components`` for the trainable encoder;
    dagger teacher configs omit it — decoder/aux/prior tensor layouts stay compatible with ``--frozen-vae-bundle``.
    """

    encoder_configs = Conv2dHeadEncoderCfg()

    init_noise_std = 0.01

    # Encoder is trainable and not loaded from --frozen-vae-bundle; may differ from dagger.
    vae_encoder_kwargs = {
        "hidden_sizes": [1024, 512, 512],
        "nonlinearity": "ELU",
    }
    # Decoder / prior must match the dagger bundle (e.g. g1_hsidagger_perceptive_vae ... vae_phase_bundle_*.pt).
    vae_decoder_kwargs = {
        "hidden_sizes": [2048, 1024, 512, 256, 128],
        "nonlinearity": "ELU",
    }
    vae_prior_kwargs = {
        "hidden_sizes": [512, 256, 128],
        "nonlinearity": "ELU",
    }
    vae_latent_size = 32
    vae_decode_add_prior_mean = True
    vae_project_to_sphere = True
    vae_encoder_mean_only = True
    explore_latent = True
    vae_input_subobs_components = [
        # "joint_pos_ref",
        # "joint_vel_ref",
        # "position_ref",
        # "rotation_ref",
        "parallel_latent_0_depth_image",
        "projected_gravity",
        "root_position_commands",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
    ]
    vae_aux_subobs_components = [
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
    ]
    vae_prior_subobs_components = [
        "parallel_latent_0_depth_image",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
    ]


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    """Pure PPO; load frozen decoder/prior from dagger via ``frozen_vae_bundle`` (CLI ``--frozen-vae-bundle``)."""

    class_name = "VaeFrozenPriorPPO"
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
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1PerceptiveVaePPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: VaePolicyCfg = VaePolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_hsidownstream_gen_perceptive_vae"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
