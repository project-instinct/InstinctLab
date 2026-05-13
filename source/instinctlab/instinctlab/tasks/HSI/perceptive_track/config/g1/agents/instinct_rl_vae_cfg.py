import os

from isaaclab.utils import configclass

from instinctlab.envs.mdp.observations.exteroception import visualizable_image
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
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
    encoder_configs = Conv2dHeadEncoderCfg()

    vae_encoder_kwargs = {
        "hidden_sizes": [256, 128, 64],
        "nonlinearity": "ELU",
    }
    vae_decoder_kwargs = {
        "hidden_sizes": [512, 256, 128],
        "nonlinearity": "ELU",
    }
    vae_latent_size = 16
    vae_input_subobs_components = [
        "parallel_latent_0_depth_image",  # based on the encoder_configs in Conv2dHeadEncoderCfg
        # "projected_gravity",
        # "base_ang_vel",
        # "joint_pos",
        # "joint_vel",
        # "last_action",
    ]
    vae_aux_subobs_components = [
        # "parallel_latent_0_depth_image",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
    ]


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "VaeDistill"
    kl_loss_func = "kl_divergence"
    kl_loss_coef = 1.0
    using_ppo = False
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    # PPO parameters should not affect anything.
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    teacher_act_prob = 0.2
    # update_times_scale = 20 * int(1e3)

    teacher_policy_class_name = InstinctRlEncoderActorCriticCfg().class_name
    teacher_policy: dict = {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "encoder_configs": {
            "depth_image": {
                "class_name": "Conv2dHeadModel",
                "component_names": ["depth_image"],
                "output_size": 32,
                "takeout_input_components": True,
                "channels": [32, 32],
                "kernel_sizes": [3, 3],
                "strides": [1, 1],
                "hidden_sizes": [32],
                "paddings": [1, 1],
                "nonlinearity": "ReLU",
                "use_maxpool": False,
            }
        },
        "critic_encoder_configs": None,
        "obs_format": {
            "policy": {
                "joint_pos_ref": (10, 29),
                "joint_vel_ref": (10, 29),
                "position_ref": (10, 3),
                "rotation_ref": (10, 6),
                "depth_image": (1, 18, 32),
                "projected_gravity": (24,),
                "base_ang_vel": (24,),
                "joint_pos": (232,),
                "joint_vel": (232,),
                "last_action": (232,),
            },
            "critic": {
                "joint_pos_ref": (10, 29),
                "joint_vel_ref": (10, 29),
                "position_ref": (10, 3),
                "link_pos": (14, 3),
                "link_rot": (14, 6),
                "height_scan": (187,),
                "base_lin_vel": (24,),
                "base_ang_vel": (24,),
                "joint_pos": (232,),
                "joint_vel": (232,),
                "last_action": (232,),
            },
        },
        "num_actions": 29,
        "num_rewards": 1,
    }
    teacher_logdir = os.path.expanduser(
        "~/Data/instinctlab_logs/instinct_rl/g1_perceptive_shadowing/20260111_103654_g1Perceptive_4MotionsKneelClimbStep1_concatMotionBins__GPU0_from20260108_032900"
    )


@configclass
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    # critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    # NOTE: No critic normalizer, must be loaded from the teacher policy.


@configclass
class G1PerceptiveVaePPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: VaePolicyCfg = VaePolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_perceptive_vae"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
