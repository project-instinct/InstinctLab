from dataclasses import MISSING
from typing import Literal, Sequence

from isaaclab.utils.configclass import configclass

from instinctlab.utils.wrappers.instinct_rl.module_cfg import InstinctRlParallelBlockCfg


@configclass
class InstinctRlActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    critic_obs_groups: list[str] | None = None
    """Optional observation group name per critic when using multiple critics with distinct inputs."""


@configclass
class InstinctRlActorCriticRecurrentCfg(InstinctRlActorCriticCfg):
    class_name: str = "ActorCriticRecurrent"

    rnn_type: str = "gru"
    """The type of RNN to use. Default is GRU."""

    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""

    rnn_num_layers: int = 1
    """The number of layers in the RNN."""

    multireward_multirnn: bool = False
    """Whether to use multiple RNN critics for multiple rewards."""


@configclass
class EncoderCfgMixin:
    encoder_configs: object = MISSING
    """ A configclass containing InstinctRlEncoderCfg.
    """

    critic_encoder_configs: object | Literal["shared"] | None = None
    """ A configclass containing InstinctRlEncoderCfg for building the critic encoders.
    """


@configclass
class EstimatorCfgMixin:
    """Mixin for state estimator."""

    estimator_obs_components: list[str] = MISSING
    """The components of the observation used for the estimator."""

    estimator_target_components: list[str] = MISSING
    """The components of the observation used as the target for the estimator."""

    estimator_configs: object = MISSING
    """A configclass containing GalaxeaRlEncoderCfg for building the estimator."""

    replace_state_prob: float = 1.0


@configclass
class EstimatorActorCriticCfg(
    EstimatorCfgMixin,
    InstinctRlActorCriticCfg,
):
    """Configuration for the estimator actor-critic networks."""

    class_name = "EstimatorActorCritic"


@configclass
class EstimatorActorCriticRecurrentCfg(
    EstimatorCfgMixin,
    InstinctRlActorCriticRecurrentCfg,
):
    """Configuration for the estimator actor-critic-recurrent networks."""

    class_name = "EstimatorActorCriticRecurrent"


@configclass
class InstinctRlMoEActorCriticCfg(InstinctRlActorCriticCfg):
    class_name: str = "MoEActorCritic"

    num_moe_experts: int = 8
    """The number of Mixture of Experts (MoE) experts."""

    moe_gate_hidden_dims: list[int] = []
    """The hidden dimensions of the MoE gate network."""


@configclass
class InstinctRlVaeActorCriticCfg(InstinctRlActorCriticCfg):
    class_name: str = "VaeActor"

    vae_encoder_kwargs: dict = MISSING
    """ A dict building the MLP-based VAE encoder."""

    vae_decoder_kwargs: dict = MISSING
    """ A dict building the MLP-based VAE decoder."""

    vae_latent_size: int = 16
    """ The latent size of the VAE."""

    critic_hidden_dims: list[int] = [512, 256, 128]
    """The hidden dimensions of the critic network (typically not used for VAE distillation)."""

    init_noise_std: float = 1e-4
    """The initial noise standard deviation for the critic network (typically not used for VAE distillation)."""

    activation: str = "elu"
    """The activation function for the critic network (typically not used for VAE distillation)."""


@configclass
class InstinctRlEncoderActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlActorCriticCfg,
):
    """Configuration for the encoder actor-critic networks."""

    class_name = "EncoderActorCritic"


@configclass
class InstinctRlEncoderActorCriticRecurrentCfg(
    EncoderCfgMixin,
    InstinctRlActorCriticRecurrentCfg,
):
    """Configuration for the encoder actor-critic-recurrent networks."""

    class_name = "EncoderActorCriticRecurrent"


@configclass
class InstinctRlEncoderMoEActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlMoEActorCriticCfg,
):
    """Configuration for the encoder actor-critic networks."""

    class_name = "EncoderMoEActorCritic"


@configclass
class InstinctRlEncoderVaeActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlVaeActorCriticCfg,
):
    """Configuration for the encoder actor networks."""

    class_name = "EncoderVaeActorCritic"


@configclass
class InstinctRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    optimizer_class_name: str = "AdamW"
    """The optimizer class name. Default is AdamW."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    advantage_mixing_weights: float | Sequence[float] = 1.0
    """The weights for the mixing advantages and compute surrogate loss when multiple rewards are returned."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    clip_min_std: float = 1e-12
    """The minimum standard deviation for the policy when computing distribution.
    Default: 1e-12 to prevent numerical instability.
    """

    value_mask_group: str | None = None
    """Optional observation group ``(N, num_rewards)`` indicating which critic/reward column applies to each env."""

    critic_learning_rates: list[float | None] | None = None
    """Optional per-critic fixed learning rates (``None`` follows the adaptive PPO learning rate)."""

    critic_warmup_iterations: list[int] | None = None
    """Optional per-critic warmup iterations during which that critic's envs are excluded from the actor surrogate."""

    critic_value_clipping: list[bool] | None = None
    """Optional per-critic value clipping flags (off during warmup)."""


@configclass
class InstinctRlContactPlannerPpoAlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    """Configuration for ``ContactPlannerPPO``: PPO plus a contact planner trained from hindsight samples.

    See ``instinct_rl.algorithms.planner.PlannerAlgoMixin`` for the meaning of every ``planner_*`` field.
    """

    class_name: str = "ContactPlannerPPO"

    planner_class_name: str = "AttentionContactPlanner"
    """``AttentionContactPlanner`` or the baseline ``MlpContactPlanner`` (or ``"module.path:Class"``)."""

    planner_kwargs: dict = MISSING
    """Planner constructor arguments, besides ``input_segments``."""

    planner_obs_group: str = "planner"
    planner_target_components: list[str] = ["contact_targets"]
    planner_gate_group: str = "contact_source"
    planner_critic_targets: list[tuple[str, tuple[str, ...]]] | None = None
    planner_target_noise: float = 0.0
    planner_positions_feedback_key: str = "planned_contacts"
    planner_occupancy_feedback_key: str = "planned_occupancy"
    planner_samples_key: str = "planner_samples"
    planner_optimizer_class_name: str = "Adam"
    planner_optimizer_kwargs: dict = {"lr": 1e-4}
    planner_max_grad_norm: float | None = None
    planner_num_updates: int = 10
    planner_batch_size: int = 4096
    planner_accumulation_steps: int = 8
    planner_min_samples: int = 1000
    planner_replay_capacity: int = 100_000
    planner_replay_num_partitions: int | None = None
    planner_replay_partition_capacity: int = 2000
    planner_goal_replay_kwargs: dict | None = None
    planner_replay_device: str | None = None
    planner_replay_checkpoint: str = "latest"


@configclass
class InstinctRlNormalizerCfg:
    class_name: str = "EmpiricalNormalization"
    until: int | None = None


@configclass
class InstinctRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    ckpt_manipulator: str | None = None
    """A string calling the checkpoint manipulator when loading. Typically, user has to implement their own
    checkpoint manipulator to load the model weights if the loaded model is different from the current training
    model. Default is None for no manipulation.
    """

    policy: InstinctRlActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: InstinctRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    normalizers = dict()
    """The configs for each observation group when they are still flattened tensors
    Empty dict for no normalizer running in the RL runner.
    """

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    log_interval: int = 1
    """The number of iterations between logs."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    # logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    # """The logger to use. Default is tensorboard."""

    # neptune_project: str = "instinctlab"
    # """The neptune project name. Default is "instinctlab"."""

    # wandb_project: str = "instinctlab"
    # """The wandb project name. Default is "instinctlab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str | None = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
