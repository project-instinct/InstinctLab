"""Agent configuration for the G1 SuperMimic task.

Two things distinguish this from a PPO config:

* ``inputs`` names the runtime providers the algorithm needs. They cannot be config values because they
  are built from the live scene, and their layout is only known from ``obs_format``. The runner resolves
  each dotted path and passes the result as a keyword argument (see
  ``instinct_rl.runners.on_policy_runner._resolve_algorithm_inputs``).
* The algorithm config is **standalone**, not a ``InstinctRlPpoAlgorithmCfg`` subclass. ``to_dict``
  serialises every attribute including the base class's ``MISSING`` sentinels, and ``SuperMimic``
  rejects unknown keyword arguments by design, so inheriting PPO's fields would fail at construction.

Hyperparameters follow the paper (arXiv:2610.28480v1, Sec. 3): H = 32 imagined steps, a k = 8
world-model fit window, 4096 imagined rollouts per update, and 2e-4 for each learner.
"""

import os

from isaaclab.utils.configclass import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlEncoderActorCriticCfg,
    InstinctRlMlpCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
)


@configclass
class MotionEncoderCfg(InstinctRlMlpCfg):
    """Paper Table A.4: motion encoder, MLP (2048, 1024, 512, 256) over the K = 10 future reference frames,
    to a 32-d latent bounded by tanh."""

    component_names = ["joint_pos_ref", "joint_vel_ref", "position_ref", "rotation_ref"]
    hidden_sizes = [2048, 1024, 512, 256]
    output_size = 32
    nonlinearity = "ELU"
    output_nonlinearity = "Tanh"
    # the decoder sees the latent in place of the raw reference terms
    takeout_input_components = True


@configclass
class SuperMimicEncoderConfigs:
    motion = MotionEncoderCfg()


@configclass
class SuperMimicPolicyCfg(InstinctRlEncoderActorCriticCfg):
    """Paper Table A.4 networks.

    * policy: motion encoder -> 32-d latent z, then the decoder MLP (2048, 1024, 512, 256) over z and the
      last 10 frames of the proprioceptive observation;
    * critic: MLP (256, 128) over state, goal and simulation context (the ``critic`` group).
    """

    class_name = "EncoderActorCritic"
    init_noise_std = 1.0
    encoder_configs: SuperMimicEncoderConfigs = SuperMimicEncoderConfigs()
    critic_encoder_configs = None
    actor_hidden_dims = [2048, 1024, 512, 256]
    critic_hidden_dims = [256, 128]
    activation = "elu"


@configclass
class SuperMimicAlgorithmCfg:
    """SuperMimic's algorithm configuration.

    Field names are the ``SuperMimic.__init__`` parameter names: the runner forwards this dict
    verbatim, so a typo here is a construction error rather than a silently ignored setting.
    """

    class_name: str = "SuperMimic"

    inputs: dict = {
        # Built from the live scene; see instinctlab.tasks.shadowing.supermimic.supermimic_provider.
        "imagination": "instinctlab.tasks.shadowing.supermimic.supermimic_provider:build_imagination",
    }
    """Runtime constructor arguments, resolved by the runner. Not YAML-serialisable by design."""

    # -- paper settings --
    horizon: int = 32
    """H: imagined rollout length (paper Sec. 3.3, Table A.5)."""
    wm_horizon: int = 8
    """k: autoregressive window the world model is fit on (Eq. 1)."""
    num_imagined: int = 4096
    """Imagined rollouts per policy update."""
    wm_batches_per_update: int = 10
    wm_batch_size: int = 2048
    critic_epochs: int = 5
    critic_mini_batches: int = 4
    updates_per_iteration: int = 2
    wm_warmup_iterations: int = 250
    """World-model-only iterations before the first policy update.

    The paper trains the two in one loop from the start (Sec. 3.1), which assumes a replay buffer that is
    already populated. Here the buffer starts empty. 250 iterations (1250 env steps, filling the
    1000-step buffer, and 5000 world-model batches) match the 500 warmup iterations at 2.5 env steps
    each of the public reproduction that reaches the paper's 0.82 return
    (github.com/zita-ch/supermimic_vibed). With 40 warmup iterations every run with the critic diverged.
    """

    world_model_learning_rate: float = 2e-4
    policy_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    policy_grad_value_clip: float = 0.1
    policy_chunk_size: int = 2048
    """Imagined rollouts per differentiable policy pass; bounds activation memory, not the batch (the
    chunks' gradients are accumulated into one step). The pass is launch-bound, so fewer chunks are
    faster: 512 -> 2048 cut the policy fit from 7.7 s to 2.0 s per update at ~9 GB of activations."""
    policy_cuda_graph: bool = True
    """Replay each policy chunk (rollout + backward) as one captured CUDA graph instead of launching its
    ~10^5 kernels from Python."""
    world_model_cuda_graph: bool = True
    """Likewise for each world-model batch (k-step loss + backward)."""
    collection_noise_std: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    use_critic: bool = True
    replay_capacity: int = 1000
    learning_rate: float = 2e-4
    """Reported by the runner's logger; the effective rates are the per-learner ones above."""

    # Every component is regressed as a one-step change, so the fitting error of every dimension is
    # measured on the same scale (its typical change per step). Quaternions included: a 20 ms rotation is
    # small, an additive change followed by the world model's renormalisation is its first-order step,
    # and fitting them absolutely would weight their error by their full spread, about 25x less.
    world_model_predict_delta: str = "all"
    world_model_hidden_sizes: tuple = (2048, 2048, 2048)


@configclass
class SuperMimicNormalizersCfg:
    """Normalizers for the observation groups the networks consume."""

    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1SuperMimicRunnerCfg(InstinctRlOnPolicyRunnerCfg):
    """G1 SuperMimic runner configuration."""

    policy: SuperMimicPolicyCfg = SuperMimicPolicyCfg()
    algorithm: SuperMimicAlgorithmCfg = SuperMimicAlgorithmCfg()
    normalizers: SuperMimicNormalizersCfg = SuperMimicNormalizersCfg()

    num_steps_per_env = 5
    """Paper Table A.5: update-to-data ratio 0.4, i.e. 0.4 learner updates (one world-model fit plus one
    policy update) per vectorised environment step. With ``updates_per_iteration = 2`` that is 5 steps.
    Collecting 24 steps per 2 updates gave the world model about 5x fewer fits per transition."""
    max_iterations = 30000
    save_interval = 1000
    log_interval = 10
    experiment_name = "g1_supermimic"

    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = self.load_run is not None
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )
