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
    InstinctRlActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
)


@configclass
class SuperMimicPolicyCfg(InstinctRlActorCriticCfg):
    """Actor/critic sizes shared with the PPO baseline so the comparison is like for like."""

    class_name = "ActorCritic"
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
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
    horizon: int = 8
    """H: imagined rollout length.

    The paper uses H = 32 (Sec. 3.3). Measured on this task, the world model's open-loop error reaches
    its own data spread (normalised error 1.0, i.e. the prediction is no better than the mean) at step 9
    and grows to 5x by step 32 -- see ``instinctlab/horizonchk.py``. Backpropagating through steps the
    model cannot predict trains the policy on fabricated dynamics, which in testing drove the real
    environment reward *down* (0.93 -> 0.37) while the imagined return rose. H is therefore held at the
    edge of the model's competence. Raise it only alongside a measured increase in usable horizon.
    """
    wm_horizon: int = 8
    """k: autoregressive window the world model is fit on (Eq. 1)."""
    num_imagined: int = 4096
    """Imagined rollouts per policy update."""
    wm_batches_per_update: int = 10
    wm_batch_size: int = 2048
    critic_epochs: int = 5
    critic_mini_batches: int = 4
    updates_per_iteration: int = 2
    wm_warmup_iterations: int = 3
    """World-model-only iterations before the first policy update.

    The paper trains the two in one loop from the start (Sec. 3.1), which assumes a replay buffer that is
    already populated. In this implementation the buffer starts empty, so an H = 32 rollout through an
    untrained model diverges immediately and would poison the critic. Three warmup iterations cost
    seconds and give the model a usable k-step fit first.
    """

    world_model_learning_rate: float = 2e-4
    policy_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    policy_grad_value_clip: float = 0.1
    collection_noise_std: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    use_critic: bool = True
    replay_capacity: int = 1000
    learning_rate: float = 2e-4
    """Reported by the runner's logger; the effective rates are the per-learner ones above."""

    # Quaternions are not a vector space, so they are regressed absolutely; everything else as a delta.
    world_model_predict_delta: tuple = (
        "root_pos_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "joint_pos",
        "joint_vel",
    )
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

    num_steps_per_env = 24
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
