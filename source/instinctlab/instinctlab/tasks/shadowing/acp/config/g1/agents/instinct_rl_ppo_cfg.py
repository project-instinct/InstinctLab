import os

from isaaclab.utils.configclass import configclass

from instinctlab.tasks.shadowing.perceptive.config.g1.agents.instinct_rl_ppo_cfg import Conv2dHeadEncoderCfg
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlContactPlannerPpoAlgorithmCfg,
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
class PlannerAlgorithmCfg(InstinctRlContactPlannerPpoAlgorithmCfg):
    """Stage-2 PPO settings of the paper (Table S6) and the planner's optimization (Supp. Table S5)."""

    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 8
    learning_rate = 1e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    planner_class_name = "AttentionContactPlanner"
    planner_kwargs = {
        "points_xy_component": "terrain_xy",
        "points_z_component": "terrain_height",
        "context_components": ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "last_action", "goal"],
        "num_bodies": 9,
    }
    # the tracker's observation noise on contact targets (Supp. Table S2), also on the planner's
    planner_target_noise = 0.1
    planner_optimizer_kwargs = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
    planner_num_updates = 10
    planner_batch_size = 4096
    planner_accumulation_steps = 8
    planner_min_samples = 1000
    planner_replay_capacity = 100_000


@configclass
class G1ACPPlannerPPORunnerCfg(G1ACPTrackerPPORunnerCfg):
    """Stage 2, started from a stage-1 tracker checkpoint (``--load_run <stage-1 run> --checkpoint model_N.pt
    --resume``): the checkpoint manipulator adds the fresh planner and restarts the iteration count at 0."""

    algorithm: PlannerAlgorithmCfg = PlannerAlgorithmCfg()
    ckpt_manipulator = "add_missing_algorithm_state"
    max_iterations = 5000
    experiment_name = "g1_acp_planner"


@configclass
class BigReplayPlannerAlgorithmCfg(PlannerAlgorithmCfg):
    """Experiment variant: a 10x replay (about 4 GB of 1029-float samples on the GPU). The 100,000-sample FIFO holds
    only 10-40 iterations of labels, so the planner forgets the labels of the pre-hand-off tracker within the ramp."""

    planner_replay_capacity = 1_000_000


@configclass
class G1ACPBigReplayPlannerPPORunnerCfg(G1ACPPlannerPPORunnerCfg):
    algorithm: BigReplayPlannerAlgorithmCfg = BigReplayPlannerAlgorithmCfg()


@configclass
class DepthPlannerAlgorithmCfg(PlannerAlgorithmCfg):
    """The planner also reads the depth image, through a small AME-2 encoder whose query is its context."""

    planner_kwargs = {
        **PlannerAlgorithmCfg().planner_kwargs,
        "image_component": "depth_image",
        "image_encoder_kwargs": {
            "num_heads": 16,
            "local_dim": 48,
            "global_dim": 32,
            "prop_dim": 64,
            "hidden_size": 256,
        },
    }


@configclass
class G1ACPDepthPlannerPPORunnerCfg(G1ACPPlannerPPORunnerCfg):
    algorithm: DepthPlannerAlgorithmCfg = DepthPlannerAlgorithmCfg()


def _num_motions(metadata_yaml: str) -> int:
    import yaml

    with open(metadata_yaml) as f:
        return len(yaml.safe_load(f)["motion_files"])


@configclass
class V2PlannerAlgorithmCfg(PlannerAlgorithmCfg):
    """Second stage-2 setup: one replay partition of 2000 samples per reference clip, sampled balanced over the terrain
    categories, then clips (InstinctMJ's reference replay), instead of one 100k FIFO."""

    planner_replay_partition_capacity = 2000


@configclass
class G1ACPPlannerV2PPORunnerCfg(G1ACPPlannerPPORunnerCfg):
    algorithm: V2PlannerAlgorithmCfg = V2PlannerAlgorithmCfg()
    max_iterations = 10000

    def __post_init__(self):
        super().__post_init__()
        from ..acp_tracker_cfg import METADATA_YAML  # the env cfg module, loaded with the task anyway

        self.algorithm.planner_replay_num_partitions = _num_motions(METADATA_YAML)


@configclass
class G1ACPPlannerV3Occ5PPORunnerCfg(G1ACPPlannerV2PPORunnerCfg):
    """V2 planner with 5x occupancy loss weight to balance occupancy BCE (~0.07) against affordance CE (~3.2)."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.planner_kwargs = {**self.algorithm.planner_kwargs, "occupancy_loss_coef": 5.0}


@configclass
class G1ACPPlannerV3NoActPPORunnerCfg(G1ACPPlannerV2PPORunnerCfg):
    """V2 planner without ``last_action`` / ``joint_vel`` in context to avoid closed-loop action feedback shift."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.planner_kwargs = {
            **self.algorithm.planner_kwargs,
            "context_components": ["base_ang_vel", "projected_gravity", "joint_pos", "goal"],
        }


@configclass
class G1ACPPlannerV3NoActOcc5PPORunnerCfg(G1ACPPlannerV2PPORunnerCfg):
    """V2 planner combining no ``last_action`` / ``joint_vel`` in context with 5x occupancy loss weight."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.planner_kwargs = {
            **self.algorithm.planner_kwargs,
            "context_components": ["base_ang_vel", "projected_gravity", "joint_pos", "goal"],
            "occupancy_loss_coef": 5.0,
        }
