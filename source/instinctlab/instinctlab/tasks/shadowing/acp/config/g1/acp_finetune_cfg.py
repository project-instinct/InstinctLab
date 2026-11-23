"""ACP stage 3 for the G1: goal-reaching RL fine-tuning on procedural terrains mixed with reference-following envs.

Starts from a stage-2 checkpoint (``noActOcc5@10k``) via ``ckpt_manipulator = "add_goal_critic"``:
- 50:50 mix of reference-following envs (first half) and procedural goal-reaching envs (last ``num_goal_envs``) (D1).
- Mixed terrain: 2 ``motion_matched`` columns + 6 procedural columns (stairs up/down, mesh boxes, square gaps,
  stepping stones, slope) with target flat patches (D2, D3).
- Planner targets are used throughout (``planner_gate = 1.0``), reference-contact rewards removed,
  ``track_planned_contact`` weight 1.5 in both reward groups.
- Two reward groups (``rewards`` -> ``V_m``, ``goal_rewards`` -> ``V_fs``) dispatched by ``env_type``.
- Opt-in extensions: D6 ``enable_no_progress_termination`` (off by default) and D7 ``use_terrain_masks`` (off by default).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.terrains as isaac_terrains
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.terrains as terrain_gen
from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.managers import MultiRewardCfg
from instinctlab.tasks.shadowing.acp import contact_schedule, goal_reaching, planner_handoff, procedural_terrains
from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_rewards_cfg import (
    PAPER_HEIGHT_THRESHOLD,
    PAPER_TILT_THRESHOLD,
)
from instinctlab.utils.config import set_cfg_joint_order

from .acp_planner_cfg import ACPPlannerObservationsCfg, ACPPlannerRewardsCfg, G1ACPPlannerEnvCfg
from .acp_tracker_cfg import ACPCommandCfg, CONTACT_BODIES

NUM_PROCEDURAL_TYPES = 10
NUM_TERRAIN_LEVELS = 10
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _procedural_sub_terrains() -> dict:
    """K = 10 directional trimesh procedural terrains covering ACP's Stage-3 groups (L1013-1043).

    Every tile spawns at ``x_local = -2.2`` m facing ``+x`` and samples ``"target"`` flat patches
    strictly on the far-side landing platform beyond the obstacle. ``box_ascent``, ``box_descent``,
    and ``box_up_down`` scale up to **1.0 m** high platforms for arms+legs whole-body climbing.
    """
    prop = 1.0 / (2 + NUM_PROCEDURAL_TYPES)
    far_landing = {
        "target": FlatPatchSamplingCfg(
            num_patches=12,
            patch_radius=[0.10, 0.20],
            max_height_diff=0.05,
            x_range=(1.8, 4.2),
            y_range=(-1.8, 1.8),
        ),
    }
    walled_landing = {
        "target": FlatPatchSamplingCfg(
            num_patches=12,
            patch_radius=[0.10, 0.20],
            max_height_diff=0.05,
            x_range=(1.8, 4.2),
            y_range=(-0.6, 0.6),
        ),
    }
    box_landing = {
        "target": FlatPatchSamplingCfg(
            num_patches=12,
            patch_radius=[0.10, 0.20],
            max_height_diff=0.05,
            x_range=(0.4, 3.0),
            y_range=(-1.8, 1.8),
        ),
    }
    sub_terrains = {
        # Dense foot
        "stairs_up": procedural_terrains.DirectionalStairsTerrainCfg(
            proportion=prop,
            terrain_name="stairs_up",
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            num_steps=7,
            descending=False,
            flat_patch_sampling=far_landing,
        ),
        "stairs_down": procedural_terrains.DirectionalStairsTerrainCfg(
            proportion=prop,
            terrain_name="stairs_down",
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            num_steps=7,
            descending=True,
            flat_patch_sampling=far_landing,
        ),
        "walled_stairs": procedural_terrains.DirectionalStairsTerrainCfg(
            proportion=prop,
            terrain_name="walled_stairs",
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            num_steps=7,
            descending=False,
            walled=True,
            corridor_width=2.0,
            flat_patch_sampling=walled_landing,
        ),
        "slope_rough": procedural_terrains.DirectionalSlopeTerrainCfg(
            proportion=prop,
            terrain_name="slope_rough",
            slope_range=(0.04, 0.35),
            roughness_range=(0.0, 0.04),
            flat_patch_sampling=far_landing,
        ),
        # Dense whole-body (up to 1.0 m high platforms for arms+legs climbing)
        "box_ascent": procedural_terrains.HighPlatformTerrainCfg(
            proportion=prop,
            terrain_name="box_ascent",
            height_range=(0.15, 1.00),
            mode="ascent",
            flat_patch_sampling=box_landing,
        ),
        "box_descent": procedural_terrains.HighPlatformTerrainCfg(
            proportion=prop,
            terrain_name="box_descent",
            height_range=(0.15, 1.00),
            mode="descent",
            flat_patch_sampling=box_landing,
        ),
        "box_up_down": procedural_terrains.HighPlatformTerrainCfg(
            proportion=prop,
            terrain_name="box_up_down",
            height_range=(0.15, 1.00),
            mode="up_down",
            box_length=2.0,
            flat_patch_sampling=far_landing,
        ),
        # Sparse foot (exact trimesh)
        "gaps": procedural_terrains.DirectionalGapTerrainCfg(
            proportion=prop,
            terrain_name="gaps",
            gap_width_range=(0.15, 0.70),
            gap_depth=1.0,
            flat_patch_sampling=box_landing,
        ),
        "double_stones": procedural_terrains.DoubleStonesTerrainCfg(
            proportion=prop,
            terrain_name="double_stones",
            stone_gap_range=(0.12, 0.42),
            stone_size_range=(0.35, 0.55),
            flat_patch_sampling=far_landing,
        ),
        "beam_pallets": procedural_terrains.BeamAndPalletsTerrainCfg(
            proportion=prop,
            terrain_name="beam_pallets",
            beam_width_range=(0.25, 0.60),
            slat_depth_range=(0.16, 0.30),
            slat_gap_range=(0.10, 0.28),
            flat_patch_sampling=far_landing,
        ),
    }
    return sub_terrains


@configclass
class ACPFinetuneCommandCfg(ACPCommandCfg):
    goal_reaching: goal_reaching.GoalReachingCommandCfg = goal_reaching.GoalReachingCommandCfg(
        num_levels=NUM_TERRAIN_LEVELS,
        max_init_level=0,
    )


@configclass
class ACPFinetuneObservationsCfg(ACPPlannerObservationsCfg):
    @configclass
    class CriticGoalObsCfg(ObsGroupCfg):
        """V_fs input (316 features for 29-DoF G1; paper L523-525, L1060-1061).

        ``planned_contacts`` is overwritten with fresh ``p_hat`` by ``PlannerAlgoMixin`` (D13).
        """

        base_ang_vel = ObsTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObsTermCfg(func=mdp.projected_gravity)
        joint_pos = ObsTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObsTermCfg(func=mdp.joint_vel_rel)
        last_action = ObsTermCfg(func=mdp.last_action)
        base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel)
        planner_height_scan = ObsTermCfg(
            func=contact_schedule.scan_height_rel_base,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-2.0, 2.0),
        )
        goal = ObsTermCfg(
            func=goal_reaching.combined_goal,
            params={"ref_command_name": "contact_schedule", "goal_command_name": "goal_reaching"},
        )
        goal_dist_xy = ObsTermCfg(
            func=goal_reaching.goal_dist_xy,
            params={"command_name": "goal_reaching"},
        )
        goal_delta_z = ObsTermCfg(
            func=goal_reaching.goal_delta_z,
            params={"command_name": "goal_reaching"},
        )
        normalized_time = ObsTermCfg(
            func=goal_reaching.goal_normalized_time,
            params={"command_name": "goal_reaching"},
        )
        planned_contacts = ObsTermCfg(
            func=goal_reaching.planned_contacts_obs,
            params={"num_bodies": len(CONTACT_BODIES)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class EnvTypeObsCfg(ObsGroupCfg):
        env_type = ObsTermCfg(func=goal_reaching.env_type_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    critic_goal: CriticGoalObsCfg = CriticGoalObsCfg()
    env_type: EnvTypeObsCfg = EnvTypeObsCfg()


@configclass
class ACPFinetuneRefRewardsCfg(ACPPlannerRewardsCfg):
    """Reference-env rewards: stage-2 set without the 3 reference-contact terms; track_planned_contact at 1.5."""

    contact_label = None
    contact_pos_base = None
    contact_pos_global = None


@configclass
class ACPFinetuneGoalRewardsCfg:
    """Goal-env rewards: shared regularization + track_planned_contact + 6 † terms (paper Table 3)."""

    # Goal-reaching † terms (D7 terrain masks opt-in, off by default)
    move2goal = RewTermCfg(
        func=goal_reaching.move2goal,
        weight=40.0,
        params={"command_name": "goal_reaching", "use_terrain_masks": False},
    )
    standatgoal = RewTermCfg(
        func=goal_reaching.standatgoal,
        weight=20.0,
        params={"command_name": "goal_reaching", "foot_body_names": FOOT_BODIES},
    )
    flat_orientation = RewTermCfg(
        func=goal_reaching.goal_flat_orientation,
        weight=-10.0,
        params={"command_name": "goal_reaching", "use_terrain_masks": False},
    )
    stumbling = RewTermCfg(
        func=goal_reaching.goal_stumbling,
        weight=-10.0,
        params={
            "command_name": "goal_reaching",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "use_terrain_masks": False,
        },
    )
    slippage = RewTermCfg(
        func=goal_reaching.goal_slippage,
        weight=-10.0,
        params={
            "command_name": "goal_reaching",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "use_terrain_masks": False,
        },
    )
    ang_vel_xy = RewTermCfg(
        func=goal_reaching.goal_ang_vel_xy,
        weight=-0.1,
        params={"command_name": "goal_reaching", "use_terrain_masks": False},
    )
    # Planner contact tracking (weight 1.5 in every env)
    track_planned_contact = RewTermCfg(
        func=planner_handoff.track_planned_contact,
        weight=1.5,
        params={"command_name": "contact_schedule", "std": 1.0},
    )
    # Shared regularization terms (D14: same as stage-1/2)
    action_rate_l2 = RewTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    joint_torques_l2 = RewTermCfg(func=mdp.joint_torques_l2, weight=-1e-5)
    applied_torque_limits = RewTermCfg(func=mdp.applied_torque_limits, weight=-1e-3)
    joint_pos_limits = RewTermCfg(func=mdp.joint_pos_limits, weight=-10.0)
    joint_vel_limits = RewTermCfg(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9})


@configclass
class ACPFinetuneRewardGroupsCfg(MultiRewardCfg):
    rewards: ACPFinetuneRefRewardsCfg = ACPFinetuneRefRewardsCfg()
    goal_rewards: ACPFinetuneGoalRewardsCfg = ACPFinetuneGoalRewardsCfg()


@configclass
class ACPFinetuneTerminationsCfg:
    """Reference-env terminations wrapped for ref envs only + goal-env terminations (D6)."""

    time_out = DoneTermCfg(
        func=goal_reaching.for_env_type,
        time_out=True,
        params={"func": mdp.time_out, "goal": False},
    )
    base_pos_too_far = DoneTermCfg(
        func=goal_reaching.for_env_type,
        time_out=False,
        params={
            "func": instinct_mdp.pos_far_from_ref,
            "goal": False,
            "func_params": {
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "distance_threshold": PAPER_HEIGHT_THRESHOLD,
                "vel_scale": 0.5,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        },
    )
    base_pg_too_far = DoneTermCfg(
        func=goal_reaching.for_env_type,
        time_out=False,
        params={
            "func": instinct_mdp.projected_gravity_far_from_ref,
            "goal": False,
            "func_params": {
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "projected_gravity_threshold": PAPER_TILT_THRESHOLD,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
            },
        },
    )
    dataset_exhausted = DoneTermCfg(
        func=goal_reaching.for_env_type,
        time_out=True,
        params={
            "func": instinct_mdp.dataset_exhausted,
            "goal": False,
            "func_params": {"reference_cfg": SceneEntityCfg("motion_reference"), "print_reason": False},
        },
    )
    out_of_border = DoneTermCfg(
        func=goal_reaching.for_env_type,
        time_out=True,
        params={
            "func": instinct_mdp.terrain_out_of_bounds,
            "goal": False,
            "func_params": {"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
        },
    )
    # Goal-env terminations (D6)
    goal_time_out = DoneTermCfg(
        func=goal_reaching.goal_time_out,
        time_out=True,
        params={"max_episode_length_s": 15.0},
    )
    goal_out_of_bounds = DoneTermCfg(
        func=goal_reaching.goal_out_of_bounds,
        time_out=False,
        params={"command_name": "goal_reaching", "margin": 0.25},
    )
    goal_bad_orientation = DoneTermCfg(
        func=goal_reaching.goal_bad_orientation,
        time_out=False,
        params={"limit": 0.8},
    )
    # Optional D6 progress-timeout termination (disabled by default; enable via enable_no_progress_termination)
    goal_no_progress: DoneTermCfg | None = None


@configclass
class ACPFinetuneCurriculumCfg:
    goal_terrain_levels = CurriculumTermCfg(
        func=goal_reaching.GoalTerrainLevels,
        params={"command_name": "goal_reaching"},
    )


@configclass
class G1ACPFinetuneEnvCfg(G1ACPPlannerEnvCfg):
    """ACP stage-3 goal-reaching RL fine-tuning environment (50:50 ref:goal mix, D1–D15)."""

    num_goal_envs: int = 1050
    """Number of goal-reaching envs (last ``num_goal_envs`` envs; 50% of 2100 envs/GPU for 4-GPU 8400 total)."""

    enable_no_progress_termination: bool = False
    """Opt-in D6 extension: terminate goal envs that make no progress toward the goal within ``no_progress_timeout_s``."""
    no_progress_timeout_s: float = 8.0

    use_terrain_masks: bool = False
    """Opt-in D7 extension: enable per-terrain-type reward masks (T_ori, T_st, T_sl, T_omega, v_max)."""

    commands: ACPFinetuneCommandCfg = ACPFinetuneCommandCfg()
    observations: ACPFinetuneObservationsCfg = ACPFinetuneObservationsCfg()
    rewards: ACPFinetuneRewardGroupsCfg = ACPFinetuneRewardGroupsCfg()
    terminations: ACPFinetuneTerminationsCfg = ACPFinetuneTerminationsCfg()
    curriculum: ACPFinetuneCurriculumCfg = ACPFinetuneCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2100  # 2100/GPU x 4 GPUs = 8400 total (paper L811)
        self.num_goal_envs = self.scene.num_envs // 2

        # Reference envs start at phase 0 in stage 3 (paper L979)
        motion = self.scene.motion_reference.motion_buffers["TerrainMotion"]
        motion.motion_start_from_middle_range = [0.0, 0.0]

        # Mixed terrain: 2 motion_matched cols + K procedural cols (D2, D3)
        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains["motion_matched"].proportion = 2.0 / (2 + NUM_PROCEDURAL_TYPES)
        generator.sub_terrains.update(_procedural_sub_terrains())
        generator.num_cols = 2 + NUM_PROCEDURAL_TYPES
        generator.seed = 42
        generator.use_cache = True
        generator.cache_dir = "/home/leo/.cache/instinctlab/acp_stage3_terrain_v2"

        # G1ACPPlannerEnvCfg.__post_init__ injects vel_scale at top-level of base_pos_too_far.params;
        # our wrapped term already carries vel_scale=0.5 inside func_params.
        self.terminations.base_pos_too_far.params.pop("vel_scale", None)

        # Combined goal obs in policy and planner (with same U(-0.2, 0.2) noise)
        combined_goal_term = ObsTermCfg(
            func=goal_reaching.combined_goal,
            params={"ref_command_name": "contact_schedule", "goal_command_name": "goal_reaching"},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        self.observations.policy.goal = combined_goal_term
        self.observations.planner.goal = combined_goal_term

        # Stage 3 uses planner predictions throughout (gate = 1.0)
        self.observations.contact_source.planner_gate.params = {
            "start_iteration": -1,
            "ramp_iterations": 1,
            "eta_max": 1.0,
        }

        # Policy joint order for critic_goal
        joint_order = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
        self.observations.critic_goal.joint_pos.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.critic_goal.joint_vel.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )

        # Reset event for goal envs (runs after reset_robot)
        self.events.reset_goal = EventTerm(
            func=goal_reaching.reset_goal_envs,
            mode="reset",
            params={"command_name": "goal_reaching"},
        )

        # Dual-buffer partitioned hindsight labeler (D8, D12)
        self.scene.contact_forces.track_contact_points = True
        self.contact_labeler.partition_samples = True
        self.contact_labeler.dual_buffer = True
        self.contact_labeler.contact_point = True
        self.contact_labeler.num_slots = 1024

        # Opt-in D6 progress-timeout termination
        if self.enable_no_progress_termination:
            self.terminations.goal_no_progress = DoneTermCfg(
                func=goal_reaching.goal_no_progress,
                time_out=False,
                params={"command_name": "goal_reaching", "timeout_s": self.no_progress_timeout_s},
            )

        # Opt-in D7 terrain masks
        if self.use_terrain_masks:
            for term_name in ("move2goal", "flat_orientation", "stumbling", "slippage", "ang_vel_xy"):
                getattr(self.rewards.goal_rewards, term_name).params["use_terrain_masks"] = True

        self.run_name = "g1ACPFinetune"


@configclass
class G1ACPFinetuneEnvCfg_PLAY(G1ACPFinetuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.num_goal_envs = 16
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.reset_robot.params["randomize_pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
        self.events.reset_goal.params["pos_z_range"] = (0.0, 0.0)
        self.events.reset_goal.params["yaw_range"] = (0.0, 0.0)
        self.events.reset_goal.params["joint_pos_range"] = (0.0, 0.0)
