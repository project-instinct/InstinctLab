"""ACP stage-1 tracker for the G1 on terrain-interactive reference motions (PHP + Deep Whole-body Parkour).

The actor sees proprioception (5-step history, 5 x 93), the depth image of the head camera, the next contact target
of 9 bodies (27) and a goal (3). The paper's 9 x 7 height scan and the 17 x 11 planner scan are critic inputs only.
Reference kinematics enter only the rewards and the critic. Choices not
given by the paper (kernel widths, goal definition, terminations, start sampling) are recorded in
``agents/AttnContacts_SuperMimic_REPRODUCE_PLAN.md``.
"""

import os
import yaml

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptive_cfg
from instinctlab.managers import MultiRewardCfg
from instinctlab.motion_reference import ContactMotionReferenceData
from instinctlab.motion_reference.motion_files.terrain_contact_motion_cfg import TerrainContactMotionCfg
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.tasks.shadowing.acp import contact_schedule
from instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg import (
    G1_CFG,
    G1PerceptiveShadowingEnvCfg,
    motion_reference_cfg,
)
from instinctlab.tasks.utils.backends import physics_preset
from instinctlab.utils.urdf import urdf_importer_link_prim_path

MOTION_FOLDER = os.path.expanduser("~/Datasets/ACP-TerrainContacts_G1")
TERRAIN_SIZE = (14.0, 12.0)  # must match the size used by scripts/php_to_instinct.py
CONTACT_BODIES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "pelvis",
]
PROPRIO_HISTORY_LENGTH = 5


@configclass
class ACPTerrainContactMotionCfg(TerrainContactMotionCfg):
    """Settings of the perceptive shadowing task's terrain motion, on the ACP dataset."""

    path = MOTION_FOLDER
    metadata_yaml = os.path.join(MOTION_FOLDER, "metadata.yaml")
    contact_body_names = CONTACT_BODIES
    max_origins_per_motion = 49
    ensure_link_below_zero_ground = False
    motion_start_from_middle_range = [0.0, 0.0]
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    env_starting_stub_sampling_strategy = "concat_motion_bins"


@configclass
class ACPSceneCfg(perceptive_cfg.PerceptiveShadowingSceneCfg):
    # 9 x 7 tracker scan; the inherited 17 x 11 ``height_scanner`` is the planner scan
    tracker_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.8, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ACPCommandCfg(perceptive_cfg.CommandCfg):
    contact_schedule = contact_schedule.ContactScheduleCommandCfg()


@configclass
class ACPObservationsCfg:
    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05), history_length=PROPRIO_HISTORY_LENGTH
        )
        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.0125, n_max=0.0125),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5), history_length=PROPRIO_HISTORY_LENGTH
        )
        last_action = ObsTermCfg(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH)
        # User decision: the actor perceives terrain through the onboard depth camera (as in the perceptive
        # shadowing task) instead of the paper's 9 x 7 height scan, which stays a critic input.
        depth_image = ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"},
        )
        contact_targets = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "contact_schedule"},
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )
        goal = ObsTermCfg(
            func=contact_schedule.contact_schedule_goal,
            params={"command_name": "contact_schedule"},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticObsCfg(ObsGroupCfg):
        base_ang_vel = ObsTermCfg(func=mdp.base_ang_vel, history_length=PROPRIO_HISTORY_LENGTH)
        projected_gravity = ObsTermCfg(func=mdp.projected_gravity, history_length=PROPRIO_HISTORY_LENGTH)
        joint_pos = ObsTermCfg(func=mdp.joint_pos_rel, history_length=PROPRIO_HISTORY_LENGTH)
        joint_vel = ObsTermCfg(func=mdp.joint_vel_rel, history_length=PROPRIO_HISTORY_LENGTH)
        last_action = ObsTermCfg(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH)
        base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel)
        height_scan = ObsTermCfg(
            func=contact_schedule.scan_height_rel_base,
            params={"sensor_cfg": SceneEntityCfg("tracker_height_scanner")},
            clip=(-2.0, 2.0),
        )
        planner_height_scan = ObsTermCfg(
            func=contact_schedule.scan_height_rel_base,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-2.0, 2.0),
        )
        contact_targets = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "contact_schedule"})
        goal = ObsTermCfg(func=contact_schedule.contact_schedule_goal, params={"command_name": "contact_schedule"})
        reference_contacts = ObsTermCfg(
            func=contact_schedule.reference_contacts, params={"command_name": "contact_schedule"}
        )
        # privileged reference kinematics
        joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
        joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
        position_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "position_ref_command"})
        rotation_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "rotation_ref_command"})
        link_pos = ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=CONTACT_BODIES, preserve_order=True)},
        )
        link_rot = ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=CONTACT_BODIES, preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class ACPRewardsCfg:
    # reference tracking (paper Table 3 weights)
    base_position = RewTermCfg(func=instinct_mdp.base_position_imitation_gauss, weight=3.0, params={"std": 0.3})
    base_velocity = RewTermCfg(func=instinct_mdp.base_velocity_imitation_gauss, weight=1.0, params={"std": 0.5})
    base_rotation = RewTermCfg(
        func=instinct_mdp.base_rot_imitation_gauss, weight=1.0, params={"std": 0.4, "difference_type": "axis_angle"}
    )
    joint_position = RewTermCfg(
        func=instinct_mdp.joint_pos_imitation_gauss, weight=2.5, params={"std": 0.3, "combine_method": "mean_prod"}
    )
    joint_velocity = RewTermCfg(
        func=instinct_mdp.joint_vel_imitation_gauss, weight=0.5, params={"std": 3.0, "combine_method": "mean_prod"}
    )
    # reference contacts
    contact_label = RewTermCfg(
        func=contact_schedule.contact_label,
        weight=5.0,
        params={
            "command_name": "contact_schedule",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=CONTACT_BODIES, preserve_order=True),
            "threshold": 1.0,
        },
    )
    contact_pos_base = RewTermCfg(
        func=contact_schedule.contact_link_pos_gauss,
        weight=2.0,
        params={"command_name": "contact_schedule", "std": 0.15, "in_base_frame": True},
    )
    contact_pos_global = RewTermCfg(
        func=contact_schedule.contact_link_pos_gauss,
        weight=1.0,
        params={"command_name": "contact_schedule", "std": 0.15, "in_base_frame": False},
    )
    # regularization
    action_rate_l2 = RewTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    joint_torques_l2 = RewTermCfg(func=mdp.joint_torques_l2, weight=-1e-5)
    applied_torque_limits = RewTermCfg(func=mdp.applied_torque_limits, weight=-1e-3)
    joint_pos_limits = RewTermCfg(func=mdp.joint_pos_limits, weight=-10.0)
    joint_vel_limits = RewTermCfg(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9})


@configclass
class ACPRewardGroupsCfg(MultiRewardCfg):
    rewards = ACPRewardsCfg()


def _num_terrains(metadata_yaml: str) -> int:
    with open(metadata_yaml) as f:
        return len(yaml.safe_load(f)["terrains"])


@configclass
class G1ACPTrackerEnvCfg(G1PerceptiveShadowingEnvCfg):
    scene: ACPSceneCfg = ACPSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg.replace(
            data_class_type=ContactMotionReferenceData,
            contact_body_names=CONTACT_BODIES,
            motion_buffers={"TerrainMotion": ACPTerrainContactMotionCfg()},
        ),
    )
    commands: ACPCommandCfg = ACPCommandCfg()
    observations: ACPObservationsCfg = ACPObservationsCfg()
    rewards: ACPRewardGroupsCfg = ACPRewardGroupsCfg()

    def __post_init__(self):
        if self.sim.physics is None:
            from isaaclab_physx.physics import PhysxCfg

            from instinctlab.tasks.utils.newton import newton_physics_cfg

            # Smaller PhysX GPU buffers than the perceptive task (2**27), so the depth encoder's PPO update fits
            # next to 4096 envs on a 24 GB GPU.
            self.sim.physics = physics_preset(
                PhysxCfg(
                    gpu_max_rigid_patch_count=10 * 2**15,
                    gpu_max_rigid_contact_count=2**23,
                    gpu_collision_stack_size=2**26,
                ),
                newton=newton_physics_cfg(nconmax=128),
            )
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.tracker_height_scanner.prim_path = urdf_importer_link_prim_path(
            G1_CFG.spawn.asset_path, "torso_link"
        )
        # One terrain per row: curriculum mode gives row r the difficulty (r + U) / num_rows, which
        # motion_matched_terrain maps to terrain r when num_rows equals the number of terrains.
        generator = self.scene.terrain.terrain_generator
        generator.size = TERRAIN_SIZE
        generator.curriculum = True
        generator.num_rows = _num_terrains(os.path.join(MOTION_FOLDER, "metadata.yaml"))
        generator.num_cols = 2
        # friction range of the paper (Table S2)
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.2)
        self.run_name = "g1ACPTracker"


@configclass
class G1ACPTrackerEnvCfg_PLAY(G1ACPTrackerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        motion = self.scene.motion_reference.motion_buffers["TerrainMotion"]
        motion.motion_bin_length_s = None
        motion.env_starting_stub_sampling_strategy = "independent"
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.reset_robot.params["randomize_pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
