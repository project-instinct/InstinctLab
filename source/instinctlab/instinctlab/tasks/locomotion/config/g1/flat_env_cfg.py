import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ViewerCfg, mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as Event
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_POLICY_JOINT_ORDER_V1,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
)
from instinctlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinctlab.monitors import MonitorTermCfg
from instinctlab.sensors import HierarchicalContactSensorCfg
from instinctlab.utils.config import set_cfg_joint_order

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG


# ============================================================================
# Scene Configuration
# ============================================================================


@configclass
class G1FlatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = HierarchicalContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
# Actions Configuration
# ============================================================================


@configclass
class G1FlatActionsCfg:
    joint_pos = instinct_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


# ============================================================================
# Commands Configuration
# ============================================================================


@configclass
class G1FlatCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi)
        ),
    )


# ============================================================================
# Observations Configuration
# ============================================================================


@configclass
class G1FlatPolicyObsCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=instinct_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=instinct_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class G1FlatCriticObsCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=instinct_mdp.base_ang_vel)
    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity,
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel)
    actions = ObsTerm(func=instinct_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class G1FlatObservationsCfg:
    policy: G1FlatPolicyObsCfg = G1FlatPolicyObsCfg()
    critic: G1FlatCriticObsCfg = G1FlatCriticObsCfg()


# ============================================================================
# Rewards Configuration
# ============================================================================


@configclass
class G1FlatRewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=locomotion_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=locomotion_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=instinct_mdp.contact_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.8})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    stand_still = RewTerm(func=locomotion_mdp.stand_still, weight=-0.8, params={"command_name": "base_velocity"})
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    dof_torques_l2 = RewTerm(
        func=instinct_mdp.joint_torques_l2,
        weight=-4.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )


# ============================================================================
# Terminations Configuration
# ============================================================================


@configclass
class G1FlatTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_.*",
                    ".*_hip_.*",
                    ".*_knee_.*",
                ],
            ),
            "threshold": 1.0,
        },
    )


# ============================================================================
# Events Configuration
# ============================================================================


@configclass
class G1FlatEventsCfg:
    physics_material = Event(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 0.8),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.8),
            "num_buckets": 64,
        },
    )
    add_base_mass = Event(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_external_force_torque = Event(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = Event(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = Event(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (-1.0, 1.0),
        },
    )
    push_robot = Event(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# ============================================================================
# Curriculum Configuration
# ============================================================================


@configclass
class G1FlatCurriculumCfg:
    # terrain_levels = CurrTerm(func=locomotion_mdp.terrain_levels_vel)
    pass


@configclass
class G1FlatMonitorCfg:
    actuator = MonitorTermCfg(
        func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "torque_plot_scale": 1.0,
            "joint_vel_plot_scale": 1.0e-1,
            "joint_power_plot_scale": 1.0e-1,
        },
    )


# ============================================================================
# Environment Configuration
# ============================================================================


@configclass
class G1FlatEnvCfg(InstinctLabRLEnvCfg):
    scene: G1FlatSceneCfg = G1FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: G1FlatActionsCfg = G1FlatActionsCfg()
    commands: G1FlatCommandsCfg = G1FlatCommandsCfg()
    observations: G1FlatObservationsCfg = G1FlatObservationsCfg()
    rewards: G1FlatRewardsCfg = G1FlatRewardsCfg()
    terminations: G1FlatTerminationsCfg = G1FlatTerminationsCfg()
    monitors: G1FlatMonitorCfg = G1FlatMonitorCfg()
    events: G1FlatEventsCfg = G1FlatEventsCfg()
    curriculum: G1FlatCurriculumCfg = G1FlatCurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.0, 2.0, 0.5), lookat=(0.0, 0.0, 0.0), origin_type="asset_root", asset_name="robot"
    )

    def __post_init__(self):
        joint_order = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
        set_cfg_joint_order(self.actions.joint_pos, joint_order)
        self.observations.policy.joint_pos.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.critic.joint_pos.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.critic.joint_vel.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.actions.joint_pos.scale = beyondmimic_action_scale

        self.run_name = "".join(
            [
                "G1Flat",
                (
                    f"_feetAirTime{self.rewards.feet_air_time.weight:.2f}"
                    if self.rewards.feet_air_time is not None
                    else ""
                ),
                f"_standStill{-self.rewards.stand_still.weight:.2f}" if self.rewards.stand_still is not None else "",
                (
                    f"_actionRate{-self.rewards.action_rate_l2.weight:.2f}"
                    if self.rewards.action_rate_l2 is not None
                    else ""
                ),
                (
                    f"_jointDeviationKnee{-self.rewards.joint_deviation_knee.weight:.2f}"
                    if self.rewards.joint_deviation_knee is not None
                    else ""
                ),
            ]
        )


@configclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    scene: G1FlatSceneCfg = G1FlatSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        # self.observations.policy.enable_corruption = False
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
