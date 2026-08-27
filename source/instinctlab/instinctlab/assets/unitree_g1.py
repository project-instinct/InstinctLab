"""
Customized Unitree G1 asset for Isaac Sim
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.schemas import CollisionBaseCfg, JointDriveBaseCfg
from isaaclab_assets import G1_CFG

from instinctlab.sim import UrdfFileCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

G1_29DOF_POLICY_JOINT_ORDER_V1 = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
"""Versioned policy joint order retained from the Isaac Lab 2.3.2 experiments.

Changing this tuple is a breaking policy-interface change. Define a new version instead.
"""


G1_29DOF_TORSOBASE_CFG = G1_CFG.copy()
G1_29DOF_TORSOBASE_CFG.spawn = UrdfFileCfg(
    asset_path=os.path.join(__file_dir__, "resources/unitree_g1/urdf/g1_29dof_torsobase_simplified.urdf"),
    merge_fixed_joints=True,
    fix_base=False,
    self_collision=True,
    activate_contact_sensors=True,
)
G1_29DOF_TORSOBASE_CFG.soft_joint_pos_limit_factor = 0.95
G1_29DOF_TORSOBASE_CFG.actuators = {
    # NOTE: checked, delayed PD actuator has same time-lag when computing torques; and no lag when
    # the num_pushes does not reach the lag time.
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "waist_.*_joint",
        ],
        effort_limit={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 88.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
            "waist_roll_joint": 50.0,
            "waist_pitch_joint": 50.0,
            "waist_yaw_joint": 88.0,
        },
        velocity_limit=60.0,
        stiffness={
            ".*_hip_yaw_joint": 90.0,
            ".*_hip_roll_joint": 90.0,
            ".*_hip_pitch_joint": 90.0,
            ".*_knee_joint": 140.0,
            "waist_roll_joint": 60.0,
            "waist_pitch_joint": 60.0,
            "waist_yaw_joint": 90.0,
        },
        damping={
            ".*_hip_yaw_joint": 2.0,
            ".*_hip_roll_joint": 2.0,
            ".*_hip_pitch_joint": 2.0,
            ".*_knee_joint": 2.5,
            "waist_.*_joint": 2.5,
        },
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
    "feet": DelayedPDActuatorCfg(
        effort_limit=20,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=20.0,
        damping=1.0,
        velocity_limit=60.0,
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
    "arms": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
        ],
        effort_limit=25,
        velocity_limit=60.0,
        stiffness=25,
        damping={
            ".*_shoulder_.*_joint": 1.0,
            ".*_elbow_joint": 1.0,
        },
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
    "wrist": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*wrist_roll_joint",
            ".*wrist_pitch_joint",
            ".*wrist_yaw_joint",
        ],
        effort_limit={
            ".*wrist_roll_joint": 25.0,
            ".*wrist_pitch_joint": 5.0,
            ".*wrist_yaw_joint": 5.0,
        },
        velocity_limit=25.0,
        stiffness={
            ".*wrist_roll_joint": 25.0,
            ".*wrist_pitch_joint": 5.0,
            ".*wrist_yaw_joint": 5.0,
        },
        damping={
            ".*wrist_roll_joint": 1.0,
            ".*wrist_pitch_joint": 0.5,
            ".*wrist_yaw_joint": 0.5,
        },
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
}
G1_29DOF_TORSOBASE_CFG.init_state = G1_CFG.init_state.copy()
G1_29DOF_TORSOBASE_CFG.init_state.joint_pos = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
}

G1_29DOF_TORSOBASE_CLOG_CFG = G1_29DOF_TORSOBASE_CFG.copy()
G1_29DOF_TORSOBASE_CLOG_CFG.spawn = UrdfFileCfg(
    asset_path=os.path.join(__file_dir__, "resources/unitree_g1/urdf/g1_29dof_torsobase_clog.urdf"),
    merge_fixed_joints=True,
    fix_base=False,
    self_collision=True,
    activate_contact_sensors=True,
    collision_type="Convex Decomposition",
)

G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping = [
    1,
    0,
    2,  # waist pitch
    4,
    3,
    5,  # waist roll
    7,
    6,
    8,  # waist yaw
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
    18,
    17,
    20,
    19,
    22,
    21,
    24,
    23,
    26,
    25,
    28,
    27,
]
G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf = [
    1,
    1,
    1,  # waist pitch
    -1,
    -1,  # shoulder roll
    -1,  # waist roll
    -1,
    -1,  # shoulder yaw
    -1,  # waist yaw
    1,
    1,
    1,
    1,
    -1,
    -1,  # wrist roll
    -1,
    -1,  # hip roll
    1,
    1,  # wrist pitch
    -1,
    -1,  # hip yaw
    -1,
    -1,  # wrist yaw
    1,
    1,
    1,
    1,
    -1,
    -1,  # ankle roll
]


# Following the principles of BeyondMimic, and the kp/kd computation logic.
# NOTE: These logic are still being tested, so we put them here for substitution in users Cfg class
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

beyondmimic_g1_29dof_actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_7520_14,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_knee_joint": ARMATURE_7520_22,
        },
    ),
    "feet": ImplicitActuatorCfg(
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    "waist": ImplicitActuatorCfg(
        effort_limit_sim=50,
        velocity_limit_sim=37.0,
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    "waist_yaw": ImplicitActuatorCfg(
        effort_limit_sim=88,
        velocity_limit_sim=32.0,
        joint_names_expr=["waist_yaw_joint"],
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        armature=ARMATURE_7520_14,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        damping={
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
    ),
}


beyondmimic_g1_29dof_delayed_actuators = {
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_7520_14,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_knee_joint": ARMATURE_7520_22,
        },
        min_delay=0,
        max_delay=2,
    ),
    "feet": DelayedPDActuatorCfg(
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
        min_delay=0,
        max_delay=2,
    ),
    "waist": DelayedPDActuatorCfg(
        effort_limit_sim=50,
        velocity_limit_sim=37.0,
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
        min_delay=0,
        max_delay=2,
    ),
    "waist_yaw": DelayedPDActuatorCfg(
        effort_limit_sim=88,
        velocity_limit_sim=32.0,
        joint_names_expr=["waist_yaw_joint"],
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        armature=ARMATURE_7520_14,
        min_delay=0,
        max_delay=2,
    ),
    "arms": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        damping={
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
        min_delay=0,
        max_delay=2,
    ),
}


beyondmimic_action_scale = {}
for a in beyondmimic_g1_29dof_actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            beyondmimic_action_scale[n] = 0.25 * e[n] / s[n]

G1_SOURCE_URDF_PATH = os.path.join(__file_dir__, "resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf")

G1_29DOF_TORSOBASE_POPSICLE_CFG = G1_29DOF_TORSOBASE_CFG.copy()
G1_29DOF_TORSOBASE_POPSICLE_CFG.spawn = UrdfFileCfg(
    asset_path=G1_SOURCE_URDF_PATH,
    replace_cylinders_with_capsules=True,
    merge_fixed_joints=True,
    fix_base=False,
    self_collision=True,
    activate_contact_sensors=True,
    joint_drive_props=JointDriveBaseCfg(ensure_drives_exist=True),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    ),
)
G1_29DOF_TORSOBASE_POPSICLE_CFG.spawn.activate_contact_sensors = True
G1_29DOF_TORSOBASE_POPSICLE_CFG.soft_joint_pos_limit_factor = 0.9
G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state = G1_29DOF_TORSOBASE_CFG.init_state.copy()
G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.pos = (0.0, 0.0, 0.82)
G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.joint_pos = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.joint_vel = {".*": 0.0}
G1_29DOF_TORSOBASE_POPSICLE_CFG.actuators = beyondmimic_g1_29dof_actuators

_G1_REFERENCE_SPAWN_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG.spawn.copy()
_G1_REFERENCE_SPAWN_CFG.activate_contact_sensors = False
_G1_REFERENCE_SPAWN_CFG.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
_G1_REFERENCE_SPAWN_CFG.collision_props = CollisionBaseCfg(collision_enabled=False)
_G1_REFERENCE_SPAWN_CFG.joint_drive_props = JointDriveBaseCfg(stiffness=0.0, damping=0.0, max_force=1.0e6)
G1_REFERENCE_CFG = AssetBaseCfg(
    spawn=_G1_REFERENCE_SPAWN_CFG,
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.pos,
        rot=G1_29DOF_TORSOBASE_POPSICLE_CFG.init_state.rot,
    ),
)

G1_29DOF_LINKS = [  # Order not guaranteed.
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
    "waist_roll_link",
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
