import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
from instinctlab.assets.unitree_g1 import (
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

from .perceptive_vae_cfg import G1_CFG, G1PerceptiveVaeEnvCfg, G1PerceptiveVaeEnvCfg_PLAY, ObservationsCfg


_PROJECT_INSTINCT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 9)
)
_DATASET_FOLDER = os.path.join(_PROJECT_INSTINCT_ROOT, "data", "dataset_folder")
_WALK_AMP_SELECTION_YAML = os.path.join(_DATASET_FOLDER, "walk_amp.yaml")
MOTION_NAME = os.path.splitext(os.path.basename(_WALK_AMP_SELECTION_YAML))[0]


@configclass
class WalkAmpMotionCfg(AmassMotionCfgBase):
    path = _DATASET_FOLDER
    retargetting_func = None
    filtered_motion_selection_filepath = _WALK_AMP_SELECTION_YAML
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    motion_target_framerate = 50.0
    assumed_file_framerate = 50.0
    velocity_estimation_method = "frontbackward"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    symmetric_augmentation_link_mapping=[0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12],
    symmetric_augmentation_joint_mapping=G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    symmetric_augmentation_joint_reverse_buf=G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    motion_buffers={
        MOTION_NAME: WalkAmpMotionCfg(),
    },
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    mp_split_method="Even",
)


@configclass
class AmpObservationsCfg(ObservationsCfg):
    @configclass
    class AmpPolicyStateObsCfg(ObsGroupCfg):
        concatenate_terms = False

        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            history_length=10,
        )
        joint_pos_rel = ObsTermCfg(
            func=mdp.joint_pos_rel,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=True)},
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=True)},
        )
        base_lin_vel = ObsTermCfg(
            func=mdp.base_lin_vel,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    @configclass
    class AmpReferenceStateObsCfg(ObsGroupCfg):
        concatenate_terms = False

        projected_gravity = ObsTermCfg(
            func=instinct_mdp.projected_gravity_reference_as_state,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
            history_length=10,
        )
        joint_pos_rel = ObsTermCfg(
            func=instinct_mdp.joint_pos_rel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
        )
        joint_vel = ObsTermCfg(
            func=instinct_mdp.joint_vel_rel_reference_as_state,
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
        )
        base_lin_vel = ObsTermCfg(
            func=instinct_mdp.base_lin_vel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
        )
        base_ang_vel = ObsTermCfg(
            func=instinct_mdp.base_ang_vel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
        )

    amp_policy: AmpPolicyStateObsCfg = AmpPolicyStateObsCfg()
    amp_reference: AmpReferenceStateObsCfg = AmpReferenceStateObsCfg()


@configclass
class G1PerceptiveVaeAmpEnvCfg(G1PerceptiveVaeEnvCfg):
    observations: AmpObservationsCfg = AmpObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference = motion_reference_cfg
        self.run_name += "_amp"


@configclass
class G1PerceptiveVaeAmpEnvCfg_PLAY(G1PerceptiveVaeEnvCfg_PLAY):
    observations: AmpObservationsCfg = AmpObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference = motion_reference_cfg
        self.run_name += "_amp"
