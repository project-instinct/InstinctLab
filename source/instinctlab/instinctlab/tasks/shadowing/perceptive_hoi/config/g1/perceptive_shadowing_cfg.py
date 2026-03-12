import numpy as np
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.mdp as shadowing_mdp
import instinctlab.tasks.shadowing.perceptive_hoi.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
    beyondmimic_g1_29dof_delayed_actuators,
)
from instinctlab.monitors import ActuatorMonitorTerm, MonitorTermCfg, ShadowingBasePosMonitorTerm
from instinctlab.motion_reference import HoiMotionReferenceData, HoiMotionReferenceState, MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.omomo_motion_cfg import OmomoMotionCfg as OmomoMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.sim import MeshFileCfg

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

OMOMO_DATASET_PATH = "/localhdd/Datasets/OMOMO"

MESH_FILE_PATHS = {
    "floorlamp": "/localhdd/Datasets/OMOMO/data/captured_objects/floorlamp_cleaned_simplified.obj",
    "largebox": "/localhdd/Datasets/OMOMO/data/captured_objects/largebox_cleaned_simplified.obj",
    "whitechair": "/localhdd/Datasets/OMOMO/data/captured_objects/whitechair_cleaned_simplified.obj",
    "trashcan": "/localhdd/Datasets/OMOMO/data/captured_objects/trashcan_cleaned_simplified.obj",
    "smalltable": "/localhdd/Datasets/OMOMO/data/captured_objects/smalltable_cleaned_simplified.obj",
    "suitcase": "/localhdd/Datasets/OMOMO/data/captured_objects/suitcase_cleaned_simplified.obj",
}


@configclass
class OmomoMotionCfg(OmomoMotionCfgBase):
    path = os.path.expanduser(OMOMO_DATASET_PATH)
    ensure_link_below_zero_ground = False
    motion_start_from_middle_range = [0.0, 0.0]
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    env_starting_stub_sampling_strategy = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    data_class_type=HoiMotionReferenceData,
    state_class_type=HoiMotionReferenceState,
    scene_object_names=list(MESH_FILE_PATHS.keys()),
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
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.1,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    # set the robot_reference directly at where they are in the scene
    # DO NOT FORGET to change this when in actual training
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    motion_buffers={
        "OmomoMotion": OmomoMotionCfg(),
    },
    mp_split_method="None",
)


@configclass
class G1PerceptiveHoiShadowingEnvCfg(perceptual_cfg.PerceptiveHoiShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveHoiShadowingSceneCfg = perceptual_cfg.PerceptiveHoiShadowingSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    def __post_init__(self):
        super().__post_init__()

        # add mesh objects as scene rigid objects
        for object_name, mesh_file_path in MESH_FILE_PATHS.items():
            setattr(
                self.scene,
                object_name,
                RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/{object_name}",
                    spawn=MeshFileCfg(
                        asset_path=mesh_file_path,
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    ),
                ),
            )

        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))
        for object_name in list(MESH_FILE_PATHS.keys()):
            self.scene.camera.mesh_prim_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"/World/envs/env_.*/{object_name}")
            )

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        # self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 0.3
        self.actions.joint_pos.scale = beyondmimic_action_scale

        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]

        # match key links for observation terms
        self.observations.critic.link_pos.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests
        self.observations.critic.link_rot.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests

        self.run_name = "g1PerceptiveHoi" + "".join(
            [
                (
                    "_concatMotionBins"
                    if self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy
                    == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
            ]
        )
        # HOI task does not use terrain constraints.
        self.terminations.out_of_border = None


@configclass
class G1PerceptiveHoiShadowingEnvCfg_PLAY(G1PerceptiveHoiShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveHoiShadowingSceneCfg = perceptual_cfg.PerceptiveHoiShadowingSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=[1.5, 0.0, 1.5],
        lookat=[0.0, 0.0, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        # deactivate adaptive sampling and start from the 0.0s of the motion
        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_bin_length_s = None
        self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy = "independent"
        # BeyondConcatMotionAdaptiveWeighting requires _motion_bin_weights, which is only created when motion_bin_length_s is set
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        # self.scene.motion_reference.motion_buffers[MOTION_NAME].path = (
        #     "/localhdd/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall"
        # )
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True

        # change reset robot event with more pitch_down randomization (since the robot is facing -y axis)
        # self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.6)

        # remove some terimation terms
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None
        self.terminations.out_of_border = None

        # put the reference in scene and move the robot elsewhere and visualize the reference
        # self.events.reset_robot.params["position_offset"] = [0.0, 1.0, 2.0]
        # self.scene.motion_reference.visualizing_robot_offset = (0.0, 0.0, 0.0)
        # self.viewer.asset_name = "robot_reference"

        # remove some randomizations
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.reset_robot.params["randomize_pose_range"]["x"] = [0.0] * 2  # (+-0.6)
        self.events.reset_robot.params["randomize_pose_range"]["y"] = [0.0] * 2  # (+-0.6)
        self.events.reset_robot.params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)

        # add some additional monitor terms
        self.monitors.shadowing_position_stats = MonitorTermCfg(
            func=ShadowingBasePosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
            ),
        )
        self.monitors.right_ankle_pitch_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
            ),
        )
        self.monitors.left_ankle_pitch_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
            ),
        )
        self.monitors.right_knee_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
            ),
        )
        self.monitors.left_knee_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
            ),
        )
