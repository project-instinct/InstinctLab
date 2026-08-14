import numpy as np
import os

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.mdp as shadowing_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_REFERENCE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
    beyondmimic_g1_29dof_delayed_actuators,
    configure_g1_29dof_policy_io,
)
from instinctlab.monitors import MonitorTermCfg
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg as TerrainMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.utils.urdf import urdf_importer_link_prim_path

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

MOTION_FOLDER = "~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep6"


@configclass
class TerrainMotionCfg(TerrainMotionCfgBase):
    path = os.path.expanduser(MOTION_FOLDER)
    metadata_yaml = os.path.expanduser(f"{MOTION_FOLDER}/metadata.yaml")
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
class AMASSMotionCfg(AmassMotionCfgBase):
    path = os.path.expanduser(MOTION_FOLDER)
    filtered_motion_selection_filepath = None
    ensure_link_below_zero_ground = False
    motion_start_from_middle_range = [0.0, 0.0]
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    env_starting_stub_sampling_strategy = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference",
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
        "TerrainMotion": TerrainMotionCfg(),
    },
    mp_split_method="None",
)


@configclass
class G1PerceptiveShadowingEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    def __post_init__(self):
        super().__post_init__()
        configure_g1_29dof_policy_io(self)

        self.scene.height_scanner.prim_path = urdf_importer_link_prim_path(G1_CFG.spawn.asset_path, "torso_link")
        self.scene.camera.prim_path = urdf_importer_link_prim_path(G1_CFG.spawn.asset_path, "torso_link")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS, G1_CFG.spawn.asset_path))

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        # self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 0.3
        self.actions.joint_pos.scale = beyondmimic_action_scale

        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml = os.path.join(
            self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
        )
        PLANE_TERRAIN = False
        if PLANE_TERRAIN:
            self.scene.motion_reference.motion_buffers.pop(MOTION_NAME)
            self.scene.motion_reference.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
        else:
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].path = (
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path
            )
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = os.path.join(
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
            )

        # match key links for observation terms
        self.observations.critic.link_pos.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests
        self.observations.critic.link_rot.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests

        self.run_name = "g1Perceptive" + "".join(
            [
                (
                    "_concatMotionBins"
                    if self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy
                    == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
            ]
        )


@configclass
class G1PerceptiveShadowingEnvCfg_PLAY(G1PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        # deactivate adaptive sampling and start from the 0.0s of the motion
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_bin_length_s = None
        self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy = "independent"
        # self.scene.motion_reference.motion_buffers[MOTION_NAME].path = (
        #     "/localhdd/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall"
        # )
        self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml = os.path.join(
            self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
        )
        if self.scene.terrain.terrain_type == "hacked_generator":
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].path = (
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path
            )
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = os.path.join(
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
            )

        # Use non-terrain-matching motion and plane to hack the scene.
        self.scene.terrain.terrain_generator.num_rows = 6
        self.scene.terrain.terrain_generator.num_cols = 6
        # self.scene.motion_reference.motion_buffers.pop(MOTION_NAME)
        # self.scene.motion_reference.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_start_from_middle_range = [0.0, 0.0]
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_bin_length_s = None
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True

        # change reset robot event with more pitch_down randomization (since the robot is facing -y axis)
        # self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.6)

        # remove some terimation terms
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None

        # put the reference in scene and move the robot elsewhere and visualize the reference
        # self.events.reset_robot.params["position_offset"] = [0.0, 1.0, 2.0]
        # self.scene.motion_reference.visualizing_robot_offset = (0.0, 0.0, 0.0)

        # remove some randomizations
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None
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
            func="instinctlab.monitors.monitors:ShadowingBasePosMonitorTerm",
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
            ),
        )
        self.monitors.right_ankle_pitch_actuator = MonitorTermCfg(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
            ),
        )
        self.monitors.left_ankle_pitch_actuator = MonitorTermCfg(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
            ),
        )
        self.monitors.right_knee_actuator = MonitorTermCfg(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
            ),
        )
        self.monitors.left_knee_actuator = MonitorTermCfg(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
            ),
        )

        # add another box to the scene (to test visual generalization)
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(4.8, 0.6, 0.5),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.35, 0.25)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cube/.*")
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(4.8, 2.6, 0.5),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.35, 0.25)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cube/.*")
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(0.1, 4.6, 1.8),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.8, -1.35, 0.25)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cube/.*")

        # self.scene.distractor1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cone1",
        #     spawn=sim_utils.MeshConeCfg(
        #         radius=0.22,
        #         height=0.55,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.0, 0.3)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cone1/.*")

        # see the reference robot
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/RobotReference/.*")
