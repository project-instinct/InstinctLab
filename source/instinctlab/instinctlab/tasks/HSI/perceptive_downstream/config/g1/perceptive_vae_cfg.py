import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.HSI.perceptive_downstream.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import ActuatorMonitorTerm, MonitorTermCfg
from instinctlab.sensors import get_link_prim_targets

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG
# Must match frozen VAE bundle (dagger run name uses propHistory4_depthHist10Skip3).
PROPRIO_HISTORY_LENGTH = 4
TEACHER_PROPRIO_HISTORY_LENGTH = 8


@configclass
class ObservationsCfg:
    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        depth_image = ObsTermCfg(
            func=instinct_mdp.visualizable_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised_history",
                "history_skip_frames": 2,
            },
        )

        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTermCfg(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticObsCfg(ObsGroupCfg):
        height_scan = ObsTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=[-20.0, 20.0],
        )

        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTermCfg(func=mdp.last_action, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class G1PerceptiveVaePlayMonitorCfg:
    right_ankle_pitch_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
        ),
    )
    left_ankle_pitch_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
        ),
    )
    right_knee_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
        ),
    )
    left_knee_actuator = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
        ),
    )


@configclass
class G1PerceptiveVaeEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))
        self.scene.camera.data_histories["distance_to_image_plane_noised"] = 10
        self.observations.policy.depth_image.params["history_skip_frames"] = 3
        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale

        self.run_name = "g1PerceptiveVae" + "".join(
            [
                f"_propHistory{PROPRIO_HISTORY_LENGTH}",
                f"_depthHist{self.scene.camera.data_histories['distance_to_image_plane_noised']}Skip{self.observations.policy.depth_image.params['history_skip_frames']}",
            ]
        )


@configclass
class G1PerceptiveVaeEnvCfg_PLAY(G1PerceptiveVaeEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=10,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    monitors: G1PerceptiveVaePlayMonitorCfg | None = G1PerceptiveVaePlayMonitorCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=[0.0, 2.0, 2.5],
        lookat=[0.0, 0.0, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator.num_rows = 13
        self.scene.terrain.terrain_generator.num_cols = 13

        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True

        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None

        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
