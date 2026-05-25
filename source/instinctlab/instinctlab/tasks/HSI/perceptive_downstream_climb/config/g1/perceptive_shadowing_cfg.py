import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.tasks.HSI.perceptive_downstream_climb.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import ActuatorMonitorTerm, MonitorTermCfg
from instinctlab.sensors import get_link_prim_targets

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG


@configclass
class G1PerceptiveShadowingPlayMonitorCfg:
    """Minimal monitors for PLAY (motion reference disabled)."""

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
class G1PerceptiveShadowingEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale

        self.observations.critic.link_pos.params["asset_cfg"].body_names = G1_29DOF_LINKS
        self.observations.critic.link_rot.params["asset_cfg"].body_names = G1_29DOF_LINKS

        self.run_name = "g1PerceptiveClimb"


@configclass
class G1PerceptiveShadowingEnvCfg_PLAY(G1PerceptiveShadowingEnvCfg):
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=10,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    monitors: G1PerceptiveShadowingPlayMonitorCfg | None = G1PerceptiveShadowingPlayMonitorCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=[1.5, 0.0, 1.5],
        lookat=[0.0, 0.0, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2

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
