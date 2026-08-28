from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from instinctlab.assets.unitree_g1 import G1_REFERENCE_CFG
from instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg import (
    G1_CFG,
    G1PerceptiveShadowingEnvCfg,
    G1PerceptiveShadowingEnvCfg_PLAY,
    motion_reference_cfg,
)
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import PerceptiveShadowingSceneCfg
from instinctlab.tasks.utils.newton import InstinctNewtonVisualizerCfg, newton_sim_cfg


@configclass
class G1PerceptiveNewtonSceneCfg(PerceptiveShadowingSceneCfg):
    contact_forces = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class G1PerceptiveNewtonEnvCfg(G1PerceptiveShadowingEnvCfg):
    scene: G1PerceptiveNewtonSceneCfg = G1PerceptiveNewtonSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=256, nconmax=128, margin=0.01, gap=0.01, use_mujoco_contacts=False)

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = newton_sim_cfg(
            njmax=256, nconmax=128, margin=0.01, gap=0.01, use_mujoco_contacts=False
        ).physics
        self.sim.use_newton_actuators = True
        self.events.reset_robot.params["position_offset"] = [0.0, 0.0, 0.05]


@configclass
class G1PerceptiveNewtonEnvCfg_PLAY(G1PerceptiveShadowingEnvCfg_PLAY):
    scene: G1PerceptiveNewtonSceneCfg = G1PerceptiveNewtonSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=256, nconmax=128, margin=0.01, gap=0.01, use_mujoco_contacts=False)

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = newton_sim_cfg(
            njmax=256, nconmax=128, margin=0.01, gap=0.01, use_mujoco_contacts=False
        ).physics
        self.sim.use_newton_actuators = True
        self.sim.visualizer_cfgs = [
            InstinctNewtonVisualizerCfg(show_collision=True, show_contacts=True, show_visual=False, follow_body=True)
        ]
