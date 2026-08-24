from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from instinctlab.assets.unitree_g1 import G1_REFERENCE_CFG
from instinctlab.tasks.shadowing.perceptive_hoi.config.g1.perceptive_shadowing_cfg import (
    G1_CFG,
    G1PerceptiveHoiShadowingEnvCfg,
    G1PerceptiveHoiShadowingEnvCfg_PLAY,
    motion_reference_cfg,
)
from instinctlab.tasks.shadowing.perceptive_hoi.perceptive_env_cfg import PerceptiveHoiShadowingSceneCfg
from instinctlab.tasks.utils.newton import apply_newton_robot_cfg, newton_material_cfg, newton_sim_cfg


@configclass
class G1PerceptiveHoiNewtonSceneCfg(PerceptiveHoiShadowingSceneCfg):
    contact_forces = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    def __post_init__(self):
        super().__post_init__()
        self.terrain.physics_material = newton_material_cfg()


@configclass
class G1PerceptiveHoiNewtonEnvCfg(G1PerceptiveHoiShadowingEnvCfg):
    scene: G1PerceptiveHoiNewtonSceneCfg = G1PerceptiveHoiNewtonSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01)

    def __post_init__(self):
        super().__post_init__()
        apply_newton_robot_cfg(self.scene.robot)
        self.sim.physics = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01).physics
        self.sim.use_newton_actuators = True


@configclass
class G1PerceptiveHoiNewtonEnvCfg_PLAY(G1PerceptiveHoiShadowingEnvCfg_PLAY):
    scene: G1PerceptiveHoiNewtonSceneCfg = G1PerceptiveHoiNewtonSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01)

    def __post_init__(self):
        super().__post_init__()
        apply_newton_robot_cfg(self.scene.robot)
        self.sim.physics = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01).physics
        self.sim.use_newton_actuators = True
