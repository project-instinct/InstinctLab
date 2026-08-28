from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from instinctlab.assets.unitree_g1 import G1_REFERENCE_CFG
from instinctlab.tasks.shadowing.beyondmimic.beyondmimic_env_cfg import BeyondMimicSceneCfg
from instinctlab.tasks.shadowing.beyondmimic.config.g1.beyondmimic_plane_cfg import (
    G1_CFG,
    G1BeyondMimicPlaneEnvCfg,
    G1BeyondMimicPlaneEnvCfg_PLAY,
    motion_reference_cfg,
)
from instinctlab.tasks.utils.newton import newton_sim_cfg


@configclass
class G1BeyondMimicNewtonSceneCfg(BeyondMimicSceneCfg):
    contact_forces = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class G1BeyondMimicNewtonEnvCfg(G1BeyondMimicPlaneEnvCfg):
    scene: G1BeyondMimicNewtonSceneCfg = G1BeyondMimicNewtonSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01)

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01).physics
        self.sim.use_newton_actuators = True


@configclass
class G1BeyondMimicNewtonEnvCfg_PLAY(G1BeyondMimicPlaneEnvCfg_PLAY):
    scene: G1BeyondMimicNewtonSceneCfg = G1BeyondMimicNewtonSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01)

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = newton_sim_cfg(njmax=224, nconmax=56, margin=0.0, gap=0.01).physics
        self.sim.use_newton_actuators = True
