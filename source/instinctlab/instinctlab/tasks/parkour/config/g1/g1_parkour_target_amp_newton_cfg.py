from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY
from instinctlab.tasks.parkour.config.parkour_env_cfg import SceneCfg
from instinctlab.tasks.utils.newton import InstinctNewtonVisualizerCfg, newton_sim_cfg


@configclass
class G1ParkourNewtonSceneCfg(SceneCfg):
    contact_forces = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class G1ParkourNewtonEnvCfg(G1ParkourEnvCfg):
    scene: G1ParkourNewtonSceneCfg = G1ParkourNewtonSceneCfg(num_envs=4096, env_spacing=2.5)
    sim: SimulationCfg = newton_sim_cfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = newton_sim_cfg().physics
        self.sim.use_newton_actuators = True


@configclass
class G1ParkourNewtonEnvCfg_PLAY(G1ParkourEnvCfg_PLAY):
    scene: G1ParkourNewtonSceneCfg = G1ParkourNewtonSceneCfg(num_envs=10, env_spacing=2.5)
    sim: SimulationCfg = newton_sim_cfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = newton_sim_cfg().physics
        self.sim.use_newton_actuators = True
        self.sim.visualizer_cfgs = [
            InstinctNewtonVisualizerCfg(
                follow_body=True,
            )
        ]
