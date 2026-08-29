from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG, beyondmimic_g1_29dof_actuators

from .flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY, G1FlatSceneCfg


def _newton_sim_cfg() -> SimulationCfg:
    """Return the fixed MJWarp configuration for the Newton task IDs."""
    return SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=112,
                nconmax=28,
                iterations=100,
                ls_iterations=50,
                solver="newton",
                integrator="implicitfast",
                cone="pyramidal",
                impratio=1.0,
                ls_parallel=False,
                use_mujoco_contacts=True,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.01),
        ),
        use_newton_actuators=True,
    )


def _newton_robot_cfg():
    """Return the self-colliding G1 asset with the locomotion actuators."""
    robot_cfg = G1_29DOF_TORSOBASE_POPSICLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.actuators = beyondmimic_g1_29dof_actuators
    return robot_cfg


@configclass
class G1FlatNewtonSceneCfg(G1FlatSceneCfg):
    """G1 flat scene using Newton's native contact sensor."""

    robot = _newton_robot_cfg()
    contact_forces = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    def __post_init__(self):
        self.terrain.physics_material = NewtonMaterialPropertiesCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        )


@configclass
class G1FlatNewtonEnvCfg(G1FlatEnvCfg):
    """Registered Newton training configuration."""

    scene: G1FlatNewtonSceneCfg = G1FlatNewtonSceneCfg(num_envs=4096, env_spacing=2.5)
    sim: SimulationCfg = _newton_sim_cfg()


@configclass
class G1FlatNewtonEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Registered Newton play configuration."""

    scene: G1FlatNewtonSceneCfg = G1FlatNewtonSceneCfg(num_envs=1, env_spacing=2.5)
    sim: SimulationCfg = _newton_sim_cfg()
