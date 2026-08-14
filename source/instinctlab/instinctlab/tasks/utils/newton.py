from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import (
    NewtonArticulationRootPropertiesCfg,
    NewtonMaterialPropertiesCfg,
)


def newton_sim_cfg(
    njmax: int = 256,
    nconmax: int = 128,
    margin: float = 0.01,
    gap: float = 0.01,
    use_mujoco_contacts: bool = True,
) -> SimulationCfg:
    """Return the shared Newton MJWarp simulation configuration."""
    return SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=njmax,
                nconmax=nconmax,
                iterations=100,
                ls_iterations=50,
                solver="newton",
                integrator="implicitfast",
                cone="pyramidal",
                impratio=1.0,
                ls_parallel=False,
                use_mujoco_contacts=use_mujoco_contacts,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
            default_shape_cfg=NewtonShapeCfg(margin=margin, gap=gap),
        ),
        use_newton_actuators=True,
    )


def apply_newton_robot_cfg(robot_cfg) -> None:
    """Apply Newton's stable articulation-root setting to a G1 robot config."""
    robot_cfg.spawn.articulation_props = NewtonArticulationRootPropertiesCfg(self_collision_enabled=False)


def newton_material_cfg() -> NewtonMaterialPropertiesCfg:
    """Return the default high-friction Newton terrain material."""
    return NewtonMaterialPropertiesCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
    )
