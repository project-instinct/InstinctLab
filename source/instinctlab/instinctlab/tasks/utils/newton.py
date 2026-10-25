from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

if TYPE_CHECKING:
    from .newton_visualizer import InstinctNewtonGLVisualizer


def newton_physics_cfg(
    njmax: int = 384,
    nconmax: int = 192,
    margin: float = 0.001,
    gap: float = 0.01,
    use_mujoco_contacts: bool = False,
) -> NewtonCfg:
    """Return the shared Newton MJWarp physics configuration."""
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=njmax,
            nconmax=nconmax,
            iterations=100,
            ls_iterations=50,
            solver="newton",
            integrator="implicitfast",
            cone="pyramidal",
            impratio=1.0,
            use_mujoco_contacts=use_mujoco_contacts,
        ),
        collision_cfg=(
            None
            if use_mujoco_contacts
            else NewtonCollisionPipelineCfg(
                broad_phase="explicit",
                max_triangle_pairs=2_500_000,
            )
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(margin=margin, gap=gap),
    )


@configclass
class InstinctNewtonBackendCfg(NewtonGLBackendCfg):
    """Newton GL backend with InstinctLab's collision-shape visibility and follow-camera options.

    Isaac Lab's Newton visualizer wrapper does not surface Newton's ``show_visual`` setting.
    Collision-only shapes are rendered behind opaque visual meshes, which makes
    ``show_collision=True`` appear to do nothing on the robot even though the collision shapes are
    present in the Newton model.
    """

    class_type: type[InstinctNewtonGLVisualizer] | str = "{DIR}.newton_visualizer:InstinctNewtonGLVisualizer"

    show_visual: bool | None = None
    """When set, override the underlying Newton viewer's visual-shape visibility."""

    follow_body: bool = False
    """When true, recenter the Newton camera on the active G1 torso every frame."""


@configclass
class InstinctNewtonVisualizerCfg(VisualizerCfg):
    """Newton GL view backed by :class:`InstinctNewtonBackendCfg`."""

    backend: InstinctNewtonBackendCfg = InstinctNewtonBackendCfg()
    streaming_view: bool = True
    """Keep the streaming camera panel enabled, as Isaac Lab's deprecated Newton GL config did."""
