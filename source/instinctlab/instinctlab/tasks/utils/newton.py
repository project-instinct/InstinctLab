from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonGLVisualizer

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

if TYPE_CHECKING:
    from isaaclab.sensors import Camera
    from isaaclab.sim import SimulationContext
    from isaaclab.visualizers import PerspectiveCameraCfg


def newton_sim_cfg(
    njmax: int = 384,
    nconmax: int = 192,
    margin: float = 0.001,
    gap: float = 0.01,
    use_mujoco_contacts: bool = False,
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
        ),
        use_newton_actuators=True,
    )


class InstinctNewtonGLVisualizer(NewtonGLVisualizer):
    """Newton GL visualizer that can expose collision-only shapes clearly and follow the robot.

    Newton's viewer has a ``show_visual`` toggle but Isaac Lab's Newton backend configuration has no
    counterpart, so the viewer attribute is set directly after the backend has configured the viewer.
    Follow mode recenters the camera on the robot torso of the first environment.
    """

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        super().initialize(sim, cameras=cameras)
        if self.cfg.backend.show_visual is not None:
            self._viewer.show_visual = self.cfg.backend.show_visual
        if self.cfg.backend.follow_body:
            self._update_camera_to_body()

    def step(self, dt: float) -> None:
        super().step(dt)
        if self.cfg.backend.follow_body:
            self._update_camera_to_body()

    def _update_camera_to_body(self) -> None:
        """Point the Newton camera at the active G1 torso."""
        body_q = self.backend.state_0.body_q
        if body_q is None:
            return

        labels = list(self.backend.model.body_label)
        body_q = body_q.numpy()
        if not labels or len(body_q) != len(labels):
            return

        body_index = next((i for i, label in enumerate(labels) if "/Robot/torso_link" in label), None)
        if body_index is None:
            body_index = next((i for i, label in enumerate(labels) if "/Robot/pelvis" in label), None)
        if body_index is None:
            return

        target = body_q[body_index][0:3]
        eye = (float(target[0] + 1.5), float(target[1]), float(target[2] + 1.5))
        target = (float(target[0]), float(target[1]), float(target[2] + 0.4))
        self.set_camera_view(eye, target)


@configclass
class InstinctNewtonBackendCfg(NewtonGLBackendCfg):
    """Newton GL backend with InstinctLab's collision-shape visibility and follow-camera options.

    Isaac Lab's Newton visualizer wrapper does not surface Newton's ``show_visual`` setting.
    Collision-only shapes are rendered behind opaque visual meshes, which makes
    ``show_collision=True`` appear to do nothing on the robot even though the collision shapes are
    present in the Newton model.
    """

    class_type: type[InstinctNewtonGLVisualizer] | str = "{DIR}.newton:InstinctNewtonGLVisualizer"

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
