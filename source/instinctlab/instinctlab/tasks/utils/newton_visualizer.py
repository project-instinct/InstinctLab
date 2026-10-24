"""Newton GL visualizer subclass, kept apart from the task configs.

Importing it loads Newton and its USD bindings, which must not happen before a Kit app starts; the configs in
:mod:`instinctlab.tasks.utils.newton` select it lazily through ``class_type``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_visualizers.newton import NewtonGLVisualizer

if TYPE_CHECKING:
    from isaaclab.sensors import Camera
    from isaaclab.sim import SimulationContext
    from isaaclab.visualizers import PerspectiveCameraCfg


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
