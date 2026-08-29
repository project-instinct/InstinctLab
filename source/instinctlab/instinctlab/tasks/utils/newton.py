from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass


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
                ls_parallel=False,
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


@configclass
class InstinctNewtonVisualizerCfg(NewtonVisualizerCfg):
    """Newton visualizer config that can expose collision-only shapes clearly.

    Isaac Lab's Newton visualizer wrapper currently does not surface Newton's
    ``show_visual`` setting.  Collision-only shapes are rendered behind opaque
    visual meshes, which makes ``show_collision=True`` appear to do nothing on
    the robot even though the collision shapes are present in the Newton model.
    """

    show_visual: bool | None = None
    """When set, override the underlying Newton viewer's visual-shape visibility."""

    follow_body: bool = False
    """When true, recenter the Newton camera on the active G1 torso every frame."""

    def create_visualizer(self):
        visualizer = super().create_visualizer()
        original_initialize = visualizer.initialize

        def initialize(scene_data_provider):
            original_initialize(scene_data_provider)
            if self.show_collision:
                visualizer._viewer.show_collision = True
            if self.show_visual is not None:
                visualizer._viewer.show_visual = self.show_visual
            if self.follow_body:
                visualizer._viewer.model_changed = True
                self._update_camera_to_body(visualizer)

        visualizer.initialize = initialize

        if self.follow_body:
            original_step = visualizer.step

            def step(dt):
                visualizer._viewer.model_changed = True
                if self.show_collision:
                    visualizer._viewer.show_collision = True
                if self.show_visual is not None:
                    visualizer._viewer.show_visual = self.show_visual
                self._update_camera_to_body(visualizer)
                original_step(dt)

            visualizer.step = step

        return visualizer

    def _update_camera_to_body(self, visualizer) -> None:
        """Point the Newton camera at the active G1 torso."""
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        if model is None or state is None:
            return

        labels = list(model.body_label)
        body_q = state.body_q.numpy()
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
        visualizer._apply_camera_pose((eye, target))
