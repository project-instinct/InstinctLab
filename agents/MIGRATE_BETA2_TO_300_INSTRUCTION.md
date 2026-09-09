# Migrating a downstream project from Isaac Lab 3.0.0-beta2 to 3.0.0

This is a practical guide for projects built on InstinctLab, such as GI Lab.
Use the current InstinctLab migration as the reference implementation, then port
only the API areas your project uses.

## Target environment

Use a separate checkout and environment while migrating. Do not switch an
existing beta2 checkout in place.

| Component | Pinned target |
|---|---|
| Python | `>=3.12,<3.13` |
| Isaac Sim | `6.0.1.0` |
| Isaac Lab source | `release/3.0.0` at `6e0cbe2c3953b2e32fefc1065f4c4f5818f458c3` |
| `isaaclab` | `16.4.0` |
| `isaaclab-assets` | `0.6.4` |
| `isaaclab-physx` | `5.1.0` |
| `isaaclab-tasks` | `17.0.0` |
| `isaaclab-visualizers` | `1.7.0` |
| `isaaclab-newton` | `5.3.0` |
| Torch / Vision / Audio | `2.11.0` / `0.26.0` / `2.11.0` |
| Warp | `1.16.0` |
| Gymnasium | `1.3.0` |

The Isaac Lab SHA is authoritative. Do not use a floating branch name in a
release, experiment record, or downstream README.

Create the isolated Isaac Lab checkout first:

```bash
git clone --branch release/3.0.0 --single-branch \
  https://github.com/isaac-sim/IsaacLab.git isaaclab300
git -C isaaclab300 switch --detach 6e0cbe2c3953b2e32fefc1065f4c4f5818f458c3
```

Install Isaac Lab with its own installer and lock file, then install
InstinctLab and the downstream project as editable packages in that order.
Downstream projects that intentionally leave `INSTALL_REQUIRES` empty must say
so in their README and document this installation order.

## Migration order

### 1. Update metadata and documentation

- Pin the Isaac Lab source SHA and package versions above.
- Update the Isaac Sim, Isaac Lab, Python, Warp, and Gymnasium requirements.
- Record the compatible InstinctLab commit in the downstream README.
- Keep the beta2 environment intact until the new environment passes the
  validation below.

### 2. Port launch and video code

Import launch helpers from `isaaclab.app`:

```python
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import get_checkpoint_path
```

Remove the beta2-only `--/isaaclab/has_gui=true` workaround. Also remove
`render_mode="rgb_array"` from `gym.make` and replace
`gym.wrappers.RecordVideo` with configuration-owned recording before creating
the environment:

```python
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

env_cfg.video_recorders = [
    VideoRecorderCfg(
        source="visualizer",
        output_dir=video_dir,
        video_interval=args_cli.video_interval,
        video_length=args_cli.video_length,
    )
]
env = gym.make(args_cli.task, cfg=env_cfg)
```

Set an explicit recording source suitable for the project, then verify the
first frame, frame interval, resolution, and output length. Do not assume an
old `RecordVideo` callback has identical timing semantics.

For a downstream project, keep the default `play.py` and `train.py` control
flow aligned with InstinctLab. Their only normal differences should be task
registration and any extra repository metadata recorded by the training
runner.

### 3. Port visualizer configuration

- Replace `env_cfg.viewer` usage with
  `env_cfg.sim.default_visualizer_cfg` or `env_cfg.sim.visualizer_cfgs`.
- Configure the selected visualizer's camera, origin, and resolution there.
- For custom Newton visualization, use `NewtonGLVisualizerCfg` and a concrete
  visualizer `class_type`; beta2 factory overrides are no longer the extension
  point.

### 4. Port changed sensor, path, and physics APIs

Apply only the rows that match code in the downstream project.

| Area | Required migration |
|---|---|
| Clone/path queries | Replace `iter_clone_plan_matches` with `isaaclab.cloner.query.iter_sources`. Recheck every path expression: matching is whole-path and descendant-aware. |
| Contact sensors | Prefer upstream `ContactSensorCfg` over copied private implementations. Revalidate body ordering and force semantics; use `net_normal_forces_w` when thresholds mean normal force. |
| Custom Newton ray/camera code | Preserve existing multi-mesh behavior with the explicit legacy ray-caster classes, then revalidate custom kernels and lifecycle hooks. |
| Newton solver config | Remove deprecated `MJWarpSolverCfg.ls_parallel`. |
| Actuators | Replace private joint-parameter parsers with `isaaclab.actuators.resolve_joint_parameter`. Use `actuator_effort_limit` / `actuator_velocity_limit` for actuator-model limits and `joint_effort_limit` / `joint_velocity_limit` for solver limits. |

Review every use of custom or private Isaac Lab internals rather than applying
a blind rename. In particular, path regular expressions, contact-force
thresholds, and actuator limits can change behavior without causing an import
error.

### 5. Resolve changed defaults deliberately

- `PhysxCfg.enable_external_forces_every_iteration` now defaults to `True`.
- Contact-force buffers are backend-dependent: Newton total force includes
  friction, while PhysX normal-force behavior differs.
- Beta2 checkpoints may no longer be compatible if observations or actions
  changed.

Save the resolved simulator, visualizer, timing, and environment configuration
for each production task. This makes intentional behavior changes reviewable.

## Minimum validation

Run validation on the target Linux compute machine, not macOS.

1. Install the pinned environment and import Isaac Lab, InstinctLab, and the
   downstream package.
2. Resolve each production task configuration, then construct and reset it on
   every supported backend.
3. Run finite zero-action and fixed-action steps; check observation, action,
   reward, termination, and contact shapes and values.
4. Exercise custom sensors and visualizers through create, update, reset, and
   close, including partial-environment updates when supported.
5. Run one training update, checkpoint resume, play, and video export for each
   task family that supports them.
6. Compare representative beta2 and target results with fixed seeds, resolved
   configs, initial state, backend, and environment count. Review any changed
   contact, gait, actuator, or motion-reference signal.

## Keep the migration reviewable

- Do not mix unrelated changes with the API port.
- Commit metadata, launcher/video, sensor/physics, and task-config changes in
  small reviewable groups.
- Record full upstream and downstream SHAs with results.
- Keep the beta2 checkout and environment until rollback has been tested.
