# Migrating Isaac Lab 2.3.2 to 3.0.0-beta2 with PhysX and Newton task configs

Updated: 2026-08-17

## Purpose

This guide captures the verified InstinctLab migration from Isaac Lab 2.3.2 / Isaac Sim 5.1 to
Isaac Lab `3.0.0-beta2` / Isaac Sim `6.0.1`. It covers PhysX migration gates and a Newton
task-config recipe. The exact target is pinned below; do not use a floating branch.

| Component | Verified target |
|---|---|
| Python | `3.12.13` |
| Isaac Sim | `6.0.1` |
| Isaac Lab commit | `6a7acb0320a0bdc15b13e44e83b575e00797faf4` |
| `isaaclab` | `6.1.17` |
| `isaaclab-physx` | `1.1.3` |
| `isaaclab-newton` | `0.13.6` |
| `newton[sim]` | `1.2.1` |
| Torch / Vision / Audio | `2.10.0` |
| Warp | `1.13.0` |
| Gymnasium | `1.2.1` |
| NumPy | `>=2` |

## Migration in one page

Five changes interact:

1. Python 3.12 / Isaac Sim 6.0.1 runtime.
2. Backend-neutral APIs separate from `isaaclab_physx` and `isaaclab_newton`.
3. In-memory quaternions change from WXYZ to XYZW.
4. Asset and sensor data becomes Warp-backed `ProxyArray`.
5. URDF importer, lifecycle, USD hierarchy, and generated assets change.

| Phase | Gate |
|---|---|
| 0 Freeze baseline | Reproducible reference and exact repository/runtime manifest |
| 1 Build target runtime | Pinned imports pass |
| 2 Port imports/config/lifecycle | Configs resolve without simulation |
| 3 Convert quaternion semantics | Focused math/file tests pass |
| 4 Port `ProxyArray` writes | Torch/Warp ownership and partial writes verified |
| 5 Port sensors/PhysX views | Partial/reset/recreate and numerical gates pass |
| 6 Rebuild assets | Hierarchy, physics, collision geometry pass |
| 7 Integrate tasks/RL | Reset, rollout, train, resume, video, export pass |
| 8 Compare/release | Numerical, learning, performance, rollback accepted |

Commit phases separately. Do not combine quaternion, asset, sensor, and launcher changes into one
unreviewable patch.

## Rules that prevent false success

- Keep the 2.3.2 checkout/environment read-only.
- Record full SHAs and dirty worktree state for every editable repository.
- Preserve user-owned paths, datasets, assets, logs, and checkpoints.
- Do not rely on deprecated aliases merely because imports succeed.
- Do not bulk-replace quaternion literals or `.data.*` expressions.
- Do not require bitwise physics equality across Isaac Sim generations.
- Run runtime verification on the real Linux/CUDA host.
- Treat registration, construction, reset, rollout, training, video, and export as separate gates.
## Phase 0: freeze a reproducible 2.3.2 baseline

Capture repository state:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
git diff --name-status
```
Record runtime versions, GPU/driver, SHAs, resolved env/agent configs, dataset/asset checksums,
task IDs, spaces, manager terms, seeds, timesteps, fixed-action rollout, observations, rewards,
terminations, contacts, sensor outputs, throughput, and host/device memory.
Define tolerances now for root pose, quaternion angle error, joint position/velocity, reward,
contact rate/force, ray hits/depth, and termination timing.
## Phase 1: create and pin the target runtime

```bash
git fetch origin release/3.0.0-beta2
git switch --detach 6a7acb0320a0bdc15b13e44e83b575e00797faf4
git rev-parse HEAD
```
Create a Python 3.12 environment and install the pinned Isaac Sim and Isaac Lab packages. Install
the downstream project and RL library as editable packages only after selecting revisions.
PhysX extension dependency:

```toml
[dependencies]
"isaaclab" = {}
"isaaclab_physx" = {}
```
Newton requires the Newton packages and Newton-compatible task validation. `isaaclab_visualizers`
is a Python package, not an Omniverse extension dependency.
Gate: a fresh process imports core Isaac Lab, both backends, project package, RL package, and task
registration without missing-extension or dependency errors.
## Phase 2: port imports, configuration, and lifecycle

### Backend-neutral vs backend-specific

Keep factory-backed assets/sensors on the `isaaclab` surface:

```python
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, FrameTransformer, RayCaster
```

Import PhysX implementations only when selecting or subclassing PhysX:

```python
from isaaclab_physx.physics import PhysxCfg, PhysxManager
from isaaclab_physx.sensors.ray_caster import MultiMeshRayCaster
```
### Common imports

| 2.3.2 | 3.0.0-beta2 |
|---|---|
| `omni.physics.tensors.impl.api` | `omni.physics.tensors.api` |
| `isaacsim.core.simulation_manager.SimulationManager` | `isaaclab_physx.physics.PhysxManager` |
| `XformPrimView` | `isaaclab.sim.views.FrameView` |
| `root_physx_view` | `root_view` or public asset data |
| old configclass locations | `isaaclab.utils.configclass` |

### Explicit PhysX config

`SimulationCfg.physx` is gone. Assign a `PhysxCfg` to `SimulationCfg.physics`:

```python
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg

sim = SimulationCfg(dt=1.0 / 120.0, physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
```
### Schema split

| Legacy alias | Common base | PhysX |
|---|---|---|
| `RigidBodyPropertiesCfg` | `RigidBodyBaseCfg` | `PhysxRigidBodyPropertiesCfg` |
| `JointDrivePropertiesCfg` | `JointDriveBaseCfg` | `PhysxJointDrivePropertiesCfg` |
| `CollisionPropertiesCfg` | `CollisionBaseCfg` | `PhysxCollisionPropertiesCfg` |
| `ArticulationRootPropertiesCfg` | `ArticulationRootBaseCfg` | `PhysxArticulationRootPropertiesCfg` |
| `RigidBodyMaterialCfg` | `RigidBodyMaterialBaseCfg` | `PhysxRigidBodyMaterialCfg` |

Migrate `max_velocity` to `max_joint_velocity` and `max_effort` to `max_force`.
### Simulator lifecycle

```python
from isaaclab_tasks.utils import add_launcher_args, launch_simulation

add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

with launch_simulation(env_cfg, args_cli):
    env = gym.make(task_id, cfg=env_cfg)
```
Resolve pure config, checkpoint paths, and CLI values before launch. Import USD, PhysX, simulator
implementations, and wrappers only after launch when possible. Task registration/config import must
not initialize Kit or load `pxr`, `omni`, or `carb`.
## Phase 3: migrate quaternion semantics

Isaac Lab 3.0 uses XYZW in memory:

| Meaning | 2.3.2 | 3.0.0-beta2 |
|---|---|---|
| Order | `(w, x, y, z)` | `(x, y, z, w)` |
| Identity | `(1, 0, 0, 0)` | `(0, 0, 0, 1)` |

Audit asset/sensor offsets, poses, goals, direct physics-view reads, quaternion buffers,
custom Torch/Warp kernels, observations, rewards, interpolation, symmetry augmentation, datasets,
caches, checkpoints, ONNX inputs, and metadata.

The suffix `_w` means world frame, not WXYZ.

Use the upstream finder in report mode:

```bash
python /path/to/IsaacLab/scripts/tools/find_quaternions.py --path source --base <sha>
python /path/to/IsaacLab/scripts/tools/find_quaternions.py --path scripts --base <sha>
```

Rule:

> Every in-memory quaternion is XYZW. Non-XYZW may exist only at a named file/protocol boundary and
> is converted exactly once.

If policy observations contain quaternion components, old checkpoints may be schema-incompatible;
retrain unless there is a verified compatibility policy.

## Phase 4: port `ProxyArray` and writes

Asset/sensor properties often return `ProxyArray`:

```python
joint_pos_torch = robot.data.joint_pos.torch
joint_pos_warp = robot.data.joint_pos.warp
```

- Use `.torch` for Torch indexing, slicing, cloning, concatenation, and third-party Torch code.
- Pass `ProxyArray` directly to `wp.launch()` when its CUDA-array interface is sufficient.
- Use `.warp` only when a concrete Warp array/pointer/stride/type is required.
- Replace `wp.to_torch(proxy_array)` with `proxy_array.torch`.
- Never rebind simulator-owned `ProxyArray` fields; write into their backing buffers.

Unsuffixed write methods split into:

- `_index`: compact partial data matching selected environment/body IDs.
- `_mask`: full-size data plus boolean Warp masks.

Do not mix shapes. Prove unselected environments remain unchanged.

Other changed APIs:

- Full-state `Imu` is now `Pva`; new `Imu` exposes angular velocity and linear acceleration only.
- `body_incoming_joint_wrench_b` removed; add a `JointWrenchSensor`.
- Contact-sensor `pose_w`, `pos_w`, and `quat_w` deprecated; use a `FrameTransformer` or dedicated
  pose sensor.
- Deformable schemas and materials changed and need independent validation.

## Phase 5: port custom sensors and low-level PhysX code

### PhysX tensor views

```python
import omni.physics.tensors.api as physx
from isaaclab_physx.physics import PhysxManager

physics_sim_view = PhysxManager.get_physics_sim_view()
```

Prefer public asset/sensor data. Isolate unavoidable raw-view use in one small component responsible
for path resolution, device semantics, indexing, shape, and XYZW.

### Contact sensors and imported hierarchy

Importer 3.0 may produce nested rigid links. Audit contact-report APIs, full descendant prim paths,
environment/body ordering, filtered-contact matrices, and thresholds.

Do not copy InstinctLab's old hierarchical contact workaround blindly. Reproduce the downstream
asset issue first and remove the workaround when upstream behavior is sufficient.

### Ray casters and cameras

Review subclass base, child sensor prim/offset, `ray_alignment`, mesh-cache keys, environment masks,
Warp buffer ownership, dynamic/static transforms, and partial updates. For grouped dynamic meshes,
store an explicit world ID per ray and a checked world-to-entity membership table.

Camera output is `ProxyArray`-backed. Cross noise/history code into Torch explicitly and preserve
channel-vector dimensions such as `(N, T, H, W, C)`.

### Sensor lifecycle gate

Test full update, selected-env update, selected-env reset, create/step/close cycles, recreation,
debug visualization, cache isolation, and stable memory. Empty outdated masks must be cheap and
side-effect free.

## Phase 6: rebuild and validate converted assets

Do not reuse old generated USD without proof. Important importer changes:

- `{usd_dir}/{robot_name}/{robot_name}.usda` structured output.
- Nested rigid-body hierarchy.
- Removed old `usd_file_name` workflow.
- `replace_cylinders_with_capsules` is a deprecated importer no-op.
- Changed instanceability/editable-layer assumptions.

Regenerate into staging and compare articulation/fixed-base behavior, joint/body names and order,
limits/axes/inertias/masses/COM, drive gains/limits/armature/friction, collision shapes/materials/
filters/self-collision/contact reporting, default pose, rollout, and sensor prim paths.

Never let raw USD traversal order define policy semantics. Resolve policy-facing joint/body indices
by name.

If a required old importer behavior is gone, compose the standard importer with a narrow offline
postprocessor. Do not fork the importer or mutate runtime prims.

InstinctLab's G1 collision cylinders follow this pattern:

1. Inventory the exact 2.3.2 capsules.
2. Keep a manifest of expected prim paths and geometry.
3. Run standard 3.0 conversion in disposable staging.
4. Edit the defining USD layer offline.
5. Preserve schemas, transforms, physics attributes, materials, and extents.
6. Fail on path/count mismatch.
7. Hash inputs, config, tool revisions, and postprocessor version.
8. Publish an immutable digest-scoped asset.
9. Load only that final artifact and record its digest.

`UsdGeom.Capsule.height` is the cylindrical spine length, excluding hemispherical caps.

## Adding a Newton-backend environment config

Follow `source/instinctlab/instinctlab/tasks/locomotion/config/g1/flat_newton_env_cfg.py`. Keep the
shared MDP, commands, rewards, and events backend-neutral. The Newton config owns only
backend-specific simulation, sensor, material, and articulation schema.

```python
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_newton.sim.schemas import (
    NewtonArticulationRootPropertiesCfg,
    NewtonMaterialPropertiesCfg,
)
from isaaclab.utils.configclass import configclass

def _newton_sim_cfg() -> SimulationCfg:
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
    robot = G1_29DOF_TORSOBASE_POPSICLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.articulation_props = NewtonArticulationRootPropertiesCfg(self_collision_enabled=False)
    robot.actuators = beyondmimic_g1_29dof_actuators
    return robot

@configclass
class G1FlatNewtonSceneCfg(G1FlatSceneCfg):
    robot = _newton_robot_cfg()
    contact_forces = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    def __post_init__(self):
        self.terrain.physics_material = NewtonMaterialPropertiesCfg(
            static_friction=1.0, dynamic_friction=1.0
        )

@configclass
class G1FlatNewtonEnvCfg(G1FlatEnvCfg):
    scene: G1FlatNewtonSceneCfg = G1FlatNewtonSceneCfg(num_envs=4096, env_spacing=2.5)
    sim: SimulationCfg = _newton_sim_cfg()
```

Register separate train and Play IDs:

```python
gym.register(id="...-Newton-v0", ..., kwargs={"env_cfg_entry_point": G1FlatNewtonEnvCfg, ...})
gym.register(id="...-Newton-Play-v0", ..., kwargs={"env_cfg_entry_point": G1FlatNewtonEnvCfg_PLAY, ...})
```

Rules:

- Do not put Newton schemas or `NewtonCfg` in the shared base config.
- Do not hot-swap `sim.physics` by task-name string.
- Keep `UrdfFileCfg` as the asset interface. `asset_cache.py` derives and validates flattened USD
  under `$INSTINCTLAB_ASSET_CACHE/urdf/<asset>/<digest>/`.
- `train.py` and `play.py` call `ensure_asset_cache_kit_args(env_cfg, args_cli)` before
  `launch_simulation`. Cold caches transparently build with windowless Kit; warm `--viz none` runs
  stay kitless.
- `InstinctNewtonVisualizerCfg` is Play-only debug tooling; training does not need it.

Validate by importing config without Kit, instantiating train/Play peers, running one
create/reset/four-step/close smoke, running one `--viz none` training update, and comparing
action/observation/reward/termination schemas with the PhysX peer by name.

## Phase 7: port viewer, video, task registration, and RL integration

Owner split:

- `ViewerCfg` owns camera pose, origin type, and asset tracking.
- `SimulationCfg.visualizer_cfgs` selects visualization backends.
- Renderer produces frames.
- `RecordVideo` owns trigger, length, encoding, naming, and output.

Use `--viz none` for no viewer/offscreen video and `--viz kit` for the Kit window. Do not add
`SimulationCfg.default_visualizer_cfg`.

For video:

```python
if args_cli.video:
    args_cli.enable_cameras = True

with launch_simulation(env_cfg, args_cli):
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
```

Validate pixels, not only the MP4 container.

For every task, compare observation/action names/order/shapes/dtypes/spaces, reward/termination/
event/curriculum/command order, history reset, timeout/dataset-exhaustion semantics, reference
synchronization, and extras/log keys.

Keep custom RL runners inside `with launch_simulation(...)`. Log commit/status/diff for every
editable repository. Test resume by advancing checkpoint number and changing parameters. Export ONNX
at opset 18 and run `onnx.checker`.

## Phase 8: verification and release

Use a ladder:

1. Source compiles.
2. Package metadata resolves.
3. Registration imports without simulator modules.
4. Every config instantiates.
5. Asset hierarchy/collision audits pass.
6. Every task constructs and resets.
7. Four finite zero/fixed-action steps pass.
8. Partial update/reset leaves unselected envs unchanged.
9. Longer reset and sensor stress pass.
10. One optimizer update per task family passes with finite losses.
11. Resume, play, video, and export pass.
12. Distributed init/update/save passes.
13. Controlled 2.3.2-vs-3.0 comparisons pass.
14. Steady-state performance/memory comparisons pass.
15. Rollback to 2.3.2 succeeds from a clean process.

Align joints/bodies by name for comparisons. Document discontinuous threshold behavior. Make release
decisions from representative end-to-end workloads, not short profiles alone.

## Static migration audit

```bash
rg -n 'omni\.physics\.tensors\.impl\.api|obtain_world_pose_from_view|XformPrimView' source scripts
rg -n 'isaacsim\.core\.simulation_manager|root_physx_view' source scripts
rg -n 'SimulationCfg\.physx|\.sim\.physx\.' source scripts
rg -n '\.write_[A-Za-z_]+_to_sim\(' source scripts
rg -n 'convert_quat|quat_|rot=' source scripts
rg -n '\.data\.[A-Za-z_][A-Za-z_0-9]*' source scripts
rg -n 'default_visualizer_cfg|ViewerCfg|\.viewer\.|visualizer_cfgs' source scripts
rg -n 'RecordVideo|render_mode=.rgb_array.|enable_cameras' source scripts
rg -n -- '--headless' README.md source scripts
```

Interpret matches semantically. Run formatters and `pre-commit run --all-files` when configured.

## Common failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Wrong orientations | WXYZ data crossed into XYZW memory | identities, camera offsets, file boundaries |
| Torch warning/type failure | implicit `ProxyArray` path | add `.torch` |
| Warp pointer/stride failure | concrete Warp storage required | use `.warp` |
| Unselected envs change | `_index`/`_mask` shape mismatch | compact vs full-size data |
| Missing `omni.physics.tensors.impl` | private module removed | `omni.physics.tensors.api` |
| Wrong/no contacts | nested hierarchy/view grouping | full prim paths and env-major order |
| Duplicate/self ray hits | descendants traversed more than once | stop at rigid-body boundary |
| Gray but valid video | absolute camera or cameras enabled late | `ViewerCfg` tracking/order |
| Import abort before launch | eager Kit/USD imports | `.pyi` + `lazy_export()` |
| Recreate loop grows memory | global caches retain objects | close/reset invalidation |
| ONNX downgrade traceback | Torch 2.10 cannot lower requested opset | export opset 18 |
| Good rollout, bad learning | physics/contact or observation drift | fixed-seed curves and longer canary |

## Completion checklist

- [ ] Runtime and editable repos pinned by full SHA/version.
- [ ] 2.3.2 reference remains read-only.
- [ ] No removed private imports or `SimulationCfg.physx`.
- [ ] Schemas are intentionally common or backend-specific.
- [ ] Internal quaternions are XYZW; other conventions are named boundaries.
- [ ] Checkpoint/observation compatibility decision recorded.
- [ ] `ProxyArray` crosses to Torch/Warp explicitly.
- [ ] All writes verify `_index`/`_mask` semantics.
- [ ] Policy-facing joint/body order is name-defined.
- [ ] Generated assets record inputs, config, tool revision, and digest.
- [ ] Custom sensor full/partial/reset/recreate/debug gates pass.
- [ ] Every task family passes reset, rollout, and finite training canary.
- [ ] Play, resume, video pixel content, and opset-18 export pass.
- [ ] Numerical and performance comparisons accepted.
- [ ] Known deviations/untested gates recorded.
- [ ] Rollback exercised from clean processes.

## Evidence

The InstinctLab migration passed PhysX config instantiation, dataset-ready reset/rollout, one-update
training, sensor stress, video inspection, checkpoint resume, ONNX variants, a world-size-one
distributed canary, controlled 2.3.2/3.0 comparisons, and representative performance tests.
Remaining caveats at the time of writing were true multi-GPU execution and multi-device ray-cache
isolation on more than one GPU.
Project-specific evidence:
- `ISAACLAB_3_0_0_BETA2_UPGRADE_PLAN.md`
- `PROGRESS.md`
- `UPGRADE_REQUIREMENTS_and_INFO.md`

## Upstream references

- [Isaac Lab 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/migration/migrating_to_isaaclab_3-0.html)
- [Working with ProxyArray](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/how-to/proxy_array.html)
- [Pinned Isaac Lab branch](https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0-beta2)
- [Pinned target commit](https://github.com/isaac-sim/IsaacLab/commit/6a7acb0320a0bdc15b13e44e83b575e00797faf4)
