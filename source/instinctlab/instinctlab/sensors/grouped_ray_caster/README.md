# Grouped ray casters

This package extends Isaac Lab's multi-mesh ray casters with per-environment target groups, dynamic target tracking,
near-hit rejection, and optional mesh IDs. It supports the PhysX and Newton backends without importing both backend
packages into the same sensor process.

## Why the implementation is split

PhysX and Newton expose similar ray-caster results, but their pose-tracking lifecycles are different:

- PhysX creates rigid-body views after the physics simulation view is initialized.
- Newton registers the sensor site during sensor construction, before cloning and model finalization, and later reads
  that site directly from Newton model/state arrays.

Ray-cast *targets* are the same on both backends: Isaac Lab binds every tracked target through the scene-data
provider, so InstinctLab registers no target sites of its own.

That difference cannot be hidden safely by importing both implementations and selecting a branch during an update.
Doing so loads an unused physics integration, makes optional backend installations impossible, and allows backend-only
types to leak into shared code.

The package therefore has one shared layer and one implementation module per backend:

| Module | Responsibility |
| --- | --- |
| `grouped_ray_caster_cfg.py` | Backend-neutral ray-caster configuration and public `class_type`. |
| `grouped_ray_caster_camera_cfg.py` | Backend-neutral camera configuration and public `class_type`. |
| `grouped_ray_caster.py` | Public lazy dispatcher and shared non-camera Warp update. |
| `grouped_ray_caster_camera.py` | Public lazy dispatcher and shared camera Warp update. |
| `grouped_kernels.py` | Shared near-hit-rejecting ray-cast kernel over the per-environment mesh tables. |
| `flat_target_prim_registry.py` | Shared target resolution: articulation-root targets and uniform per-environment prim counts. |
| `physx.py` | The two concrete PhysX sensors. |
| `newton.py` | The two concrete Newton sensors. |
| `instinctlab.utils.backend_dispatch` | Public helper that imports only the implementation selected by the active `SimulationContext`. |

This is the minimum useful split for import isolation. Combining `physx.py` and `newton.py` would import both backend
packages before a sensor could choose one. Splitting every concrete class into its own file would add files without
improving isolation, so each backend's caster and camera stay together.

The noisy ray-caster cameras in `instinctlab.sensors.noisy_camera` use the same dispatch rule and compose their noise
mixin with the concrete class from only the selected backend.

## Import invariant

Maintain these rules:

1. Public configuration, dispatch, shared registry, and shared kernel modules must not import `isaaclab_physx` or
   `isaaclab_newton`.
2. `physx.py` may import `isaaclab_physx`, but must not import `isaaclab_newton`.
3. `newton.py` may import `isaaclab_newton`, but must not import `isaaclab_physx`.
4. Task configurations must use the public config's default `class_type`; they must not substitute a backend-specific
   concrete class.
5. Backend differences belong in narrow pose-registration/access methods. Mesh bookkeeping and ray-cast kernels remain
   shared.
6. Python imports that cross package boundaries use absolute `instinctlab...` paths. Package `__init__.py` files may
   use `.` for local re-exports, but modules do not use `..` to reach outside their package. Configuration `class_type`
   values keep using `{DIR}` so Isaac Lab resolves them relative to the configuration module.

External users should construct `GroupedRayCasterCfg` or `GroupedRayCasterCameraCfg`. The concrete `Physx*` and
`Newton*` classes are implementation details and may move as Isaac Lab's backend APIs evolve.

## Grouping semantics

Grouping is what Isaac Lab's multi-mesh tables already provide: every configured target contributes one row of mesh
slots to each environment, padded to the widest row, so a ray of environment `i` only ever hits environment `i`'s
targets. InstinctLab adds two rules on top:

- **One mount per environment.** Each sensor instance is attached to exactly one body in every environment. Several
  instances of the same class may live in one scene; on Newton, instances on the same body share one site label.
- **Uniform target resolution.** A target expression must resolve the same number of prims in every environment it
  exists in; a ragged expression raises instead of silently padding the narrower environments. An environment without
  any match is accepted, so a static target may exist in only some prototypes.

A global target expression such as `/World/ground` resolves to one mesh that every environment shares. An
env-regex expression such as `{ENV_REGEX_NS}/Obstacle` resolves to one record per environment, with the pose taken
from that environment's prim; use it when the same geometry can sit at different places or be transformed per
environment. Warp meshes are cached per source prim and device, so identical geometry referenced by several targets or
sensors is loaded once.

An articulation-root prim is one tracked body, not a request to expand the whole robot: InstinctLab lists robot links
explicitly, and expanding the torso root would add every link a second time.

## Backend support policy

InstinctLab follows Isaac Lab's supported physics backends. PhysX support here is compatibility with the current Isaac
Lab release, not a promise to preserve PhysX after Isaac Lab removes or stops supporting it. If upstream Isaac Lab
retires PhysX, InstinctLab will follow that transition and remove the PhysX dispatcher entries and implementations in a
corresponding compatibility update. Users who still require PhysX at that point should pin matching Isaac Lab and
InstinctLab versions.

When changing this package, verify both runtime selection and import isolation: a Newton sensor must not load
`isaaclab_physx.sensors.ray_caster`, and a PhysX sensor must not load `isaaclab_newton.sensors.ray_caster`.
