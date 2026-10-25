# Plan: OvPhysX backend support in InstinctLab

Status: implemented 2026-10-25 (commits `2da9821`, `9235e39`, `2e3ba3a` and the validation commit; results in §10). Applies to branch `migrate/isaaclab3-main` (Isaac Lab `main` at `8c9b995cb`, `ovphysx` wheel 0.6.4 in the image).

## 1. Is a backend switch only a `sim` switch today?

No, for both backends.

What follows `sim.physics` automatically at runtime:

- Isaac Lab assets and sensors are factories (`FactoryBase`): `ArticulationCfg`, `isaaclab.sensors.ContactSensorCfg`, the ray casters and the multi-mesh ray casters pick `isaaclab_physx`, `isaaclab_newton` or `isaaclab_ov` from the active backend.
- InstinctLab's own backend components pick their implementation through `instinctlab.utils.backend_dispatch.create_backend_component`: grouped ray caster and camera, noisy cameras, volume points, motion reference.

What does not: every Newton task is a separate env cfg class and task id (`*-Newton-v0`) that overrides, besides `sim`:

| Override in the Newton cfg | Example |
|---|---|
| Physics cfg and solver capacities | `newton_sim_cfg(njmax=..., nconmax=...)`, `max_triangle_pairs` |
| Contact sensor cfg class | `isaaclab_newton.sensors.ContactSensorCfg` instead of `isaaclab.sensors.ContactSensorCfg` |
| Terrain material class | `NewtonMaterialPropertiesCfg` instead of `PhysxRigidBodyMaterialCfg` (locomotion) |
| Robot actuators | locomotion uses the BeyondMimic actuator set on Newton |
| Task tuning | perceptive shadowing sets the reset `position_offset` to `[0, 0, 0.05]` |
| Play visualizer | `InstinctNewtonVisualizerCfg` instead of the Kit viewer |

So switching backends means picking a different task id. Isaac Lab main has the mechanism that would make it a `sim`-level switch: `PresetCfg` fields selected with `physics=<name>` / `presets=<name>` on the command line (see `isaaclab_tasks/benchmark/render_benchmark/render_benchmark_env_cfg.py`), and `PhysxAutoCfg`, which resolves to Isaac Sim PhysX under Kit and to OvPhysX kitless. InstinctLab does not use presets yet.

## 2. Goal

One backend assignment mechanism for PhysX, Newton and OvPhysX: every backend-specific value of a task lives in a backend preset of its env cfg, and selecting the backend selects the preset. Two equivalent entry points:

```bash
# a task id per backend; the id only selects the preset
isaaclab -p scripts/instinct_rl/train.py --task Instinct-Parkour-Target-Amp-G1-OvPhysx-v0
# or the preset selector on the PhysX id
isaaclab -p scripts/instinct_rl/train.py --task Instinct-Parkour-Target-Amp-G1-v0 physics=ovphysx
```

- Task ids: `*-v0` (PhysX, default), `*-Newton-v0` and the new `*-OvPhysx-v0`, plus their `-Play-v0` variants. The Newton and OvPhysX ids register the same env cfg class as the PhysX id with the backend preset selected; they carry no other overrides.
- OvPhysX is PhysX without Kit, so the PhysX-side cfgs (base `ContactSensorCfg`, `Physx*Cfg` schema fragments, `PhysxRigidBodyMaterialCfg`, GPU buffer sizes) are expected to carry over. Phase 1 verifies this; any field that does not carry over becomes part of the preset.
- Network I/O layout per backend stays as it is today (D1: PhysX and Newton order bodies/joints differently).

Non-goal: distributed training.

## 3. Phase 0: runtime check

- Run an upstream task on OvPhysX in the container (`physics=ovphysx` or `presets=ovphysx` on a task that defines it) and confirm the wheel, kitless launch and GPU pipeline work on aorua (RTX 5070 Ti, driver 580).
- Record what is unavailable kitless: Kit viewer, Kit video capture, RTX cameras. InstinctLab uses ray-caster cameras only, so observations are unaffected.
- Confirm the cold-cache path on OvPhysX: the URDF converts in the Kit-free child process (§7 decision 3) and the run stays kitless.

Gate: an upstream task trains one iteration on OvPhysX; an InstinctLab task converts its URDF on a cold cache and then runs kitless on OvPhysX.

## 4. Phase 1: dispatch and components

- `utils/backend_dispatch.py`: accept `"ovphysx"`; components without an OvPhysX implementation raise `NotImplementedError` naming the component (no silent mapping to PhysX).
- Add an OvPhysX implementation next to each PhysX one, on `isaaclab_ov` APIs (`isaaclab_ov.assets`, `isaaclab_ov.sensors.ray_caster`, the OvPhysX physics manager):

| Component | PhysX implementation today | OvPhysX work |
|---|---|---|
| Grouped ray caster, grouped camera | `sensors/grouped_ray_caster/physx.py` (`_PhysXRayCasterMixin`) | same classes on `isaaclab_ov`'s ray-caster pose mixin |
| Noisy cameras | `sensors/noisy_camera/physx_ray_caster_cameras.py` | follows the grouped/multi-mesh camera classes |
| Motion reference | `motion_reference/physx.py` (PhysX tensor views) | OvPhysX articulation view |
| Volume points | `sensors/volume_points/physx.py` (physics sim view) | OvPhysX body-pose access |
| Torque monitor | `monitors/monitor_manager.py` (`get_physics_sim_view`) | OvPhysX view or explicit rejection if unused |

- Unit tests per component with fake managers, as for Newton (`test_volume_points_newton.py`, `test_backend_dispatch.py`).

Gate: each component constructs, resets (`slice(None)`, `slice(1,4,2)`, int32, int64) and updates on OvPhysX; two instances on different links stay independent; pytest passes.

## 5. Phase 2: backend presets in the task cfgs

- One backend preset per family (locomotion, whole-body shadowing, BeyondMimic, perceptive shadowing, perceptive VAE, perceptive HOI, parkour) with `physx` (default), `newton` and `ovphysx` entries. A preset entry owns every backend-specific value; today these are spread over the Newton cfg classes:

| Value | `physx` | `newton` | `ovphysx` |
|---|---|---|---|
| `sim.physics` | current `PhysxCfg` (GPU buffers) | current `newton_sim_cfg(...)` (`njmax`, `nconmax`, triangle pairs) | `OvPhysxCfg` with the PhysX buffer values |
| contact sensor cfg | `isaaclab.sensors.ContactSensorCfg` | `isaaclab_newton.sensors.ContactSensorCfg` (check whether the base factory cfg suffices) | base `ContactSensorCfg` |
| terrain material | `PhysxRigidBodyMaterialCfg` | `NewtonMaterialPropertiesCfg` (locomotion) | `PhysxRigidBodyMaterialCfg` |
| robot actuators | task default | BeyondMimic set (locomotion) | task default |
| task tuning | - | reset `position_offset = [0, 0, 0.05]` (perceptive shadowing) | - |
| Play viewer | Kit | `InstinctNewtonGLVisualizer` (follow mode) | `viser` |

- Preset scope: Isaac Lab presets replace cfg sections, so values outside `sim` (contact sensor, material, actuators, event params) need a preset on their own section or a small per-backend hook in the env cfg's `__post_init__`. Pick the form after reading `isaaclab_tasks/utils/presets.py` and `hydra.py` at the synced SHA; it must also work from the task id registration, not only from the CLI.
- Registration: `*-Newton-v0` and `*-OvPhysx-v0` register the base env cfg with the preset selected; delete the per-backend env cfg classes once their values live in the presets.
- Regression gate for the refactor: the PhysX and Newton snapshots of every row must match `main/c8_audit` exactly (layout and values), so the move to presets changes no behavior.
- CLI: verify `physics=` / `presets=` in `utils/instinct_rl_cli.py` (already parsed as preset selectors) and that `--task *-OvPhysx-v0 physics=newton` is rejected or documented (a task id and a selector naming different backends).
- Play viewer: OvPhysX Play cfgs default to `viser`; `--viz newton_gl|rerun|viser` overrides it (§7 decision 4).

Gate: every family builds on all three backends from both entry points with no other override; PhysX and Newton snapshots unchanged.

## 6. Phase 3: validation

There is no EA baseline for OvPhysX; the reference is main's own PhysX row of the same task.

- Layout: identical to the PhysX snapshot (`compare_snapshot.py` layout section).
- Rollout: reset frames exact; physics-derived terms within the PhysX tolerance (RMS ratio <= 0.05) or each difference recorded with evidence, as in the EA comparison.
- Contacts: in-contact agreement >= 0.90; check the friction-included `net_forces_w` semantics (manifest H1) on OvPhysX.
- Reset gate (`main/run_gate.sh`) on all 7 OvPhysX rows.
- Training: 128 envs x 1 iteration + resume on aorua for parkour, perceptive shadowing, HOI; one larger run on marsbrain.
- Checkpoints: DWP and Hiking play on OvPhysX; fall rate measured with `probe_dwp_falls/probe_play_falls.py` against the PhysX rows (DWP: 0 falls on PhysX).
- Performance: steps/s against PhysX at 4096 envs.

Gate: all rows pass or every difference is recorded in the manifest; no deprecation warnings from InstinctLab code.

## 7. Decisions

1. Task ids: separate `*-OvPhysx-v0` ids (decided). They select the backend preset and carry no other overrides.
2. Newton: moves to the same backend preset mechanism (decided); `*-Newton-v0` ids stay and only select the preset.
3. URDF conversion without Kit (decided, implemented): conversion always runs in a Kit-free child process with the standalone importer (`sim/spawners/from_files/urdf_conversion.py`), for every backend and launcher. See `URDF_CONVERTER_EVALUATION.md` and the upgrade log.
4. OvPhysX Play viewer: `viser` by default, switchable with `--viz` (decided).

## 8. Risks

- OvPhysX 0.6.x is new; API changes at the next Isaac Lab sync are likely.
- Kitless URDF conversion and the inertia guard (`_check_urdf_inertia`) must agree with the Kit importer output.
- HOI's large contact and collision-stack buffers (open item Q2) may behave differently on OvPhysX.
- Contact force semantics or solver defaults may differ from Isaac Sim PhysX in ways the cfg cannot restore; those become accepted differences.

## 9. Commits

1. Backend presets for PhysX and Newton, Newton ids re-registered, per-backend cfg classes removed (Phase 2 refactor; PhysX and Newton snapshots unchanged).
2. Backend dispatch accepts OvPhysX; OvPhysX components and tests (Phase 1).
3. OvPhysX preset entries and `*-OvPhysx-v0` ids (Phase 2).
4. Validation fixes, manifest entries, README (unsupported combinations updated) and the upgrade log (Phase 3).

## 10. Results

Logs under `~/Data/instinctlab_logs/upgrade_3main/ovphysx/` unless noted.

- Presets (commit 1): resolved cfgs of all 28 PhysX/Newton ids are unchanged (`cfg_before` vs `cfg_after`) except the factory contact sensor cfg on Newton, the explicit `PhysxCfg()` of the locomotion tasks (the launch default) and the perceptive Newton Play reset offset, which now matches Newton training. PhysX and Newton snapshots match the previous sweep on all 14 rows (`main/c13_presets`, parkour Newton on marsbrain). Each `*-Newton-v0`/`*-OvPhysx-v0` id gives the same cfg as `physics=` on the PhysX id; a second backend selector is rejected.
- Presets in practice: only `sim.physics`, the shadowing contact sensors' `force_threshold` (Newton: unset), the locomotion terrain material and actuators, the perceptive shadowing reset offset and the viewers differ per backend. OvPhysX takes the PhysX values except `sim.physics` (`OvPhysxCfg` with the PhysX GPU buffers) and the viewers (`viser`). The base `ContactSensorCfg` and `PhysxRigidBodyMaterialCfg` work on OvPhysX unchanged.
- Layout (`s1/layout_check.txt`): all 7 OvPhysX rows identical to main's PhysX rows up to backend class names and backend-order name lists. In every term the list the term reads is identical; the permuted lists are the unused all-joints/all-bodies defaults of its `SceneEntityCfg`. OvPhysX orders joints and bodies as `G1_29DOF_POLICY_JOINT_ORDER_V1`, unlike PhysX on main (manifest D9). Because reset randomization draws per backend joint, OvPhysX rollouts are not value-comparable with PhysX; the checkpoint plays below replace the value comparison.
- Reset gate (`s1/gate`): 7/7 PASS. Training: 128 envs x 1 iteration and resume for parkour, perceptive shadowing and HOI, kitless (`train/`).
- Checkpoints: DWP on `Instinct-Perceptive-Shadowing-G1-OvPhysx-Play-v0`, 32 envs, 10 min: 785 episodes, 0 falls (PhysX 0/934, Newton 27%; `probe_dwp_falls/ovphysx_main.jsonl`). Hiking (`--useonnx`, 1 env, 10 min): 30/30 episodes end in `time_out`, as on PhysX (29/29; `hiking/`).
- Throughput on marsbrain, 4096 envs (`marsbrain/summary.txt`): parkour 9.2k-12.6k steps/s on PhysX, 19.0k-21.6k on OvPhysX; perceptive shadowing 10.0k vs 11.4k (one sample each).
- Viewer: OvPhysX Play opens `viser` on port 8080 without Kit (`viser_play.log`). OvPhysX also runs kitless under `isaaclab -p`.
- Not covered: contact-force semantics (H1) on OvPhysX were not compared per body, since the body order differs; the `ovphysx` tensor-binding API that `OvPhysxView` uses emits a `DeprecationWarning` from Isaac Lab's own code.
- Joint order (commit `3b1422e`): the G1 cfgs set `joint_ordering = G1_29DOF_POLICY_JOINT_ORDER_V1` (OvPhysX's native order, so a no-op there); the motion reference follows the articulation cfg. Network I/O is unchanged. Snapshot layouts against EA differ only in the unused all-joints default lists and the motion reference's joint list (`main/c14_joint_order`, parkour Newton in `c14_joint_order_marsbrain`). Reset draws now follow the policy order on every backend: PhysX and Newton resets no longer match EA value for value, and OvPhysX resets now match main's PhysX resets exactly (`ovphysx/s2`; only stepped physics differs). OvPhysX snapshots are unchanged (`s2` vs `s1`). DWP on PhysX: 1 fall in 765 episodes (end of a dive roll; 0/934 before), on OvPhysX 0/778; Hiking on PhysX 29/29 time-outs.
