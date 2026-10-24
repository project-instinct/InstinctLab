# Upgrade log: Isaac Lab 3.0.0 EA to Isaac Lab 3.0.0 GA (`main`)

Findings and decisions made while porting InstinctLab from Isaac Lab EA `release/3.0.0` (Isaac Sim 6.0.1) to Isaac Lab `main` (Isaac Sim 6.2). The plan is `agents/UPGRADE_TO_3main_LMUOpus.md`. Evidence files are under `~/Data/instinctlab_logs/upgrade_3main/`; the manifest with every tolerance, accepted difference and sync point is `~/Data/instinctlab_logs/upgrade_3main/manifest.yaml`.

## Summary of differences against EA

Network I/O layout (observation names, order, shapes, dtypes, joint/body lists, ray patterns, ONNX signatures) is unchanged on all 14 rows (7 task families × PhysX/Newton). The DWP checkpoint re-exports to ONNX bit-identically on both backends.

Value-level differences, all accepted (manifest ids):

| Id | Change | Effect on InstinctLab |
|---|---|---|
| D6 | URDF importer 3.x writes correct link inertia (EA rotated it) | G1 dynamics change; largest on dynamic motions. Details below |
| D7 | `BaseRayCaster.reset` no longer draws zero-range drifts | Later CUDA random draws shift; same distributions |
| D8 | PhysX 6.2 contact response at deep self-contact | HOI PhysX rollouts diverge after the first step |
| H1 | PhysX contact `net_forces_w` includes friction | Contact rewards/terminations see a larger force; no observation term |
| H2 | float64 air/contact time | Removes a tiny float32 drift (locomotion air-time reward) |
| H3 | PhysX ray casters refresh kinematics first | No measured change |
| H4 | Height-field `to_discrete_units` | No change for InstinctLab terrains |
| H5 | Default ground-plane look | Visual only |
| H6 | `string_to_callable` raises the original error | Error messages only |
| H7 | `update_from_dict` no longer type-checks scalars | Mistyped Hydra overrides are no longer rejected |
| H8 | `use_newton_actuators` defaults to True | Numerically identical on PhysX |
| D1 | PhysX and Newton order bodies/joints differently (EA too) | Each backend is compared with its own baseline |
| D2-D4 | PhysX filter-pair log message; normalizer `until=2**62`; ONNX external data file | None (same on EA) |
| D5 | Managers copy omitted keyword defaults into term params | Layout dump only |

Open: Q1, a Newton height-scan difference of up to 1.6 mm on 10 of 748 rays at reset (perceptive shadowing; 7.6 mm on HOI), inside the Newton ray-cast path. Meshes, rays and sensor pose are identical.

Restored to EA behavior in InstinctLab: terrain origins and the parkour height-field scales (Phase 3), the torque-monitor prim path, and non-deprecated schema/material cfgs with identical values.

Not validated: distributed training; hosts with NVIDIA drivers older than 580 (the image's PyTorch is built for CUDA 13.0).

Training at scale (marsbrain, RTX 4090 D 24 GB, 20 iterations then resume +5):

| Task | PhysX | Newton |
|---|---|---|
| Parkour | 4096 envs OK | 4096 envs OK |
| Perceptive shadowing | 4096 envs OK with `env.sim.physics.gpu_max_rigid_contact_count=67108864` (the cfg's `2**27` leaves no memory for the depth encoder) | 4096 envs OK |
| Perceptive HOI | 2048 envs OK; resume needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 4096 envs stalls after the first iteration; 1024 envs OK |

Open: Q2, HOI exceeds its cfg capacities at scale (Newton triangle pairs about 3.3M against a 2.5M cap at 4096 envs; PhysX collision stack overflow at 2048 envs). EA never ran HOI at that scale, so there is no baseline.

## 2026-10-23: URDF link inertia, EA importer bug fixed on main

### Finding

The EA URDF importer (`isaacsim.asset.importer.urdf` 2.x, Isaac Sim 6.0.1) writes each link's inertia rotated the wrong way. Main's importer (3.x, Isaac Sim 6.2) writes it correctly. Every G1 policy trained on EA saw slightly wrong link inertias.

### Mechanism

A URDF `<inertial>` block gives a full 3×3 tensor `I`. USD stores it as principal moments `D` (`physics:diagonalInertia`) and a rotation `R` (`physics:principalAxes`), and the simulator rebuilds `I = R·D·Rᵀ`.

- EA writes the inverse rotation `Rᵀ`: the quaternion's vector part has the wrong sign. The simulator then uses `Rᵀ·D·R`. Mass, center of mass and principal moments are correct; the principal axes are not.
- Main writes `R`, and also writes the full tensor as `newton:inertia`.

Example, `torso_link` `principalAxes` (w, x, y, z):

| Importer | Quaternion |
|---|---|
| EA | (0.7210, −0.00089, 0.6929, −0.00098) |
| main | (0.7210, 0.00089, −0.6929, 0.00098) |

### Evidence

- `main/check_urdf_inertia.py` rebuilds `R·D·Rᵀ` from each converted USD and compares it with the URDF tensor (24 of 30 bodies; the 6 compound bodies that absorb fixed-joint children are compared with `newton:inertia` instead). Both URDFs the tasks use, `assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf` and `tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf`, give the same result:

  | USD | Max relative error vs URDF | Bodies matching only with `Rᵀ` |
  |---|---|---|
  | EA asset cache (Isaac Sim 6.0.1) | 5.9e-1 | 23 of 24 |
  | main asset cache | 1.8e-7 | 0 |

- `left_elbow_link`: URDF `ixy = 6.53e-5`. EA's PhysX view reports `ixy = −1.34e-5`; main's reports 6.53e-5 (max error 7e-11).
- Simulated inertia on main matches the USD tensor on both backends: PhysX `get_inertias()` 5.8e-7, Newton `model.body_inertia` 1.8e-7 (max relative error over 30 bodies, `probe_randvals.py`).
- Masses, COM positions, default joint positions and friction from the startup randomization are identical between EA and main (`probe_norand/randvals`).
- With EA's converted USD swapped into main (`probe_eausd.py`), the §6 snapshots of `Instinct-BeyondMimic-Plane-G1-v0` and `Instinct-Shadowing-WholeBody-Plane-G1-v0` match the EA baseline exactly: every stepped-term RMS ratio is 0.0000 and terminated/time_out agreement is 1.000. All PhysX drift on these rows therefore comes from the inertia, not from the port.
- Size of the effect with main's correct inertia, 32 steps, 4 envs: BeyondMimic (LAFAN sprint clip) policy `joint_vel` RMS ratio 0.31, `base_ang_vel` 0.17; WholeBody `joint_vel` 0.044; locomotion (flat, slow) 0.022.
- EA run twice gives bit-identical snapshots (`ea_repeat_probe`), so the drift is not run-to-run noise.

### Decision

Use main's importer output, i.e. the correct inertia. No InstinctLab code reproduces the EA rotation, and none is needed for the correct conversion: the asset cache digest includes the Isaac Lab version, so main never reuses an EA-converted USD.

### Consequences

- Policies trained on EA run on a physically different G1 on main. Expect a behavior change for dynamic motions; fine-tune or retrain where it matters.
- Snapshot parity: robot rows are gated on code parity with EA's converted USD swapped in (exact match expected). Runs with main's USD are recorded, not gated, under manifest accepted difference D6.

## 2026-10-24: Kit-free URDF conversion

Following `agents/URDF_CONVERTER_EVALUATION.md`, the InstinctLab image adds Isaac Lab's `importers` extra (`isaacsim-asset-isolated==6.2.0.0`) through `uv sync --frozen` on Isaac Lab's lockfile, so its overrides keep torch, Newton, MuJoCo and `usd-exchange` at the image's versions (checked with `uv pip list`; `uv pip check` lists only the overridden pins).

The evaluation's step 2 ("`ensure_asset_cache_kit_args` no longer needs to launch Kit") holds only for plain `python`:

- Under `isaaclab -p`, Isaac Sim's `python.sh` puts Isaac Sim's regular `isaacsim` package first on `PYTHONPATH`, which hides the wheel's `isaacsim.asset` namespace portion (`isaaclab.utils.version.standalone_importers_available()` is `False`). A cold cache there still converts through a windowless Kit app.
- Under plain `python` (kitless backends), `ensure_asset_cache_kit_args` keeps the launch kitless. It also imports the importer's PhysX schema fallback before the launch: registered later, inside the running simulation, the importer's `ApplyAPI("PhysxJointAPI")` fails.
- The snapshot tool (`network_io_snapshot.py`) now calls the same function.

Verification (`main/c11_kitless_probe`, `main/c11_layout`):

- Cold cache, `python scripts/instinct_rl/train.py --task Instinct-Locomotion-Flat-G1-Newton-v0 --viz none`: no Kit launched, one iteration trained, cache written.
- That kitless USD vs. the Kit-converted cache: 127 prims, 0 differences (`urdf_converter_eval/kitless_convert.py` diff).
- Layout audit on the rebuilt image: 13 of 14 rows have an identical layout to the EA baseline; value differences are the recorded D6/D8/Q1 ones. Parkour Newton runs out of host memory on aorua and was checked on marsbrain.

## 2026-10-25: URDF conversion in a Kit-free child process

Requirement: URDF conversion must not depend on Kit; every other component keeps its dependencies (PhysX and the Kit viewer still need Kit).

- `spawn_from_urdf` converts a cold cache through `urdf_conversion.convert_urdf`: a child process of the virtual environment's Python, with the variables Isaac Sim's `python.sh` exports for Kit removed (`PYTHONPATH`, `LD_LIBRARY_PATH`, `LD_PRELOAD`, `CARB_APP_PATH`, `ISAAC_PATH`, `EXP_PATH`), runs the standalone importer, `_flatten_urdf_geometry` and `_check_urdf_inertia`. The parent publishes the result to the asset cache and spawns it, inside Kit or not.
- This replaces the 2026-10-24 approach: `ensure_asset_cache_kit_args` (forced a windowless Kit for cold caches under `isaaclab -p`) and its calls in the train/play scripts and the snapshot tool are removed, together with the early PhysX schema registration.
- Verification (`main/c12_subprocess`): cold-cache training of `Instinct-Locomotion-Flat-G1-v0` under `isaaclab -p` (PhysX, Kit for the simulation only), `-Newton-v0` under `isaaclab -p` (no Kit) and under `python` (no Kit) all train one iteration; each converted USD matches the Kit-converted cache (127 prims, 0 differences). `test_urdf_conversion.py` converts a two-link URDF with an `isaacsim` package shadowing PYTHONPATH as `python.sh` does; without the environment stripping that test fails with `No module named 'isaacsim.asset'`.
