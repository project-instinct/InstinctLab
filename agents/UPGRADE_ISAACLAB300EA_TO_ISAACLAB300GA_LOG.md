# Upgrade log: Isaac Lab 3.0.0 EA to Isaac Lab 3.0.0 GA (`main`)

Findings and decisions made while porting InstinctLab from Isaac Lab EA `release/3.0.0` (Isaac Sim 6.0.1) to Isaac Lab `main` (Isaac Sim 6.2). The plan is `agents/UPGRADE_TO_3main_LMUOpus.md`. Evidence files are under `~/Data/instinctlab_logs/upgrade_3main/`.

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
