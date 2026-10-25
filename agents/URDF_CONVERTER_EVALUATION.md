# Kit-free URDF to USD conversion: evaluation

Date: 2026-10-24. Isaac Lab `main` at `8c9b995cb`, Isaac Sim 6.2.0, Newton 1.8.0.dev0, image `instinctlab` (aorua).
Evidence: `~/Data/instinctlab_logs/upgrade_3main/urdf_converter_eval/` (scripts, logs, converted USDs, `robot_smoke.json`).

## Conclusion

Do not write a dedicated converter. Isaac Sim 6.2's URDF importer (`isaacsim.asset.importer.urdf` 3.11.12) is
`newton-physics/urdf-usd-converter` 0.3.2 plus Isaac Sim post-processing. The same pipeline ships Kit-free as
`isaacsim-asset-isolated==6.2.0.0` (Apache-2.0, `pypi.nvidia.com`), which Isaac Lab exposes as its `importers`
extra and calls from `UrdfConverter` when Kit is absent. For InstinctLab's G1 it produces a USD identical to the
Kit conversion. Adding it to the image removes the one-time Kit launch for URDF conversion on every backend.

## Findings

### `urdf-usd-converter` alone (0.3.4)

Tested on 7 public Unitree robots from `unitreerobotics/unitree_ros` (`15ebe53`): G1 29-DoF, G1 23-DoF,
G1 29-DoF with hands, H1, H1-2, Go2, B2. Also tested on InstinctLab's `g1_29dof_torsobase_simplified.urdf`
and `g1_29dof_torsobase_popsicle.urdf`.

- All converted in under 1 s with no warnings.
- Exact to float precision against the URDF: link mass, COM, inertia tensor, body poses at q = 0, joint
  axes and anchors, limits (radians to degrees), collider types and counts.
- Not usable as-is:
  - Adds a fixed `root_joint` to the world, so every robot has a fixed base.
  - No fixed-joint merging, so body count and order differ.
  - No joint drives.
  - Massless links are kept as bodies (B2 sensor links, G1 `LL_FOOT`/`LR_FOOT`).
  - Bodies are nested in the prim hierarchy. InstinctLab's `_flatten_urdf_geometry` already handles this.

The Isaac Sim post-processing fixes all of these (`fix_base`, `merge_fixed_joints`, drives, self-collision,
robot schema, physics variants).

### Standalone importer (`isaacsim-asset-isolated`), no Kit

1. InstinctLab's G1 (`G1_29DOF_TORSOBASE_POPSICLE_CFG`), run through InstinctLab's own path (Isaac Lab
   `UrdfConverter`, `_flatten_urdf_geometry` with capsules, `_check_urdf_inertia`): 0 differences from the
   Kit-converted asset cache across all 127 prims, applied schemas, attribute values and relationships.
2. The 7 public robots, imported with InstinctLab's settings (merge fixed joints, floating base) and loaded
   in Newton:

   | Check | Result |
   |---|---|
   | Mass, COM, inertia per body vs. a fixed-joint merge computed directly from the URDF | ≤ 5e-8 relative error on all robots |
   | Base | floating on all robots; no enabled world joint |
   | Drives | on every actuated joint |
   | Joint limits, effort, velocity, damping, armature vs. Newton's URDF parser | identical (≤ 5e-7) |
   | Free swing, fixed base, no contacts, 0.5 s | humanoids ≤ 4e-6 rad; B2 1.3e-4 rad; Go2 1e-3 rad (not investigated; a joint hitting its limit is the likely cause) |
   | Drop on ground with PD hold, 2 s | stable on all 7 |
   | DOF order vs. Newton's URDF parser | same, except on the two robots with hands (the loaders traverse the tree in a different order) |

## Pitfalls found

- Newton's own URDF parser (`add_urdf(collapse_fixed_joints=True)`) gets merged inertias wrong, for example
  G1 `torso_link` and the wrists. It also gives links without `<inertial>` mass from collider density, adding
  0.19 kg on Go2. Do not use it as a reference.
- `isaacsim-asset-isolated` pins `usd-exchange==2.3.0` and older `mujoco`/`mujoco-warp`. Install it with
  Isaac Lab's overrides; otherwise it downgrades MuJoCo and breaks Newton.
- The importer leaves the `Physics` variant set unselected. Isaac Lab selects `physics`
  (`PhysicsVariant.PHYSICS`); other consumers must do the same, or they see no bodies or joints.
- MuJoCo-Warp's default `njmax`/`nconmax` overflow on the hand models and corrupt the simulation. This is
  test-harness tuning, not a converter issue.

## Adoption

1. Install `isaacsim-asset-isolated==6.2.0.0` in the image through Isaac Lab's `importers` extra, keeping
   Isaac Lab's `override-dependencies` (`usd-exchange==3.0.1`, `mujoco~=3.15.0`, `mujoco-warp~=3.15.0`,
   `newton-usd-schemas>=0.5.0`, the pinned Newton).
2. Then `asset_cache.ensure_asset_cache_kit_args` no longer needs to launch Kit. Snapshot tools and other
   direct env builders can convert in-process.
3. Verify by deleting the asset cache, converting without Kit, and diffing against a Kit conversion (the
   check in `kitless_convert.py`). Then rerun the layout audit.

This resolves decision 3 of `UPGRADE_OVPHYSX_PLAN.md`.
