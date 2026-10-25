# InstinctLab Upgrade Plan: Isaac Lab `release/3.0.0` (EA) → Isaac Lab `main`

Author: LMU Opus agent. Date: 2026-10-22. Status: plan only; nothing in this document is implemented yet.

This plan replaces the earlier EA → `v3.0.0` GA plan (`agents/UPGRADE_PLAN.md`) for the next migration. That plan was not implemented, and the file is not in the checkout or in git history (checked 2026-10-22). Its decisions are **not** inherited. Every decision that applies here is restated in "Decisions".

An alternative plan that migrated in place in the Nutstore checkouts was reviewed and discarded on 2026-10-22. Its useful parts (rollback, risks, baseline facts, freeze gate, site lists) are merged into this plan. Working in place was rejected because switching the Nutstore `isaaclab` checkout to `main` removes the EA environment this plan uses as its reference (§14).

Facts are verified against the local trees at the SHAs in §5 unless marked *(changelog)*: those come from upstream `CHANGELOG.rst`/`changelog.d` text and are re-checked at the phase's sync point.

## 1. Goal and invariants

Move InstinctLab from the Isaac Lab 3.0 Early Access snapshot (`release/3.0.0` at `e180372f3`, `isaaclab 16.4.0`) to upstream Isaac Lab `main` (3.x line; at writing `8c9b995cb`, `isaaclab 41.0.1`).

The following invariants must hold. A change that breaks one of them is a blocker, not a migration note.

1. **Network input/output layout is exact** for every registered task family, as defined in §6:
   - observation group names, term names, term order, per-term shapes and dtypes
   - history length, stacking order and flattening
   - joint and body order, scales, clips and normalization
   - action dimension, joint order, scale and offset
   - ONNX file names and their input/output names and shapes
2. **Main InstinctLab features keep working** on PhysX and Newton:
   - `InstinctRlEnv` with `MultiRewardManager` and `MonitorManager`
   - motion reference: `AmassMotion`, `AistppMotion`, `EmberUcb`, `HeightScannedMotion`, `OmomoMotion` (HOI), `TerrainMotion`, and the `StayStillMotion` generator
   - grouped ray caster, noisy camera (noise pipeline, history, delayed skip), volume points
   - custom terrains: the 17 `Perlin*` height-field cfgs with `WallTerrainCfgMixin`, the 2 `PerlinMesh*` trimesh cfgs, `FiledTerrainGenerator`, InstinctLab `TerrainImporter`, virtual obstacles
   - asset cache (`sim/spawners/from_files/`)
   - train/resume/play/video/ONNX export, plus the task-local play scripts (`tasks/shadowing/play.py`, `tasks/parkour/scripts/play.py --useonnx` with keyboard control)
   - motion tools under `scripts/`, including `multi_play.py`, which launches `scripts/instinct_rl/play.py` as a subprocess
3. **Instinct-RL stays at 1.0.3, unchanged.** All Isaac Lab adaptation lives in `InstinctRlEnv` (`envs/manager_based_rl_env.py`) and the Instinct-RL vec-env wrapper (`utils/wrappers/instinct_rl/`).
4. **Value-level drift policy:** if an upstream change alters values that feed the network while keeping the layout, restore the EA value wherever a configuration exists. Changes that cannot be configured back (upstream bug fixes) are accepted and recorded in the manifest with affected tasks and checkpoint impact (§10).

GHSi and PGMT stay outside the migration (`stash@{0}: preserve GHSi and PGMT outside Isaac Lab 3.0 GA migration`). Do not restore them or count them.

## 2. Decisions (user, 2026-10-22)

| Topic | Decision |
|---|---|
| InstinctLab branch | Create `release/3.0.0` from `migrate/isaaclab3.0.0` (HEAD `7b61512`, 4 commits ahead of origin). All migration commits land there. It is merged into InstinctLab `main` later, in a separate step. |
| Uncommitted work | The dataset-default edits in 4 task cfgs (parkour, perceptive shadowing, perceptive VAE, perceptive HOI) become the first commit on `release/3.0.0`. Confirm the diff is intended first. |
| Isaac Lab ref | Track `upstream/main` HEAD. Sync only at defined points (§4). Every artifact records the exact SHA. |
| Location | New git clones under `~/workspace/instinctlab_ws/`, outside Nutstore. The Nutstore checkouts and the `instinctlab` conda env stay on EA for the baseline. |
| Runtime | Docker image built from the Isaac Lab clone (`isaac-lab-base`), with an InstinctLab image on top. |
| Logs | `logs -> ../../../Data/instinctlab_logs` in the migration clone. The container mounts `~/Data` and `~/Datasets`. |
| CLI | Port off `hydra_task_config` to main's `resolve_task_config` / `setup_preset_cli`. Adopt three rules: (a) `train.py` requires a non-empty `--run_name` whenever a Hydra-style argument is present; (b) no Hydra-derived postfix in log-directory names; (c) `play.py` precedence is task defaults → saved YAML → Hydra → dedicated CLI options. |
| HOI meshes | Remove InstinctLab `MeshFileCfg`/`spawn_from_mesh`, which shadow upstream names. Use upstream `isaaclab.sim.MeshFileCfg` with in-memory trimesh (`TrimeshObjectCfg`). Prim layout changes from `{object}/geometry/mesh` to `{object}/mesh`. |
| Validation | Host `aorua` (this machine; RTX 5070 Ti 16 GB). One-iteration train plus save/resume only for parkour, perceptive shadowing and perceptive HOI, on both backends, with `--num_envs 128`. Other families get config, smoke and playback checks only. Any other training run needs approval. |
| Instinct-RL | Unchanged (1.0.3). |
| `ovphysx` | Rejected for now with a clear error; documented as unsupported. |
| Grouped sensor mounts | One body per env per sensor instance (main's rule is kept). Several instances of the same class on different bodies must work; see Phase 4. |
| Container user | UID 1000 first; the root fallback is acceptable. |

Naming: Isaac Lab's EA branch is also called `release/3.0.0`. This plan always writes **InstinctLab `release/3.0.0`** for the migration branch and **Isaac Lab `release/3.0.0`** for the EA branch.

## 3. dev-workflow alignment

These were the points where the existing setup contradicted dev-workflow. The user resolved them as follows:

- **Build artifacts in auto-sync folders.** Resolved by using clones in `~/workspace/instinctlab_ws/`. The only writes into `~/NutstoreFiles` are git operations in the `instinctlab` checkout before the clone exists (§5.1 and commit 1 in §11). The EA baseline run writes only to `~/Data/instinctlab_logs`. Builds, Docker state and all migration commits after commit 1 live outside Nutstore.
- **Agent isolation.** Each agent gets its own clone, branch and container. Do not use `git worktree` for the editable-installed repos, because the import path can silently resolve to another checkout. Verify `__file__` paths in the container (§5.4).
- **Logs.** Use `~/Data/instinctlab_logs` through a relative symlink. Run names include the agent tag when agents run in parallel.
- **Transfer.** Never rsync the Nutstore tree. Clones come from git (local path or remote). Line endings, modes, symlinks and `uv.lock` must survive.
- **Leaving the project folder.** Working in `~/workspace/instinctlab_ws/` and `~/Data/instinctlab_logs` was approved through this plan's decisions. Any other path outside the project needs a new question.

Ask again before acting if a new conflict appears, for example Docker caches that need a path outside `~/workspace` or `~/Data`.

## 4. Isaac Lab `main` tracking policy

Facts at writing (local refs, fetched 2026-10-22):

- `upstream/main` is at `8c9b995cb`. It is 886 commits ahead of `v3.0.0`.
- It diverged from `release/3.0.0` at `7d61327c4` (2026-08-20).
- The isaaclab major version moved 28 → 41 in about four weeks, roughly three breaking majors per week.
- HEAD carries unreleased `changelog.d` fragments (`pr8-consumer-bindings`) that rewrite multi-mesh ray-caster internals.
- The old Nutstore local `main` is the archived 2.x line (`upstream/archive/main-2.x`). It is not used.

Rules:

1. In the clone, local `main` tracks `upstream/main`. Update only with `git pull --ff-only` at a **sync point**:
   - the start of each phase in §8
   - right before final acceptance (§9)

   Never sync in the middle of a phase.
2. At every sync point, run the following and append the result to "Sync log" (§15):
   - record `OLD..NEW`
   - `git diff OLD NEW -- 'source/*/docs/CHANGELOG.rst' 'source/*/changelog.d' pyproject.toml uv.lock docker/`
   - re-run the AST import scan from §8.1 against the new tree
3. If `pyproject.toml`, `uv.lock` or `docker/` changed, rebuild `isaac-lab-base` and the InstinctLab image. Source edits alone do not need a rebuild, because the sources are bind-mounted.
4. Update the `source/instinctlab/setup.py` pins to the versions at the synced SHA. They stay exact pins.
5. Final acceptance (§9) runs the **whole** matrix on one frozen SHA. That SHA, the image ID and the package versions go into `README.md`.

## 5. Workspace, environment and hygiene

State of the Nutstore checkouts at writing (verified 2026-10-22):

- `instinctlab`: branch `migrate/isaaclab3.0.0`, HEAD `7b61512`. Real content changes only in the 4 dataset cfgs. Mode-only changes on `docker/docker-{attach,compose,stop}.sh` and `tasks/shadowing/grid_search.sh`. Almost every tracked text file is CRLF in the working tree (sync damage; `git add` normalizes to LF). Untracked: both `agents/UPGRADE_TO_3main_*.md` plans. No `logs` entry yet.
- `isaaclab`: Isaac Lab `release/3.0.0` at `e180372f3` (`isaaclab 16.4.0`, assets 0.6.4, tasks 17.0.0). 7 files show as modified, with no content diff: 5 mode/CRLF (`.github/actions/{multi-gpu,run-tests}/*.py`, `docker/container.py`, `scripts/tools/find_quaternions.py`, `tools/run_install_ci.py`) and the two `uv.lock` files. None of them is imported at runtime. Leave them alone and record `git status` in the baseline.
- `instinct_rl`: clean at `574c8d5`.
- Conda env `instinctlab` imports all three Nutstore checkouts through editable installs.

### 5.1 Nutstore checkout (EA, untouched except for the branch)

1. In `~/NutstoreFiles/2Projects/isaacSim/instinctlab`, run the hygiene inspection:

   ```bash
   git diff --summary HEAD
   git diff --numstat HEAD | awk '$1!=0||$2!=0'
   git ls-files --eol | awk '$1=="i/lf" && $2!="w/lf" && $2!="w/-text" && $2!="w/none"'
   ```

   Current real content changes are only the 4 dataset cfgs. Mode-only changes (`docker/*.sh`, `grid_search.sh`) and CRLF-only files are sync damage.
2. Create the branch and commit the dataset edits as LF:

   ```bash
   git switch -c release/3.0.0 migrate/isaaclab3.0.0
   ```

   Then `git add` the 4 cfg files explicitly. Restore only mode-damaged paths with `git restore --source=HEAD -- <paths>`, and rewrite CRLF files with `git checkout-index --force -- <paths>`. No `git clean` and no `reset --hard`.
3. Land commit 1 (§11: `--useonnx` fix and snapshot helper) here, because the EA baseline (§6) runs from this checkout.
4. Do not touch the Nutstore `isaaclab` or `instinct_rl` checkouts. The EA conda env depends on them.

### 5.2 Clones

```text
~/workspace/instinctlab_ws/
├── isaaclab/        # git clone of isaac-sim/IsaacLab, branch main (tracks upstream/main)
├── instinct_rl/     # clone at 1.0.3 (current HEAD 574c8d5; the license-only change since 846008a is fine)
└── instinctlab/     # clone of the Nutstore repo, branch release/3.0.0
    └── logs -> ../../../Data/instinctlab_logs
```

- Clone `isaaclab` from `git@github.com:isaac-sim/IsaacLab.git` as `upstream`, and add the fork `origin` for pushes if needed.
- Clone `instinctlab` from the Nutstore repo path **after** commits 0 and 1 exist there (§5.1). Then set `origin` to the GitHub remote and push InstinctLab `release/3.0.0` only when the user asks. Later commits live only in the clone. Do not push them back into the Nutstore checkout while it is the EA reference.
- `logs` is a relative symlink. Add it to `.git/info/exclude` (local) rather than to the tracked `.gitignore`.
- Create `~/Data/instinctlab_logs` if it is missing.

### 5.3 Docker

- Build the base image from the clone: `./docker/container.py build base`.
  - Main's flow uses uv and `/opt/isaaclab-venv`, plus `uv run --no-sync`.
  - `isaaclab.sh` and the conda guides are removed on main (isaaclab 35.0.0).
  - Tag the image with the Isaac Lab SHA.
- `docker compose` 2.40 and the `nvidia` runtime are present on `aorua`. Pulling `nvcr.io/nvidia/isaac-sim:6.2.0` requires accepting the EULA.
- Rewrite InstinctLab `docker/Dockerfile`, `docker-compose.yaml` and `.env.base` against main's base image:
  - Install `instinct_rl` and `instinctlab` with `uv pip install -e` into `/opt/isaaclab-venv`.
  - Never run an exact `uv sync` inside the container. It removes the editable installs.
  - Drop the stale `isaaclab.sh` alias and the `ziwenz-instinctlab` `/public/MARS/...` service, or move it to an override file.
- Mounts follow dev-workflow, so the symlinks resolve:
  - the three clones at their **host paths** (`/home/leo/workspace/instinctlab_ws/...`), with `isaaclab/source` also over `/workspace/isaaclab/source` (the image's editable install)
  - `/home/leo/Data` and `/home/leo/Datasets` at host paths, read-only for `Datasets`
  - named volumes for the asset cache (`~/.cache/instinctlab`), the Kit/OV caches, and the uv and Warp caches
- Run as UID/GID 1000 with `HOME=/home/leo` so `~/Datasets/...` defaults resolve and outputs stay user-owned.
  - Fallback (approved by the user, 2026-10-22) if Isaac Sim refuses to run as non-root: run as root, additionally mount the same directories at `/root/Data` and `/root/Datasets`, and chown outputs afterwards.
  - Set `git config --global safe.directory` inside the container only, so `store_code_state` can read the repos.

### 5.4 Environment verification (each image build)

Inside the container, record:

- `isaaclab*.__file__`, `instinct_rl.__file__` and `instinctlab.__file__`. They must resolve into the bind-mounted clones, not wheels or copied sources.
- the SHAs of all three clones and the image ID
- `torch.__version__`, which must be `2.11.0`, and `torch.version.cuda`
- `warp` (1.17.0) and the `newton` version/commit. Main pins newton to git `71eeba16…` (1.8.0.dev0) through the uv lock.
- the output of `uv pip check`

Fail on any mismatch.

## 6. Network I/O contract (freeze before any code change)

Every policy input is a flat vector. The vec-env wrapper flattens each term (`flatten(start_dim=1)`) and concatenates terms in configclass order (`utils/wrappers/instinct_rl/vecenv_wrapper.py` `_flatten_obs_group`, `get_obs_segments`). Every group uses `concatenate_terms=False` (16 sites in the task cfgs). `get_obs_format()`/`get_obs_segments()` (`{group: {term: shape}}`) have three consumers, and all three must keep working:

- the external Instinct-RL runner
- the hard-coded teacher `obs_format` in `tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vae_cfg.py`
- `tasks/parkour/scripts/play.py --useonnx`, which slices `obs[:, :proprio]` and assumes depth is last Isaac Lab's per-term order is: func → modifiers → noise → clip → scale → history. The history is flattened oldest → newest, and the buffer is backfilled with the first observation after a reset. Joint order is `G1_29DOF_POLICY_JOINT_ORDER_V1` (`assets/unitree_g1.py`), applied with `set_cfg_joint_order` to:

- the action
- policy/critic joint observations
- shadowing joint-reference commands
- parkour AMP joint terms

There is one action term (`JointPositionActionCfg`, 29 dof, `use_default_offset=True`, `beyondmimic_action_scale`, 50 Hz).

The numbers below are derived from the cfgs. **The baseline freeze is authoritative**: if it disagrees, correct this table before Phase 1.

| Family (base / Play / Newton share the layout) | Policy input (derived; confirm in baseline) | Network-side contract |
|---|---|---|
| Locomotion flat | 96: ang_vel 3, gravity 3, cmd 3, joint_pos_rel 29, joint_vel 29, actions 29 | MLP; policy/critic normalizers; `actor.onnx` |
| Whole-body shadowing | 763: refs ×10 frames (joint_pos, joint_vel, pos 3, rot 6), then proprio | MoE 8 experts; critic adds link_pos/rot over 14 links |
| BeyondMimic | 160 (refs ×1 frame, adds base_lin_vel) | MLP |
| Perceptive shadowing / HOI | 1990, of which depth (1,18,32) from 27×48 ray-cast camera → normalize(0,2) → crop 2/2/2/2 → resize; proprio history 8 | Conv2d `depth_image` → 32-d latent appended last; actor input 1446; `0-depth_image.onnx` + `actor.onnx` + `policy_normalizer.npz` |
| Perceptive VAE | 3048, depth history 10 with skip 3 → (4,18,32) | Teacher `obs_format` is hard-coded in `instinct_rl_vae_cfg.py`; `vae_actor.onnx` with baked gather indices |
| Parkour target AMP | 5376: proprio history 8 (ang_vel ×0.25, joint_vel ×0.05), depth (8,18,32) **last** (delayed, skip 5, 36×64 → crop → blur → normalize) | Depth encoder → 128; actor 896; `--useonnx` loads `0-depth_encoder.onnx` + `actor.onnx`, slices `obs[:, :proprio]` and assumes depth is last; AMP groups feed the discriminator only |

Freeze result (2026-10-23, `ea_baseline_20261023_001046`): the table matches for all 14 rows. Critic dims: 99, 889, 286, 1667, 1667, 1990, 5400. In perceptive shadowing/HOI policy obs, `depth_image` sits after the four reference terms and before proprio; only the network appends its latent last. Overrides and accepted differences are in the manifest.

Freeze procedure (EA, Nutstore checkout, conda `instinctlab`):

1. Write a small package-level snapshot helper. It goes in a test utility module under `source/instinctlab`, never in `scripts/`. For every task in the matrix it dumps:
   - `get_obs_format()` per group (term → shape, in order)
   - dtypes
   - action dim and resolved joint names
   - resolved `SceneEntityCfg` joint/body names for every observation, command and action term
   - contact-sensor body list
   - camera/ray-caster resolution, pattern point count and order hash
   - noise-pipeline order
   - history settings
   - the resolved env/agent cfg YAML
2. Capture fixed-seed rollouts per task: seed 42, 4 envs, full reset, partial reset `[1, 3]`, 16 zero-action steps and 16 fixed-action steps. Record observations, rewards, terminations and contact outputs.
3. Run checkpoint playback and ONNX export on the two reference checkpoints:
   - Hiking `parkour_onboard_preview_stair` (ONNX only, via `tasks/parkour/scripts/play.py --useonnx`)
   - Deep Whole-body Parkour `20260121_085042_…/model_48000.pt` (via `play.py --exportonnx`)

   Record the ONNX input/output names and shapes.

   Prerequisite commit (lands before the baseline, on `release/3.0.0`): `--useonnx` must accept an ONNX-only run directory. Today it always resolves `model_*.pt`.
4. Store everything in `~/Data/instinctlab_logs/upgrade_3main/ea_baseline_<timestamp>/` with all three repo SHAs, git status and diffs.
5. Before the first main run, write the manifest. It holds the tolerances (rtol 1e-4, atol 1e-5 for deterministic non-contact values; contact metrics per task, set from the task configs) and the pre-declared accepted differences.

On main, the same helper must produce **identical** names, order, shapes, dtypes, joint/body lists and ONNX I/O signatures. There is no tolerance on layout.

Gate (before Phase 1): the freeze folder has every item above for all 14 matrix rows (§8.8) plus the two checkpoints, and the table above matches it. The EA env stays available, but every later comparison uses this frozen folder. Regenerate it only from the same three SHAs.

## 7. Known `main` changes that hit InstinctLab

Static scans were run against the local `upstream/main` (`8c9b995cb`), and the main-side code for each row was read at that SHA. Version numbers in parentheses are *(changelog)* attributions. Re-verify each item at the phase's sync point.

Rows marked **runtime-only** are not found by the Phase-1 AST import scan. They pass import and fail, or warn, only once a scene is built.

| Area | InstinctLab site | Change on main | Action |
|---|---|---|---|
| Hydra CLI | `scripts/instinct_rl/train.py`, `play.py` | `hydra_task_config` removed; `resolve_task_config`, `register_task`, `setup_preset_cli` in `isaaclab_tasks.utils` (`preset_cli` merged into `hydra`, tasks 28.0.0); `--physics/--renderer/--visualizer` pre-parsed | Phase 2 |
| Cloner | `sensors/grouped_ray_caster/flat_target_prim_registry.py` | `isaaclab.cloner.query.iter_sources` removed; `cloner/path.py` unchanged vs v3.0.0 | Phase 4 |
| Multi-mesh ray caster | same file + `grouped_ray_caster/*` | pr8: `_build_mesh_records(target_cfg, plan)` (no `dummy_mesh_id`, 4-tuples with body path); `_create_tracked_target_view`, `_mesh_views` removed; `_mesh_body_indices`, `_scene_data_provider` added; kernel renamed `copy_mesh_transforms_to_table_kernel`; `_initialize_impl` requires `_view_count == _num_envs`; static targets per-env on kitless backends (40.0.1) | Phase 4 |
| Spawners | `sim/spawners/from_files/from_files.py` | `_spawn_from_usd_file` → public `spawn_from_usd_file`; upstream `MeshFileCfg`/`spawn_from_mesh` exist | Phases 1, 6 |
| Stale import | `tasks/shadowing/mdp/curriculums.py:12`, `envs/mdp/curriculums/motion_reference.py:15` | `TYPE_CHECKING` import from `isaaclab.motion_reference...` never existed | Phase 1 (→ `instinctlab.motion_reference...`) |
| Physics managers: hard failures (**runtime-only**) | `sensors/volume_points/newton.py`: `NewtonManager._model` (:94), `_state_0` (:103), `_cl_site_index_map` (:74); `utils/backend_dispatch.py:18`: `sim.physics_manager.__name__` | `SimulationContext` owns one manager instance (`self.backend_manager`, `simulation_context.py:190`). The class-level forwarding metaclass (`physics_manager.py` `_PhysicsManagerMeta`) skips every `_`-prefixed name: `_model`/`_state_0` raise `AttributeError`. `_cl_site_index_map` resolves to the **class-level empty dict** (`newton_manager.py:564`), so volume points raise a misleading "site … was not registered" error instead of an attribute error. `sim.physics_manager` is a deprecated alias that returns the instance, which has no `__name__` | Phase 5 |
| Physics managers: deprecation warnings (**runtime-only**) | `motion_reference/newton.py` (7 calls: `get_model`, `get_state_0`, `get_physics_sim_view`), `sensors/grouped_ray_caster/newton.py:33,41` (`cl_register_site`), `tasks/utils/newton.py:96-99` (`get_model`, `get_state_0`), `monitors/monitor_manager.py:25-27` and `sensors/volume_points/physx.py:33` (`PhysxManager.get_physics_sim_view`), `sensors/volume_points/newton.py:71` (`cl_register_site`) | Public instance methods called on the class are forwarded to the active manager with a `DeprecationWarning`; they work but fail the "no deprecation warnings" gate | Phase 5 |
| Backend names | `utils/backend_dispatch.py` | `InteractiveScene.physics_backend = sim.backend_manager.backend_name` (`"physx"`, `"ovphysx"`, `"newton"`). Upstream itself tests `"physx" in self.physics_backend` (`interactive_scene.py:154`), which also matches `"ovphysx"` | Phase 5: compare exactly; never substring-match |
| Visualizers | 12 files. `KitVisualizerCfg` + `sim.default_visualizer_cfg` in 7 task cfgs (locomotion `flat_env_cfg.py`, whole-body `plane_shadowing_cfg.py`, `beyondmimic_plane_cfg.py`, perceptive `perceptive_shadowing_cfg.py`, `perceptive_vae_cfg.py`, perceptive HOI `perceptive_shadowing_cfg.py`, `g1_parkour_target_amp_cfg.py`); `tasks/shadowing/play.py:343-344` reads `default_visualizer_cfg.lookat/eye`; `scripts/amass_visualize.py:221` passes `visualizer_cfgs=KitVisualizerCfg(...)`; `tasks/utils/newton.py` `InstinctNewtonVisualizerCfg(NewtonGLVisualizerCfg).create_visualizer` (dead since GA), used by the parkour and perceptive Newton cfgs | `initialize(sim, *, cameras)`; `NewtonVisualizerBase`/`NewtonGLVisualizer`; `VisualizerCfg(backend=…)` (`visualizer_cfg.py:145`). `KitVisualizerCfg` and `NewtonGLVisualizerCfg` are deprecated combined cfgs. `default_visualizer_cfg` is "Deprecated in Isaac Lab 3.2; translated once into `visualizer_cfgs` before launch". Configured visualizers run unless `--viz none` (40.0.0) | Phase 5 |
| Sensor base | `sensors/volume_points/volume_points.py`, `sensors/noisy_camera/noisy_camera.py` (override `_initialize_impl`) | `SensorBase` timing buffers (`_create_timing_buffers`, float64 `_elapsed_since_update`); delayed `update_period` refresh fixed (36.0.0); camera images invalidated after pose writes | Phase 4 |
| Env reset | `envs/manager_based_rl_env.py:44` `_reset_idx(env_ids: Sequence[int])`; 72 sites: 59 `len(env_ids)` in 16 files, 9 `torch.tensor/as_tensor(env_ids`, 4 `env_ids.tolist()`/`meshgrid(env_ids` | `_reset_idx(env_ids: torch.Tensor \| slice)` (`manager_based_rl_env.py:382`); `reset(env_ids=slice(None), *, seed, options)` | Phase 3 |
| Manager internals | `managers/reward_manager.py`, `InstinctRlEnv.load_managers` / `setup_manager_visualizers` | Class-based term funcs accepted (35.0.0); startup events at end of `load_managers`; live-plot contract | Phase 3 |
| Height field | `terrains/height_field/hf_terrains.py` (decorated by upstream `height_field_to_mesh`, which unpacks `z_gen, origin`) | Generators must return `(hf, origin)` | Phase 3 |
| Terrain scales | `tasks/parkour/config/parkour_env_cfg.py` (`TerrainGeneratorCfg(horizontal_scale=0.05)`) | Sub-terrains no longer inherit generator scales unless their field is `None` (33.0.0); sub-terrain default 0.1 → parkour terrain silently coarsens | Phase 3: set sub-terrain scales to `None` in InstinctLab Perlin cfg defaults or per task, restoring EA values |
| Height-field discretization | all hf terrains | `to_discrete_units` fix changes geometry (38.0.0) | Not configurable → manifest |
| String resolution | `sensors/noisy_camera/noisy_camera.py` (`string_to_callable`) | Raises `ModuleNotFoundError`/`AttributeError` instead of `ValueError` (37.0.0) | Audit `except` clauses |
| Launch | `tasks/*/play.py` (`launch_simulation(env_cfg, args_cli)`), `tasks/parkour/tools/convert_urdf.py` (`AppLauncher`) | `headless`/`visualizer_intent` removed; lowercase `--visualizer`; `visualizer_explicit` removed | Phase 2 |
| Video | train/play `--video` | `VideoRecorderCfg.source` grammar; `apply_video_recording` inside `launch_simulation` | Phase 2 |
| Seeding | train/play | `configure_seed(torch_deterministic=False)` now disables determinism (34.0.0) | Phase 2: keep the EA effective setting explicitly |
| Cloning | `scripts/amass_visualize.py:228` (`replicate_physics=False`); `scene.env_origins` users: `tasks/locomotion/mdp/curriculums.py`, `tasks/parkour/mdp/terminations.py`, `tasks/parkour/mdp/curriculums.py` (`update_env_origins`), `envs/mdp/events/reset_to_fall.py`, `motion_reference/motion_reference_manager.py:511`, `terrains/terrain_importer.py:113` | `InteractiveSceneCfg.replicate_physics` is deprecated and ignored with a `DeprecationWarning` (`interactive_scene_cfg.py:104-110`); non-replicated path removed; env transforms not authored by scene (38/39.0.0) | Phase 2 (`amass_visualize.py`); Phase 7 audit (`env_origins` users) |
| Tooling | repo | `isaaclab.sh` removed; uv-only; Docker build context is repo root | §5.3 |

Unchanged on main relative to v3.0.0 (verified by diff): `utils/math.py` (quaternion math), `cloner/path.py`, `clone_plan.py`, `preserve_order` / joint-ordering code, `actuator_base.py`, `actuator_pd.py`. EA → v3.0.0 deltas in these files still apply. Example: `quat_apply` broadcasting and `joint_ordering`/`body_ordering` defaults. Audit those in Phase 7.

## 8. Phases

Order of work. Each step needs the previous step's gate:

1. §5.1 in the Nutstore checkout: hygiene, branch, commit 0 (dataset defaults), commit 1 (`--useonnx` + snapshot helper).
2. §6 EA baseline freeze in conda `instinctlab`, with its gate.
3. §5.2 clones (InstinctLab cloned from the Nutstore repo after step 1), then the §5.3 Docker build and §5.4 verification.
4. Phases 1–9 below, in the clones only.
5. §9 final acceptance.

Each phase starts with a sync point (§4) and ends with its gate. Commit progressively on InstinctLab `release/3.0.0` in the clone. An intermediate commit may leave tasks unrunnable; its message says so. Before each rewrite, read the relevant upstream code at the synced SHA. Follow `agents/CommonAgentInstructions.md`:

- smallest correct change
- no shims or aliases for removed upstream names
- no defensive fallbacks
- run `pre-commit run --all-files`

### 8.1 Phase 1: metadata, Docker, import blockers

- `setup.py`:
  - pins to the synced main versions (at writing: isaaclab 41.0.1, isaaclab-assets 2.1.0, isaaclab-physx 9.0.0, isaaclab-tasks 29.0.1, isaaclab-visualizers 4.0.0, isaaclab-newton 10.0.0)
  - `warp-lang==1.17.0` (main's `pyproject.toml` at writing; EA pins 1.16.0)
  - keep `torch==2.11.0` and `torchvision==0.26.0`
  - drop `torchaudio` (main dropped it; InstinctLab never imports it)
  - classifier `Isaac Sim :: 6.2.0`
  - keep `instinct-rl==1.0.3`
- Docker files per §5.3.
- Import blockers:
  - `spawn_from_usd_file`
  - the stale `TYPE_CHECKING` imports
  - a temporary removal of the `iter_sources` import is **not** allowed; it is fixed in Phase 4, and until then the grouped ray caster is knowingly broken
  - `hydra_task_config` is fixed in Phase 2, not stubbed
- AST import scan: list every `from isaaclab* import X` and attribute access on Isaac Lab objects in `source/` and `scripts/`, and check that each resolves at the synced SHA. The current scan found exactly `cloner.query.iter_sources`, `_spawn_from_usd_file`, `hydra_task_config` and `isaaclab.motion_reference…`.
- The scan cannot see attributes resolved at runtime through the manager metaclass. The **runtime-only** rows in §7 (`NewtonManager._model`/`_state_0`/`_cl_site_index_map`, `sim.physics_manager.__name__`, class-level manager calls) import cleanly and break only at scene construction. Keep them on the Phase 5 list. A clean scan is not evidence that they are fixed.

Gate: the image builds; §5.4 passes; `import instinctlab` succeeds; `import instinctlab.tasks` succeeds except for the documented Phase 2/4 items.

### 8.2 Phase 2: CLI, Hydra, launch, video

- Mirror the structure of `isaaclab_rl/entrypoints/backends/train_rsl_rl.py` and `play_rsl_rl.py`. Keep InstinctLab's scripts as thin entry points.
- Put parser construction, validation, config layering, run-name validation and log naming in importable helpers in the `instinctlab` package. `scripts/instinct_rl/cli_args.py` stays the only documented exception to the scripts rule. Tests target the helpers only.
- Parsing: `add_launcher_args` → InstinctLab args → `setup_preset_cli(parser, argv)`. The remainder holds Hydra overrides, preset selectors (`physics=`, `renderer=`, `presets=`) and unknown flags.
- Training: `resolve_task_config(task, "instinct_rl_cfg_entry_point", overrides=...)`, then dedicated CLI options.
- Playback precedence: task defaults → saved `env.yaml`/`agent.yaml` → Hydra → dedicated CLI.
  - Resolve the log directory first.
  - Then layer the saved YAML onto the registered cfg instances.
  - Then compose Hydra.
  - Do not call `resolve_task_config` after loading the YAML.
  - Apply the same string→slice/space post-processing main applies.
- `train.py` run-name rule:
  - A Hydra-style argument is `[+|++]env.*|agent.*=…`, `~key[=…]`, or a preset selector.
  - When any is present, `--run_name` is required and non-empty. Both `--run_name v` and `--run_name=v` are accepted.
  - `agent.run_name=` does not satisfy the rule.
  - Malformed residual tokens give an argument error first.
- Log-directory name: `<timestamp>[_<env run_name>][_<agent run_name>][_from<date>_<time>]`.
- `play.py` writes files only for `--exportonnx` and `--video`. Suppress Hydra's `outputs/` directory if main still creates it. Asset/Kit/Warp cache writes are allowed.
- Preserve every existing flag and behavior:
  - checkpoint selection; absolute/relative `--load_run`; resume suffix
  - distributed (`LOCAL_RANK`, `--local-rank`, rank seed offset, CPU affinity) and `--logroot`
  - `--debug`, `--cprofile`, `--video*`
  - play flags: `--exportonnx`, `--env_cfg`, `--agent_cfg`, `--sample`, `--zero_act_until`, `--no_terminate`, `--aux_reward`, `--no_resume`, `--disable_fabric`, `--video_start_step`
  - task-local play flags: `tasks/parkour/scripts/play.py` `--useonnx`, `--keyboard_control`, `--keyboard_angvel`, `--keyboard_linvel_step`; `tasks/shadowing/play.py` `--x_offset/--y_offset` (used by `grid_search.sh`), `--cam_rotate_speed`, `--mpjpe`, `--play_teacher`, `--compare_actions`
  - `scripts/multi_play.py` forwards `--task`, `--load_run` and unknown args to `play.py` as a subprocess, so the play CLI must keep accepting them
- Before editing, spawn a subagent to inventory these flags exactly. The lists above are a lower bound.
- Launch and video:
  - Drop `headless`/`visualizer_explicit` plumbing in the asset-cache launch args.
  - Use lowercase visualizer names.
  - Port `--video` to main's `VideoRecorderCfg.source` and call `apply_video_recording` inside `launch_simulation`.
  - Set `configure_seed`'s `torch_deterministic` to the EA effective value.
- Other `AppLauncher` scripts (`amass_filter.py`, `amass_visualize.py`, `list_envs.py`, `convert_urdf.py`): fix only removed or deprecated fields. Known: `replicate_physics=False` at `amass_visualize.py:228` (deprecated and ignored on main). The `KitVisualizerCfg` at `amass_visualize.py:221` moves in Phase 5.

Gate:
- `--help` subprocess checks pass for both scripts.
- Helper tests pass for: precedence; preset passthrough; `+`/`++`/`~` forms; the run-name cases (train rejects missing or empty, rejects `agent.run_name=` alone, accepts no-Hydra; play never requires it); malformed input; log naming.
- The no-write test for `play.py` passes.
- A representative task reaches environment construction.

### 8.3 Phase 3: envs, managers, resets, terrains

- `InstinctRlEnv`:
  - `_reset_idx(env_ids: torch.Tensor | slice)`: normalize once for `MonitorManager.reset` (`torch.arange(num_envs, device)[env_ids]`) and pass the original selector to `super()`.
  - `load_managers`: keep the `MultiRewardCfg` → `DummyRewardCfg` swap. Check it against main's ordering (startup events run at the end of `super().load_managers()`).
  - `setup_manager_visualizers`: follow main's live-plot contract, adding `monitor_manager`. The reward scalar must handle the dict from `MultiRewardManager`.
- `MultiRewardManager`: verify it against main's `RewardManager` internals (`_episode_sum_buf`, `_step_reward`, `_class_term_cfgs`, scene-entity finalization) and the class-based term changes.
- `extras["log"]`: 0-d device tensors are fine for Instinct-RL 1.0.3. Re-check at the synced SHA.
- Reset selectors: fix every hit of the grep `len(env_ids)|torch.tensor(env_ids|torch.as_tensor(env_ids|env_ids.tolist()|meshgrid(env_ids` across `source/instinctlab` (72 at writing). Areas: commands, events, curricula, motion reference/buffers/files/generators, noisy camera, noise models.
  - Normalize at public entry points, and derive counts from the selected tensor.
  - Keep int dtypes consistent: main step resets pass int32.
  - Type hints become `torch.Tensor | slice`.
- Selector use in term calls:
  - Use resolved `joint_ids`/`body_ids` directly, not `torch.tensor(cfg.joint_ids)`.
  - Replace `data[:, cfg.body_ids[0]]` with `data[:, cfg.body_ids][:, 0]`.
  - Class-based terms read host lists in `__init__`.
- Noise callbacks must not mutate their input in place.
- Height field:
  - Generators return `(hf, origin)`, with origin = `[size_x/2, size_y/2, max(central 2 m patch)·vertical_scale]` (EA wrapper semantics).
  - `generate_wall` and any `TerrainGenerator` override must follow the same contract.
- Terrain scales: restore EA inheritance by giving InstinctLab Perlin sub-terrain cfgs `horizontal_scale`/`vertical_scale`/`slope_threshold = None`, or by setting the values explicitly in the parkour generator cfg.
  - Prefer the cfg-default change if no task relies on the 0.1 default without a generator; check this first.
  - The rough-terrain scan (187 points) must keep its order.

Gate: full, partial-slice and int32/int64 tensor resets pass for every matrix task on both backends; terrain origins match the baseline; the §6 snapshot matches for every task.

### 8.4 Phase 4: grouped ray caster, noisy camera, volume points

- Rewrite `flat_target_prim_registry.py` against main's `BaseMultiMeshRayCaster` internals at the synced SHA (pr8 contract):
  - prototype/world resolution via `cloner.path` and `plan.topology`
  - body binding through the scene-data provider (`_mesh_body_indices`)
  - delegate to `super()._build_mesh_records` for every case the override does not change
- Keep InstinctLab semantics:
  - an explicitly selected articulation root is one tracked body
  - world-to-entity ordering is deterministic
  - static mesh reuse is shared
- Under the per-env static-target rule, express shared static meshes with env-regex paths.
- Mount rule (user clarification, 2026-10-22): each sensor instance mounts on exactly one body per env. Main's `BaseRayCaster._initialize_impl` check ("expected one mount in every environment") and InstinctLab's own `_view_count == _num_envs` check both stay. The requirement is **several instances of the same sensor class in one scene, each on a different body**:
  - Parkour already does this with `left_height_scanner`/`right_height_scanner` (`RayCasterCfg`).
  - The grouped and noisy cameras must support it too, for example a second `NoisyGroupedRayCasterCameraCfg` on another link.
  - Many raycast targets per env (ground, the 29 G1 links, HOI objects via `RaycastTargetCfg`) is the grouped feature itself and is unaffected by the mount rule.
- Multi-instance isolation. Main keeps class-level state on `BaseRayCaster` (`meshes: ClassVar[...]`, `_instance_count`), so verify that two instances:
  - each get their own warp tables, world-ID tables, tracked-target bindings and noise/history buffers
  - share only static meshes keyed identically by the upstream cache
  - tear down without invalidating the other instance
- On Newton, `cl_register_site` (grouped ray caster `newton.py`, volume points) must return distinct labels for distinct bodies and the same label for the same body. Each instance resolves its own site indices.
- Noisy camera and volume points:
  - adopt `SensorBase` timing buffers
  - verify `update_period` timing against the baseline (the delayed-refresh fix may change when depth refreshes; restore through cfg if possible, otherwise record it in the manifest)
  - verify that the noise pipeline order, `AsyncCircularBuffer` history and delayed-image skip/delay are unchanged
  - verify the effect of image invalidation after pose writes

Gate: initialize/reset/update/teardown pass on both backends for static shared, tracked per-env, articulation-root, multi-body and multi-instance (two grouped cameras plus two grouped ray casters on different links in one scene, partial reset, the outputs of each instance unchanged by the other's presence) cases; ragged resolution gives a clear error; depth observations match the baseline within tolerance, or the difference is pre-declared.

### 8.5 Phase 5: physics backend managers and visualizers

- Replace class-level access with the instance: `sim.backend_manager.get_model()`, `get_state_0()`, `cl_register_site(...)`, and `self._backend_manager._cl_site_index_map`. The private map read follows main's `FrameTransformer` (`frame_transformer.py:126`) and `Imu` (`imu.py:113`). Sites, all from the two §7 physics-manager rows:
  - hard failures first: `sensors/volume_points/newton.py` (`_model`, `_state_0`, `_cl_site_index_map`) and `utils/backend_dispatch.py`
  - then the warning sites: `motion_reference/newton.py`, `sensors/grouped_ray_caster/newton.py`, `tasks/utils/newton.py`, `sensors/volume_points/newton.py` (`cl_register_site`)
- `PhysxManager.get_physics_sim_view` moves to the instance (`monitors/monitor_manager.py`, `sensors/volume_points/physx.py`).
- Test for the stale-dict trap: a Newton volume-points test must fail if the sensor reads `NewtonManager._cl_site_index_map` (the class-level `{}`) instead of the instance map.
- `utils/backend_dispatch.py`: use `sim.backend_manager` and `InteractiveScene.physics_backend` (`"physx"` | `"ovphysx"` | `"newton"`), not `physics_manager.__name__`. Compare with `==`; `"physx" in backend` also matches `"ovphysx"`.
  - `ovphysx` is **rejected for now** (user decision, 2026-10-22). Dispatch raises a clear `NotImplementedError` naming the backend and the supported set (`physx`, `newton`). Do not silently map it to PhysX.
  - Document it in the README's unsupported combinations and the manifest's unvalidated paths. Revisiting it is a separate decision.
- Do not cache the model across rebuilds. Reacquire `asset.data` after hard resets (newton 9.2.0).
- Port `InstinctNewtonVisualizerCfg` to an `InstinctNewtonGLVisualizer(NewtonGLVisualizer)` subclass selected by `class_type`.
  - Override `initialize(sim, *, cameras)` with a `super()` call, and `step`.
  - `show_visual`, `follow_body` and `show_contacts` must work again.
  - The private `_viewer` access stays only where no config field exists; cover it with a lifecycle test.
- Migrate `KitVisualizerCfg` + `sim.default_visualizer_cfg` to main's non-deprecated `visualizer_cfgs`/`VisualizerCfg(backend=KitBackendCfg())` form. Sites (§7 Visualizers row): the 7 task cfgs; the camera-orbit read of `default_visualizer_cfg.lookat/eye` in `tasks/shadowing/play.py:343-344`; `scripts/amass_visualize.py:221`; the parkour and perceptive Newton cfgs that use `InstinctNewtonVisualizerCfg`. Tracking HEAD makes removal of the deprecated forms likely during the migration's life.
- Newton contact sensors: confirm that `prim_path="{ENV_REGEX_NS}/Robot/.*"` resolves the same bodies in the same order as the baseline.
- Revalidate MJWarp options (mujoco-warp ~=3.15), collision capacities and CUDA-graph operation.

Gate: Newton volume points, motion references, visualizer follow mode, contacts and resets run without deprecation warnings from InstinctLab code; contact body lists match the baseline.

### 8.6 Phase 6: spawners, assets, datasets

- Remove InstinctLab `MeshFileCfg`, `spawn_from_mesh` and their re-exports, leaving a one-line removal comment at each site.
- Port the perceptive HOI cfg to `isaaclab.sim.MeshFileCfg(mesh=MeshFileCfg.TrimeshObjectCfg(mesh=…), scale, mass_props, collision_props, rigid_props, mesh_collision_props)`. Pass `mesh_collision_props` as a cfg object; the mapping form is rejected.
- Load meshes with a small InstinctLab utility that keeps the old semantics: `expanduser`, `trimesh.load_mesh(process=False)`, concatenate a `Scene`, reject non-triangle meshes.
- The `params/env.yaml` dump writes a placeholder for `TrimeshObjectCfg.mesh`, and `--env_cfg` loading skips it.
- Test the residual-Hydra (`register_task` → dict) path on HOI. If trimesh cannot pass through it, reject those forms for HOI with a clear error.
- Confirm the ray-cast depth camera still sees the objects at `{object}/mesh` on both backends.
- Dataset defaults are already committed (§5.1):
  - Hiking: `~/Datasets/hiking-in-the-wild_Data&Model/...`
  - Deep Whole-body Parkour: `~/Datasets/Deep-Wholebody-Parkour_Dataset`
  - OMOMO: `~/Datasets/OMOMO/...`

  Expand `~` at the filesystem consumers (`AmassMotion`, `TerrainMotion`, `mesh_terrains`), after overrides. Document the overridable fields and their derived companions. `__post_init__` does not re-run after CLI overrides.
- Asset cache: URDF importer 3.x dropped options, so check the post-processing. The cache digest includes the versions, so the first run rebuilds. Remove any use of `retrieve_git_asset_path`/`NEWTON_ASSET_REPO_URL`.

Gate: HOI spawns and is visible to the depth camera on both backends; dataset defaults resolve in the container; overrides with derived fields work.

### 8.7 Phase 7: behavior audit (restore-or-document)

Grep-driven audit over `source/instinctlab`. Each finding gets one of two outcomes:

- **Restored**: a cfg override reproduces the EA value, with a short comment.
- **Accepted**: recorded in the manifest with affected tasks and checkpoint impact.

Do not change behavior just to match a new default, unless the default is a bug fix that cannot be overridden.

Areas:

- physics-schema fragments and spawner props
- `AssetBaseCfg`/`scene.extras` changes
- URDF importer options
- actuator and solver limit ownership (`_clip_effort(effort, joint_vel)`; `"controller"` → `"drive"`)
- contact semantics and float64 air/contact time
- reset persistence of contact values
- articulation `joint_ordering`/`body_ordering` defaults
- quaternion broadcasting
- depth normalization: NaN/-inf → 0, returns a new tensor
- camera first-frame fix
- PhysX kinematics refresh before ray-caster reads
- height-field discretization
- ground-plane appearance
- `replicate_physics`/clone path removal and the `scene.env_origins` users listed in the §7 Cloning row
- `randomize_rigid_body_scale` errors on shared bodies
- `string_to_callable` exceptions
- `configure_seed` determinism
- `update_from_dict` no longer type-checks scalars (Hydra overrides)
- TorchScript load errors

Gate: every difference against the EA baseline is either gone or recorded in the manifest before the acceptance run.

### 8.8 Phase 8: validation matrix (on `aorua`, in the container)

| Family | PhysX | Newton |
|---|---|---|
| Locomotion | `Instinct-Locomotion-Flat-G1-v0` | `…-Newton-v0` |
| Whole-body shadowing | `Instinct-Shadowing-WholeBody-Plane-G1-v0` | `…-Newton-v0` |
| Perceptive shadowing | `Instinct-Perceptive-Shadowing-G1-v0` | `…-Newton-v0` |
| Perceptive VAE | `Instinct-Perceptive-Vae-G1-v0` | `…-Newton-v0` |
| Perceptive HOI | `Instinct-Perceptive-HOI-Shadowing-G1-v0` | `…-Newton-v0` |
| BeyondMimic | `Instinct-BeyondMimic-Plane-G1-v0` | `…-Newton-v0` |
| Parkour | `Instinct-Parkour-Target-Amp-G1-v0` | `…-Newton-v0` |

The `-Play-v0` variants are used for playback. Checks run in increasing cost order:

1. Static:
   - imports
   - no removed or deprecated upstream API used by InstinctLab
   - task enumeration
   - every env/agent cfg resolves
   - helper tests and `--help` subprocess checks
2. **Layout:** the §6 snapshot matches exactly for every row, including resolved joint/body lists, ONNX signatures and noise/history order.
3. Smoke: construct with 4 envs; full, slice `slice(1,4,2)` and int32/int64 tensor `[0,2]` resets; 16 zero-action and 16 fixed-action steps; finiteness; close and reconstruct.
4. Components:
   - grouped ray caster / noisy camera
   - volume points and virtual obstacles
   - motion reference and backend views
   - monitor dispatch
   - terrain origins and scales
   - asset cache cold and warm
   - HOI meshes
   - Newton visualizer lifecycle
   - video frame count and size
5. Checkpoints:
   - Hiking via `--useonnx` on PhysX, then Newton.
   - Deep Whole-body Parkour via `play.py` on both backends, plus `--exportonnx` with 1 env. The output must numerically match the shipped `exported/*.onnx` and `policy_normalizer.npz` on identical inputs.
   - Fixed-seed short rollouts must match EA within the manifest tolerances.
   - Play from the current registration, not `--env_cfg`. The saved YAML predates 3.0.
6. Training (approved list only): parkour, perceptive shadowing, perceptive HOI on both backends, `--num_envs 128 --max_iterations 1 --run_name upgrade3main_<agent>`, then save/resume. Distributed training is unvalidated (single GPU) and recorded in the manifest.

Every run folder follows experiment conventions:

- under `~/Data/instinctlab_logs/`
- git records for all three repos, including the Isaac Lab SHA
- `params/*.yaml`, checkpoints and events
- the comparison results

### 8.9 Phase 9: documentation and merge preparation

- `README.md`:
  - the Docker workflow on main (base build, InstinctLab image, mounts, `uv pip install -e`, no exact `uv sync`)
  - the frozen acceptance SHA, image ID, versions and the Instinct-RL revision
  - a note that the EA conda env `instinctlab` (Isaac Lab `release/3.0.0`) does not run InstinctLab `release/3.0.0` after commit 1
- `agents/computation/aorua.md`: container usage and paths.
- Dataset locations and override rules, including the Deep Whole-body Parkour download URL.
- Mark `agents/MIGRATE_BETA2_TO_300_INSTRUCTION.md` as superseded by this plan, and `agents/UPGRADE_PLAN.md` too if it exists by then. Write a downstream note (beta2/EA → main) only if the user asks.
- The validated matrix, known unsupported combinations and the manifest location.
- Merging InstinctLab `release/3.0.0` → InstinctLab `main` is a separate, user-initiated step. Nothing is pushed without an explicit request.

## 9. Final acceptance

1. Sync once more and freeze the SHA (§4).
2. Rebuild both images if `pyproject.toml`, `uv.lock` or `docker/` changed since the last build (§4 rule 3), then re-run §5.4.
3. Run all of Phase 8 on that SHA.

Done means:

- §6 layout matches exactly for all 14 rows (7 families × 2 backends).
- The main features in §1.2 pass their component checks.
- The CLI rules from §2 hold.
- Reference checkpoints play, and the ONNX export matches.
- Approved training and resume runs pass.
- Every value-level difference is restored or recorded.
- No InstinctLab code uses removed upstream APIs or emits upstream deprecation warnings.
- The Nutstore checkouts and the EA env are unchanged apart from §5.1 (branch, commits 0–1, restored sync damage).
- No user work is lost.

## 10. Manifest

`~/Data/instinctlab_logs/upgrade_3main/manifest.yaml` contains:

- host; image IDs; all repo SHAs per sync point; package versions; newton commit
- dataset paths
- reference checkpoint paths and hashes
- tolerances
- accepted differences (id, upstream change and version, affected tasks/backends, observation terms touched, checkpoint impact, rationale)
- unvalidated paths (distributed; `ovphysx` rejected)

## 11. Commit sequence on InstinctLab `release/3.0.0`

Commits 0–1 land in the Nutstore checkout before the clone exists. Commits 2–9 land in the clone.

0. Dataset-default edits (§5.1).
1. `--useonnx` accepts ONNX-only run directories (needed for the EA baseline). Add the §6 snapshot helper.
2. Metadata, Docker, and Phase 1 import blockers. Known breakage: CLI, grouped ray caster, and the §7 runtime-only physics-manager sites.
3. CLI helpers, the train/play port, the launch/video/seed fixes, and helper tests.
4. `InstinctRlEnv`, managers, reset selectors, height-field contract and terrain scales.
5. Grouped ray caster, noisy camera and volume points on main's sensor internals, with multi-instance tests.
6. Backend-manager instances, backend dispatch, Newton visualizer and visualizer cfg migration.
7. `MeshFileCfg` removal, HOI port, `~` expansion and asset cache.
8. Audit restorations.
9. Documentation.

If a sync point changes an already-ported area, fix it in a follow-up commit named after the upstream change.

## 12. Open decisions

None at writing. Resolved 2026-10-22:

- `ovphysx`: rejected with a clear error and documented (Phase 5).
- Sensor mounts: one body per env per instance; several same-class instances on different bodies must work (Phase 4).
- Container user: UID 1000 first, root fallback approved (§5.3).

Open to the user, not blocking: the InstinctLab branch name `release/3.0.0` (§2) reuses the name of Isaac Lab's EA branch. The discarded in-place plan proposed `migrate/isaaclab3-main` instead. Renaming is a user decision. Until then, the "Naming" rule in §2 applies.

## 13. Rollback

Nothing in this plan changes the EA setup beyond §5.1, so rollback is cheap at every point before the user merges:

- EA work continues at any time from the Nutstore checkouts with conda `instinctlab`. The Nutstore InstinctLab branch holds only commits 0–1. To return to the previous branch, use `git switch migrate/isaaclab3.0.0`; commit 0 is the only content difference.
- To abandon the migration, stop using `~/workspace/instinctlab_ws/` and prune the Docker images. The clones and the freeze folder stay until the user asks to remove them.
- The manifest records the pre-migration SHAs (`isaaclab e180372f3`, `instinct_rl 574c8d5`, `instinctlab 7b61512`) as explicit rollback targets.

## 14. Risks and assumptions

- **EA checkout moved.** Switching the Nutstore `isaaclab` checkout to `main` (for example, by following an in-place migration) makes conda `instinctlab` unable to run EA. If that happens before the §6 gate, the EA reference is lost until Isaac Lab `release/3.0.0` is checked out again. Mitigation: never switch that checkout during this plan. An agent executing this plan checks `git -C ~/NutstoreFiles/2Projects/isaacSim/isaaclab rev-parse HEAD` = `e180372f3` before every EA run.
- **Runtime-only breakage.** The manager-metaclass failures (§7) pass imports and the AST scan. The stale `_cl_site_index_map` read gives a misleading "site not registered" error. Mitigation: Phase 5 lists the sites explicitly and adds the stale-dict test.
- **Tracking a moving target.** The isaaclab major version moved 28 → 41 in about four weeks. Mitigation: sync points, a frozen acceptance SHA, the AST scan at every sync.
- **Unreleased `changelog.d` fragments** (`pr8-consumer-bindings`: multi-mesh body binding, mount rejection) may change again before acceptance. Phase 4 re-reads the internals at its sync point.
- **Nutstore CRLF damage.** Almost every working-tree file in the Nutstore `instinctlab` is CRLF. `git add` normalizes to LF. Never copy files from the Nutstore tree into the clone; transfer only through git (§3).
- **Single-GPU host.** Distributed training stays unvalidated and is recorded in the manifest.
- **Assumption:** the 4 dataset cfg edits are intended. Confirm before commit 0.
- **Assumption:** the §6 layout numbers are derived from cfgs. The freeze is authoritative.

## 15. Sync log

| Date | Phase | Isaac Lab `OLD..NEW` | Rebuild | Notes |
|---|---|---|---|---|
| 2026-10-22 | plan | `—..8c9b995cb` (local `upstream/main`, fetched 2026-10-22) | — | Scan basis for this plan |
