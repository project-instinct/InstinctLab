# AttnContacts + SuperMimic Reproduction Plan (Instinct-RL + InstinctLab)

Status: plan only; nothing in this document is implemented. Revision 2, 2026-10-26 (user answers of
2026-10-26 applied: data, base branch, replay checkpoints, no dead-code removal, LAFAN1-only SuperMimic,
compute, dedicated worktrees, reusable interfaces). Revision 3, same day: reuse of existing `EstimatorMixin`,
`TransformerHeadModel`, `QueueRolloutStorage` / `utils/buffer.py`, TPPO LR schedulers, existing DDP code,
parkour `PoseVelocityCommand`, `BeyondMimicAdaptiveWeighting`, motion-reference formulas, monitors and events
(§5); interface fixes from a DeepSeek review; marsbrain code location.
Revision 4, same day: **ACP scope cut to module delivery plus one small demonstration** (§3.2); GRAIL dropped
(not the paper's data); ACP data taken from what is already on marsbrain, no downloads, no Mixamo retargeting;
build directly in marsbrain `~/workspace/isaacSim6` (§9.1); compute re-estimated (§9.2); only the last two
replay checkpoints kept (§5.1 U5).
**Revision 5, same day (setup done).** Branch `integrate/ACP-SuperMimic` created on marsbrain in both repos
(InstinctLab @ `dfa08ee`, Instinct-RL @ `574c8d5`); the two local scratch worktrees deleted; InstinctLab's 44
dirty files stashed; Instinct-RL overridden with local `main` (its 7 dirty files were already in `574c8d5`, and
are stashed as well). **Both papers now reproduce under LAFAN1 only** (user decision): ACP drops the NoKov box
set and the terrain zoo and runs on LAFAN1 clips; M5 export manifest skipped.

Merged from two earlier plans (kept as history, not to be edited further):

- `~/.opencode/plan/INSTINCT_RL_UPGRADE_PLAN.md` (Instinct-RL upgrade for AttnContacts / ACP, 2026-10-23)
- `~/.opencode/plan/supermimic-reproduction.md` (SuperMimic reproduction, 2026-10-23)

Development branch, created 2026-10-26: **`integrate/ACP-SuperMimic`** in both repos on marsbrain.

| Repo on marsbrain | Branch | HEAD | Base |
|---|---|---|---|
| `~/workspace/isaacSim6/instinctlab` | `integrate/ACP-SuperMimic` | `dfa08ee` | local `migrate/isaaclab3-main` (Isaac Lab `main`, 54 commits ahead of the old `update/isaaclab3.0.0-beta2` tree; contains `main` as an ancestor) |
| `~/workspace/isaacSim6/instinct_rl` | `integrate/ACP-SuperMimic` | `574c8d5` | local `main` (1.0.3), a complete override per user decision |

Reference clones `~/workspace/instinctlab_ws/` are now the same commits (`574c8d5`, `dfa08ee`); the local
`*-repro` worktrees are deleted. Line numbers below were read on the local clones and must be re-checked on
marsbrain (they match, since the commits are identical).
`IL` = `instinctlab/source/instinctlab/instinctlab`, `IR` = `instinct_rl/instinct_rl`.

## 0. Purpose

The point of reproducing these papers is to **build reusable modules from each paper's design**, and to show
those modules work. The two papers differ in how far that goes (user decisions of 2026-10-26):

1. Each module is a standalone, tested, plug-and-play piece of Instinct-RL or InstinctLab with a config
   interface, so later tasks can use it without the rest of the paper's pipeline (§5 gives every interface).
   **This is required of both papers.**
2. **SuperMimic**: the full pipeline built from these modules matches §3.1 on our stack (a real reproduction).
3. **ACP**: the modules are exercised end to end once, on one skill family, against the gates in §3.2. The
   paper's numbers stay for reference only; the full benchmark is a later, separate decision.

Papers:

| Short name | Paper | Core idea |
|---|---|---|
| **ACP** (AttnContacts) | He, Wen, Zhang et al., "Humanoid Mobility via Attention-based Contacts", arXiv:2610.31608 | A cross-attention planner picks the next contact for 9 bodies from a 17×11 terrain point scan; a tracker that sees contacts (not reference kinematics) executes them; 3 stages: tracking (S1), planner hand-off (S2), RL fine-tuning on goal-reaching terrains with two critics (S3). |
| **SuperMimic** | Liao et al., arXiv:2610.28480 (CoRL 2026) | Motion tracking trained by backpropagating through H=32 steps of a learned world model (k=8-step fit), with a critic that bootstraps the return after step H. |

Paper notes: `~/workspace/paper-intro-videos/{AttnContacts,SuperMimic}/` (`FACTS.md`, `paper/src/main_arxiv.tex`, `paper.txt`).

## 1. Recorded user decisions

| Topic | Decision | Applies to |
|---|---|---|
| Scope | Sim-only this time. Final policies must still be deployable: exportable, inputs available onboard, `instinct_onboard`-compatible files. SuperMimic is reproduced fully; ACP is module delivery + one demo (§3.2). | both |
| Robot | 29-DoF G1 (no 23-DoF variant), although ACP's paper uses 23 DoF. | both |
| Physics | ACP: PhysX only. SuperMimic: Newton with the MuJoCo-Warp solver (as in the paper); PhysX variant optional. | per paper |
| Base | **Decided:** Instinct-RL is **overridden completely with local `main`** (`574c8d5`, 1.0.3). InstinctLab is based on the current `main` line (`migrate/isaaclab3-main` @ `dfa08ee`, which contains `main` and is 54 commits newer than the previous working tree). | both |
| Worktrees | Edit only marsbrain `~/workspace/isaacSim6/{instinctlab,instinct_rl}` on `integrate/ACP-SuperMimic`; never the Nutstore tree or the local clones. The local `*-repro` worktrees are deleted. | both |
| RL library | Extend Instinct-RL (one `release/1.1` line, version 1.1.0). Do not move to rsl_rl. Instinct-RL is a **collection of plug-and-play algorithms**: every new class is registered and usable alone from a config. | both |
| Reuse | Where a paper needs something close to an existing Instinct-RL piece, extend that piece (default behavior unchanged) instead of writing a parallel one. Where both papers need the same thing, it is one shared module (§5.1). | both |
| Replay buffers | Saved with checkpoints, so resume is exact (§5.1 U5). | both |
| Dead code | Not removed. F1 only reports it. | Instinct-RL |
| SuperMimic placement | New algorithm `SuperMimic`, run by the existing `OnPolicyRunner`. No runner split. | SuperMimic |
| SuperMimic data | LAFAN1 only (the paper's 40 clips). No BONES-SEED flips this time. | SuperMimic |
| ACP scope | **Module delivery, not full reproduction.** The delivered modules (A-M1 … A-M8, A-M11) are built and tested; A-M9 and the stage-3 schedule are later/optional. The pipeline is exercised once end to end on one skill family (§3.2). Paper figure numbers are reference, not gates. | ACP |
| ACP data | The authors released no code or data ("Code TBD"). **Both papers are reproduced under LAFAN1 only** (user decision 2026-10-26): ACP uses the same LAFAN1 clips as SuperMimic, on flat terrain, with our own contact labels (§7.3). No downloads, no retargeting, no box-terrain set. Add a contact-scheduled motion reference data class under `MotionReferenceManager` (§5.4). | ACP |
| Export manifest (M5) | **Skipped** (user decision). `instinct_onboard` keeps its per-policy wiring; ACP's `planner.onnx` → `actor.onnx` link is documented in the export README per run instead. | both |
| GRAIL | **Dropped.** It is an NVIDIA locomanipulation dataset, not a source in the ACP paper (their Table S1 lists only OmniRetarget, Mixamo, LAFAN, Kimodo, AME rollouts), and it is 224 GB / CC BY-NC. | ACP |
| Where code is built | Directly in marsbrain `~/workspace/isaacSim6/{instinctlab,instinct_rl}` (user decision), on branch **`integrate/ACP-SuperMimic`** in both. Created 2026-10-26; local scratch worktrees deleted; dirty files stashed (§9.1). | both |
| Reference code | `zita-ch/supermimic_vibed` (MIT) may be consulted for design choices, not copied. Local read-only copy: `/tmp/opencode/ref/supermimic_vibed`. `lucidrains/attention-contact-planning` (MIT, WIP, planner only) likewise for ACP. | per paper |
| Compute | `marsbrain` (1× RTX 4090 D, 62 GB RAM, 117 GB disk) is free and does everything: build, tests, training. `baidu-cluster-kengo-zzw` and `ziwen-galaxea-desktop` are **not** used; `aorua` is not required (§9.1) unless a LAFAN1 clip is missing from marsbrain, which would first be reported as a blocker. | both |
| The "second algorithm" left open in the ACP plan (§9 there) | It is SuperMimic. It replaces PPO's update, so it is its own algorithm class, not a mixin. The "learner list instead of mixins" refactor is **not** needed. | both |

## 2. Invariants and coding rules

Blockers if broken:

1. **Default behavior unchanged.** With no new config keys, every existing algorithm gives the same losses and
   parameters as 1.0.3 on the golden tests (F0).
2. **Every existing InstinctLab task config runs unchanged**; the network I/O snapshot
   (`IL/utils/network_io_snapshot.py`) is identical for all task families.
3. **Existing checkpoints load** (model, optimizer, normalizers). The Deep Whole-body Parkour checkpoint
   re-exports to numerically identical `actor.onnx` / `policy_normalizer.npz`.
4. **Export file names and signatures** used by `instinct_onboard` stay as they are (`actor.onnx`,
   `0-<encoder>.onnx`, `policy_normalizer.npz`).
5. **Boundary:** Instinct-RL owns networks, losses, optimizers, rollout/replay storage and export.
   InstinctLab owns simulation, labels, rewards, curricula, env types and the imagination interface (its formulas).

Rules (from `CommonAgentInstructions.md`, `ProjectSpecificInstructions.md`, core coding principles):

- Smallest correct change; follow existing patterns (mixins such as `WasabiAlgoMixin`; base interface plus
  `*Newton` / `*PhysX` backend files with dynamic dispatch in InstinctLab).
- Reuse first (§1). A new shared base class needs at least two concrete users in this plan; §5.1 names them.
- No silent fallbacks: new config keys fail loudly on wrong shapes or missing obs groups.
- Explain the root cause of each defect to the user before fixing it (F1). Do not bundle fixes with features.
- Policy joint order via `instinctlab.utils.config.set_cfg_joint_order` in the robot `EnvCfg.__post_init__`.
- No commits, pushes or PRs unless the user asks in that conversation. Run `pre-commit run --all-files`
  where configured (Instinct-RL has none; ask before adding one).
- Logs under `~/Data/instinctlab_logs/`, datasets under `~/Datasets/`.

## 3. Reproduction targets (what "matches the paper" means)

### 3.1 SuperMimic (LAFAN1 only)

| ID | Target | Paper value | Source |
|---|---|---|---|
| SM-T1 | LAFAN1-40 eval normalized return ≥ 0.82 **and** 0/40 clips failed | 0.82 at 1.2 h, 1× H100 (supermimic_vibed: 0.829, 0/40 at 1.47 h on 1× RTX PRO 6000) | §4.1, Fig. 4a |
| SM-T2 | GPU-hours to SM-T1 recorded and compared with the paper after adjusting for GPU type | 1.2 H100-h | Fig. 4a |
| SM-T3 | World-model 8-step state error (normalized) in range | 0.007–0.011 | §4.2, Fig. 3c |
| SM-T4 | Critic ablation: without critic, H ≤ 16 does not converge; with critic, H=16 ≈ critic-free H=32; H=8 fails both | qualitative | §4.1, Fig. 4b,c |

Evaluation protocol (App. F.1): every clip once from frame 0, friction 1.0, no randomization or noise,
deterministic policy. Failure = training termination (root height error > 0.25 m or z-only tilt error > 0.8).
Normalized return = undiscounted episode return / largest possible return (the formula is not given; use
max per-step reward × (clip length − 1), as supermimic_vibed does, and state it in results).
Note: the paper uses 0.82 both as an eval return (Fig. 4a) and as a training return (Fig. 4b,c). SM-T1 uses
the eval meaning; SM-T4 compares runs with each other, not with the paper's absolute numbers.

Out of scope: BONES-SEED flips / side rolls (Table A.8) and the 264 h benchmark, SONIC comparison, PPO-WM
comparison, velocity interface, multi-GPU SuperMimic training (U3 makes it possible later), real robot.
PPO-Sim baseline (paper: 0.60–0.72 within 18 h) is optional, using the existing BeyondMimic PPO task.

### 3.2 ACP (module delivery, not full reproduction)

User decisions 2026-10-26: do not fully reproduce ACP in the paper's sense, and reproduce both papers **under
LAFAN1 only**. The deliverable is **reusable modules that work**, shown by one small end-to-end demonstration on
LAFAN1. Gates are module-level plus that demo; the paper's numbers (kept below for reference) are **not** pass
bars.

**What the paper's dataset actually is** (Table S1, verified in the TeX on 2026-10-26). 13.2 min total:
box climbing from OmniRetarget (already G1, need only contact labels); stairs, ladder, sit from Mixamo,
retargeted with their unreleased tool; walk and get-up from LAFAN; vault from Kimodo; sparse-terrain
reference motions are rollouts of their own AME policy (no contact labels). GRAIL is **not** one of them.
Under the LAFAN1-only decision we use the LAFAN row of that table (walk, get-up) plus the contact-rich LAFAN
clips the paper did not use (dance, fight, ground, jumps); the other six categories are out of scope. §7.3
records what each would have needed.

**Demo motion set: LAFAN1 only** (user decision). The same 40 clips as SuperMimic, from
`~/Datasets/LAFAN1_Retargeting_Dataset/g1/` (already on marsbrain, 40 CSVs) or the 77 converted npz. Chosen
because it is the one dataset both papers can share, and because it already contains genuine multi-contact
motion: `fallAndGetUp*` (6 clips: hands, knees and pelvis on the ground), `ground*` (5: sitting / lying), `dance*` (8),
`fight*` (3), `fightAndSports*` (2), `jumps*` (3), alongside `walk` (7), `run` (4) and `sprint` (2) for the
foot-only baseline — 40 in total. Contact labels come
from our own annotation (A-M8), as the paper's are unpublished.

Honest limitation, recorded up front: LAFAN1 is flat-ground motion, so the planner's 17×11 height scan carries
almost no terrain structure, and the attention-over-terrain mechanism is only *exercised*, not *validated*.
The demo therefore proves the pipeline (labels → replay → planner → composed tracker obs → export), and the
contact-prediction gate is on flat ground. A cheap way to give the scan real geometry later exists (place a
subset of the clips on a single box mesh from the existing terrain generators and re-annotate those clips), but
it is **out of scope now** and would be a separate decision, together with the box-terrain and full-benchmark
options.

| ID | Target | Verdict |
|---|---|---|
| ACP-G1 | Every delivered module (A-M1 … A-M8, A-M11) has unit tests and passes; golden suite (F0) unchanged. A-M9 / A-M10-stage3 are later and not gated | **gate** |
| ACP-G2 | Demo, stage 1: tracker tracks the contact schedule on LAFAN1 — ≥ 80 % of episodes reach the clip end without termination, on ≥ 3 clips of ≥ 2 motion types, including at least one `fallAndGetUp` clip (multi-contact) | **gate** |
| ACP-G3 | Demo, stage 2: planner trained on hindsight labels from those rollouts; top-1 contact prediction within 0.15 m of the label on held-out episodes | **gate** |
| ACP-G4 | Planner vs MLP baseline, same labels and budget: attention ≥ MLP in contact error | reported, not a gate |
| ACP-G5 | Stage 3 fine-tuning on goal-reaching terrain | **not attempted** under LAFAN1-only |
| ACP-G6 | Export `planner.onnx` + `actor.onnx`, reload, identical outputs; `instinct_onboard` can be fed | **gate** |
| ACP-G7 | Repo-wide regression: every existing task config still runs (F4) | **gate** |

Paper values for reference only (Fig. 7, checked by eye 2026-10-26): stage-1 tracker ≥ 98.5 % success in every
reference-anchored category; S2 → S3 gain +54.8 pp on sparse whole-body (27 → 82 %); S3 contact error
0.12–0.21 m; attention vs MLP 82 vs 38 (sparse whole-body). Our 0.15 m G3 threshold is a deliberate relaxation
of the paper's ~0.12 m, appropriate for one dataset on flat ground.

The paper's "11 categories" = 4 reference-anchored + 4 goal-reaching (dense/sparse × foot/whole-body) + 3
unseen (composition, distractors, roughness). The demo uses none of those axes: it is one dataset on flat
ground.

Out of scope now: that full benchmark, the 3,000-episode × 3-seed protocol, MGMT comparison, 8 composite
courses, the box / stairs / ladder / sit / vault / sparse categories, LiDAR neural mapping, real robot. Deployability still means: planner and
tracker scans are yaw-aligned height grids the existing perceptive tasks already produce, and both networks
export to ONNX.

If the modules land well and the demo works, extending to the full benchmark is a separate decision later
(§9.2 keeps the numbers for that case).

## 4. Paper facts used by the design (condensed)

### 4.1 SuperMimic

- Observation (Table A.1), 10-frame history: joint pos rel (29, U±0.01), joint vel (29, U±0.5), base ang vel
  (3, U±0.2), projected gravity (3, U±0.05), previous action (29). Goal: K=10 future reference frames.
- Action `q_sp = α·a + q0`, BeyondMimic gains and action scale; 50 Hz; 4 × 5 ms substeps.
- Reward: BeyondMimic tracking without self-collision penalty; joint-limit weight −2 (not −10); every term × 0.02 s.
- Termination: root height error > 0.25 m or z-only tilt error > 0.8; no end-effector termination.
- Sim context ξ: friction U[0.3, 1.2]; joint offsets U±0.01 rad; torso CoM offset x ±0.025, y ±0.05, z ±0.05 m; no pushes.
- Reset sampling over 1 s bins: 50 % uniform, 40 % softmax(−25·G_b/G_max), 10 % running average of past softmax weights.
- Networks: world model MLP (2048, 2048, 2048); motion encoder MLP (2048, 1024, 512, 256) → 32-d tanh latent;
  decoder MLP (2048, 1024, 512, 256) on latent + last 10 observations; critic MLP (256, 128) on (state, goal, ξ).
- Table A.5: k=8, H=32, γ=0.99, λ=0.95, 4096 envs, action noise N(0, 0.1²), buffer 1000 steps/env,
  10 × 2048 windows per WM update, 4096 imagined rollouts per policy update, Adam 2e-4 (policy β=(0.9, 0.95)),
  cosine decay to 0.03× over 250k iterations (320k total), grad clip 1.0 (policy then element-wise 0.1),
  critic 5 epochs × 4 minibatches, update-to-data ratio 0.4.
- supermimic_vibed deviations worth copying as options: bf16 MLP matmuls, analytic reward/termination heads,
  sigmoid-softened termination in imagination, extra obs = reference root orientation in root frame (6D),
  500 world-model-only warm-up iterations, MJWarp `njmax=500`, `nconmax=128` (overflow during falls).

### 4.2 ACP

- Tracker: MLP {512, 256, 128} ELU. Paper input 468 = 5 × 75 proprio + 63 scan (9×7) + 27 contact targets + 3 goal.
  **With 29 DoF: 558** = 5 × 93 + 63 + 27 + 3. Output 29 joint offsets × 0.25 + default pose; 50 Hz policy, 200 Hz PD.
- Planner: 187 terrain points (17×11, 1.6 × 1.0 m, 0.1 m, body-yaw frame), 5-step proprio history, goal.
  2-layer 5×5 CNN + per-point features → keys/values (96); 9 learned body embeddings → FiLM → 9 queries;
  1-layer cross-attention, 32 heads; top-k decoding (k=1 in evaluation); occupancy head, keep target if > 0.5.
  Paper size 786,993 params (MLP baseline 833,322); ours will differ because of the 29-DoF proprio input
  (encoder input 441 → 531: the 27 contact targets enter after the encoder). Tests check our own computed count.
- Planner loss: soft-target cross-entropy (Gaussian σ_a = 0.1 m over the 187 points, masked to valid contacts)
  + occupancy BCE. Adam 1e-4, 10 updates per PPO iteration × 4096 entries × 8 accumulation sub-batches,
  start after ≥ 1000 entries.
- Hindsight labels: scan a finished episode backward; contact onset when vertical force > 1 N; drop segments
  < 10 steps; next onset within 1.5 s (75 steps); express in that step's body-yaw frame; keep if inside the
  [−0.8, 0.8] × [−0.5, 0.5] m footprint. Goal-reaching segments are cut at the first step within 0.2 m of
  the goal (Table S5). Admission: reference envs start in the first 10 % of the clip and reach
  ≥ 90 %; goal envs come within 0.5 m horizontally and 0.1 m vertically of the goal. Quality = positive
  mechanical work / initial goal distance (denominator ≥ 0.5 m).
- Critics: V_m (reference) paper 353 → **383 with 29 DoF**; V_fs (goal) paper 298 → **316** (V_m gains the
  +6-DoF joint pos/vel plus the extra base-reference terms; V_fs gains only the joint terms, hence the smaller
  delta). Separate advantage normalization per env type; V_fs fixed lr 1e-3 and 500-iteration warmup (its samples excluded
  from the actor loss, no value clipping).
- PPO (Table S6): 24 steps/env, 5 epochs, 8 minibatches, clip 0.2, value coef 1.0, grad norm 1.0, γ 0.99,
  λ 0.95, lr 1e-4 KL-adaptive, target KL 0.02/0.01/0.01, entropy 0.006/0.005/0.002 (S1/S2/S3).
  Envs per worker: 12,000 / 2,400 / 2,100 (workers 3 / 2 / 4, so totals 36,000 / 4,800 / 8,400).
- Rewards (paper Table 3, Δt = 0.02 s): reference tracking (base pos 3.0, base vel 1.0, joint pos 2.5, joint vel
  0.5); contact label 5.0, contact pos base 2.0, contact pos global 1.0 (scaled by 1 − η_rew in S2, off in S3);
  track_planned_contact 1.5·η_rew (S2), 1.5 (S3); move2goal 40, standatgoal 20 (S3 goal envs);
  regularization (action rate −0.1, 2nd order −0.05, torque −1e-5, torque limits −1e-3, dof pos limits −10,
  dof vel limits −1, contact_acc −0.01); S3 goal envs: flat orientation, stumbling, slippage −10 each, ang vel xy −0.1.
- Curricula: S1 CSI (start phase 0.9, −0.05 after reaching 0.99, +0.05 otherwise); S2 start phase U(0, 0.98);
  S2 η_obs/η_rew linear ramps (Eq. 3) with **unknown** start iterations and lengths; S3 10-level terrain
  curriculum (promote: reached within 0.5 m and success EMA > 0.5; demote: closest distance > 1.0 m), 15 s episodes.
- Domain randomization (Table S2): friction U(0.3, 1.2), restitution U(0, 0.5), torso mass U(−2, 4) kg, leg
  mass ×U(0.85, 1.15), torso CoM as SuperMimic, joint bias ±0.01 rad (ankle ±0.03), armature and PD gains
  ×U(0.9, 1.1), action delay U(0, 20) ms. Obs noise: ω U±0.05 rad/s, projected gravity U±0.0125, q U±0.01 rad,
  q̇ U±1.5 rad/s, contact targets U±0.1 m, goal direction/yaw U±0.2 rad. Exteroception: planner scan
  N(0, 0.03²) m, tracker scan U±0.1 m, odometry delay U(50, 100) ms, XY drift ≤ 0.05 m (disk), Z drift U±0.05 m.
  Resets: joints U±0.2 rad around reference/default, base height +U(0, 0.05) m, yaw U±0.4 rad (goal envs).
  Physics randomization is sampled once per env and is active in all stages; evaluation turns all of it off.
- Motion data (Table S1, 13.2 min total, no per-clip counts; read from the TeX 2026-10-26): box climbing
  (OmniRetarget), stairs / ladder / sit (Mixamo, retargeted with the unreleased MGMT tool), walk / get-up
  (LAFAN), vault (Kimodo), sparse terrains (rollouts of their own AME policy, unlabeled). None of it is
  published, and the retargeting tool with it. GRAIL is not involved.
- Not in the paper, must be chosen by us and written down: η schedule numbers, iteration counts per stage,
  termination conditions, nominal PD gains, sim dt/substeps, reference episode length, replay capacity.

## 5. Module design and interfaces

Notation: signatures are Python; `obs_format` is Instinct-RL's `{group: OrderedDict[component, shape]}`.
"Registered" means exported from `IR/algorithms/__init__.py` (in `__all__`) or `IR/modules/__init__.py`, so a
config can name the class directly; every class is also reachable as `"pkg.module:Class"`.

### 5.0 What is reused from Instinct-RL as it is today

| Existing piece | Reused by | How |
|---|---|---|
| `PPO` (+ `init_storage`, `act`, `process_env_step`, `compute_losses`, `compute_auxiliary_reward`, `update`, `state_dict` hooks) | ACP | `ContactPlannerPPO = PlannerAlgoMixin + PPO`; multi-critic options added to `PPO` itself (A-M1) |
| `PPO(advantage_mixing_weights=float \| list)` (`ppo.py:53,96-99,296`, mean over the reward dim) | ACP | A-M1 extends this existing option with a `str` (per-env obs group) form and a reduction mode; float/list behavior unchanged |
| Mixin pattern of `WasabiAlgoMixin` (`super().__init__` first, nets + optimizer + storage built in `init_storage`, own loop after `super().update()`) | ACP `PlannerAlgoMixin` | same hook order |
| `ActorCritic` multi-critic `critics` (built when `num_rewards > 1`) and `evaluate(list[Tensor])` | ACP V_m / V_fs | A-M1 makes `_build_critic`, `critic_obs_segments` and the `EncoderActorCriticMixin` critic path per group (`critic_obs_groups`); one group = today |
| `EncoderActorCritic` + `ParallelLayer` + `MlpModel` | SuperMimic policy (motion encoder → 32-d latent, decoder on latent + history) | config only, plus `MlpModel(output_nonlinearity=...)` for the tanh latent |
| `DeterministicPolicyMixin` (`modules/deterministic_policy.py`) | SuperMimic policy | `EncoderDeterministicActorCritic` = mixin + `EncoderActorCritic` |
| `Conv2dModel` / `MlpModel` | ACP planner CNN, per-point MLP, MLP baseline; SuperMimic world model | building blocks inside the new modules |
| `EmpiricalNormalization` (+ DDP sync, `export`) | runner normalizers for both; world-model input/delta normalizers | as is, plus module kwarg `exclude_slices` and builder kwargs `exclude_components` + `obs_format` (A-M3) |
| `RolloutStorage` (+ per-reward GAE) and its `MiniBatch` subclass pattern (`ActionLabelRollout`, `rollout_storage.py:482-490`) | ACP | per-critic obs and mask added the same way (subclass `MiniBatch` with extra fields); GAE math moved into U2 unchanged |
| `QueueRolloutStorage` + `utils/buffer.py` (`buffer_expand`, `buffer_swap`, `buffer_to`, `buffer_method`, `buffer_from_example`) | U5, S-M1 | ring-buffer cursor logic and allocation / device moves reused inside `TensorDictBuffer` / `ReplayStorage` |
| `rollout_files/rollout_dataset.py` (`RolloutDataset`: sample stored experience into minibatches by obs component) | A-M4 | prior art for obs-component assembly of sampled batches; reuse its assembly helpers |
| TPPO LR scheduler convention (`lr_scheduler_class_name` + kwargs → `torch.optim.lr_scheduler.*`; `lr_scheduler_state_dict` in `PPO.state_dict`) | U4 | `ModuleLearner` uses the same convention; cosine schedule = `torch.optim.lr_scheduler.LambdaLR` with a small `cosine_with_floor` lambda, no new scheduler protocol |
| Existing DDP code (`ppo.py:358-383` broadcast / per-tensor all-reduce, `wasabi.py:299-306`, `normalizer.py:119-169`, `on_policy_runner.py:520-548`) | U3 | U3 helpers are extracted from these call sites; PPO's own per-tensor path is left as is (float order) |
| `ckpt_manipulator` (`ignore_missing_key`, `fit_smaller_weight`) | ACP S1 → S2 → S3 transitions | as is |
| `get_obs_slice`, `get_subobs_by_components`, `replace_obs_components` | planner input slicing, composed policy obs, world-model state layout | as is; masked use goes through U7 |
| `ParallelLayer.export_as_onnx`, `ActorCritic.export_as_onnx` | both exports | as is. Two signatures, kept distinct: network level `export_as_onnx(obs, filedir)` (new modules follow it); algorithm level `export_as_onnx(obs, extra_obs, export_dir)` (U1 hook, A-M5, S-M4) |
| `OnPolicyRunner` (learn loop, logging, save/load, export) | both | small generic fixes (U1); no new runner |
| `EstimatorMixin` (`modules/state_estimator.py`) + `EstimatorAlgoMixin` (`algorithms/estimator.py`): an upstream net whose output replaces policy-obs components, trained by a supervised loss | ACP planner | closest existing pattern to A-M5; both use the slot-fill helper U7. A-M5 uses the same list-style keys (`planner_target_components`, like `estimator_target_components`). Why not `EstimatorPPO` itself: see A-M5 |
| `TransformerHeadModel` (`modules/transformer.py`) and its ONNX SDP note | ACP attention blocks (A-M2) | same constructor conventions (`MlpModel` input layer, `d_model`, `num_heads`, `dim_feedforward`, `norm_first`), same ONNX workaround; A-M2 is cross-attention, which `TransformerHeadModel` (self-attention + pooling) cannot express |
| `LipschitzyMixin`, `TPPO`, `EstimatorTPPO` | later combinations | new mixins follow the same `Mixin + PPO` pattern so e.g. `PlannerAlgoMixin + TPPO` works without new code |

Unknown kwargs: `PPO` and `ActorCritic` only warn on unknown kwargs today (`ppo.py:69-71`, `actor_critic.py:58-61`);
that stays (invariant 1). The new classes (`ContactPlannerPPO`'s mixin, `SuperMimic`, `ModuleLearner`, the new
modules) raise on unknown kwargs, so a mistyped new option fails loudly.

### 5.1 Shared Instinct-RL modules (new; each has two or more users in this plan)

**U1. Runner generalization** (`IR/runners/on_policy_runner.py`). Users: ACP, SuperMimic, any out-of-tree algorithm.

- `"pkg.module:Class"` algorithm names (fix of `:73`), via one helper used everywhere a class is named:
  ```python
  # IR/utils/utils.py
  def resolve_class(name: str, default_namespace: types.ModuleType) -> type: ...
  ```
  used by the runner (algorithms), `build_actor_critic`, `build_normalizer`, and a new last branch in
  `ParallelLayer._build_one_block` (unknown names resolve through it instead of returning `None`; known names
  keep their current branches).
- No config mutation (copy before `pop("class_name")`).
- Optional, duck-typed algorithm hooks (called only if the algorithm defines them; PPO defines none, so
  default behavior is unchanged):
  ```python
  alg.set_normalizers(normalizers: dict[str, nn.Module])      # SuperMimic needs the policy normalizer inside imagination
  alg.get_policy_feedback() -> dict[str, Tensor]               # runner forwards to env.set_policy_feedback() before env.step()
  alg.export_as_onnx(obs: Tensor, extra_obs: dict, export_dir: str)  # if present, the runner calls it instead of actor_critic.export_as_onnx
  ```
- Algorithm construction: `alg_class(actor_critic, device=..., **alg_cfg, **env.get_algorithm_inputs())`.
- `log()`: `action_std` via `getattr(alg.actor_critic, "action_std", None)`; learning rate from
  `alg.learning_rate` or, if the algorithm has several, from `losses`/`stats` already logged.
- Checkpoint extras: adaptive LR per param group, critic-warmup state, RNG states (torch CPU/CUDA, numpy).
- `torch.load(..., map_location=device)`; final save works with `log_dir=None`.

**U2. GAE as a function** (`IR/utils/returns.py`). Users: `RolloutStorage.compute_returns` (bit-identical),
A-M1 masked per-critic advantages, SuperMimic critic targets on imagined rollouts.
```python
def compute_gae(rewards, values, dones, last_values, gamma, lam, time_dim=0) -> tuple[Tensor, Tensor]:
    """rewards/values: (T, N, R); dones: (T, N, 1); last_values: (N, R). Returns (returns, advantages), unnormalized."""
def normalize_advantages(advantages, mode="global", mask=None, eps=1e-8) -> Tensor:  # "global" == today
```

**U3. Distributed helpers** (`IR/utils/distributed.py`, earlier M4). Users: ACP planner, SuperMimic world
model / policy / critic. Extracted from the existing DDP call sites (§5.0); PPO's own per-tensor path stays
(changing it alters floating-point order), `WasabiAlgoMixin` and `EmpiricalNormalization` may switch to the
helpers only where the golden suite stays bit-identical.
```python
def is_distributed() -> bool
def broadcast_parameters(params: Iterable[Tensor], src: int = 0) -> None
def allreduce_gradients(params: Iterable[Tensor], op: Literal["mean", "sum"] = "mean") -> None   # one flattened buffer
def allreduce_scalar(value: Tensor | float, op: Literal["mean", "sum", "min", "max"]) -> Tensor
```

**U4. `ModuleLearner`** (`IR/algorithms/module_learner.py`): one parameter set with its own optimizer, LR
schedule, clipping and DDP. Users: ACP planner; SuperMimic world model, policy, critic (4 users). Existing
algorithms are not migrated (golden tests).
```python
class ModuleLearner:
    def __init__(self, params: nn.Module | Iterable[nn.Parameter], name: str,
                 optimizer_class_name: str = "Adam", optimizer_kwargs: dict | None = None,
                 max_grad_norm: float | None = None, max_grad_value: float | None = None,
                 lr_scheduler_class_name: str | None = None,  # TPPO convention: torch.optim.lr_scheduler name; None = fixed LR
                 lr_scheduler: dict | None = None,            # its kwargs
                 grad_accumulation_steps: int = 1)
    def step(self, loss: Tensor) -> dict[str, Tensor]   # backward, (accumulate), allreduce, clip, step, schedule; returns {f"{name}_grad_norm", f"{name}_lr"}
    def distributed_data_parallel(self) -> None          # broadcast parameters from rank 0
    @property learning_rate -> float
    def state_dict(self) -> dict; def load_state_dict(self, state: dict) -> None   # optimizer + schedule state
```
Cosine decay to 0.03× (SuperMimic) is `LambdaLR` with `IR/utils/lr_schedule.py::cosine_with_floor(final_ratio,
decay_steps, warmup_steps=0)`; ACP's planner and V_fs use a fixed LR. The state is saved as
`lr_scheduler_state_dict`, as PPO does.

**U5. Tensor buffers and checkpointed replay** (`IR/storage/tensor_buffer.py`). Base for the two replay
storages (S-M1 and A-M4), which store different things but share allocation, device moves and checkpointing.
```python
@dataclass
class TensorSpec: shape: tuple[int, ...]; dtype: torch.dtype = torch.float32
class TensorDictBuffer:
    def __init__(self, leading_shape: tuple[int, ...], specs: dict[str, TensorSpec], device: str | torch.device)
    def write(self, index, data: dict[str, Tensor]) -> None          # index into leading dims
    def gather(self, index, keys: Sequence[str] | None = None) -> dict[str, Tensor]
    def state_dict(self) -> dict[str, Tensor | int]                  # only tensors + ints: loads with weights_only=True
    def load_state_dict(self, state: dict) -> None                   # shapes/specs must match, else raise
    @classmethod def specs_from_obs_format(cls, obs_format, groups: Sequence[str]) -> dict[str, TensorSpec]
```
Replay checkpointing (user decisions: replay buffers are saved; **only the last two replay files are kept**).
Each algorithm returns its buffer under `state_dict()["replay_state_dict"]` when
`save_replay_in_checkpoint=True` (default for both new algorithms; False keeps the old refill-after-resume
behavior). Sizes: SuperMimic 1000 steps × 4096 envs ≈ 3–4 GB fp32 (≈ 2 GB with the bf16 storage option); ACP
planner buffer ≈ 0.3–1 GB depending on capacity. The replay part goes to a sidecar `model_{it}.replay.pt`
next to `model_{it}.pt`; `load()` reads it back when present. The runner keeps exactly the **2 most recent**
sidecars per run and per rank (`keep_replay_checkpoints=2`, pruned on every save) and keeps all model checkpoints; if the
two newest sidecars are both unreadable or from a different config, resume starts with an empty buffer and logs
a warning. Under DDP each rank has its own buffer and saves `model_{it}.replay.rank{r}.pt`; nothing is
gathered to rank 0. Disk on marsbrain is 117 GB, so this matters: a SuperMimic run with 3 sidecars would cost
≈ 12 GB instead of 8.

**U6. Imagination interface protocol** (`IR/env/imagination.py`, earlier S-M5 protocol). Users: SuperMimic;
later any model-based method. Implemented in InstinctLab (pure torch, differentiable).
`ref_idx` everywhere (U6, S-M1, S-M5, obs group `ref_index`) is one `LongTensor (B, 2)` = (motion id, frame),
so large datasets stay exact (a single concatenated float index would not).
```python
class ImaginationInterface(Protocol):
    state_format: OrderedDict[str, tuple]        # world-model state layout (root pos/quat, joints, vels ...)
    context_format: OrderedDict[str, tuple]      # sim context ξ
    def policy_obs(self, state_hist, ctx, prev_action, ref_idx) -> Tensor   # unnormalized, same layout as obs group "policy"
    def add_obs_noise(self, obs) -> Tensor
    def critic_obs(self, state, ctx, ref_idx) -> Tensor
    def reward(self, state, action, prev_action, ref_idx) -> Tensor          # (B, R)
    def termination_prob(self, state, ref_idx) -> Tensor                     # (B,), differentiable (sigmoid-softened)
    def advance(self, ref_idx) -> Tensor
    def valid_start(self, ref_idx, horizon: int) -> Tensor                   # (B,) bool
```

**U7. Obs slot filling** (`IR/utils/utils.py`). Users: `EstimatorMixin.act` (rewritten to call it,
bit-identical on the golden suite), A-M5 `PlannerAlgoMixin.act`, SuperMimic imagination (writes predicted
state into policy-obs layout).
```python
def fill_obs_components(obs: Tensor, component_names: Sequence[str], values: Tensor,
                        obs_segments: OrderedDict, mask: Tensor | None = None) -> Tensor
    # returns a copy; rows where mask is False keep obs. Wraps replace_obs_components.
```

**Runner–env channels** (`IR/env/vec_env.py` + InstinctLab wrapper, earlier M2). Users: ACP (iteration for η,
planned contacts for rewards), SuperMimic (imagination interface, final obs).
```python
class VecEnv(ABC):
    def set_training_iteration(self, iteration: int) -> None
    def set_policy_feedback(self, feedback: dict[str, Tensor]) -> None
    def get_algorithm_inputs(self) -> dict[str, Any]      # {} for tasks without extras
    # step(): infos["final_obs"] = {group: Tensor} for envs that reset this step (same format as observations)
```

Other shared modules (InstinctLab or scripts):

| # | Module | Path | ACP use | SuperMimic use | Later reuse |
|---|---|---|---|---|---|
| M1 | Golden test suite: CPU `FakeVecEnv` (configurable obs groups, `num_rewards`, time-outs, resets) + tiny training driver + recorded 1.0.3 fixtures | IR `tests/` | regression for A-M1 | regression for U1 | every future Instinct-RL change |
| M5 | ~~Export manifest (`export_manifest.json`)~~ — **dropped** (user decision). Instead, each run's export README documents how `planner.onnx` output feeds the `contact_targets_b` slot of `actor.onnx` | IR export path | documented per run | — | reconsider only if `instinct_onboard` needs multi-network wiring |
| M6 | Motion-data converters: LAFAN1 Unitree CSV → `*retargetted.npz`; retarget/asset check (joint zeros, foot height after FK, feasibility filter). OmniRetarget / PHP npz → npz + terrain metadata: **later, not built now** | IL `scripts/` | the 40 LAFAN1 clips (shared with SuperMimic) | the same 40 clips (same conversion) | any new dataset into the motion reference stack |
| M7 | Shared domain randomization as reusable `EventTerm` config groups, both backends. Mostly config reuse: joint calibration offsets = existing `randomize_default_joint_pos`; friction and torso CoM = the terms BeyondMimic already configures (`beyondmimic_env_cfg.py`); new code only for what is missing (ACP mass/gain/delay checks per backend) | IL `envs/mdp/events/` | yes (plus mass, gains, delay) | yes | all G1 sim-to-real tasks |
| M8 | Motion-library evaluator over `play.py` + existing monitors: deterministic per-clip runs, no randomization, CSV/JSON report of success / normalized return / failure / contact error. Wilson CI and per-category aggregation only when the full benchmark is attempted | IL `scripts/instinct_rl/evaluate.py` | demo: per-clip / per-skill (ACP-G2, ACP-G4); 11 × 3,000 later | every clip once from frame 0 | every tracking or goal-reaching policy |
| M9 | Stateless batched motion lookup by (motion id, frame): `MotionBuffer.lookup(motion_ids, frame_ids, fields) -> dict[str, Tensor]`, factored out of `AmassMotion.fill_motion_data` (which then calls it) | IL `motion_reference/` | contact-onset precompute checks; goal sampling | imagination goals | any model-based or offline use of motion data |

### 5.2 ACP Instinct-RL modules

**A-M1. Per-env multi-critic PPO** (options on `PPO`, `ActorCritic`, `RolloutStorage`; all default to today).
```python
PPO(...,
    advantage_mixing_weights: float | list[float] | str = 1.0,   # str: obs group of shape (N, R) giving per-env weights (env-type one-hot)
    advantage_mixing_reduction: Literal["mean", "sum"] = "mean",  # per-env group requires "sum"
    value_mask_group: str | None = None,                         # obs group (N, R) in {0,1}: which critic trains on which env (ACP: "env_type", the same one-hot)
    advantage_normalization: Literal["global", "per_critic_masked"] = "global",
    critic_learning_rates: list[float | None] | None = None,     # None entry = follows the adaptive actor LR
    critic_warmup_iterations: list[int] | None = None,           # critic r trains alone; its envs give 0 actor gradient
    critic_value_clipping: list[bool] | None = None)
ActorCritic(..., critic_obs_groups: list[str] | None = None)   # one critic per group, own input size; None = today
RolloutStorage(..., critic_obs_shape: list | list[list] , mask_shape=None)   # list of shapes when several groups
# MiniBatch gains critic_obs_list / value_mask through the existing MiniBatch-subclass pattern (ActionLabelRollout)
```
`advantage_mixing_weights` already exists for float/list (mean over the reward dim); only the `str` form and
`advantage_mixing_reduction` are new.
Grad-norm clipping stays joint over all parameters (paper gives one max norm). With `critic_learning_rates`
unset there is one param group, so old optimizer states load. Later use: any multi-task PPO mixing env types.

**A-M2. Attention blocks** (`IR/modules/attention.py`, registered).
```python
class FiLM(nn.Module):
    def __init__(self, cond_size: int, feature_size: int, hidden_sizes: Sequence[int] = ())
    def forward(self, x: Tensor, cond: Tensor) -> Tensor              # x * (1 + γ(cond)) + β(cond)
class CrossAttentionHead(nn.Module):
    def __init__(self, query_size: int, kv_size: int, d_model: int, num_heads: int, num_layers: int = 1,
                 dim_feedforward: int | None = None, dropout: float = 0.0)
    def forward(self, queries: Tensor, keys_values: Tensor, key_padding_mask: Tensor | None = None
                ) -> tuple[Tensor, Tensor]   # (B, Q, d_model), head-averaged weights (B, Q, P)
```
ONNX: same SDP-kernel workaround as `ParallelLayer.export_one_block_as_onnx`. Built on `nn.MultiheadAttention`
with the constructor conventions of `TransformerHeadModel`. They take two inputs and are used inside modules such
as the planner, not as `ParallelLayer` blocks (which take one flat tensor). Later use: any query-over-points perception (footholds, grasp points, HOI).

**A-M3. `EmpiricalNormalization(shape, ..., exclude_slices: Sequence[slice] = ())`** and
`build_normalizer(input_shape, ..., obs_format: dict | None = None, exclude_components: list[str] = [])`
resolving the slices from `obs_format` (both new kwargs required together). Excluded slots keep mean 0
/ std 1, so `policy_normalizer.npz` keeps its format. Later use: any obs slot filled after normalization.

**A-M4. `PartitionedReplayBuffer`** (`IR/storage/replay_buffer.py`, on U5).
```python
class PartitionedReplayBuffer:
    def __init__(self, capacity: int, specs: dict[str, TensorSpec], num_partitions: int,
                 eviction: Literal["fifo", "lowest_quality"] = "fifo",
                 sampling: Literal["uniform", "balanced"] = "uniform", device="cpu")
    def add(self, samples: dict[str, Tensor], partition: Tensor, quality: Tensor | None = None) -> int
    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> dict[str, Tensor]
    def __len__(self) -> int
    def ready(self, min_size: int) -> bool                     # allreduce MIN across ranks (U3)
    def state_dict(self) -> dict; def load_state_dict(self, state) -> None
class ReplayMixer:   # several buffers sampled with fixed ratios (reference vs goal-reaching samples)
    def __init__(self, buffers: dict[str, PartitionedReplayBuffer], ratios: dict[str, float])
    def sample(self, batch_size) -> dict[str, Tensor]; def ready(self, min_size) -> bool; state_dict / load_state_dict
```
Later use: self-imitation, hindsight relabeling, any supervised head trained from rollouts.

**A-M5. Planner modules and the generic upstream-planner mixin.**

Planners (`IR/modules/contact_planner.py`, registered), one interface so the MLP baseline is a config switch:
```python
class PlannerOutput(NamedTuple): logits: Tensor; occupancy_logits: Tensor   # (B, Q, P), (B, Q)
class AttentionContactPlanner(nn.Module):
    def __init__(self, input_segments: OrderedDict[str, tuple], scan_component: str, scan_grid: tuple[int, int],
                 point_xy: Sequence[tuple[float, float]] | None, proprio_components: Sequence[str], goal_component: str,
                 num_queries: int = 9, d_model: int = 96, num_heads: int = 32,
                 cnn_kwargs: dict = ..., point_mlp_sizes=..., film_hidden_sizes=..., top_k: int = 1,
                 temperature: float = 1.0, occupancy_threshold: float = 0.5)
    def forward(self, planner_obs: Tensor) -> PlannerOutput
    def decode(self, out: PlannerOutput, deterministic: bool = True) -> tuple[Tensor, Tensor]  # (B, Q, 3) in body-yaw frame, (B, Q) bool
    def compute_loss(self, planner_obs: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]  # soft CE (σ=0.1 m) + occupancy BCE
    @property output_size -> int                                   # Q*3 (+ Q occupancy)
    def export_as_onnx(self, planner_obs: Tensor, filedir: str, filename="planner.onnx")
class MlpContactPlanner(nn.Module): same signature minus the attention arguments, MlpModel backbone
```
Uses `Conv2dModel`, `MlpModel`, A-M2. Point heights come from the scan; the point grid xy is fixed and passed in.

Mixin (`IR/algorithms/planner.py`). Written for any "upstream network fills a slot of the downstream policy
obs and is trained by supervision from env samples", not just contacts:
```python
class PlannerAlgoMixin:
    def __init__(self, *args,
                 planner_class_name: str = "AttentionContactPlanner", planner_kwargs: dict = {},
                 planner_obs_group: str = "planner",
                 planner_target_components: list[str] = ["contact_targets_b"],   # components of obs group "policy" to fill (A-M11 obs term name)
                 planner_gate_group: str | None = "contact_source",  # (N,1): 1 = use planner, 0 = keep env value
                 planner_feedback_keys: dict[str, str] = {"positions": "planned_contacts", "occupancy": "planned_occupancy"},
                 planner_samples_key: str = "planner_samples",       # infos key with labeled samples (§8)
                 planner_learner_kwargs: dict = {"optimizer_kwargs": {"lr": 1e-4}},   # U4
                 planner_num_updates: int = 10, planner_batch_size: int = 4096,
                 planner_accumulation_steps: int = 8, planner_min_samples: int = 1000,
                 planner_replay_kwargs: dict = {}, planner_replay_mix: dict[str, float] | None = None,  # A-M4
                 freeze_planner: bool = False, save_replay_in_checkpoint: bool = True, **kwargs)
    def init_storage(...)             # super(); build planner via resolve_class, ModuleLearner, replay
    def act(self, obs, extra_obs)     # planner(extra_obs[planner_obs_group]) under no_grad; composed obs via
                                      # fill_obs_components (U7) with mask gate==1; super().act(composed) stores the composed obs,
                                      # so PPO minibatches never re-run the planner
    def get_policy_feedback(self) -> dict[str, Tensor]   # U1 hook
    def process_env_step(...)         # pop infos[planner_samples_key] into replay; super()
    def update(self, it)              # super().update(); then planner updates (planner frozen during rollout and PPO update)
    def distributed_data_parallel(self); state_dict / load_state_dict (+ "planner", "planner_learner", "replay_state_dict")
    def export_as_onnx(self, obs, extra_obs, export_dir)   # planner.onnx + actor.onnx (+ manifest if M5 accepted)
class ContactPlannerPPO(PlannerAlgoMixin, PPO): pass      # registered
```
The contact-target slot is listed in the policy normalizer's `exclude_components` (A-M3), because the planner
writes it after normalization. Later use: any hierarchical planner + tracker, teacher-command injection.

Relation to the existing estimator (reuse decision). `EstimatorMixin` / `EstimatorAlgoMixin` already do
"upstream net fills policy-obs components, trained by supervision"; A-M5 shares U7 and their list-style keys.
`EstimatorPPO` is not used as the base because ACP differs in four ways that would change its default behavior:
(1) labels come from the future (hindsight, A-M7) and live in a replay buffer, not in the same rollout step's
critic obs; (2) the planner has its own optimizer and 10 updates per iteration (U4), not a term in the PPO loss;
(3) the planner reads its own obs group (`planner`), while `EstimatorMixin` only reads `policy`; (4) a per-env
gate decides replacement, not a random `replace_state_prob`, and the composed obs is stored so PPO minibatches
do not re-run the planner. If a later task needs a replay-trained estimator, `PlannerAlgoMixin` with an MSE
planner module covers it.

### 5.3 SuperMimic Instinct-RL modules

**S-M1. `ReplayStorage`** (`IR/storage/replay_storage.py`, on U5): time × env ring buffer.
```python
class ReplayStorage:
    def __init__(self, num_envs: int, capacity_per_env: int, specs: dict[str, TensorSpec],
                 episode_key: str = "episode_step", storage_dtype: torch.dtype | None = None, device="cpu")
    def add(self, step: dict[str, Tensor]) -> None                       # each (N, ...)
    def sample_windows(self, batch_size: int, length: int, keys: Sequence[str]) -> tuple[dict[str, Tensor], Tensor]
        # (B, length, ...) + valid mask (B, length): mask stops at episode ends
    def sample_starts(self, batch_size: int, history: int, keys: Sequence[str],
                      valid_fn: Callable[[Tensor], Tensor] | None = None) -> dict[str, Tensor]
        # start states with `history` earlier rows rebuilt from the ring (no per-step history copy: saves ≈ 15 GB)
    def __len__(self) -> int; state_dict / load_state_dict
```
Ring cursor and growth logic come from `QueueRolloutStorage` / `buffer_expand` / `buffer_swap`.
Stored per step: single-frame state, ξ, `ref_idx` (int64 (motion id, frame), see U6), executed noisy action `a+ε`,
clean previous action, reward, termination, episode step. Later use: SAC, TD-MPC, Dreamer-style methods.

**S-M2. `WorldModel`** (`IR/modules/world_model.py`, registered). **Written 2026-10-26** (first version; the
SuperMimic algorithm in S-M4 is not written yet). Kept deliberately generic per user emphasis: it knows nothing
about G1, SuperMimic or any reward, and the state layout is the same `OrderedDict[str, tuple]` used by
`obs_format`, so the existing helpers apply to states too. `predict_delta` accepts `"all"`/`"none"`/a per-component
mix; `frame_of_component` records `world`/`root`/`invariant` so heading-invariance is a property of the layout
rather than hardcoded robot math. Measured: 6.0 ms per k=8-step training step at B=512 (≈86k windows/s), 112k
parameters at hidden=(256,256), 44 MB GPU.
```python
class WorldModel(nn.Module):
    def __init__(self, state_segments: OrderedDict[str, tuple], action_size: int, context_size: int,
                 root_pos_component: str, root_quat_component: str,
                 frame_of_component: dict[str, Literal["world", "root", "invariant"]],
                 hidden_sizes: Sequence[int] = (2048, 2048, 2048), nonlinearity: str = "SiLU",
                 normalizer_class_name: str | None = "EmpiricalNormalization", matmul_dtype: str | None = None)
    def forward(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor        # next state (world frame)
    def rollout(self, state0, actions: Tensor, context) -> Tensor                      # (B, k, ...) open loop
    def compute_loss(self, window: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]  # k-step loss (Eq. 1) + per-component errors
```
Predicts increments in the root frame from heading-invariant inputs; normalizers inside, so the loss scale
does not depend on env units. Uses `MlpModel`, `EmpiricalNormalization`. Later use: model-based RL, MPC,
sim-to-sim gap studies.

**S-M3. Policy and critic: reuse, no new network class.** `EncoderDeterministicActorCritic =
DeterministicPolicyMixin + EncoderActorCritic` (one-line registered class). The motion encoder is a
`ParallelLayer` block (`MlpModel` over the K=10 goal components, `output_size=32`, new
`MlpModel(output_nonlinearity="Tanh")` option, default `None` = today); the decoder is the actor MLP; the
critic is `ActorCritic`'s critic on the `critic` group (state, goal, ξ). Export is the existing
`EncoderActorCritic.export_as_onnx` (`0-<encoder>.onnx` + `actor.onnx`) plus `policy_normalizer.npz`.

**S-M4. `SuperMimic` algorithm** (`IR/algorithms/supermimic.py`, registered). Not a PPO subclass; duck-types
the runner interface (`act`, `process_env_step`, `compute_returns`, `update`, `state_dict`,
`load_state_dict`, `distributed_data_parallel`, `train_mode`, `test_mode`, `actor_critic`, `learning_rate`).
```python
class SuperMimic:
    def __init__(self, actor_critic, imagination: ImaginationInterface,           # from env.get_algorithm_inputs()
                 world_model_kwargs: dict, world_model_learner_kwargs: dict, policy_learner_kwargs: dict,
                 critic_learner_kwargs: dict, replay_kwargs: dict,
                 horizon: int = 32, wm_horizon: int = 8, gamma: float = 0.99, lam: float = 0.95,
                 use_critic: bool = True, collection_noise_std: float = 0.1,
                 wm_batches_per_update: int = 10, wm_batch_size: int = 2048,
                 num_imagined: int = 4096, critic_epochs: int = 5, critic_mini_batches: int = 4,
                 updates_per_iteration: int = 2, wm_warmup_iterations: int = 0,
                 policy_grad_value_clip: float | None = 0.1, compile: bool = False,
                 save_replay_in_checkpoint: bool = True, device="cpu", **kwargs)
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1)
    def set_normalizers(self, normalizers)        # U1 hook: imagination normalizes policy obs with the runner's normalizer
    def act(self, obs, extra_obs)                 # deterministic action + N(0, σ²) noise
    def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs)  # replay add; resets use infos["final_obs"]
    def compute_returns(self, last_extra_obs)     # no-op
    def update(self, it) -> tuple[dict, dict]     # updates_per_iteration × (WM fit, BPTT policy step, critic fit via U2)
```
Runner mapping: `num_steps_per_env = 5`, 2 updates per 5 sim steps = UTD 0.4; one runner iteration = 2 paper
iterations, and LR schedules count paper iterations. Later use: any task with a differentiable imagination
interface (U6).

### 5.4 InstinctLab modules

**A-M11. Contact-scheduled motion reference** (user request), following the HOI chain
(`motion_reference_hoi_data.py`, `omomo_motion.py`) exactly:

1. `IL/motion_reference/motion_reference_contact_data.py`
   ```python
   @dataclass class ContactMotionSequence(MotionSequence):         # load-time, [N, T, ...]
       contact_flag: Tensor            # (N, T, B) bool
       contact_pos_w: Tensor           # (N, T, B, 3), clip frame
       next_contact_onset_time: Tensor # (N, T, B) s until the next onset (0 if in contact now), -1 if none within horizon
       next_contact_pos_w: Tensor      # (N, T, B, 3)
       contact_body_names: list[str]
       attrs_with_frame_dim = MotionSequence.attrs_with_frame_dim + (4 fields)   # else silently dropped by the loader
       make_empty(batch_size, num_frames, num_joints, num_links, num_contact_bodies, device)
       make_emtpy_concat_batch(buffer_lengths, num_joints, num_links, num_contact_bodies, device)
   @dataclass class ContactMotionReferenceData(MotionReferenceData):   # sim-side, ProxyArray fields, same 4 names
       make_empty(num_envs, num_frames, num_joints, num_links, num_contact_bodies, device, contact_body_names=None)
       reset(env_ids)        # super().reset(); flags False, onset -1, positions 0 (no "*_mask" names: base reset would set them True)
       mirror_(env_mask, body_mapping: list[int])   # symmetric augmentation of the new fields (swap L/R, negate y)
   ```
   No `ContactMotionReferenceState`: reset states do not need contacts.
2. `IL/motion_reference/motion_files/contact_motion.py` + `contact_motion_cfg.py`
   ```python
   @configclass class ContactMotionCfg(AmassMotionCfg):
       class_type = "{DIR}.contact_motion:ContactMotion"
       contact_body_names: list[str] = MISSING      # 9 bodies: ankles, knees, wrists, elbows, pelvis
       contact_onset_horizon_s: float = 1.5
       contact_keys_required: bool = True           # a clip without contact keys raises
   class ContactMotion(AmassMotion):
       # _read_retargetted_motion_file / _pack_retargetted_motion_sequence: read npz keys
       #   contact_body_names (B,), contact_flag (T, B), contact_pos_w (T, B, 3); reorder to cfg body order
       # onset fields precomputed at load (vectorized next-rising-edge search): runtime stays a nearest-frame gather
       # fill_motion_data: super(), then gather the 4 fields; add env origin to the position fields like base_pos_w
   ```
   Works with `TerrainMotion` through a `TerrainContactMotion(TerrainMotion, ContactMotion)` pairing (metadata
   yaml unchanged) for reference-anchored terrains.
3. Manager: `_prepare_data_class_kwargs` injects `num_contact_bodies` / `contact_body_names` when the
   `make_empty` signature asks for them (same as `num_objects`); `MotionReferenceManagerCfg` gets
   `contact_body_names: list[str] = []` and `symmetric_augmentation_contact_body_mapping: list[int] | None`.
   Mirroring has no hook today: `_symmetric_augment_reference_data` (`motion_reference_manager.py:1062`) edits a
   hard-coded list of base/joint/link fields. Its body moves unchanged into a new
   `MotionReferenceData.mirror_(env_ids, manager)` method and the manager calls `data_buf.mirror_(...)`
   (bit-identical test on BeyondMimic data); `ContactMotionReferenceData.mirror_` calls `super()` and then
   mirrors the 4 contact fields. HOI data can add its own later; one mechanism only.
4. Obs / reward terms (`IL/envs/mdp/observations/contact_reference.py`, `rewards/contact_reference.py`):
   `contact_targets_b(env, asset_cfg, reference_name, body_names, frame="yaw")` → (N, 9·3),
   `contact_target_valid`, rewards `contact_label`, `contact_pos_base`, `contact_pos_global` (A-M6).
5. Writer: A-M8 annotation script adds the contact keys to existing `*retargetted.npz` files (new files,
   originals untouched). Test with the `test/test_dataset_paths.py` `_TerrainStub` pattern.

Later use: contact-conditioned control, contact-aware datasets for any motion task, teleoperation by contacts.

Other InstinctLab modules:

| # | Module | Path | Later reuse |
|---|---|---|---|
| A-M6 | Contact-target tracker interface: obs terms above, `track_planned_contact` reward reading `policy_feedback`. `contact_pos_base` / `contact_pos_global` reuse the math of `link_pos_tracking_square/gauss` (`rewards/motion_reference.py:740,769`) and `motion_reference/utils.py::get_link_position_distance`, with an added contact-validity mask | IL `envs/mdp/` | contact-conditioned control |
| A-M7 | Hindsight contact labeler: trajectory recorder (≤ 500 steps), backward onset scan from contact forces, footprint check, admission rules, emits `infos["planner_samples"]` (§8) | IL `managers/` or `envs/mdp/` | footstep planners, affordance labels |
| A-M8 | Offline clip contact annotation: per-frame contact flags + world positions for the 9 bodies from simulated playback on the matched terrain (forces > 1 N), written as npz keys for A-M11 | IL `scripts/` | contact-aware datasets |
| A-M9 (**later/optional**, only for ACP-G5 or the full benchmark) | Goal-reaching terrain set (ladders, pallets, vault, balance beams, stepping stones, gaps, double stones, discrete platforms, uneven bridge; + distractors, composition; 10 levels), position goal command, move2goal / standatgoal, terrain curriculum. Parkour's `PoseVelocityCommand` is a *velocity* command whose goal is sampled from terrain flat patches `"target"`; ACP needs the goal position itself (+ height for sit/stand/get-up, + yaw). So: factor its flat-patch goal sampling and `pos_command_b` update into a base `TerrainGoalCommand` (`envs/mdp/commands/`), make `PoseVelocityCommand` a subclass (parkour unchanged, output checked identical), and add `PositionGoalCommand` for ACP. The terrain curriculum is a new term following parkour's `tracking_exp_vel` and locomotion's `terrain_levels_vel`, promoting on goal distance + success EMA instead of velocity tracking | IL `terrains/`, `envs/mdp/commands/`, `envs/mdp/curriculums/` | perceptive locomotion / parkour / navigation |
| A-M10 | Stage schedules. **Now:** S1 CSI curriculum + a fixed η value. **Later:** η_obs / η_rew linear ramps from `training_iteration`, stage-3 schedule | IL curricula | staged hand-off from reference to learned commands |
| S-M5 | SuperMimic pure-torch MDP module implementing U6 (reward, termination, obs, goal), also called by the manager terms, so sim and imagination share one formula. Built on `motion_reference/utils.py` difference functions. FK: a small reusable `pytorch_kinematics` chain module in `utils/humanoid_fk.py`, which today only exports FK to ONNX; the exporter then uses the same module | IL task `mdp/` | analytic reward checks, offline relabeling |
| S-M6 | Return-based reset-bin curriculum (50/40/10 mix) writing `_motion_bin_weights`; **subclass** of `BeyondMimicAdaptiveWeighting` (reuses its per-bin `ConcatBatchTensor` counters, buffer lookup and the write into `_motion_bin_weights`; overrides the counter update, termination-based → return-based, and the weighting rule) | IL `envs/mdp/curriculums/motion_reference.py` | alternative bin weighting for any motion tracking task |
| S-M7 | SuperMimic G1 task (Newton/MJWarp) | IL `tasks/shadowing/supermimic/config/g1/` | base for model-based G1 tasks |

### 5.5 Reuse map (what each paper takes from where)

| Need | ACP | SuperMimic | Shared piece |
|---|---|---|---|
| Extra network with own optimizer, DDP, checkpoint | planner | WM, policy, critic | U4 `ModuleLearner` + U3 |
| Replay with checkpointing | planner samples | transitions | U5 `TensorDictBuffer` (A-M4, S-M1 on top) |
| GAE | per-critic masked | imagined rollouts | U2 |
| Env → algorithm objects / algorithm → env data | planned contacts, iteration | imagination, final obs | runner–env channels, U1 hooks |
| Encoder + MLP actor | — (plain MLP tracker) | motion encoder + decoder | `EncoderActorCritic`, `ParallelLayer`, `MlpModel` |
| Multi-critic | V_m + V_fs | single critic | `ActorCritic.critics` + A-M1 |
| Motion data access | contact schedule | goals | `MotionReferenceManager`, M9 |
| Randomization, evaluation, converters | yes (M8 per-clip for the demo; per-category later) | yes | M7, M8, M6 |
| Upstream net writing policy-obs slots | planner → contact targets | imagination → policy obs | U7 (also used by existing `EstimatorMixin`) |
| Attention | cross-attention planner | — | A-M2 next to existing `TransformerHeadModel` |
| Goal command, terrain curriculum | goal-reaching envs | — | `TerrainGoalCommand` factored out of parkour `PoseVelocityCommand`; curriculum after `tracking_exp_vel` |
| Reward / obs formulas | contact position rewards | imagination reward | `motion_reference/utils.py`, `link_pos_tracking_*` |
| Reset-bin weighting | — | 50/40/10 bins | `BeyondMimicAdaptiveWeighting` (subclass) |

## 6. Phases

Order: shared foundation (F), then two tracks. **Recommendation:** run SuperMimic training first. Its data
is ready, its runs take hours instead of days, and it exercises U1–U6, M6–M9 end to end before ACP spends
GPU-days. ACP's CPU-only work (A1–A3) and data work (A4) run in parallel.

### F. Shared foundation (Instinct-RL, CPU only, in marsbrain `~/workspace/isaacSim6/instinct_rl`)

| Phase | Work | Gate |
|---|---|---|
| F0 | M1: `FakeVecEnv`, driver, golden fixtures recorded with the unchanged code of `integrate/ACP-SuperMimic` (Instinct-RL `574c8d5`, 1.0.3) (fixed seeds, CPU, deterministic): per-iteration losses and parameter hash after 3 iterations for `PPO` (1 and 2 rewards), `ActorCritic`, `ActorCriticRecurrent`, `EncoderActorCritic`, `MoEActorCritic`, `EncoderMoEActorCritic`, `WasabiPPO`, `EstimatorPPO`, `TPPO`, `VaeDistill`; save → load → continue equality | Fixtures in their own change; suite passes on `integrate/ACP-SuperMimic` |
| F1 | Defect fixes, one at a time, root cause shown to the user first, each with a regression test: `on_policy_runner.py:73` `importlib.import_module()` without argument; `:65,72` `pop("class_name")` mutates caller config; `:206` final save with `log_dir=None`; `:448` `torch.load` without `map_location`; `rollout_storage.py:344,374,381,388` `QueueRolloutStorage` reward width 1; `estimator.py:35` `minibatch.actor_obs`; `amp_storage.py:80` `self.observations`; `setup.py` missing `regex`, `pyyaml`. Report only: `wasabi.py:308` no discriminator grad clipping; unused code (`act_residual.py:13-14`, `demonstration.py:133`, `dagger_saver.py:74-78`, `VqVae`, `SarsaRolloutStorage`, `TwoStageRunner`) stays (user decision) | Golden suite bit-identical, except a fixture re-recorded with the user's approval when a fix changes a covered path |
| F2 | U1 runner generalization + runner–env channels; U2 GAE function; U7 slot filling (with `EstimatorMixin` moved onto it) | Fake-env tests: call order `act → feedback → step`; iteration across resume; golden suite identical (incl. `EstimatorPPO`) |
| F3 | U3, U4, U5 (+ sidecar replay checkpoints), U6 protocol; M5 if accepted | 2-process `gloo` tests; replay round trip with `weights_only=True`; ONNX vs PyTorch atol 1e-5; DWP re-export identical |
| F4 | InstinctLab side of the channels in the wrapper (delivers ACP-G7); version pin; network I/O snapshot for every task family; 1-iteration train + resume for parkour (WasabiPPO), perceptive shadowing, perceptive HOI (PhysX, `--num_envs 128`, marsbrain); DWP playback and export | All identical / passing; results under `~/Data/instinctlab_logs/` |

### S. SuperMimic track (LAFAN1 only)

| Phase | Work | Done when |
|---|---|---|
| S0 | **DONE 2026-10-26.** Environment: Docker (`instinctlab:3main` on `isaac-lab-base:8c9b995cb`, torch 2.11.0+cu130, driver 580.178.04); the `isaaclab3` conda env was deleted (it could not run this branch) after saving its package lists. LAFAN1-40 conversion (M6) via new `scripts/lafan1_csv_to_instinct.py`; new task `Instinct-SuperMimic-Plane-G1-{,-Newton,-OvPhysx}-v0` (S-M7) on the converted data. | **Measured:** 40 clips, 264,705 frames, 147.1 min (paper: 2.45 h) — matches. Conversion validates (median foot offset 0.8 cm; the repo's existing GMR retargets measure 6.4 cm). Training at **4096 envs: 0.83 s/iteration steady state ≈ 118k env-steps/s, peak 5.07 GB of 24.5 GB** — the paper's env count fits on one 24 GB card with ~19 GB left for replay and BPTT. |
| S1 | Instinct-RL: S-M1, S-M2, S-M3 config, S-M4 | PPO regression unchanged; `SuperMimic` runs on the fake env and a small config |
| S2 | InstinctLab: S-M7 task (obs per §4.1, groups `wm_state`, `sim_context`, `ref_index`), S-M5, M9, S-M6, M7 events (no pushes), rewards (no `undesired_contacts`, joint limit −2), terminations (root height, z-tilt), `final_obs`, MJWarp limits; M8 evaluator | Analytic reward/termination match manager terms to 1e-5; FK matches sim body poses; `final_obs` holds the pre-reset state |
| S3 | World model only | SM-T3; imagined rollout speed vs Newton reported |
| S4 | Full loop on LAFAN1-40 | SM-T1, SM-T2 |
| S5 | Critic / horizon ablation: {critic, no critic} × H ∈ {8, 16, 32, 64} (H=128 only on an 80 GB GPU) | SM-T4 |
| Optional | PPO-Sim baseline; ONNX export and `instinct_onboard` agent; sim-to-sim to PhysX | |

### A. ACP track (module delivery, §3.2)

| Phase | Work | Done when |
|---|---|---|
| A1 | A-M1 per-env multi-critic PPO (part of ACP-G1) | Golden suite identical; 2-critic fake env shows masked value targets, per-group advantage mean 0 / std 1, fixed-LR group keeps its LR, warmup samples give zero actor gradient, checkpoint round trip |
| A2 | A-M2, A-M3, A-M4 (part of ACP-G1) | Unit tests per block, including 2-process `ready()` and replay round trip |
| A3 | A-M5 planners and `ContactPlannerPPO`; export `planner.onnx` + `actor.onnx` (part of ACP-G1) | Fake-env end-to-end: labels → buffer → planner loss decreases on a synthetic target; feedback reaches the env; composed obs stored; the S1 → S2 transition works via `ckpt_manipulator` (S2 → S3 once stage 3 exists); ONNX matches PyTorch |
| A4 | Data (§7.3), all LAFAN1, no downloads; part of ACP-G1. M6: Unitree `g1/` CSV → `*retargetted.npz` (the same conversion SuperMimic's S0 needs, so it is done once); A-M8 contact annotation of all 40 clips on flat ground; A-M11 data class + loader | Contact flags agree with sim contact forces on 2 clips; onset fields agree with M9 lookups; clips play back on the plane |
| A5 | ACP G1-robot task (PhysX, 29 DoF) on the LAFAN1 flat-ground scene; ACP-G1 close-out (all delivered modules tested together): obs groups `policy`, `planner`, `contact_source` (N,1), `critic`; `env_type` (N,2) and `critic_goal` only if
the S2 run actually trains V_fs (with LAFAN1-only there are no goal envs, so by default the demo runs one
critic and A-M1's multi-critic path is covered by unit tests instead); A-M6, A-M7, M7 + ACP randomization, terminations documented. A-M10 needed now: the S1 CSI curriculum and a **fixed** η (no ramps — a single stage-2 η value, recorded in the config). No A-M9 terrain zoo, no stage-3 schedule | 128-env smoke on marsbrain; `infos["planner_samples"]` contract (§8) checked by a test |
| A6 | Demo stage 1 → 2 on the LAFAN1 clips | ACP-G2, ACP-G3 |
| A7 | Planner vs MLP | ACP-G4 (report) |
| A8 | Export and onboarding check | ACP-G6 |
| Optional | Only if taken up later, as a separate decision: box-mesh variant (real scan geometry, §3.2) and the full benchmark. ACP-G5 (stage 3) is out of scope under LAFAN1-only | report only |

M6, M7 and M9 are used by the demo (M6 path check, M7 randomization, M9 onset cross-check); M8 is built now in
its per-clip form only. A-M9 and A-M10's stage-3 ramps are deferred and are built only if the full benchmark or
ACP-G5 is taken up.

## 7. Data

### 7.1 Sources already on hosts

- marsbrain `~/Datasets/`: `LAFAN1_Retargeting_Dataset/` (raw), `UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz/`
  (77 clips, GMR retarget; not the paper's Unitree retarget), `mixamo_adobe_motions/`,
  `NoKov-Marslab-Motions-instinctnpz/` (box climbing, dive roll, knee climb — **not used** under LAFAN1-only;
  kept as the cheapest way to add a real-terrain variant later).
- marsbrain cannot reach Hugging Face. Everything the demo needs is already there, so nothing is downloaded for
  ACP, and the 40 Unitree `g1/` CSVs are confirmed present on marsbrain, so this plan has no download step. If a
  clip ever turns out to be missing, that is a blocker to report; the fallback would be to fetch it on `aorua`.

### 7.2 SuperMimic (LAFAN1 only)

1. LAFAN1 Unitree G1 CSVs (HF `lvhaidong/LAFAN1_Retargeting_Dataset`, `g1/`, 30 fps, root pos, xyzw quat,
   29 joints) → instinct npz with M6; the 40 paper clips (2.45 h): dance, fallAndGetUp, fight,
   fightAndSports, jumps, run, sprint, walk. **S0 first checks whether `~/Datasets/LAFAN1_Retargeting_Dataset/g1/`
   holds all of them** (marsbrain has the raw dataset and 77 GMR-retargeted npz, which are a *different*
   retarget and not a substitute — the paper's numbers are tied to the Unitree retarget). The card declares no license on top of LAFAN1's CC BY-NC-ND:
   research use only, no redistribution of converted files.
2. Check retargets against the InstinctLab G1 asset; run the paper's feasibility filter (App. E).

### 7.3 ACP data (LAFAN1 only)

The ACP authors published nothing (project page "Code TBD"; the retargeting tool is unreleased). Under the
LAFAN1-only decision the demo uses the same 40 clips as SuperMimic — already on marsbrain in two forms:
`~/Datasets/LAFAN1_Retargeting_Dataset/g1/` (40 Unitree-retargeted CSVs, 300 MB with `h1`/`h1_2`) and
`~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz/` (77 GMR-retargeted npz).

Use the **Unitree `g1/` CSVs** converted with M6, for consistency with SuperMimic's reproduction target (the
paper's numbers are tied to that retarget). The 77 GMR npz are usable for ACP's demo alone if the conversion is
slow, but then ACP and SuperMimic run on different retargets and the results note must say so.

Clip → role in the demo:

| LAFAN1 clips | Count | Role |
|---|---|---|
| `fallAndGetUp1-3_subject*` | 6 | multi-contact: hands, knees, pelvis on the ground |
| `ground1/2_subject*` | 5 | sitting / lying transitions (also in the paper's "get up" family) |
| `dance1/2_subject*` | 8 | upright whole-body, occasional hand and knee contacts |
| `fight1_*`, `fightAndSports1_*` | 5 | hand contacts, fast transitions |
| `jumps1_*` | 3 | flight phases (contact-free intervals) |
| `walk1-4_*`, `run1/2_*`, `sprint1_*` | 13 | foot-only baseline |

What each dropped paper category would have needed (recorded so a later extension is cheap):

| Paper category | Paper source | Would need |
|---|---|---|
| Box climbing | OmniRetarget | HF download (431 MB, MIT); NoKov `60cmBoxes`/`kneeClimbStep` already on marsbrain works too |
| Stairs, ladder, sit | Mixamo | Mixamo FBX + terrain-constrained retargeting (Holosoma, Apache-2.0); no ladder motion exists anywhere on the host |
| Vault | Kimodo | Kimodo generation (weights only, no clip collection) or NoKov `rollVault11` (already there) |
| Sparse terrain | AME rollouts | re-run an AME/AME-2 teacher and harvest rollouts, or AMASS KIT `step_stones*` (already there) |

Consequences for the build:
- **No downloads, no retargeting, no terrain zoo.** A-M9 (goal-reaching terrain set) and stage 3 leave scope.
- **Contact labels must be produced by us** (A-M8): no npz on marsbrain carries them (every `*retargetted.npz`
  has 5 keys: `framerate`, `joint_names`, `joint_pos`, `base_pos_w`, `base_quat_w`).
- Annotate **all 40 clips** once (cheap, it is a sim playback), so any subset can be used later.
- Licenses: LAFAN1 upstream is CC BY-NC-ND 4.0; derived files stay research-only and are not redistributed.

## 8. Cross-repo contracts

- Instinct-RL `VecEnv`: `set_training_iteration`, `set_policy_feedback`, `get_algorithm_inputs`,
  `infos["final_obs"]` (same group/format as observations). The wrapper returns `{}` from
  `get_algorithm_inputs()` for tasks without extras; an env term reading a missing feedback key raises.
- ACP `infos["planner_samples"]`: dict with shared leading dim K (K may be 0): `planner_obs (K, D)`,
  `target_pos (K, 9, 3)`, `target_valid (K, 9)`, `partition (K,)` int, `quality (K,)`, `buffer_id (K,)` int.
  The env reads `training_iteration` only for the later η ramps (the demo uses one fixed η) and
  `policy_feedback["planned_contacts"]` / `["planned_occupancy"]` for `track_planned_contact`.
- SuperMimic: `get_algorithm_inputs()["imagination"]` implements U6.
- Motion data: `*retargetted.npz` gains optional keys `contact_body_names`, `contact_flag (T, B)`,
  `contact_pos_w (T, B, 3)` (clip frame, meters). Existing loaders ignore them.

## 9. Computation resources

### 9.1 Hosts

| Host | GPU | RAM | Disk | Role |
|---|---|---|---|---|
| marsbrain | RTX 4090 D, 24 GB | 62 GB | 117 GB free on `~` | **everything**: build, tests, SuperMimic training, ACP demo |
| aorua | RTX 5070 Ti, 16 GB | 15 GB | — | optional: editing, docs, small CPU runs. Not required. |

marsbrain (checked 2026-10-26, SSH reachable again): 24 cores, torch 2.10.0+cu128, conda env `isaaclab3`
(Python 3.12.13), `instinct_rl`, `instinctlab` and `isaaclab` editable-installed from `~/workspace/isaacSim6/`,
`pytorch_kinematics` 0.10.0 present, datasets on `/localhdd/Datasets` (`~/Datasets`), 117 GB free on `~`.

Setup done 2026-10-26 (user decisions):

| Repo on marsbrain | Branch | HEAD | State |
|---|---|---|---|
| `instinctlab` | `integrate/ACP-SuperMimic` | `dfa08ee` | clean; the previous 44 modified files are in `stash@{0}` |
| `instinct_rl` | `integrate/ACP-SuperMimic` | `574c8d5` (1.0.3, local `main`, complete override) | clean; the previous 7 modified files are in `stash@{0}` and were already present in `574c8d5` |
| `IsaacLab` | unchanged | `6a7acb032` | not modified |

Bases: `instinctlab` is based on local `migrate/isaaclab3-main` (`dfa08ee`), which contains the old working tree
(`94f8c55`) and `main` (`ba28d3d`) as ancestors — it is 54 commits newer than what marsbrain ran before, and it
carries the Newton / PhysX backend work SuperMimic needs. `instinct_rl` was overridden outright with local
`main` per user decision. Backups kept: `/tmp/g1.png.modified-backup`, `/tmp/irl_dirty_backup.patch`, and the
two stashes. Do not drop the stashes without asking.

One env note: `setup.py` on `574c8d5` pins `torch==2.11.0` while the `isaaclab3` env has 2.10.0+cu128. The
editable install already in place works; do not `pip install -e .` into that env without resolving the pin.

### 9.2 GPU-hours on marsbrain (1× 4090 D)

"4090-h" = GPU-hours on that card. All estimates include ×1.3–1.5 margin.

**SuperMimic, LAFAN1 only** (unchanged from revision 2)

| Item | 4090-h |
|---|---|
| Dev, smoke tests, throughput probe, WM-only run (S0, S3) | ≈ 15–20 |
| Main run to SM-T1 (paper 1.2 H100-h; 4090 D bf16 ≈ 1/4–1/6 of an H100) | ≈ 4–8 per run |
| Main runs incl. 2–3 reruns / seeds | ≈ 15–25 |
| **Core (SM-T1 … SM-T3)** | **≈ 30–45** (≈ 2 days) |
| Ablation S5: 8 runs (4 non-converging capped at 12–20 h, 4 converging at 4–8 h) | ≈ 65–110 |
| (rerun allowance for failed ablation runs) | included below |
| Ablation reruns | ≈ 20–35 |
| **With ablation (SM-T4)** | **≈ 115–190** (≈ 5–8 days) |

**ACP, module delivery + demo** (this revision; was ≈ 1,200 for the full reproduction)

| Item | 4090-h |
|---|---|
| A1–A3 (Instinct-RL modules, CPU/unit tests, fake env) | ≈ 3–5 (mostly CPU) |
| A4 data: M6 CSV conversion (shared with SuperMimic S0), A-M8 annotation of all 40 clips, A-M11 loader, playback checks (mostly CPU) | ≈ 8–12 |
| A5 task build + 128-env smoke tests, debugging | ≈ 15–25 |
| A6 demo stage 1 (tracker; 4,096 envs on one GPU — 1/9 of the paper's S1 count, sized for 24 GB and a ~20-clip LAFAN1 subset) | ≈ 20–30 |
| A6 demo stage 2 (planner supervision + co-training, shorter) | ≈ 10–15 |
| A7 planner-vs-MLP (2 short runs) | ≈ 8–12 |
| A8 export/onboarding, evaluations, reruns after bugs | ≈ 10–15 |
| **Total ACP** | **≈ 74–114** (≈ 3–5 days) |

**Combined:** ≈ 115–190 (SuperMimic core+ablation) + 74–114 (ACP) ≈ **190–305 4090-h**, i.e. **8–13 days**
of continuous GPU time. Assume ~50 % duty cycle (agent iterations, re-exports, evaluation pauses, human review),
so ≈ 2–4 weeks of calendar time on a machine used only by this work. SuperMimic runs first.

If the full ACP benchmark is chosen later (11 categories × 3,000 episodes, 3 seeds, three stages at paper env
counts), the earlier estimate was ≈ 1,200 4090-h total; subtracting the ≈ 74–114 above leaves ≈ 1,090
additional (≈ 6–7 weeks on this one GPU), dominated by stage 3 (≈ 300) and its MLP ablation (≈ 420). That needs
the 4-GPU node in §9.3 to be practical.

Storage: datasets already on marsbrain (no new download). Each SuperMimic run keeps its models plus its 2 most
recent replay sidecars (≈ 2 × 4 GB) — sidecars of *finished* runs are deleted, only the live/current run holds
them. So the peak is ≈ 10 GB live + models for ~12–15 runs, ≈ 35 GB; ACP demo ≈ 5 GB. Fine within 117 GB, but
prune logs as runs are retired.

### 9.3 More GPUs

Not needed for the reduced scope. If the full ACP benchmark or several SuperMimic seeds run in parallel later,
4 × 24 GB on one node with ≥ 256 GB RAM would cut the ACP benchmark from ≈ 6–7 weeks to ≈ 2, while one 80 GB
GPU would allow SuperMimic H=128 and full-size runs.

## 10. Risks

- ACP: η schedule numbers, iteration counts and terminations are not in the paper; our choices may explain
  differences. Record every choice in the run config and the results note.
- ACP: reduced scope, LAFAN1-only. The demo is flat-ground, so the attention-over-terrain mechanism is
  exercised but not validated (a plane scan has almost no structure). Module-level tests carry the reproduction
  claim; the results note must state this plainly, and the optional box variant (§3.2) is the mitigation.
- ACP: the composed-obs design needs the planner frozen between `act` and the PPO update of the same iteration.
- ACP: excluded normalizer components must be fixed before the first run; changing them invalidates statistics.
- ACP: contact labels are ours (A-M8), so a systematic labelling error (force threshold 1 N, terrain mismatch)
  would propagate into the planner targets. Validate on 2 clips against sim forces before training.
- Setup is done: both repos are on `integrate/ACP-SuperMimic` and clean (`instinctlab` @ `dfa08ee`,
  `instinct_rl` @ `574c8d5`). Two things to remember: InstinctLab's old 44 dirty files are in `stash@{0}` and
  Instinct-RL's 7 in `stash@{0}` (they were already in `574c8d5`); do not drop those stashes without asking.
  Line numbers in this document were read from the reference clones, which are now the same commits, so they
  apply as written.
- ACP: V_fs uses planner predictions; with feedback set before `step`, obs t+1 sees p̂_t (one-step lag).
  Alternative: the mixin appends p̂ to the V_fs input itself. Decide in A3.
- ACP: the next-contact definition while a body is already in contact (we use onset time 0 and the current
  position) must be checked against the paper text in A4.
- SuperMimic: manager-based env step overhead may be high; measure in S0; fallback is a direct env inside InstinctLab.
- SuperMimic: MJWarp contact/constraint overflow during falls; runtime CoM changes may destabilize MJWarp.
- SuperMimic: retarget/asset mismatch between Unitree LAFAN1 and the InstinctLab torso-base G1.
- Both: replay sidecars make checkpoints large; pruning keeps only the last two.
- Both: distributed training is not yet validated on the upgraded InstinctLab branch.
- Both: Isaac Lab `main` imports link inertia correctly while EA did not; old EA checkpoints are not comparable.

## 11. Open items (ask the user)

All decisions from the earlier rounds are settled and recorded in §1: marsbrain as the only host; branch
`integrate/ACP-SuperMimic`; Instinct-RL overridden with local `main`; InstinctLab based on the current `main`
line; dirty files stashed; local scratch worktrees deleted; both papers under LAFAN1 only; ACP reduced to
modules + one demo; GRAIL dropped; export manifest skipped; last two replay sidecars kept.

Nothing is blocking. Two things will be raised when they come up, rather than guessed:

1. If `~/Datasets/LAFAN1_Retargeting_Dataset/g1/` turns out not to hold all 40 clips, or marsbrain cannot reach
   Hugging Face for the rest, that is a blocker to report — not a silent switch to `aorua`.
2. If the LAFAN1-only demo leaves the planner's terrain attention untested in a way you want covered, the
   optional box-mesh variant (§3.2) or the full benchmark (§9.2) are separate decisions at that point.

## 12. Agent workflow (planning and evaluation)

To save tokens of the main agent, spawn OpenCode subagents with **DeepSeek V4.1 Flash 1017**
(model id `deepseek/deepseek-flash`) for work that is reading, searching, running, or summarizing:

| Delegate to `deepseek/deepseek-flash` subagents | Keep in the main agent |
|---|---|
| Paper fact extraction and cross-checks against TeX/PDF | Design decisions, interfaces, contracts (§5, §8) |
| Codebase surveys (where is X, who calls Y) | Defect root-cause explanations to the user |
| Running golden tests, smoke tests and parity checks; returning pass/fail and diffs | Code changes that touch invariants (§2) |
| Parsing TensorBoard logs and evaluator CSV/JSON into result tables per §3 targets | Accepting or rejecting a gate |
| Monitoring long runs on marsbrain (GPU memory, steps/s, NaNs) | Questions to the user (§11) |
| First-pass reviews of each phase against this plan | Final results note |

Each subagent prompt states: read-only or allowed edits, exact paths, host, and the expected output format
(short markdown table). Independent subagents run in parallel. Revision 2 used three such subagents
(Instinct-RL interface survey, motion-reference survey, public data search); revision 3 used one more to check
§5 against the code for missed reuse and interface mismatches.

## 13. References

- ACP: `~/workspace/paper-intro-videos/AttnContacts/{FACTS.md,paper/src/main_arxiv.tex,paper/paper.txt}`;
  project page https://leggedrobotics.github.io/ACP/ (code TBD).
- SuperMimic: `~/workspace/paper-intro-videos/SuperMimic/{FACTS.md,paper.txt}`.
- `zita-ch/supermimic_vibed` (MIT): LAFAN1 return 0.829, 0/40 failed, 1.47 h on 1× RTX PRO 6000.
  `Motphys/Uni-SuperMimic` (Apache-2.0): LAFAN1 0.827, 0/40 failed. `lucidrains/attention-contact-planning` (MIT, WIP).
- Data, used: LAFAN1 only, already on marsbrain (`~/Datasets/LAFAN1_Retargeting_Dataset/g1/`, 40 Unitree CSVs
  confirmed present). HF `lvhaidong/LAFAN1_Retargeting_Dataset` is the source only if any clip is missing.
- Data, **not used** now (kept for a later extension): HF `omniretarget/OmniRetarget_Dataset` (the paper's box
  source; NoKov covers it), `amazon-far/php_parkour`, `amazon-far/holosoma` (would retarget Mixamo
  stairs/ladder/sit), `nv-tlabs/kimodo`, AMASS KIT stairs / stepping stones. GRAIL is **not** a paper source
  and is not used.
- marsbrain state, checked 2026-10-26: `~/workspace/isaacSim6/{instinctlab,instinct_rl,IsaacLab}`; conda env
  `isaaclab3` (py 3.12.13, torch 2.10.0+cu128); `pytorch_kinematics` 0.10.0; datasets under
  `/localhdd/Datasets` (`~/Datasets`); no contact labels in any `*retargetted.npz`; no Mixamo content beyond a
  single `Dive_Roll.bvh/fbx`; no ladder motion anywhere.
- InstinctLab pieces: `motion_reference/` (`MotionReferenceManager` `_prepare_data_class_kwargs` :601-626,
  `motion_reference_hoi_data.py`, `motion_files/omomo_motion.py`, `amass_motion.py` `fill_motion_data`
  :250-413, `terrain_motion.py`), `tasks/shadowing/beyondmimic/`, `tasks/shadowing/perceptive/`,
  `terrains/trimesh/mesh_terrains.py::motion_matched_terrain`, `monitors/monitors.py`, `utils/humanoid_fk.py`,
  `scripts/motion_matched_metadata_generator.py`.
- Instinct-RL gap survey for ACP: `~/.opencode/plan/INSTINCT_RL_UPGRADE_PLAN.md` §3.
- InstinctLab upgrade plan: `~/workspace/instinctlab_ws/instinctlab/agents/UPGRADE_TO_3main_LMUOpus.md`.
