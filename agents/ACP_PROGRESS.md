# ACP reproduction progress (as of 2026-11-12)

Reproduction of ACP (Attention-based Contact Planning, arXiv:2610.31608) on the G1, PHP and DWP terrain-contact
data. Stage 1 trains the contact-conditioned tracker, stage 2 the contact planner and the hand-off from reference
targets to planner targets, stage 3 is not started (plan: `ACP_STAGE3_PLAN.md`, decisions D1-D15 open).

Success = a clip reaches its end from the given start phase without a failure termination; 206 clips, friction 1.0,
no randomization (`scripts/acp/evaluate.py`). "From phase 0" (the clip start) is the main number.

## Stage 1: tracker

Best setup: old rewards, old critic, BeyondMimic adaptive start sampling, Conv2d depth CNN (branch
`exp/acp-s1-beyondInit`, `45a959a`). With all contact targets: 88-95 % from phase 0 across hosts and seeds
(95.1 % at 12k, `marsrepro/model_10000` 95.6 %).

Rejected: the paper's Table-3 rewards with its V_m critic (stall at 0-1 %); the paper critic alone on the best
setup (66-82 %, slower).

Footprint tracker (branch `exp/acp-s1-beyondInit-footprint`, `d07c95d`, merged into `feat/acp-stage2` as
`c61e319`): contact targets zeroed outside the planner scan (+-0.8 x +-0.5 m), because the planner can only output
targets inside it. Reason: the all-targets tracker fell from 93 % to 4 % when its targets were cut to the scan.

| run | 2k | 4k | 6k | 8k | 10k | 12k | 14k |
|---|---|---|---|---|---|---|---|
| fp_seed42 | 44.7 | 50.5 | 85.9 | 74.3 | 89.8 | 92.7 | 88.3 |
| fp_seed2 | 41.3 | 53.9 | 76.2 | 68.4 | 76.7 | 82.0 | 77.7 |
| fp_seed1 (dropped) | 39.8 | 53.9 | 68.0 | | | | |

Weakest category from phase 0: DWP (20-40 %). Stage-1 training is stopped (seed42 at model_14000, seed2 at
model_15000).

## Stage 2: planner and hand-off

All runs start from `fp_seed42/model_6000` (85.9 % from phase 0 with reference targets), 5000 iterations, seed 42.
Hand-off (paper Eq. S1, timing ours): planner targets reach the tracker from iteration 500 over a 2000-iteration
ramp, the reward moves to them from 1000.

Fixes on the way: hindsight labels count terrain contacts only (`16d48cc`; self-contacts had labeled the pelvis);
labeler reference lookup order (`9548460`). Labels and reference contacts agree except ~14 % of robot footfalls,
inherent to hindsight labeling.

Success from phase 0, planner targets / reference targets:

| variant (task suffix) | 1k | 2k | 3k | 4k | 5k |
|---|---|---|---|---|---|
| paper (`-G1-v0`) | 1.9 / 80.1 | 7.3 / 57.8 | 17.0 / 42.7 | 17.5 / 31.6 | 14.6 / 29.6 |
| depth planner (`-Depth-`) | 1.9 / 71.4 | 16.0 / 54.4 | 18.0 / 31.6 | 23.3 / 30.1 | 18.4 / 29.1 |
| reference-contact labels (`-RefLabels-`) | 3.9 / 78.6 | 4.4 / 72.3 | 15.0 / 51.0 | 16.5 / 55.3 | 16.5 / 54.4 |
| RefLabels + late hand-off (from 2000) | 3.4 / 82.5 | 1.5 / 87.4 | 1.9 / 82.5 | 20.4 / 67.5 | 20.4 / 53.4 |
| adaptive start sampling (`-AdaptiveStart-`) | 2.4 / 80.1 | 9.2 / 57.8 | 17.0 / 45.2 | 18.5 / 33.5 | 16.5 / 32.5 |
| RefLabels + adaptive start | 2.9 / 91.8 | 6.8 / 83.0 | 19.9 / 60.7 | 18.9 / 49.0 | 16.0 / 52.9 |

From phase 0.5 the same checkpoints reach 75-85 % with planner targets, but that start skips the obstacle of most
step clips (low-step: 100 % from 0.5, 20-45 % from 0).

### Findings

- Every variant ends at 15-23 % with planner targets. Failed clips stop at a median 32-38 % of the clip, at the
  obstacle, in every run and checkpoint.
- The tracker also loses its reference-target skill during stage 2 (85.9 % to 30-55 %), which caps any planner fix.
  RefLabels and the late hand-off slow this loss.
- Swap test at model_5000 (`PLANNER_MIX`, `cca6461`), success from phase 0, RefLabels + adaptive start / paper:
  planner targets 16.0 / 14.6 %, reference 52.9 / 29.6 %; reference targets for the feet only 47.6 / 25.7 %;
  reference targets where the planner has none 45.6 / 21.4 %; reference positions where both have a target
  22.8 / 12.1 %. The gap comes from foot targets the reference has and the planner misses, not from the positions
  of the targets it gives.
- A lower occupancy threshold (0.35 / 0.2 / 0.1) raises foot recall from 0.62 to 0.82 but not success
  (15-22 %): the extra low-confidence targets are badly placed.
- With reference labels the planner still recalls only ~60 % of reference foot targets (precision 0.92-0.95).

### Missed foot targets (`FOOTHOLD_LOG=1` + `scripts/acp/analyze_footholds.py`, model_5000, from phase 0)

Recall of the reference foot targets the tracker would get (inside the footprint), paper / RefLabels + adaptive:

- all 75 / 59 %; ground 80 / 57 %; obstacle (> 0.08 m above the clip's ground) 59 / 69 %;
- obstacle by category: low-climb 34 / 91 %, high-step 48 / 20 %, low-step 76 / 58 %, high-climb 86 / 83 %;
- contact onset within 0.3 s 70 / 42-48 %, 0.6-1.0 s ahead 87 / 74 %; targets behind the base (x < 0) 47-56 / 34-45 %;
  distance to the scan edge makes no difference;
- failed clips, last second before the failure 58 / 46 %, elsewhere 79-82 / 62-63 %. 139 / 141 of the failed clips
  have an obstacle foot target in their last second; the planner misses it on every step of that second in 15.

On its own training labels (replay, 100k samples) the paper planner reaches recall 96.5 %, precision 90 %, xy error
0.075 m. The RefLabels planner reaches only 77 / 82 %: the reference's contact timing is hard to predict from the
robot's state. Obstacle-height labels are rare in the paper run's replay (target z >= -0.6 m in the body-yaw frame:
6.4 % of labels, against 11.9 % of reference targets on the evaluation rollouts).

Two label properties that differ from the tracker's targets:

- Labels use the body link origin (robot or reference), the tracker's targets the annotated terrain contact point:
  reference-source labels sit 0.067 m higher (median) and 0.05 m away in xy. Robot labels are 0.13-0.14 m away in xy
  (17-21 % beyond 0.25 m).
- Only episodes that start within the first 10 % of the clip and reach 90 % are labeled. After the hand-off the label
  rate falls from ~370 to ~100 samples per step and the height termination rises from 0.11 to 0.33, so the planner
  learns from successful crossings only and has no labels for the states just before a failure.

### Training curves of the stage-2 runs (TensorBoard, all six variants alike)

- The replay (100,000 FIFO) is full from iteration 100 on; labels arrive at 7-10k samples per iteration before the
  hand-off (370-430 per step), so the planner only ever sees the last 10-15 iterations of the tracker's behaviour.
- During the gate ramp (500-2500) the label rate falls to 1.3-3.5k per iteration (20-130 per step) and the
  admitted episodes from ~0.9 to 0.1-0.3 per step; `base_pos_too_far` rises from 0.11 to 0.30-0.37 and
  `dataset_exhausted` (training-time success, random start phase) falls from 0.87-0.89 to 0.59-0.67.
- The planner loss stops improving at ~3.2 once the hand-off starts (the late hand-off reaches 2.95 at 2000 before
  its ramp, then 3.26-3.38). The KL-adaptive learning rate stays at 1e-4 or below.

### Next experiment (launched 2026-11-12 21:10 CST, `95050ca`, GPUs 0-2 of r1522, from `fp_seed42/model_6000`)

Both effects above cut the planner off from the labels of the good pre-hand-off tracker. Two config knobs, three
runs:

| run | task | change |
|---|---|---|
| `acpS2fp_partialHandoff_seed42` | `-PartialHandoff-` | both ramps stop at eta_max = 0.5: every step keeps the reference targets with probability 0.5, half of the reference-contact reward stays |
| `acpS2fp_bigReplay_seed42` | `-BigReplay-` | replay 1,000,000 samples (~4 GB on the GPU), 100-400 iterations of labels |
| `acpS2fp_partialHandoffBigReplay_seed42` | `-PartialHandoff-BigReplay-` | both |

Seed-2 replicates of all three (`acpS2fp_*_seed2`, containers `acp_s2fp_*_s2`) run on gthintel218 (r1611,
manip_3:4-7, GPUs 0-2 there), launched 2026-11-12 21:32 CST by `~/workspace/isaacSim/acp_partial_launch_seed2.sh`.

Evaluate with `scripts/acp/evaluate.py` (planner and reference targets, phases 0 and 0.5, `EVERY=1000`) against
the paper variant's 14.6 / 29.6 % at model_5000. Then the hand-off question is whether a tracker that keeps its
reference skill under a partial hand-off also crosses obstacles with planner targets alone.

### Results of the partial hand-off / big replay runs (from phase 0, seed 42 / seed 2)

Planner targets at 3k: partial hand-off 2.9 / 6.8 %, big replay 18.0 / 18.9 %, both 4.4 / 4.9 % (paper 17.0 %).
Reference targets at 3k: 71.8 / 72.3 %, 38.8 / 38.8 %, 69.9 / 83.0 % (paper 42.7 %). The partial hand-off keeps the
tracker's reference skill but the tracker never learns to work on planner targets alone; the big replay changes
nothing. (The first evaluations of the partial hand-off runs mixed in reference targets on half the steps,
`528fc70`; those results are in `eval_gate0.5/` of the run dirs, 65-72 % at 2k.)

### Step 0: how much missing foot targets hurt the stage-1 tracker (`FOOT_TARGET_DROPOUT`, `9dbd1e4`)

`fp_seed42/model_6000`, reference targets, success from phase 0, foot targets withheld at random:

| withheld | per contact | per step |
|---|---|---|
| none | 84.0 % | |
| 10 % | 66.5 % | |
| 25 % | 41.3 % | 62.6 % |
| 50 % | 15.5 % | |

The planner's foot recall is ~75 %, so even perfectly placed planner targets would leave the tracker near 40 %. The
tracker is fragile to missing foot targets; the swap test (45.6 % with the reference filling the planner's gaps)
agrees.

### Run A: labels from all episodes (`Instinct-ACP-Planner-LabelAll-G1-v0`, `9dbd1e4`)

Episodes of any start phase are recorded (2400 slots), and the ones that end before the admission end phase are
labeled up to their last step with a complete lookahead. Seed 42 on gthintel201 GPU0 from 2026-11-13 05:32 CST, seed
2 on gthintel218 GPU1 from 06:04 CST. About 2,000 samples per step (paper ~370), 240-270 of them from failed episodes
after the hand-off; the 100k replay holds about 2 iterations of labels.

Success from phase 0, seed 42 / seed 2, planner targets at 1k-5k: 1.0/7.8/17.5/17.0/19.4 % and
4.4/10.7/13.1/18.0/18.9 %; reference targets 78.6/59.7/35.4/34.0/20.9 % and 77.7/54.4/36.9/38.4/30.1 %. The same as
the paper planner (14.6 / 29.6 % at 5k); failures stay at the obstacle (median progress 0.34-0.36). Labels of the
failed approaches do not help.

### Conclusion so far

No planner-side change (reference labels, late or partial hand-off, adaptive start sampling, 10x replay, labels from
all episodes) moves stage-2 success with planner targets beyond 15-20 %. The bottleneck is the tracker's dependence on
complete foot targets (step 0). Proposed next: fine-tune the stage-1 tracker with foot targets withheld per contact
(e.g. 25 %), check it with `FOOT_TARGET_DROPOUT`, then rerun stage 2 from it.

### Comparison with InstinctMJ's ACP branch (2026-11-13) and the next setup

Differences that bear on the stage-2 ceiling (details in the conversation of 2026-11-13):

1. Our stage-1 contact rewards are sharp (std 0.15 m); the paper's planned-contact term that replaces them in stage
   2 is nearly flat (sigma 1 m: 0.99 at 0.1 m error). A full reward hand-off removes the contact guidance.
2. Hindsight labels used the body origin; the dataset's contact targets are terrain contact points.
3. One 100k FIFO replay, filled mostly by the clips that succeed; InstinctMJ keeps one ring per clip and samples
   balanced over terrain categories, then clips.
4. Our hand-off is about twice as fast (InstinctMJ: planner targets from 1000 over 4000, reward from 3000 over 4000,
   10k iterations).
5. Our tracker sees a depth image instead of the paper's 9 x 7 height scan (user decision).

Also from 2026-11-13: all ACP tasks use the 196 PHP clips (`metadata_php.yaml`); the 10 DWP clips are left out
(stage-1 fails 6-8 of them at every checkpoint; 8 of the 171 stage-2 failures).

Stage-1 fine-tunes, 196 PHP clips, success from phase 0 (all foot targets / 25 % withheld):

| run | 14000 | 16000 |
|---|---|---|
| parent `fp_seed42` model_12000 | 99.5 / 46.4 % | - |
| `acpS1_php_control_seed42` (no dropout) | 85.7 / 44.9 % | 93.4 / 39.3 % |
| `acpS1_php_footDrop25_seed42` | 82.7 / 61.7 % | 89.3 / 86.2 % |
| `acpS1_php_footDrop25_seed2` | 76.0 / 68.4 % | 86.2 / 76.0 % |

The dropouts trade a few points of clean-target success for 30-40 points of robustness to missing foot
targets. Continuing the fine-tune also recovers most of the clean-target loss the plain continuation showed at 14000.

Running (gthintel218, r1647, from `fp_seed42/model_12000`, 4000 iterations to model_16000, `3061521`):
`acpS1_php_footDrop25_seed42` / `_seed2` (`Instinct-ACP-Tracker-FootDropout-G1-v0`, foot targets withheld with
p 0.25 per contact) and `acpS1_php_control_seed42` (`Instinct-ACP-Tracker-G1-v0`). Then stage 2 with
`Instinct-ACP-Planner-V2-G1-v0` (`21d0cdd` + instinct_rl `6d25f66`: items 1-4 above; reward hand-off capped at 0.5)
from each, 10k iterations.

### Stage-2 V2 evaluation (2026-11-14)

ACP stage-2 V2 checkpoint evaluation (0-10k sweep), booking r1658, 3 runs (`acpS2v2_footDrop_seed42`,
`acpS2v2_footDrop_seed2`, `acpS2v2_control_seed42`). Success rate (%) at start phase 0.0:

| iteration | footDrop_seed42 planner | footDrop_seed42 reference | footDrop_seed2 planner | footDrop_seed2 reference | control_seed42 planner | control_seed42 reference |
|---|---|---|---|---|---|---|
| 0 | 2.0 | 92.3 | 0.0 | 89.3 | 0.0 | 92.9 |
| 1000 | 1.0 | 95.9 | 5.1 | 89.3 | 1.5 | 95.9 |
| 2000 | 10.7 | 90.3 | 5.1 | 90.8 | 2.5 | 93.9 |
| 3000 | 4.6 | 78.1 | 7.6 | 88.8 | 9.2 | 85.2 |
| 4000 | 12.8 | 80.6 | 5.6 | 70.9 | 8.2 | 81.6 |
| 5000 | 14.3 | 58.7 | 16.3 | 55.6 | 13.8 | 59.7 |
| 6000 | 16.3 | 48.5 | 18.4 | 50.5 | 17.3 | 54.6 |
| 7000 | 18.4 | 46.9 | 18.9 | 59.7 | 20.4 | 57.6 |
| 8000 | 19.9 | 56.6 | 21.4 | 46.9 | 19.9 | 49.0 |
| 9000 | 21.4 | 49.5 | 24.0 | 43.9 | 21.9 | 44.4 |
| 10000 | 22.4 | 48.5 | 16.3 | 41.3 | 17.9 | 41.3 |

Start phase 0.5 planner success at 10k: footDrop_seed42 88.8 %; footDrop_seed2 88.8 %; control_seed42 89.8 %.

### Stage-2 V2 diagnostics (2026-11-15/16, offline scripts on `acpS2v2_footDrop_seed42` checkpoints)

Phase-0 rollouts of the 196 clips, one env per clip, no randomization. "Tracker@N" = the policy of `model_N.pt`;
planners are V2's (`model_9000`) or retrained offline on labels from Tracker rollouts. The tracker gets a 27-value
target vector; below, *positions* (where) and *occupancy* (which bodies have a target, i.e. when) are swapped
separately.

Label bugs in the robot hindsight labels (fixed in the labeler, see below):

- No gap debouncing: PPO action noise (std 0.53) makes the stance foot chatter (force on/off), so every chatter gap
  became a fake contact onset at the stance foot. On V2 replay samples 73 % of foot steps were valid (32k fake
  stance labels), left/right knee labels 7:1. With 5-step gap filling and the 3D force norm: feet 0.51 / 0.48,
  knees 0.046 / 0.047.
- `max_recorded_steps = 500` cut the 91 clips longer than 500 steps (all low-/high-climb clips) at step ~416.

Swap tests on Tracker@1000 (success from phase 0, overall / high-climb / high-step / low-climb / low-step):

| targets fed to the tracker | overall |
|---|---|
| reference (positions and occupancy) | ~95 % |
| V2 planner positions, reference occupancy | 63.3 % (40 / 95 / 10 / 84) |
| planner retrained on debounced labels: positions, reference occupancy | 82.1 % (80 / 98 / 68 / 77) |
| the same planner's own occupancy too | 3-11 % |
| reference positions, oracle hindsight occupancy (labels of the reference-driven rollout of the same clip) | 65.8 % (55 / 92 / 0 / 93) |
| oracle hindsight positions and occupancy | 12.8 % |

- Positions are fine after the label fixes; occupancy is what fails.
- Elbows: the robot never touches the terrain with them (hindsight rate 0.000), the annotation marks them in 3.1-3.5 %
  of low-climb frames (within 4 cm of the box while the hands support). Oracle occupancy without elbow targets: 0 %
  on low-climb.
- The retrained planner matches its labels on the rollout they came from (feet P/R 0.98), but in closed loop on its
  own occupancy its foot P/R against the reference falls from 0.885 / 0.790 to 0.593 / 0.570. Its input holds 465
  values of 5-step proprioception (joint pos/vel, last action); once its occupancy changes the tracker's actions,
  those inputs leave the training distribution.
- Robot touchdown lags the reference's annotated onset by a few frames, and the reference marks onsets earlier
  (12 frames on a traced low-step clip).

`_run_diag_v13`: planner from scratch on debounced labels of admitted (successful) rollouts of Tracker@1000/2000
(394k samples, foot valid 0.51 / 0.48, elbows 0): own occupancy 5.1 % on Tracker@1000/2000; its positions with
reference occupancy 80.1 / 76.0 %; its foot occupancy with reference occupancy elsewhere 17.9 / 13.8 %.

Labeler changes (this commit): `label_next_contacts(min_gap_steps=...)` fills interior non-contact gaps shorter
than `HindsightContactLabelerCfg.min_gap_steps` (default 5 steps = 0.1 s, as `annotate_motion_contacts.py`)
before the segment-length filter; robot contacts use the norm of the terrain contact force (was |F_z|);
`max_recorded_steps` 500 -> 1000.

`_run_diag_v15` on `Tracker@1000` (196 clips, planner trained on debounced Rollout 0 labels) isolates the remaining
closed-loop foot-occupancy gap: `pl_pos_ref_occ` 81.6 % (160/196), `ref_pos_pl_feet_occ` 63.3 % (124/196,
`hc76=95 %`, `lc76=85 %`), `pl_pos_oracle_feet_occ` 32.7 % (64/196), `pl_pos_pl_feet_occ` 19.9 % (39/196). Because
`ref_t` is zero whenever `ref_on` is False, false-positive `occ` steps (~11-17 %) are harmless with `ref_pos`,
whereas `pl_pos` injects non-zero target coordinates on those false-positive steps and destabilizes the stage-1
tracker before stage-2 PPO adaptation.

### Stage-2 V3 runs (launched 2026-11-17 00:07 EST on `gthintel201` GPUs 0-3, `r1730`)

Four 10k-iteration runs from `acpS1_php_footDrop25_{seed42,seed2}/model_16000.pt` with the fixed hindsight labeler:

- GPU 0: `acpS2v3_fixed_seed42` (`Instinct-ACP-Planner-V2-G1-v0`, seed 42; pure ablation of V2 + labeler fixes)
- GPU 1: `acpS2v3_epGate85_seed42` (`Instinct-ACP-Planner-V3-G1-v0`, seed 42; episode-level `planner_gate` with `eta_max=0.85`)
- GPU 2: `acpS2v3_epGate85_seed2` (`Instinct-ACP-Planner-V3-G1-v0`, seed 2)
- GPU 3: `acpS2v3_fixed_seed2` (`Instinct-ACP-Planner-V2-G1-v0`, seed 2)

### Stage-2 velocity-dependent termination & final V3 evaluation (2026-11-19/20)

Diagnosis on `@9k` checkpoints showed 100 % of standard-threshold failures (`21.9 %` overall) were `base_pos_too_far`
(`|z_robot(t) - z_ref(t)| > 0.25 m` at exact timestamp `t`), killing valid timing-shifted climbs that otherwise finish
within `1.3-1.6 cm` in `z` and `2.5-4.5 cm` in `xy`. Added configurable `vel_scale` to `pos_far_from_ref`
(`threshold = distance_threshold + vel_scale * ||v_ref||`) and set `vel_scale = 0.5` in `G1ACPPlannerEnvCfg` (`75fd03d`).

Final evaluation (`scripts/acp/evaluate.py`, `vel_scale=0.5`, 196 PHP clips) after resuming training with `vel_scale=0.5`
(training `dataset_exhausted` rose from `~72 %` to `91.9-93.7 %`):

| Run (`*vel05*`) | Ckpt | Planner phase 0.0 (hc76 / hs / lc76 / ls) | Planner phase 0.5 | Reference phase 0.0 |
|---|---|---|---|---|
| `acpS2v3_noAct_seed42_vel05` | `model_9000.pt` | **81.1 %** (85 / 100 / 23 / 100) | 98.0 % | 73.5 % |
| `acpS2v3_noActOcc5_seed42_vel05` | `model_10000.pt` | **79.6 %** (73 / 100 / 35 / 95) | 100.0 % | 77.0 % |
| `acpS2v3_fixed_seed42_vel05` | `model_13000.pt` | **79.1 %** (78 / 100 / 20 / 100) | 100.0 % | 74.0 % |
| `acpS2v3_refLabels_seed42_vel05` | `model_9000.pt` | **74.0 %** (62 / 95 / 17 / 100) | 99.0 % | 81.6 % |

Planner targets now outperform reference targets on the 3 robot-hindsight runs (`79.1-81.1 %` vs `73.5-77.0 %`); ready to move to stage 3 (`ACP_STAGE3_PLAN.md`).

### Stage-3 implementation, zero-shot baseline & 4-GPU `torchrun` launch (2026-11-21)

Implemented Stage-3 goal-reaching RL fine-tuning on `feat/acp-stage3` in both `instinct_rl` (`R1–R6` + `D13` + DDP `context_normalizer.sync_across_processes()` + `map_location`) and `instinctlab` (`L1–L10`, user decisions D1–D15, starting from `acpS2v3_noActOcc5_seed42_vel05/model_10000.pt`). All 47 CPU unit tests pass (`test_contact_planner.py` 25/25, `test_stage3_rl.py` 5/5, `test_hindsight_labeler.py` 12/12, `test_stage3_env.py` 5/5).

Key fixes from GPU smoke testing:
- `GoalReachingCommand._resample_command` made a no-op so `command_manager.reset` does not overwrite `tile_origin_w`/`goal_pos_w` after `reset_goal_envs`.
- Switched the 6 procedural columns from dense `Perlin*` heightfields (~40.7M tris, which overflowed PhysX BV4 cooking) to exact trimesh (`MeshPyramidStairs`, `MeshInvertedPyramidStairs`, `MeshBox`, `MeshGap`) + coarse-scale heightfields (`HfSteppingStones` 0.25 m, `HfPyramidSloped` 0.5 m), cutting startup to ~38 s with clean PhysX cooking.

Zero-shot baseline (`scripts/acp/eval_stage3.py`, 256 envs, level 0, `noActOcc5@10k`):
- Goal envs overall: **36.9 %** (86/233) — `perlin_slope` 100.0 %, `square_gaps` 95.5 %, `pyramid_stairs` 95.2 %, `mesh_boxes` 90.9 %, `pyramid_stairs_inv` 6.1 %, `stepping_stones` 1.0 %.
- Reference envs (early-stopped at 200 goal episodes): 60.3 % (85/141).

Launched 10,000-iteration 4-GPU `torchrun` Stage-3 run `acpS3_noActOcc5_4gpu_seed42` (2,100 envs/GPU × 4 = 8,400 envs, ~6.95 s/iter) on `gthintel201` (`manip_0:0,1,2,3`, reservation `r1973`, tmux `acp_s3`, guarded by `acp_window_guard.sh 1973`).

## Infrastructure

- gthintel cluster via cardbot; runs in `humix-docker` containers launched by `~/workspace/isaacSim/il_run.sh`,
  checkouts `instinctlab_stage2` / `instinct_rl_stage2`, logs and run dirs under `~/Data/instinctlab_logs`.
- `~/workspace/isaacSim/acp_window_guard.sh <reservation>` stops our containers 6 min before the reservation ends.
