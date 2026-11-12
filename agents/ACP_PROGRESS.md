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
2 on gthintel218. At iteration 50: 2,040 samples per step (paper ~370), 87 of them from failed episodes; the 100k
replay now holds about 2 iterations of labels.

## Infrastructure

- gthintel cluster via cardbot; runs in `humix-docker` containers launched by `~/workspace/isaacSim/il_run.sh`,
  checkouts `instinctlab_stage2` / `instinct_rl_stage2`, logs and run dirs under `~/Data/instinctlab_logs`.
- `~/workspace/isaacSim/acp_window_guard.sh <reservation>` stops our containers 6 min before the reservation ends.
