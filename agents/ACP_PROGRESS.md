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

### Next

Find which foot targets the planner misses (time to onset, on the obstacle or the ground, distance to the scan edge,
category) and how often such targets occur in its training labels.

## Infrastructure

- gthintel cluster via cardbot; runs in `humix-docker` containers launched by `~/workspace/isaacSim/il_run.sh`,
  checkouts `instinctlab_stage2` / `instinct_rl_stage2`, logs and run dirs under `~/Data/instinctlab_logs`.
- `~/workspace/isaacSim/acp_window_guard.sh <reservation>` stops our containers 6 min before the reservation ends.
