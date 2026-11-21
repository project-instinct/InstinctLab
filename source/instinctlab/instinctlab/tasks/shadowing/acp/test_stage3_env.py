"""Pure-torch unit tests for ACP stage-3 InstinctLab components (L1, L3, L5, L6, L7).

Run with ``pytest source/instinctlab/instinctlab/tasks/shadowing/acp/test_stage3_env.py`` (CPU only).
"""

from types import SimpleNamespace

import pytest
import torch

from instinctlab.motion_reference.motion_files.terrain_motion import TerrainMotion
from instinctlab.tasks.shadowing.acp import goal_reaching
from instinctlab.tasks.shadowing.acp.hindsight_labeler import (
    DUAL_BUFFER_KEYS,
    PARTITION_KEYS,
    SAMPLE_KEYS,
    HindsightContactLabeler,
    HindsightContactLabelerCfg,
)
from instinctlab.terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg


# -- L1: TerrainMotion.match_scene ignores non-motion-matched columns ---------------------------


def test_l1_match_scene_skips_non_motion_matched_tiles():
    motion = TerrainMotion.__new__(TerrainMotion)
    motion.yaml_data = {"terrains": [{"terrain_id": 0}, {"terrain_id": 1}]}
    motion._all_motion_terrain_ids = torch.tensor([0, 1], dtype=torch.long)
    motion.cfg = SimpleNamespace(max_origins_per_motion=4)
    motion._all_motion_origins_in_scene = torch.full((2, 4, 3), float("nan"))
    motion._all_motion_num_selectable_origins = torch.zeros(2, dtype=torch.int)

    # 2 rows x 4 cols: cols 0-1 are MotionMatchedTerrainCfg, cols 2-3 are procedural
    origins = torch.zeros(2, 4, 3)
    for r in range(2):
        for c in range(4):
            origins[r, c] = torch.tensor([float(r), float(c), 0.0])

    sub_cfgs = []
    for r in range(2):
        diff = (r + 0.1) / 2.0
        for c in range(4):
            if c < 2:
                cfg = MotionMatchedTerrainCfg(metadata_yaml="dummy.yaml")
                cfg.difficulty = diff
            else:
                cfg = SimpleNamespace(difficulty=diff)
            sub_cfgs.append(cfg)

    scene = SimpleNamespace(terrain=SimpleNamespace(terrain_origins=origins, subterrain_specific_cfgs=sub_cfgs))
    motion.match_scene(scene)

    assert int(motion._all_motion_num_selectable_origins[0]) == 2
    assert int(motion._all_motion_num_selectable_origins[1]) == 2
    # All matched origins must come from cols 0 and 1 (y in {0, 1}), never cols 2 or 3
    for t in range(2):
        matched_y = motion._all_motion_origins_in_scene[t, :2, 1]
        assert set(matched_y.tolist()) == {0.0, 1.0}


# -- L3: goal encoding and env-type helpers -----------------------------------------------------


def test_l3_encode_goal_b_and_env_type():
    root_pos = torch.tensor([[0.0, 0.0, 0.78], [0.0, 0.0, 0.78]])
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])  # yaw = 0
    # env 0: 2 m ahead (> r=0.5 -> unit vector [1, 0]); env 1: 0.25 m ahead (< r=0.5 -> [0.5, 0])
    goal_pos = torch.tensor([[2.0, 0.0, 0.78], [0.25, 0.0, 0.78]])
    goal_heading = torch.tensor([0.0, 0.5])
    encoded = goal_reaching.encode_goal_b(root_pos, root_quat, goal_pos, goal_heading, goal_radius=0.5)
    torch.testing.assert_close(encoded[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(encoded[1], torch.tensor([0.5, 0.0, 0.5]), atol=1e-5, rtol=0)

    env = SimpleNamespace(num_envs=6, device=torch.device("cpu"), cfg=SimpleNamespace(num_goal_envs=2))
    mask = goal_reaching.is_goal_env(env)
    assert mask.tolist() == [False, False, False, False, True, True]
    one_hot = goal_reaching.env_type_obs(env)
    assert one_hot.shape == (6, 2)
    assert one_hot[0].tolist() == [1.0, 0.0]
    assert one_hot[5].tolist() == [0.0, 1.0]


# -- L5 + D7: move2goal & opt-in terrain masks --------------------------------------------------


def test_l5_move2goal_and_opt_in_terrain_masks():
    # 2 envs: both moving at 1.2 m/s toward goal (distance 3 m)
    # Env 0 is on "pyramid_stairs" (v_max=0.8 when masks ON), env 1 is on "perlin_slope" (v_max=2.0)
    cmd = SimpleNamespace(
        goal_pos_w=torch.tensor([[3.0, 0.0, 0.78], [3.0, 0.0, 0.78]]),
        cfg=SimpleNamespace(success_radius=0.5),
        procedural_type_names=["pyramid_stairs", "perlin_slope"],
        terrain_type=torch.tensor([0, 1], dtype=torch.long),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3)),
            root_lin_vel_w=SimpleNamespace(torch=torch.tensor([[1.2, 0.0, 0.0], [1.2, 0.0, 0.0]])),
        )
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
    )
    # Default (use_terrain_masks=False): v_max=2.0 everywhere -> both get 1.0
    rew_default = goal_reaching.move2goal(env, "goal_reaching", use_terrain_masks=False)
    assert rew_default.tolist() == [1.0, 1.0]

    # Opt-in extension (use_terrain_masks=True): stairs cap v_max=0.8 -> env 0 gets 0.0, env 1 gets 1.0
    rew_masked = goal_reaching.move2goal(env, "goal_reaching", use_terrain_masks=True)
    assert rew_masked.tolist() == [0.0, 1.0]


# -- L6: GoalTerrainLevels curriculum -----------------------------------------------------------


def test_l6_goal_terrain_levels_promote_and_demote():
    cmd = SimpleNamespace(
        num_levels=10,
        terrain_level=torch.tensor([0, 0, 2, 2], dtype=torch.long),
        reached=torch.tensor([False, False, True, False]),
        min_qualifying_dist=torch.tensor([5.0, 5.0, 0.1, 2.5]),
    )
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(num_goal_envs=2),  # envs 2, 3 are goal envs
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, False, True])),
    )
    curr = goal_reaching.GoalTerrainLevels(cfg=SimpleNamespace(), env=env)
    env_ids = torch.tensor([2, 3], dtype=torch.long)

    # Step 1: env 2 succeeds (ema=0.2 <= 0.5 -> stay at 2); env 3 fails with dist 2.5 > 1.0 -> demote to 1
    curr(env, env_ids)
    assert cmd.terrain_level[2].item() == 2
    assert cmd.terrain_level[3].item() == 1

    # Push env 2 EMA past 0.5 (0.2 -> 0.36 -> 0.488 -> 0.5904 -> promote to 3!)
    for _ in range(3):
        curr(env, torch.tensor([2], dtype=torch.long))
    assert cmd.terrain_level[2].item() == 3


# -- L7: HindsightContactLabeler dual_buffer, cut at 0.2m, and D12 quality ----------------------


def test_l7_labeler_dual_buffer_cut_and_quality():
    cfg = HindsightContactLabelerCfg(
        body_names=["left_ankle_roll_link"],
        min_gap_steps=0,
        min_contact_steps=2,
        horizon_s=0.4,
        max_recorded_steps=20,
        num_slots=4,
        partition_samples=True,
        dual_buffer=True,
    )
    labeler = HindsightContactLabeler(cfg, num_envs=2, obs_size=4, step_dt=0.05, device="cpu")
    assert set(labeler.sample_keys) == set(SAMPLE_KEYS + PARTITION_KEYS + DUAL_BUFFER_KEYS)
    # Empty pop still carries all keys with length 0
    empty = labeler.pop_samples()
    assert set(empty.keys()) == set(labeler.sample_keys)
    assert empty["buffer_id"].shape == (0,) and empty["quality"].shape == (0,)

    # Start both envs (env 0 = ref, env 1 = goal)
    labeler.start_episodes(torch.tensor([0, 1]), eligible=torch.tensor([True, True]))
    # Record 10 steps; env 0 does work=1.0/step, env 1 does work=2.0/step
    for t in range(10):
        obs = torch.full((2, 4), float(t))
        contact = torch.tensor([[t >= 4], [t >= 4]])
        body_pos = torch.zeros(2, 1, 3)
        root_pos = torch.zeros(2, 3)
        root_yaw = torch.zeros(2)
        work = torch.tensor([1.0, 2.0])
        labeler.record(obs, contact, body_pos, root_pos, root_yaw, positive_work=work)

    # Finish both admitted: env 0 no cut (-1), env 1 cut at step 4 (emits steps 0..4 = 5 steps), d_init=4.0
    labeler.finish_episodes(
        env_ids=torch.tensor([0, 1]),
        episode_length=torch.tensor([10, 10]),
        admitted=torch.tensor([True, True]),
        partition=torch.tensor([7, 12]),
        group=torch.tensor([1, 2]),
        cut_length=torch.tensor([-1, 4]),
        quality_denom=torch.tensor([1.0, 4.0]),
        buffer_id=torch.tensor([0, 1]),
    )
    samples = labeler.pop_samples()
    # 10 samples from env 0 + 5 samples from env 1 = 15 samples
    assert samples["planner_obs"].shape == (15, 4)
    assert (samples["buffer_id"] == 0).sum().item() == 10
    assert (samples["buffer_id"] == 1).sum().item() == 5
    # Env 1 quality = -(5 steps * 2.0 work) / max(4.0, 0.5) = -2.5
    goal_q = samples["quality"][samples["buffer_id"] == 1]
    torch.testing.assert_close(goal_q, torch.full((5,), -2.5))
