"""Evaluate an ACP stage-2 (zero-shot) or stage-3 checkpoint on the mixed reference + procedural goal-reaching task.

Reports:
- Reference-env success rate (phase 0.0 -> >= 0.9, no failure termination).
- Goal-env success rate overall, per procedural terrain type, and per difficulty level.
- Mean min_qualifying_dist and termination breakdown for goal envs.

Usage (inside the container)::

    isaaclab -p scripts/acp/eval_stage3.py \
        --task Instinct-ACP-Finetune-G1-v0 \
        --load_run <run_dir> --checkpoint model_10000.pt \
        --num_envs 256 --num_episodes 200 --level 0 --headless
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import os

import gymnasium as gym
import torch

from instinct_rl.runners import OnPolicyRunner
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ACP Stage-3 goal-reaching + reference retention.")
parser.add_argument("--task", type=str, default="Instinct-ACP-Finetune-G1-v0")
parser.add_argument("--load_run", type=str, required=True, help="Run directory (or full path to run dir).")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename (e.g. model_10000.pt).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_episodes", type=int, default=200, help="Target completed goal episodes to collect.")
parser.add_argument("--level", type=int, default=0, help="Fixed procedural terrain level (0..9) or -1 for uniform.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max rollout steps.")
parser.add_argument("--debug_ref", action="store_true", help="Print ref-env 0..3 diagnostics for first 25 steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import instinctlab.tasks  # noqa: E402, F401
from instinctlab.tasks.shadowing.acp.goal_reaching import is_goal_env  # noqa: E402
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _resolve_checkpoint(load_run: str, checkpoint: str, experiment_name: str) -> str:
    if os.path.isfile(os.path.join(load_run, checkpoint)):
        return os.path.join(load_run, checkpoint)
    for base in (
        os.path.expanduser(f"~/Data/instinctlab_logs/instinct_rl/{experiment_name}"),
        os.path.expanduser("~/Data/instinctlab_logs/instinct_rl/g1_acp_planner"),
        os.path.expanduser("~/Data/instinctlab_logs/instinct_rl/g1_acp_finetune"),
    ):
        cand = os.path.join(base, load_run, checkpoint)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"Could not find {checkpoint} in {load_run}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.num_goal_envs = args_cli.num_envs // 2
    env_cfg.curriculum = None  # keep eval levels fixed
    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    if args_cli.level >= 0:
        env_cfg.commands.goal_reaching.max_init_level = args_cli.level

    # Load agent cfg from task registration
    spec = gym.spec(args_cli.task)
    agent_entry = spec.kwargs["instinct_rl_cfg_entry_point"]
    mod_name, cls_name = agent_entry.split(":")
    import importlib

    agent_cfg = getattr(importlib.import_module(mod_name), cls_name)()
    agent_cfg.device = args_cli.device

    ckpt_path = _resolve_checkpoint(args_cli.load_run, args_cli.checkpoint, agent_cfg.experiment_name)
    print(f"[INFO] Evaluating checkpoint: {ckpt_path}")

    env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    runner.eval_mode()

    alg = runner.alg
    policy_norm = runner.normalizers.get("policy")
    unwrapped = env.unwrapped
    device = unwrapped.device
    goal_cmd = unwrapped.command_manager.get_term("goal_reaching")
    goal_mask = is_goal_env(unwrapped)

    if args_cli.level >= 0:
        goal_cmd.terrain_level.fill_(args_cli.level)
    else:
        goal_cmd.terrain_level[:] = torch.arange(unwrapped.num_envs, device=device) % goal_cmd.num_levels

    env.set_training_iteration(10**9)
    obs, extras = env.get_observations()

    ref_total = 0
    ref_success = 0
    goal_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "succ": 0, "min_d": 0.0, "oob": 0, "tilt": 0, "to": 0})
    goal_total = 0

    for step in range(args_cli.max_steps):
        # Snapshot pre-step episode metrics before auto-reset wipes them
        pre_reached = goal_cmd.reached.clone()
        pre_min_d = goal_cmd.min_qualifying_dist.clone()
        pre_type = goal_cmd.terrain_type.clone()
        pre_ref_phase = unwrapped.reference_phase(torch.arange(unwrapped.num_envs, device=device)).clone()

        with torch.inference_mode():
            obs_t = obs.to(device)
            extra_obs = {k: v.to(device) for k, v in extras["observations"].items()}
            norm_obs = policy_norm(obs_t) if policy_norm is not None else obs_t
            composed = alg.compose_policy_obs(norm_obs, extra_obs, target_noise=False)
            actions = alg.actor_critic.act_inference(composed)
            env.set_policy_feedback(alg.get_policy_feedback())
            obs, _, dones, extras = env.step(actions)

        done_ids = dones.bool().nonzero().squeeze(-1)
        if args_cli.debug_ref and step < 25:
            rob = unwrapped.scene["robot"]
            mr = unwrapped.scene["motion_reference"]
            tm = unwrapped.termination_manager
            rpos = rob.data.root_pos_w.torch[:4]
            refpos = mr.data.base_pos_w[mr.ALL_INDICES[:4], mr.aiming_frame_idx[:4]]
            dz = (rpos[:, 2] - refpos[:, 2]).abs()
            dxy = (rpos[:, :2] - refpos[:, :2]).norm(dim=-1)
            bptf = tm.get_term("base_pos_too_far")[:4]
            bpg = tm.get_term("base_pg_too_far")[:4]
            print(
                f"[DEBUG step={step:2d}] rpos0={rpos[0].round(decimals=2).tolist()} "
                f"ref0={refpos[0].round(decimals=2).tolist()} dz={dz.round(decimals=3).tolist()} "
                f"dxy={dxy.round(decimals=2).tolist()} bptf={bptf.tolist()} bpg={bpg.tolist()} done0..3={dones[:4].tolist()}"
            )
        if done_ids.numel() > 0:
            tm = unwrapped.termination_manager
            terminated = tm.terminated[done_ids]
            for i_idx, env_id in enumerate(done_ids.tolist()):
                if goal_mask[env_id]:
                    tname = goal_cmd.procedural_type_names[int(pre_type[env_id])]
                    st = goal_stats[tname]
                    st["n"] += 1
                    goal_total += 1
                    succ = bool(pre_reached[env_id]) and (not bool(terminated[i_idx]))
                    st["succ"] += int(succ)
                    md = float(pre_min_d[env_id])
                    st["min_d"] += md if md < 100.0 else 5.0
                    if bool(tm.get_term("goal_out_of_bounds")[env_id]):
                        st["oob"] += 1
                    elif bool(tm.get_term("goal_bad_orientation")[env_id]):
                        st["tilt"] += 1
                    elif bool(tm.get_term("goal_time_out")[env_id]):
                        st["to"] += 1
                else:
                    ref_total += 1
                    if float(pre_ref_phase[env_id]) >= 0.9 and not bool(terminated[i_idx]):
                        ref_success += 1
        if goal_total >= args_cli.num_episodes and ref_total >= args_cli.num_episodes // 2:
            break

    print("\n=== ACP Stage-3 Evaluation Summary ===")
    ref_rate = 100.0 * ref_success / max(ref_total, 1)
    print(f"Reference envs (phase 0.0 -> >=0.9): {ref_success}/{ref_total} ({ref_rate:.1f}%)")
    total_g_succ = sum(s["succ"] for s in goal_stats.values())
    total_g_n = sum(s["n"] for s in goal_stats.values())
    print(f"Goal envs overall (level={args_cli.level}): {total_g_succ}/{total_g_n} ({100.0 * total_g_succ / max(total_g_n, 1):.1f}%)")
    print(f"{'Terrain Type':<22} {'Succ':>10} {'Rate':>7} {'MinDist':>8} {'Tilt':>6} {'OOB':>6} {'TO':>6}")
    for tname in sorted(goal_stats):
        s = goal_stats[tname]
        n = max(s["n"], 1)
        print(
            f"{tname:<22} {int(s['succ']):>4}/{int(s['n']):<5} {100.0*s['succ']/n:>6.1f}% "
            f"{s['min_d']/n:>8.2f} {int(s['tilt']):>6} {int(s['oob']):>6} {int(s['to']):>6}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
