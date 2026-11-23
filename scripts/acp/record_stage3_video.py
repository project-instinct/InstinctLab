"""Record a multi-terrain showcase MP4 of an ACP stage-3 checkpoint using IsaacLab's Kit visualizer + cv2.

Records consecutive clips of ``model_10000.pt`` on Level-5 procedural terrains (stairs up, stairs down,
mesh boxes, square gaps, slope) with a follow-cam, a green goal sphere marker, and a status HUD overlay.

Usage (inside the container)::

    isaaclab -p scripts/acp/record_stage3_video.py \
        --task Instinct-ACP-Finetune-G1-Play-v0 \
        --load_run <run_dir> --checkpoint model_10000.pt \
        --viz kit --level 5 --steps_per_terrain 250 \
        --out /home/leo/Data/instinctlab_logs/stage3_10k_l5.mp4
"""

from __future__ import annotations

import os
import torch

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

parser = cli.build_play_parser()
parser.add_argument("--level", type=int, default=5)
parser.add_argument("--steps_per_terrain", type=int, default=250)
parser.add_argument(
    "--terrains",
    nargs="+",
    default=["pyramid_stairs", "pyramid_stairs_inv", "mesh_boxes", "square_gaps", "perlin_slope"],
)
parser.add_argument("--out", type=str, required=True)
args_cli, overrides = cli.parse_cli(parser, require_run_name=False)


def main():
    num_envs = 6
    env_cfg, agent_cfg = cli.resolve_play_cfg(args_cli.task, overrides)
    cli.update_instinct_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = num_envs
    env_cfg.num_goal_envs = num_envs
    env_cfg.curriculum = None
    env_cfg.commands.goal_reaching.max_init_level = args_cli.level
    for visualizer_cfg in env_cfg.sim.visualizer_cfgs:
        visualizer_cfg.backend.origin_type = "world"
        visualizer_cfg.headless = True
        visualizer_cfg.window.size = (960, 540)

    ckpt_path = os.path.join(args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Recording video for {ckpt_path} -> {args_cli.out}")

    with launch_simulation(env_cfg, args_cli):
        import cv2
        import gymnasium as gym
        import numpy as np

        from instinct_rl.runners import OnPolicyRunner
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt_path)
        runner.eval_mode()

        alg = runner.alg
        policy_norm = runner.normalizers.get("policy")
        unwrapped = env.unwrapped
        device = unwrapped.device
        goal_cmd = unwrapped.command_manager.get_term("goal_reaching")
        robot = unwrapped.scene["robot"]
        viz = next((v for v in unwrapped.sim.visualizers if hasattr(v, "render_rgb_array")), None)
        if viz is None:
            raise RuntimeError("No rgb-capable visualizer found; pass --viz kit.")

        goal_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Stage3Goal",
                markers={
                    "goal": sim_utils.SphereCfg(
                        radius=0.18,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.95, 0.3)),
                    )
                },
            )
        )

        name_to_env = {name: i for i, name in enumerate(goal_cmd.procedural_type_names) if i < num_envs}
        selected = [(t, name_to_env[t]) for t in args_cli.terrains if t in name_to_env]
        print(f"[INFO] Recording terrains (level {args_cli.level}): {selected}")

        env.set_training_iteration(10**9)
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        writer = None
        fps = int(round(1.0 / unwrapped.step_dt))
        total_frames = 0

        for tname, env_idx in selected:
            goal_cmd.terrain_level.fill_(args_cli.level)
            obs, extras = env.reset()
            for step in range(args_cli.steps_per_terrain):
                rpos = robot.data.root_pos_w.torch[env_idx].cpu().numpy()
                gpos = goal_cmd.goal_pos_w[env_idx : env_idx + 1]
                goal_marker.visualize(translations=gpos)

                eye = (float(rpos[0] - 2.6), float(rpos[1] - 2.2), float(rpos[2] + 1.4))
                target = (float(rpos[0] + 0.3), float(rpos[1]), float(rpos[2] - 0.1))
                unwrapped.sim.set_camera_view(eye=eye, target=target)

                with torch.no_grad():
                    obs_t = obs.to(device)
                    extra_obs = {k: v.to(device) for k, v in extras["observations"].items()}
                    norm_obs = policy_norm(obs_t) if policy_norm is not None else obs_t
                    composed = alg.compose_policy_obs(norm_obs, extra_obs, target_noise=False)
                    actions = alg.actor_critic.act_inference(composed)
                    env.set_policy_feedback(alg.get_policy_feedback())
                    obs, _, dones, extras = env.step(actions)

                frame = viz.render_rgb_array()
                if frame is not None:
                    bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                    if writer is None:
                        h, w = bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args_cli.out, fourcc, fps, (w, h))
                    d_g = float(goal_cmd.dist_xy[env_idx])
                    reached = bool(goal_cmd.reached[env_idx])
                    label = (
                        f"ACP Stage-3 @10k | {tname} (L{args_cli.level}) | "
                        f"d_g={d_g:.2f}m | reached={'YES' if reached else 'no'}"
                    )
                    cv2.rectangle(bgr, (10, 10), (780, 48), (20, 20, 24), -1)
                    cv2.putText(
                        bgr,
                        label,
                        (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (0, 255, 120) if reached else (240, 240, 240),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(bgr)
                    total_frames += 1

                if bool(dones[env_idx]):
                    break

        if writer is not None:
            writer.release()
            print(f"[INFO] Saved video ({total_frames} frames): {args_cli.out}")
        else:
            print("[WARN] No frames captured!")
        env.close()


if __name__ == "__main__":
    main()
