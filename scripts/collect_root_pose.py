"""Interactively collect manual root-pose trajectories on an IsaacLab terrain.

This script intentionally stays outside task configs: it launches an existing
environment, teleports the robot root to a keyboard-controlled reference pose,
and saves both the trajectory and reproducibility metadata.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect manual root pose trajectories.")
parser.add_argument(
    "--task",
    type=str,
    default="Instinct-HSIDownstreamClimb-Perceptive-Vae-G1-Play-v0",
    help="Task id to launch for terrain initialization.",
)
parser.add_argument("--output_dir", type=str, default="data/root_pose_trajs", help="Directory for saved trajectory runs.")
parser.add_argument("--name", type=str, default=None, help="Optional run directory name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs. This collector supports one env.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--speed", type=float, default=0.4, help="Horizontal movement speed in m/s.")
parser.add_argument("--z_speed", type=float, default=0.25, help="Vertical movement speed in m/s.")
parser.add_argument("--yaw_speed", type=float, default=1.0, help="Yaw speed in rad/s.")
parser.add_argument("--history_stride", type=int, default=5, help="Visualize one historical marker every N samples.")
parser.add_argument("--max_vis_points", type=int, default=1000, help="Maximum number of historical markers to show.")
parser.add_argument(
    "--auto_face_motion",
    action="store_true",
    default=False,
    help="Automatically set yaw to the horizontal motion direction.",
)
parser.add_argument(
    "--follow_robot",
    action="store_true",
    default=True,
    help="Continuously write the manual root pose to the robot.",
)
parser.add_argument(
    "--no_follow_robot",
    action="store_false",
    dest="follow_robot",
    help="Only move the marker and record the trajectory; do not teleport the robot.",
)
parser.add_argument(
    "--disable_camera_frame",
    action="store_true",
    default=False,
    help="Use robot yaw instead of the active viewport camera for WASD motion.",
)
parser.add_argument(
    "--keep_terminations",
    action="store_true",
    default=False,
    help="Keep configured termination terms. By default they are disabled for uninterrupted collection.",
)
parser.add_argument("--save_mesh", action="store_true", default=False, help="Also export the generated terrain mesh as PLY.")
parser.add_argument("--debug", action="store_true", default=False, help="Wait for debugger attach.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("collect_root_pose.py supports exactly one environment. Use --num_envs 1.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import carb.input
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.appwindow
import omni.usd
from carb.input import KeyboardEventType
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Gf

import instinctlab.tasks  # noqa: F401


if args_cli.debug:
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Waiting for debugger attach at %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def _make_output_dir() -> str:
    run_name = args_cli.name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args_cli.task}_{timestamp}"
    out_dir = os.path.abspath(os.path.join(args_cli.output_dir, run_name))
    os.makedirs(out_dir, exist_ok=False)
    return out_dir


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _wrap_pi(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _normal_clone(tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_to_numpy(tensor), device=tensor.device, dtype=tensor.dtype)


def _matches_key(event_input, *names: str) -> bool:
    for name in names:
        key = getattr(carb.input.KeyboardInput, name, None)
        if key is not None and event_input == key:
            return True
    return False


def _key_is_down(keys: set, *names: str) -> bool:
    return any(getattr(carb.input.KeyboardInput, name, None) in keys for name in names)


def _get_active_camera_yaw() -> float | None:
    """Return active viewport camera yaw in world frame, or None if unavailable."""
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            return None
        camera_path = viewport.get_active_camera()
        if camera_path is None:
            return None
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(str(camera_path))
        if not camera_prim or not camera_prim.IsValid():
            return None
        transform = omni.usd.get_world_transform_matrix(camera_prim)
        forward = transform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        forward_xy = np.array([float(forward[0]), float(forward[1])], dtype=np.float64)
        norm = np.linalg.norm(forward_xy)
        if norm < 1.0e-6:
            return None
        forward_xy /= norm
        return float(np.arctan2(forward_xy[1], forward_xy[0]))
    except Exception:
        return None


def _build_markers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    current_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ManualRootPose/current",
        markers={
            "current": sim_utils.SphereCfg(
                radius=0.12,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.05)),
            ),
        },
    )
    history_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ManualRootPose/history",
        markers={
            "history": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(current_cfg), VisualizationMarkers(history_cfg)


def _zero_action(env) -> torch.Tensor:
    if hasattr(env.unwrapped, "action_manager"):
        return torch.zeros_like(env.unwrapped.action_manager.action)
    action_shape = env.unwrapped.single_action_space.shape
    return torch.zeros((env.unwrapped.num_envs, *action_shape), device=env.unwrapped.device)


def _disable_terminations(env) -> None:
    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    if termination_manager is None:
        return
    termination_manager._term_names = []
    termination_manager._term_cfgs = []
    termination_manager._class_term_cfgs = []
    termination_manager._term_name_to_term_idx = {}
    termination_manager._term_dones = torch.zeros(
        (env.unwrapped.num_envs, 0), device=env.unwrapped.device, dtype=torch.bool
    )
    termination_manager._last_episode_dones = torch.zeros_like(termination_manager._term_dones)


def _write_root_pose(robot, root_pose: torch.Tensor, device: torch.device) -> None:
    with torch.inference_mode():
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


def _save_trajectory(
    out_dir: str,
    env,
    env_cfg,
    samples: list[dict[str, float]],
    root_pos: torch.Tensor,
    root_yaw: torch.Tensor,
) -> None:
    if not samples:
        print("[collector] No samples collected; nothing saved.")
        return

    t = np.asarray([row["t"] for row in samples], dtype=np.float32)
    step = np.asarray([row["step"] for row in samples], dtype=np.int64)
    root_pos_w = np.asarray([[row["x"], row["y"], row["z"]] for row in samples], dtype=np.float32)
    root_yaw_w = np.asarray([row["yaw"] for row in samples], dtype=np.float32)
    root_quat_w = np.asarray(
        [[row["qw"], row["qx"], row["qy"], row["qz"]] for row in samples],
        dtype=np.float32,
    )

    terrain = env.unwrapped.scene["terrain"]
    env_origin = _to_numpy(env.unwrapped.scene.env_origins[0])
    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)

    np.savez_compressed(
        os.path.join(out_dir, "trajectory.npz"),
        t=t,
        step=step,
        dt=np.float32(env.unwrapped.step_dt),
        root_pos_w=root_pos_w,
        root_yaw_w=root_yaw_w,
        root_quat_w=root_quat_w,
        final_root_pos_w=_to_numpy(root_pos),
        final_root_yaw_w=np.float32(root_yaw.item()),
        env_origin_w=env_origin,
        task=np.asarray(args_cli.task),
        auto_face_motion=np.asarray(args_cli.auto_face_motion),
        camera_frame=np.asarray(not args_cli.disable_camera_frame),
        terrain_levels=_to_numpy(terrain_levels) if terrain_levels is not None else np.asarray([]),
        terrain_types=_to_numpy(terrain_types) if terrain_types is not None else np.asarray([]),
        terrain_origins=_to_numpy(terrain_origins) if terrain_origins is not None else np.asarray([]),
    )

    with open(os.path.join(out_dir, "trajectory.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "t", "x", "y", "z", "yaw", "qw", "qx", "qy", "qz"])
        writer.writeheader()
        writer.writerows(samples)

    dump_yaml(os.path.join(out_dir, "env_cfg.yaml"), env_cfg)

    terrain_cfg = getattr(terrain, "cfg", None)
    if terrain_cfg is not None:
        dump_yaml(os.path.join(out_dir, "terrain_cfg.yaml"), terrain_cfg)

    terrain_meta = {
        "task": args_cli.task,
        "step_dt": float(env.unwrapped.step_dt),
        "num_samples": len(samples),
        "env_origin_w": env_origin.tolist(),
        "terrain_levels": _to_numpy(terrain_levels).tolist() if terrain_levels is not None else None,
        "terrain_types": _to_numpy(terrain_types).tolist() if terrain_types is not None else None,
        "terrain_origins_shape": list(terrain_origins.shape) if terrain_origins is not None else None,
        "terrain_generator_class": None,
        "terrain_generator_seed": None,
        "controls": {
            "wasd": "move horizontally in active viewport camera frame",
            "qe": "yaw left/right unless --auto_face_motion is set",
            "zx": "move z up/down",
            "space": "pause/resume recording",
            "enter": "save and exit",
            "escape": "exit without saving",
        },
        "args": vars(args_cli),
    }
    terrain_generator = getattr(terrain, "terrain_generator", None)
    if terrain_generator is not None:
        terrain_meta["terrain_generator_class"] = type(terrain_generator).__name__
        terrain_meta["terrain_generator_seed"] = getattr(getattr(terrain_generator, "cfg", None), "seed", None)
        if args_cli.save_mesh and hasattr(terrain_generator, "terrain_mesh"):
            mesh_path = os.path.join(out_dir, "terrain_mesh.ply")
            terrain_generator.terrain_mesh.export(mesh_path)
            terrain_meta["terrain_mesh_file"] = os.path.basename(mesh_path)
    dump_yaml(os.path.join(out_dir, "terrain_meta.yaml"), terrain_meta)

    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write("Manual root pose trajectory collected by WBCHSI/scripts/collect_root_pose.py\n")
        f.write(f"task: {args_cli.task}\n")
        f.write(f"samples: {len(samples)}\n")
        f.write("trajectory.npz contains root_pos_w [N,3], root_yaw_w [N], root_quat_w [N,4].\n")

    print(f"[collector] Saved {len(samples)} samples to: {out_dir}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1.0e6
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if not args_cli.keep_terminations:
        _disable_terminations(env)

    obs, _ = env.reset()
    del obs

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    root_pos = _normal_clone(robot.data.root_pos_w[0])
    root_quat = _normal_clone(robot.data.root_quat_w[0])
    root_yaw = math_utils.euler_xyz_from_quat(root_quat.unsqueeze(0))[2][0].detach().clone()

    key_down: set = set()
    paused = False
    should_save = False
    should_exit = False
    camera_warning_printed = False

    def on_keyboard_input(event):
        nonlocal paused, should_save, should_exit
        if event.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
            key_down.add(event.input)
            if _matches_key(event.input, "SPACE") and event.type == KeyboardEventType.KEY_PRESS:
                paused = not paused
                print(f"[collector] recording {'paused' if paused else 'resumed'}")
            elif _matches_key(event.input, "ENTER", "NUMPAD_ENTER", "RETURN") and event.type == KeyboardEventType.KEY_PRESS:
                should_save = True
                should_exit = True
            elif _matches_key(event.input, "ESCAPE", "ESC") and event.type == KeyboardEventType.KEY_PRESS:
                should_exit = True
        elif event.type == KeyboardEventType.KEY_RELEASE:
            key_down.discard(event.input)
        return True

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    current_marker, history_marker = _build_markers()
    samples: list[dict[str, float]] = []
    out_dir = _make_output_dir()
    action = _zero_action(env)
    timestep = 0

    print("[collector] Controls: WASD move, Q/E yaw, Z/X height, Space pause, Enter save+exit, Esc exit.")
    print(f"[collector] Output directory reserved: {out_dir}")

    try:
        while simulation_app.is_running() and not should_exit:
            dt = float(env.unwrapped.step_dt)
            frame_yaw = None
            if not args_cli.disable_camera_frame:
                frame_yaw = _get_active_camera_yaw()
                if frame_yaw is None and not camera_warning_printed:
                    print("[collector] Could not read viewport camera; falling back to robot yaw frame.")
                    camera_warning_printed = True
            if frame_yaw is None:
                frame_yaw = float(root_yaw.item())

            forward = torch.tensor([np.cos(frame_yaw), np.sin(frame_yaw), 0.0], device=device, dtype=torch.float32)
            right = torch.tensor([-np.sin(frame_yaw), np.cos(frame_yaw), 0.0], device=device, dtype=torch.float32)
            move = torch.zeros(3, device=device)
            if _key_is_down(key_down, "W"):
                move += forward
            if _key_is_down(key_down, "S"):
                move -= forward
            if _key_is_down(key_down, "D"):
                move += right
            if _key_is_down(key_down, "A"):
                move -= right
            norm = torch.linalg.norm(move[:2])
            if norm > 1.0e-6:
                move[:2] /= norm
            if _key_is_down(key_down, "Z"):
                move[2] += 1.0
            if _key_is_down(key_down, "X"):
                move[2] -= 1.0

            root_pos[:2] += move[:2] * args_cli.speed * dt
            root_pos[2] += move[2] * args_cli.z_speed * dt

            yaw_delta = 0.0
            if _key_is_down(key_down, "Q"):
                yaw_delta += args_cli.yaw_speed * dt
            if _key_is_down(key_down, "E"):
                yaw_delta -= args_cli.yaw_speed * dt
            if args_cli.auto_face_motion and norm > 1.0e-6:
                root_yaw = torch.atan2(move[1], move[0])
            else:
                root_yaw = _wrap_pi(root_yaw + torch.tensor(yaw_delta, device=device))

            root_quat = _quat_from_yaw(root_yaw.unsqueeze(0))[0]
            root_pose = torch.cat([root_pos, root_quat]).unsqueeze(0)

            if args_cli.follow_robot:
                _write_root_pose(robot, root_pose, device)

            current_marker.visualize(root_pos.unsqueeze(0))
            if samples and args_cli.max_vis_points > 0:
                hist = np.asarray(
                    [[row["x"], row["y"], row["z"]] for row in samples[:: max(1, args_cli.history_stride)]],
                    dtype=np.float32,
                )
                if len(hist) > args_cli.max_vis_points:
                    hist = hist[-args_cli.max_vis_points :]
                history_marker.visualize(torch.tensor(hist, device=device))

            if not paused:
                record_t = len(samples) * dt
                samples.append(
                    {
                        "step": timestep,
                        "t": record_t,
                        "x": float(root_pos[0].item()),
                        "y": float(root_pos[1].item()),
                        "z": float(root_pos[2].item()),
                        "yaw": float(root_yaw.item()),
                        "qw": float(root_quat[0].item()),
                        "qx": float(root_quat[1].item()),
                        "qy": float(root_quat[2].item()),
                        "qz": float(root_quat[3].item()),
                    }
                )

            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(action)
                if torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any():
                    _write_root_pose(robot, root_pose, device)
                    print("[collector] Env reset/termination detected; root pose was rewritten.")

            timestep += 1

        if should_save:
            _save_trajectory(out_dir, env, env_cfg, samples, root_pos, root_yaw)
        else:
            print(f"[collector] Exit without saving. Reserved output directory remains empty: {out_dir}")
    finally:
        if keyboard_sub is not None and hasattr(keyboard_sub, "unsubscribe"):
            keyboard_sub.unsubscribe()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
