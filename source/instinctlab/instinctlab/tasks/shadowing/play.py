"""Script to play a checkpoint if an RL agent from Instinct-RL."""

import argparse
import subprocess
import sys

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument(
    "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
)
parser.add_argument(
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)
parser.add_argument(
    "--mpjpe",
    action="store_true",
    default=False,
    help=(
        "Whether to print the mpjpe statistics, both local frame and global frame. Must have shadowing_link_pos_b and"
        " shadowing_link_pos_w monitors"
    ),
)
parser.add_argument(
    "--play_teacher",
    action="store_true",
    default=False,
    help="Whether to play the teacher policy instead of the policy. Must be a distillation task with TPPO algorithm.",
)
parser.add_argument(
    "--compare_actions",
    action="store_true",
    default=False,
    help=(
        "Whether to compare the actions of the policy and the teacher policy. Must be a distillation task with TPPO"
        " algorithm."
    ),
)

parser.add_argument("--x_offset", type=float, default=None, help="Override reset range X")
parser.add_argument("--y_offset", type=float, default=None, help="Override reset range Y")
parser.add_argument(
    "--cam_rotate_speed",
    type=float,
    default=None,
    help="If set, the camera will rotate from initial pose at this speed (rad/s)",
)

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append simulation launcher cli args
from isaaclab.app import add_launcher_args  # isort: skip

add_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

import gymnasium as gym
import numpy as np
import os
import time
import torch

from instinct_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.app import launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # override
    if args_cli.x_offset is not None:
        env_cfg.events.reset_robot.params["randomize_pose_range"]["x"] = [args_cli.x_offset] * 2
    if args_cli.y_offset is not None:
        env_cfg.events.reset_robot.params["randomize_pose_range"]["y"] = [args_cli.y_offset] * 2

    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    # configure the environment-owned recorder before environment construction
    if args_cli.video:
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer",
                output_dir=os.path.join(log_dir, "videos", "play"),
                video_length=args_cli.video_length,
                step_offset=args_cli.video_start_step,
                output_filename_prefix=f"model_{resume_path.split('_')[-1].split('.')[0]}",
            )
        ]

    with launch_simulation(env_cfg, args_cli):
        return _run_play(env_cfg, agent_cfg, agent_cfg_dict, log_dir, resume_path)


def _get_kit_visualizer(env):
    """Return the active Kit visualizer from the environment's simulation, if any."""
    sim = getattr(env.unwrapped, "sim", None)
    if sim is None:
        return None
    for viz in getattr(sim, "visualizers", []):
        if getattr(getattr(viz, "cfg", None), "visualizer_type", None) == "kit":
            return viz
    return None


def _run_play(env_cfg, agent_cfg, agent_cfg_dict, log_dir: str, resume_path: str):
    """Run shadowing policy inference inside an active simulation runtime."""
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

    from instinctlab.managers.reward_manager import MultiRewardManager
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    kit_visualizer = _get_kit_visualizer(env)
    if args_cli.cam_rotate_speed is not None and kit_visualizer is None:
        raise RuntimeError("--cam_rotate_speed requires --viz kit so Isaac Lab creates a Kit visualizer.")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # react to custom play arguments
    if args_cli.no_terminate:
        # NOTE: This is only applicable with shadowing task
        env.unwrapped.termination_manager._term_cfgs = [
            env.unwrapped.termination_manager._term_cfgs[
                env.unwrapped.termination_manager._term_names.index("dataset_exhausted")
            ]
        ]
        env.unwrapped.termination_manager._term_names = ["dataset_exhausted"]

    # start mpjpe buffer if assigned.
    if args_cli.mpjpe:
        mpjpe_b, mpjpe_w = [], []

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # obtain the teacher policy for analysis
    if args_cli.play_teacher:
        teacher_policy = ppo_runner.alg.get_teacher_actions
    else:
        teacher_policy = None
    # obtain the live plotter for comparison
    if args_cli.compare_actions:
        from instinctlab.utils.live_plotter import LivePlotter

        live_plotter = LivePlotter(keys=["policy", "teacher"])
    else:
        live_plotter = None

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, infos = env.get_observations()
            ppo_runner.export_as_onnx(obs, export_model_dir)

    # reset environment
    obs, infos = env.get_observations()
    timestep = 0
    print("[INFO]: Reset environment")
    # simulate environment
    print(env.unwrapped.monitor_manager)
    total_success = 0
    total_traj = 0
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # get teacher actions
            if teacher_policy is not None:
                teacher_actions = teacher_policy(infos["observations"]["critic"])
            else:
                teacher_actions = None
            # get base errors
            monitor = env.unwrapped.monitor_manager._terms["shadowing_position_stats"]
            robot_base_pos = monitor._robot_base_pos  # [number_env, 3]
            reference_base_pos = monitor._reference_base_pos  # [number_env, 3]
            pos_error = robot_base_pos - reference_base_pos  # (N, 3)
            if timestep == 1:
                print(f"First frame after reset, example robot base position: {robot_base_pos[0]}")

            xy_error = torch.norm(pos_error[:, :2], dim=1)  # (N,)
            z_error = torch.abs(pos_error[:, 2])  # (N,)

            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            if teacher_policy is not None:
                obs, rewards, dones, infos = env.step(teacher_actions)
            else:
                obs, rewards, dones, infos = env.step(actions)
            if live_plotter is not None:
                live_plotter.plot([actions[0, 12].cpu().numpy(), teacher_actions[0, 12].cpu().numpy()], x=timestep)
            # print(infos)
            # print(f"obs: {obs}")
            # print(f"dones: {dones}, dones.shape: {dones.shape}")
        timestep += 1

        # override reward terms if auxiliary reward is enabled
        if args_cli.aux_reward:
            # NOTE: This is only applicable when reward_term has `.reward` to be overridden
            aux_rewards = ppo_runner.alg.compute_auxiliary_reward(infos["observations"])
            for aux_reward_name, aux_reward in aux_rewards.items():
                aux_term_cfg = env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)  # type: ignore
                aux_term_cfg.func.reward[:] = aux_reward * getattr(ppo_runner.alg, aux_reward_name + "_coef", 1.0)

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if dones.sum() == dones.numel():  # all environments has ended
            print(
                "info[log][Episode_Monitor/shadowing_link_pos_w_link_pos_error]:",
                infos["log"]["Episode_Monitor/shadowing_link_pos_w_link_pos_error"],
            )
            print(
                "info[log][Episode_Monitor/shadowing_link_pos_b_link_pos_error]:",
                infos["log"]["Episode_Monitor/shadowing_link_pos_b_link_pos_error"],
            )
            # 1. get xy_error and z_error
            print(xy_error)
            print(z_error)
            # 2. identify if the trajectory is successful
            XY_THRESHOLD = 1.0
            Z_THRESHOLD = 0.1

            xy_success = xy_error <= XY_THRESHOLD
            z_success = z_error <= Z_THRESHOLD
            success_per_env = xy_success & z_success  # (N,)
            traj_success = success_per_env.sum().item()
            num_traj = dones.numel()
            # 3. write back or update status
            total_traj += num_traj
            total_success += traj_success
            print(f"Accumulated Success Rate: {total_success}/{total_traj}")
            if total_traj >= 100:
                print(f"Final Success Rate: {total_success}/{total_traj}")
                # write to txt in the following format:
                # randomize_pose_range_x | randomize_pose_range_y | total_success | total_traj # the first two, see: perceptive_shadowing_cfg.py
                break

        if args_cli.cam_rotate_speed is not None:
            # VisualizerCfg eye/lookat are offsets from the controller's current asset-root origin.
            lookat = np.asarray(env_cfg.sim.default_visualizer_cfg.lookat)
            eye_offset = np.asarray(env_cfg.sim.default_visualizer_cfg.eye) - lookat
            angle = args_cli.cam_rotate_speed * timestep * env.unwrapped.step_dt
            rotmat = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            eye_offset = rotmat @ eye_offset
            eye = lookat + eye_offset
            kit_visualizer.cfg.eye = tuple(eye.tolist())
            kit_visualizer.cfg.lookat = tuple(lookat.tolist())
            kit_visualizer.reapply_origin()

    # close the simulator
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}_0000.mp4"),
            ]
        )
    return total_success, total_traj


if __name__ == "__main__":
    # run the main function
    import numpy as np

    results = []
    x = f"{args_cli.x_offset:.2f}" if args_cli.x_offset is not None else None
    y = f"{args_cli.y_offset:.2f}" if args_cli.y_offset is not None else None
    print(f"\n=== Running grid: x={x}, y={y} ===")
    total_success, total_traj = main()
    if args_cli.x_offset is not None and args_cli.y_offset is not None:
        import pandas as pd

        results.append(
            {"x": args_cli.x_offset, "y": args_cli.y_offset, "total_success": total_success, "total_traj": total_traj}
        )
        print(f"x: {x}, y: {y}, total_success: {total_success}, total_traj: {total_traj}")
        csv_file = "grid_search_results.csv"
        df = pd.DataFrame(results)

        file_exists = os.path.isfile(csv_file)
        df.to_csv(csv_file, mode="a", header=not file_exists, index=False)
