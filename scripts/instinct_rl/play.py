"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import subprocess
import sys

from isaaclab.app import AppLauncher

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
    "--viz_z_sphere",
    action="store_true",
    default=False,
    help="Record latent z each step; on exit save log_dir/latent_viz/ (PCA PNG + CSV by motion color).",
)
parser.add_argument(
    "--viz_z_project_mode",
    type=str,
    default="pca",
    choices=("pca", "first3"),
    help="3D sphere view: NumPy SVD PCA (with fixed basis npz under latent_viz/) or first3 dims.",
)
parser.add_argument(
    "--viz_z_pca_basis",
    type=str,
    default=None,
    help="Optional path to fixed PCA basis npz (mean, basis). Default: latent_viz/z_sphere_pca_basis.npz when mode=pca.",
)
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

import latent_viz  # isort: skip

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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

    viz_hook = None
    viz_storage = None
    viz_rows: list = []
    viz_motions: list[str] = []
    viz_steps: list[int] = []
    viz_envs: list[int] = []
    viz_out_dir = os.path.join(log_dir, "latent_viz")

    def save_viz_rows(reason: str):
        if not (args_cli.viz_z_sphere and viz_hook is not None):
            return
        if not viz_rows:
            print(f"[latent_viz] No samples captured on {reason}; nothing saved.")
            return
        latent_viz.save_latent_visualization(
            viz_rows,
            viz_motions,
            viz_steps,
            viz_envs,
            viz_out_dir,
            project_mode=args_cli.viz_z_project_mode,
            pca_basis_path=args_cli.viz_z_pca_basis,
        )
        print(f"[latent_viz] Saved {len(viz_rows)} samples on {reason} to {viz_out_dir}")

    if args_cli.viz_z_sphere:
        viz_hook, viz_storage = latent_viz.try_install_vae_decoder_z_hook(ppo_runner.alg.actor_critic)
        if viz_hook is None:
            print("\033[93m[latent_viz] No VAE decoder hook; viz disabled.\033[0m")
        else:
            print(f"[latent_viz] Hook installed. Saving to: {viz_out_dir}")
            latent_viz.warn_if_not_project_to_sphere(ppo_runner.alg.actor_critic)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.export_as_onnx(obs, export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                if viz_hook is not None and viz_storage is not None:
                    z_batch = viz_storage[0]
                    if z_batch is not None:
                        n_e = z_batch.shape[0]
                        mids = latent_viz.get_motion_identifiers_for_envs(env.unwrapped, n_e)
                        for ei in range(n_e):
                            viz_rows.append(z_batch[ei : ei + 1].clone())
                            viz_motions.append(mids[ei] if ei < len(mids) else "unknown_motion")
                            viz_steps.append(timestep)
                            viz_envs.append(ei)
                    viz_storage[0] = None
                if timestep < args_cli.zero_act_until:
                    actions[:] = 0.0
                # env stepping
                obs, rewards, dones, infos = env.step(actions)
                if args_cli.viz_z_sphere and viz_hook is not None and torch.any(dones.to(bool)):
                    save_viz_rows("episode reset")
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
    finally:
        if viz_hook is not None:
            latent_viz.remove_hook(viz_hook)
        if args_cli.viz_z_sphere and viz_hook is not None and viz_rows:
            save_viz_rows("play exit")
        elif args_cli.viz_z_sphere and viz_hook is not None:
            print("[latent_viz] No samples captured; nothing saved.")

    # close the simulator
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
