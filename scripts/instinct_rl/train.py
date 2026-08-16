# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an Instinct-RL policy."""

import argparse
import gymnasium as gym
import multiprocessing as mp
import os
import sys
import torch
import torch.distributed as dist
from datetime import datetime

from instinct_rl.runners import OnPolicyRunner

import isaaclab
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

import instinctlab.tasks  # noqa: F401
from instinctlab.utils.wrappers.instinct_rl.rl_cfg import InstinctRlOnPolicyRunnerCfg

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--logroot", type=str, default=None, help="Override the default log root, typically logs/instinct_rl/."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run distributed training.")
parser.add_argument("--local-rank", type=int, help="Local rank assigned by the distributed launcher.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--cprofile", action="store_true", default=False, help="Enable cProfile.")
cli_args.add_instinct_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# TODO: Remove this workaround once Isaac Lab initializes `/isaaclab/has_gui` itself.
# release/3.0.0-beta2 leaves it unset, preventing the Kit `IsaacLab` window and live monitors
# from being created. This setting concerns the Kit GUI only, not the selected physics backend.
if "kit" in (args_cli.visualizer or []):
    args_cli.kit_args = f"{args_cli.kit_args} --/isaaclab/has_gui=true".strip()

if "LOCAL_RANK" in os.environ:
    args_cli.distributed = True
if args_cli.video:
    args_cli.enable_cameras = True

# Hydra consumes only the arguments not handled above.
sys.argv = [sys.argv[0]] + hydra_args


if args_cli.debug:
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def auto_affinity():
    """Assign the current distributed process a disjoint CPU-core range."""
    rank = int(os.environ["RANK"])
    num_cores = mp.cpu_count() // torch.cuda.device_count()
    core_range = range(rank * num_cores, (rank + 1) * num_cores)
    core_mask = ",".join(map(str, core_range))
    os.system(f"taskset -cp {core_mask} {os.getpid()}")
    print("Affinity auto updated to:", core_mask, "for rank:", rank)


@hydra_task_config(args_cli.task, "instinct_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: InstinctRlOnPolicyRunnerCfg,
):
    """Launch the configured simulator and train with Instinct-RL."""
    from instinctlab.sim.spawners.from_files.asset_cache import ensure_asset_cache_kit_args

    ensure_asset_cache_kit_args(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        # Import the simulator-dependent wrapper only after SimulationApp starts.
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        agent_cfg = cli_args.update_instinct_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        env_cfg.seed = agent_cfg.seed
        if not args_cli.distributed:
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training requires a CUDA device.")

        if args_cli.distributed:
            dist.init_process_group(backend="nccl")
            auto_affinity()
            rank, world_size = dist.get_rank(), dist.get_world_size()
            env_cfg.seed += rank
            agent_cfg.device = env_cfg.sim.device
            print(f"[INFO] Distributed training with rank {rank} of {world_size} on {agent_cfg.device}.")

        log_root_path = (
            os.path.abspath(os.path.join("logs", "instinct_rl", agent_cfg.experiment_name))
            if args_cli.logroot is None
            else os.path.abspath(args_cli.logroot)
        )
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        if getattr(env_cfg, "run_name", None):
            log_dir += f"_{env_cfg.run_name}"
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
            for hydra_arg in hydra_args:
                key, value = hydra_arg.split("=", maxsplit=1)
                log_dir += f"_{key.split('.')[-1]}-{value}"
        log_dir = os.path.join(log_root_path, log_dir)

        resume_path = None
        if agent_cfg.resume:
            if os.path.isabs(agent_cfg.load_run):
                resume_path = get_checkpoint_path(
                    os.path.dirname(agent_cfg.load_run),
                    os.path.basename(agent_cfg.load_run),
                    agent_cfg.load_checkpoint,
                )
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO] Resuming experiment from checkpoint: {resume_path}")
            resume_run_name = os.path.basename(os.path.dirname(resume_path))
            resume_name_parts = resume_run_name.split("_")
            if len(resume_name_parts) >= 2:
                log_dir += f"_from{resume_name_parts[0]}_{resume_name_parts[1]}"

        env_cfg.log_dir = log_dir
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = InstinctRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        runner.add_git_repo_to_log(isaaclab.__file__)
        if resume_path is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        if not args_cli.distributed or dist.get_rank() == 0:
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        profiler = None
        if args_cli.cprofile:
            import cProfile

            profiler = cProfile.Profile()
            print("Profiling enabled; cprofile_stats.profile will be written to the run directory.")
            profiler.enable()

        try:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
            )
        finally:
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(os.path.join(log_dir, "cprofile_stats.profile"))
            env.close()
            if args_cli.distributed:
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
