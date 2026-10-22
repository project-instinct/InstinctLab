# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an Instinct-RL policy."""

import gymnasium as gym
import multiprocessing as mp
import os
import torch
import torch.distributed as dist

import warp as wp
from instinct_rl.runners import OnPolicyRunner

import isaaclab
from isaaclab.app import launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.io import dump_yaml

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(parser, require_run_name=True)

if "LOCAL_RANK" in os.environ:
    args_cli.distributed = True

cli.wait_for_debugger(args_cli.debug)

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


def main():
    """Launch the configured simulator and train with Instinct-RL."""
    from instinctlab.sim.spawners.from_files.asset_cache import ensure_asset_cache_kit_args

    env_cfg, agent_cfg = cli.resolve_train_cfg(args_cli.task, overrides)
    agent_cfg = cli.update_instinct_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires a CUDA device.")

    ensure_asset_cache_kit_args(env_cfg, args_cli)
    cli.pre_launch_video_config(env_cfg, args_cli, video_interval=args_cli.video_interval)
    with launch_simulation(env_cfg, args_cli):
        # Import the simulator-dependent wrapper only after the simulator starts.
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        if args_cli.distributed:
            # Keep device-less Warp launches on this process's resolved distributed device.
            wp.set_device(env_cfg.sim.device)
            dist.init_process_group(backend="nccl")
            auto_affinity()
            rank, world_size = dist.get_rank(), dist.get_world_size()
            env_cfg.seed += rank
            agent_cfg.device = env_cfg.sim.device
            print(f"[INFO] Distributed training with rank {rank} of {world_size} on {agent_cfg.device}.")

        log_root_path = (
            os.path.abspath(os.path.join(cli.LOG_ROOT, agent_cfg.experiment_name))
            if args_cli.logroot is None
            else os.path.abspath(args_cli.logroot)
        )
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

        resume_path = None
        if agent_cfg.resume:
            resume_path = cli.checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO] Resuming experiment from checkpoint: {resume_path}")
        log_dir = os.path.join(
            log_root_path,
            cli.train_log_dir_name(getattr(env_cfg, "run_name", None), agent_cfg.run_name, resume_path),
        )

        env_cfg.log_dir = log_dir
        cli.apply_video_recording(env_cfg, args_cli, os.path.join(log_dir, "videos", "train"))
        env = gym.make(args_cli.task, cfg=env_cfg)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

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
