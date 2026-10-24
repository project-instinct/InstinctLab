"""Script to play a checkpoint of an RL agent from Instinct-RL."""

import gymnasium as gym
import os
import subprocess
import torch

from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.loggings import pack_checkpoint_folder

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

parser = cli.build_play_parser()
cli.add_play_runtime_args(parser)
args_cli, overrides = cli.parse_cli(parser, require_run_name=False)

cli.wait_for_debugger(args_cli.debug)


def main():
    """Launch the configured simulator and play an Instinct-RL policy."""
    # Resolve the run folder from the task defaults, Hydra and the CLI before any saved YAML is layered.
    env_cfg, agent_cfg = cli.resolve_play_cfg(args_cli.task, overrides)
    cli.update_instinct_rl_cfg(agent_cfg, args_cli)
    log_dir, resume_path = cli.play_run_dir(args_cli, agent_cfg)

    env_yaml, agent_yaml = cli.load_saved_cfgs(args_cli, log_dir)
    if env_yaml is not None or agent_yaml is not None:
        env_cfg, agent_cfg = cli.resolve_play_cfg(args_cli.task, overrides, env_yaml=env_yaml, agent_yaml=agent_yaml)
    cli.apply_play_cli(args_cli, env_cfg, agent_cfg)
    agent_cfg.load_run = args_cli.load_run
    env_cfg.log_dir = log_dir

    cli.pre_launch_video_config(
        env_cfg,
        args_cli,
        step_offset=args_cli.video_start_step,
        prefix=cli.checkpoint_video_prefix(resume_path),
    )
    with launch_simulation(env_cfg, args_cli):
        cli.apply_video_recording(env_cfg, args_cli, os.path.join(log_dir, "videos", "play"))
        return _run_play(env_cfg, agent_cfg, log_dir, resume_path)


def _run_play(env_cfg, agent_cfg, log_dir: str, resume_path: str):
    """Run Instinct-RL policy inference inside an active simulation runtime."""
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    env = gym.make(args_cli.task, cfg=env_cfg)

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
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    if agent_cfg.load_run is not None and args_cli.exportonnx:
        log_dir = pack_checkpoint_folder(
            log_dir,
            resume_path,
            ppo_runner.current_learning_iteration,
            additional_file_regex=[r"^driver_info\.txt$"],
        )
        export_model_dir = os.path.join(log_dir, "exported")
        assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
        os.makedirs(export_model_dir, exist_ok=True)
        obs, _ = env.get_observations()
        ppo_runner.export_as_onnx(obs, export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1

        # override reward terms if auxiliary reward is enabled
        if args_cli.aux_reward:
            # NOTE: This is only applicable when reward_term has `.reward` to be overridden
            aux_rewards = ppo_runner.alg.compute_auxiliary_reward(infos["observations"])
            for aux_reward_name, aux_reward in aux_rewards.items():
                aux_term_cfg = env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)  # type: ignore
                aux_term_cfg.func.reward[:] = aux_reward * getattr(ppo_runner.alg, aux_reward_name + "_coef", 1.0)

        # Exit the play loop after recording one video
        if args_cli.video and timestep == args_cli.video_length:
            break

    # close the simulator
    env.close()

    if args_cli.video:
        recorder = env_cfg.video_recorders[0]
        subprocess.run(["code", "-r", os.path.join(recorder.output_dir, f"{recorder.output_filename_prefix}_0000.mp4")])


if __name__ == "__main__":
    # run the main function
    main()
