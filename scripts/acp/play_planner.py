"""Play an ACP stage-2 checkpoint: the contact planner feeds its targets to the tracker (``ContactPlannerPPO``).

``scripts/instinct_rl/play.py`` runs a single actor network. Here every step the planner predicts the contacts from
its observation group, they replace the tracker's contact targets, and the env receives them as policy feedback (for
``track_planned_contact`` and its markers). ``--contact_source reference`` keeps the reference contacts instead, to
compare with stage 1. Both set the hand-off schedules to their end value (eta = 1 or 0).

Usage (inside the container)::

    isaaclab -p scripts/acp/play_planner.py --task Instinct-ACP-Planner-G1-Play-v0 --load_run <run> [--checkpoint ...]
    isaaclab -p scripts/acp/play_planner.py ... --num_envs 1 --exportonnx   # planner.onnx + actor.onnx
"""

import gymnasium as gym
import os
import torch

from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.loggings import pack_checkpoint_folder

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

parser = cli.build_play_parser()
cli.add_play_runtime_args(parser)
parser.add_argument(
    "--contact_source",
    choices=("planner", "reference"),
    default="planner",
    help="Which contact targets the tracker receives",
)
args_cli, overrides = cli.parse_cli(parser, require_run_name=False)

cli.wait_for_debugger(args_cli.debug)

# A training iteration far past the hand-off ramps (eta = 1), or iteration 0 (eta = 0).
HANDOFF_ITERATION = {"planner": 10**9, "reference": 0}


def main():
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
        env_cfg, args_cli, step_offset=args_cli.video_start_step, prefix=cli.checkpoint_video_prefix(resume_path)
    )
    with launch_simulation(env_cfg, args_cli):
        cli.apply_video_recording(env_cfg, args_cli, os.path.join(log_dir, "videos", "play"))
        return _run_play(env_cfg, agent_cfg, log_dir, resume_path)


def _run_play(env_cfg, agent_cfg, log_dir: str, resume_path: str):
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is None:
        raise SystemExit("--load_run is required: the planner and the tracker come from a stage-2 checkpoint")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    alg = runner.alg
    if not hasattr(alg, "compose_policy_obs"):
        raise SystemExit(f"{type(alg).__name__} has no planner; play it with scripts/instinct_rl/play.py")
    runner.eval_mode()
    device = env.unwrapped.device
    policy_normalizer = runner.normalizers.get("policy")

    if args_cli.exportonnx:
        log_dir = pack_checkpoint_folder(log_dir, resume_path, runner.current_learning_iteration)
        export_model_dir = os.path.join(log_dir, "exported")
        if env.unwrapped.num_envs != 1:
            raise SystemExit("Exporting to ONNX needs --num_envs 1.")
        os.makedirs(export_model_dir, exist_ok=True)
        obs, _ = env.get_observations()
        runner.export_as_onnx(obs, export_model_dir)

    env.set_training_iteration(HANDOFF_ITERATION[args_cli.contact_source])
    obs, extras = env.get_observations()
    timestep = 0
    while True:
        with torch.inference_mode():
            obs = obs.to(device)
            extra_obs = {k: v.to(device) for k, v in extras["observations"].items()}
            normalized = policy_normalizer(obs) if policy_normalizer is not None else obs
            composed = alg.compose_policy_obs(normalized, extra_obs, target_noise=False)
            actions = alg.actor_critic.act_inference(composed)
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            env.set_policy_feedback(alg.get_policy_feedback())
            obs, _, _, extras = env.step(actions)
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
    env.close()


if __name__ == "__main__":
    main()
