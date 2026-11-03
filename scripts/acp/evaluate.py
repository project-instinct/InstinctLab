"""Checkpoint evaluation of an ACP tracker on its reference clips (stage-1 gate ACP-G2 of the reproduction plan).

Every clip of the task's motion buffer plays once from each start phase in ``START_PHASES`` (fractions of the clip
duration; default the first frame and the middle), one environment per clip, with friction 1.0, no domain
randomization, no observation noise and a deterministic policy. A clip succeeds when it reaches its end from that
start without a failure termination (the task's base height and tilt terminations). Progress is the fraction of the
remaining clip that was played. During the
rollouts the geometric contact labels are compared with the simulated contacts (net force above 1 N) of the 9
contact bodies, and the contact position error is the distance between robot and reference body while the
reference body is in contact.

Usage (inside the container)::

    CKPT=<run>/model_N.pt isaaclab -p scripts/acp/evaluate.py
    RUN=<run dir> EVERY=2000 isaaclab -p scripts/acp/evaluate.py      # every such checkpoint not yet in eval.csv
    CKPT=... START_PHASES=0,0.25,0.5,0.75 isaaclab -p scripts/acp/evaluate.py

Results go to ``<run dir>/eval.csv`` (one row per checkpoint and start phase) and ``<checkpoint>_eval.json``.
"""

import glob
import json
import os
import re
import time
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = os.environ.get("TASK", "Instinct-ACP-Tracker-G1-v0")
CKPT = os.environ.get("CKPT")
RUN = os.environ.get("RUN")
EVERY = int(os.environ.get("EVERY", "1"))
START_PHASES = [float(p) for p in os.environ.get("START_PHASES", "0,0.5").split(",")]
CONTACT_FORCE_THRESHOLD = 1.0
if (CKPT is None) == (RUN is None):
    raise SystemExit("set exactly one of CKPT (one checkpoint) or RUN (every checkpoint of a run)")

FAILURE_TERMS = ("base_pos_too_far", "base_pg_too_far", "link_pos_too_far")

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", "1", "--run_name", "evaluate", "--viz", "none"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, agent_cfg = cli.resolve_train_cfg(TASK, overrides)
motion_name, motion_cfg = next(iter(env_cfg.scene.motion_reference.motion_buffers.items()))
motion_cfg.motion_start_from_middle_range = [0.0, 0.0]
motion_cfg.motion_bin_length_s = None
motion_cfg.env_starting_stub_sampling_strategy = "independent"
env_cfg.observations.policy.enable_corruption = False
env_cfg.episode_length_s = 1.0e5  # the clip end, not a time limit, ends the episode
env_cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
env_cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
env_cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
for name in (
    "add_joint_default_pos",
    "base_com",
    "randomize_actuator_gains",
    "randomize_rigid_body_mass",
    "randomize_ray_offsets",
    "bin_fail_counter_smoothing",
):
    setattr(env_cfg.events, name, None)
for name in list(vars(env_cfg.curriculum)):  # start sampling curricula (CSI) would move the start from frame 0
    setattr(env_cfg.curriculum, name, None)
env_cfg.events.reset_robot.params["randomize_pose_range"] = {}
env_cfg.events.reset_robot.params["randomize_velocity_range"] = {}
env_cfg.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)

import yaml  # noqa: E402

with open(os.path.expanduser(motion_cfg.metadata_yaml)) as f:
    NUM_CLIPS = len(yaml.safe_load(f)["motion_files"])
env_cfg.scene.num_envs = NUM_CLIPS


def checkpoints() -> list[str]:
    if CKPT is not None:
        return [CKPT]
    found = [(int(re.search(r"model_(\d+)\.pt$", p).group(1)), p) for p in glob.glob(os.path.join(RUN, "model_*.pt"))]
    return [p for it, p in sorted(found) if it % EVERY == 0]


with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinct_rl.runners import OnPolicyRunner

    from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    # Environment i always plays clip i.
    def _clip_per_env(self, assigned_ids):
        if len(assigned_ids) > 0:
            selection = self._assigned_env_motion_selection
            selection[assigned_ids] = (assigned_ids.long() % self.num_trajectories).to(selection)

    AmassMotion._sample_assigned_env_motion_selection = _clip_per_env

    # Every clip starts at the evaluated phase of its duration.
    start_phase = [0.0]

    def _start_at_phase(self, assigned_ids):
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        sequences = self._all_motion_sequences
        clip_length_s = sequences.buffer_length[motion_ids] / sequences.framerate[motion_ids]
        self._motion_buffer_start_time_s[assigned_ids] = start_phase[0] * clip_length_s

    AmassMotion._sample_env_motion_start_time = _start_at_phase

    sim = gym.make(TASK, cfg=env_cfg)
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    buffer = unwrapped.scene["motion_reference"].motion_buffers[motion_name]
    if buffer.num_trajectories != NUM_CLIPS:
        raise RuntimeError(f"the buffer holds {buffer.num_trajectories} clips, expected {NUM_CLIPS}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    device = runner.device
    root = os.path.expanduser(motion_cfg.path)
    clip_names = [os.path.relpath(p, root) for p in buffer._all_motion_files]
    categories = [n.split("/")[1] if n.startswith("php/") else n.split("/")[0] for n in clip_names]
    term = unwrapped.command_manager.get_term("contact_schedule")
    sensor = unwrapped.scene.sensors["contact_forces"]
    sensor_body_ids, _ = sensor.find_bodies(term.body_names, preserve_order=True)
    term_names = list(unwrapped.termination_manager.active_terms)
    step_dt = float(unwrapped.step_dt)

    @torch.no_grad()
    def evaluate(policy, phase: float) -> dict:
        start_phase[0] = phase
        sim.reset()
        obs, _ = env.get_observations()
        clip = buffer._assigned_env_motion_selection.to(device)
        if not torch.equal(clip, torch.arange(NUM_CLIPS, device=device)):
            raise RuntimeError("environments are not assigned one clip each")
        # control steps from the start to the clip's last frame
        clip_steps = (torch.floor(buffer.assigned_motion_lengths.to(device) / step_dt + 1e-6) - 1.0).clamp_min(1.0)

        running = torch.ones(NUM_CLIPS, dtype=torch.bool, device=device)
        failed = torch.zeros(NUM_CLIPS, dtype=torch.bool, device=device)
        steps = torch.zeros(NUM_CLIPS, device=device)
        num_bodies = len(term.body_names)
        agree = torch.zeros(num_bodies, device=device)
        label_pos = torch.zeros(num_bodies, device=device)
        label_and_sim = torch.zeros(num_bodies, device=device)
        sim_pos = torch.zeros(num_bodies, device=device)
        frames = torch.zeros((), device=device)
        pos_error = torch.zeros(num_bodies, device=device)
        limit = int(clip_steps.max()) + 10
        for _ in range(limit):
            # contact statistics of the state the policy acts from
            label = term.reference_contact
            sim_contact = sensor.data.net_forces_w.torch[:, sensor_body_ids].norm(dim=-1) > CONTACT_FORCE_THRESHOLD
            live = running.unsqueeze(-1)
            agree += ((label == sim_contact) & live).sum(0)
            label_pos += (label & live).sum(0)
            sim_pos += (sim_contact & live).sum(0)
            label_and_sim += (label & sim_contact & live).sum(0)
            frames += running.sum()
            body_pos = term.robot.data.body_link_pos_w.torch[:, term.robot_body_ids]
            ref_pos = term.motion_reference.reference_frame.link_pos_w[:, 0, term.reference_link_ids]
            pos_error += ((body_pos - ref_pos).norm(dim=-1) * (label & live)).sum(0)

            obs, _, dones, _ = env.step(policy(obs))
            steps += running.float()
            done = dones.reshape(-1).bool().to(device)
            fail = torch.stack([unwrapped.termination_manager.get_term(n) for n in FAILURE_TERMS if n in term_names])
            failed |= running & done & fail.any(0).to(device)
            running &= ~done
            if not bool(running.any()):
                break
        if bool(running.any()):
            raise RuntimeError(f"{int(running.sum())} clips did not end within {limit} steps")

        success = ~failed
        per_category = {}
        for c in sorted(set(categories)):
            mask = torch.tensor([x == c for x in categories], device=device)
            per_category[c] = {"clips": int(mask.sum()), "success": float(success[mask].float().mean())}
        per_body = {
            name: {
                "agreement": round(float(agree[i] / frames), 4),
                "label_precision": round(float(label_and_sim[i] / label_pos[i].clamp_min(1)), 4),
                "label_recall": round(float(label_and_sim[i] / sim_pos[i].clamp_min(1)), 4),
                "contact_pos_error_m": round(float(pos_error[i] / label_pos[i].clamp_min(1)), 4),
            }
            for i, name in enumerate(term.body_names)
        }
        return {
            "start_phase": phase,
            "success": float(success.float().mean()),
            "n_failed": int(failed.sum()),
            "per_category": per_category,
            "per_body": per_body,
            "failed": [clip_names[i] for i in failed.nonzero().flatten().tolist()],
            "progress": {clip_names[i]: round(float(steps[i] / clip_steps[i]), 3) for i in range(NUM_CLIPS)},
        }

    out_csv = None if RUN is None else os.path.join(RUN, "eval.csv")
    done_paths = set()
    if out_csv is not None and os.path.exists(out_csv):
        done_paths = {line.split(",")[1] for line in open(out_csv).read().strip().splitlines()[1:]}
    elif out_csv is not None:
        with open(out_csv, "w") as f:
            f.write("iteration,checkpoint,start_phase,success,n_failed,per_category,eval_at\n")
    for path in [p for p in checkpoints() if os.path.basename(p) not in done_paths]:
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
        for name, normalizer in runner.normalizers.items():
            normalizer.load_state_dict(ckpt[f"{name}_normalizer_state_dict"])
        policy = runner.get_inference_policy(device)
        results = {"checkpoint": path, "iteration": ckpt.get("iter"), "by_start_phase": {}}
        for phase in START_PHASES:
            t0 = time.time()
            res = evaluate(policy, phase)
            results["by_start_phase"][str(phase)] = res
            summary = " ".join(f"{c}={v['success']:.0%}" for c, v in res["per_category"].items())
            print(
                f"[eval] {os.path.basename(path)} (iteration {results['iteration']}) from phase {phase}: success"
                f" {res['success']:.1%}, failed {res['n_failed']}/{NUM_CLIPS}; {summary} [{time.time() - t0:.0f} s]",
                flush=True,
            )
            for name, v in res["per_body"].items():
                print(f"[eval]   {name:22s} {v}", flush=True)
            if out_csv is not None:
                per_category = "|".join(f"{c}:{v['success']:.3f}" for c, v in res["per_category"].items())
                with open(out_csv, "a") as f:
                    f.write(
                        f"{results['iteration']},{os.path.basename(path)},{phase},{res['success']:.4f},"
                        f"{res['n_failed']},{per_category},{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
        with open(path.replace(".pt", "_eval.json"), "w") as f:
            json.dump(results, f, indent=1)
    sim.close()
