"""Checkpoint evaluation of an ACP tracker on its reference clips (stage-1 gate ACP-G2 of the reproduction plan).

Every clip of the task's motion buffer plays once from each start phase in ``START_PHASES`` (fractions of the clip
duration; default the first frame and the middle), one environment per clip, with friction 1.0, no domain
randomization, no observation noise and a deterministic policy. A clip succeeds when it reaches its end from that
start without a failure termination (the task's base height and tilt terminations). Progress is the fraction of the
remaining clip that was played. During the
rollouts the geometric contact labels are compared with the simulated contacts (net force above 1 N) of the 9
contact bodies, and the contact position error is the distance between robot and reference body while the
reference body is in contact.

A stage-2 checkpoint (``ContactPlannerPPO``, e.g. ``TASK=Instinct-ACP-Planner-G1-v0``) plays with its planner in the
loop: ``CONTACT_SOURCE=planner`` (default) gives the tracker the planner's targets, ``CONTACT_SOURCE=reference`` the
reference's, as in stage 1. With the planner, its targets are also compared with the reference targets per body:
occupancy precision and recall (planned occupied vs a reference target within the onset horizon) and the position
error where both have a target. Stage-2 results go to ``eval_<source>.csv`` and ``<checkpoint>_eval_<source>.json``.

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
CONTACT_SOURCE = os.environ.get("CONTACT_SOURCE")  # stage-2 checkpoints only: "planner" (default) or "reference"
if CONTACT_SOURCE not in (None, "planner", "reference"):
    raise SystemExit(f"CONTACT_SOURCE must be planner or reference, got {CONTACT_SOURCE!r}")
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
if getattr(env_cfg.observations, "planner", None) is not None:
    env_cfg.observations.planner.enable_corruption = False
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
    alg = runner.alg
    with_planner = hasattr(alg, "compose_policy_obs")
    if CONTACT_SOURCE is not None and not with_planner:
        raise SystemExit(f"CONTACT_SOURCE is for stage-2 checkpoints; {type(alg).__name__} has no planner")
    source = (CONTACT_SOURCE or "planner") if with_planner else None
    suffix = "" if source is None else f"_{source}"
    if with_planner:
        # the planner gate observation at the end of its schedule: 1 (planner) or 0 (reference)
        env.set_training_iteration(10**9 if source == "planner" else 0)
        policy_normalizer = runner.normalizers.get("policy")
    root = os.path.expanduser(motion_cfg.path)
    clip_names = [os.path.relpath(p, root) for p in buffer._all_motion_files]
    categories = [n.split("/")[1] if n.startswith("php/") else n.split("/")[0] for n in clip_names]
    term = unwrapped.command_manager.get_term("contact_schedule")
    sensor = unwrapped.scene.sensors["contact_forces"]
    sensor_body_ids, _ = sensor.find_bodies(term.body_names, preserve_order=True)
    term_names = list(unwrapped.termination_manager.active_terms)
    step_dt = float(unwrapped.step_dt)

    def planner_act(obs, extras):
        """Actions with the planner's targets composed into the policy obs (where the gate is set); the env gets
        the planner's targets as policy feedback, as in training."""
        extra_obs = {k: v.to(device) for k, v in extras["observations"].items()}
        normalized = policy_normalizer(obs) if policy_normalizer is not None else obs
        composed = alg.compose_policy_obs(normalized, extra_obs, target_noise=False)
        env.set_policy_feedback(alg.get_policy_feedback())
        return alg.actor_critic.act_inference(composed)

    @torch.no_grad()
    def evaluate(policy, phase: float) -> dict:
        start_phase[0] = phase
        sim.reset()
        obs, extras = env.get_observations()
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
        # planner vs reference targets
        plan_and_ref = torch.zeros(num_bodies, device=device)
        plan_occupied = torch.zeros(num_bodies, device=device)
        ref_valid = torch.zeros(num_bodies, device=device)
        plan_error = torch.zeros(num_bodies, device=device)
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

            obs = obs.to(device)
            if with_planner:
                actions = planner_act(obs, extras)
                feedback = alg.get_policy_feedback()
                occupied = (feedback[alg.planner_occupancy_feedback_key] > 0.5) & live
                planned = feedback[alg.planner_positions_feedback_key].reshape(NUM_CLIPS, num_bodies, 3)
                reference_target = term.command.reshape(NUM_CLIPS, num_bodies, 3)
                valid = term.target_valid & live
                both = occupied & valid
                plan_occupied += occupied.sum(0)
                ref_valid += valid.sum(0)
                plan_and_ref += both.sum(0)
                plan_error += ((planned - reference_target).norm(dim=-1) * both).sum(0)
            else:
                actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
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
        if with_planner:
            for i, name in enumerate(term.body_names):
                per_body[name].update(
                    {
                        "planned_precision": round(float(plan_and_ref[i] / plan_occupied[i].clamp_min(1)), 4),
                        "planned_recall": round(float(plan_and_ref[i] / ref_valid[i].clamp_min(1)), 4),
                        "planned_vs_reference_m": round(float(plan_error[i] / plan_and_ref[i].clamp_min(1)), 4),
                    }
                )
        return {
            "start_phase": phase,
            "success": float(success.float().mean()),
            "n_failed": int(failed.sum()),
            "per_category": per_category,
            "per_body": per_body,
            "failed": [clip_names[i] for i in failed.nonzero().flatten().tolist()],
            "progress": {clip_names[i]: round(float(steps[i] / clip_steps[i]), 3) for i in range(NUM_CLIPS)},
        }

    out_csv = None if RUN is None else os.path.join(RUN, f"eval{suffix}.csv")
    done_paths = set()
    if out_csv is not None and os.path.exists(out_csv):
        done_paths = {line.split(",")[1] for line in open(out_csv).read().strip().splitlines()[1:]}
    elif out_csv is not None:
        with open(out_csv, "w") as f:
            f.write("iteration,checkpoint,start_phase,success,n_failed,per_category,eval_at\n")
    for path in [p for p in checkpoints() if os.path.basename(p) not in done_paths]:
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
        if with_planner:
            alg.planner.load_state_dict(ckpt["planner_state_dict"])
        for name, normalizer in runner.normalizers.items():
            normalizer.load_state_dict(ckpt[f"{name}_normalizer_state_dict"])
        policy = runner.get_inference_policy(device)
        results = {"checkpoint": path, "iteration": ckpt.get("iter"), "contact_source": source, "by_start_phase": {}}
        for phase in START_PHASES:
            t0 = time.time()
            res = evaluate(policy, phase)
            results["by_start_phase"][str(phase)] = res
            summary = " ".join(f"{c}={v['success']:.0%}" for c, v in res["per_category"].items())
            print(
                f"[eval] {os.path.basename(path)} (iteration {results['iteration']}, contacts {source or 'reference'})"
                f" from phase {phase}: success"
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
        with open(path.replace(".pt", f"_eval{suffix}.json"), "w") as f:
            json.dump(results, f, indent=1)
    sim.close()
