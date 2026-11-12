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
occupancy precision and recall (planned occupied vs a reference target within the onset horizon and inside the
command's target footprint, i.e. a target the tracker would get) and the position error where both have a target. Stage-2 results go to ``eval_<source>.csv`` and ``<checkpoint>_eval_<source>.json``.
``PLANNER_MIX`` (see below) replaces one part of the planner's targets with the reference's, to find which part the
tracker fails on; its results get the suffix ``_mix-<PLANNER_MIX>``.

Usage (inside the container)::

    CKPT=<run>/model_N.pt isaaclab -p scripts/acp/evaluate.py
    RUN=<run dir> EVERY=2000 isaaclab -p scripts/acp/evaluate.py      # every such checkpoint not yet in eval.csv
    CKPT=... START_PHASES=0,0.25,0.5,0.75 isaaclab -p scripts/acp/evaluate.py

Results go to ``<run dir>/eval.csv`` (one row per checkpoint and start phase) and ``<checkpoint>_eval.json``.
"""

import glob
import json
import numpy as np
import os
import re
import time
import torch
from collections import defaultdict

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = os.environ.get("TASK", "Instinct-ACP-Tracker-G1-v0")
CKPT = os.environ.get("CKPT")
RUN = os.environ.get("RUN")
EVERY = int(os.environ.get("EVERY", "1"))
START_PHASES = [float(p) for p in os.environ.get("START_PHASES", "0,0.5").split(",")]
CONTACT_SOURCE = os.environ.get("CONTACT_SOURCE")  # stage-2 checkpoints only: "planner" (default) or "reference"
# diagnostic: the tracker's reference targets are zeroed outside the hindsight labels' scan footprint
REFERENCE_FOOTPRINT = os.environ.get("REFERENCE_FOOTPRINT", "0") == "1"
if CONTACT_SOURCE not in (None, "planner", "reference"):
    raise SystemExit(f"CONTACT_SOURCE must be planner or reference, got {CONTACT_SOURCE!r}")
# diagnostic (planner contacts only): the tracker gets the planner's targets with one part replaced by the reference's
#   feet_reference      reference targets for the feet, the planner's for the other bodies
#   feet_planner        the planner's targets for the feet, reference targets for the other bodies
#   reference_positions the planner's occupancy, but the reference position where both have a target
#   reference_occupancy the reference's occupancy, the planner's position where both have a target
PLANNER_MIX = os.environ.get("PLANNER_MIX")
# diagnostic: the planner keeps a target when its occupancy probability exceeds this (default: the planner's setting)
OCCUPANCY_THRESHOLD = os.environ.get("OCCUPANCY_THRESHOLD")
# diagnostic (planner contacts only): per-step foot targets of the reference and the planner go to
# <checkpoint>_footholds<suffix>_phase<p>.npz, and the planner's foot occupancy on (up to FOOTHOLD_REPLAY_SAMPLES)
# samples of the run's planner_replay_latest.pt to <checkpoint>_footholds_replay.npz; scripts/acp/analyze_footholds.py
# reads them.
FOOTHOLD_LOG = os.environ.get("FOOTHOLD_LOG", "0") == "1"
FOOTHOLD_REPLAY_SAMPLES = int(os.environ.get("FOOTHOLD_REPLAY_SAMPLES", "200000"))
PLANNER_MIXES = ("feet_reference", "feet_planner", "reference_positions", "reference_occupancy")
if PLANNER_MIX not in (None, *PLANNER_MIXES):
    raise SystemExit(f"PLANNER_MIX must be one of {PLANNER_MIXES}, got {PLANNER_MIX!r}")
if (PLANNER_MIX is not None or FOOTHOLD_LOG) and CONTACT_SOURCE == "reference":
    raise SystemExit("PLANNER_MIX and FOOTHOLD_LOG need CONTACT_SOURCE=planner")
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

    if REFERENCE_FOOTPRINT:
        from instinctlab.tasks.shadowing.acp.contact_schedule import ContactScheduleCommand

        (x_min, x_max), (y_min, y_max) = env_cfg.contact_labeler.footprint
        update_command = ContactScheduleCommand._update_command

        def _update_command_in_footprint(self):
            update_command(self)
            targets = self._target_b.view(self.num_envs, -1, 3)
            inside = (targets[..., 0] >= x_min) & (targets[..., 0] <= x_max)
            inside &= (targets[..., 1] >= y_min) & (targets[..., 1] <= y_max)
            targets *= inside.unsqueeze(-1)

        ContactScheduleCommand._update_command = _update_command_in_footprint

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
    suffix = ("" if source is None else f"_{source}") + ("_footprint" if REFERENCE_FOOTPRINT else "")
    if PLANNER_MIX is not None:
        if not with_planner:
            raise SystemExit("PLANNER_MIX is for stage-2 checkpoints")
        suffix += f"_mix-{PLANNER_MIX}"
    if OCCUPANCY_THRESHOLD is not None:
        if not with_planner:
            raise SystemExit("OCCUPANCY_THRESHOLD is for stage-2 checkpoints")
        alg.planner.occupancy_threshold = float(OCCUPANCY_THRESHOLD)
        suffix += f"_occ{OCCUPANCY_THRESHOLD}"
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
    feet_ids = [term.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    term_names = list(unwrapped.termination_manager.active_terms)
    step_dt = float(unwrapped.step_dt)

    def planner_act(obs, extras):
        """Actions with the planner's targets composed into the policy obs (where the gate is set); the env gets
        the planner's targets as policy feedback, as in training."""
        extra_obs = {k: v.to(device) for k, v in extras["observations"].items()}
        normalized = policy_normalizer(obs) if policy_normalizer is not None else obs
        composed = alg.compose_policy_obs(normalized, extra_obs, target_noise=False)
        feedback = alg.get_policy_feedback()
        env.set_policy_feedback(feedback)
        if PLANNER_MIX is not None:
            composed[:, alg._target_index] = mixed_targets(feedback)
        return alg.actor_critic.act_inference(composed)

    def reference_targets_on() -> torch.Tensor:
        """``(N, B)`` reference targets the tracker gets: valid and inside the command's target footprint (the
        command is zero elsewhere)."""
        return term.target_valid & (term.command.reshape(NUM_CLIPS, -1, 3).abs().sum(dim=-1) > 0.0)

    def mixed_targets(feedback) -> torch.Tensor:
        """Normalized target slot of the PLANNER_MIX diagnostic."""
        planned = feedback[alg.planner_positions_feedback_key].reshape(NUM_CLIPS, -1, 3)
        planned_on = feedback[alg.planner_occupancy_feedback_key] > 0.5
        reference = term.command.reshape(NUM_CLIPS, -1, 3)
        reference_on = reference_targets_on()
        if PLANNER_MIX in ("feet_reference", "feet_planner"):
            feet = torch.zeros(planned.shape[1], dtype=torch.bool, device=device)
            feet[feet_ids] = True
            use_reference = feet if PLANNER_MIX == "feet_reference" else ~feet
            mixed = torch.where(use_reference[None, :, None], reference, planned)
        elif PLANNER_MIX == "reference_positions":
            mixed = torch.where((planned_on & reference_on).unsqueeze(-1), reference, planned)
        else:  # reference_occupancy
            mixed = torch.where((reference_on & ~planned_on).unsqueeze(-1), reference, planned)
            mixed = mixed * reference_on.unsqueeze(-1)
        mixed = mixed.reshape(NUM_CLIPS, -1)
        if alg.policy_normalizer is not None:
            index = alg._target_index
            mixed = (mixed - alg.policy_normalizer.mean[index]) / (
                alg.policy_normalizer.std[index] + alg.policy_normalizer.eps
            )
        return mixed

    last_footholds = [None]

    @torch.no_grad()
    def replay_footholds(run_dir: str) -> dict | None:
        """The planner's foot occupancy and positions on samples of the run's planner replay (its training labels)."""
        path = os.path.join(run_dir, alg.REPLAY_FILE)
        if not os.path.exists(path):
            print(f"[eval] no {path}; skipping the replay footholds", flush=True)
            return None
        replay = torch.load(path, weights_only=True, map_location="cpu")["replay_state_dict"]
        filled = int(replay["filled"])
        index = torch.randperm(filled)[:FOOTHOLD_REPLAY_SAMPLES]
        planner_obs = replay["buffer/planner_obs"][index]
        target_pos = replay["buffer/target_pos"][index][:, feet_ids]
        target_valid = replay["buffer/target_valid"][index][:, feet_ids]
        positions, occupied = [], []
        for start in range(0, len(index), 8192):
            p, o = alg.planner.predict(planner_obs[start : start + 8192].to(device))
            positions.append(p[:, feet_ids].cpu())
            occupied.append(o[:, feet_ids].cpu())
        return {
            "label_on": target_valid.bool().numpy(),
            "label_b": target_pos.numpy(),
            "planned_on": torch.cat(occupied).numpy(),
            "planned_b": torch.cat(positions).numpy(),
        }

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
        foothold_log = defaultdict(list)
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
                valid = reference_targets_on() & live
                both = occupied & valid
                plan_occupied += occupied.sum(0)
                ref_valid += valid.sum(0)
                plan_and_ref += both.sum(0)
                plan_error += ((planned - reference_target).norm(dim=-1) * both).sum(0)
                if FOOTHOLD_LOG:
                    step_log = {
                        "live": running,
                        "reference_on": reference_targets_on()[:, feet_ids],
                        "reference_valid": term.target_valid[:, feet_ids],
                        "reference_target_b": reference_target[:, feet_ids],
                        "reference_target_w": term.target_pos_w[:, feet_ids],
                        "reference_contact": label[:, feet_ids],
                        "planned_on": occupied[:, feet_ids],
                        "planned_b": planned[:, feet_ids],
                        "robot_contact": sim_contact[:, feet_ids],
                        "root_pos_w": term.robot.data.root_pos_w.torch,
                    }
                    for key, value in step_log.items():
                        foothold_log[key].append(value.clone().cpu())
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
        if foothold_log:
            footholds = {key: torch.stack(values).numpy() for key, values in foothold_log.items()}  # (T, N, ...)
            footholds.update(
                failed=failed.cpu().numpy(),
                steps=steps.cpu().numpy(),
                clip_steps=clip_steps.cpu().numpy(),
                categories=np.array(categories),
                clips=np.array(clip_names),
                step_dt=np.array(step_dt),
            )
            last_footholds[0] = footholds

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
        if FOOTHOLD_LOG and with_planner:
            replay = replay_footholds(os.path.dirname(path))
            if replay is not None:
                np.savez_compressed(path.replace(".pt", "_footholds_replay.npz"), **replay)
        results = {"checkpoint": path, "iteration": ckpt.get("iter"), "contact_source": source, "by_start_phase": {}}
        for phase in START_PHASES:
            t0 = time.time()
            res = evaluate(policy, phase)
            results["by_start_phase"][str(phase)] = res
            if last_footholds[0] is not None:
                np.savez_compressed(path.replace(".pt", f"_footholds{suffix}_phase{phase}.npz"), **last_footholds[0])
                last_footholds[0] = None
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
