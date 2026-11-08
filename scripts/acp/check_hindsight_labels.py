"""Compare the stage-2 hindsight contact labels with the reference contact targets of the same rollout steps.

Each clip plays once from its first frame (one environment per clip, no randomization or observation noise) with the
tracker of ``CKPT`` fed the reference targets. Every step records the reference targets the tracker observes
(``contact_schedule`` command) and what the hindsight labeler records (vertical contact force, contact-body
positions, base pose). The recorded trajectories are then labeled with the labeler's own function
(``label_next_contacts``, the task's ``contact_labeler`` settings), and per body the labels are compared with the
reference targets of the same step: how often each is valid, their agreement, and the distance (xy and z) where both
are valid. Episodes are compared whether or not they would be admitted.

Usage (inside the container)::

    TASK=Instinct-ACP-Planner-G1-v0 CKPT=<stage-2 run>/model_0.pt isaaclab -p scripts/acp/check_hindsight_labels.py
"""

import os
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = os.environ.get("TASK", "Instinct-ACP-Planner-G1-v0")
CKPT = os.environ["CKPT"]

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser, argv=["--task", TASK, "--num_envs", "1", "--run_name", "check", "--viz", "none"], require_run_name=True
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, agent_cfg = cli.resolve_train_cfg(TASK, overrides)
motion_name, motion_cfg = next(iter(env_cfg.scene.motion_reference.motion_buffers.items()))
motion_cfg.motion_start_from_middle_range = [0.0, 0.0]
motion_cfg.motion_bin_length_s = None
motion_cfg.env_starting_stub_sampling_strategy = "independent"
for group in ("policy", "planner"):
    if getattr(env_cfg.observations, group, None) is not None:
        getattr(env_cfg.observations, group).enable_corruption = False
env_cfg.episode_length_s = 1.0e5
for name in (
    "add_joint_default_pos",
    "base_com",
    "randomize_actuator_gains",
    "randomize_rigid_body_mass",
    "randomize_ray_offsets",
    "bin_fail_counter_smoothing",
):
    setattr(env_cfg.events, name, None)
for name in list(vars(env_cfg.curriculum)):
    setattr(env_cfg.curriculum, name, None)
env_cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
env_cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
env_cfg.events.reset_robot.params["randomize_pose_range"] = {}
env_cfg.events.reset_robot.params["randomize_velocity_range"] = {}
env_cfg.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
# also report the contact force from the terrain alone (the labeler uses the net force, which includes self-contacts)
env_cfg.scene.contact_forces.filter_prim_paths_expr = ["/World/ground/terrain/mesh"]

import yaml  # noqa: E402

with open(os.path.expanduser(motion_cfg.metadata_yaml)) as f:
    NUM_CLIPS = len(yaml.safe_load(f)["motion_files"])
env_cfg.scene.num_envs = NUM_CLIPS

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    from instinct_rl.runners import OnPolicyRunner

    import isaaclab.utils.math as math_utils

    from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
    from instinctlab.tasks.shadowing.acp.contact_schedule import heading
    from instinctlab.tasks.shadowing.acp.hindsight_labeler import label_next_contacts
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    def _clip_per_env(self, assigned_ids):
        if len(assigned_ids) > 0:
            selection = self._assigned_env_motion_selection
            selection[assigned_ids] = (assigned_ids.long() % self.num_trajectories).to(selection)

    def _start_at_zero(self, assigned_ids):
        self._motion_buffer_start_time_s[assigned_ids] = 0.0

    AmassMotion._sample_assigned_env_motion_selection = _clip_per_env
    AmassMotion._sample_env_motion_start_time = _start_at_zero

    sim = gym.make(TASK, cfg=env_cfg)
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    device = runner.device
    alg = runner.alg
    ckpt = torch.load(CKPT, weights_only=True, map_location="cpu")
    alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
    alg.planner.load_state_dict(ckpt["planner_state_dict"])
    for name, normalizer in runner.normalizers.items():
        normalizer.load_state_dict(ckpt[f"{name}_normalizer_state_dict"])
    runner.eval_mode()
    policy_normalizer = runner.normalizers.get("policy")
    env.set_training_iteration(0)  # gate 0: the tracker gets the reference targets

    term = unwrapped.command_manager.get_term("contact_schedule")
    labeler_cfg = env_cfg.contact_labeler
    sensor = unwrapped.scene.sensors[labeler_cfg.sensor_name]
    robot = unwrapped.scene[labeler_cfg.asset_name]
    sensor_ids = sensor.find_bodies(labeler_cfg.body_names, preserve_order=True)[0]
    robot_ids = robot.find_bodies(labeler_cfg.body_names, preserve_order=True)[0]
    if list(labeler_cfg.body_names) != list(term.body_names):
        raise RuntimeError(f"labeler bodies {labeler_cfg.body_names} differ from the command's {term.body_names}")
    num_bodies = len(term.body_names)
    horizon_steps = int(round(labeler_cfg.horizon_s / unwrapped.step_dt))
    max_steps = 1500

    sim.reset()
    obs, extras = env.get_observations()
    running = torch.ones(NUM_CLIPS, dtype=torch.bool, device=device)
    length = torch.zeros(NUM_CLIPS, dtype=torch.long, device=device)
    rec = {k: [] for k in ("ref_target", "ref_valid", "contact", "terrain_contact", "body_pos", "root_pos", "root_yaw")}
    with torch.no_grad():
        for _ in range(max_steps):
            # the state the tracker acts from, as the labeler records it after each step
            rec["ref_target"].append(term.command.reshape(NUM_CLIPS, num_bodies, 3).clone())
            rec["ref_valid"].append(term.target_valid.clone())
            rec["contact"].append(sensor.data.net_forces_w.torch[:, sensor_ids, 2].abs() > labeler_cfg.force_threshold)
            terrain_force_z = sensor.data.force_matrix_w.torch[:, sensor_ids, :, 2].sum(dim=-1)
            rec["terrain_contact"].append(terrain_force_z.abs() > labeler_cfg.force_threshold)
            rec["body_pos"].append(robot.data.body_link_pos_w.torch[:, robot_ids].clone())
            rec["root_pos"].append(robot.data.root_pos_w.torch.clone())
            rec["root_yaw"].append(heading(math_utils.yaw_quat(robot.data.root_quat_w.torch)))
            extra_obs = {k: v.to(device) for k, v in extras["observations"].items()}
            normalized = policy_normalizer(obs) if policy_normalizer is not None else obs
            composed = alg.compose_policy_obs(normalized, extra_obs, target_noise=False)
            env.set_policy_feedback(alg.get_policy_feedback())
            obs, _, dones, extras = env.step(alg.actor_critic.act_inference(composed))
            length += running.long()
            running &= ~dones.reshape(-1).bool().to(device)
            if not bool(running.any()):
                break
    data = {k: torch.stack(v, dim=1) for k, v in rec.items()}  # (N, L, ...)
    label_pos, label_valid = label_next_contacts(
        data["contact"],
        data["body_pos"],
        data["root_pos"],
        data["root_yaw"],
        length,
        horizon_steps,
        labeler_cfg.min_contact_steps,
        labeler_cfg.footprint,
    )
    steps = torch.arange(data["contact"].shape[1], device=device)
    in_range = (steps.unsqueeze(0) < length.unsqueeze(1)).unsqueeze(-1)  # (N, L, 1)
    ref_valid = data["ref_valid"] & in_range
    label_valid = label_valid & in_range
    both = ref_valid & label_valid
    diff = label_pos - data["ref_target"]
    print(
        f"[check] {NUM_CLIPS} clips, {int(length.sum())} steps, horizon {horizon_steps} steps, "
        f"min contact {labeler_cfg.min_contact_steps} steps, footprint {labeler_cfg.footprint}"
    )
    for b, name in enumerate(term.body_names):
        n_ref, n_lab, n_both = int(ref_valid[..., b].sum()), int(label_valid[..., b].sum()), int(both[..., b].sum())
        mask = both[..., b]
        d = diff[..., b, :][mask]
        xy = d[:, :2].norm(dim=-1) if n_both else torch.zeros(1, device=device)
        z = d[:, 2] if n_both else torch.zeros(1, device=device)
        print(
            f"[check] {name:22s} ref valid {n_ref:6d}  label valid {n_lab:6d}  both {n_both:6d}  "
            f"xy diff median {float(xy.median()):.3f} p90 {float(xy.quantile(0.9)):.3f} m  "
            f"z diff median {float(z.median()):+.3f} m"
        )
    # for the feet: is the xy difference along the walking direction (onset timing) or sideways?
    for b, name in enumerate(term.body_names[:2]):
        mask = both[..., b]
        d = diff[..., b, :][mask]
        ref_x = data["ref_target"][..., b, 0][mask]
        print(
            f"[check] {name}: mean diff x {float(d[:, 0].mean()):+.3f} y {float(d[:, 1].mean()):+.3f} m; "
            f"corr(diff x, ref x) {float(torch.corrcoef(torch.stack([d[:, 0], ref_x]))[0, 1]):+.2f}"
        )
    terrain_pos, terrain_valid = label_next_contacts(
        data["terrain_contact"],
        data["body_pos"],
        data["root_pos"],
        data["root_yaw"],
        length,
        horizon_steps,
        labeler_cfg.min_contact_steps,
        labeler_cfg.footprint,
    )
    terrain_valid = terrain_valid & in_range
    print("[check] contact steps (net force | terrain force) and labels from the terrain force only:")
    for b, name in enumerate(term.body_names):
        net = int((data["contact"][..., b] & in_range[..., 0]).sum())
        ter = int((data["terrain_contact"][..., b] & in_range[..., 0]).sum())
        mask = ref_valid[..., b] & terrain_valid[..., b]
        xy = (terrain_pos[..., b, :2] - data["ref_target"][..., b, :2])[mask].norm(dim=-1)
        med = float(xy.median()) if mask.any() else 0.0
        print(
            f"[check] {name:22s} contact steps net {net:6d} terrain {ter:6d}  terrain-label valid "
            f"{int(terrain_valid[..., b].sum()):6d}  xy diff median {med:.3f} m"
        )
    sim.close()
