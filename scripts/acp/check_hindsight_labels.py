"""Compare the stage-2 hindsight contact labels with the reference contact targets of the same rollout steps.

Each clip plays once from its first frame (one environment per clip, no randomization or observation noise) with the
tracker of ``CKPT`` fed the reference targets. Every step records the reference targets the tracker observes
(``contact_schedule`` command) and what the hindsight labeler records (vertical contact force, contact-body
positions, base pose). The recorded trajectories are then labeled with the labeler's own function
(``label_next_contacts``, the task's ``contact_labeler`` settings), and per body the labels are compared with the
reference targets of the same step: how often each is valid and the distance (xy and z) where both are valid.
Episodes are compared whether or not they would be admitted. The labels are computed several ways: from the robot's
terrain contacts with the labeler's settings and with its differences from the dataset's contact definition (minimum
contact length, scan footprint) removed one by one, and from the reference's own contact flags and body positions,
which separates the definition from the robot's timing.

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
    camera = unwrapped.scene.sensors["camera"]
    horizon_steps = int(round(labeler_cfg.horizon_s / unwrapped.step_dt))
    max_steps = 1500

    sim.reset()
    obs, extras = env.get_observations()
    running = torch.ones(NUM_CLIPS, dtype=torch.bool, device=device)
    length = torch.zeros(NUM_CLIPS, dtype=torch.long, device=device)
    rec = {
        k: []
        for k in (
            "ref_target",
            "ref_valid",
            "ref_contact",
            "ref_body_pos",
            "contact",
            "terrain_contact",
            "body_pos",
            "root_pos",
            "root_yaw",
            "target_w",
            "cam_pos",
            "cam_quat",
            "planned",
            "planned_occ",
        )
    }
    with torch.no_grad():
        for _ in range(max_steps):
            # the state the tracker acts from, as the labeler records it after each step
            rec["ref_target"].append(term.command.reshape(NUM_CLIPS, num_bodies, 3).clone())
            rec["ref_valid"].append(term.target_valid.clone())
            rec["ref_contact"].append(term.reference_contact.clone())
            rec["ref_body_pos"].append(
                term.motion_reference.reference_frame.link_pos_w[:, 0, term.reference_link_ids].clone()
            )
            rec["contact"].append(sensor.data.net_forces_w.torch[:, sensor_ids, 2].abs() > labeler_cfg.force_threshold)
            terrain_force_z = sensor.data.force_matrix_w.torch[:, sensor_ids, :, 2].sum(dim=-1)
            rec["terrain_contact"].append(terrain_force_z.abs() > labeler_cfg.force_threshold)
            rec["body_pos"].append(robot.data.body_link_pos_w.torch[:, robot_ids].clone())
            rec["root_pos"].append(robot.data.root_pos_w.torch.clone())
            rec["root_yaw"].append(heading(math_utils.yaw_quat(robot.data.root_quat_w.torch)))
            rec["target_w"].append(term.target_pos_w.clone())
            rec["cam_pos"].append(camera.data.pos_w.torch.clone())
            rec["cam_quat"].append(camera.data.quat_w_ros.torch.clone())
            extra_obs = {k: v.to(device) for k, v in extras["observations"].items()}
            normalized = policy_normalizer(obs) if policy_normalizer is not None else obs
            composed = alg.compose_policy_obs(normalized, extra_obs, target_noise=False)
            feedback = alg.get_policy_feedback()
            env.set_policy_feedback(feedback)
            rec["planned"].append(feedback[alg.planner_positions_feedback_key].reshape(NUM_CLIPS, num_bodies, 3))
            rec["planned_occ"].append(feedback[alg.planner_occupancy_feedback_key] > 0.5)
            obs, _, dones, extras = env.step(alg.actor_critic.act_inference(composed))
            length += running.long()
            running &= ~dones.reshape(-1).bool().to(device)
            if not bool(running.any()):
                break
    data = {k: torch.stack(v, dim=1) for k, v in rec.items()}  # (N, L, ...)
    steps = torch.arange(data["contact"].shape[1], device=device)
    in_range = (steps.unsqueeze(0) < length.unsqueeze(1)).unsqueeze(-1)  # (N, L, 1)
    ref_valid = data["ref_valid"] & in_range
    print(f"[check] {NUM_CLIPS} clips, {int(length.sum())} steps, horizon {horizon_steps} steps")
    no_footprint = ((-1.0e3, 1.0e3), (-1.0e3, 1.0e3))
    # (name, contact flags, body positions, min contact steps, footprint): the labeler's settings, then each of its
    # differences from the dataset's contact definition removed, on the robot's contacts and on the reference's
    variants = [
        (
            "robot, labeler (min 10, footprint)",
            "terrain_contact",
            "body_pos",
            labeler_cfg.min_contact_steps,
            labeler_cfg.footprint,
        ),
        ("robot, min 3, footprint", "terrain_contact", "body_pos", 3, labeler_cfg.footprint),
        ("robot, min 10, no footprint", "terrain_contact", "body_pos", labeler_cfg.min_contact_steps, no_footprint),
        ("robot, min 3, no footprint", "terrain_contact", "body_pos", 3, no_footprint),
        (
            "reference, labeler (min 10, footprint)",
            "ref_contact",
            "ref_body_pos",
            labeler_cfg.min_contact_steps,
            labeler_cfg.footprint,
        ),
        ("reference, min 3, no footprint", "ref_contact", "ref_body_pos", 3, no_footprint),
    ]
    for title, contact_key, pos_key, min_steps, footprint in variants:
        label_pos, label_valid = label_next_contacts(
            data[contact_key],
            data[pos_key],
            data["root_pos"],
            data["root_yaw"],
            length,
            horizon_steps,
            min_steps,
            footprint,
        )
        label_valid = label_valid & in_range
        both = ref_valid & label_valid
        diff = label_pos - data["ref_target"]
        print(f"[check] == labels from {title}")
        if (
            contact_key == "terrain_contact"
            and min_steps == labeler_cfg.min_contact_steps
            and footprint is not no_footprint
        ):
            # the planner of CKPT on the same steps, against the labels it is trained on and against the reference
            planned_occ = data["planned_occ"] & in_range
            for b, name in enumerate(term.body_names[:2]):
                for other, valid, pos in (
                    ("labels", label_valid, label_pos),
                    ("reference", ref_valid, data["ref_target"]),
                ):
                    m = planned_occ[..., b] & valid[..., b]
                    xy = (data["planned"][..., b, :2] - pos[..., b, :2])[m].norm(dim=-1)
                    print(
                        f"[check]   planner vs {other:9s} {name:22s} precision"
                        f" {float(m.sum() / planned_occ[..., b].sum().clamp_min(1)):.2f} recall"
                        f" {float(m.sum() / valid[..., b].sum().clamp_min(1)):.2f} | xy median"
                        f" {float(xy.median()):.3f} mean {float(xy.mean()):.3f} p90 {float(xy.quantile(0.9)):.3f}"
                    )
        for b, name in enumerate(term.body_names):
            mask = both[..., b]
            xy = diff[..., b, :2][mask].norm(dim=-1)
            z = diff[..., b, 2][mask]
            if not bool(mask.any()):
                xy = z = torch.zeros(1, device=device)
            print(
                f"[check]   {name:22s} ref {int(ref_valid[..., b].sum()):6d} label {int(label_valid[..., b].sum()):6d}"
                f" both {int(mask.sum()):6d} | xy median {float(xy.median()):.3f} p90 {float(xy.quantile(0.9)):.3f}"
                f" >0.25 m {float((xy > 0.25).float().mean()):5.1%} | z median {float(z.median()):+.3f}"
            )
    # Can the planner see the next contact? For every step with a reference target: is the target inside the scan
    # footprint (the planner can only select scan points), and inside the depth image (in front of the camera,
    # inside the cropped field of view, within the normalized depth range)? By time until the reference onset.
    contact = data["ref_contact"] & in_range
    previous = torch.cat([torch.zeros_like(contact[:, :1]), contact[:, :-1]], dim=1)
    onset = torch.where(contact & ~previous, steps.view(1, -1, 1), torch.full_like(contact, 10**6, dtype=torch.long))
    next_onset = torch.flip(torch.cummin(torch.flip(onset, dims=[1]), dim=1).values, dims=[1])
    time_to_onset = (next_onset - steps.view(1, -1, 1)).float() * unwrapped.step_dt  # (N, L, B)
    (x_min, x_max), (y_min, y_max) = labeler_cfg.footprint
    target_b = data["ref_target"]
    in_footprint = (target_b[..., 0] >= x_min) & (target_b[..., 0] <= x_max)
    in_footprint &= (target_b[..., 1] >= y_min) & (target_b[..., 1] <= y_max)
    offset = data["target_w"] - data["cam_pos"].unsqueeze(2)
    cam_quat = data["cam_quat"].unsqueeze(2).expand(-1, -1, num_bodies, -1)
    p_cam = math_utils.quat_apply_inverse(cam_quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape(offset.shape)
    intrinsics = camera.data.intrinsic_matrices.torch[0]
    height, width = camera.data.image_shape
    top, bottom, left, right = env_cfg.scene.camera.noise_pipeline["crop_and_resize"].crop_region
    depth_max = env_cfg.scene.camera.noise_pipeline["normalize"].depth_range[1]
    z = p_cam[..., 2]
    u = intrinsics[0, 0] * p_cam[..., 0] / z.clamp_min(1e-6) + intrinsics[0, 2]
    v = intrinsics[1, 1] * p_cam[..., 1] / z.clamp_min(1e-6) + intrinsics[1, 2]
    in_view = (z > 0) & (z <= depth_max) & (u >= left) & (u <= width - right) & (v >= top) & (v <= height - bottom)
    print(
        f"[check] == can the planner see the next reference contact? (scan footprint {labeler_cfg.footprint};"
        f" depth image {height}x{width} cropped {top},{bottom},{left},{right} px, depth <= {depth_max} m)"
    )
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.51)]
    for b, name in enumerate(term.body_names):
        valid = ref_valid[..., b]
        if int(valid.sum()) < 100:
            continue
        cells = []
        for lo, hi in bins:
            m = valid & (time_to_onset[..., b] >= lo) & (time_to_onset[..., b] < hi)
            n = int(m.sum())
            fp = float(in_footprint[..., b][m].float().mean()) if n else 0.0
            vw = float(in_view[..., b][m].float().mean()) if n else 0.0
            cells.append(f"{lo:.2f}-{hi:.2f}s n {n:6d} scan {fp:5.1%} depth {vw:5.1%}")
        far = valid & ~in_footprint[..., b]
        print(f"[check]   {name:22s} " + " | ".join(cells))
        if bool(far.any()):
            xb = target_b[..., b, 0][far]
            print(
                f"[check]   {'':22s} outside the scan: target x median {float(xb.median()):+.2f} m,"
                f" |y| > {y_max} m in {float((target_b[..., b, 1][far].abs() > y_max).float().mean()):.0%}"
            )
    sim.close()
