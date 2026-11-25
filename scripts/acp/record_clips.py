"""Record videos of an ACP tracker checkpoint on chosen reference clips, with the contact debug view.

One environment per clip, each playing its clip once from ``START_PHASE`` with the Play task (no randomization or
observation noise, deterministic policy, reference robot and contact markers). The camera looks at the robot from
its front-right, turning with the reference heading. Every captured step re-aims the camera at each running clip in
turn; a clip's video ends with its episode (the clip end, or a failure). Files are named
``<category>__<clip>__<ok|FAILED>.mp4``.

Usage (inside the container)::

    CKPT=<run>/model_N.pt CLIPS=dwp/,low-climb-76/right_45_walk1.5s_step0.7s isaaclab -p scripts/acp/record_clips.py

``CLIPS`` is a comma list of substrings of the motion files (relative to the dataset root); every clip containing
one is recorded. ``OUT`` is the output directory (default ``<run>/videos/<checkpoint>``).
"""

import os
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = os.environ.get("TASK", "Instinct-ACP-Tracker-G1-Play-v0")
CKPT = os.environ["CKPT"]
CLIPS = [c for c in os.environ["CLIPS"].split(",") if c]
START_PHASE = float(os.environ.get("START_PHASE", "0"))
STRIDE = int(os.environ.get("STRIDE", "2"))  # one frame every n control steps (2 -> 25 fps)
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(CKPT), "videos", os.path.splitext(os.path.basename(CKPT))[0]))
# camera relative to the robot base: ahead, to the right, up (metres); looks at the base raised by LOOK_Z
CAMERA_AHEAD, CAMERA_RIGHT, CAMERA_UP, LOOK_Z = 2.5, 2.0, 2.6, 0.3  # high enough to see over the DWP walls

parser = cli.build_play_parser()
args_cli, overrides = cli.parse_cli(parser, argv=["--task", TASK, "--viz", "kit"], require_run_name=False)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, agent_cfg = cli.resolve_play_cfg(TASK, overrides)
env_cfg.observations.policy.enable_corruption = False
env_cfg.episode_length_s = 1.0e5  # the clip end, not a time limit, ends the episode
env_cfg.scene.motion_reference.visualizing_marker_types = []  # only the contact markers, not every reference link
# This script aims the camera itself (an asset-tracking camera would override it at every render) and renders
# offscreen: a headless Kit visualizer makes the launcher enable the capture extensions render_rgb_array needs.
for visualizer_cfg in env_cfg.sim.visualizer_cfgs:
    visualizer_cfg.backend.origin_type = "world"
    visualizer_cfg.headless = True
motion_cfg = env_cfg.scene.motion_reference.motion_buffers["TerrainMotion"]

import yaml  # noqa: E402

with open(os.path.expanduser(motion_cfg.metadata_yaml)) as f:
    all_files = [m["motion_file"] for m in yaml.safe_load(f)["motion_files"]]
CLIP_IDS = [i for i, name in enumerate(all_files) if any(c in name for c in CLIPS)]
if not CLIP_IDS:
    raise SystemExit(f"no motion file matches {CLIPS}")
env_cfg.scene.num_envs = len(CLIP_IDS)


with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym

    import cv2
    from instinct_rl.runners import OnPolicyRunner

    import isaaclab.utils.math as math_utils

    from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
    from instinctlab.tasks.shadowing.acp.contact_schedule import heading
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

    # environment i plays clip CLIP_IDS[i] from START_PHASE
    clip_ids = torch.tensor(CLIP_IDS)

    def _clip_per_env(self, assigned_ids):
        if len(assigned_ids) > 0:
            selection = self._assigned_env_motion_selection
            selection[assigned_ids] = clip_ids.to(selection.device)[assigned_ids.long() % len(CLIP_IDS)]

    def _start_at_phase(self, assigned_ids):
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        sequences = self._all_motion_sequences
        self._motion_buffer_start_time_s[assigned_ids] = (
            START_PHASE * sequences.buffer_length[motion_ids] / sequences.framerate[motion_ids]
        )

    AmassMotion._sample_assigned_env_motion_selection = _clip_per_env
    AmassMotion._sample_env_motion_start_time = _start_at_phase

    sim = gym.make(TASK, cfg=env_cfg)
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ckpt = torch.load(CKPT, weights_only=True, map_location="cpu")
    runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
    for name, normalizer in runner.normalizers.items():
        normalizer.load_state_dict(ckpt[f"{name}_normalizer_state_dict"])
    policy = runner.get_inference_policy(unwrapped.device)

    viz = next(v for v in unwrapped.sim.visualizers if hasattr(v, "render_rgb_array"))
    robot = unwrapped.scene["robot"]
    manager = unwrapped.scene["motion_reference"]
    reference_offset = torch.tensor(manager.cfg.visualizing_robot_offset)
    os.makedirs(OUT, exist_ok=True)
    names = [all_files[i] for i in CLIP_IDS]
    stems = [
        n.split("/")[-3 if n.startswith("php/") else -2].replace("php-motions-", "")
        + "__"
        + n.split("/")[-1].replace("_retargeted.npz", "").replace("-retargeted.npz", "")
        for n in names
    ]
    fps = max(1, round(1.0 / (float(unwrapped.step_dt) * STRIDE)))
    writers = {}
    num = len(CLIP_IDS)
    running = torch.ones(num, dtype=torch.bool)
    failed = torch.zeros(num, dtype=torch.bool)
    camera_yaw = None

    def frame(eye, target):
        viz.set_camera_view(tuple(eye), tuple(target))
        image = viz.render_rgb_array()
        image = image.numpy() if hasattr(image, "numpy") else image
        return cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)

    sim.reset()
    obs, _ = env.get_observations()
    step = 0
    with torch.inference_mode():
        while bool(running.any()):
            if step % STRIDE == 0:
                base = robot.data.root_pos_w.torch.cpu()
                # turn with the reference heading (smoothed), so the view follows the clip rather than the robot's sway
                yaw = heading(manager.reference_frame.base_quat_w[:, 0]).cpu()
                camera_yaw = yaw if camera_yaw is None else camera_yaw + 0.1 * math_utils.wrap_to_pi(yaw - camera_yaw)
                forward = torch.stack([torch.cos(camera_yaw), torch.sin(camera_yaw), torch.zeros(num)], -1)
                right = torch.stack([torch.sin(camera_yaw), -torch.cos(camera_yaw), torch.zeros(num)], -1)
                for i in running.nonzero().flatten().tolist():
                    look = base[i] + 0.5 * reference_offset + torch.tensor([0.0, 0.0, LOOK_Z])
                    eye = (
                        base[i] + CAMERA_AHEAD * forward[i] + CAMERA_RIGHT * right[i] + torch.tensor([0, 0, CAMERA_UP])
                    )
                    image = frame(eye.tolist(), look.tolist())
                    if stems[i] not in writers:
                        h, w = image.shape[:2]
                        writers[stems[i]] = cv2.VideoWriter(
                            os.path.join(OUT, f"{stems[i]}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    writers[stems[i]].write(image)
            obs, _, dones, _ = env.step(policy(obs))
            done = dones.reshape(-1).bool().cpu()
            failure = (
                torch.stack(
                    [unwrapped.termination_manager.get_term(n) for n in ("base_pos_too_far", "base_pg_too_far")]
                )
                .any(0)
                .cpu()
            )
            failed |= running & done & failure
            running &= ~done
            step += 1

    for writer in writers.values():
        writer.release()
    for i, stem in enumerate(stems):
        tag = "FAILED" if failed[i] else "ok"
        os.replace(os.path.join(OUT, f"{stem}.mp4"), os.path.join(OUT, f"{stem}__{tag}.mp4"))
        print(f"[record] {stem}: {tag}", flush=True)
    print(f"[record] {num} videos in {OUT} ({fps} fps)", flush=True)
    sim.close()
