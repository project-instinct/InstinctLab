"""Checkpoint evaluation in the SuperMimic paper's protocol (arXiv:2610.28480v1, Sec. 4 and App. F.1).

Every clip of the training library plays once from its first frame, one environment per clip, with
foot friction 1.0, no domain randomization, no observation noise and a deterministic policy. A clip
fails when the rollout triggers a training termination (root height or tilt deviation). Its normalized
return is the undiscounted reward sum over the episode divided by the largest value that sum can
take on the clip, so a policy that terminates early scores low.

Usage (inside the container)::

    CKPT=<run>/model_N.pt python scripts/supermimic/evaluate.py
    RUN=<run dir> python scripts/supermimic/evaluate.py           # every checkpoint not yet in eval.csv
    RUN=<run dir> EVERY=1000 WATCH=1 python scripts/supermimic/evaluate.py   # keep scoring new ones
    CKPT=<ckpt> VIZ=newton CLIPS=walk1_subject1,dance2_subject3 python scripts/supermimic/evaluate.py
        # watch it: a viewer, paced at real time, with the reference shown as markers; CLIPS limits
        # the evaluation to the named clips (default all 40)
    CKPT=<ckpt> VIZ=newton_gl VIDEO=1 python scripts/supermimic/evaluate.py
        # record one video per clip (camera following the robot) plus an overview, under
        # <run dir>/videos/<checkpoint>/

Results go to ``<run dir>/eval.csv`` (one row per checkpoint) and ``<checkpoint>_eval.json``.
"""

import glob
import json
import os
import re
import time
import torch

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

TASK = os.environ.get("TASK", "Instinct-SuperMimic-Plane-G1-Newton-v0")
CKPT = os.environ.get("CKPT")
RUN = os.environ.get("RUN")
EVERY = int(os.environ.get("EVERY", "1"))  # only checkpoints whose iteration is a multiple of this
WATCH = os.environ.get("WATCH", "0") == "1"  # keep polling the run directory for new checkpoints
VIZ = os.environ.get("VIZ")  # a visualizer (e.g. "newton"): watch the evaluation at real time
CLIPS = [c for c in os.environ.get("CLIPS", "").split(",") if c]
VIDEO = os.environ.get("VIDEO", "0") == "1"  # record a video per clip (needs VIZ)
VIDEO_STRIDE = int(os.environ.get("VIDEO_STRIDE", "2"))  # one frame every n control steps (2 -> 25 fps)
if VIDEO and not VIZ:
    raise SystemExit("VIDEO=1 needs a visualizer, e.g. VIZ=newton_gl")
if (CKPT is None) == (RUN is None):
    raise SystemExit("set exactly one of CKPT (one checkpoint) or RUN (every checkpoint of a run)")

# Training terminations that mark a clip as failed. Reaching the clip end (``dataset_exhausted``) is
# the successful end of an evaluation episode.
FAILURE_TERMS = ("base_pos_too_far", "base_pg_too_far")
# Largest per-step reward: the positive tracking weights times the control interval (paper Table A.2).
POSITIVE_TERMS = (
    "body_position",
    "body_orientation",
    "body_linear_velocity",
    "body_angular_velocity",
    "root_position",
    "root_orientation",
)

parser = cli.build_train_parser()
args_cli, overrides = cli.parse_cli(
    parser,
    argv=["--task", TASK, "--num_envs", "1", "--run_name", "evaluate"] + (["--viz", VIZ] if VIZ else []),
    require_run_name=True,
)

from isaaclab.app import launch_simulation  # noqa: E402

env_cfg, agent_cfg = cli.resolve_train_cfg(TASK, overrides)

# -- the evaluation simulation (App. F.1) ---------------------------------------------------------------
motion_name, motion_cfg = next(iter(env_cfg.scene.motion_reference.motion_buffers.items()))
motion_cfg.motion_start_from_middle_range = [0.0, 0.0]  # from the first frame
motion_cfg.motion_bin_length_s = None  # no return-based start sampling
motion_cfg.env_starting_stub_sampling_strategy = "independent"  # a clip per env, chosen below
env_cfg.observations.policy.enable_corruption = False  # no observation noise
env_cfg.episode_length_s = 1.0e5  # the clip end, not a time limit, ends the episode
env_cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
env_cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
env_cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
for name in (
    "add_joint_default_pos",
    "base_com",
    "push_robot",
    "bin_fail_counter_smoothing",
    "return_based_reset_sampling",
):
    if hasattr(env_cfg.events, name):
        setattr(env_cfg.events, name, None)
if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "beyond_adaptive_sampling"):
    env_cfg.curriculum.beyond_adaptive_sampling = None
reset_params = env_cfg.events.reset_robot.params
for key in ("randomize_pose_range", "randomize_velocity_range"):
    if key in reset_params:
        reset_params[key] = {axis: (0.0, 0.0) for axis in reset_params[key]}
if "randomize_joint_pos_range" in reset_params:
    reset_params["randomize_joint_pos_range"] = (0.0, 0.0)

# one environment per clip; the clip list is the motion buffer's own file selection
with open(os.path.expanduser(motion_cfg.filtered_motion_selection_filepath)) as f:
    import yaml

    selection = yaml.safe_load(f)["selected_files"]
if CLIPS:
    chosen = [f for f in selection if f.replace("_retargetted.npz", "") in CLIPS]
    if len(chosen) != len(CLIPS):
        raise SystemExit(
            f"unknown clips in CLIPS: {sorted(set(CLIPS) - {f.replace('_retargetted.npz', '') for f in chosen})}"
        )
    os.makedirs("logs/supermimic_eval", exist_ok=True)
    motion_cfg.filtered_motion_selection_filepath = os.path.abspath("logs/supermimic_eval/clips.yaml")
    with open(motion_cfg.filtered_motion_selection_filepath, "w") as f:
        yaml.dump({"selected_files": chosen}, f)
    selection = chosen
NUM_CLIPS = len(selection)
env_cfg.scene.num_envs = NUM_CLIPS
if VIZ:
    # show the reference: a ghost robot posed from the clip, and its links as markers on the robot
    from instinctlab.assets.unitree_g1 import G1_REFERENCE_CFG

    env_cfg.scene.robot_reference = G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference")
    env_cfg.scene.motion_reference.visualizing_marker_types = ["relative_links"]
    env_cfg.scene.motion_reference.debug_vis = True


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

    # Environment i always plays clip i. The buffer otherwise draws a clip per reset from its weights.
    def _clip_per_env(self, assigned_ids):
        if len(assigned_ids) > 0:
            selection = self._assigned_env_motion_selection
            selection[assigned_ids] = (assigned_ids.long() % self.num_trajectories).to(selection)

    AmassMotion._sample_assigned_env_motion_selection = _clip_per_env

    sim = gym.make(TASK, cfg=env_cfg)
    unwrapped = sim.unwrapped
    env = InstinctRlVecEnvWrapper(sim)
    buffer = unwrapped.scene["motion_reference"].motion_buffers[motion_name]
    if buffer.num_trajectories != NUM_CLIPS:
        raise RuntimeError(f"the buffer holds {buffer.num_trajectories} clips, expected {NUM_CLIPS}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    device = runner.device
    clip_names = [os.path.basename(p).replace("_retargetted.npz", "") for p in buffer._all_motion_files]

    step_dt = float(unwrapped.step_dt)
    weights = {n: unwrapped.reward_manager.get_term_cfg(n).weight for n in POSITIVE_TERMS}
    max_step_reward = sum(weights.values()) * step_dt
    term_names = list(unwrapped.termination_manager.active_terms)

    class ClipRecorder:
        """One video per clip from a camera that follows its robot, plus a fixed overview of all clips.

        The visualizer renders one view at a time, so every captured step re-aims the camera at each
        running clip in turn. A clip's video stops when its episode ends (its last frame is the end of
        the clip, or the fall); the robot then resets and is no longer recorded.
        """

        def __init__(self, checkpoint: str):
            import cv2

            self.cv2 = cv2
            self.viz = unwrapped.sim.visualizers[0]
            self.robot = unwrapped.scene["robot"]
            name = os.path.splitext(os.path.basename(checkpoint))[0]
            self.out_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "videos", name)
            os.makedirs(self.out_dir, exist_ok=True)
            self.fps = max(1, round(1.0 / (step_dt * VIDEO_STRIDE)))
            self.writers: dict[str, object] = {}
            self.open = torch.ones(NUM_CLIPS, dtype=torch.bool)
            origins = unwrapped.scene.env_origins.cpu()
            centre = origins.mean(0)
            spread = float((origins - centre).norm(dim=-1).max()) + 3.0
            self.overview = (
                (centre + torch.tensor([spread, -spread, 0.8 * spread])).tolist(),
                (centre + torch.tensor([0.0, 0.0, 0.5])).tolist(),
            )

        def _frame(self, eye, target):
            self.viz.set_camera_view(eye, target)
            frame = self.viz.render_rgb_array()
            frame = frame.numpy() if hasattr(frame, "numpy") else frame
            return self.cv2.cvtColor(frame[..., :3], self.cv2.COLOR_RGB2BGR)

        def _write(self, key: str, frame):
            writer = self.writers.get(key)
            if writer is None:
                h, w = frame.shape[:2]
                writer = self.cv2.VideoWriter(
                    os.path.join(self.out_dir, f"{key}.mp4"), self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
                )
                self.writers[key] = writer
            writer.write(frame)

        def capture(self, running: torch.Tensor):
            self.open &= running.cpu()
            root = self.robot.data.root_pos_w.torch.cpu()
            for i in self.open.nonzero().flatten().tolist():
                x, y, z = root[i].tolist()
                self._write(clip_names[i], self._frame((x + 1.9, y - 1.9, 1.35), (x, y, 0.7)))
            self._write("overview_all_clips", self._frame(*self.overview))

        def close(self, failed: torch.Tensor):
            for writer in self.writers.values():
                writer.release()
            # mark the clips that failed in the file name
            for i in failed.nonzero().flatten().tolist():
                src = os.path.join(self.out_dir, f"{clip_names[i]}.mp4")
                if os.path.exists(src):
                    os.replace(src, os.path.join(self.out_dir, f"{clip_names[i]}_FAILED.mp4"))
            print(f"[eval] videos in {self.out_dir} ({self.fps} fps)", flush=True)

    @torch.no_grad()
    def evaluate(path: str) -> dict:
        # the learners and normalisers only: the replay in the checkpoint is not needed
        # on the CPU: the checkpoint also holds the replay buffer (several GB), which is not needed here
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
        for name, normalizer in runner.normalizers.items():
            normalizer.load_state_dict(ckpt[f"{name}_normalizer_state_dict"])
        policy = runner.get_inference_policy(device)

        sim.reset()
        obs, _ = env.get_observations()
        clip = buffer._assigned_env_motion_selection.to(device)
        if not torch.equal(clip, torch.arange(NUM_CLIPS, device=device)):
            raise RuntimeError("environments are not assigned one clip each")
        if float(buffer._motion_buffer_start_time_s.abs().max()) > 0.0:
            raise RuntimeError("clips do not start from their first frame")
        # largest attainable sum: one maximal reward per control step until the clip's last frame
        clip_steps = torch.floor(buffer.complete_motion_lengths.to(device) / step_dt + 1e-6) - 1.0

        ret = torch.zeros(NUM_CLIPS, device=device)
        steps = torch.zeros(NUM_CLIPS, device=device)
        running = torch.ones(NUM_CLIPS, dtype=torch.bool, device=device)
        failed = torch.zeros(NUM_CLIPS, dtype=torch.bool, device=device)
        limit = int(clip_steps.max()) + 10
        recorder = ClipRecorder(path) if VIDEO else None
        for step in range(limit):
            tic = time.time()
            obs, reward, dones, _ = env.step(policy(obs))
            if recorder is None and VIZ:
                time.sleep(max(0.0, step_dt - (time.time() - tic)))
            reward = reward.reshape(NUM_CLIPS, -1).sum(-1).to(device)
            ret += torch.where(running, reward, 0.0)
            steps += running.float()
            done = dones.reshape(-1).bool().to(device)
            fail = torch.stack(
                [unwrapped.termination_manager.get_term(n) for n in FAILURE_TERMS if n in term_names]
            ).any(0)
            failed |= running & done & fail.to(device)
            running &= ~done
            # after this step's resets are known: a clip that just ended shows its reset pose now
            if recorder is not None and step % VIDEO_STRIDE == 0:
                recorder.capture(running)
            if not bool(running.any()):
                break
        if recorder is not None:
            recorder.close(failed)
        if bool(running.any()):
            raise RuntimeError(f"{int(running.sum())} clips did not end within {limit} steps")

        norm = (ret / (clip_steps * max_step_reward)).clamp(max=1.0)
        return {
            "checkpoint": path,
            "iteration": ckpt.get("iter"),
            "norm_return": float(norm.mean()),
            "norm_return_framewise": float(ret.sum() / (clip_steps.sum() * max_step_reward)),
            "n_failed": int(failed.sum()),
            "failed": [clip_names[i] for i in failed.nonzero().flatten().tolist()],
            "per_clip": {
                clip_names[i]: {
                    "norm_return": round(float(norm[i]), 4),
                    "steps": int(steps[i]),
                    "clip_steps": int(clip_steps[i]),
                }
                for i in range(NUM_CLIPS)
            },
        }

    out_csv = None if RUN is None else os.path.join(RUN, "eval.csv")
    done_paths = set()
    if out_csv is not None and os.path.exists(out_csv):
        done_paths = {line.split(",")[1] for line in open(out_csv).read().strip().splitlines()[1:]}
    elif out_csv is not None:
        with open(out_csv, "w") as f:
            f.write("iteration,checkpoint,norm_return,norm_return_framewise,n_failed,failed,eval_at\n")
    while True:
        todo = [p for p in checkpoints() if os.path.basename(p) not in done_paths]
        if not todo:
            if not WATCH:
                break
            time.sleep(60)
            continue
        path = todo[0]
        # Checkpoints hold the replay (several GB), so one may still be being written: take it only once
        # it has not changed for a minute, and retry later if it still cannot be read.
        if WATCH and time.time() - os.path.getmtime(path) < 60:
            time.sleep(30)
            continue
        t0 = time.time()
        try:
            res = evaluate(path)
        except RuntimeError as err:
            if not WATCH or "zip archive" not in str(err):
                raise
            print(f"[eval] {os.path.basename(path)} not readable yet ({err}); retrying", flush=True)
            time.sleep(60)
            continue
        done_paths.add(os.path.basename(path))
        with open(path.replace(".pt", "_eval.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(
            f"[eval] {os.path.basename(path)} (iteration {res['iteration']}): normalized return"
            f" {res['norm_return']:.4f} (frame-weighted {res['norm_return_framewise']:.4f}),"
            f" failed {res['n_failed']}/{NUM_CLIPS} {res['failed']} [{time.time() - t0:.0f} s]",
            flush=True,
        )
        if out_csv is not None:
            with open(out_csv, "a") as f:
                f.write(
                    f"{res['iteration']},{os.path.basename(path)},{res['norm_return']:.4f},"
                    f"{res['norm_return_framewise']:.4f},{res['n_failed']},{'|'.join(res['failed'])},"
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
    sim.close()
