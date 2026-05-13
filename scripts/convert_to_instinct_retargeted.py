#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Isaac / BeyondMimic expected joint order.
JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def quat_norm_stats(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    norms = np.linalg.norm(quat_wxyz, axis=1)
    return float(norms.min()), float(norms.max()), float(norms.mean())


def save_retargeted_npz(
    out_file: Path,
    fps: float,
    joint_pos: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
) -> None:
    np.savez(
        out_file,
        framerate=np.float32(fps),
        joint_names=np.array(JOINT_NAMES, dtype=object),
        joint_pos=joint_pos.astype(np.float32),
        base_pos_w=base_pos_w.astype(np.float32),
        base_quat_w=base_quat_w.astype(np.float32),
    )


def convert_robot_terrain_npz(src_file: Path, default_fps: float) -> dict:
    data = np.load(src_file, allow_pickle=True)
    if "qpos" not in data.files:
        raise ValueError(f"{src_file} 缺少 qpos 字段")

    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{src_file} qpos 形状异常: {qpos.shape}, 期望 (T, 36)")

    fps = float(np.asarray(data["fps"]).reshape(())) if "fps" in data.files else float(default_fps)

    # OmniRetarget qpos order: [qw, qx, qy, qz, x, y, z, 29 joints]
    base_quat_w = qpos[:, 0:4]  # already wxyz
    base_pos_w = qpos[:, 4:7]
    joint_pos = qpos[:, 7:36]

    return {
        "fps": fps,
        "joint_pos": joint_pos,
        "base_pos_w": base_pos_w,
        "base_quat_w": base_quat_w,
        "T": qpos.shape[0],
    }


def convert_lafan_csv(src_file: Path, csv_fps: float) -> dict:
    arr = np.loadtxt(src_file, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 36:
        raise ValueError(f"{src_file} 列数异常: {arr.shape[1]}, 期望 36")

    # LAFAN G1 order from README:
    # [x, y, z, qx, qy, qz, qw, 29 joints]
    base_pos_w = arr[:, 0:3]
    quat_xyzw = arr[:, 3:7]
    base_quat_w = quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    joint_pos = arr[:, 7:36]

    return {
        "fps": float(csv_fps),
        "joint_pos": joint_pos,
        "base_pos_w": base_pos_w,
        "base_quat_w": base_quat_w,
        "T": arr.shape[0],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OmniRetarget/LAFAN G1 data to instinct retargeted.npz format.")
    parser.add_argument("--robot_terrain_dir", type=str, required=True, help="robot-terrain .npz directory")
    parser.add_argument("--lafan_csv_dir", type=str, required=True, help="LAFAN G1 .csv directory")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument("--robot_terrain_default_fps", type=float, default=30.0)
    parser.add_argument("--lafan_csv_fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rt_dir = Path(args.robot_terrain_dir).expanduser().resolve()
    lf_dir = Path(args.lafan_csv_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    out_rt = out_dir / "robot-terrain"
    out_lf = out_dir / "lafan1_g1"
    ensure_dir(out_rt)
    ensure_dir(out_lf)

    rt_files = sorted(rt_dir.glob("*.npz"))
    lf_files = sorted(lf_dir.glob("*.csv"))

    print(f"[INFO] robot-terrain files: {len(rt_files)}")
    print(f"[INFO] lafan csv files: {len(lf_files)}")

    for src_file in rt_files:
        out_file = out_rt / f"{src_file.stem}.retargeted.npz"
        data = convert_robot_terrain_npz(src_file, args.robot_terrain_default_fps)
        save_retargeted_npz(out_file, data["fps"], data["joint_pos"], data["base_pos_w"], data["base_quat_w"])

        qmin, qmax, qmean = quat_norm_stats(data["base_quat_w"])
        zmin = float(data["base_pos_w"][:, 2].min())
        zmax = float(data["base_pos_w"][:, 2].max())
        print(
            f"[RT ] {src_file.name} -> {out_file.name} | "
            f"T={data['T']} fps={data['fps']:.2f} "
            f"qnorm(min/max/mean)=({qmin:.6f},{qmax:.6f},{qmean:.6f}) "
            f"z(min/max)=({zmin:.6f},{zmax:.6f})"
        )

    for src_file in lf_files:
        out_file = out_lf / f"{src_file.stem}.retargeted.npz"
        data = convert_lafan_csv(src_file, args.lafan_csv_fps)
        save_retargeted_npz(out_file, data["fps"], data["joint_pos"], data["base_pos_w"], data["base_quat_w"])

        qmin, qmax, qmean = quat_norm_stats(data["base_quat_w"])
        zmin = float(data["base_pos_w"][:, 2].min())
        zmax = float(data["base_pos_w"][:, 2].max())
        print(
            f"[LF ] {src_file.name} -> {out_file.name} | "
            f"T={data['T']} fps={data['fps']:.2f} "
            f"qnorm(min/max/mean)=({qmin:.6f},{qmax:.6f},{qmean:.6f}) "
            f"z(min/max)=({zmin:.6f},{zmax:.6f})"
        )

    print(f"[DONE] Output directory: {out_dir}")


if __name__ == "__main__":
    main()
