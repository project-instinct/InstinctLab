"""Convert PHP (Perceptive Humanoid Parkour) G1 motion/terrain bundles to an InstinctLab terrain-motion dataset.

Input: the extracted PHP training bundles (``php-motions-*-v1/``), each holding pairs of

- ``<clip>_motion.npz``: Holosoma format at 50 fps. ``joint_pos`` (T, 36) = root pos, root quat (WXYZ), 29 joints;
  ``body_pos_w`` / ``body_quat_w`` (T, 32, ...) for ``body_names``; ``joint_names`` (29,).
- ``<clip>_terrain.npy``: (num_boxes, num_variants, 10) boxes ``[pos_xyz, quat_wxyz, size_xyz]``, box bottoms on the
  ground plane z = 0. An empty box array means flat ground.

Output, one folder per clip (the layout ``motion_matched_metadata_generator.py`` scans)::

    <tgt>/<bundle>/<clip>/<clip>_retargeted.npz   framerate, joint_names, joint_pos, base_pos_w, base_quat_w (WXYZ)
    <tgt>/<bundle>/<clip>/<clip>_terrain.stl      ground plane of ``--size`` + the boxes of one variant
    <tgt>/metadata.yaml

InstinctLab's G1 is torso-rooted, so the base pose is PHP's ``torso_link`` body pose. Each clip and its boxes are
shifted together in xy so that the bounding box of all body positions and box footprints is centered at the origin,
which is where ``motion_matched_terrain`` puts the subterrain center. A clip that does not fit ``--size`` with
``--margin`` raises.

Usage::

    python scripts/php_to_instinct.py --src ~/Datasets/PHP-Parkour_Motions/php-motions-low-step-v1 ... \
        --tgt ~/Datasets/PHP-Parkour_Motions/instinct_variant0 --size 14 12 --validate
"""

import argparse
import glob
import numpy as np
import os
import subprocess
import sys
import torch
import trimesh

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_URDF = os.path.join(
    _THIS_DIR,
    "..",
    "source",
    "instinctlab",
    "instinctlab",
    "assets",
    "resources",
    "unitree_g1",
    "urdf",
    "g1_29dof_torsobase_popsicle.urdf",
)
ROOT_BODY = "pelvis"  # PHP's floating base (joint_pos[:, :7])
BASE_BODY = "torso_link"  # InstinctLab G1 floating base
REVERSED_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
# Bodies compared against InstinctLab forward kinematics in --validate.
VALIDATE_BODIES = [
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def pelvis_to_base(root_pos, root_quat_wxyz, joint_pos, joint_names, chain):
    """PHP pelvis pose -> InstinctLab torso_link pose, through FK of the torso-rooted URDF."""
    import pytorch_kinematics as pk

    order = [joint_names.index(n) for n in chain.get_joint_parameter_names()]
    frames = chain.forward_kinematics(
        torch.as_tensor(joint_pos[:, order], dtype=torch.float64), chain.get_frame_indices(ROOT_BODY, BASE_BODY)
    )
    base_in_root = frames[ROOT_BODY].inverse().compose(frames[BASE_BODY])
    root = pk.Transform3d(
        rot=torch.as_tensor(root_quat_wxyz, dtype=torch.float64),
        pos=torch.as_tensor(root_pos, dtype=torch.float64),
        dtype=torch.float64,
    )
    base = root.compose(base_in_root).get_matrix()
    return base[:, :3, 3].numpy(), pk.matrix_to_quaternion(base[:, :3, :3]).numpy()


def quat_wxyz_to_matrix(quat):
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def box_footprints_xy(boxes):
    """(num_boxes, 10) boxes -> (num_boxes * 4, 2) xy corners of their footprints."""
    if len(boxes) == 0:
        return np.zeros((0, 2))
    signs = np.array([[sx, sy, 0.0] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)])
    corners = signs[None] * boxes[:, None, 7:10]  # (B, 4, 3) in the box frame
    rot = quat_wxyz_to_matrix(boxes[:, 3:7])  # (B, 3, 3)
    corners_w = np.einsum("bij,bkj->bki", rot, corners) + boxes[:, None, 0:3]
    return corners_w[..., :2].reshape(-1, 2)


def build_terrain_mesh(boxes, size):
    """Flat ground plane of ``size`` at z = 0 (a two-triangle surface, so the border height reads 0) plus the boxes."""
    hx, hy = size[0] / 2, size[1] / 2
    ground = trimesh.Trimesh(
        vertices=np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )
    meshes = [ground]
    for box in boxes:
        transform = np.eye(4)
        transform[:3, :3] = quat_wxyz_to_matrix(box[3:7])
        transform[:3, 3] = box[0:3]
        meshes.append(trimesh.creation.box(extents=box[7:10], transform=transform))
    return trimesh.util.concatenate(meshes)


def convert_clip(motion_file, terrain_file, tgt_dir, size, margin, variant, chain):
    motion = np.load(motion_file)
    fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
    joint_names = [str(n) for n in motion["joint_names"]]
    joint_pos = motion["joint_pos"][:, 7:].copy()
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(f"{motion_file}: {joint_pos.shape[1]} joint columns but {len(joint_names)} joint names")
    # The torso-rooted URDF inverts the waist chain, so a waist angle has the opposite sign (as in GMR_to_instinct.py).
    for name in REVERSED_JOINTS:
        joint_pos[:, joint_names.index(name)] *= -1.0
    base_pos_w, base_quat_w = pelvis_to_base(
        motion["joint_pos"][:, 0:3], motion["joint_pos"][:, 3:7], joint_pos, joint_names, chain
    )

    terrain = np.load(terrain_file)
    if terrain.ndim != 3 or terrain.shape[2] != 10:
        raise ValueError(f"{terrain_file}: expected (num_boxes, num_variants, 10), got {terrain.shape}")
    if terrain.shape[0] and not 0 <= variant < terrain.shape[1]:
        raise ValueError(f"{terrain_file}: variant {variant} out of range ({terrain.shape[1]} variants)")
    boxes = terrain[:, variant].astype(np.float64) if terrain.shape[0] else np.zeros((0, 10))

    points_xy = np.concatenate([motion["body_pos_w"][..., :2].reshape(-1, 2), box_footprints_xy(boxes)])
    lower, upper = points_xy.min(0), points_xy.max(0)
    center = (lower + upper) / 2
    extent = upper - lower
    if np.any(extent + 2 * margin > np.asarray(size)):
        raise ValueError(
            f"{motion_file}: extent {extent.round(2).tolist()} m + 2 x {margin} m margin does not fit size {list(size)}"
        )
    shift = np.array([-center[0], -center[1], 0.0])
    base_pos_w = base_pos_w + shift
    boxes = boxes.copy()
    boxes[:, 0:3] += shift

    clip = os.path.basename(motion_file)[: -len("_motion.npz")]
    os.makedirs(tgt_dir, exist_ok=True)
    out_npz = os.path.join(tgt_dir, f"{clip}_retargeted.npz")
    np.savez(
        out_npz,
        framerate=np.asarray(fps),
        joint_names=np.asarray(joint_names),
        joint_pos=joint_pos.astype(np.float32),
        base_pos_w=base_pos_w.astype(np.float32),
        base_quat_w=base_quat_w.astype(np.float32),
    )
    build_terrain_mesh(boxes, size).export(os.path.join(tgt_dir, f"{clip}_terrain.stl"))
    return out_npz, {"frames": joint_pos.shape[0], "num_boxes": len(boxes), "extent": extent, "shift": shift}


def validate(motion_file, out_npz, chain, shift):
    """Max distance between PHP body positions and InstinctLab FK of the converted base pose + joints.

    A mismatch here means PHP's G1 model and the InstinctLab torso-rooted asset disagree (link offsets, joint
    zeros or base frame).
    """
    motion = np.load(motion_file)
    data = np.load(out_npz)
    body_names = [str(n) for n in motion["body_names"]]
    joint_names = [str(n) for n in data["joint_names"]]
    order = [joint_names.index(n) for n in chain.get_joint_parameter_names()]
    joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float64)[:, order]
    frames = chain.forward_kinematics(joint_pos, chain.get_frame_indices(*VALIDATE_BODIES))
    rot = torch.as_tensor(quat_wxyz_to_matrix(data["base_quat_w"].astype(np.float64)))
    base_pos = torch.as_tensor(data["base_pos_w"], dtype=torch.float64)
    errors = {}
    for name in VALIDATE_BODIES:
        pos_b = frames[name].get_matrix()[:, :3, 3]
        pos_w = (rot @ pos_b.unsqueeze(-1)).squeeze(-1) + base_pos
        ref = torch.as_tensor(motion["body_pos_w"][:, body_names.index(name)] + shift)
        errors[name] = float((pos_w - ref).norm(dim=-1).max())
    lowest_foot_z = float(
        np.minimum(
            motion["body_pos_w"][:, body_names.index("left_ankle_roll_link"), 2],
            motion["body_pos_w"][:, body_names.index("right_ankle_roll_link"), 2],
        ).min()
    )
    return errors, lowest_foot_z


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", nargs="+", required=True, help="Extracted PHP bundle directories")
    parser.add_argument("--tgt", required=True, help="Output dataset root")
    parser.add_argument("--size", nargs=2, type=float, required=True, help="Subterrain size (x, y) in metres")
    parser.add_argument("--margin", type=float, default=0.5, help="Minimum free ground around each clip, metres")
    parser.add_argument("--variant", type=int, default=0, help="Which of the randomized box variants to build")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Torso-rooted robot URDF for the base shift")
    parser.add_argument(
        "--validate-tol", type=float, default=0.02, help="Max FK body-position error (m) accepted by --validate"
    )
    parser.add_argument("--validate", action="store_true", help="Check the base frame against InstinctLab FK")
    args = parser.parse_args()

    import pytorch_kinematics as pk

    with open(args.urdf) as f:
        chain = pk.build_chain_from_urdf(f.read()).to(dtype=torch.float64)

    failures = []
    num_clips = 0
    for src in args.src:
        src = os.path.expanduser(src)
        bundle = os.path.basename(os.path.normpath(src))
        motion_files = sorted(glob.glob(os.path.join(src, "*_motion.npz")))
        if not motion_files:
            raise SystemExit(f"no *_motion.npz in {src}")
        for motion_file in motion_files:
            terrain_file = motion_file[: -len("_motion.npz")] + "_terrain.npy"
            clip = os.path.basename(motion_file)[: -len("_motion.npz")]
            out_npz, stats = convert_clip(
                motion_file,
                terrain_file,
                os.path.join(args.tgt, bundle, clip),
                args.size,
                args.margin,
                args.variant,
                chain,
            )
            num_clips += 1
            line = (
                f"{bundle}/{clip}: {stats['frames']} frames, {stats['num_boxes']} boxes,"
                f" extent {stats['extent'].round(1).tolist()} m"
            )
            if args.validate:
                errors, lowest_foot_z = validate(motion_file, out_npz, chain, stats["shift"])
                worst = max(errors.values())
                line += f", FK error max {worst * 100:.2f} cm, lowest ankle z {lowest_foot_z:+.3f} m"
                if worst > args.validate_tol:
                    failures.append((f"{bundle}/{clip}", errors))
            print(line)

    if failures:
        for name, errors in failures:
            print(f"FK mismatch {name}: {errors}", file=sys.stderr)
        raise SystemExit(f"validation failed for {len(failures)} clip(s)")

    subprocess.run(
        [sys.executable, os.path.join(_THIS_DIR, "motion_matched_metadata_generator.py"), "--path", args.tgt],
        check=True,
    )
    print(f"\nconverted {num_clips} clip(s) -> {args.tgt} (metadata.yaml written)")


if __name__ == "__main__":
    main()
