"""Annotate terrain-motion clips with per-frame body contacts (flags and world positions) from geometry.

For every motion in a terrain-motion dataset (``metadata.yaml`` as read by ``TerrainMotionCfg``), the collision
geometry of each contact body is posed by forward kinematics of the torso-rooted G1 URDF, and a body is in contact
on a frame when

- the signed distance of its closest collision point to the matched terrain mesh is below ``--distance``
  (negative = penetrating, down to ``--max-penetration``), and
- that point moves slower than ``--speed``.

Contact segments shorter than ``--min-contact`` seconds are dropped after gaps shorter than ``--min-gap`` seconds are
filled. The contact position is the closest terrain point of that body point, in the clip (mesh) frame.

The originals are not modified. Each motion is written to ``<tgt>/<same relative path>`` with the added keys

- ``contact_body_names`` (B,): body names, in the simulated (fixed-joint-merged) asset
- ``contact_flag`` (T, B) bool
- ``contact_pos_w`` (T, B, 3) float32, metres in the clip frame; 0 where ``contact_flag`` is False

Terrain files are linked (relative symlinks) and a ``metadata.yaml`` with the same entries is written to ``<tgt>``.

Usage::

    python scripts/annotate_motion_contacts.py --path <dataset root> --tgt <output root> [--metadata <yaml>]
"""

import argparse
import json
import multiprocessing
import numpy as np
import os
import torch
import trimesh
import xml.etree.ElementTree as ET
import yaml

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
    "g1_29dof_torsobase_simplified.urdf",
)
# ACP's 9 contact bodies (ankles, knees, wrists, elbows, pelvis), named as in the simulated asset.
DEFAULT_CONTACT_BODIES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "pelvis",
]
# The torso-rooted URDF inverts the waist chain; motion files store the angles for that URDF already
# (see GMR_to_instinct.py), so no sign change is applied here.


def rpy_to_matrix(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def geometry_points(geometry):
    """Sample points on the surface of one URDF collision geometry, in the geometry frame."""
    if geometry.tag == "box":
        half = np.array([float(v) for v in geometry.get("size").split()]) / 2
        grid = np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
        )
        return grid * half
    if geometry.tag == "cylinder":
        radius, length = float(geometry.get("radius")), float(geometry.get("length"))
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
        return np.concatenate(
            [np.concatenate([ring, np.full((8, 1), z)], axis=-1) for z in (-length / 2, 0, length / 2)]
        )
    if geometry.tag == "sphere":
        radius = float(geometry.get("radius"))
        return np.concatenate([np.eye(3), -np.eye(3)]) * radius
    raise ValueError(f"unsupported collision geometry <{geometry.tag}>")


def collision_points_per_body(urdf_path, body_names):
    """{body: [(urdf link, (P, 3) points in that link frame)]}, merging fixed-joint children into their parent body."""
    root = ET.parse(urdf_path).getroot()
    fixed_children = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            fixed_children.setdefault(joint.find("parent").get("link"), []).append(joint.find("child").get("link"))
    links = {link.get("name"): link for link in root.findall("link")}

    result = {}
    for body in body_names:
        if body not in links:
            raise ValueError(f"contact body {body} is not a link of {urdf_path}")
        merged, stack = [], [body]
        while stack:
            name = stack.pop()
            stack.extend(fixed_children.get(name, []))
            points = []
            for collision in links[name].findall("collision"):
                origin = collision.find("origin")
                xyz = (
                    np.zeros(3) if origin is None else np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                )
                rpy = (
                    np.zeros(3) if origin is None else np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
                )
                points.append(geometry_points(collision.find("geometry")[0]) @ rpy_to_matrix(rpy).T + xyz)
            if points:
                merged.append((name, np.concatenate(points)))
        if not merged:
            raise ValueError(f"contact body {body} has no collision geometry in {urdf_path}")
        result[body] = merged
    return result


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


def body_points_world(motion, chain, body_points):
    """{body: (T, P, 3)} world positions of the collision points of each body."""
    joint_names = [str(n) for n in motion["joint_names"]]
    order = [joint_names.index(n) for n in chain.get_joint_parameter_names()]
    links = sorted({link for parts in body_points.values() for link, _ in parts})
    frames = chain.forward_kinematics(
        torch.as_tensor(np.asarray(motion["joint_pos"])[:, order], dtype=torch.float64), chain.get_frame_indices(*links)
    )
    base_rot = quat_wxyz_to_matrix(np.asarray(motion["base_quat_w"], dtype=np.float64))
    base_pos = np.asarray(motion["base_pos_w"], dtype=np.float64)
    result = {}
    for body, parts in body_points.items():
        posed = []
        for link, points in parts:
            link_tf = frames[link].get_matrix().numpy()  # (T, 4, 4) in the base frame
            in_base = np.einsum("tij,pj->tpi", link_tf[:, :3, :3], points) + link_tf[:, None, :3, 3]
            posed.append(np.einsum("tij,tpj->tpi", base_rot, in_base) + base_pos[:, None])
        result[body] = np.concatenate(posed, axis=1)
    return result


def debounce(flag, min_gap, min_contact):
    """Fill False runs shorter than ``min_gap`` frames, then drop True runs shorter than ``min_contact`` frames."""

    def runs(values):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], values.astype(np.int8), [0]])))
        return edges[0::2], edges[1::2]

    flag = flag.copy()
    starts, ends = runs(~flag)
    for s, e in zip(starts, ends):
        if 0 < s and e < len(flag) and e - s < min_gap:
            flag[s:e] = True
    starts, ends = runs(flag)
    for s, e in zip(starts, ends):
        if e - s < min_contact:
            flag[s:e] = False
    return flag


def annotate(motion, terrain, chain, body_points, params):
    fps = float(np.asarray(motion["framerate"]).item())
    points = body_points_world(motion, chain, body_points)
    num_frames = len(motion["joint_pos"])
    contact_flag = np.zeros((num_frames, len(points)), dtype=bool)
    contact_pos = np.zeros((num_frames, len(points), 3), dtype=np.float32)
    query = trimesh.proximity.ProximityQuery(terrain)
    for b, (body, p) in enumerate(points.items()):
        num_points = p.shape[1]
        closest, _, triangle = query.on_surface(p.reshape(-1, 3))
        signed = np.einsum("ni,ni->n", p.reshape(-1, 3) - closest, terrain.face_normals[triangle])
        signed = signed.reshape(num_frames, num_points)
        closest = closest.reshape(num_frames, num_points, 3)
        best = np.argmin(signed, axis=1)
        rows = np.arange(num_frames)
        best_signed = signed[rows, best]
        velocity = np.gradient(p, 1.0 / fps, axis=0)[rows, best]
        flag = (
            (best_signed < params["distance"])
            & (best_signed > -params["max_penetration"])
            & (np.linalg.norm(velocity, axis=-1) < params["speed"])
        )
        flag = debounce(flag, round(params["min_gap"] * fps), round(params["min_contact"] * fps))
        contact_flag[:, b] = flag
        contact_pos[flag, b] = closest[rows, best][flag]
    return contact_flag, contact_pos


_WORKER = {}


def _init_worker(urdf, bodies):
    import pytorch_kinematics as pk

    torch.set_num_threads(1)
    with open(urdf) as f:
        _WORKER["chain"] = pk.build_chain_from_urdf(f.read()).to(dtype=torch.float64)
    _WORKER["body_points"] = collision_points_per_body(urdf, bodies)


def _process(job):
    src_npz, terrain_file, tgt_npz, params = job
    motion = dict(np.load(src_npz, allow_pickle=True))
    terrain = trimesh.load(terrain_file, force="mesh")
    flag, pos = annotate(motion, terrain, _WORKER["chain"], _WORKER["body_points"], params)
    os.makedirs(os.path.dirname(tgt_npz), exist_ok=True)
    np.savez(
        tgt_npz,
        **motion,
        contact_body_names=np.asarray(list(_WORKER["body_points"])),
        contact_flag=flag,
        contact_pos_w=pos,
    )
    return tgt_npz, flag.mean(axis=0)


def relative_symlink(target, link):
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.relpath(target, os.path.dirname(link)), link)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, help="Dataset root (motion_file / terrain_file are relative to it)")
    parser.add_argument("--metadata", default=None, help="metadata.yaml, default <path>/metadata.yaml")
    parser.add_argument("--tgt", required=True, help="Output root, keeps the relative layout")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--bodies", nargs="+", default=DEFAULT_CONTACT_BODIES)
    parser.add_argument("--distance", type=float, default=0.04, help="Contact if signed distance below this (m)")
    parser.add_argument("--max-penetration", type=float, default=0.3, help="Ignore points deeper than this (m)")
    parser.add_argument("--speed", type=float, default=1.0, help="Contact point speed limit (m/s)")
    parser.add_argument("--min-gap", type=float, default=0.06, help="Fill contact gaps shorter than this (s)")
    parser.add_argument("--min-contact", type=float, default=0.06, help="Drop contacts shorter than this (s)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    path = os.path.expanduser(args.path)
    tgt = os.path.expanduser(args.tgt)
    with open(os.path.expanduser(args.metadata or os.path.join(path, "metadata.yaml"))) as f:
        metadata = yaml.safe_load(f)
    terrain_files = {t["terrain_id"]: t["terrain_file"] for t in metadata["terrains"]}
    params = {
        "distance": args.distance,
        "max_penetration": args.max_penetration,
        "speed": args.speed,
        "min_gap": args.min_gap,
        "min_contact": args.min_contact,
    }

    for terrain_file in terrain_files.values():
        relative_symlink(os.path.join(path, terrain_file), os.path.join(tgt, terrain_file))
    jobs = [
        (
            os.path.join(path, m["motion_file"]),
            os.path.join(path, terrain_files[m["terrain_id"]]),
            os.path.join(tgt, m["motion_file"]),
            params,
        )
        for m in metadata["motion_files"]
    ]
    with multiprocessing.get_context("spawn").Pool(args.workers, _init_worker, (args.urdf, args.bodies)) as pool:
        for tgt_npz, fraction in pool.imap(_process, jobs):
            summary = " ".join(f"{name}={v:.0%}" for name, v in zip(args.bodies, fraction))
            print(f"{os.path.relpath(tgt_npz, tgt)}: {summary}")

    with open(os.path.join(tgt, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)
    with open(os.path.join(tgt, "contact_annotation.json"), "w") as f:
        json.dump({"source": path, "urdf": os.path.realpath(args.urdf), "bodies": args.bodies, **params}, f, indent=2)
    print(f"\nannotated {len(jobs)} motion(s) -> {tgt}")


if __name__ == "__main__":
    main()
