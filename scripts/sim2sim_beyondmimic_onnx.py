#!/usr/bin/env python3
"""
BeyondMimic G1 Sim2Sim using ONNX policy exported by scripts/instinct_rl/play.py --exportonnx.

Based on scripts/sim2sim_base.py control flow, but:
- ONNXRuntime inference (+ optional policy_normalizer.npz).
- Observation layout matches BeyondMimic policy observations (single-frame buffer).
- MuJoCo loading matches downstream ``sim2sim_downstream_mujoco_onnx.py`` style:
  URDF temp floating-base insertion → MuJoCo extension ``discardvisual=false`` + ``strippath=false`` (keep STL paths/visuals) →
  ``MjModel.from_xml_path`` → ``mj_saveLastXML``
  → MJCF injection (joint armature + ``position`` actuators) → minimal scene wrapper
  → final ``MjModel.from_xml_path`` with ``nu == NUM_ACTIONS``.
- PD targets via MuJoCo ``position`` actuators (``data.ctrl[:] = target_q``), not Python torque injection.
- Viewer defaults to hiding collision geom group 0 (floor uses group 2); use ``--show_collision_geoms`` to show primitives.
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import os
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    import mujoco_viewer
except ImportError:  # pragma: no cover - optional GUI dependency
    mujoco_viewer = None


# --- Repo paths ---
_SCRIPTS_DIR = Path(__file__).resolve().parent
_WBCHSI_ROOT = _SCRIPTS_DIR.parent
_PROJECT_ROOT = _WBCHSI_ROOT.parent
_DEFAULT_URDF = (
    _WBCHSI_ROOT
    / "source/instinctlab/instinctlab/assets/resources/unitree_g1/omniretarget_models/g1/g1_29dof_spherehand.urdf"
)
_DEFAULT_DATASET_YAML = _PROJECT_ROOT / "data/dataset_folder/beyondmimic.yaml"
_DEFAULT_LOG_ROOT = _WBCHSI_ROOT / "logs/instinct_rl/g1_beyondmimic"


# --- Isaac / BeyondMimic joint order (matches instinctlab/assets/unitree_g1.py header) ---
ISAAC_JOINT_ORDER: list[str] = [
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

NUM_ACTIONS = len(ISAAC_JOINT_ORDER)

# Popsicle URDF: MuJoCo URDF importer discards ``<visual>`` meshes unless compiler sets discardvisual=false.
_EXPECTED_POP_MESHES = 35
_EXPECTED_FLOATING_BASE_NQ = 36
_EXPECTED_FLOATING_BASE_NV = 35

# Popsicle default pose (G1_29DOF_OMNIRETARGET_CFG.init_state.joint_pos)
_POPSICLE_DEFAULT_RULES: tuple[tuple[str, float], ...] = (
    (".*_hip_pitch_joint", -0.312),
    (".*_knee_joint", 0.669),
    (".*_ankle_pitch_joint", -0.363),
    (".*_elbow_joint", 0.6),
    ("left_shoulder_roll_joint", 0.2),
    ("left_shoulder_pitch_joint", 0.2),
    ("right_shoulder_roll_joint", -0.2),
    ("right_shoulder_pitch_joint", 0.2),
)

# --- BeyondMimic natural actuator stiffness/damping (same numeric derivation as unitree_g1.py) ---
_NATURAL_FREQ = 10 * 2.0 * math.pi
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425

_STIFFNESS_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFFNESS_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFFNESS_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2

_DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMPING_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ

# Per-joint armature matching instinctlab ``beyondmimic_g1_29dof_actuators`` / downstream script.
_ARMATURE_PER_JOINT: dict[str, float] = {
    "left_hip_pitch_joint": _ARMATURE_7520_14,
    "right_hip_pitch_joint": _ARMATURE_7520_14,
    "left_hip_yaw_joint": _ARMATURE_7520_14,
    "right_hip_yaw_joint": _ARMATURE_7520_14,
    "left_hip_roll_joint": _ARMATURE_7520_22,
    "right_hip_roll_joint": _ARMATURE_7520_22,
    "left_knee_joint": _ARMATURE_7520_22,
    "right_knee_joint": _ARMATURE_7520_22,
    "left_ankle_pitch_joint": 2.0 * _ARMATURE_5020,
    "right_ankle_pitch_joint": 2.0 * _ARMATURE_5020,
    "left_ankle_roll_joint": 2.0 * _ARMATURE_5020,
    "right_ankle_roll_joint": 2.0 * _ARMATURE_5020,
    "waist_yaw_joint": _ARMATURE_7520_14,
    "waist_roll_joint": 2.0 * _ARMATURE_5020,
    "waist_pitch_joint": 2.0 * _ARMATURE_5020,
    "left_shoulder_pitch_joint": _ARMATURE_5020,
    "right_shoulder_pitch_joint": _ARMATURE_5020,
    "left_shoulder_roll_joint": _ARMATURE_5020,
    "right_shoulder_roll_joint": _ARMATURE_5020,
    "left_shoulder_yaw_joint": _ARMATURE_5020,
    "right_shoulder_yaw_joint": _ARMATURE_5020,
    "left_elbow_joint": _ARMATURE_5020,
    "right_elbow_joint": _ARMATURE_5020,
    "left_wrist_roll_joint": _ARMATURE_5020,
    "right_wrist_roll_joint": _ARMATURE_5020,
    "left_wrist_pitch_joint": _ARMATURE_4010,
    "right_wrist_pitch_joint": _ARMATURE_4010,
    "left_wrist_yaw_joint": _ARMATURE_4010,
    "right_wrist_yaw_joint": _ARMATURE_4010,
}


def _first_matching(patterns: Iterable[tuple[str, float]], name: str, default: float = 0.0) -> float:
    for pat, val in patterns:
        if fnmatch.fnmatch(name, pat):
            return float(val)
    return default


def default_joint_positions_lab() -> np.ndarray:
    return np.array([_first_matching(_POPSICLE_DEFAULT_RULES, jn, 0.0) for jn in ISAAC_JOINT_ORDER], dtype=np.float64)


def stiffness_for_joint(name: str) -> float:
    if name == "waist_yaw_joint" or fnmatch.fnmatch(name, "*_hip_yaw_joint") or fnmatch.fnmatch(name, "*_hip_pitch_joint"):
        return _STIFFNESS_7520_14
    if fnmatch.fnmatch(name, "*_hip_roll_joint") or fnmatch.fnmatch(name, "*_knee_joint"):
        return _STIFFNESS_7520_22
    if fnmatch.fnmatch(name, "*_ankle_*"):
        return 2.0 * _STIFFNESS_5020
    if name in ("waist_roll_joint", "waist_pitch_joint"):
        return 2.0 * _STIFFNESS_5020
    if fnmatch.fnmatch(name, "*_shoulder_*") or fnmatch.fnmatch(name, "*_elbow_joint"):
        return _STIFFNESS_5020
    if fnmatch.fnmatch(name, "*_wrist_roll_joint"):
        return _STIFFNESS_5020
    if fnmatch.fnmatch(name, "*_wrist_pitch_joint") or fnmatch.fnmatch(name, "*_wrist_yaw_joint"):
        return _STIFFNESS_4010
    raise KeyError(f"No stiffness rule for joint {name}")


def damping_for_joint(name: str) -> float:
    if name == "waist_yaw_joint" or fnmatch.fnmatch(name, "*_hip_yaw_joint") or fnmatch.fnmatch(name, "*_hip_pitch_joint"):
        return _DAMPING_7520_14
    if fnmatch.fnmatch(name, "*_hip_roll_joint") or fnmatch.fnmatch(name, "*_knee_joint"):
        return _DAMPING_7520_22
    if fnmatch.fnmatch(name, "*_ankle_*"):
        return 2.0 * _DAMPING_5020
    if name in ("waist_roll_joint", "waist_pitch_joint"):
        return 2.0 * _DAMPING_5020
    if fnmatch.fnmatch(name, "*_shoulder_*") or fnmatch.fnmatch(name, "*_elbow_joint"):
        return _DAMPING_5020
    if fnmatch.fnmatch(name, "*_wrist_roll_joint"):
        return _DAMPING_5020
    if fnmatch.fnmatch(name, "*_wrist_pitch_joint") or fnmatch.fnmatch(name, "*_wrist_yaw_joint"):
        return _DAMPING_4010
    raise KeyError(f"No damping rule for joint {name}")


def effort_limit_for_joint(name: str) -> float:
    if fnmatch.fnmatch(name, "*_hip_yaw_joint") or fnmatch.fnmatch(name, "*_hip_pitch_joint"):
        return 88.0
    if fnmatch.fnmatch(name, "*_hip_roll_joint"):
        return 139.0
    if fnmatch.fnmatch(name, "*_knee_joint"):
        return 139.0
    if fnmatch.fnmatch(name, "*_ankle_pitch_joint") or fnmatch.fnmatch(name, "*_ankle_roll_joint"):
        return 50.0
    if name == "waist_yaw_joint":
        return 88.0
    if name in ("waist_roll_joint", "waist_pitch_joint"):
        return 50.0
    if fnmatch.fnmatch(name, "*_shoulder_*") or fnmatch.fnmatch(name, "*_elbow_joint") or fnmatch.fnmatch(
        name, "*_wrist_roll_joint"
    ):
        return 25.0
    if fnmatch.fnmatch(name, "*_wrist_pitch_joint") or fnmatch.fnmatch(name, "*_wrist_yaw_joint"):
        return 5.0
    raise KeyError(f"No effort rule for joint {name}")


def action_scale_for_joint(name: str) -> float:
    k = stiffness_for_joint(name)
    e = effort_limit_for_joint(name)
    return 0.25 * e / k


def build_gain_vectors_mj(hinge_joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kp = np.zeros(NUM_ACTIONS, dtype=np.float64)
    kd = np.zeros(NUM_ACTIONS, dtype=np.float64)
    tau_lim = np.zeros(NUM_ACTIONS, dtype=np.float64)
    ascale = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for i, jn in enumerate(hinge_joint_names):
        kp[i] = stiffness_for_joint(jn)
        kd[i] = damping_for_joint(jn)
        tau_lim[i] = effort_limit_for_joint(jn)
        ascale[i] = action_scale_for_joint(jn)
    return kp, kd, tau_lim, ascale


# --- Quaternion utils (w, x, y, z) ---


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product q * p, both wxyz."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz)."""
    q = quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:4]


def subtract_frame_transforms(t01: np.ndarray, q01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same semantics as isaaclab subtract_frame_transforms when only frame 1 is given."""
    q10 = quat_conj(quat_normalize(q01))
    t12 = quat_apply(q10, -np.asarray(t01, dtype=np.float64))
    return t12, q10


def transform_points(points: np.ndarray, pos: np.ndarray | None, quat: np.ndarray | None) -> np.ndarray:
    """Matches isaaclab.utils.math.transform_points for batched (P,3)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = pts.copy()
    if quat is not None:
        q = quat_normalize(quat)
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy xyzw
        out = rot.apply(out)
    if pos is not None:
        out = out + np.asarray(pos, dtype=np.float64).reshape(1, 3)
    return out.reshape(-1)


def quat_to_tan_norm(q: np.ndarray) -> np.ndarray:
    """Copied from instinctlab.utils.math.quat_to_tan_norm (numpy)."""
    q = quat_normalize(q)
    tan = quat_apply(q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    norm = quat_apply(q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return np.concatenate([tan, norm], axis=-1)


def axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()


def quat_angular_velocity(q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity from q_prev to q_next over dt (world-ish frame; matches motion_reference usage)."""
    q_prev = quat_normalize(q_prev)
    q_next = quat_normalize(q_next)
    if float(np.dot(q_prev, q_next)) < 0.0:
        q_next = -q_next
    q_diff = quat_mul(q_next, quat_conj(q_prev))
    aa = axis_angle_from_quat(q_diff)
    return aa / dt


def estimate_velocity_np(x: np.ndarray, dt: float, mode: str = "frontbackward") -> np.ndarray:
    """Finite-difference velocity for shape (T, D)."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2
    if mode == "frontbackward":
        prev = np.roll(x, 1, axis=0)
        prev[0] = x[0]
        nxt = np.roll(x, -1, axis=0)
        nxt[-1] = x[-1]
        return (nxt - prev) / (2.0 * dt)
    raise ValueError(mode)


def estimate_angular_velocity_np(qs: np.ndarray, dt: float, mode: str = "frontbackward") -> np.ndarray:
    """Angular velocity series from quaternion series (T,4) wxyz."""
    qs = np.asarray(qs, dtype=np.float64)
    assert qs.ndim == 2 and qs.shape[-1] == 4
    if mode == "frontbackward":
        qp = np.roll(qs, 1, axis=0)
        qp[0] = qs[0]
        qn = np.roll(qs, -1, axis=0)
        qn[-1] = qs[-1]
        out = np.zeros((qs.shape[0], 3), dtype=np.float64)
        for t in range(qs.shape[0]):
            out[t] = quat_angular_velocity(qp[t], qn[t], 2.0 * dt)
        return out
    raise ValueError(mode)


class RetargetedNpzMotion:
    """Reads BeyondMimic-style retargeted .npz (see AmassMotion._read_retargetted_motion_file)."""

    def __init__(self, filepath: str):
        data = np.load(filepath, mmap_mode="r", allow_pickle=True)
        framerate = float(np.asarray(data["framerate"]).reshape(()))
        dt = 1.0 / framerate

        jraw = data["joint_names"]
        joint_names = jraw.tolist() if isinstance(jraw, np.ndarray) else list(jraw)
        def _decode_jname(jn):
            if isinstance(jn, bytes):
                return jn.decode("utf-8")
            if isinstance(jn, np.bytes_):
                return bytes(jn).decode("utf-8")
            return str(jn)

        joint_names = [_decode_jname(jn) for jn in joint_names]
        joint_pos_full = np.asarray(data["joint_pos"], dtype=np.float64)
        root_trans = np.asarray(data["base_pos_w"], dtype=np.float64)
        root_quat = np.asarray(data["base_quat_w"], dtype=np.float64)

        joint_ids = [joint_names.index(jn) for jn in ISAAC_JOINT_ORDER]
        joint_pos = joint_pos_full[:, joint_ids]

        joint_vel = estimate_velocity_np(joint_pos, dt, "frontbackward")
        base_lin_vel_w = estimate_velocity_np(root_trans, dt, "frontbackward")
        base_ang_vel_w = estimate_angular_velocity_np(root_quat, dt, "frontbackward")

        self.dt = dt
        self.framerate = framerate
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.base_pos_w = root_trans
        self.base_quat_w = np.apply_along_axis(quat_normalize, 1, root_quat)
        self.base_lin_vel_w = base_lin_vel_w
        self.base_ang_vel_w = base_ang_vel_w
        self.time_step_total = joint_pos.shape[0]

    def get_duration(self) -> float:
        return self.dt * max(self.time_step_total - 1, 0)


def resolve_motion_path(args: argparse.Namespace) -> Path:
    if args.motion_file:
        return Path(args.motion_file).expanduser().resolve()
    ypath = Path(args.dataset_yaml).expanduser().resolve()
    if not ypath.is_file():
        raise FileNotFoundError(f"beyondmimic dataset yaml not found: {ypath}")
    with open(ypath, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rel = cfg["selected_files"][0]
    folder = ypath.parent
    return (folder / rel).resolve()


def resolve_onnx_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.load_model:
        onnx_p = Path(args.load_model).expanduser().resolve()
        norm_p = Path(args.policy_normalizer).expanduser().resolve() if args.policy_normalizer else None
        if norm_p is None:
            cand = onnx_p.parent / "policy_normalizer.npz"
            norm_p = cand if cand.is_file() else None
        return onnx_p, norm_p

    if not args.load_run:
        raise ValueError("Provide --load_model or --load_run")
    exp_dir = Path(args.log_root).expanduser().resolve() / args.load_run
    exported = exp_dir / "exported"
    onnx_p = exported / "actor.onnx"
    norm_p = exported / "policy_normalizer.npz"
    norm_p = norm_p if norm_p.is_file() else None
    if args.policy_normalizer:
        norm_p = Path(args.policy_normalizer).expanduser().resolve()
    return onnx_p, norm_p


def load_normalizer(path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    if path is None or not path.is_file():
        return None, None, 1e-2
    z = np.load(path, allow_pickle=True)
    mean = np.asarray(z["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(z["std"], dtype=np.float64).reshape(-1)
    eps = float(np.asarray(z["eps"]).reshape(()))
    return mean, std, eps


def _make_absolute_meshdir(root: ET.Element, xml_path: str) -> None:
    """Resolve relative compiler meshdir against ``xml_path`` so temp MJCF loads meshes."""
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        if meshdir and not os.path.isabs(meshdir):
            compiler.set(
                "meshdir",
                os.path.join(os.path.dirname(os.path.abspath(xml_path)), meshdir),
            )


def _inject_joint_armature(root: ET.Element) -> int:
    """Set joint armature from BeyondMimic actuator definitions."""
    n_set = 0
    for joint in root.iter("joint"):
        jname = joint.get("name")
        if not jname:
            continue
        arm = _ARMATURE_PER_JOINT.get(jname)
        if arm is None:
            continue
        joint.set("armature", f"{arm:.10f}")
        n_set += 1
    return n_set


def _inject_position_actuators(root: ET.Element, body_joint_names: list[str]) -> None:
    """Replace actuator section with MuJoCo ``position`` PD actuators (downstream style)."""
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator_el = ET.SubElement(root, "actuator")
    for jname in body_joint_names:
        kp = float(stiffness_for_joint(jname))
        kd = float(damping_for_joint(jname))
        tau_limit = float(effort_limit_for_joint(jname))
        ET.SubElement(
            actuator_el,
            "position",
            {
                "name": jname,
                "joint": jname,
                "kp": f"{kp:.6f}",
                "kv": f"{kd:.6f}",
                "ctrlrange": "-100 100",
                "forcerange": f"{-tau_limit:.6f} {tau_limit:.6f}",
                "forcelimited": "true",
            },
        )


def _materialize_popsicle_urdf_as_mjcf(urdf_path: str) -> tuple[ET.ElementTree, ET.Element, str, list[str]]:
    """Floating ``world``→``torso_link``, URDF import → MJCF + armature + position actuators.

    Returns (tree, root, robot_xml_path, paths_to_delete_excluding_robot_xml).
    Caller deletes robot_xml_path after loading the scene wrapper.
    """
    src = os.path.abspath(urdf_path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"URDF robot model not found: {src}")

    with open(src, encoding="utf-8") as f:
        urdf_text = f.read()

    urdf_text = _ensure_mujoco_keep_visuals_in_urdf(urdf_text)

    if "floating_base_joint" not in urdf_text:
        m_robot = re.search(r"<robot\s[^>]*>", urdf_text)
        if not m_robot:
            raise RuntimeError("Could not find <robot ...> tag in URDF.")
        floating_insert = """
<link name="world"></link>
<joint name="floating_base_joint" type="floating">
  <parent link="world"/>
  <child link="torso_link"/>
</joint>
"""
        insert_at = m_robot.end()
        urdf_text = urdf_text[:insert_at] + floating_insert + urdf_text[insert_at:]

    meshes_dir = os.path.normpath(os.path.join(os.path.dirname(src), "..", "meshes"))
    urdf_text = urdf_text.replace('filename="../meshes/', f'filename="{meshes_dir}/')

    tmp_urdf = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w", encoding="utf-8")
    tmp_urdf.write(urdf_text)
    tmp_urdf.close()

    model_preview = mujoco.MjModel.from_xml_path(tmp_urdf.name)

    robot_xml_fd = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    robot_xml_path = robot_xml_fd.name
    robot_xml_fd.close()

    mujoco.mj_saveLastXML(robot_xml_path, model_preview)

    tree = ET.parse(robot_xml_path)
    root = tree.getroot()
    _make_absolute_meshdir(root, robot_xml_path)

    body_joint_names: list[str] = []
    for joint_id in range(1, model_preview.njnt):
        jname = mujoco.mj_id2name(model_preview, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not jname:
            raise RuntimeError(f"Converted MuJoCo model has unnamed joint id {joint_id}.")
        body_joint_names.append(str(jname))

    if len(body_joint_names) != NUM_ACTIONS:
        warnings.warn(
            f"Converted MJCF has {len(body_joint_names)} body joints, expected {NUM_ACTIONS}.",
            stacklevel=2,
        )

    _inject_position_actuators(root, body_joint_names)

    n_arm = _inject_joint_armature(root)
    if n_arm < NUM_ACTIONS:
        warnings.warn(
            f"Injected armature on only {n_arm}/{NUM_ACTIONS} joints — dynamics may differ from IsaacLab.",
            stacklevel=2,
        )

    tree.write(robot_xml_path, encoding="unicode")

    return tree, root, robot_xml_path, [tmp_urdf.name]


def _ensure_mujoco_keep_visuals_in_urdf(urdf_text: str) -> str:
    """Insert MuJoCo URDF extension so STL visuals are kept (default importer drops them).

    ``strippath="false"`` keeps absolute mesh filenames from being stripped to basenames
    (otherwise URDF loaded from ``/tmp`` fails with e.g. ``left_hip_roll_link.STL``).
    """
    if 'discardvisual="false"' in urdf_text and 'strippath="false"' in urdf_text:
        return urdf_text
    if 'discardvisual="false"' in urdf_text and 'strippath="false"' not in urdf_text:
        patched, n = re.subn(
            r'(<compiler\b[^>]*\bdiscardvisual\s*=\s*"false")(\s*/>)',
            r'\1 strippath="false"\2',
            urdf_text,
            count=1,
        )
        if n:
            return patched
    m_robot = re.search(r"<robot\s[^>]*>", urdf_text)
    if not m_robot:
        raise RuntimeError("Could not find <robot ...> tag in URDF.")
    mujoco_insert = """
<mujoco>
  <compiler discardvisual="false" strippath="false"/>
</mujoco>
"""
    insert_at = m_robot.end()
    return urdf_text[:insert_at] + mujoco_insert + urdf_text[insert_at:]


def _count_mesh_geoms(model: mujoco.MjModel) -> int:
    n = 0
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            n += 1
    return n


def build_model_like_downstream(urdf_path: str) -> mujoco.MjModel:
    """Minimal scene + downstream-style URDF → MJCF materialization."""
    enforce_pop_visual_meshes = "popsicle" in os.path.basename(urdf_path).lower()
    _, _, robot_xml_path, extra_cleanup = _materialize_popsicle_urdf_as_mjcf(urdf_path)
    robot_xml_abs = os.path.abspath(robot_xml_path)

    scene_xml = f"""\
<mujoco model="sim2sim_beyondmimic">
  <include file="{robot_xml_abs}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="50 50 0.05" type="plane" material="groundplane" group="2"/>
  </worldbody>
</mujoco>"""

    tmp_scene = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", encoding="utf-8")
    tmp_scene.write(scene_xml)
    tmp_scene.close()

    model = mujoco.MjModel.from_xml_path(tmp_scene.name)

    for p in [robot_xml_path, tmp_scene.name, *extra_cleanup]:
        try:
            os.unlink(p)
        except OSError:
            pass

    mesh_geoms = _count_mesh_geoms(model)
    print(
        f"[INFO] MuJoCo model loaded (downstream-style): nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"nmesh={model.nmesh}, mesh_geoms={mesh_geoms}"
    )
    if enforce_pop_visual_meshes:
        if model.nmesh != _EXPECTED_POP_MESHES or mesh_geoms != _EXPECTED_POP_MESHES:
            raise RuntimeError(
                f"Expected {_EXPECTED_POP_MESHES} visual STL meshes (nmesh and mesh geoms), "
                f"got nmesh={model.nmesh}, mesh_geoms={mesh_geoms}. "
                "Check URDF <mujoco><compiler discardvisual=\"false\"/></mujoco> and mesh paths."
            )
        if model.nq != _EXPECTED_FLOATING_BASE_NQ or model.nv != _EXPECTED_FLOATING_BASE_NV:
            raise RuntimeError(
                "Floating-base degree counts regressed: "
                f"expected nq={_EXPECTED_FLOATING_BASE_NQ}, nv={_EXPECTED_FLOATING_BASE_NV}, "
                f"got nq={model.nq}, nv={model.nv}."
            )
    elif model.nmesh == 0 and mesh_geoms == 0:
        warnings.warn(
            "MuJoCo loaded zero meshes/visual mesh geoms — URDF visuals were likely discarded. "
            "For popsicle STL visuals use a URDF filename containing 'popsicle' or add "
            '<mujoco><compiler discardvisual="false"/></mujoco> to the URDF.',
            stacklevel=2,
        )
    if model.nu != NUM_ACTIONS:
        warnings.warn(
            f"Expected nu={NUM_ACTIONS} position actuators, got nu={model.nu}. Check actuator injection.",
            stacklevel=2,
        )
    return model


def hinge_joint_metadata(model: mujoco.MjModel) -> tuple[list[int], list[str]]:
    hinge_ids: list[int] = []
    hinge_names: list[str] = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        hinge_ids.append(j)
        hinge_names.append(name)
    if len(hinge_names) != NUM_ACTIONS:
        raise RuntimeError(f"Expected {NUM_ACTIONS} hinge joints, got {len(hinge_names)}: {hinge_names}")
    missing = [jn for jn in ISAAC_JOINT_ORDER if jn not in hinge_names]
    if missing:
        raise RuntimeError(f"Hinge joints missing expected BeyondMimic names: {missing}")
    return hinge_ids, hinge_names


def lab_vec_to_mj_order(vec_lab: np.ndarray, mj_joint_names: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(ISAAC_JOINT_ORDER)}
    out = np.zeros_like(vec_lab)
    for mj_i, name in enumerate(mj_joint_names):
        out[mj_i] = vec_lab[idx[name]]
    return out


def mj_vec_to_lab_order(vec_mj: np.ndarray, mj_joint_names: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(mj_joint_names)}
    return np.array([vec_mj[idx[jn]] for jn in ISAAC_JOINT_ORDER], dtype=np.float64)


def build_policy_obs(
    motion: RetargetedNpzMotion,
    frame_idx: int,
    robot_root_pos_w: np.ndarray,
    robot_root_quat_w: np.ndarray,
    v_body: np.ndarray,
    omega_body: np.ndarray,
    q_lab: np.ndarray,
    dq_lab: np.ndarray,
    default_lab: np.ndarray,
    last_action_lab: np.ndarray,
) -> np.ndarray:
    """Flattened policy observation (BeyondMimicObservationsCfg.PolicyObsCfg term order)."""
    jp_m = motion.joint_pos[frame_idx]
    jv_m = motion.joint_vel[frame_idx]
    bp_m = motion.base_pos_w[frame_idx]
    bq_m = motion.base_quat_w[frame_idx]

    joint_pos_ref = jp_m - default_lab
    joint_vel_ref = jv_m - 0.0

    anchor_pos, anchor_quat = subtract_frame_transforms(robot_root_pos_w, robot_root_quat_w)
    position_ref = transform_points(bp_m, anchor_pos, anchor_quat)

    quat_rel = quat_mul(anchor_quat, bq_m)
    rotation_ref = quat_to_tan_norm(quat_rel)

    # NOTE: base_lin_vel / base_ang_vel in BeyondMimic are robot proprioception (root frame), not motion reference.
    obs_terms = [
        joint_pos_ref.astype(np.float32),
        joint_vel_ref.astype(np.float32),
        position_ref.astype(np.float32),
        rotation_ref.astype(np.float32),
        v_body.astype(np.float32),
        omega_body.astype(np.float32),
        (q_lab - default_lab).astype(np.float32),
        dq_lab.astype(np.float32),
        last_action_lab.astype(np.float32),
    ]
    return np.concatenate(obs_terms, axis=0)


def apply_normalizer(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None, eps: float) -> np.ndarray:
    if mean is None:
        return x
    return (x - mean) / (std + eps)


def write_dummy_onnx(path: Path, obs_dim: int = 160, act_dim: int = NUM_ACTIONS) -> None:
    import onnx
    from onnx import TensorProto, helper
    from onnx.numpy_helper import from_array

    w = np.zeros((obs_dim, act_dim), dtype=np.float32)
    b = np.zeros((act_dim,), dtype=np.float32)
    nodes = [
        helper.make_node("MatMul", ["input", "W"], ["gemm_out"]),
        helper.make_node("Add", ["gemm_out", "B"], ["output"]),
    ]
    W = from_array(w, name="W")
    B = from_array(b, name="B")
    graph = helper.make_graph(
        nodes,
        "dummy_actor",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, obs_dim])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, act_dim])],
        initializer=[W, B],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def write_dummy_motion_npz(path: Path, frames: int = 50, fps: float = 50.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    T = frames
    joint_names = np.array(ISAAC_JOINT_ORDER, dtype=object)
    joint_pos = np.zeros((T, NUM_ACTIONS), dtype=np.float64)
    base_pos_w = np.zeros((T, 3), dtype=np.float64)
    base_pos_w[:, 2] = 0.82
    base_quat_w = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (T, 1))
    np.savez(
        path,
        framerate=np.float64(fps),
        joint_names=joint_names,
        joint_pos=joint_pos,
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
    )


def run_mujoco(args: argparse.Namespace) -> None:
    if args.dummy_assets:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "beyondmimic_sim2sim_dummy"
        tmp.mkdir(parents=True, exist_ok=True)
        onnx_path = tmp / "actor.onnx"
        norm_path = None
        write_dummy_onnx(onnx_path)
        motion_path = tmp / "motion.npz"
        write_dummy_motion_npz(motion_path)
        print(f"[dummy_assets] wrote {onnx_path} and {motion_path}")
    else:
        onnx_path, norm_path = resolve_onnx_paths(args)
        motion_path = resolve_motion_path(args)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    mean_npz, std_npz, eps_npz = load_normalizer(norm_path)

    motion = RetargetedNpzMotion(str(motion_path))
    print(f"Using motion file: {motion_path}")
    print(f"Motion duration: {motion.get_duration():.2f}s, frames={motion.time_step_total}, fps={motion.framerate:.2f}")

    model = build_model_like_downstream(str(Path(args.mujoco_urdf).expanduser().resolve()))
    model.opt.timestep = float(args.dt)

    hinge_ids, hinge_names = hinge_joint_metadata(model)
    _, _, _, action_scale_mj = build_gain_vectors_mj(hinge_names)

    mj_body_name = args.root_body
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_body_name)
    if bid < 0:
        raise RuntimeError(f"Body {mj_body_name!r} not found in compiled model.")

    data = mujoco.MjData(model)
    default_lab = default_joint_positions_lab()

    # Initial floating-base pose (torso root)
    data.qpos[0:3] = [0.0, 0.0, 0.82]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    q_defaults_mj = lab_vec_to_mj_order(default_lab, hinge_names)
    data.qpos[7 : 7 + NUM_ACTIONS] = q_defaults_mj

    mujoco.mj_forward(model, data)
    data.ctrl[:] = q_defaults_mj

    viewer = None
    if not args.no_viewer:
        if mujoco_viewer is None:
            raise RuntimeError("mujoco_viewer is not installed; use --no_viewer for headless.")
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = 5.0
        # Default: hide collision primitives (group 0); show URDF visuals (group 1) and scene floor (group 2).
        if not args.show_collision_geoms:
            vopt = getattr(viewer, "vopt", None)
            if vopt is not None:
                vopt.geomgroup[0] = 0
                vopt.geomgroup[1] = 1
                vopt.geomgroup[2] = 1

    sess_opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
    in_name = session.get_inputs()[0].name
    in_shape = session.get_inputs()[0].shape
    obs_dim_onnx = int(in_shape[-1]) if isinstance(in_shape[-1], int) else None

    last_action_lab = np.zeros(NUM_ACTIONS, dtype=np.float32)
    action_mj = np.zeros(NUM_ACTIONS, dtype=np.float64)
    target_q_mj = q_defaults_mj.copy()

    control_dt = float(args.dt) * int(args.decimation)
    motion_time = 0.0

    n_steps = int(float(args.sim_duration) / float(args.dt))

    for step_i in tqdm(range(n_steps), desc="Simulating..."):
        mujoco.mj_forward(model, data)

        robot_root_pos = data.xpos[bid].astype(np.float64).copy()
        robot_root_quat = data.xquat[bid].astype(np.float64).copy()

        res_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, res_vel, 1)
        omega_body = res_vel[0:3].copy()
        v_body = res_vel[3:6].copy()

        q_mj = np.array([data.qpos[model.jnt_qposadr[jid]] for jid in hinge_ids], dtype=np.float64)
        dq_mj = np.array([data.qvel[model.jnt_dofadr[jid]] for jid in hinge_ids], dtype=np.float64)
        q_lab = mj_vec_to_lab_order(q_mj, hinge_names)
        dq_lab = mj_vec_to_lab_order(dq_mj, hinge_names)

        if step_i % int(args.decimation) == 0:
            motion_time += control_dt
            m_step = min(int(motion_time / motion.dt), motion.time_step_total - 1)

            obs = build_policy_obs(
                motion,
                m_step,
                robot_root_pos,
                robot_root_quat,
                v_body,
                omega_body,
                q_lab,
                dq_lab,
                default_lab,
                last_action_lab,
            )
            if obs_dim_onnx is not None and obs.shape[-1] != obs_dim_onnx:
                raise RuntimeError(f"Observation dim {obs.shape[-1]} != ONNX expected {obs_dim_onnx}")

            obs_f = obs.astype(np.float32).reshape(1, -1)
            obs_n = apply_normalizer(obs_f.reshape(-1), mean_npz, std_npz, eps_npz).astype(np.float32).reshape(1, -1)

            action_out = session.run(None, {in_name: obs_n})[0]
            action_lab = np.asarray(action_out, dtype=np.float32).reshape(-1)
            if action_lab.shape[0] != NUM_ACTIONS:
                raise RuntimeError(f"Policy output dim {action_lab.shape[0]} != {NUM_ACTIONS}")

            last_action_lab = action_lab.copy()
            action_lab64 = action_lab.astype(np.float64)
            action_mj = lab_vec_to_mj_order(action_lab64, hinge_names)
            target_q_mj = action_mj * action_scale_mj + q_defaults_mj

        data.ctrl[:] = target_q_mj

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.render()
            viewer.cam.lookat[:2] = data.qpos[:2]

    if viewer is not None:
        viewer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BeyondMimic ONNX Sim2Sim (downstream-style MJCF + position actuators).")
    p.add_argument("--task", type=str, default="Instinct-BeyondMimic-Plane-G1-Play-v0", help="Task id (info only).")
    p.add_argument("--load_run", type=str, default=None, help="Experiment folder name under logs/.../g1_beyondmimic/")
    p.add_argument("--log_root", type=str, default=str(_DEFAULT_LOG_ROOT), help="Root log dir for g1_beyondmimic.")
    p.add_argument("--load_model", type=str, default=None, help="Path to actor.onnx (overrides --load_run).")
    p.add_argument("--policy_normalizer", type=str, default=None, help="Path to policy_normalizer.npz (optional).")
    p.add_argument("--motion_file", type=str, default=None, help="Explicit motion .npz (retargeted AMASS format).")
    p.add_argument("--dataset_yaml", type=str, default=str(_DEFAULT_DATASET_YAML), help="beyondmimic.yaml path.")
    p.add_argument("--mujoco_urdf", type=str, default=str(_DEFAULT_URDF), help="Robot URDF for MuJoCo compilation.")
    p.add_argument("--root_body", type=str, default="torso_link", help="Body name used as Isaac robot root frame.")
    p.add_argument("--sim_duration", type=float, default=1200.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument("--no_viewer", action="store_true", help="Do not open mujoco_viewer window.")
    p.add_argument(
        "--show_collision_geoms",
        action="store_true",
        help="In viewer, show collision primitive group (0); default hides them so STL visuals are unobstructed.",
    )
    p.add_argument(
        "--dummy_assets",
        action="store_true",
        help="Write dummy ONNX+motion under $TMPDIR and run a short smoke test without logs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dummy_assets:
        args.sim_duration = min(args.sim_duration, 0.25)
        args.no_viewer = True
    run_mujoco(args)


if __name__ == "__main__":
    main()
