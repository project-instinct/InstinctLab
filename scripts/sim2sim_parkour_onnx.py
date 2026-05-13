#!/usr/bin/env python3
"""
Parkour G1 Sim2Sim using ONNX exported by scripts/instinct_rl/play.py --exportonnx.

Matches ``Instinct-Parkour-Target-Amp-G1-Play-v0`` policy observations (MoE + depth encoder):
- Two-stage ONNX: ``exported/0-depth_encoder.onnx`` then ``exported/actor.onnx`` (same wiring as
  instinctlab/tasks/parkour/scripts/onnxer.py).
- Proprio terms (history_length=8, flattened) in Isaac declaration order; depth stack aligned with
  ``delayed_visualizable_image`` subsampling over a short FIFO.

MuJoCo pipeline is shared with sim2sim_beyondmimic_onnx (URDF → MJCF, position actuators).
Adds a fixed ``parkour_depth_cam`` on ``torso_link`` for off-screen depth rendering.

Visualization: ``mujoco_viewer`` is created **before** ``mujoco.Renderer`` so GLFW gets the first GL
context (avoids a blank window on some Linux stacks). If the window still fails, try
``export MUJOCO_GL=glfw`` (see MuJoCo rendering docs).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

try:
    import mujoco_viewer
except ImportError:  # pragma: no cover
    mujoco_viewer = None


def _load_beyondmimic_module():
    """Import sibling script as a module (filename is not a valid package name)."""
    p = Path(__file__).resolve().parent / "sim2sim_beyondmimic_onnx.py"
    spec = importlib.util.spec_from_file_location("_sim2sim_beyondmimic_helpers", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bm = _load_beyondmimic_module()

# Re-export commonly used symbols from BeyondMimic helper script
ISAAC_JOINT_ORDER: list[str] = _bm.ISAAC_JOINT_ORDER
NUM_ACTIONS: int = _bm.NUM_ACTIONS
default_joint_positions_lab = _bm.default_joint_positions_lab
build_gain_vectors_mj = _bm.build_gain_vectors_mj
hinge_joint_metadata = _bm.hinge_joint_metadata
lab_vec_to_mj_order = _bm.lab_vec_to_mj_order
mj_vec_to_lab_order = _bm.mj_vec_to_lab_order
quat_normalize = _bm.quat_normalize
quat_conj = _bm.quat_conj
quat_mul = _bm.quat_mul
quat_apply = _bm.quat_apply
_inject_joint_armature = _bm._inject_joint_armature
_inject_position_actuators = _bm._inject_position_actuators
_make_absolute_meshdir = _bm._make_absolute_meshdir
_ensure_mujoco_keep_visuals_in_urdf = _bm._ensure_mujoco_keep_visuals_in_urdf
_count_mesh_geoms = _bm._count_mesh_geoms


_SCRIPTS_DIR = Path(__file__).resolve().parent
_WBCHSI_ROOT = _SCRIPTS_DIR.parent
_PROJECT_ROOT = _WBCHSI_ROOT.parent

_DEFAULT_URDF = (
    _WBCHSI_ROOT
    / "source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
)
_DEFAULT_LOG_ROOT = _WBCHSI_ROOT / "logs/instinct_rl/g1_parkour"

# --- Camera / depth (ParkourEnvCfg camera + crop pipeline) ---
CAMERA_BODY_NAME = "torso_link"
CAMERA_NAME = "parkour_depth_cam"
CAMERA_POS_BODY = np.array([0.0487988662332928, 0.01, 0.4378029937970051], dtype=np.float64)
CAMERA_QUAT_WXYZ = np.array([0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0], dtype=np.float64)
CAMERA_FOVY_DEG = 58.29
RENDER_H, RENDER_W = 36, 64
# CropAndResizeCfg crop_region: up, down, left, right on (H, W)
CROP_UP, CROP_DOWN, CROP_LEFT, CROP_RIGHT = 18, 0, 16, 16
DEPTH_AFTER_CROP_H = RENDER_H - CROP_UP - CROP_DOWN
DEPTH_AFTER_CROP_W = RENDER_W - CROP_LEFT - CROP_RIGHT

HISTORY_LEN = 8
SENSOR_HISTORY_LEN = 37  # NoisyGroupedRayCasterCfg data_histories distance_to_image_plane_noised
HISTORY_SKIP = 5
NUM_OUTPUT_FRAMES = 8
# frame_indices = sensor_history_length - frame_offset - delay - 1  (delay=0), offsets [35..0] step 5
_DEPTH_SUB_INDICES = tuple(SENSOR_HISTORY_LEN - off - 1 for off in range((NUM_OUTPUT_FRAMES - 1) * HISTORY_SKIP, -1, -HISTORY_SKIP))

DEPTH_CLIP_MIN = 0.0
DEPTH_CLIP_MAX = 2.5

_GRAVITY_DIR_W = np.array([0.0, 0.0, -1.0], dtype=np.float64)


class Hist8:
    """Rolling buffer (oldest row 0, newest row -1)."""

    def __init__(self, feat_dim: int):
        self.buf = np.zeros((HISTORY_LEN, feat_dim), dtype=np.float32)

    def push(self, row: np.ndarray) -> None:
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = np.asarray(row, dtype=np.float32).reshape(-1)

    def flat(self) -> np.ndarray:
        return self.buf.reshape(-1)


def quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse rotation (matches isaaclab quat_apply_inverse)."""
    return quat_apply(quat_conj(quat_normalize(q_wxyz)), np.asarray(v, dtype=np.float64))


def projected_gravity_b(quat_root_wxyz: np.ndarray) -> np.ndarray:
    g = _GRAVITY_DIR_W / (np.linalg.norm(_GRAVITY_DIR_W) + 1e-12)
    return quat_apply_inverse(quat_root_wxyz, g).astype(np.float32)


def crop_depth_hw(depth_hw: np.ndarray) -> np.ndarray:
    """Crop Isaac pipeline (up, down, left, right)."""
    h, w = depth_hw.shape
    r0, r1 = CROP_UP, h - CROP_DOWN
    c0, c1 = CROP_LEFT, w - CROP_RIGHT
    return np.asarray(depth_hw[r0:r1, c0:c1], dtype=np.float64)


def blur_depth(d: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur kernel 3 equivalent (sigma=1)."""
    return gaussian_filter(d, sigma=sigma, mode="nearest")


def normalize_depth_range(d: np.ndarray) -> np.ndarray:
    d = np.clip(d, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
    return ((d - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN + 1e-12)).astype(np.float32)


def process_raw_depth(depth_hw: np.ndarray) -> np.ndarray:
    """Full CPU depth pipeline → (crop_h, crop_w) float32 [0,1]."""
    cropped = crop_depth_hw(depth_hw)
    blurred = blur_depth(cropped.astype(np.float64))
    return normalize_depth_range(blurred)


class DepthRing:
    """FIFO of processed depth frames (newest at end); maxlen matches Isaac sensor history."""

    def __init__(self, maxlen: int = SENSOR_HISTORY_LEN):
        self._dq: deque[np.ndarray] = deque(maxlen=maxlen)
        self._zeros = np.zeros((DEPTH_AFTER_CROP_H, DEPTH_AFTER_CROP_W), dtype=np.float32)

    def push(self, frame_hw: np.ndarray) -> None:
        self._dq.append(frame_hw.astype(np.float32).copy())

    def stack_for_encoder(self) -> np.ndarray:
        """Shape (8, H, W) oldest slice first — subsampled along FIFO like delayed_visualizable_image."""
        buf_len = len(self._dq)
        frames = []
        if buf_len == 0:
            for _ in range(NUM_OUTPUT_FRAMES):
                frames.append(self._zeros)
            return np.stack(frames, axis=0)

        arr = np.stack(list(self._dq), axis=0)  # (buf_len, H, W)
        for idx in _DEPTH_SUB_INDICES:
            j = int(idx)
            if j < 0:
                j = 0
            if j >= buf_len:
                j = buf_len - 1
            frames.append(arr[j])
        return np.stack(frames, axis=0)


def resolve_parkour_onnx_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.exported_dir:
        d = Path(args.exported_dir).expanduser().resolve()
        enc = d / "0-depth_encoder.onnx"
        act = d / "actor.onnx"
        return enc, act

    if not args.load_run:
        raise ValueError("Provide --load_run or --exported_dir")
    exp_dir = Path(args.log_root).expanduser().resolve() / args.load_run
    exported = exp_dir / "exported"
    return exported / "0-depth_encoder.onnx", exported / "actor.onnx"


def _patch_urdf_mesh_paths(urdf_text: str, src_path: str) -> str:
    """Fix mesh paths for multiple G1 URDF layouts loaded from /tmp via MuJoCo.

    - Omniretarget spherehand: ``<compiler meshdir="assets"/>`` plus ``filename="assets/foo.obj"``
      becomes ``assets/assets/foo.obj`` — use absolute ``meshdir`` and strip the redundant ``assets/``
      prefix from filenames.
    - BeyondMimic popsicle: ``../meshes/``.
    - Parkour shoe: ``../../../assets/resources/unitree_g1/meshes/``.
    """
    dirname = os.path.dirname(os.path.abspath(src_path))

    if 'filename="assets/' in urdf_text or "filename='assets/" in urdf_text:
        assets_abs = os.path.normpath(os.path.join(dirname, "assets"))
        urdf_text = urdf_text.replace('meshdir="assets"', f'meshdir="{assets_abs}"')
        urdf_text = urdf_text.replace("meshdir='assets'", f"meshdir='{assets_abs}'")
        urdf_text = urdf_text.replace('filename="assets/', 'filename="')
        urdf_text = urdf_text.replace("filename='assets/", "filename='")

    meshes_alt = os.path.normpath(os.path.join(dirname, "..", "meshes"))
    urdf_text = urdf_text.replace('filename="../meshes/', f'filename="{meshes_alt}/')

    rel_assets = "../../../assets/resources/unitree_g1/meshes/"
    abs_assets = os.path.normpath(os.path.join(dirname, rel_assets))
    urdf_text = urdf_text.replace(f'filename="{rel_assets}', f'filename="{abs_assets}/')
    urdf_text = urdf_text.replace(f"filename='{rel_assets}", f"filename='{abs_assets}/")
    return urdf_text


def _find_body_et(elem: ET.Element, name: str) -> ET.Element | None:
    if elem.tag == "body" and elem.get("name") == name:
        return elem
    for ch in elem:
        hit = _find_body_et(ch, name)
        if hit is not None:
            return hit
    return None


def _inject_depth_camera_mjcf(root: ET.Element) -> None:
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("MJCF has no <worldbody>")
    body = _find_body_et(wb, CAMERA_BODY_NAME)
    if body is None:
        raise RuntimeError(f"Body {CAMERA_BODY_NAME!r} not found in MJCF (needed for depth cam)")
    w, x, y, z = CAMERA_QUAT_WXYZ.tolist()
    ET.SubElement(
        body,
        "camera",
        {
            "name": CAMERA_NAME,
            "pos": f"{CAMERA_POS_BODY[0]} {CAMERA_POS_BODY[1]} {CAMERA_POS_BODY[2]}",
            "quat": f"{w} {x} {y} {z}",
            "fovy": f"{CAMERA_FOVY_DEG}",
        },
    )


def _materialize_parkour_urdf_as_mjcf(urdf_path: str) -> tuple[str, list[str]]:
    """Return path to robot MJCF XML written on disk + paths to delete."""
    src = os.path.abspath(urdf_path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"URDF robot model not found: {src}")

    with open(src, encoding="utf-8") as f:
        urdf_text = f.read()

    urdf_text = _ensure_mujoco_keep_visuals_in_urdf(urdf_text)
    urdf_text = _patch_urdf_mesh_paths(urdf_text, src)

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
        urdf_text = urdf_text[: m_robot.end()] + floating_insert + urdf_text[m_robot.end() :]

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

    _inject_position_actuators(root, body_joint_names)
    _inject_joint_armature(root)
    _inject_depth_camera_mjcf(root)

    tree.write(robot_xml_path, encoding="unicode")
    return robot_xml_path, [tmp_urdf.name]


def build_model_parkour(urdf_path: str) -> mujoco.MjModel:
    robot_xml_path, extra_cleanup = _materialize_parkour_urdf_as_mjcf(urdf_path)
    robot_xml_abs = os.path.abspath(robot_xml_path)

    scene_xml = f"""\
<mujoco model="sim2sim_parkour">
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
        f"[INFO] MuJoCo parkour model: nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"nmesh={model.nmesh}, mesh_geoms={mesh_geoms}"
    )
    if model.nu != NUM_ACTIONS:
        warnings.warn(
            f"Expected nu={NUM_ACTIONS} position actuators, got nu={model.nu}.",
            stacklevel=2,
        )
    return model


def expected_proprio_dim() -> int:
    """Default PolicyCfg slice without depth (8-frame stacks)."""
    d_ang, d_grav, d_cmd = 3, 3, 3
    d_j = NUM_ACTIONS
    return HISTORY_LEN * (d_ang + d_grav + d_cmd + d_j + d_j + d_j)


def flatten_policy_obs(
    h_base_ang: Hist8,
    h_grav: Hist8,
    h_cmd: Hist8,
    h_jp: Hist8,
    h_jv: Hist8,
    h_act: Hist8,
    depth_stack: np.ndarray,
) -> np.ndarray:
    """Same term order as ObservationsCfg.PolicyCfg / VecEnvWrapper flatten."""
    depth_flat = depth_stack.reshape(-1)
    return np.concatenate(
        [
            h_base_ang.flat(),
            h_grav.flat(),
            h_cmd.flat(),
            h_jp.flat(),
            h_jv.flat(),
            h_act.flat(),
            depth_flat.astype(np.float32),
        ],
        axis=0,
    )


def write_dummy_onnx_pair(
    export_dir: Path,
    depth_shape: tuple[int, int, int],
    proprio_dim: int,
    latent_dim: int,
    act_dim: int,
) -> None:
    import onnx
    from onnx import TensorProto, helper
    from onnx.numpy_helper import from_array

    export_dir.mkdir(parents=True, exist_ok=True)
    c, h, w = depth_shape
    din = c * h * w

    # depth_encoder: flatten + linear → latent
    flatten_out = helper.make_tensor_value_info("flatten_out", TensorProto.FLOAT, [None, din])
    encoder_nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "We"], ["enc_lin"]),
        helper.make_node("Add", ["enc_lin", "Be"], ["output"]),
    ]
    We = from_array(np.zeros((din, latent_dim), dtype=np.float32), name="We")
    Be = from_array(np.zeros((latent_dim,), dtype=np.float32), name="Be")
    enc_graph = helper.make_graph(
        encoder_nodes,
        "dummy_depth_encoder",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, c, h, w])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, latent_dim])],
        initializer=[We, Be],
    )
    enc_model = helper.make_model(enc_graph, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(enc_model)
    onnx.save(enc_model, str(export_dir / "0-depth_encoder.onnx"))

    ain = proprio_dim + latent_dim
    Wa = from_array(np.zeros((ain, act_dim), dtype=np.float32), name="Wa")
    Ba = from_array(np.zeros((act_dim,), dtype=np.float32), name="Ba")
    actor_nodes = [
        helper.make_node("MatMul", ["input", "Wa"], ["al"]),
        helper.make_node("Add", ["al", "Ba"], ["output"]),
    ]
    actor_graph = helper.make_graph(
        actor_nodes,
        "dummy_actor",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, ain])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, act_dim])],
        initializer=[Wa, Ba],
    )
    actor_model = helper.make_model(actor_graph, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(actor_model)
    onnx.save(actor_model, str(export_dir / "actor.onnx"))
    print(f"[dummy_assets] wrote ONNX pair under {export_dir}")


def run_mujoco(args: argparse.Namespace) -> None:
    sess_opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]

    if args.dummy_assets:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "parkour_sim2sim_dummy_exported"
        depth_shape = (NUM_OUTPUT_FRAMES, DEPTH_AFTER_CROP_H, DEPTH_AFTER_CROP_W)
        latent_dim = 128
        proprio_dim = expected_proprio_dim()
        write_dummy_onnx_pair(tmp, depth_shape, proprio_dim, latent_dim, NUM_ACTIONS)
        encoder_path = tmp / "0-depth_encoder.onnx"
        actor_path = tmp / "actor.onnx"
        args.sim_duration = min(float(args.sim_duration), 0.25)
        args.no_viewer = True
    else:
        encoder_path, actor_path = resolve_parkour_onnx_paths(args)

    if not encoder_path.is_file():
        raise FileNotFoundError(f"Depth encoder ONNX not found: {encoder_path}")
    if not actor_path.is_file():
        raise FileNotFoundError(f"Actor ONNX not found: {actor_path}")

    enc_sess = ort.InferenceSession(str(encoder_path), sess_opts, providers=providers)
    act_sess = ort.InferenceSession(str(actor_path), sess_opts, providers=providers)
    enc_in_name = enc_sess.get_inputs()[0].name
    act_in_name = act_sess.get_inputs()[0].name

    enc_in_shape = enc_sess.get_inputs()[0].shape
    latent_dim = int(enc_sess.get_outputs()[0].shape[-1])
    actor_in_dim = act_sess.get_inputs()[0].shape[-1]
    if not isinstance(actor_in_dim, int):
        raise RuntimeError(f"Actor ONNX needs fixed last dim; got {act_sess.get_inputs()[0].shape}")

    enc_elided = 1
    for d in enc_in_shape[1:]:
        if isinstance(d, int):
            enc_elided *= d
        else:
            raise RuntimeError(f"Depth encoder needs fixed H,W,C dims for proprio split; got {enc_in_shape}")

    proprio_dim_expect = int(actor_in_dim) - latent_dim
    proprio_dim_cfg = expected_proprio_dim()
    if proprio_dim_expect != proprio_dim_cfg:
        raise RuntimeError(
            f"Actor ONNX expects proprio dim {proprio_dim_expect} (actor_in {actor_in_dim} - latent {latent_dim}) "
            f"but this script builds {proprio_dim_cfg} from Instinct-Parkour PolicyCfg. "
            "Export ONNX from the same task / update observation builder if you changed PolicyCfg."
        )

    depth_nelems = enc_elided

    model = build_model_parkour(str(Path(args.mujoco_urdf).expanduser().resolve()))
    model.opt.timestep = float(args.dt)

    hinge_ids, hinge_names = hinge_joint_metadata(model)
    _, _, _, action_scale_mj = build_gain_vectors_mj(hinge_names)

    mj_body_name = args.root_body
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_body_name)
    if bid < 0:
        raise RuntimeError(f"Body {mj_body_name!r} not found in compiled model.")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera {CAMERA_NAME!r} missing from model.")

    data = mujoco.MjData(model)
    default_lab = default_joint_positions_lab()

    init_z = float(args.init_height)
    data.qpos[0:3] = [0.0, 0.0, init_z]
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
        if not args.show_collision_geoms:
            vopt = getattr(viewer, "vopt", None)
            if vopt is not None:
                vopt.geomgroup[0] = 0
                vopt.geomgroup[1] = 1
                vopt.geomgroup[2] = 1

    # Off-screen depth: create after interactive viewer so GLFW wins first GL context on picky drivers.
    skip_renderer = bool(args.no_viewer and args.dummy_depth)
    renderer: mujoco.Renderer | None = None
    cam_mjv = mujoco.MjvCamera()
    cam_mjv.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam_mjv.fixedcamid = cam_id
    if not skip_renderer:
        renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer.enable_depth_rendering()

    cmd_vec = np.array([args.cmd_lin_vel_x, args.cmd_lin_vel_y, args.cmd_ang_vel_z], dtype=np.float32)

    h_ang = Hist8(3)
    h_grav = Hist8(3)
    h_cmd = Hist8(3)
    h_jp = Hist8(NUM_ACTIONS)
    h_jv = Hist8(NUM_ACTIONS)
    h_act = Hist8(NUM_ACTIONS)
    depth_ring = DepthRing()

    target_q_mj = q_defaults_mj.copy()
    dec = int(args.decimation)

    n_steps = int(float(args.sim_duration) / float(args.dt))

    for step_i in tqdm(range(n_steps), desc="Simulating..."):
        mujoco.mj_forward(model, data)

        robot_root_quat = data.xquat[bid].astype(np.float64).copy()

        res_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, res_vel, 1)
        omega_body = res_vel[0:3].copy()

        q_mj = np.array([data.qpos[model.jnt_qposadr[jid]] for jid in hinge_ids], dtype=np.float64)
        dq_mj = np.array([data.qvel[model.jnt_dofadr[jid]] for jid in hinge_ids], dtype=np.float64)
        q_lab = mj_vec_to_lab_order(q_mj, hinge_names)
        dq_lab = mj_vec_to_lab_order(dq_mj, hinge_names)

        policy_tick = step_i % dec == 0

        if policy_tick:
            ang_scaled = (omega_body * 0.25).astype(np.float32)
            grav_b = projected_gravity_b(robot_root_quat)
            jp_rel = (q_lab - default_lab).astype(np.float32)
            jv_scaled = (dq_lab.astype(np.float32) * 0.05)

            h_ang.push(ang_scaled)
            h_grav.push(grav_b)
            h_cmd.push(cmd_vec)
            h_jp.push(jp_rel)
            h_jv.push(jv_scaled)

            if renderer is not None:
                renderer.update_scene(data, camera=cam_mjv)
                if args.dummy_depth:
                    depth_hw = np.full((RENDER_H, RENDER_W), 1.0, dtype=np.float32)
                else:
                    depth_hw = renderer.render()
                    depth_hw = np.asarray(depth_hw, dtype=np.float32)
            else:
                depth_hw = np.full((RENDER_H, RENDER_W), 1.0, dtype=np.float32)

            depth_ring.push(process_raw_depth(depth_hw))
            d_stack = depth_ring.stack_for_encoder()

            obs_flat = flatten_policy_obs(h_ang, h_grav, h_cmd, h_jp, h_jv, h_act, d_stack)

            if obs_flat.shape[-1] != proprio_dim_cfg + depth_nelems:
                raise RuntimeError(
                    f"Built obs dim {obs_flat.shape[-1]} != proprio+depth {proprio_dim_cfg + depth_nelems}."
                )

            proprio_flat = obs_flat[:proprio_dim_cfg].astype(np.float32).reshape(1, -1)
            depth_in = d_stack.reshape(1, *enc_in_shape[1:]).astype(np.float32)

            latent = enc_sess.run(None, {enc_in_name: depth_in})[0]
            actor_in = np.concatenate([proprio_flat, latent.astype(np.float32)], axis=1)
            if actor_in.shape[1] != actor_in_dim:
                raise RuntimeError(f"Actor input dim {actor_in.shape[1]} != ONNX {actor_in_dim}")

            action_lab = np.asarray(act_sess.run(None, {act_in_name: actor_in})[0], dtype=np.float32).reshape(-1)
            if action_lab.shape[0] != NUM_ACTIONS:
                raise RuntimeError(f"Policy output dim {action_lab.shape[0]} != {NUM_ACTIONS}")

            h_act.push(action_lab)

            action_mj = lab_vec_to_mj_order(action_lab.astype(np.float64), hinge_names)
            target_q_mj = action_mj * action_scale_mj + q_defaults_mj

        data.ctrl[:] = target_q_mj
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.render()
            viewer.cam.lookat[:2] = data.qpos[:2]

    if renderer is not None:
        renderer.close()
    if viewer is not None:
        viewer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parkour G1 ONNX Sim2Sim (depth encoder + MoE actor ONNX).",
        epilog="If the viewer window is blank, try: export MUJOCO_GL=glfw",
    )
    p.add_argument("--task", type=str, default="Instinct-Parkour-Target-Amp-G1-Play-v0", help="Task id (info only).")
    p.add_argument("--load_run", type=str, default=None, help="Experiment folder under logs/.../g1_parkour/")
    p.add_argument("--log_root", type=str, default=str(_DEFAULT_LOG_ROOT), help="Root log dir for g1_parkour.")
    p.add_argument("--exported_dir", type=str, default=None, help="Path to exported/ with both ONNX files (overrides load_run).")
    p.add_argument("--mujoco_urdf", type=str, default=str(_DEFAULT_URDF), help="Robot URDF for MuJoCo.")
    p.add_argument("--root_body", type=str, default="torso_link", help="Body name used as Isaac robot root frame.")
    p.add_argument("--cmd_lin_vel_x", type=float, default=0.6, help="Constant velocity command x (base frame).")
    p.add_argument("--cmd_lin_vel_y", type=float, default=0.0)
    p.add_argument("--cmd_ang_vel_z", type=float, default=0.0)
    p.add_argument("--init_height", type=float, default=0.9, help="Initial floating-base height (matches training spawn z).")
    p.add_argument("--sim_duration", type=float, default=1200.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument(
        "--no_viewer",
        action="store_true",
        help="Headless run. With --dummy_depth, skips creating an off-screen Renderer.",
    )
    p.add_argument("--show_collision_geoms", action="store_true")
    p.add_argument(
        "--dummy_assets",
        action="store_true",
        help="Write random ONNX pair under $TMPDIR and run a short smoke test.",
    )
    p.add_argument("--dummy_depth", action="store_true", help="Skip rendering; fill depth with constants (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_mujoco(args)


if __name__ == "__main__":
    main()
