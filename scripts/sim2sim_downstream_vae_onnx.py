#!/usr/bin/env python3
"""
HSI Downstream G1 Sim2Sim using ONNX exported by scripts/instinct_rl/play.py --exportonnx.

Expected exported files under a run's exported/ directory:
- 0-depth_image.onnx
- vae_actor.onnx
- policy_normalizer.npz (optional but strongly recommended)
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
from scipy.ndimage import zoom
from tqdm import tqdm

try:
    import mujoco_viewer
except ImportError:  # pragma: no cover
    mujoco_viewer = None


def _load_beyondmimic_module():
    p = Path(__file__).resolve().parent / "sim2sim_beyondmimic_onnx.py"
    spec = importlib.util.spec_from_file_location("_sim2sim_beyondmimic_helpers", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bm = _load_beyondmimic_module()

ISAAC_JOINT_ORDER = _bm.ISAAC_JOINT_ORDER
NUM_ACTIONS = _bm.NUM_ACTIONS
default_joint_positions_lab = _bm.default_joint_positions_lab
build_gain_vectors_mj = _bm.build_gain_vectors_mj
hinge_joint_metadata = _bm.hinge_joint_metadata
lab_vec_to_mj_order = _bm.lab_vec_to_mj_order
mj_vec_to_lab_order = _bm.mj_vec_to_lab_order
quat_normalize = _bm.quat_normalize
quat_conj = _bm.quat_conj
quat_apply = _bm.quat_apply
apply_normalizer = _bm.apply_normalizer
load_normalizer = _bm.load_normalizer
_inject_joint_armature = _bm._inject_joint_armature
_inject_position_actuators = _bm._inject_position_actuators
_make_absolute_meshdir = _bm._make_absolute_meshdir
_ensure_mujoco_keep_visuals_in_urdf = _bm._ensure_mujoco_keep_visuals_in_urdf

_SCRIPTS_DIR = Path(__file__).resolve().parent
_WBCHSI_ROOT = _SCRIPTS_DIR.parent
_DEFAULT_LOG_ROOT = _WBCHSI_ROOT / "logs/instinct_rl/g1_hsidownstream_perceptive_vae"
_DEFAULT_URDF = (
    _WBCHSI_ROOT
    / "source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle_spherehand.urdf"
)

# downstream depth camera from params/env.yaml
CAMERA_BODY_NAME = "torso_link"
CAMERA_NAME = "downstream_depth_cam"
CAMERA_POS_BODY = np.array([0.0487988662332928, 0.015, 0.4378029937970051], dtype=np.float64)
CAMERA_QUAT_WXYZ = np.array([0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0], dtype=np.float64)
CAMERA_FOVY_DEG = 58.29
RENDER_H, RENDER_W = 27, 48
DEPTH_MIN, DEPTH_MAX = 0.0, 2.0
CROP_UP, CROP_DOWN, CROP_LEFT, CROP_RIGHT = 2, 2, 2, 2
RESIZE_H, RESIZE_W = 18, 32
DEPTH_HISTORY = 37
DEPTH_SKIP = 5
DEPTH_OUT_FRAMES = 8
DEPTH_FRAME_IDXS = tuple(
    DEPTH_HISTORY - off - 1 for off in range((DEPTH_OUT_FRAMES - 1) * DEPTH_SKIP, -1, -DEPTH_SKIP)
)

PROPRIO_HISTORY = 4
CMD_HISTORY = 8
ANG_DIM = 3
CMD_DIM = 3
JOINT_DIM = NUM_ACTIONS
LATENT_DIM_CFG = 32

DEPTH_FEATURE_DIM = DEPTH_OUT_FRAMES * RESIZE_H * RESIZE_W
PROPRIO_FEATURE_DIM = (
    PROPRIO_HISTORY * ANG_DIM  # projected_gravity
    + CMD_HISTORY * CMD_DIM  # velocity_commands
    + PROPRIO_HISTORY * ANG_DIM  # base_ang_vel
    + PROPRIO_HISTORY * JOINT_DIM  # joint_pos
    + PROPRIO_HISTORY * JOINT_DIM  # joint_vel
    + PROPRIO_HISTORY * JOINT_DIM  # last_action
)
RAW_OBS_DIM = DEPTH_FEATURE_DIM + PROPRIO_FEATURE_DIM
ENCODER_OBS_DIM = PROPRIO_FEATURE_DIM + LATENT_DIM_CFG


def quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(quat_normalize(q_wxyz)), np.asarray(v, dtype=np.float64))


def projected_gravity_b(quat_root_wxyz: np.ndarray) -> np.ndarray:
    g = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)
    return quat_apply_inverse(quat_root_wxyz, g).astype(np.float32)


class Hist:
    def __init__(self, history_len: int, feat_dim: int):
        self.buf = np.zeros((history_len, feat_dim), dtype=np.float32)

    def push(self, row: np.ndarray) -> None:
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = np.asarray(row, dtype=np.float32).reshape(-1)

    def flat(self) -> np.ndarray:
        return self.buf.reshape(-1)


class DepthRing:
    def __init__(
        self,
        maxlen: int = DEPTH_HISTORY,
        frame_idxs: tuple[int, ...] = DEPTH_FRAME_IDXS,
        frame_shape: tuple[int, int] = (RESIZE_H, RESIZE_W),
    ):
        self._dq: deque[np.ndarray] = deque(maxlen=maxlen)
        self._frame_idxs = tuple(int(i) for i in frame_idxs)
        self._zeros = np.zeros(frame_shape, dtype=np.float32)

    def push(self, frame_hw: np.ndarray) -> None:
        self._dq.append(np.asarray(frame_hw, dtype=np.float32).copy())

    def sampled_stack(self) -> np.ndarray:
        if len(self._dq) == 0:
            return np.stack([self._zeros for _ in self._frame_idxs], axis=0)
        arr = np.stack(list(self._dq), axis=0)
        out = []
        for idx in self._frame_idxs:
            j = int(idx)
            if j < 0:
                j = 0
            if j >= arr.shape[0]:
                j = arr.shape[0] - 1
            out.append(arr[j])
        return np.stack(out, axis=0).astype(np.float32)


def process_depth(depth_hw: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_hw, dtype=np.float64)
    d = np.clip(d, DEPTH_MIN, DEPTH_MAX)
    d = (d - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-12)
    d = d[CROP_UP : d.shape[0] - CROP_DOWN, CROP_LEFT : d.shape[1] - CROP_RIGHT]
    z_h = RESIZE_H / float(d.shape[0])
    z_w = RESIZE_W / float(d.shape[1])
    d = zoom(d, zoom=(z_h, z_w), order=1)
    return d.astype(np.float32)


def restore_viewer_context(viewer) -> None:
    """mujoco.Renderer can leave its offscreen context current; switch back before window rendering."""
    if viewer is None:
        return
    try:
        import glfw

        window = getattr(viewer, "window", None)
        if window is not None:
            glfw.make_context_current(window)
    except Exception:
        pass


def depth_frame_indices(history_len: int, history_skip: int, num_output_frames: int) -> tuple[int, ...]:
    history_len = int(history_len)
    history_skip = max(int(history_skip), 1)
    num_output_frames = max(int(num_output_frames), 1)
    frames_needed = (num_output_frames - 1) * history_skip + 1
    if frames_needed > history_len:
        raise RuntimeError(
            "Depth history is too short for exported encoder: "
            f"history={history_len}, skip={history_skip}, frames={num_output_frames}, needs at least {frames_needed}"
        )
    return tuple(history_len - off - 1 for off in range((num_output_frames - 1) * history_skip, -1, -history_skip))


def infer_depth_sampling(args: argparse.Namespace, depth_path: Path, num_output_frames: int) -> tuple[int, int, tuple[int, ...]]:
    history_len: int | None = None
    history_skip: int | None = None

    run_dir = depth_path.parent.parent if depth_path.parent.name == "exported" else depth_path.parent
    env_yaml = run_dir / "params" / "env.yaml"
    if env_yaml.is_file():
        text = env_yaml.read_text(encoding="utf-8", errors="ignore")
        m_hist = re.search(r"distance_to_image_plane_noised:\s*(\d+)", text)
        m_skip = re.search(r"history_skip_frames:\s*(\d+)", text)
        if m_hist:
            history_len = int(m_hist.group(1))
        if m_skip:
            history_skip = int(m_skip.group(1))

    if history_len is None or history_skip is None:
        for s in (args.load_run or "", str(depth_path)):
            m = re.search(r"depthHist(\d+)Skip(\d+)", s)
            if m:
                history_len = int(m.group(1))
                history_skip = int(m.group(2))
                break

    if history_len is None:
        history_len = DEPTH_HISTORY
    if history_skip is None:
        history_skip = DEPTH_SKIP

    frame_idxs = depth_frame_indices(history_len, history_skip, num_output_frames)
    return history_len, history_skip, frame_idxs


def flatten_raw_obs(
    depth_stack: np.ndarray,
    h_gravity: Hist,
    h_cmd: Hist,
    h_ang: Hist,
    h_jp: Hist,
    h_jv: Hist,
    h_act: Hist,
    expected_dim: int = RAW_OBS_DIM,
) -> np.ndarray:
    parts = [
        depth_stack.reshape(-1).astype(np.float32),
        h_gravity.flat(),
        h_cmd.flat(),
        h_ang.flat(),
        h_jp.flat(),
        h_jv.flat(),
        h_act.flat(),
    ]
    obs = np.concatenate(parts, axis=0).astype(np.float32)
    if obs.shape[0] != expected_dim:
        raise RuntimeError(f"Raw obs dim {obs.shape[0]} != expected {expected_dim}")
    return obs


def resolve_model_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    if args.depth_encoder and args.vae_actor:
        enc = Path(args.depth_encoder).expanduser().resolve()
        vae = Path(args.vae_actor).expanduser().resolve()
        norm = Path(args.policy_normalizer).expanduser().resolve() if args.policy_normalizer else None
        if norm is None:
            cand = vae.parent / "policy_normalizer.npz"
            norm = cand if cand.is_file() else None
        return enc, vae, norm

    if args.exported_dir:
        exported = Path(args.exported_dir).expanduser().resolve()
    else:
        if not args.load_run:
            raise ValueError("Provide --load_run, --exported_dir, or explicit --depth_encoder/--vae_actor")
        exported = Path(args.log_root).expanduser().resolve() / args.load_run / "exported"

    enc = exported / "0-depth_image.onnx"
    vae = exported / "vae_actor.onnx"
    norm = exported / "policy_normalizer.npz"
    norm = norm if norm.is_file() else None
    if args.policy_normalizer:
        norm = Path(args.policy_normalizer).expanduser().resolve()
    return enc, vae, norm


def write_dummy_assets(export_dir: Path, latent_dim: int = LATENT_DIM_CFG) -> tuple[Path, Path, Path]:
    import onnx
    from onnx import TensorProto, helper
    from onnx.numpy_helper import from_array

    export_dir.mkdir(parents=True, exist_ok=True)
    depth_path = export_dir / "0-depth_image.onnx"
    vae_path = export_dir / "vae_actor.onnx"
    norm_path = export_dir / "policy_normalizer.npz"

    d_in = DEPTH_OUT_FRAMES * RESIZE_H * RESIZE_W
    w_depth = np.zeros((d_in, latent_dim), dtype=np.float32)
    b_depth = np.zeros((latent_dim,), dtype=np.float32)
    depth_nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["output"]),
    ]
    depth_graph = helper.make_graph(
        depth_nodes,
        "dummy_depth",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, DEPTH_OUT_FRAMES, RESIZE_H, RESIZE_W])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, latent_dim])],
        initializer=[from_array(w_depth, "W"), from_array(b_depth, "B")],
    )
    depth_model = helper.make_model(depth_graph, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(depth_model)
    onnx.save(depth_model, str(depth_path))

    in_dim = PROPRIO_FEATURE_DIM + latent_dim
    w_actor = np.zeros((in_dim, NUM_ACTIONS), dtype=np.float32)
    b_actor = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    z_mean_w = np.zeros((in_dim, latent_dim), dtype=np.float32)
    z_std_w = np.zeros((in_dim, latent_dim), dtype=np.float32)
    actor_nodes = [
        helper.make_node("MatMul", ["input", "W_act"], ["act_mm"]),
        helper.make_node("Add", ["act_mm", "B_act"], ["output"]),
        helper.make_node("MatMul", ["input", "W_zm"], ["latent_mean"]),
        helper.make_node("MatMul", ["input", "W_zs"], ["latent_std"]),
    ]
    actor_graph = helper.make_graph(
        actor_nodes,
        "dummy_vae_actor",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, in_dim])],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, NUM_ACTIONS]),
            helper.make_tensor_value_info("latent_mean", TensorProto.FLOAT, [None, latent_dim]),
            helper.make_tensor_value_info("latent_std", TensorProto.FLOAT, [None, latent_dim]),
        ],
        initializer=[
            from_array(w_actor, "W_act"),
            from_array(b_actor, "B_act"),
            from_array(z_mean_w, "W_zm"),
            from_array(z_std_w, "W_zs"),
        ],
    )
    actor_model = helper.make_model(actor_graph, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(actor_model)
    onnx.save(actor_model, str(vae_path))

    np.savez(norm_path, mean=np.zeros((RAW_OBS_DIM,), dtype=np.float32), std=np.ones((RAW_OBS_DIM,), dtype=np.float32), eps=1e-2)
    return depth_path, vae_path, norm_path


def _patch_urdf_mesh_paths(urdf_text: str, src_path: str) -> str:
    dirname = os.path.dirname(os.path.abspath(src_path))
    if 'filename="assets/' in urdf_text or "filename='assets/" in urdf_text:
        assets_abs = os.path.normpath(os.path.join(dirname, "assets"))
        urdf_text = urdf_text.replace('meshdir="assets"', f'meshdir="{assets_abs}"')
        urdf_text = urdf_text.replace("meshdir='assets'", f"meshdir='{assets_abs}'")
        urdf_text = urdf_text.replace('filename="assets/', 'filename="')
        urdf_text = urdf_text.replace("filename='assets/", "filename='")
    meshes_alt = os.path.normpath(os.path.join(dirname, "..", "meshes"))
    urdf_text = urdf_text.replace('filename="../meshes/', f'filename="{meshes_alt}/')
    omni_assets = os.path.normpath(os.path.join(dirname, "..", "omniretarget_models", "g1", "assets"))
    urdf_text = urdf_text.replace(
        'filename="../omniretarget_models/g1/assets/',
        f'filename="{omni_assets}/',
    )
    return urdf_text


def _find_body_et(elem: ET.Element, name: str) -> ET.Element | None:
    if elem.tag == "body" and elem.get("name") == name:
        return elem
    for ch in elem:
        got = _find_body_et(ch, name)
        if got is not None:
            return got
    return None


def _inject_depth_camera_mjcf(root: ET.Element) -> None:
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("MJCF has no <worldbody>")
    body = _find_body_et(wb, CAMERA_BODY_NAME)
    if body is None:
        raise RuntimeError(f"Body {CAMERA_BODY_NAME!r} not found in MJCF.")
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


def _materialize_urdf_as_mjcf(urdf_path: str) -> tuple[str, list[str]]:
    src = os.path.abspath(urdf_path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"URDF not found: {src}")
    with open(src, encoding="utf-8") as f:
        urdf_text = f.read()
    urdf_text = _ensure_mujoco_keep_visuals_in_urdf(urdf_text)
    urdf_text = _patch_urdf_mesh_paths(urdf_text, src)
    if "floating_base_joint" not in urdf_text:
        m_robot = re.search(r"<robot\s[^>]*>", urdf_text)
        if not m_robot:
            raise RuntimeError("Could not find <robot ...> in URDF.")
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
            raise RuntimeError(f"Converted model has unnamed joint id {joint_id}.")
        body_joint_names.append(str(jname))
    _inject_position_actuators(root, body_joint_names)
    _inject_joint_armature(root)
    _inject_depth_camera_mjcf(root)
    tree.write(robot_xml_path, encoding="unicode")
    return robot_xml_path, [tmp_urdf.name]


def build_model_downstream(urdf_path: str) -> mujoco.MjModel:
    robot_xml_path, extra_cleanup = _materialize_urdf_as_mjcf(urdf_path)
    robot_xml_abs = os.path.abspath(robot_xml_path)
    scene_xml = f"""\
<mujoco model="sim2sim_downstream_vae">
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
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane" group="2"/>
  </worldbody>
</mujoco>
"""
    scene_fd = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", encoding="utf-8")
    scene_fd.write(scene_xml)
    scene_fd.close()
    model = mujoco.MjModel.from_xml_path(scene_fd.name)
    try:
        os.remove(scene_fd.name)
        os.remove(robot_xml_path)
        for p in extra_cleanup:
            if os.path.exists(p):
                os.remove(p)
    except OSError:
        pass
    if model.nu != NUM_ACTIONS:
        raise RuntimeError(f"Expected nu={NUM_ACTIONS}, got {model.nu}")
    return model


def run_mujoco(args: argparse.Namespace) -> None:
    if args.dummy_assets:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "downstream_sim2sim_dummy"
        depth_path, vae_path, norm_path = write_dummy_assets(tmp)
        print(f"[dummy_assets] wrote models under {tmp}")
    else:
        depth_path, vae_path, norm_path = resolve_model_paths(args)

    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth encoder ONNX not found: {depth_path}")
    if not vae_path.is_file():
        raise FileNotFoundError(
            f"VAE actor ONNX not found: {vae_path}\nRun play export first (scripts/instinct_rl/play.py --exportonnx)."
        )
    providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    depth_sess = ort.InferenceSession(str(depth_path), sess_opts, providers=providers)
    vae_sess = ort.InferenceSession(str(vae_path), sess_opts, providers=providers)
    depth_in_name = depth_sess.get_inputs()[0].name
    vae_in_name = vae_sess.get_inputs()[0].name

    depth_shape = depth_sess.get_inputs()[0].shape
    if not all(isinstance(d, int) for d in depth_shape[1:]):
        raise RuntimeError(f"Depth encoder requires fixed [C,H,W], got {depth_shape}")
    d_f, d_h, d_w = int(depth_shape[1]), int(depth_shape[2]), int(depth_shape[3])
    if (d_h, d_w) != (RESIZE_H, RESIZE_W):
        raise RuntimeError(
            f"Depth encoder expects image size {(d_h, d_w)} but script builds {(RESIZE_H, RESIZE_W)}"
        )
    depth_history, depth_skip, depth_frame_idxs = infer_depth_sampling(args, depth_path, d_f)
    depth_feature_dim = d_f * d_h * d_w
    raw_obs_dim = depth_feature_dim + PROPRIO_FEATURE_DIM
    print(
        "[sim2sim] depth sampling "
        f"history={depth_history}, skip={depth_skip}, frames={d_f}, frame_idxs={depth_frame_idxs}; "
        f"raw_obs_dim={raw_obs_dim}"
    )

    mean_npz, std_npz, eps_npz = load_normalizer(norm_path)
    if mean_npz is None:
        warnings.warn("policy_normalizer.npz missing; running without obs normalization.")
    elif mean_npz.shape[0] != raw_obs_dim:
        raise RuntimeError(
            f"Normalizer dim {mean_npz.shape[0]} != expected raw obs dim {raw_obs_dim} "
            f"(depth {depth_feature_dim} + proprio {PROPRIO_FEATURE_DIM}). "
            "Check that --load_run/--exported_dir/--policy_normalizer belong to the same export."
        )
    latent_dim = int(depth_sess.get_outputs()[0].shape[-1])
    if latent_dim != LATENT_DIM_CFG:
        warnings.warn(f"Depth latent dim is {latent_dim}, expected cfg {LATENT_DIM_CFG}; using ONNX-reported dim.")

    vae_in_dim = vae_sess.get_inputs()[0].shape[-1]
    if not isinstance(vae_in_dim, int):
        raise RuntimeError(f"vae_actor.onnx needs fixed last dim; got {vae_sess.get_inputs()[0].shape}")
    expected_vae_in = PROPRIO_FEATURE_DIM + latent_dim
    if vae_in_dim != expected_vae_in:
        raise RuntimeError(f"vae_actor input dim {vae_in_dim} != expected {expected_vae_in}")

    model = build_model_downstream(str(Path(args.mujoco_urdf).expanduser().resolve()))
    model.opt.timestep = float(args.dt)
    hinge_ids, hinge_names = hinge_joint_metadata(model)
    _, _, _, action_scale_mj = build_gain_vectors_mj(hinge_names)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.root_body)
    if bid < 0:
        raise RuntimeError(f"Body {args.root_body!r} not found in model.")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera {CAMERA_NAME!r} missing from model.")

    data = mujoco.MjData(model)
    default_lab = default_joint_positions_lab()
    data.qpos[0:3] = [0.0, 0.0, float(args.init_height)]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    q_defaults_mj = lab_vec_to_mj_order(default_lab, hinge_names)
    data.qpos[7 : 7 + NUM_ACTIONS] = q_defaults_mj
    mujoco.mj_forward(model, data)
    data.ctrl[:] = q_defaults_mj

    viewer = None
    if not args.no_viewer:
        if mujoco_viewer is None:
            raise RuntimeError("mujoco_viewer not installed; use --no_viewer")
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = 5.0
        if not args.show_collision_geoms:
            vopt = getattr(viewer, "vopt", None)
            if vopt is not None:
                vopt.geomgroup[0] = 0
                vopt.geomgroup[1] = 1
                vopt.geomgroup[2] = 1

    skip_renderer = bool(args.dummy_depth)
    renderer: mujoco.Renderer | None = None
    cam_mjv = mujoco.MjvCamera()
    cam_mjv.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam_mjv.fixedcamid = cam_id
    if not skip_renderer:
        renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer.enable_depth_rendering()
        restore_viewer_context(viewer)

    cmd_vec = np.array([args.cmd_lin_vel_x, args.cmd_lin_vel_y, args.cmd_ang_vel_z], dtype=np.float32)
    h_gravity = Hist(PROPRIO_HISTORY, ANG_DIM)
    h_cmd = Hist(CMD_HISTORY, CMD_DIM)
    h_ang = Hist(PROPRIO_HISTORY, ANG_DIM)
    h_jp = Hist(PROPRIO_HISTORY, JOINT_DIM)
    h_jv = Hist(PROPRIO_HISTORY, JOINT_DIM)
    h_act = Hist(PROPRIO_HISTORY, JOINT_DIM)
    depth_ring = DepthRing(maxlen=depth_history, frame_idxs=depth_frame_idxs, frame_shape=(d_h, d_w))

    target_q_mj = q_defaults_mj.copy()
    dec = int(args.decimation)
    n_steps = int(float(args.sim_duration) / float(args.dt))

    for step_i in tqdm(range(n_steps), desc="Simulating..."):
        mujoco.mj_forward(model, data)
        root_quat = data.xquat[bid].astype(np.float64).copy()
        res_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, res_vel, 1)
        omega_body = res_vel[0:3].copy()

        q_mj = np.array([data.qpos[model.jnt_qposadr[jid]] for jid in hinge_ids], dtype=np.float64)
        dq_mj = np.array([data.qvel[model.jnt_dofadr[jid]] for jid in hinge_ids], dtype=np.float64)
        q_lab = mj_vec_to_lab_order(q_mj, hinge_names)
        dq_lab = mj_vec_to_lab_order(dq_mj, hinge_names)

        if step_i % dec == 0:
            h_gravity.push(projected_gravity_b(root_quat))
            h_cmd.push(cmd_vec)
            h_ang.push((omega_body * 0.25).astype(np.float32))
            h_jp.push((q_lab - default_lab).astype(np.float32))
            h_jv.push((dq_lab * 0.05).astype(np.float32))

            if renderer is not None:
                renderer.update_scene(data, camera=cam_mjv)
                depth_hw = np.full((RENDER_H, RENDER_W), 1.0, dtype=np.float32) if args.dummy_depth else renderer.render()
                restore_viewer_context(viewer)
                depth_hw = np.asarray(depth_hw, dtype=np.float32)
            else:
                depth_hw = np.full((RENDER_H, RENDER_W), 1.0, dtype=np.float32)

            depth_ring.push(process_depth(depth_hw))
            depth_stack = depth_ring.sampled_stack()
            raw_obs = flatten_raw_obs(
                depth_stack,
                h_gravity,
                h_cmd,
                h_ang,
                h_jp,
                h_jv,
                h_act,
                expected_dim=raw_obs_dim,
            )
            obs_n = apply_normalizer(raw_obs, mean_npz, std_npz, eps_npz).astype(np.float32)
            depth_flat = obs_n[:depth_feature_dim]
            proprio_flat = obs_n[depth_feature_dim:]

            depth_in = depth_flat.reshape(1, d_f, d_h, d_w).astype(np.float32)
            latent = np.asarray(depth_sess.run(None, {depth_in_name: depth_in})[0], dtype=np.float32).reshape(1, -1)
            if latent.shape[1] != latent_dim:
                raise RuntimeError(f"Depth encoder output dim {latent.shape[1]} != expected {latent_dim}")

            vae_in = np.concatenate([proprio_flat.reshape(1, -1), latent], axis=1)
            if vae_in.shape[1] != vae_in_dim:
                raise RuntimeError(f"vae_actor input dim {vae_in.shape[1]} != ONNX {vae_in_dim}")

            vae_outputs = vae_sess.run(None, {vae_in_name: vae_in})
            action_lab = np.asarray(vae_outputs[0], dtype=np.float32).reshape(-1)
            if action_lab.shape[0] != NUM_ACTIONS:
                raise RuntimeError(f"Action dim {action_lab.shape[0]} != {NUM_ACTIONS}")
            h_act.push(action_lab)

            action_mj = lab_vec_to_mj_order(action_lab.astype(np.float64), hinge_names)
            target_q_mj = action_mj * action_scale_mj + q_defaults_mj

        data.ctrl[:] = target_q_mj
        mujoco.mj_step(model, data)
        if viewer is not None:
            restore_viewer_context(viewer)
            viewer.render()
            viewer.cam.lookat[:2] = data.qpos[:2]

    if renderer is not None:
        renderer.close()
    if viewer is not None:
        viewer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream VAE ONNX Sim2Sim (depth encoder + vae actor).")
    p.add_argument("--task", type=str, default="Instinct-HSIDownstream-Perceptive-Vae-G1-Play-v0")
    p.add_argument("--load_run", type=str, default=None, help="Run name under logs root.")
    p.add_argument("--log_root", type=str, default=str(_DEFAULT_LOG_ROOT))
    p.add_argument("--exported_dir", type=str, default=None, help="Path to exported/ directory.")
    p.add_argument("--depth_encoder", type=str, default=None, help="Path to 0-depth_image.onnx")
    p.add_argument("--vae_actor", type=str, default=None, help="Path to vae_actor.onnx")
    p.add_argument("--policy_normalizer", type=str, default=None, help="Path to policy_normalizer.npz")
    p.add_argument("--mujoco_urdf", type=str, default=str(_DEFAULT_URDF))
    p.add_argument("--root_body", type=str, default="torso_link")
    p.add_argument("--init_height", type=float, default=0.82)
    p.add_argument("--cmd_lin_vel_x", type=float, default=0.0)
    p.add_argument("--cmd_lin_vel_y", type=float, default=0.0)
    p.add_argument("--cmd_ang_vel_z", type=float, default=0.0)
    p.add_argument("--sim_duration", type=float, default=1200.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument("--no_viewer", action="store_true")
    p.add_argument("--show_collision_geoms", action="store_true")
    p.add_argument("--dummy_depth", action="store_true", help="Use constant depth image instead of renderer output.")
    p.add_argument("--dummy_assets", action="store_true", help="Write temporary dummy ONNX+normalizer assets and run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dummy_assets:
        args.no_viewer = True
        args.sim_duration = min(args.sim_duration, 0.25)
    run_mujoco(args)


if __name__ == "__main__":
    main()
