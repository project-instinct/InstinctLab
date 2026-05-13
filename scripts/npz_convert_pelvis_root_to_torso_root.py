#!/usr/bin/env python3
"""Convert OmniRetarget / instinct *.retargeted.npz motion from pelvis floating base to torso_link base.

Pipeline (per frame):
1. FK on pelvis-root URDF (`g1_29dof_spherehand.urdf`) with full joint vector + pelvis pose in world
   → world pose of `torso_link` becomes the new ``base_*`` trajectory (same quaternion convention wxyz).
2. FK on the pelvis-root chain yields ``T_{pelvis → torso}``; invert to get ``T_{torso → pelvis}``.
3. On torso-root URDF (`g1_29dof_torsobase_popsicle.urdf`), optimize the 3-DOF torso→pelvis serial waist chain
   so its FK matches that target; overwrite ``waist_pitch / waist_roll / waist_yaw`` in joint_pos.
4. Hip–ankle + arm joint values are preserved (by joint name).

Requires: torch, scipy, pytorch_kinematics (see instinctlab/pyproject/setup).

Example:
  python WBCHSI/scripts/npz_convert_pelvis_root_to_torso_root.py \\
    --input /path/to/motion1.retargeted.npz \\
    --output /path/out/motion1.torso_base.retargeted.npz \\
    --urdf_pelvis_base .../g1_29dof_spherehand.urdf \\
    --urdf_torso_base .../g1_29dof_torsobase_popsicle.urdf
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np


def _require_pk():
    try:
        import pytorch_kinematics as pk

        return pk
    except ImportError as e:
        raise ImportError(
            "This converter needs pytorch_kinematics (+ torch). "
            "Install inside your instinctlab env, e.g.: pip install pytorch_kinematics torch scipy numpy."
        ) from e


WAIST_SYNONYMS = frozenset({"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"})


def _normalize_joint_names(names: Sequence) -> list[str]:
    return [str(n) for n in names]


def _joint_names_for_savez(names: list[str]) -> np.ndarray:
    """Unicode array safe for np.savez / Isaac (avoid object dtype + pickle / numpy._core)."""
    if not names:
        return np.array([], dtype="U1")
    max_len = max(len(s) for s in names)
    return np.array(names, dtype=f"U{max_len}")


def _scene_object_names_for_savez(extra) -> np.ndarray:
    """Same rationale as joint_names: never persist object arrays that pickle numpy internals."""
    arr = np.asarray(extra, dtype=object)
    strings = [str(arr.flat[i]) for i in range(arr.size)]
    max_len = max((len(s) for s in strings), default=1)
    return np.array(strings, dtype=f"U{max_len}").reshape(arr.shape)


def _transform_target_res(
    tgt_mat_n: np.ndarray, pred_mat_n: np.ndarray, rot_weight: float
) -> np.ndarray:
    """6-dim residual (translation + scaled rotation flattened)."""
    dt = pred_mat_n[:3, 3] - tgt_mat_n[:3, 3]
    drt = rot_weight * (pred_mat_n[:3, :3] - tgt_mat_n[:3, :3]).reshape(-1)
    return np.concatenate([dt, drt]).astype(np.float64)


def _solve_waist_ik_serial(
    serial_chain,
    target_mat_torso_to_pelvis: np.ndarray,
    init_rad: np.ndarray,
    jac: str,
    rot_weight: float,
    tol: float,
):
    """Minimize FK errors for 3 waist joints (SerialChain torso→pelvis)."""
    import torch
    from scipy.optimize import least_squares

    tgt = np.asarray(target_mat_torso_to_pelvis, dtype=np.float64)

    bounds_lo, bounds_hi = serial_chain.get_joint_limits()
    if bounds_lo is not None:
        lb = np.maximum(np.asarray(bounds_lo[:3], dtype=np.float64), -np.inf)
        ub = np.minimum(np.asarray(bounds_hi[:3], dtype=np.float64), np.inf)
    else:
        lb, ub = -np.ones(3) * np.pi * 4, np.ones(3) * np.pi * 4

    x0 = np.clip(np.asarray(init_rad, dtype=np.float64).reshape(-1), lb + 1e-6, ub - 1e-6)

    def fun(q: np.ndarray) -> np.ndarray:
        with torch.enable_grad():  # some pk internals expect grad context when jac="2-point"
            T = serial_chain.forward_kinematics(torch.as_tensor(q, dtype=torch.float32).reshape(1, -1), end_only=True)
        M = T.get_matrix()[0].detach().cpu().numpy().astype(np.float64)
        return _transform_target_res(tgt, M, rot_weight)

    ls_base = dict(
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=120,
        bounds=(lb, ub),
        method="trf",
        verbose=0,
    )
    jac_kw: str | None = None if jac == "auto" else jac
    if jac_kw:
        ls_base["jac"] = jac_kw
    sol = least_squares(fun, x0, **ls_base)
    if jac_kw != "3-point":
        lb2 = dict(ls_base)
        lb2.pop("jac", None)
        lb2["jac"] = "3-point"
        retry = least_squares(fun, x0, **lb2)
        if retry.cost < sol.cost:
            sol = retry
    return sol.x.astype(np.float32)


def convert_npz_arrays(
    data: dict,
    urdf_pelvis_path: str,
    urdf_torso_path: str,
    jac: str = "3-point",
    rot_weight: float = 10.0,
    ik_tol: float = 5e-5,
):
    pk = _require_pk()
    import torch

    with open(urdf_pelvis_path, "rb") as f:
        chain_p = pk.build_chain_from_urdf(f.read())
    with open(urdf_torso_path, "rb") as f:
        chain_t = pk.build_chain_from_urdf(f.read())

    jp_names_sphere = chain_p.get_joint_parameter_names()
    serial_waist = pk.SerialChain(chain_t, end_frame_name="pelvis")
    waist_names_pop_serial = serial_waist.get_joint_parameter_names()
    assert len(waist_names_pop_serial) == 3, f"Expected 3-DOF torso→pelvis chain, got {waist_names_pop_serial}"

    sphere_frame_idx = chain_p.get_frame_indices("pelvis", "torso_link")

    jn_arr = data["joint_names"]
    if isinstance(jn_arr, np.ndarray):
        joint_names_npz = [str(jn_arr[i]) for i in range(jn_arr.shape[0])]
    else:
        joint_names_npz = _normalize_joint_names(list(jn_arr))

    sphere_order = torch.tensor([joint_names_npz.index(n) for n in jp_names_sphere], dtype=torch.long)

    jp = np.asarray(data["joint_pos"], dtype=np.float32)
    T_frames = jp.shape[0]
    bp = np.asarray(data["base_pos_w"], dtype=np.float32)
    bq = np.asarray(data["base_quat_w"], dtype=np.float32)

    src_base_tf = pk.Transform3d(device="cpu", rot=torch.as_tensor(bq, dtype=torch.float32), pos=torch.as_tensor(bp, dtype=torch.float32))

    joint_pos_torso = jp.copy()

    jp_t_full = torch.as_tensor(jp, dtype=torch.float32)
    joint_pos_pk_sphere = jp_t_full[:, sphere_order]

    frame_poses = chain_p.forward_kinematics(joint_pos_pk_sphere, sphere_frame_idx)
    pel_vis = frame_poses["pelvis"]
    torso_in_chain = frame_poses["torso_link"]
    rel_p_to_t = pel_vis.inverse().compose(torso_in_chain)
    tgt_root = src_base_tf.compose(rel_p_to_t)

    tgt_m = tgt_root.get_matrix().numpy()
    torso_base_pos = tgt_m[:, :3, 3].astype(np.float32)
    torso_base_quat = pk.matrix_to_quaternion(torch.as_tensor(tgt_m[:, :3, :3], dtype=torch.float32)).numpy().astype(np.float32)

    wi_pop = np.array([joint_names_npz.index(n) for n in waist_names_pop_serial], dtype=int)

    # Sphere serial pelvis→torso is yaw→roll→pitch; torso-base serial pelvis←torso is pitch→roll→yaw — reuse values as IK init (rough).
    name_to_idx_sphere = {n: jp_names_sphere.index(n) for n in WAIST_SYNONYMS}
    q_sphere_yaw_roll_pitch = joint_pos_pk_sphere[..., [name_to_idx_sphere["waist_yaw_joint"], name_to_idx_sphere["waist_roll_joint"], name_to_idx_sphere["waist_pitch_joint"],]].numpy()

    def sphere_waist_to_pop_guess(row_yrp: np.ndarray) -> np.ndarray:
        yaw, roll, pitch = row_yrp
        pitch_idx, roll_idx, yaw_idx = (
            waist_names_pop_serial.index("waist_pitch_joint"),
            waist_names_pop_serial.index("waist_roll_joint"),
            waist_names_pop_serial.index("waist_yaw_joint"),
        )
        out = np.zeros(3, dtype=np.float64)
        out[pitch_idx] = pitch
        out[roll_idx] = roll
        out[yaw_idx] = yaw
        return out

    tgt_torso_to_p_inv = []
    # T_p→t from pelvis-fixed FK frame; IK needs T_{torso→pelvis} serial from torso.identity
    for i in range(T_frames):
        M_i = torso_in_chain.get_matrix()[i].detach().cpu().numpy().astype(np.float64)
        tgt_torso_to_p_inv.append(np.linalg.inv(M_i))
    tgt_torso_to_p_inv = np.stack(tgt_torso_to_p_inv)

    failures = 0
    q_prev = sphere_waist_to_pop_guess(q_sphere_yaw_roll_pitch[0]).astype(np.float32)

    for i in range(T_frames):
        tgt_mat = tgt_torso_to_p_inv[i]
        init = q_prev.astype(np.float32) if i else sphere_waist_to_pop_guess(q_sphere_yaw_roll_pitch[i]).astype(np.float32)
        try:
            qi = _solve_waist_ik_serial(serial_waist, tgt_mat, init, jac, rot_weight, ik_tol)
            q_prev = qi
            joint_pos_torso[i, wi_pop] = qi
            # sanity FK check
            Tchk = serial_waist.forward_kinematics(torch.as_tensor(qi, dtype=torch.float32).reshape(1, 3), end_only=True)
            Mchk = Tchk.get_matrix()[0].detach().cpu().numpy()
            er = np.linalg.norm(_transform_target_res(tgt_mat, Mchk, rot_weight))
            if er > ik_tol * 100:
                failures += 1
        except Exception:
            failures += 1
            joint_pos_torso[i, wi_pop] = init.astype(np.float32)
            q_prev = joint_pos_torso[i, wi_pop]

    if failures > 0:
        print(f"WARN: waist IK flagged {failures} / {T_frames} frames with high residuals or solver issues.")

    out = {
        "framerate": np.float32(np.asarray(data["framerate"]).reshape(())) if "framerate" in data else np.float32(30.0),
        "joint_names": _joint_names_for_savez(joint_names_npz),
        "joint_pos": np.asarray(joint_pos_torso, dtype=np.float32).copy(),
        "base_pos_w": np.asarray(torso_base_pos, dtype=np.float32).copy(),
        "base_quat_w": np.asarray(torso_base_quat, dtype=np.float32).copy(),
    }

    extras = {"object_pos_w", "object_quat_w", "object_validity", "scene_object_names"}
    for key in extras:
        if key not in data:
            continue
        extra = data[key]
        if key == "scene_object_names":
            out[key] = (
                np.asarray(extra).copy()
                if isinstance(extra, np.ndarray) and extra.dtype.kind in ("U", "S")
                else _scene_object_names_for_savez(extra)
            )
            continue
        out[key] = np.asarray(extra).copy()

    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_wbchsi = os.path.normpath(os.path.join(here, ".."))

    defaults = dict(
        urdf_pelvis_base=os.path.join(
            repo_wbchsi,
            "source/instinctlab/instinctlab/assets/resources/unitree_g1/omniretarget_models/g1/g1_29dof_spherehand.urdf",
        ),
        urdf_torso_base=os.path.join(
            repo_wbchsi,
            "source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf",
        ),
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="Input *.npz motion (pelvis-root retarget)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output *.npz (torso_link root)")
    parser.add_argument(
        "--urdf_pelvis_base",
        type=str,
        default=defaults["urdf_pelvis_base"],
        help="Pelvis-based G1 omniretarget URDF.",
    )
    parser.add_argument(
        "--urdf_torso_base",
        type=str,
        default=defaults["urdf_torso_base"],
        help="Torso-root G1 popsicle URDF.",
    )
    parser.add_argument("--rot_weight", type=float, default=10.0, help="IK weight on rotation residuals vs translation")
    parser.add_argument("--ik_tol", type=float, default=5e-5, help="SciPy IK stopping tolerance scalars.")
    parser.add_argument("--jac", type=str, default="3-point", choices=["auto", "2-point", "3-point"])

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    raw = dict(np.load(args.input, allow_pickle=True))

    out = convert_npz_arrays(raw, args.urdf_pelvis_base, args.urdf_torso_base, jac=args.jac, rot_weight=args.rot_weight, ik_tol=args.ik_tol)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.output, **out)
    print(f"[ok] wrote {args.output} (frames={out['joint_pos'].shape[0]})")


if __name__ == "__main__":
    main()
