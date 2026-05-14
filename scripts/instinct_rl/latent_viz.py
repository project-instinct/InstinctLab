"""VAE latent z capture + PCA plot for play.py (decoder forward hook only)."""

from __future__ import annotations

import csv
import os
from collections import Counter
from typing import Any, Literal

import numpy as np
import torch

LEGEND_TOP_K = 15

ProjectMode = Literal["pca", "first3"]

# Emit at most one warning when motion identifiers cannot be resolved (avoids silent all-unknown).
_MOTION_ID_FALLBACK_WARNED = False


def try_install_vae_decoder_z_hook(actor_critic: Any) -> tuple[Any | None, list[torch.Tensor | None]]:
    storage: list[torch.Tensor | None] = [None]
    actor = getattr(actor_critic, "actor", None)
    if actor is None or not hasattr(actor, "decoder") or not hasattr(actor, "latent_size"):
        return None, storage

    latent_size = int(actor.latent_size)
    decoder = actor.decoder

    def _hook(_mod, inp, _out):
        if not inp or not isinstance(inp[0], torch.Tensor):
            return
        z = inp[0]
        if z.shape[-1] < latent_size:
            return
        storage[0] = z[..., :latent_size].detach().cpu()

    return decoder.register_forward_hook(_hook), storage


def remove_hook(handle: Any | None) -> None:
    if handle is not None:
        handle.remove()


def warn_if_not_project_to_sphere(actor_critic: Any) -> None:
    actor = getattr(actor_critic, "actor", None)
    if actor is not None and hasattr(actor, "project_to_sphere") and not bool(actor.project_to_sphere):
        print("\033[93m[latent_viz] project_to_sphere is False; z may not be unit sphere.\033[0m")


def _warn_motion_id_fallback_once(message: str) -> None:
    global _MOTION_ID_FALLBACK_WARNED
    if not _MOTION_ID_FALLBACK_WARNED:
        print(f"\033[93m[latent_viz] {message}\033[0m")
        _MOTION_ID_FALLBACK_WARNED = True


def _short_motion_label(raw: Any) -> str:
    """Normalize motion identifiers for CSV/plots.

    Many datasets reuse the same *filename* under different parent folders (see metadata.yaml:
    ``climb_00_z_scale_0.8/motion1....npz`` vs ``climb_00_z_scale_0.9/...``). Using basename only
    would collapse them to one label; we keep ``parent_dir/filename`` when a path has two+ segments.
    """
    s = str(raw).strip()
    if not s:
        return "empty_motion_id"
    norm = s.replace("\\", "/").rstrip("/")
    looks_like_path = "/" in norm or norm.lower().endswith((".npz", ".npy", ".pkl", ".yaml", ".yml"))
    if not looks_like_path:
        return s
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return s
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def _legend_motion_label(name: str) -> str:
    """Matplotlib legend text: do not strip ``parent/file`` back to basename."""
    s = str(name)
    if "/" in s or "\\" in s:
        return s
    return os.path.basename(s) if s else s


def get_motion_identifiers_for_envs(base_env: Any, num_envs: int) -> list[str]:
    """Resolve per-env motion ids from scene motion_reference (see MotionReferenceManager).

    Must pass env indices as a ``torch`` tensor on the manager device: a Python ``list`` triggers
    type errors inside ``get_current_motion_identifiers`` and previously produced all-unknown labels.
    """
    n = int(num_envs)
    fallback = [f"unknown_motion_e{i}" for i in range(n)]

    scene = getattr(base_env, "scene", None)
    if scene is None:
        _warn_motion_id_fallback_once("get_motion_identifiers_for_envs: base_env has no .scene; using unknown_motion_*.")
        return fallback

    try:
        motion_ref = scene["motion_reference"]
    except Exception as exc:
        _warn_motion_id_fallback_once(
            f"get_motion_identifiers_for_envs: scene['motion_reference'] missing or error: {exc}"
        )
        return fallback

    get_ids = getattr(motion_ref, "get_current_motion_identifiers", None)
    if get_ids is None or not callable(get_ids):
        _warn_motion_id_fallback_once(
            "get_motion_identifiers_for_envs: motion_reference has no get_current_motion_identifiers; using fallback."
        )
        return fallback

    device = getattr(motion_ref, "device", None) or torch.device("cpu")
    try:
        env_ids_t = torch.arange(n, device=device, dtype=torch.long)
        mids = get_ids(env_ids_t)
    except Exception as exc:
        _warn_motion_id_fallback_once(
            f"get_motion_identifiers_for_envs: get_current_motion_identifiers failed ({type(exc).__name__}: {exc})."
        )
        return fallback

    if not isinstance(mids, list) or len(mids) != n:
        _warn_motion_id_fallback_once(
            "get_motion_identifiers_for_envs: expected "
            f"list of length {n}, got {type(mids).__name__} "
            f"with len {len(mids) if hasattr(mids, '__len__') else 'n/a'}; using fallback."
        )
        return fallback

    return [_short_motion_label(m) for m in mids]


def _pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    if n <= 1:
        return torch.zeros(n, 2, dtype=x.dtype)
    _u, _s, v = torch.pca_lowrank(x, q=2, center=False)
    return x @ v[:, :2]


def _project_to_3d(z_np: np.ndarray, mode: ProjectMode | str) -> np.ndarray:
    """Reduce latent rows to 3D then L2-normalize each row to the unit sphere (S^2).

    Mirrors `/home/xufurui/Desktop/Projects/InstinctLab/scripts/instinct_rl/play_dagger.py`.
    """
    if z_np.shape[1] == 3:
        z3 = z_np
    elif z_np.shape[1] < 3:
        pad = np.zeros((z_np.shape[0], 3 - z_np.shape[1]), dtype=z_np.dtype)
        z3 = np.concatenate([z_np, pad], axis=1)
    elif mode == "first3":
        z3 = z_np[:, :3]
    else:
        centered = z_np - np.mean(z_np, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:3].T
        z3 = centered @ basis
    norms = np.linalg.norm(z3, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return z3 / norms


def _project_to_3d_with_fixed_pca(z_np: np.ndarray, basis_path: str | None) -> np.ndarray:
    """Project to 3D using a reusable PCA basis for cross-run / cross-episode consistency."""
    if z_np.shape[1] <= 3:
        return _project_to_3d(z_np, mode="first3")
    if basis_path is None:
        return _project_to_3d(z_np, mode="pca")

    basis_abspath = os.path.abspath(basis_path)
    basis_dir = os.path.dirname(basis_abspath)
    if basis_dir:
        os.makedirs(basis_dir, exist_ok=True)

    if os.path.isfile(basis_abspath):
        data = np.load(basis_abspath)
        mean = data["mean"].astype(np.float32)
        basis = data["basis"].astype(np.float32)
        if mean.ndim != 1 or basis.shape != (z_np.shape[1], 3):
            raise ValueError(
                f"[latent_viz] Invalid PCA basis file: expected mean=({z_np.shape[1]},), basis=({z_np.shape[1]},3), "
                f"got mean={mean.shape}, basis={basis.shape}"
            )
        centered = z_np.astype(np.float32, copy=False) - mean[None, :]
        z3 = centered @ basis
        print(f"[latent_viz] Loaded fixed PCA basis from: {basis_abspath}")
    else:
        centered = z_np - np.mean(z_np, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:3].T.astype(np.float32)
        mean = np.mean(z_np, axis=0).astype(np.float32)
        np.savez(basis_abspath, mean=mean, basis=basis)
        z3 = (z_np.astype(np.float32, copy=False) - mean[None, :]) @ basis
        print(f"[latent_viz] Computed and saved fixed PCA basis to: {basis_abspath}")

    norms = np.linalg.norm(z3, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return z3 / norms


def _add_unit_sphere_wireframe(ax: Any, u_segments: int = 40, v_segments: int = 20) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, u_segments)
    v = np.linspace(0.0, np.pi, v_segments)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.4, alpha=0.25)


def save_latent_visualization(
    z_rows: list[torch.Tensor],
    motion_labels: list[str],
    step_indices: list[int],
    env_indices: list[int],
    out_dir: str,
    *,
    project_mode: ProjectMode | str = "pca",
    pca_basis_path: str | None = None,
) -> None:
    """Writes z_raw.npy; z_pca_points.csv, z_pca_by_motion.png; z_sphere_points.csv, z_sphere_by_motion.png.

    Sphere coordinates follow `play_dagger.py`: NumPy SVD PCA (or first3), then row-wise normalization to S^2.
    When ``project_mode=="pca"``, a fixed basis file under ``out_dir`` is used unless ``pca_basis_path`` is set.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not z_rows:
        print("[latent_viz] No samples; skip save.")
        return

    if project_mode not in ("pca", "first3"):
        raise ValueError(f"[latent_viz] project_mode must be 'pca' or 'first3', got {project_mode!r}")

    z = torch.cat(z_rows, dim=0).float()
    z_np = z.detach().cpu().numpy().astype(np.float32)
    motions = np.asarray(motion_labels, dtype=object)
    steps = np.asarray(step_indices, dtype=np.int64)
    envs = np.asarray(env_indices, dtype=np.int64)

    z_raw_path = os.path.join(out_dir, "z_raw.npy")
    np.save(z_raw_path, z_np)
    print(f"[latent_viz] Wrote raw latents: {z_raw_path} shape={z_np.shape}")

    pcs = _pca_2d(z).numpy()

    if project_mode == "pca":
        basis_path = pca_basis_path if pca_basis_path is not None else os.path.join(out_dir, "z_sphere_pca_basis.npz")
        sphere3 = _project_to_3d_with_fixed_pca(z_np, basis_path)
    else:
        sphere3 = _project_to_3d(z_np, mode="first3")

    csv_path = os.path.join(out_dir, "z_pca_points.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "env_idx", "motion_id", "pca_1", "pca_2"])
        for i in range(len(pcs)):
            w.writerow([int(steps[i]), int(envs[i]), str(motions[i]), float(pcs[i, 0]), float(pcs[i, 1])])

    sphere_csv_path = os.path.join(out_dir, "z_sphere_points.csv")
    with open(sphere_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "env_idx", "motion_id", "x", "y", "z"])
        for i in range(len(sphere3)):
            w.writerow(
                [
                    int(steps[i]),
                    int(envs[i]),
                    str(motions[i]),
                    float(sphere3[i, 0]),
                    float(sphere3[i, 1]),
                    float(sphere3[i, 2]),
                ]
            )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as e:
        print(
            f"[latent_viz] matplotlib missing; wrote numpy/csv only: {z_raw_path}, {csv_path}, {sphere_csv_path}. ({e})"
        )
        return

    counts = Counter(str(m) for m in motions)
    top = [m for m, _ in counts.most_common(LEGEND_TOP_K)]
    cmap = plt.get_cmap("tab20")
    color_for = {name: cmap(i % 20) for i, name in enumerate(top)}
    gray = (0.45,) * 3

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in top:
        mask = motions == name
        if mask.any():
            c = tuple(float(x) for x in color_for[name][:3])
            lab = _legend_motion_label(name)
            lab = lab[:45] + "..." if len(lab) > 48 else lab
            ax.scatter(pcs[mask, 0], pcs[mask, 1], color=c, alpha=0.35, s=10, label=lab)

    other = np.array([str(motions[i]) not in color_for for i in range(len(motions))])
    if other.any():
        ax.scatter(pcs[other, 0], pcs[other, 1], color=gray, alpha=0.35, s=10, label="other")

    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    png = os.path.join(out_dir, "z_pca_by_motion.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig3 = plt.figure(figsize=(9, 8))
    ax3 = fig3.add_subplot(111, projection="3d")
    _add_unit_sphere_wireframe(ax3)

    for name in top:
        mask = motions == name
        if mask.any():
            c = tuple(float(x) for x in color_for[name][:3])
            lab = _legend_motion_label(name)
            lab = lab[:45] + "..." if len(lab) > 48 else lab
            ax3.scatter(
                sphere3[mask, 0],
                sphere3[mask, 1],
                sphere3[mask, 2],
                color=c,
                alpha=0.9,
                s=12,
                label=lab,
            )

    other3 = np.array([str(motions[i]) not in color_for for i in range(len(motions))])
    if other3.any():
        ax3.scatter(
            sphere3[other3, 0],
            sphere3[other3, 1],
            sphere3[other3, 2],
            color=gray,
            alpha=0.9,
            s=12,
            label="other",
        )

    title = f"VAE z on sphere (N={sphere3.shape[0]}, z_dim={z_np.shape[1]}, mode={project_mode})"
    ax3.set_title(title)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend(loc="upper right", fontsize=7)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlim((-1.05, 1.05))
    ax3.set_ylim((-1.05, 1.05))
    ax3.set_zlim((-1.05, 1.05))
    plt.tight_layout()
    png_sphere = os.path.join(out_dir, "z_sphere_by_motion.png")
    fig3.savefig(png_sphere, dpi=220, bbox_inches="tight")
    plt.close(fig3)
    print(
        f"[latent_viz] Wrote {z_raw_path}, {csv_path}, {png}, {sphere_csv_path}, {png_sphere}"
    )
