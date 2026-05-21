"""VAE latent z capture + PCA plot for play.py (decoder forward hook only)."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from typing import Any, Literal

import numpy as np
import torch

LEGEND_TOP_K = 15
_COLOR_PALETTE = (
    (0.1215686275, 0.4666666667, 0.7058823529),
    (1.0, 0.4980392157, 0.0549019608),
    (0.1725490196, 0.6274509804, 0.1725490196),
    (0.8392156863, 0.1529411765, 0.1568627451),
    (0.5803921569, 0.4039215686, 0.7411764706),
    (0.5490196078, 0.3372549020, 0.2941176471),
    (0.8901960784, 0.4666666667, 0.7607843137),
    (0.4980392157, 0.4980392157, 0.4980392157),
    (0.7372549020, 0.7411764706, 0.1333333333),
    (0.0901960784, 0.7450980392, 0.8117647059),
    (0.6823529412, 0.7803921569, 0.9098039216),
    (1.0, 0.7333333333, 0.4705882353),
    (0.5960784314, 0.8745098039, 0.5411764706),
    (1.0, 0.5960784314, 0.5882352941),
    (0.7725490196, 0.6901960784, 0.8352941176),
    (0.7686274510, 0.6117647059, 0.5803921569),
    (0.9686274510, 0.7137254902, 0.8235294118),
    (0.7803921569, 0.7803921569, 0.7803921569),
    (0.8588235294, 0.8588235294, 0.5529411765),
    (0.6196078431, 0.8549019608, 0.8980392157),
)

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


def _require_vae_actor(actor_critic: Any) -> Any:
    """Return MlpVae actor or raise if the checkpoint is not a VAE policy."""
    actor = getattr(actor_critic, "actor", None)
    if actor is None or not hasattr(actor, "decoder") or not hasattr(actor, "latent_size"):
        raise RuntimeError(
            "[latent_viz] --decoder_random_z requires a VAE actor_critic with actor.decoder and actor.latent_size."
        )
    return actor


def decode_with_random_z(actor_critic: Any, observations: torch.Tensor, *, std: float = 1.0) -> torch.Tensor:
    """Decode actions from z ~ N(0, std^2 I) without running the VAE encoder (play diagnostic)."""
    vae = _require_vae_actor(actor_critic)
    gather = getattr(actor_critic, "_gather_vae_inputs", None)
    if gather is None or not callable(gather):
        raise RuntimeError(
            "[latent_viz] --decoder_random_z requires VaeActorCritic._gather_vae_inputs (not a VAE policy?)."
        )

    if hasattr(actor_critic, "encoders"):
        observations = actor_critic.encoders(observations)

    ctx = gather(observations)
    batch = observations.shape[0]
    z = torch.randn(batch, int(vae.latent_size), device=observations.device, dtype=observations.dtype) * std

    prior_mean = None
    if vae.decode_add_prior_mean and vae.prior_net is not None and ctx.get("prior_cond") is not None:
        prior_mean, _ = vae.prior_net(ctx["prior_cond"]).chunk(2, dim=-1)

    return vae.decode_latent(z, decoder_aux_input=ctx.get("decoder_aux_input"), prior_mean=prior_mean)


def make_decoder_random_z_policy(
    actor_critic: Any,
    normalizer: Any | None = None,
    *,
    std: float = 1.0,
):
    """Callable policy for play.py: normalize obs, then decode_with_random_z each step."""

    def _policy(observations: torch.Tensor) -> torch.Tensor:
        x = normalizer(observations) if normalizer is not None else observations
        return decode_with_random_z(actor_critic, x, std=std)

    return _policy


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


def _rgba_to_hex(color: tuple[float, ...]) -> str:
    r = max(0, min(255, int(round(float(color[0]) * 255.0))))
    g = max(0, min(255, int(round(float(color[1]) * 255.0))))
    b = max(0, min(255, int(round(float(color[2]) * 255.0))))
    return f"#{r:02x}{g:02x}{b:02x}"


def _write_interactive_sphere_html(
    out_path: str,
    *,
    sphere3: np.ndarray,
    motions: np.ndarray,
    steps: np.ndarray,
    envs: np.ndarray,
    top: list[str],
    color_for: dict[str, Any],
    gray: tuple[float, float, float],
    title: str,
) -> None:
    legend_names = set(top)
    gray_hex = _rgba_to_hex(gray)
    points: list[dict[str, Any]] = []
    for i in range(int(sphere3.shape[0])):
        motion_name = str(motions[i])
        color = color_for.get(motion_name)
        if motion_name in legend_names and color is not None:
            color_hex = _rgba_to_hex(tuple(float(c) for c in color[:3]))
            group = motion_name
        else:
            color_hex = gray_hex
            group = "other"
        points.append(
            {
                "x": float(sphere3[i, 0]),
                "y": float(sphere3[i, 1]),
                "z": float(sphere3[i, 2]),
                "step": int(steps[i]),
                "env_idx": int(envs[i]),
                "motion_id": motion_name,
                "group": group,
                "color": color_hex,
            }
        )

    legend_items = [{"name": name, "color": _rgba_to_hex(tuple(float(c) for c in color_for[name][:3]))} for name in top]
    if any(p["group"] == "other" for p in points):
        legend_items.append({"name": "other", "color": gray_hex})

    payload = {"title": title, "points": points, "legend": legend_items}
    payload_json = json.dumps(payload, ensure_ascii=True)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>latent_viz sphere</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f1116;
      --panel: #181c23;
      --text: #f2f5fa;
      --muted: #9aa6b2;
      --border: #2b3440;
    }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .wrap {{
      display: flex;
      flex-direction: row;
      gap: 16px;
      padding: 16px;
      box-sizing: border-box;
      min-height: 100vh;
    }}
    .viewer {{
      flex: 1 1 auto;
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      box-sizing: border-box;
    }}
    .title {{
      font-size: 16px;
      font-weight: 600;
      margin: 0 0 10px 0;
    }}
    canvas {{
      width: 100%;
      height: min(74vh, 900px);
      display: block;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #0a0d12;
      cursor: grab;
    }}
    canvas.dragging {{
      cursor: grabbing;
    }}
    .hint {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }}
    .side {{
      width: 280px;
      flex: 0 0 280px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      box-sizing: border-box;
      max-height: calc(100vh - 32px);
      overflow: auto;
    }}
    .section-title {{
      margin: 0 0 10px 0;
      font-weight: 600;
      font-size: 14px;
    }}
    .legend-controls {{
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .control-btn {{
      border: 1px solid var(--border);
      background: #11161e;
      color: var(--text);
      border-radius: 6px;
      font-size: 12px;
      padding: 4px 8px;
      cursor: pointer;
    }}
    .control-btn:hover {{
      background: #1b2330;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 6px;
      font-size: 12px;
      color: var(--text);
    }}
    .legend-item input[type="checkbox"] {{
      margin: 0;
      accent-color: #6aa6ff;
      cursor: pointer;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex: 0 0 10px;
      border: 1px solid #ffffff55;
    }}
    .legend-name {{
      flex: 1 1 auto;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .legend-count {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .tooltip {{
      margin-top: 14px;
      font-size: 12px;
      line-height: 1.45;
      color: var(--muted);
      white-space: pre-wrap;
      border-top: 1px solid var(--border);
      padding-top: 10px;
    }}
    @media (max-width: 980px) {{
      .wrap {{
        flex-direction: column;
      }}
      .side {{
        width: auto;
        flex: 1 1 auto;
      }}
      canvas {{
        height: min(62vh, 720px);
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="viewer">
      <h1 class="title" id="title"></h1>
      <canvas id="sphereCanvas"></canvas>
      <div class="hint">Drag to rotate, mouse wheel to zoom, hover points to inspect metadata.</div>
    </div>
    <aside class="side">
      <h2 class="section-title">Legend (Top Motions)</h2>
      <div class="legend-controls">
        <button class="control-btn" id="showAllBtn" type="button">Show all</button>
        <button class="control-btn" id="hideAllBtn" type="button">Hide all</button>
      </div>
      <div id="legend"></div>
      <div class="tooltip" id="tooltip">Hover a point to view details.</div>
    </aside>
  </div>
  <script>
    "use strict";
    const payload = {payload_json};
    const canvas = document.getElementById("sphereCanvas");
    const ctx = canvas.getContext("2d");
    const titleNode = document.getElementById("title");
    const legendNode = document.getElementById("legend");
    const tooltipNode = document.getElementById("tooltip");
    const showAllBtn = document.getElementById("showAllBtn");
    const hideAllBtn = document.getElementById("hideAllBtn");
    titleNode.textContent = payload.title;

    const groupCounts = Object.create(null);
    for (const p of payload.points) {{
      const key = p.group;
      groupCounts[key] = (groupCounts[key] || 0) + 1;
    }}

    const state = {{
      yaw: 0.8,
      pitch: 0.35,
      zoom: 1.0,
      dragging: false,
      lastX: 0,
      lastY: 0,
      projected: [],
      visibleGroups: new Set(payload.legend.map((item) => item.name)),
    }};

    function renderLegend() {{
      legendNode.replaceChildren();
      for (const item of payload.legend) {{
        const row = document.createElement("label");
        row.className = "legend-item";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = state.visibleGroups.has(item.name);
        checkbox.addEventListener("change", () => {{
          if (checkbox.checked) {{
            state.visibleGroups.add(item.name);
          }} else {{
            state.visibleGroups.delete(item.name);
          }}
          draw();
        }});
        const dot = document.createElement("span");
        dot.className = "dot";
        dot.style.background = item.color;
        const text = document.createElement("span");
        text.className = "legend-name";
        text.textContent = item.name;
        text.title = item.name;
        const count = document.createElement("span");
        count.className = "legend-count";
        count.textContent = "(" + String(groupCounts[item.name] || 0) + ")";
        row.appendChild(checkbox);
        row.appendChild(dot);
        row.appendChild(text);
        row.appendChild(count);
        legendNode.appendChild(row);
      }}
    }}

    function setAllGroupsVisible(visible) {{
      state.visibleGroups.clear();
      if (visible) {{
        for (const item of payload.legend) {{
          state.visibleGroups.add(item.name);
        }}
      }}
      renderLegend();
      draw();
    }}

    function resizeCanvas() {{
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function rotatePoint(p) {{
      const cy = Math.cos(state.yaw);
      const sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch);
      const sp = Math.sin(state.pitch);
      const x1 = cy * p.x + sy * p.z;
      const z1 = -sy * p.x + cy * p.z;
      const y2 = cp * p.y - sp * z1;
      const z2 = sp * p.y + cp * z1;
      return {{ x: x1, y: y2, z: z2 }};
    }}

    function projectPoint(p3, width, height) {{
      const cameraZ = 3.2;
      const persp = cameraZ / (cameraZ - p3.z);
      const scale = Math.min(width, height) * 0.36 * state.zoom;
      return {{
        x: width * 0.5 + p3.x * scale * persp,
        y: height * 0.5 - p3.y * scale * persp,
        z: p3.z,
        size: 2.0 + 2.4 * (0.3 + Math.max(0, p3.z + 1.0) * 0.5),
      }};
    }}

    function drawWireframe(width, height) {{
      const rings = 10;
      const slices = 20;
      ctx.strokeStyle = "#8da0b333";
      ctx.lineWidth = 1;

      for (let ri = 1; ri < rings; ri += 1) {{
        const phi = (Math.PI * ri) / rings;
        ctx.beginPath();
        let first = true;
        for (let si = 0; si <= slices; si += 1) {{
          const theta = (2 * Math.PI * si) / slices;
          const p = {{
            x: Math.cos(theta) * Math.sin(phi),
            y: Math.cos(phi),
            z: Math.sin(theta) * Math.sin(phi),
          }};
          const pr = projectPoint(rotatePoint(p), width, height);
          if (first) {{
            ctx.moveTo(pr.x, pr.y);
            first = false;
          }} else {{
            ctx.lineTo(pr.x, pr.y);
          }}
        }}
        ctx.stroke();
      }}

      for (let si = 0; si < slices; si += 2) {{
        const theta = (2 * Math.PI * si) / slices;
        ctx.beginPath();
        let first = true;
        for (let ri = 0; ri <= rings; ri += 1) {{
          const phi = (Math.PI * ri) / rings;
          const p = {{
            x: Math.cos(theta) * Math.sin(phi),
            y: Math.cos(phi),
            z: Math.sin(theta) * Math.sin(phi),
          }};
          const pr = projectPoint(rotatePoint(p), width, height);
          if (first) {{
            ctx.moveTo(pr.x, pr.y);
            first = false;
          }} else {{
            ctx.lineTo(pr.x, pr.y);
          }}
        }}
        ctx.stroke();
      }}
    }}

    function draw() {{
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);
      drawWireframe(width, height);

      const visiblePoints = payload.points.filter((p) => state.visibleGroups.has(p.group));
      state.projected = visiblePoints.map((p, i) => {{
        const rp = rotatePoint(p);
        const pp = projectPoint(rp, width, height);
        return {{ idx: i, p, screen: pp }};
      }});
      state.projected.sort((a, b) => a.screen.z - b.screen.z);

      for (const item of state.projected) {{
        ctx.beginPath();
        ctx.arc(item.screen.x, item.screen.y, item.screen.size, 0, 2 * Math.PI);
        ctx.fillStyle = item.p.color;
        ctx.globalAlpha = 0.93;
        ctx.fill();
      }}
      ctx.globalAlpha = 1;
    }}

    function updateHover(clientX, clientY) {{
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      let best = null;
      let bestDist2 = 64.0;
      for (const item of state.projected) {{
        const dx = item.screen.x - x;
        const dy = item.screen.y - y;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist2) {{
          bestDist2 = d2;
          best = item;
        }}
      }}
      if (best) {{
        const p = best.p;
        tooltipNode.textContent =
          "motion_id: " + p.motion_id + "\\n" +
          "group: " + p.group + "\\n" +
          "step: " + p.step + " | env_idx: " + p.env_idx + "\\n" +
          "xyz: (" + p.x.toFixed(4) + ", " + p.y.toFixed(4) + ", " + p.z.toFixed(4) + ")";
      }} else {{
        tooltipNode.textContent = "Hover a point to view details.";
      }}
    }}

    canvas.addEventListener("mousedown", (ev) => {{
      state.dragging = true;
      state.lastX = ev.clientX;
      state.lastY = ev.clientY;
      canvas.classList.add("dragging");
    }});

    window.addEventListener("mouseup", () => {{
      state.dragging = false;
      canvas.classList.remove("dragging");
    }});

    canvas.addEventListener("mousemove", (ev) => {{
      if (state.dragging) {{
        const dx = ev.clientX - state.lastX;
        const dy = ev.clientY - state.lastY;
        state.lastX = ev.clientX;
        state.lastY = ev.clientY;
        state.yaw += dx * 0.008;
        state.pitch += dy * 0.008;
        const lim = 1.4;
        if (state.pitch > lim) state.pitch = lim;
        if (state.pitch < -lim) state.pitch = -lim;
        draw();
      }}
      updateHover(ev.clientX, ev.clientY);
    }});

    canvas.addEventListener("wheel", (ev) => {{
      ev.preventDefault();
      const factor = ev.deltaY < 0 ? 1.08 : 0.92;
      state.zoom *= factor;
      if (state.zoom < 0.45) state.zoom = 0.45;
      if (state.zoom > 2.8) state.zoom = 2.8;
      draw();
    }}, {{ passive: false }});

    showAllBtn.addEventListener("click", () => setAllGroupsVisible(true));
    hideAllBtn.addEventListener("click", () => setAllGroupsVisible(false));

    window.addEventListener("resize", resizeCanvas);
    renderLegend();
    resizeCanvas();
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


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
    """Writes z_raw.npy; z_pca_points.csv, z_pca_by_motion.png; z_sphere_points.csv, z_sphere_by_motion.png/html.

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

    counts = Counter(str(m) for m in motions)
    top = [m for m, _ in counts.most_common(LEGEND_TOP_K)]
    color_for = {name: _COLOR_PALETTE[i % len(_COLOR_PALETTE)] for i, name in enumerate(top)}
    gray = (0.45,) * 3
    html_sphere = os.path.join(out_dir, "z_sphere_by_motion.html")
    _write_interactive_sphere_html(
        html_sphere,
        sphere3=sphere3,
        motions=motions,
        steps=steps,
        envs=envs,
        top=top,
        color_for=color_for,
        gray=gray,
        title=f"VAE z on sphere (N={sphere3.shape[0]}, z_dim={z_np.shape[1]}, mode={project_mode})",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as e:
        print(
            f"[latent_viz] matplotlib missing; wrote numpy/csv/html: {z_raw_path}, {csv_path}, {sphere_csv_path}, {html_sphere}. ({e})"
        )
        return

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
        f"[latent_viz] Wrote {z_raw_path}, {csv_path}, {png}, {sphere_csv_path}, {png_sphere}, {html_sphere}"
    )
