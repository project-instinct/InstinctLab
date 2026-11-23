"""Directional trimesh procedural terrains for ACP Stage-3 goal-reaching fine-tuning.

Every terrain is a clean ``+x`` directional course on a ``(14.0, 12.0)`` tile:
- Start platform on the ``-x`` side; returned ``origin`` sits on the start platform (``x_local = -2.2`` m).
- Obstacle zone in the middle (``x_local in [-1.5, +1.5]`` m), scaled across 10 difficulty levels.
- Far-side landing platform on the ``+x`` side (``x_local >= +1.5`` m), where ``"target"`` flat patches
  are sampled strictly beyond the obstacle.

Covers the ACP paper's four Stage-3 skill groups (L1013–1043), including **1.0 m high platforms** in
``box_ascent``, ``box_descent``, and ``box_up_down`` for whole-body arms+legs climbing:
- Dense foot: ``stairs_up``, ``stairs_down``, ``walled_stairs``, ``slope_rough``
- Dense whole-body: ``box_ascent`` (up to 1.0 m), ``box_descent`` (down from 1.0 m), ``box_up_down`` (over 1.0 m)
- Sparse foot: ``gaps``, ``double_stones``, ``balance_beam_pallets``

All generators use exact ``trimesh.creation.box`` primitives (~10–80 triangles/tile), keeping a
``202 x 12`` mixed scene well under 250k triangles total for fast PhysX BV4 cooking and flat-patch sampling.
"""

from __future__ import annotations

import os
from dataclasses import MISSING
import numpy as np
import trimesh
import yaml

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from instinctlab.terrains.trimesh.utils import crop_terrain_mesh_aabb


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> trimesh.Trimesh:
    """Axis-aligned box spanning ``[x0, x1] x [y0, y1] x [z0, z1]`` in tile-corner coords ``[0, Lx] x [0, Ly]``."""
    extents = [max(x1 - x0, 1e-3), max(y1 - y0, 1e-3), max(z1 - z0, 1e-3)]
    center = [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)]
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def _lerp(rng: tuple[float, float], difficulty: float) -> float:
    return float(rng[0] + np.clip(difficulty, 0.0, 1.0) * (rng[1] - rng[0]))


# -- 1 & 2 & 3: Directional Stairs (up, down, walled) -------------------------------------------


def directional_stairs_terrain(
    difficulty: float, cfg: DirectionalStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    lx, ly = cfg.size
    cx, cy = lx / 2.0, ly / 2.0
    step_h = _lerp(cfg.step_height_range, difficulty)
    step_w = cfg.step_width
    n_steps = cfg.num_steps
    total_h = n_steps * step_h
    slab = 0.4
    half_w = 0.5 * cfg.corridor_width if cfg.walled else ly / 2.0
    y0, y1 = cy - half_w, cy + half_w

    meshes: list[trimesh.Trimesh] = []
    # Base floor outside/underneath
    meshes.append(_box(0.0, lx, 0.0, ly, -slab, 0.0))

    x_start_end = cx - 1.5
    if not cfg.descending:
        # Ascending in +x: start at z=0, climb n_steps, top landing at z=total_h
        for i in range(n_steps):
            sx0 = x_start_end + i * step_w
            sx1 = sx0 + step_w
            h = (i + 1) * step_h
            meshes.append(_box(sx0, sx1, y0, y1, 0.0, h))
        top_x0 = x_start_end + n_steps * step_w
        meshes.append(_box(top_x0, lx, y0, y1, 0.0, total_h))
        z_origin = 0.0
    else:
        # Descending in +x: start platform at z=total_h, step down to z=0
        meshes.append(_box(0.0, x_start_end, y0, y1, 0.0, total_h))
        for i in range(n_steps):
            sx0 = x_start_end + i * step_w
            sx1 = sx0 + step_w
            h = (n_steps - 1 - i) * step_h
            if h > 1e-3:
                meshes.append(_box(sx0, sx1, y0, y1, 0.0, h))
        z_origin = total_h

    if cfg.walled:
        wall_t = 0.25
        wall_h = total_h + cfg.wall_height
        meshes.append(_box(0.0, lx, max(0.0, y0 - wall_t), y0, 0.0, wall_h))
        meshes.append(_box(0.0, lx, y1, min(ly, y1 + wall_t), 0.0, wall_h))

    origin = np.array([cx - 2.2, cy, z_origin], dtype=np.float64)
    return meshes, origin


@configclass
class DirectionalStairsTerrainCfg(SubTerrainBaseCfg):
    function = directional_stairs_terrain
    terrain_name: str = "stairs_up"
    step_height_range: tuple[float, float] = (0.05, 0.25)
    step_width: float = 0.35
    num_steps: int = 7
    descending: bool = False
    walled: bool = False
    corridor_width: float = 2.0
    wall_height: float = 1.5


# -- 4: Directional Slope + Roughness -----------------------------------------------------------


def directional_slope_terrain(
    difficulty: float, cfg: DirectionalSlopeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    lx, ly = cfg.size
    cx, cy = lx / 2.0, ly / 2.0
    slope = _lerp(cfg.slope_range, difficulty)
    ramp_len = 3.2
    total_h = ramp_len * np.tan(slope)
    slab = 0.4
    n_slices = 12
    dx = ramp_len / n_slices

    meshes: list[trimesh.Trimesh] = [_box(0.0, lx, 0.0, ly, -slab, 0.0)]
    x0 = cx - 1.5
    rng = np.random.default_rng(int(round(difficulty * 10000)) + 17)
    rough = _lerp(cfg.roughness_range, difficulty)

    for i in range(n_slices):
        sx0 = x0 + i * dx
        sx1 = sx0 + dx
        h = (i + 0.5) / n_slices * total_h + float(rng.uniform(-rough, rough))
        if h > 1e-3:
            meshes.append(_box(sx0, sx1, 0.0, ly, 0.0, h))
    meshes.append(_box(x0 + ramp_len, lx, 0.0, ly, 0.0, max(total_h, 1e-3)))

    origin = np.array([cx - 2.2, cy, 0.0], dtype=np.float64)
    return meshes, origin


@configclass
class DirectionalSlopeTerrainCfg(SubTerrainBaseCfg):
    function = directional_slope_terrain
    terrain_name: str = "slope_rough"
    slope_range: tuple[float, float] = (0.04, 0.35)
    roughness_range: tuple[float, float] = (0.0, 0.04)


# -- 5, 6, 7: High-Platform Box Climbing (up to 1.0 m! ascent, descent, up-and-down) -----------


def high_platform_terrain(
    difficulty: float, cfg: HighPlatformTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Single-step / high-platform obstacle scaling up to 1.0 m for arms+legs whole-body climbing."""
    lx, ly = cfg.size
    cx, cy = lx / 2.0, ly / 2.0
    height = _lerp(cfg.height_range, difficulty)
    slab = 0.4

    meshes: list[trimesh.Trimesh] = [_box(0.0, lx, 0.0, ly, -slab, 0.0)]
    if cfg.mode == "ascent":
        # Start on ground (z=0), climb UP onto high platform (z=height) starting at x_local = -0.8
        meshes.append(_box(cx - 0.8, lx, 0.0, ly, 0.0, height))
        z_origin = 0.0
    elif cfg.mode == "descent":
        # Start on high platform (z=height), climb/drop DOWN to ground (z=0) at x_local = -0.8
        meshes.append(_box(0.0, cx - 0.8, 0.0, ly, 0.0, height))
        z_origin = height
    elif cfg.mode == "up_down":
        # Start at z=0, climb UP onto central box (z=height, length=box_length), then DOWN to z=0
        bx0 = cx - 0.9
        bx1 = bx0 + cfg.box_length
        meshes.append(_box(bx0, bx1, 0.0, ly, 0.0, height))
        z_origin = 0.0
    else:
        raise ValueError(f"Unknown HighPlatformTerrainCfg mode: {cfg.mode!r}")

    origin = np.array([cx - 2.2, cy, z_origin], dtype=np.float64)
    return meshes, origin


@configclass
class HighPlatformTerrainCfg(SubTerrainBaseCfg):
    function = high_platform_terrain
    terrain_name: str = "box_ascent"
    height_range: tuple[float, float] = (0.15, 1.00)
    """Platform height range across levels 0..9; reaches 1.0 m at level 9 for whole-body climbing."""
    mode: str = "ascent"
    """``"ascent"`` (climb up), ``"descent"`` (climb down), or ``"up_down"`` (climb over box)."""
    box_length: float = 2.0


# -- 8: Directional Gap -------------------------------------------------------------------------


def directional_gap_terrain(
    difficulty: float, cfg: DirectionalGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    lx, ly = cfg.size
    cx, cy = lx / 2.0, ly / 2.0
    gap_w = _lerp(cfg.gap_width_range, difficulty)
    depth = cfg.gap_depth
    slab = 0.4

    x_gap0 = cx - 0.6
    x_gap1 = x_gap0 + gap_w
    meshes: list[trimesh.Trimesh] = [
        _box(0.0, lx, 0.0, ly, -depth - slab, -depth),  # pit floor
        _box(0.0, x_gap0, 0.0, ly, -depth, 0.0),  # start platform
        _box(x_gap1, lx, 0.0, ly, -depth, 0.0),  # far-side landing
    ]
    origin = np.array([cx - 2.2, cy, 0.0], dtype=np.float64)
    return meshes, origin


@configclass
class DirectionalGapTerrainCfg(SubTerrainBaseCfg):
    function = directional_gap_terrain
    terrain_name: str = "gaps"
    gap_width_range: tuple[float, float] = (0.15, 0.70)
    gap_depth: float = 1.0


# -- 9: Double Stepping Stones (sparse foot, exact trimesh) -------------------------------------


def double_stones_terrain(
    difficulty: float, cfg: DoubleStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Paired stepping-stone supports spanning a pit (ACP paper L1028)."""
    lx, ly = cfg.size
    cx, cy = lx / 2.0, ly / 2.0
    gap = _lerp(cfg.stone_gap_range, difficulty)
    stone_l = _lerp(cfg.stone_size_range, 1.0 - difficulty)  # smaller at higher difficulty
    stone_w = stone_l
    depth = cfg.pit_depth
    slab = 0.4
    lat_offset = cfg.lateral_offset

    x_start_end = cx - 1.4
    x_end_start = cx + 1.8
    meshes: list[trimesh.Trimesh] = [
        _box(0.0, lx, 0.0, ly, -depth - slab, -depth),  # pit floor
        _box(0.0, x_start_end, 0.0, ly, -depth, 0.0),  # start platform
        _box(x_end_start, lx, 0.0, ly, -depth, 0.0),  # far-side landing
    ]

    rng = np.random.default_rng(int(round(difficulty * 10000)) + 31)
    x_cur = x_start_end + gap
    while x_cur + stone_l <= x_end_start - 0.05:
        for sign in (-1.0, 1.0):
            yc = cy + sign * lat_offset + float(rng.uniform(-0.04, 0.04))
            dz = float(rng.uniform(-cfg.height_jitter, cfg.height_jitter)) * difficulty
            meshes.append(_box(x_cur, x_cur + stone_l, yc - stone_w / 2.0, yc + stone_w / 2.0, -depth, dz))
        x_cur += stone_l + gap

    origin = np.array([cx - 2.2, cy, 0.0], dtype=np.float64)
    return meshes, origin


@configclass
class DoubleStonesTerrainCfg(SubTerrainBaseCfg):
    function = double_stones_terrain
    terrain_name: str = "double_stones"
    stone_gap_range: tuple[float, float] = (0.12, 0.42)
    stone_size_range: tuple[float, float] = (0.35, 0.55)
    lateral_offset: float = 0.28
    height_jitter: float = 0.06
    pit_depth: float = 0.8


# -- 10: Balance Beams & Pallets (sparse foot, exact trimesh) -----------------------------------


def beam_and_pallets_terrain(
    difficulty: float, cfg: BeamAndPalletsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Alternates between narrow balance beams (L1026) and slatted pallets (L1027) spanning a pit."""
    lx, ly = cfg.size
    cx, cy = lx / 2.0, ly / 2.0
    depth = cfg.pit_depth
    slab = 0.4

    x_start_end = cx - 1.4
    x_end_start = cx + 1.8
    meshes: list[trimesh.Trimesh] = [
        _box(0.0, lx, 0.0, ly, -depth - slab, -depth),  # pit floor
        _box(0.0, x_start_end, 0.0, ly, -depth, 0.0),  # start platform
        _box(x_end_start, lx, 0.0, ly, -depth, 0.0),  # far-side landing
    ]

    use_beam = (int(round(difficulty * 1000)) % 2) == 0
    if use_beam:
        beam_w = _lerp(cfg.beam_width_range, 1.0 - difficulty)  # narrower at higher difficulty
        meshes.append(_box(x_start_end, x_end_start, cy - beam_w / 2.0, cy + beam_w / 2.0, -depth, 0.0))
    else:
        slat_d = _lerp(cfg.slat_depth_range, 1.0 - difficulty)
        slat_gap = _lerp(cfg.slat_gap_range, difficulty)
        slat_w = 2.4
        x_cur = x_start_end + slat_gap
        while x_cur + slat_d <= x_end_start - 0.05:
            meshes.append(_box(x_cur, x_cur + slat_d, cy - slat_w / 2.0, cy + slat_w / 2.0, -depth, 0.0))
            x_cur += slat_d + slat_gap

    origin = np.array([cx - 2.2, cy, 0.0], dtype=np.float64)
    return meshes, origin


@configclass
class BeamAndPalletsTerrainCfg(SubTerrainBaseCfg):
    function = beam_and_pallets_terrain
    terrain_name: str = "beam_pallets"
    beam_width_range: tuple[float, float] = (0.25, 0.60)
    slat_depth_range: tuple[float, float] = (0.16, 0.30)
    slat_gap_range: tuple[float, float] = (0.10, 0.28)
    pit_depth: float = 0.8


# -- 11 & 12: Dataset-Mesh Goal-Reaching Terrains (from ACP-TerrainContacts_G1) -----------------

_DATASET_INDEX_CACHE: dict[tuple[str, tuple[str, ...]], list[tuple[str, str]]] = {}


def _load_dataset_pairs(
    path: str, metadata_yaml: str, categories: tuple[str, ...]
) -> list[tuple[str, str]]:
    """Return ``[(terrain_abspath, motion_abspath), ...]`` ordered by ``categories`` (easy -> hard)."""
    key = (os.path.expanduser(metadata_yaml), categories)
    if key in _DATASET_INDEX_CACHE:
        return _DATASET_INDEX_CACHE[key]
    dataset_path = os.path.expanduser(path)
    with open(os.path.expanduser(metadata_yaml)) as f:
        data = yaml.safe_load(f)
    tid_to_tfile = {t["terrain_id"]: t["terrain_file"] for t in data["terrains"]}
    grouped: dict[str, list[tuple[str, str]]] = {c: [] for c in categories}
    for m in data["motion_files"]:
        mfile = m["motion_file"]
        tfile = tid_to_tfile[m["terrain_id"]]
        for c in categories:
            if c in mfile:
                grouped[c].append((os.path.join(dataset_path, tfile), os.path.join(dataset_path, mfile)))
                break
    pairs = [pair for c in categories for pair in grouped[c]]
    if not pairs:
        raise ValueError(f"No dataset terrains matched categories {categories} in {metadata_yaml}")
    _DATASET_INDEX_CACHE[key] = pairs
    return pairs


def dataset_goal_terrain(
    difficulty: float, cfg: DatasetGoalTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Load a real training-dataset terrain mesh as a goal-reaching RL terrain (no reference motion).

    Subclasses ``SubTerrainBaseCfg`` (NOT ``MotionMatchedTerrainCfg``) so ``TerrainMotion.match_scene``
    ignores it while ``GoalReachingCommand`` uses it as a procedural goal-reaching column.
    Spawns the robot at the paired clip's starting xy on the ground; ``FlatPatchSamplingCfg`` places
    goals on the far side / top of the real dataset obstacle.
    """
    pairs = _load_dataset_pairs(cfg.path, cfg.metadata_yaml, tuple(cfg.categories))
    idx = int(np.clip(difficulty * len(pairs), 0, len(pairs) - 1))
    terrain_abspath, motion_abspath = pairs[idx]

    terrain_mesh = trimesh.load(terrain_abspath, force="mesh")
    terrain_mesh = crop_terrain_mesh_aabb(
        terrain_mesh,
        x_max=cfg.size[0] / 2,
        x_min=-cfg.size[0] / 2,
        y_max=cfg.size[1] / 2,
        y_min=-cfg.size[1] / 2,
    )
    border_verts = terrain_mesh.vertices[
        np.logical_or(
            np.abs(terrain_mesh.vertices[:, 0]) > (cfg.size[0] / 2 - 0.05),
            np.abs(terrain_mesh.vertices[:, 1]) > (cfg.size[1] / 2 - 0.05),
        )
    ]
    border_height = float(np.mean(border_verts[:, 2])) if len(border_verts) > 0 else 0.0
    if np.isnan(border_height):
        border_height = 0.0

    move = np.eye(4)
    move[:2, 3] = np.array(cfg.size) / 2.0
    move[2, 3] = -border_height
    terrain_mesh.apply_transform(move)

    # Spawn at the paired clip's ground-level start xy (shifted to tile coords [0, Lx] x [0, Ly])
    start_xy = np.array([-2.2, 0.0], dtype=np.float64)
    if os.path.exists(motion_abspath):
        with np.load(motion_abspath) as npz:
            if "base_pos_w" in npz and len(npz["base_pos_w"]) > 0:
                start_xy = np.asarray(npz["base_pos_w"][0, :2], dtype=np.float64)
    start_xy[0] = np.clip(start_xy[0], -cfg.size[0] / 2 + 1.0, cfg.size[0] / 2 - 1.0)
    start_xy[1] = np.clip(start_xy[1], -cfg.size[1] / 2 + 1.0, cfg.size[1] / 2 - 1.0)
    origin = np.array([start_xy[0] + cfg.size[0] / 2.0, start_xy[1] + cfg.size[1] / 2.0, -border_height], dtype=np.float64)
    return [terrain_mesh], origin


@configclass
class DatasetGoalTerrainCfg(SubTerrainBaseCfg):
    function = dataset_goal_terrain
    terrain_name: str = "dataset_climbs"
    path: str = MISSING
    metadata_yaml: str = MISSING
    categories: list[str] = MISSING
