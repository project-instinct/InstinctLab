# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""Deterministic cache for InstinctLab's generated USD assets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ASSET_CACHE_POSTPROCESSOR_VERSION = 2

_THIS_FILE = Path(__file__).resolve()
_POSTPROCESSOR_FILES = (_THIS_FILE, _THIS_FILE.with_name("from_files.py"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    return value


def _referenced_mesh_paths(urdf_path: Path) -> list[Path]:
    root = ET.parse(urdf_path).getroot()
    mesh_paths = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith("package://"):
            continue
        path = (urdf_path.parent / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF '{urdf_path}' references missing mesh '{path}'")
        mesh_paths.add(path)
    return sorted(mesh_paths)


def _asset_cache_inputs(cfg) -> dict[str, Any]:
    source_path = Path(cfg.asset_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"URDF asset does not exist: {source_path}")

    mesh_paths = _referenced_mesh_paths(source_path)
    mesh_inputs = {}
    for path in mesh_paths:
        try:
            key = str(path.relative_to(source_path.parent))
        except ValueError:
            key = path.name
        mesh_inputs[key] = _sha256_file(path)
    postprocessor_inputs = {path.name: _sha256_file(path) for path in _POSTPROCESSOR_FILES}
    source_file_inputs = {
        "urdf": _sha256_file(source_path),
        "meshes": mesh_inputs,
        "postprocessors": postprocessor_inputs,
    }
    converter_fields = (
        "collision_from_visuals",
        "collision_type",
        "fix_base",
        "link_density",
        "merge_fixed_joints",
        "merge_mesh",
        "replace_cylinders_with_capsules",
        "robot_type",
        "ros_package_paths",
        "run_asset_transformer",
        "run_multi_physics_conversion",
        "self_collision",
    )
    converter_config = {name: _to_jsonable(getattr(cfg, name)) for name in converter_fields}
    if getattr(cfg, "joint_drive", None) is not None:
        converter_config["joint_drive"] = _to_jsonable(cfg.joint_drive)

    return {
        "asset_name": source_path.stem,
        "asset_path": str(source_path),
        "postprocessor_version": ASSET_CACHE_POSTPROCESSOR_VERSION,
        "isaaclab_version": _package_version("isaaclab"),
        "isaacsim_version": _package_version("isaacsim"),
        "converter_config": converter_config,
        "files": source_file_inputs,
    }


def asset_cache_digest(cfg) -> str:
    """Return the content address for the generated USD derived from *cfg*."""
    payload = json.dumps(_asset_cache_inputs(cfg), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def asset_cache_root() -> Path:
    """Return the configured InstinctLab generated-asset cache root."""
    configured = os.environ.get("INSTINCTLAB_ASSET_CACHE")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".cache" / "instinctlab" / "assets"


def asset_cache_dir(cfg) -> Path:
    """Return the digest-scoped directory for *cfg*."""
    return asset_cache_root() / "urdf" / Path(cfg.asset_path).stem / asset_cache_digest(cfg)


def asset_cached_usd_path(cfg) -> Path:
    """Return the expected cached flattened USD path for *cfg*."""
    return asset_cache_dir(cfg) / f"{Path(cfg.asset_path).stem}.usda"


def has_valid_asset_cache(cfg) -> bool:
    """Return whether the cached flattened USD exists and matches all build inputs."""
    cache_dir = asset_cache_dir(cfg)
    usd_path = asset_cached_usd_path(cfg)
    metadata_path = cache_dir / "instinctlab_asset.json"
    if not usd_path.is_file() or not metadata_path.is_file():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = asset_cache_digest(cfg)
        if metadata.get("asset_digest") != digest:
            return False
        if metadata.get("inputs") != _asset_cache_inputs(cfg):
            return False
        if metadata.get("final_usd_sha256") != _sha256_file(usd_path):
            return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def publish_asset_cache(cfg, flat_usd_path: str | Path) -> Path:
    """Atomically publish a flattened USD and its provenance metadata."""
    cache_dir = asset_cache_dir(cfg)
    cached_path = asset_cached_usd_path(cfg)
    if has_valid_asset_cache(cfg):
        return cached_path

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    flat_path = Path(flat_usd_path)
    if not flat_path.is_file():
        raise FileNotFoundError(f"Flattened URDF USD does not exist: {flat_path}")

    digest = asset_cache_digest(cfg)
    inputs = _asset_cache_inputs(cfg)
    metadata = {
        "asset_digest": digest,
        "final_usd_sha256": _sha256_file(flat_path),
        "inputs": inputs,
        "root_usd": flat_path.name,
    }

    promotion_dir = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=str(cache_dir.parent)))
    try:
        shutil.copy2(flat_path, promotion_dir / flat_path.name)
        with (promotion_dir / "instinctlab_asset.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        try:
            promotion_dir.rename(cache_dir)
        except FileExistsError:
            if not has_valid_asset_cache(cfg):
                raise RuntimeError(f"URDF cache race detected at '{cache_dir}'")
    finally:
        if promotion_dir.exists():
            shutil.rmtree(promotion_dir, ignore_errors=True)

    return cached_path


def find_uncached_asset_cfgs(env_cfg) -> list:
    """Return custom URDF spawn configs whose digest-scoped USD is not cached."""
    from .from_files_cfg import UrdfFileCfg

    visited: set[int] = set()
    uncached = []

    def visit(node):
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        if isinstance(node, UrdfFileCfg) and node.asset_cache_enabled and not has_valid_asset_cache(node):
            uncached.append(node)
            return
        if isinstance(node, dict):
            children = node.values()
        elif isinstance(node, (list, tuple, set)):
            children = node
        else:
            try:
                children = vars(node).values()
            except TypeError:
                return
        for child in children:
            if child is None or isinstance(child, (int, float, str, bool)):
                continue
            visit(child)

    visit(env_cfg)
    return uncached


def ensure_asset_cache_kit_args(env_cfg, launcher_args) -> bool:
    """Force a windowless Kit app when a custom URDF asset cache is cold.

    When the digest-scoped USD already exists, this is a no-op and normal kitless
    Newton launches such as ``--viz none`` remain unchanged.
    """
    if not find_uncached_asset_cfgs(env_cfg):
        return False

    kit_args = getattr(launcher_args, "kit_args", "") or ""
    if "--/app/windowless=true" not in kit_args:
        kit_args = f"{kit_args} --/app/windowless=true".strip()

    if isinstance(launcher_args, dict):
        launcher_args["visualizer"] = ["kit"]
        launcher_args["visualizer_explicit"] = True
        launcher_args["kit_args"] = kit_args
    else:
        launcher_args.visualizer = ["kit"]
        launcher_args.visualizer_explicit = True
        launcher_args.kit_args = kit_args
    return True
