"""Helpers for InstinctLab's flattened URDF importer output."""

from __future__ import annotations

def urdf_importer_link_prim_path(
    urdf_path: str,
    link_name: str,
    asset_prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> str:
    """Build a link prim path for InstinctLab's flattened URDF output.

    The flattening pass moves every imported rigid body directly below the asset
    prim and uses the URDF link name as the prim name.
    """
    return f"{asset_prim_path}/{link_name}"
