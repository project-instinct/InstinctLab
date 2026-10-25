"""Loader for mesh files spawned through Isaac Lab's in-memory mesh path."""

from __future__ import annotations

import os
import trimesh


def load_mesh_file(path: str) -> trimesh.Trimesh:
    """Load *path* as a single triangle mesh, expanding a leading ``~``.

    The mesh is loaded unprocessed so the vertices stay exactly as authored. A multi-object file becomes one
    concatenated mesh, and a file that holds no triangles is rejected because the USD mesh prim requires them.
    """
    mesh = trimesh.load_mesh(os.path.expanduser(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at '{path}', got {type(mesh).__name__}.")
    if len(mesh.faces) == 0:
        raise TypeError(f"Expected a triangle mesh at '{path}', got a mesh without faces.")
    return mesh
