"""Tests for :mod:`instinctlab.utils.mesh`.

The loader keeps the semantics of the removed InstinctLab ``spawn_from_mesh`` helper: unprocessed load, scene
concatenation, and a clear rejection of files that hold no triangles. The tests write tiny OBJ files and never
launch a simulator.
"""

from __future__ import annotations

import trimesh

import pytest

from instinctlab.utils.mesh import load_mesh_file

TRIANGLE_OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""

QUAD_OBJ = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""

LINES_OBJ = """v 0 0 0
v 1 0 0
l 1 2
"""

TWO_OBJECT_OBJ = """o a
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
o b
v 2 0 0
v 3 0 0
v 2 1 0
f 4 5 6
"""


def _write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_loads_triangles_unprocessed(tmp_path):
    mesh = load_mesh_file(_write(tmp_path, "tri.obj", TRIANGLE_OBJ))
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)


def test_quads_come_back_triangulated(tmp_path):
    mesh = load_mesh_file(_write(tmp_path, "quad.obj", QUAD_OBJ))
    assert mesh.faces.shape == (2, 3)


def test_multi_object_file_becomes_one_mesh(tmp_path):
    mesh = load_mesh_file(_write(tmp_path, "two.obj", TWO_OBJECT_OBJ))
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.vertices.shape == (6, 3)
    assert mesh.faces.shape == (2, 3)


def test_scene_is_concatenated(tmp_path, monkeypatch):
    """``trimesh.load_mesh`` returns a ``Scene`` for file types it cannot merge itself."""
    path = _write(tmp_path, "two.obj", TWO_OBJECT_OBJ)
    parts = [trimesh.creation.box(extents=(1.0, 1.0, 1.0)), trimesh.creation.box(extents=(1.0, 1.0, 1.0))]
    scene = trimesh.Scene({"a": parts[0], "b": parts[1]})
    monkeypatch.setattr(trimesh, "load_mesh", lambda *_args, **_kwargs: scene)
    mesh = load_mesh_file(path)
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) == len(parts[0].vertices) + len(parts[1].vertices)


def test_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "m.obj").write_text(TRIANGLE_OBJ)
    mesh = load_mesh_file("~/m.obj")
    assert mesh.faces.shape == (1, 3)


def test_non_triangle_file_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="without faces"):
        load_mesh_file(_write(tmp_path, "lines.obj", LINES_OBJ))


def test_non_mesh_file_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="Expected a triangle mesh"):
        load_mesh_file(_write(tmp_path, "empty.obj", ""))
