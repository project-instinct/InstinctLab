"""Tests for the converted-URDF inertia check in :mod:`instinctlab.sim.spawners.from_files.from_files`.

The tests author tiny USD stages with ``pxr`` and never launch a simulator.
"""

from __future__ import annotations

import numpy as np

import pytest
from pxr import Gf, Usd, UsdPhysics

from instinctlab.sim.spawners.from_files.from_files import _check_urdf_inertia

# One link with off-diagonal inertia and a rotated inertial frame, one massless child merged into it by a fixed joint.
URDF = """<robot name="r">
  <link name="arm">
    <inertial>
      <origin xyz="0 0 0" rpy="{rpy}"/>
      <mass value="1.0"/>
      <inertia ixx="2.891e-4" ixy="6.53e-5" ixz="1.72e-5" iyy="4.152e-4" iyz="-5.6e-6" izz="4.197e-4"/>
    </inertial>
  </link>
  <link name="hand">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>
  <link name="base">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="2e-3" iyz="0" izz="3e-3"/>
    </inertial>
  </link>
  <joint name="wrist" type="fixed"><parent link="base"/><child link="hand"/></joint>
</robot>
"""


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _principal(tensor: np.ndarray, transpose: bool) -> tuple[Gf.Quatf, Gf.Vec3f]:
    """Principal moments and the quaternion of R with tensor = R @ diag @ R.T (or its inverse, EA's bug)."""
    moments, rotation = np.linalg.eigh(tensor)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    if transpose:
        rotation = rotation.T
    # Gf matrices act on row vectors, so they hold the transpose of the column-vector rotation.
    quat = Gf.Matrix3d(*rotation.T.flatten().tolist()).ExtractRotation().GetQuat()
    return Gf.Quatf(quat), Gf.Vec3f(*moments.tolist())


def _write(tmp_path, rpy: tuple[float, float, float], transpose: bool) -> tuple[str, str]:
    urdf_path = tmp_path / "r.urdf"
    urdf_path.write_text(URDF.format(rpy=" ".join(str(v) for v in rpy)))
    arm = np.array([[2.891e-4, 6.53e-5, 1.72e-5], [6.53e-5, 4.152e-4, -5.6e-6], [1.72e-5, -5.6e-6, 4.197e-4]])
    rotation = _rpy_matrix(*rpy)
    stage = Usd.Stage.CreateNew(str(tmp_path / "r.usda"))
    for name, tensor in (("arm", rotation @ arm @ rotation.T), ("base", np.diag((5e-3, 5e-3, 5e-3)))):
        prim = stage.DefinePrim(f"/r/{name}", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        quat, moments = _principal(tensor, transpose)
        mass_api.CreatePrincipalAxesAttr(quat)
        mass_api.CreateDiagonalInertiaAttr(moments)
    stage.GetRootLayer().Save()
    return str(tmp_path / "r.usda"), str(urdf_path)


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (0.3, -0.7, 1.1)])
def test_correct_principal_axes_pass(tmp_path, rpy):
    # base has merged-in inertia (different from its URDF tensor) and must be skipped
    _check_urdf_inertia(*_write(tmp_path, rpy, transpose=False))


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (0.3, -0.7, 1.1)])
def test_transposed_principal_axes_raise(tmp_path, rpy):
    with pytest.raises(RuntimeError, match="arm"):
        _check_urdf_inertia(*_write(tmp_path, rpy, transpose=True))
