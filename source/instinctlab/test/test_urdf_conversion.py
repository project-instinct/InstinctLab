"""Tests for :mod:`instinctlab.sim.spawners.from_files.urdf_conversion` (Kit-free URDF conversion).

The conversion runs the standalone importer from Isaac Lab's ``importers`` extra in a child process, so these tests need
that extra installed (the InstinctLab image has it) and never launch Kit.
"""

from __future__ import annotations

import pytest
from pxr import Usd, UsdPhysics

from instinctlab.sim.spawners.from_files import urdf_conversion
from instinctlab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg

URDF = """<robot name="arm">
  <link name="base">
    <inertial><mass value="2.0"/><inertia ixx="1e-2" ixy="0" ixz="0" iyy="2e-2" iyz="0" izz="3e-2"/></inertial>
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="2.891e-4" ixy="6.53e-5" ixz="1.72e-5" iyy="4.152e-4" iyz="-5.6e-6" izz="4.197e-4"/>
    </inertial>
    <collision><geometry><box size="0.1 0.1 0.3"/></geometry></collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="5"/>
  </joint>
</robot>
"""


@pytest.fixture
def urdf_cfg(tmp_path):
    (tmp_path / "arm.urdf").write_text(URDF)
    return UrdfFileCfg(asset_path=str(tmp_path / "arm.urdf"), fix_base=False, merge_fixed_joints=True)


def test_converts_without_kit(urdf_cfg):
    stage = Usd.Stage.Open(urdf_conversion.convert_urdf(urdf_cfg))
    bodies = sorted(p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    joints = sorted(p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint))
    assert bodies == ["base", "link1"]
    assert joints == ["joint1"]


def test_ignores_the_kit_environment(urdf_cfg, tmp_path, monkeypatch):
    # Isaac Sim's python.sh puts its regular ``isaacsim`` package first on PYTHONPATH, which hides the standalone
    # importer's ``isaacsim.asset`` portion; the child process must not inherit that.
    shadow = tmp_path / "kit_python_packages" / "isaacsim"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", str(shadow.parent))
    monkeypatch.setenv("CARB_APP_PATH", "/nonexistent/kit")
    assert urdf_conversion.convert_urdf(urdf_cfg).endswith("arm.usda")


def test_failure_reports_the_child_output(tmp_path):
    (tmp_path / "broken.urdf").write_text(
        "<robot name='broken'><link name='a'/><joint name='j' type='revolute'/></robot>"
    )
    with pytest.raises(RuntimeError, match="Kit-free URDF conversion of .*broken.urdf"):
        urdf_conversion.convert_urdf(UrdfFileCfg(asset_path=str(tmp_path / "broken.urdf")))
