"""Tests for the grouped ray-caster target registry.

They never launch a simulator or read USD: the registry's per-environment checks and its delegation to
upstream's record builder are exercised through a stub that supplies the mesh records upstream would return.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from instinctlab.sensors.grouped_ray_caster.flat_target_prim_registry import FlatTargetPrimRegistryMixin


class _UpstreamBuilder:
    """Stand-in for upstream's ``_build_mesh_records`` that returns prepared per-environment rows."""

    def __init__(self, rows: list[list[tuple]]):
        self.rows = rows
        self.calls = []

    def _build_mesh_records(self, target_cfg, plan):
        self.calls.append((target_cfg, plan))
        return self.rows


class _Sensor(FlatTargetPrimRegistryMixin, _UpstreamBuilder):
    """A minimal sensor: the registry mixin is the only method ahead of the stub builder."""


def _target(prim_expr: str) -> SimpleNamespace:
    return SimpleNamespace(prim_expr=prim_expr, track_mesh_transforms=False)


def _record(mesh_id: int) -> tuple:
    return (mesh_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), None)


def test_uniform_records_are_returned_unchanged():
    rows = [[_record(0)], [_record(0)]]
    sensor = _Sensor(rows)
    target = _target("/World/ground")

    assert sensor._build_mesh_records(target, plan=None) is rows


def test_sensor_without_records_is_accepted():
    """A static target may exist in only some prototypes; empty environments are not ragged."""
    rows = [[_record(0), _record(1)], [], [_record(0), _record(1)]]
    sensor = _Sensor(rows)

    assert sensor._build_mesh_records(_target("/World/envs/env_[^/]+/Obstacle"), plan=None) is rows


def test_ragged_target_raises():
    """A target expression that resolves a different number of prims per environment is rejected."""
    rows = [[_record(0), _record(1)], [_record(0)]]
    sensor = _Sensor(rows)

    with pytest.raises(RuntimeError, match="different number of prims per environment"):
        sensor._build_mesh_records(_target("/World/envs/env_[^/]+/Obstacle"), plan=None)


def test_ragged_target_names_the_expression_and_counts():
    rows = [[], [_record(0)], [_record(0), _record(1), _record(2)]]
    sensor = _Sensor(rows)

    with pytest.raises(RuntimeError) as error:
        sensor._build_mesh_records(_target("/World/envs/env_[^/]+/cube"), plan=None)

    message = str(error.value)
    assert "/World/envs/env_[^/]+/cube" in message
    assert "[0, 1, 3]" in message
