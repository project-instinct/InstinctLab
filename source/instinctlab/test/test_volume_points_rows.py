"""Tests for :func:`instinctlab.sensors.volume_points.volume_points.env_body_rows`.

A PhysX rigid-body view over several body globs lists each glob's matches in turn (body-major), so the sensor must not
reshape the rows as environment-major.
"""

from __future__ import annotations

import pytest

from instinctlab.sensors.volume_points.volume_points import env_body_rows

SUFFIXES = ["/left_ankle_roll_link", "/right_ankle_roll_link"]


def _path(env_id: int, suffix: str) -> str:
    return f"/World/envs/env_{env_id}/Robot{suffix}"


def test_body_major_rows_map_to_environment_order():
    # rows: L0 L1 L2 R0 R1 R2 (body-major), with env_10 sorting after env_2
    envs = [0, 2, 10]
    paths = [_path(e, s) for s in SUFFIXES for e in envs]
    rows = env_body_rows(paths, SUFFIXES, "cpu").view(len(envs), len(SUFFIXES))
    assert [[paths[r] for r in env_rows] for env_rows in rows.tolist()] == [
        [_path(e, s) for s in SUFFIXES] for e in envs
    ]


def test_environment_major_rows_are_kept():
    paths = [_path(e, s) for e in range(3) for s in SUFFIXES]
    assert env_body_rows(paths, SUFFIXES, "cpu").tolist() == list(range(6))


def test_unknown_body_raises():
    with pytest.raises(RuntimeError, match="Cannot map"):
        env_body_rows([_path(0, "/pelvis")], SUFFIXES, "cpu")
