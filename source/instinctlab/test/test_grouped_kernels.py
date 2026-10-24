"""Tests for :mod:`instinctlab.sensors.grouped_ray_caster.grouped_kernels` (warp only, no simulator)."""

from __future__ import annotations

import numpy as np

import pytest
import warp as wp

from instinctlab.sensors.grouped_ray_caster.grouped_kernels import raycast_mesh_table_min_distance_kernel

DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


def _square(z: float, half: float = 5.0) -> wp.Mesh:
    """A horizontal square at height ``z``, facing +z."""
    points = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(points=wp.array(points, dtype=wp.vec3), indices=wp.array(indices, dtype=wp.int32))


def _cast(device: str, heights: list[float], min_dist: float, num_envs: int = 3, num_rays: int = 64):
    with wp.ScopedDevice(device):
        meshes = [_square(z) for z in heights]
        mesh_ids = np.array([[m.id for m in meshes]] * num_envs, dtype=np.uint64)
        xy = np.random.default_rng(0).uniform(-4.0, 4.0, size=(num_envs, num_rays, 2)).astype(np.float32)
        starts = np.concatenate([xy, np.full((num_envs, num_rays, 1), 3.0, np.float32)], axis=-1)
        dirs = np.tile(np.array([0.0, 0.0, -1.0], np.float32), (num_envs, num_rays, 1))
        quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (num_envs, len(heights), 1))
        hits = wp.array(np.full((num_envs, num_rays, 3), np.inf, np.float32), dtype=wp.vec3f)
        dist = wp.array(np.full((num_envs, num_rays), np.inf, np.float32), dtype=wp.float32)
        normal = wp.zeros((num_envs, num_rays), dtype=wp.vec3f)
        face = wp.zeros((num_envs, num_rays), dtype=wp.int32)
        slot = wp.zeros((num_envs, num_rays), dtype=wp.int16)
        wp.launch(
            raycast_mesh_table_min_distance_kernel,
            dim=(num_envs, num_rays),
            inputs=[
                wp.array(np.ones(num_envs, bool), dtype=wp.bool),
                wp.array2d(mesh_ids, dtype=wp.uint64),
                wp.array(starts, dtype=wp.vec3f),
                wp.array(dirs, dtype=wp.vec3f),
                wp.zeros((num_envs, len(heights)), dtype=wp.vec3f),
                wp.array(quat, dtype=wp.quatf),
                1.0e6,
                min_dist,
                1,
                1,
                1,
            ],
            outputs=[hits, dist, normal, face, slot],
        )
        wp.synchronize()
        return starts, hits.numpy(), dist.numpy(), normal.numpy(), face.numpy(), slot.numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_closest_hit_outputs_are_consistent(device):
    # slots 0..2 at z = 0, 2, 1: the closest surface below z=3 is slot 1 (distance 1)
    starts, hits, dist, normal, face, slot = _cast(device, [0.0, 2.0, 1.0], min_dist=0.0)
    np.testing.assert_allclose(dist, 1.0, atol=1e-6)
    assert (slot == 1).all()
    np.testing.assert_allclose(hits[..., 2], 2.0, atol=1e-6)
    np.testing.assert_allclose(hits[..., :2], starts[..., :2], atol=1e-6)
    np.testing.assert_allclose(normal, np.broadcast_to([0.0, 0.0, 1.0], normal.shape), atol=1e-6)
    assert (face >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_min_distance_rejects_near_hits(device):
    # slot 1 sits 0.01 m below the ray start, inside min_dist; the next surface is slot 2 at distance 1
    _, hits, dist, _, _, slot = _cast(device, [0.0, 2.99, 2.0], min_dist=0.05)
    np.testing.assert_allclose(dist, 1.0, atol=1e-6)
    assert (slot == 2).all()
    np.testing.assert_allclose(hits[..., 2], 2.0, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_miss_keeps_fill_values(device):
    # every surface is above the ray start: no hit
    _, hits, dist, _, face, slot = _cast(device, [4.0, 5.0], min_dist=0.0)
    assert np.isinf(dist).all() and np.isinf(hits).all()
    assert (face == -1).all() and (slot == -1).all()
