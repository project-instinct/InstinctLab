"""Warp kernels for grouped ray casting.

The grouping itself comes from Isaac Lab's per-environment multi-mesh tables: one environment's row holds
every mesh it may hit. The only kernel InstinctLab needs on top of upstream is near-hit rejection, which
drops hits closer than the sensor's minimum distance (the camera's own body, ground clutter under a height
scanner) before the closest hit is selected.
"""

import warp as wp


@wp.kernel(enable_backward=False)
def raycast_mesh_table_min_distance_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    mesh: wp.array2d(dtype=wp.uint64),
    ray_starts: wp.array2d(dtype=wp.vec3f),
    ray_directions: wp.array2d(dtype=wp.vec3f),
    mesh_positions: wp.array2d(dtype=wp.vec3f),
    mesh_rotations: wp.array2d(dtype=wp.quatf),
    max_dist: float,
    min_dist: float,
    return_normal: int,
    return_face_id: int,
    return_mesh_id: int,
    # output
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3f),
    ray_face_id: wp.array2d(dtype=wp.int32),
    ray_mesh_id: wp.array2d(dtype=wp.int16),
):
    """Ray-cast every environment's mesh row, rejecting hits nearer than ``min_dist``.

    Mirrors :func:`isaaclab.utils.warp.kernels.raycast_dynamic_meshes_kernel` (including its closest-hit
    ``atomic_min`` rule) with the additional near-hit rejection. Launch with
    ``dim=(meshes_per_env, num_envs, num_rays)``.

    Args:
        env_mask: Boolean mask selecting which environments to process. Shape is (num_envs,).
        mesh: Warp mesh handles per environment. Shape is (num_envs, meshes_per_env).
        ray_starts: World-frame ray start positions [m]. Shape is (num_envs, num_rays).
        ray_directions: World-frame ray directions. Shape is (num_envs, num_rays).
        mesh_positions: World-frame mesh positions [m]. Shape is (num_envs, meshes_per_env).
        mesh_rotations: World-frame mesh rotations. Shape is (num_envs, meshes_per_env).
        max_dist: Maximum ray-cast distance [m].
        min_dist: Minimum accepted hit distance [m]; closer hits are ignored.
        return_normal: Whether to write hit normals.
        return_face_id: Whether to write hit face ids.
        return_mesh_id: Whether to write the hit mesh slot.
        ray_hits: Hit positions [m]. Shape is (num_envs, num_rays).
        ray_distance: Closest-hit distances [m]. Shape is (num_envs, num_rays).
        ray_normal: Hit normals. Shape is (num_envs, num_rays).
        ray_face_id: Hit face ids. Shape is (num_envs, num_rays).
        ray_mesh_id: Hit mesh slot. Shape is (num_envs, num_rays).
    """
    tid_mesh_id, tid_env, tid_ray = wp.tid()
    if not env_mask[tid_env]:
        return

    mesh_pose = wp.transform(mesh_positions[tid_env, tid_mesh_id], mesh_rotations[tid_env, tid_mesh_id])
    mesh_pose_inv = wp.transform_inverse(mesh_pose)
    direction = wp.transform_vector(mesh_pose_inv, ray_directions[tid_env, tid_ray])
    start_pos = wp.transform_point(mesh_pose_inv, ray_starts[tid_env, tid_ray])

    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)
    if mesh_query_ray_t.result and mesh_query_ray_t.t > min_dist:
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # TODO(warp#1058): Use the return value of atomic_min to avoid the non-thread-safe equality check
        # below, as upstream does. Ties may write normals/face-ids/mesh-ids from different meshes; the hit
        # position is still correct because all tying threads compute the same world-space point.
        if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:
            hit_pos = start_pos + mesh_query_ray_t.t * direction
            ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)

            if return_normal == 1:
                ray_normal[tid_env, tid_ray] = wp.transform_vector(mesh_pose, mesh_query_ray_t.normal)
            if return_face_id == 1:
                ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face
            if return_mesh_id == 1:
                ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)
