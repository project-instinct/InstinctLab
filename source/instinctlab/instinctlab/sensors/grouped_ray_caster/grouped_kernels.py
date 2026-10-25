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

    One thread owns one ray and reduces the hits of its environment's mesh row serially, so all outputs of a ray
    describe the same (closest) intersection. Isaac Lab's :func:`~isaaclab.utils.warp.kernels.raycast_dynamic_meshes_kernel`
    launches one thread per mesh and resolves the closest hit with ``atomic_min`` plus a read-back, which lets a
    farther hit overwrite the position of a nearer one. Rays without an accepted hit keep the values written by
    the preceding fill kernel; their face and mesh ids are set to -1. Launch with ``dim=(num_envs, num_rays)``.

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
    tid_env, tid_ray = wp.tid()
    if not env_mask[tid_env]:
        return

    ray_start = ray_starts[tid_env, tid_ray]
    ray_direction = ray_directions[tid_env, tid_ray]
    closest_distance = float(max_dist)
    closest_hit = wp.vec3f()
    closest_normal = wp.vec3f()
    closest_face_id = int(-1)
    closest_mesh_id = int(-1)

    for mesh_slot in range(mesh.shape[1]):
        mesh_pose = wp.transform(mesh_positions[tid_env, mesh_slot], mesh_rotations[tid_env, mesh_slot])
        mesh_pose_inv = wp.transform_inverse(mesh_pose)
        start_local = wp.transform_point(mesh_pose_inv, ray_start)
        direction_local = wp.transform_vector(mesh_pose_inv, ray_direction)
        query = wp.mesh_query_ray(mesh[tid_env, mesh_slot], start_local, direction_local, max_dist)
        if query.result and query.t > min_dist and query.t < closest_distance:
            closest_distance = query.t
            closest_mesh_id = mesh_slot
            closest_hit = wp.transform_point(mesh_pose, start_local + query.t * direction_local)
            if return_normal == 1:
                closest_normal = wp.transform_vector(mesh_pose, query.normal)
            if return_face_id == 1:
                closest_face_id = query.face

    if closest_mesh_id >= 0:
        ray_distance[tid_env, tid_ray] = closest_distance
        ray_hits[tid_env, tid_ray] = closest_hit
        if return_normal == 1:
            ray_normal[tid_env, tid_ray] = closest_normal
    if return_face_id == 1:
        ray_face_id[tid_env, tid_ray] = closest_face_id
    if return_mesh_id == 1:
        ray_mesh_id[tid_env, tid_ray] = wp.int16(closest_mesh_id)
