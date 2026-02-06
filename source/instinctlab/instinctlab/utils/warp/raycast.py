import numpy as np
import torch

import warp as wp

from isaaclab.utils.math import convert_quat

# disable warp module initialization messages
wp.config.quiet = True
# initialize the warp module
wp.init()

from . import kernels


def raycast_mesh_grouped(
    mesh_wp_device: wp.context.Device,
    mesh_wp_ids: torch.Tensor,
    mesh_transforms: torch.Tensor,
    mesh_inv_transforms: torch.Tensor,
    ray_group_ids: torch.Tensor,
    mesh_idxs_for_group: torch.Tensor,
    meah_idxs_slice_for_group: torch.Tensor,
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    max_dist: float = 1e6,
    min_dist: float = 0.0,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
    return_mesh_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against a mesh with different collision groups.

    - Each ray and mesh has their own collision group ID.
    - The mesh with collision group ID will be hit by every ray in the space.
    - This is a extended implementation of the `raycast_mesh` in isaaclab.utils.warp.ops`

    ### NOTE
        return_face_id might not work correctly with the current implementation of warp.
    """
    # extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device
    # device of the mesh
    torch_device = wp.device_to_torch(mesh_wp_device)
    # reshape the tensors
    ray_starts = ray_starts.to(torch_device).view(-1, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(-1, 3).contiguous()
    ray_group_ids = ray_group_ids.to(torch_device).view(-1).contiguous()
    mesh_idxs_for_group = mesh_idxs_for_group.to(torch_device).contiguous()
    meah_idxs_slice_for_group = meah_idxs_slice_for_group.to(torch_device).view(-1).contiguous()
    num_rays = ray_starts.shape[0]
    # create output tensor for the ray hits
    ray_hits = torch.full((num_rays, 3), max_dist, device=torch_device).contiguous()
    # map the memory to warp arrays
    mesh_wp_ids_wp = wp.from_torch(mesh_wp_ids, dtype=wp.uint64)
    mesh_transforms_ = torch.concatenate(
        [mesh_transforms[:, :3], convert_quat(mesh_transforms[:, 3:], to="xyzw")],
        dim=-1,
    ).contiguous()
    mesh_transforms_wp = wp.from_torch(mesh_transforms_, dtype=wp.transform)
    mesh_inv_transforms_ = torch.concatenate(
        [mesh_inv_transforms[:, :3], convert_quat(mesh_inv_transforms[:, 3:], to="xyzw")],
        dim=-1,
    ).contiguous()
    mesh_inv_transforms_wp = wp.from_torch(mesh_inv_transforms_, dtype=wp.transform)
    ray_group_ids_wp = wp.from_torch(ray_group_ids, dtype=wp.int32)
    mesh_idxs_for_group_wp = wp.from_torch(mesh_idxs_for_group, dtype=wp.int32)
    meah_idxs_slice_for_group_wp = wp.from_torch(meah_idxs_slice_for_group, dtype=wp.int32)
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
    else:
        ray_distance = None
        ray_distance_wp = wp.empty((1,), dtype=wp.float32, device=torch_device)

    if return_normal:
        ray_normal = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1,), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.ones((num_rays,), dtype=torch.int32, device=torch_device).contiguous() * (-1)
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1,), dtype=wp.int32, device=torch_device)

    if return_mesh_id:
        ray_mesh_id = torch.full((num_rays,), -1, dtype=torch.int16, device=torch_device).contiguous()
        ray_mesh_id_wp = wp.from_torch(ray_mesh_id, dtype=wp.int16)
    else:
        ray_mesh_id = None
        ray_mesh_id_wp = wp.empty((1,), dtype=wp.int16, device=torch_device)

    # launch the warp kernel
    wp.launch(
        kernel=kernels.raycast_mesh_kernel_grouped_transformed,
        dim=num_rays,
        inputs=[
            mesh_wp_ids_wp,
            mesh_transforms_wp,
            mesh_inv_transforms_wp,
            ray_group_ids_wp,
            mesh_idxs_for_group_wp,
            meah_idxs_slice_for_group_wp,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distance_wp,
            ray_normal_wp,
            ray_face_id_wp,
            ray_mesh_id_wp,
            max_dist,
            min_dist,
            int(return_distance),
            int(return_normal),
            int(return_face_id),
            int(return_mesh_id),
        ],
        device=mesh_wp_device,
    )
    # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller and Original implementation in IsaacLab.
    wp.synchronize()

    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[0], shape[1])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[0], shape[1])
    if return_mesh_id:
        ray_mesh_id = ray_mesh_id.to(device).view(shape[0], shape[1])
    return ray_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id, ray_mesh_id
