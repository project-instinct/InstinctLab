from __future__ import annotations

import math
import numpy as np
import os
import random
import torch
import trimesh
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from numpy.linalg import norm
from typing import TYPE_CHECKING

import cv2
from pxr import UsdGeom, UsdPhysics
from sklearn.cluster import DBSCAN

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import patterns
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from instinctlab.utils.warp.cylinder import CylinderSpatialGrid

from .virtual_obstacle_base import VirtualObstacleBase

if TYPE_CHECKING:
    from .edge_cylinder_cfg import (
        EdgeCylinderCfg,
        GreedyconcatEdgeCylinderCfg,
        PluckerEdgeCylinderCfg,
        RansacEdgeCylinderCfg,
        RayEdgeCylinderCfg,
        FeatureEdgeCylinderCfg,
    )
import pyvista as pv


class EdgeCylinder(VirtualObstacleBase):
    """Base class for edge detectors."""

    def __init__(self, cfg: EdgeCylinderCfg):
        self.cfg: EdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def generate(self, mesh: trimesh.Trimesh, device="cpu") -> None:
        """Detect sharp edges in the mesh and store the edge cylinder as virtual obstacle.

        Args:
            mesh: The trimesh object to analyze.

        Returns:
            A np array batch indicating the edges: (num_edges, 6)
            - x, y, z coordinates of the edge start point
            - x, y, z coordinates of the edge end point
        """
        angles = mesh.face_adjacency_angles
        # convert max_angle in degrees to radians
        threshold = np.deg2rad(self.angle_threshold)
        # pick only those adjacencies whose angle exceeds threshold
        sharp_mask = angles > threshold
        if not np.any(sharp_mask):
            edge_end_points = np.empty((0, 6), dtype=np.float32)
            print("[WARNING] No sharp edges detected.")
        else:
            # get the corresponding edges (vertex index pairs)
            # face_adjacency_edges is (n_adj, 2) vertex indices for each adjacency
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]

            # look up vertex coordinates
            v = mesh.vertices
            # build (num_edges, 6) array: [x0,y0,z0, x1,y1,z1]
            # build the (num_edges, 6) array of sharp edge end‐point coordinates
            edge_coords = np.hstack([v[sharp_edges[:, 0]], v[sharp_edges[:, 1]]])
            edge_end_points = self.process_edges(edge_coords)
            print(f"Detected {edge_end_points.shape[0]} edges after processing.")
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.edges_pyt = torch.tensor(edge_end_points, dtype=torch.float32, device=self.device)
        # create a cylinder spatial grid for the edges and for penetration offset computation
        if edge_end_points.size > 0:
            self.cylinders = CylinderSpatialGrid(
                cylinders=np.concatenate(
                    [
                        edge_end_points,
                        np.ones_like(edge_end_points[:, :1]) * self.cfg.cylinder_radius,
                    ],
                    axis=1,
                ),
                num_grid_cells=self.cfg.num_grid_cells,
                device=self.device,
            )
        else:
            self.cylinders = None

    def disable_visualizer(self):
        if hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer.set_visibility(False)

    def visualize(self):
        if self.edges_pyt.numel() == 0:
            return

        if not hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer = VisualizationMarkers(self.cfg.visualizer)
            self._cylinder_rotate_y_90 = math_utils.quat_from_angle_axis(
                angle=torch.tensor([np.pi / 2], device=self.device),
                axis=torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
            )  # shape (1, 4)

        trans = (self.edges_pyt[:, :3] + self.edges_pyt[:, 3:6]) / 2
        # compute the direction quaternion
        direction = self.edges_pyt[:, 3:6] - self.edges_pyt[:, :3]
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)  # arrow-direction
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
        quat = math_utils.quat_from_angle_axis(
            angle,
            axis,
        )
        quat = math_utils.quat_mul(quat, self._cylinder_rotate_y_90.expand(quat.shape[0], -1))
        # compute the scale to match the length and the edge thickness.
        scales = torch.ones(len(self.edges_pyt), 3, device=self.device)
        scales[:, 0] = self.cfg.cylinder_radius
        scales[:, 1] = self.cfg.cylinder_radius
        scales[:, 2] = torch.norm(direction, dim=-1)
        self._cylinder_visualizer.visualize(
            translations=trans,
            orientations=quat,
            scales=scales,
        )
        self._cylinder_visualizer.set_visibility(True)

    def get_points_penetration_offset(self, points):
        return (
            self.cylinders.get_points_penetration_offset(points)
            if self.cylinders is not None
            else torch.zeros_like(points, device=self.device)
        )

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        """Process the edge coordinates.

        Args:
            edge_coords: The edge coordinates array of shape (num_edges, 6).

        Returns:
            A np array of processed edge coordinates.
        """
        return edge_coords


class PluckerEdgeCylinder(EdgeCylinder):
    """Detects sharp edges in a mesh using the Laplacian operator."""

    def __init__(self, cfg: PluckerEdgeCylinderCfg):
        self.cfg: PluckerEdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        """Process the edge coordinates using Plücker coordinates.

        Args:
            edge_coords: The edge coordinates array of shape (num_edges, 6).

        Returns:
            A np array of processed edge coordinates.
        """
        # compute Plücker coordinates for each edge (batch)
        p0 = edge_coords[:, :3]  # shape (N, 3)
        p1 = edge_coords[:, 3:]  # shape (N, 3)

        # directions (N, 3) and normalization
        d = p1 - p0
        lengths = np.linalg.norm(d, axis=1, keepdims=True)
        d_norm = d / lengths

        # enforce canonical orientation so opposite directions map to the same line
        for i, vec in enumerate(d_norm):
            # find the first non-zero component
            for comp in vec:
                if abs(comp) < 1e-8:
                    continue
                if comp < 0:
                    d_norm[i] = -vec
                break

        # moments (N, 3)
        m = np.cross(d_norm, p0)

        para = np.hstack((d_norm, m))  # shape (N, 6)
        para = np.round(para, 6)

        # find identical rows in para
        unique_rows, inv_idx = np.unique(para, axis=0, return_inverse=True)
        groups = {}
        for i, g in enumerate(inv_idx):
            groups.setdefault(g, []).append(i)
        # only keep groups with >1 element
        same_para_groups = [idx_list for idx_list in groups.values()]
        new_edge_coords = []

        for group in same_para_groups:
            d_norm_group = d_norm[group[0]]
            p0_group = p0[group[0]]
            t_group = np.zeros((len(group), 2))  # to store t values for each edge in the group
            t_group[:, 0] = np.dot(p0[group, :] - p0_group, d_norm_group)
            t_group[:, 1] = np.dot(p1[group, :] - p0_group, d_norm_group)
            events = []
            for t in t_group:
                if t[0] < t[1]:
                    events.append((t[0], 1))
                    events.append((t[1], -1))
                else:
                    events.append((t[1], 1))
                    events.append((t[0], -1))
            events.sort(key=lambda x: (x[0], -x[1]))
            count = 0
            prev = None
            raw_results = []

            for point, delta in events:
                if prev is not None and point > prev:
                    if count == 1:
                        raw_results.append([prev, point])
                count += delta
                prev = point

            if not raw_results:
                continue
            else:
                merged = [raw_results[0]]
                for curr in raw_results[1:]:
                    last = merged[-1]
                    if abs(last[1] - curr[0]) < 1e-6:
                        last[1] = curr[1]  # merge segments
                    else:
                        merged.append(curr)
                for segment in merged:
                    pa = p0_group + segment[0] * d_norm_group
                    pb = p0_group + segment[1] * d_norm_group
                    new_edge_coords.append(np.array([pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]]))

        new_edge_coords = np.array(new_edge_coords, dtype=np.float32)
        return new_edge_coords


class RansacEdgeCylinder(EdgeCylinder):
    """Detects sharp edges in a mesh using the Laplacian operator."""

    def __init__(self, cfg: RansacEdgeCylinderCfg):
        self.cfg: RansacEdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        """Process the edge coordinates using ransac.

        Args:
            edge_coords: The edge coordinates array of shape (num_edges, 6).

        Returns:
            A np array of processed edge coordinates.
        """
        # extract all start and end points
        endpoints = np.vstack((edge_coords[:, :3], edge_coords[:, 3:]))
        unique_endpoints = np.unique(endpoints, axis=0)
        max_iters = self.cfg.max_iter
        thresh = self.cfg.point_distance_threshold

        db = DBSCAN(eps=self.cfg.cluster_eps, min_samples=2, metric="euclidean")
        labels = db.fit_predict(unique_endpoints)

        # collect groups
        groups = [unique_endpoints[labels == lbl] for lbl in np.unique(labels)]
        # remove the first cluster (group 0)
        # groups = groups[1:]

        all_segments = []

        cfg_dict = dict(
            max_iter=self.cfg.max_iter,
            point_distance_threshold=self.cfg.point_distance_threshold,
            min_points=self.cfg.min_points,
        )
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as executor:
            futures = {executor.submit(_fit_segments_for_group, grp, cfg_dict): i for i, grp in enumerate(groups)}
            for future in as_completed(futures):
                segs = future.result()
                if segs:
                    all_segments.extend(segs)

        return np.array(all_segments)


class GreedyconcatEdgeCylinder(EdgeCylinder):
    """Detects sharp edges in a mesh using the Laplacian operator."""

    def __init__(self, cfg: GreedyconcatEdgeCylinderCfg):
        self.cfg: GreedyconcatEdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        line_pts = edge_coords.reshape(-1, 3)
        V, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
        E_pairs = inv_idx.reshape(-1, 2)

        adj_list = {i: set() for i in range(V.shape[0])}
        for u, v in E_pairs:
            if u != v:
                adj_list[u].add(v)
                adj_list[v].add(u)

        num_edges_V = np.array([len(adj_list[i]) for i in range(V.shape[0])], dtype=int)
        available_edges = set(np.where(num_edges_V > 0)[0])

        cos_threshold = np.cos(np.deg2rad(self.cfg.adjacent_angle_threshold))

        processed_edge_coords = []

        def compute_max_distance_to_line_vec(V, v_set):
            A = V[v_set[0]]
            B = V[v_set[-1]]
            AB = B - A
            norm_AB = np.linalg.norm(AB)
            if norm_AB == 0:
                return v_set[0], 0.0
            pts = V[v_set]
            dists = np.linalg.norm(np.cross(pts - A, pts - B), axis=1) / norm_AB
            max_idx = np.argmax(dists)
            return v_set[max_idx], dists[max_idx]

        while available_edges:
            selected_vertex = random.choice(list(available_edges))
            v_set = [selected_vertex]

            if adj_list[selected_vertex]:
                neighbor = next(iter(adj_list[selected_vertex]))
                v_set.append(neighbor)
                adj_list[selected_vertex].remove(neighbor)
                adj_list[neighbor].remove(selected_vertex)
                for vid in [selected_vertex, neighbor]:
                    num_edges_V[vid] -= 1
                    if num_edges_V[vid] == 0:
                        available_edges.discard(vid)

            while True:
                find_neighbor = False
                start, end = v_set[0], v_set[-1]

                neighbors = list(adj_list[start])
                if neighbors:
                    dirs = V[start] - V[neighbors]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    start_dir = V[v_set[1]] - V[start]
                    start_dir /= np.linalg.norm(start_dir)
                    dots = dirs @ start_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.insert(0, n)
                        adj_list[start].remove(n)
                        adj_list[n].remove(start)
                        for vid in [start, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True

                neighbors = list(adj_list[end])
                if neighbors:
                    dirs = V[neighbors] - V[end]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    end_dir = V[end] - V[v_set[-2]]
                    end_dir /= np.linalg.norm(end_dir)
                    dots = dirs @ end_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.append(n)
                        adj_list[end].remove(n)
                        adj_list[n].remove(end)
                        for vid in [end, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True

                if not find_neighbor:
                    break

            while len(v_set) >= self.cfg.min_points:
                for i in range(len(v_set) - 1):
                    max_vi, max_dist = compute_max_distance_to_line_vec(V, v_set[i:])
                    if max_dist < 0.05:
                        break
                if len(v_set) - i >= self.cfg.min_points:
                    processed_edge_coords.append(np.concatenate([V[v_set[i]], V[v_set[-1]]]))
                v_set = v_set[: i + 1]

        return np.array(processed_edge_coords, dtype=np.float32)


class RayEdgeCylinder(VirtualObstacleBase):
    """class for ray-based edge detectors."""

    def __init__(self, cfg: RayEdgeCylinderCfg):
        self.cfg: RayEdgeCylinderCfg = cfg

    def generate(self, mesh: trimesh.Trimesh, device="cpu") -> None:
        """Detect sharp edges in the mesh and store the edge cylinder as virtual obstacle.

        Args:
            mesh: The trimesh object to analyze.

        Returns:
            A np array batch indicating the edges: (num_edges, 6)
            - x, y, z coordinates of the edge start point
            - x, y, z coordinates of the edge end point
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        # extract vertices and faces from the trimesh object
        points = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.int32)
        wp_mesh = convert_to_warp_mesh(points, indices, device="cuda")

        min_bound, max_bound = mesh.bounds
        mesh_size = max_bound - min_bound  # ndarray of shape (3,)

        pattern_cfg = self.cfg.ray_pattern
        pattern_cfg.size = [
            min(mesh_size[:2]) - self.cfg.ray_offset_pos[-1],
            min(mesh_size[:2]) - self.cfg.ray_offset_pos[-1],
        ]

        # define the eight rotation axes
        axes = torch.tensor(
            self.cfg.ray_rotate_axes,
            device=self.device,
        )
        axes = axes / torch.norm(axes, dim=-1, keepdim=True)
        camera_count = axes.shape[0]
        angles = torch.tensor(self.cfg.ray_rotate_angle, device=self.device)
        quats = math_utils.quat_from_angle_axis(angles, axes)
        offset_pos = torch.tensor(self.cfg.ray_offset_pos, device=self.device)

        # sample rays in local camera frame
        ray_starts, ray_directions = pattern_cfg.func(pattern_cfg, self.device)
        num_rays = ray_directions.shape[0]

        # rotate each ray direction by each camera quaternion
        # quats_rep shape: (camera_count * num_rays, 4)
        quats_rep = quats.unsqueeze(1).repeat(1, num_rays, 1).view(-1, 4)
        dirs_rep = ray_directions.repeat(camera_count, 1)
        rotated_dirs = math_utils.quat_apply(quats_rep, dirs_rep)
        ray_directions_w = rotated_dirs.view(camera_count, num_rays, 3)

        # apply the same translation to all ray starts and replicate per camera
        ray_starts_w = (ray_starts + offset_pos).repeat(camera_count, 1, 1)
        ray_hits_w, ray_depth, ray_normal, _ = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            mesh=wp_mesh,
            max_dist=1e6,
            return_distance=True,
            return_normal=True,
        )
        # apply the maximum distance after the transformation
        distance_to_image_plane = ray_depth

        # replace NaNs and infs by linear interpolation in torch
        flat = distance_to_image_plane.flatten()
        idx = torch.arange(flat.numel(), device=flat.device)
        valid = torch.isfinite(flat)
        if (~valid).any():
            # fallback to NumPy interpolation since torch.interp may not exist
            interp_np = np.interp(
                idx.cpu().numpy(),
                idx[valid].cpu().numpy(),
                flat[valid].cpu().numpy(),
            )
            interp = torch.from_numpy(interp_np).to(flat.device)
            flat = interp
        distance_to_image_plane = flat.view_as(distance_to_image_plane)
        distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_ray_depth
        distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_ray_depth)
        depth_image = distance_to_image_plane.view(
            -1,
            int((pattern_cfg.size[0] + 1e-9) / pattern_cfg.resolution) + 1,
            int((pattern_cfg.size[1] + 1e-9) / pattern_cfg.resolution) + 1,
            1,
        )

        # prepare lists to collect edges for all cameras
        depth_edges_list = []
        normal_edges_list = []

        # reshape normal data once
        normal_image = ray_normal.view(
            camera_count,
            int((pattern_cfg.size[0] + 1e-9) / pattern_cfg.resolution) + 1,
            int((pattern_cfg.size[1] + 1e-9) / pattern_cfg.resolution) + 1,
            3,
        )

        with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:
            # Submit tasks for each camera
            futures = [
                executor.submit(process_camera_edges, i, depth_image, normal_image, self.cfg)
                for i in range(camera_count)
            ]

            # Collect the results as they complete
            for future in as_completed(futures):
                de, ne = future.result()
                depth_edges_list.append(de)
                normal_edges_list.append(ne)
        # stack into arrays of shape (camera_count, H, W)
        depth_edges = np.stack(depth_edges_list, axis=0)
        normal_edges = np.stack(normal_edges_list, axis=0)

        ray_hit_image = ray_hits_w.view(
            -1,
            int((pattern_cfg.size[0] + 1e-9) / pattern_cfg.resolution) + 1,
            int((pattern_cfg.size[1] + 1e-9) / pattern_cfg.resolution) + 1,
            3,
        )
        points_list = [None] * camera_count
        ray_hits_flat = ray_hit_image.reshape(camera_count, -1, 3).cpu().numpy()

        def process_camera(i):
            hits = ray_hits_flat[i]
            valid = np.isfinite(hits).all(axis=1)
            combined_mask = ((depth_edges[i] > 0) ^ (normal_edges[i] > 0)).flatten()
            mask = valid & combined_mask
            return hits[mask]

        with ThreadPoolExecutor(max_workers=min(camera_count, os.cpu_count() - 1)) as executor:
            futures = {executor.submit(process_camera, i): i for i in range(camera_count)}
            for future in as_completed(futures):
                i = futures[future]
                points_list[i] = future.result()

        # concatenate all points from all cameras
        points_list = np.concatenate(points_list, axis=0)
        # filter out points on the ground
        points_list = points_list[points_list[:, 2] >= self.cfg.cutoff_z_height]
        self.points_list = torch.tensor(points_list, device=self.device)

        db = DBSCAN(eps=self.cfg.cluster_eps, min_samples=2, metric="euclidean", n_jobs=8)
        labels = db.fit_predict(points_list)

        # collect groups
        groups = [points_list[labels == lbl] for lbl in np.unique(labels)]

        all_segments = []

        cfg_dict = dict(
            max_iter=self.cfg.max_iter,
            point_distance_threshold=self.cfg.point_distance_threshold,
            min_points=self.cfg.min_points,
        )
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:
            futures = {executor.submit(_fit_segments_for_group, grp, cfg_dict): i for i, grp in enumerate(groups)}
            for future in as_completed(futures):
                segs = future.result()
                if segs:
                    all_segments.extend(segs)

        edge_end_points = np.array(all_segments, dtype=np.float32).reshape(-1, 6)
        self.edges_pyt = torch.tensor(edge_end_points, dtype=torch.float32, device=self.device)
        if torch.numel(self.edges_pyt) == 0:
            print("[WARNING] No sharp edges detected.")
        else:
            print(f"Detected {edge_end_points.shape[0]} edges after processing.")
        # create a cylinder spatial grid for the edges and for penetration offset computation
        if edge_end_points.size > 0:
            self.cylinders = CylinderSpatialGrid(
                cylinders=np.concatenate(
                    [
                        edge_end_points,
                        np.ones_like(edge_end_points[:, :1]) * self.cfg.cylinder_radius,
                    ],
                    axis=1,
                ),
                num_grid_cells=self.cfg.num_grid_cells,
                device=self.device,
            )
        else:
            self.cylinders = None

    def disable_visualizer(self):
        if hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer.set_visibility(False)
        if hasattr(self, "_points_visualizer"):
            self._points_visualizer.set_visibility(False)

    def visualize(self):
        if self.edges_pyt.numel() == 0:
            return

        if not hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer = VisualizationMarkers(self.cfg.visualizer)
            self._cylinder_rotate_y_90 = math_utils.quat_from_angle_axis(
                angle=torch.tensor([np.pi / 2], device=self.device),
                axis=torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
            )  # shape (1, 4)

        trans = (self.edges_pyt[:, :3] + self.edges_pyt[:, 3:6]) / 2
        # compute the direction quaternion
        direction = self.edges_pyt[:, 3:6] - self.edges_pyt[:, :3]
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)  # arrow-direction
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
        quat = math_utils.quat_from_angle_axis(
            angle,
            axis,
        )
        quat = math_utils.quat_mul(quat, self._cylinder_rotate_y_90.expand(quat.shape[0], -1))
        # compute the scale to match the length and the edge thickness.
        scales = torch.ones(len(self.edges_pyt), 3, device=self.device)
        scales[:, 0] = self.cfg.cylinder_radius
        scales[:, 1] = self.cfg.cylinder_radius
        scales[:, 2] = torch.norm(direction, dim=-1)
        self._cylinder_visualizer.visualize(
            translations=trans,
            orientations=quat,
            scales=scales,
        )

        if self.points_list.numel() == 0:
            return

        if not hasattr(self, "_points_visualizer"):
            self._points_visualizer = VisualizationMarkers(self.cfg.points_visualizer)

        self._points_visualizer.visualize(
            translations=self.points_list,
        )
        self._cylinder_visualizer.set_visibility(True)
        self._points_visualizer.set_visibility(True)

    def get_points_penetration_offset(self, points):
        return (
            self.cylinders.get_points_penetration_offset(points)
            if self.cylinders is not None
            else torch.zeros_like(points, device=self.device)
        )


class FeatureEdgeCylinder(VirtualObstacleBase):
    """class for feature-extracted edge detectors."""

    def __init__(self, cfg: FeatureEdgeCylinderCfg):
        self.cfg: FeatureEdgeCylinderCfg = cfg

    def generate(self, mesh: trimesh.Trimesh, device="cpu") -> None:
        """Detect sharp edges in the mesh and store the edge cylinder as virtual obstacle.

        Args:
            mesh: The trimesh object to analyze.

        Returns:
            A np array batch indicating the edges: (num_edges, 6)
            - x, y, z coordinates of the edge start point
            - x, y, z coordinates of the edge end point
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        # extract vertices and faces from the trimesh object
        points = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.int32)
        pv_mesh = pv.PolyData(points, np.hstack((np.full((indices.shape[0], 1), 3), indices)))
        edges = pv_mesh.extract_feature_edges(
            feature_angle=self.cfg.feature_angle,
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=True,
            manifold_edges=False,
        )
        edge_end_points = np.empty((0, 6), dtype=np.float32)
        if edges.n_cells > 0:
            lines = edges.lines
            points = edges.points
            edge_coords = []
            i = 0
            while i < len(lines):
                num_pts = int(lines[i])
                if num_pts == 2:
                    pt_idx1 = int(lines[i + 1])
                    pt_idx2 = int(lines[i + 2])
                    start = points[pt_idx1]
                    end = points[pt_idx2]
                    edge_coords.append(np.concatenate([start, end]))
                i += num_pts + 1

            if edge_coords:
                edge_end_points = np.array(edge_coords, dtype=np.float32)
                print(f"Detected {edge_end_points.shape[0]} edges from feature extraction.")
            else:
                print("[WARNING] No edges extracted from features.")
        else:
            print("[WARNING] No sharp edges detected.")

        self.edges_pyt = torch.tensor(edge_end_points, dtype=torch.float32, device=self.device)

        # create a cylinder spatial grid for the edges and for penetration offset computation
        if edge_end_points.size > 0:
            self.cylinders = CylinderSpatialGrid(
                cylinders=np.concatenate(
                    [
                        edge_end_points,
                        np.ones_like(edge_end_points[:, :1]) * self.cfg.cylinder_radius,
                    ],
                    axis=1,
                ),
                num_grid_cells=self.cfg.num_grid_cells,
                device=self.device,
            )
        else:
            self.cylinders = None

    def disable_visualizer(self):
        if hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer.set_visibility(False)

    def visualize(self):
        if self.edges_pyt.numel() == 0:
            return

        if not hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer = VisualizationMarkers(self.cfg.visualizer)
            self._cylinder_rotate_y_90 = math_utils.quat_from_angle_axis(
                angle=torch.tensor([np.pi / 2], device=self.device),
                axis=torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
            )  # shape (1, 4)

        trans = (self.edges_pyt[:, :3] + self.edges_pyt[:, 3:6]) / 2
        # compute the direction quaternion
        direction = self.edges_pyt[:, 3:6] - self.edges_pyt[:, :3]
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)  # arrow-direction
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
        quat = math_utils.quat_from_angle_axis(
            angle,
            axis,
        )
        quat = math_utils.quat_mul(quat, self._cylinder_rotate_y_90.expand(quat.shape[0], -1))
        # compute the scale to match the length and the edge thickness.
        scales = torch.ones(len(self.edges_pyt), 3, device=self.device)
        scales[:, 0] = self.cfg.cylinder_radius
        scales[:, 1] = self.cfg.cylinder_radius
        scales[:, 2] = torch.norm(direction, dim=-1)
        self._cylinder_visualizer.visualize(
            translations=trans,
            orientations=quat,
            scales=scales,
        )
        self._cylinder_visualizer.set_visibility(True)

    def get_points_penetration_offset(self, points):
        return (
            self.cylinders.get_points_penetration_offset(points)
            if self.cylinders is not None
            else torch.zeros_like(points, device=self.device)
        )


def process_camera_edges(i, depth_image, normal_image, cfg):
    # Depth edges for camera i
    depth_i = depth_image[i, ..., 0].cpu().numpy()
    depth_norm = cv2.normalize(depth_i, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype("uint8")
    de = cv2.Canny(depth_uint8, cfg.depth_canny_thresholds[0], cfg.depth_canny_thresholds[1])

    # Normal edges for camera i
    norm0 = normal_image[i].cpu().numpy()  # (H, W, 3)
    normal_vis = ((norm0 + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
    normal_bgr = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
    mask = cv2.inRange(normal_bgr, (250, 250, 250), (255, 255, 255))
    normal_bgr = cv2.inpaint(normal_bgr, mask, 3, cv2.INPAINT_TELEA)
    gray_n = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2GRAY)
    ne = cv2.Canny(gray_n, cfg.normal_canny_thresholds[0], cfg.normal_canny_thresholds[1])

    return de, ne


def _fit_segments_for_group(grp, cfg_dict):
    max_iters = cfg_dict["max_iter"]
    thresh = cfg_dict["point_distance_threshold"]
    min_points = cfg_dict["min_points"]

    all_segments = []
    remaining = grp.copy()

    while True:
        best_model = None
        best_inliers_idx = []

        num_remaining = len(remaining)
        if num_remaining < min_points:
            break

        max_pairs = num_remaining * (num_remaining - 1) // 2
        n_iters = min(max_iters, max_pairs)
        for _ in range(n_iters):
            i1, i2 = np.random.choice(len(remaining), 2, replace=False)
            p1, p2 = remaining[i1], remaining[i2]
            v = p2 - p1
            n = np.linalg.norm(v)
            if n == 0:
                continue
            v_unit = v / n
            dists = np.linalg.norm(np.cross(remaining - p1, v_unit), axis=1)
            inliers_idx = np.where(dists < thresh)[0]

            if inliers_idx.size > len(best_inliers_idx):
                best_inliers_idx = inliers_idx
                best_model = (p1, v_unit)

        if best_model is None or len(best_inliers_idx) < min_points:
            break

        p1, v_unit = best_model
        inlier_pts = remaining[best_inliers_idx]
        t = np.dot(inlier_pts - p1, v_unit)
        tmin, tmax = t.min(), t.max()
        seg_start = p1 + tmin * v_unit
        seg_end = p1 + tmax * v_unit
        all_segments.append((seg_start, seg_end))

        mask = np.ones(remaining.shape[0], dtype=bool)
        mask[best_inliers_idx] = False
        remaining = remaining[mask]

    return all_segments
