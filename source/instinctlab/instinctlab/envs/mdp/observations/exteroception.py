from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import cv2

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import (  # This could be dangerous for code maintainability. Maybe optimize this import later.
    _randomize_prop_by_op,
)
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import Camera, TiledCamera
    from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera
    from isaaclab.sensors.ray_caster.base_ray_caster_camera import BaseRayCasterCamera


def _debug_visualize_image(
    image: torch.Tensor,
    scale_up_vis: int = 5,
    window_name: str = "vis_image",
) -> None:
    """Visualize images in a cv2 window for debugging purposes.

    This function normalizes images to [0, 255], handles different channel configurations,
    scales them up for better visualization, and displays them in an OpenCV window.

    Args:
        images: Image tensor in shape (H, W)
        scale_up_vis: The factor to scale up the image for better visualization if the
            resolution is too low. Defaults to 5.
        window_name: The name of the OpenCV window. Defaults to "vis_image".
    """
    # automatically normalize images to [0, 255]
    img = (image * 255.0 / image.max()).cpu().numpy().astype("uint8")  # (H, W)
    # Scale up the image for better visualization
    img = cv2.resize(img, (img.shape[1] * scale_up_vis, img.shape[0] * scale_up_vis), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)


def visualizable_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    debug_vis: bool = False,
    scale_up_vis: int = 5,
    history_skip_frames: int = 0,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.
        debug_vis: Whether to visualize the images in an animated cv2 window. Defaults to False.
        scale_up_vis: The factor to scale up the image for better visualization if the resolution is too low. Defaults to 5.
        history_skip_frames: The number of frames to skip to downsample the history data.
            For example, if the input sequence is 0,1,2,3,4,5,6; and history_skip_frames is 2,
            the output sequence will be 0,3,6. NOTE: This is only supported when 'history' is in the data_type.
            Defaults to 0. NOTE: It is recommended to set the camera update_period to env_step_dt.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | BaseRayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type].torch.clone()  # (N, H, W, C) or history
    if "history" in data_type:
        # NOTE: Only depth-related data types with history are supported. where C = 1.
        images = images.squeeze(
            -1
        )  # (N, history, H, W, C) -> (N, history, H, W), images[:, -1] shall be the latest frame.
        if history_skip_frames > 0:
            images = images[:, ::history_skip_frames, :, :]
    else:
        images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    # rgb/depth image normalization
    # NOTE: rgb image is not tested yet.

    if debug_vis:
        # (N, C, H, W) -> (C, H, N, W) -> (C*H, N*W)
        _debug_visualize_image(
            images.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2), scale_up_vis
        )

    return images


class delayed_visualizable_image(ManagerTermBase):
    """A callable class that could sample delayed images from camera sensor that has history data. This is initially
    designed to use NoisyGroupedRayCasterCamera. The output shape will always be (N, num_output_frames, H, W) for now.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        self.data_type = cfg.params["data_type"]  # must provide the data_type, must have "history" in the data_type
        assert "history" in self.data_type, "data_type must have 'history' in it"
        self.sensor: BaseMultiMeshRayCasterCamera = env.scene.sensors[self.sensor_cfg.name]
        self.delayed_frame_ranges = cfg.params.get("delayed_frame_ranges", (0, 0))  # (min_delay, max_delay)
        # not recommended for gaussian distribution, but it is supported.
        self.delayed_frame_distribution: Literal["uniform", "log_uniform"] = cfg.params.get(
            "delayed_frame_distribution", "uniform"
        )
        self._num_delayed_frames = torch.zeros(env.num_envs, device=env.device)  # depending on the sensor update period
        self.history_skip_frames = max(cfg.params.get("history_skip_frames", 1), 1)
        # if greater than 0, the output data from this observation term will have history dimension, else no history dimension.
        self.num_output_frames = max(cfg.params.get("num_output_frames", 0), 1)
        sensor_output = self.sensor.data.output[self.data_type].torch
        assert len(sensor_output.shape) >= 5, (
            f"sensor data of type {self.data_type} should have (N, history, H, W, C) shape, but got"
            f" {sensor_output.shape}"
        )
        self.sensor_history_length = sensor_output.shape[1]

        # build frame offset based on num_output_frames and history_skip_frames
        # use reverse order because [:, -1] gets the latest frame in sensor data. frame_offset[0] should be the largest
        # to return the oldest frame in the output.
        self.frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )  # (num_output_frames,)

        self.check_delay_bounds()

    def check_delay_bounds(self) -> None:
        """
        Check if the delayed frame ranges are within the bounds of the sensor history length.
        If not, raise an error.
        """
        max_delayed_frames = self.delayed_frame_ranges[1]
        frames_needed_if_no_delay = (self.num_output_frames - 1) * self.history_skip_frames + 1
        if (frames_needed_if_no_delay + max_delayed_frames) > self.sensor_history_length:
            raise ValueError(
                "The delayed frame ranges are too large for the sensor history length. The maximum delayed frames is"
                f" {max_delayed_frames}, but the frames needed if no delay is {frames_needed_if_no_delay}, which is"
                f" {frames_needed_if_no_delay + max_delayed_frames}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the delayed frame ranges.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._num_delayed_frames[env_ids] = _randomize_prop_by_op(
            self._num_delayed_frames[env_ids].unsqueeze(-1),
            self.delayed_frame_ranges,
            None,
            slice(None),
            operation="abs",
            distribution=self.delayed_frame_distribution,
        ).squeeze(
            -1
        )  # (N,)

    def __call__(
        self,
        env: ManagerBasedEnv,
        data_type: str,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        history_skip_frames: int = 0,
        num_output_frames: int = 0,
        delayed_frame_ranges: tuple[int, int] = (0, 0),
        delayed_frame_distribution: Literal["uniform", "log_uniform"] = "uniform",
        debug_vis: bool = False,
        scale_up_vis: int = 5,
    ) -> torch.Tensor:
        """
        Get the delayed frames from the sensor data.
        """
        # obtain the input image
        images = self.sensor.data.output[self.data_type].torch.clone()  # (N, history, H, W, C)
        # NOTE: Only depth-related data types with history are supported for now. where C = 1.
        images = images.squeeze(
            -1
        )  # (N, history, H, W, C) -> (N, history, H, W), images[:, -1] shall be the latest frame.
        # get the delayed frames
        frame_indices = (
            self.sensor_history_length - self.frame_offset.unsqueeze(0) - self._num_delayed_frames.unsqueeze(1) - 1
        )  # (N, num_output_frames)
        frame_indices = frame_indices.to(torch.long)
        # final safety check to avoid frame_indices being out of bounds
        assert (frame_indices >= 0).all(), f"frame_indices should be non-negative, but got {frame_indices}"
        assert (
            frame_indices < self.sensor_history_length
        ).all(), f"frame_indices should be less than the sensor history length {self.sensor_history_length}"
        # Use advanced indexing: create batch indices and use them together with frame_indices
        batch_indices = (
            torch.arange(images.shape[0], device=images.device)
            .unsqueeze(1)
            .expand(-1, frame_indices.shape[1])
            .to(torch.long)
        )
        delayed_frames = images[batch_indices, frame_indices]  # (N, num_output_frames, H, W)
        if debug_vis:
            # (N, num_output_frames, H, W) -> (num_output_frames, H, N, W) -> (num_output_frames * H, N * W)
            _debug_visualize_image(
                delayed_frames.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2),
                scale_up_vis,
            )
        return delayed_frames  # still delayed_frames[:, -1] shall be the latest frame.
