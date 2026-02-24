import torch
from dataclasses import MISSING
from typing import Callable, Optional

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .noise_model import (
    ImageNoiseModel,
    LatencyNoiseModel,
    SensorDeadNoiseModel,
    blind_spot_noise,
    crop_and_resize,
    depth_artifact_noise,
    depth_contour_noise,
    depth_normalization,
    depth_sky_artifact_noise,
    depth_stero_noise,
    gaussian_blur_noise,
    random_gaussian_noise,
    range_based_gaussian_noise,
    stereo_too_close_noise,
)


@configclass
class ImageNoiseCfg(NoiseCfg):
    func: Callable[[torch.Tensor, NoiseCfg, torch.Tensor], torch.Tensor] | type[ImageNoiseModel] = ImageNoiseModel
    """ The callable function to apply noise to the image.
    The function should take two arguments:
        - the image in shape (N_, H, W, C) where N_ = len(env_ids)
        - the cfg object (as this configclass's object).
        - the env_ids tensor for specifying the environment ids
    """
    device: str | torch.device = "cpu"


# -- below are configuration classes to different types of noise applied to depth image


@configclass
class DepthContourNoiseCfg(ImageNoiseCfg):
    contour_threshold: float = 2.0
    maxpool_kernel_size: int = 1
    func = depth_contour_noise
    """ The noise model class to apply depth contour noise. """


@configclass
class DepthArtifactNoiseCfg(ImageNoiseCfg):
    artifacts_prob: float = 0.0001  # should be very low
    artifacts_height_mean_std: list[float] = [2, 0.5]
    artifacts_width_mean_std: list[float] = [2, 0.5]
    noise_value: float = 0.0
    func = depth_artifact_noise


@configclass
class DepthSteroNoiseCfg(ImageNoiseCfg):
    stero_far_distance: float = 2.0
    stero_min_distance: float = 0.12  # when using (240,424) resolution
    stero_far_noise_std: float = 0.08  # the noise std of pixels that are greater than stero_far_noise_distance
    stero_near_noise_std: float = 0.02  # the noise std of pixels that are less than stero_far_noise_distance

    stero_full_block_artifacts_prob: float = (
        0.001  # the probability of adding artifacts to pixels that are less than stero_min_distance
    )
    stero_full_block_values: list = [0.0, 0.25, 0.5, 1.0, 3.0]
    stero_full_block_height_mean_std: list = [62, 1.5]
    stero_full_block_width_mean_std: list = [3, 0.01]

    stero_half_block_spark_prob: float = 0.02
    stero_half_block_value: int = 3000  # to the maximum value directly

    func = depth_stero_noise


@configclass
class DepthSkyArtifactNoiseCfg(ImageNoiseCfg):
    sky_artifacts_prob: float = 0.0001
    sky_artifacts_far_distance: float = 2.0  # pixels greater than this distance will be viewed as sky
    sky_artifacts_values: list = [0.6, 1.0, 1.2, 1.5, 1.8]
    sky_artifacts_height_mean_std: list = [2, 3.2]
    sky_artifacts_width_mean_std: list = [2, 3.2]

    func = depth_sky_artifact_noise


@configclass
class LatencyNoiseCfg(ImageNoiseCfg):
    history_length: int = 5

    # sample frequency related settings

    sample_frequency: Optional[str] = None
    """Optional frequency setting for resampling delays
        - None (default): no resampling
        - "every_n_steps": resample every n steps, with n specified by `sample_frequency_steps`
        - "random_with_probability": resample with a certain probability, specified by `sample_probability`
    """
    sample_frequency_steps: int = 50  # used when sample_frequency is "every_n_steps"
    sample_frequency_steps_offset: int = 5  # the offset for the sample frequency steps

    sample_probability: float = 0.1  # used when sample_frequency is "random_with_probability"

    # sample distribution related settings

    latency_distribution: Optional[str] = "constant"
    """Optional distribution for sampling latency steps
        - "uniform": (uniform distribution), with range specified by 'latency_range'
        - "normal": normal distribution, with mean and std specified by 'latency_mean_std', and range specified by 'latency_range'
        - "choice": choose from a predefined list of latency steps, specified by 'latency_choices'
        - "constant" (default): use a fixed number of steps, specified by 'latency_steps'
    """
    latency_range: tuple[int, int] = (1, history_length)

    latency_mean_std: tuple[float, float] = (3, 1)  # used when latency_distribution is "normal"

    latency_choices: list[int] = [1, 2, 3, 4, 5]  # used when latency_distribution is "choice"
    latency_choices_probabilities: Optional[list[float]] = (
        None  # probabilities for each choice, default to None (uniform distribution)
    )
    # The number of probabilities must match the number of choices.

    latency_steps: int = 5  # used when latency_distribution is "constant"

    func: type[LatencyNoiseModel] = LatencyNoiseModel


@configclass
class DepthNormalizationCfg(ImageNoiseCfg):
    """Configuration for normalizing depth values to a specific range."""

    depth_range: tuple[float, float] = (0.0, 10.0)
    """Depth value range for normalization."""

    normalize: bool = True
    """Whether to normalize depth values to the range [0, 1] after clipping."""

    output_range: tuple[float, float] = (0.0, 1.0)
    """Range to normalize depth values to."""

    func = depth_normalization
    """The noise model class to apply depth normalization."""


@configclass
class CropAndResizeCfg(ImageNoiseCfg):
    """Configuration for cropping and resizing images."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    """The size to be cropped, corresponding to up, down, left, right, respectively."""

    resize_shape: tuple[int, int] = None
    """The size to be reshape to, corresponding to height, width, respectively."""

    func = crop_and_resize


@configclass
class BlindSpotNoiseCfg(ImageNoiseCfg):
    """Configuration for adding blind spot noise (zeroing out regions of the image)."""

    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    """The size to be blind spotted, corresponding to up, down, left, right, respectively."""

    func = blind_spot_noise


@configclass
class GaussianBlurNoiseCfg(ImageNoiseCfg):
    """Configuration for adding Gaussian blur noise to images."""

    kernel_size: int = 3
    """The size of the Gaussian kernel. It should be an odd number."""

    sigma: float = 1.0
    """The standard deviation of the Gaussian kernel."""

    func = gaussian_blur_noise


@configclass
class RandomGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding random Gaussian noise to images."""

    probability: float = 0.1
    """The probability of applying the Gaussian noise."""

    noise_mean: float = 0.0
    """The mean of the Gaussian noise."""

    noise_std: float = 1.0
    """The standard deviation of the Gaussian noise."""

    func = random_gaussian_noise


@configclass
class RangeBasedGaussianNoiseCfg(ImageNoiseCfg):
    """Configuration for adding range-based Gaussian noise to images."""

    min_value: float | None = None
    """The minimum value of the range."""

    max_value: float | None = None
    """The maximum value of the range."""

    noise_std: float = 1.0
    """The standard deviation of the Gaussian noise."""

    func = range_based_gaussian_noise


@configclass
class StereoTooCloseNoiseCfg(ImageNoiseCfg):
    """Configuration for adding stereo too close noise to images."""

    close_threshold: float = 0.12
    """The threshold of the too close distance."""

    # full block related settings
    full_block_height_mean_std: tuple[float, float] = (62, 1.5)
    full_block_width_mean_std: tuple[float, float] = (3, 0.01)
    full_block_values: list[float] = [0.0, 0.25, 0.5, 1.0, 3.0]
    full_block_artifacts_prob: float = 0.008

    # half block related settings
    half_block_height_mean_std: tuple[float, float] = (2, 3.2)
    half_block_width_mean_std: tuple[float, float] = (2, 3.2)
    half_block_value: float = 30  # to the maximum value directly
    half_block_spark_prob: float = 0.02

    func = stereo_too_close_noise


@configclass
class SensorDeadNoiseCfg(ImageNoiseCfg):
    """Configuration for adding sensor dead behavior, which might be autonomous restarted.
    Thus causing some frames of non-refreshed data.
    """

    dead_probability: float = 0.01
    """The probability of the sensor dead."""

    dead_frames: int | list[int] = 90  # 1.5 second at 60Hz
    """The number of frames to be non-refreshed (before the sensor is restarted).
    Can be a single number or a list of numbers to be uniformly selected from.
    """

    func = SensorDeadNoiseModel
