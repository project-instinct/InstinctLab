import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as mdp
import instinctlab.terrains as terrain_gen
from instinctlab.assets.unitree_g1 import beyondmimic_action_scale
from instinctlab.managers import MultiRewardCfg
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.sensors import Grid3dPointsGeneratorCfg, NoisyGroupedRayCasterCameraCfg, VolumePointsCfg
from instinctlab.terrains import GreedyconcatEdgeCylinderCfg, TerrainImporterCfg
from instinctlab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    RandomGaussianNoiseCfg,
    RangeBasedGaussianNoiseCfg,
)

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

##
# Scene definition
##
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "perlin_rough": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.05,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "perlin_rough_stand": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.05,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "square_gaps": terrain_gen.PerlinSquareGapTerrainCfg(
            proportion=0.10,
            gap_distance_range=(0.1, 0.7),
            gap_depth=(0.4, 0.6),
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_high": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv_high": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "boxes": terrain_gen.PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            num_obstacles=20,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.8, 1.5),
            obstacle_height_range=(0.05, 0.45),
            platform_width=1.5,
            border_width=0.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
        "mesh_boxes": terrain_gen.PerlinMeshRandomMultiBoxTerrainCfg(
            proportion=0.10,
            box_height_mean=[0.1, 0.4],
            box_height_range=0.05,
            box_length_mean=0.4,
            box_length_range=0.1,
            box_width_mean=0.4,
            box_width_range=0.1,
            platform_width=1.5,
            generation_ratio=0.3,
            no_perlin_at_obstacle=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.05),
            },
        ),
        "hf_pyramid_slope_inv": terrain_gen.PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.7),
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.00,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
                ),
            },
        ),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={
            "edges": GreedyconcatEdgeCylinderCfg(
                cylinder_radius=0.05,
                min_points=2,
            ),
        },
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    left_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    right_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    leg_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        points_generator=Grid3dPointsGeneratorCfg(
            x_min=-0.025,
            x_max=0.12,
            x_num=10,
            y_min=-0.03,
            y_max=0.03,
            y_num=5,
            z_min=-0.04,
            z_max=0.0,
            z_num=2,
        ),
        debug_vis=False,
    )
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=[
            "/World/ground",
            # NOTE: Don't forget to add the robot links in robot-specific configuration file.
        ],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),  # fovx
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),  # fovy
            width=64,
            height=36,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(  # G1 Robot head camera nominal pose
            pos=(
                0.0487988662332928,
                0.01,
                0.4378029937970051,
            ),
            rot=(
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ),
            convention="world",
        ),
        min_distance=0.1,
        # noise
        noise_pipeline={
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    motion_reference: MotionReferenceManagerCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=8, flatten_history_dim=True
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=8, flatten_history_dim=True)
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=8, flatten_history_dim=True)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=8, flatten_history_dim=True)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=8, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=8, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=8, flatten_history_dim=True)
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class AmpPolicyStateObsCfg(ObsGroup):
        concatenate_terms = False
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            history_length=10,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    preserve_order=True,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    preserve_order=True,
                ),
            },
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    @configclass
    class AmpReferenceStateObsCfg(ObsGroup):
        concatenate_terms = False
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_reference_as_state,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
            history_length=10,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_reference_as_state,
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )

    # observation group
    policy: PolicyCfg = PolicyCfg()
    # critic group
    critic: CriticCfg = CriticCfg()
    # AMP training groups
    amp_policy: AmpPolicyStateObsCfg = AmpPolicyStateObsCfg()
    amp_reference: AmpReferenceStateObsCfg = AmpReferenceStateObsCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=beyondmimic_action_scale, use_default_offset=True
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05,
        ranges=mdp.PoseVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
        random_velocity_terrain=["perlin_rough_stand"],
        velocity_ranges={
            "perlin_rough": {"lin_vel_x": (0.45, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "square_gaps": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_high": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_high": {
                "lin_vel_x": (0.45, 0.8),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
            "boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "mesh_boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "hf_pyramid_slope_inv": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
        },
        only_positive_lin_vel_x=True,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
        target_dis_threshold=0.4,
    )


@configclass
class G1Rewards:
    """Reward terms for the MDP."""

    # Task rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"})
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.3, params={"command_name": "base_velocity", "offset": 4.0})

    # Regularization rewards
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "vel_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_square,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    pelvis_orientation_l2 = RewTerm(
        func=mdp.link_orientation, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}
    )
    feet_flat_ori = RewTerm(
        func=mdp.feet_orientation_contact,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_at_plane = RewTerm(
        func=mdp.feet_at_plane,
        weight=-0.1,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
            "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "height_offset": 0.035,
        },
    )
    feet_close_xy = RewTerm(
        func=mdp.feet_close_xy_gauss,
        weight=0.4,
        params={
            "threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "std": math.sqrt(0.05),
        },
    )
    energy = RewTerm(
        func=mdp.motors_power_square,
        weight=-5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            "normalize_by_stiffness": True,
        },
    )
    freeze_upper_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.004,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"]
            ),
        },
    )

    # Safety rewards
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*"),
            "threshold": 1.0,
        },
    )


@configclass
class RewardsCfg(MultiRewardCfg):
    rewards: G1Rewards = G1Rewards()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_bound = DoneTerm(func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    root_height = DoneTerm(func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.5})
    dataset_exhausted = DoneTerm(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "print_reason": False,
            "reset_without_notice": True,
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.05, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    register_virtual_obstacles = EventTerm(
        func=instinct_mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={
            "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.15, 0.15),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.tracking_exp_vel, params={"lin_vel_threshold": (0.3, 0.6), "ang_vel_threshold": (0.0, 0.0)}
    )


@configclass
class MonitorCfg:
    pass


##
# Environment configuration
##


@configclass
class ParkourEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**29
        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
