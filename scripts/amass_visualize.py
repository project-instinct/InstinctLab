import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--live_plot", action="store_true", default=False, help="Plot some critical lines alive")
parser.add_argument("--video", type=str, default=None, help="Path to save the video.")
parser.add_argument(
    "--motion_path",
    type=str,
    default=None,
    help=(
        "Directory that contains *.retargeted.npz (already retargeted G1 motion). "
        "When set, SMPL IK is disabled (retargetting_func=None)."
    ),
)
parser.add_argument(
    "--motion_yaml",
    type=str,
    default=None,
    help=(
        "Optional YAML with `selected_files:` entries relative to --motion_path "
        "(same format as AmassMotionCfg.filtered_motion_selection_filepath)."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
from collections import deque

if args_cli.video:
    import imageio.v2 as iio

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass

from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG
from instinctlab.motion_reference import MotionReferenceManager
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg import motion_reference_cfg

# from instinctlab.utils.retarget_smpl_to_joint import retarget_smpl_to_g1_29dof_joints
from instinctlab.utils.humanoid_ik import HumanoidSmplRotationalIK
from instinctlab.utils.live_plotter import LivePlotter

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

VIEWER_CFG = ViewerCfg()
VIEWER_CFG.resolution = (640, 360)

# ratio between the step_dt and sim_dt
DECIMATION = 4


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    clip_joint_ref_to_robot_limits = True
    path = os.path.expanduser("~/Datasets/AMASS/")
    retargetting_func = HumanoidSmplRotationalIK
    retargetting_func_kwargs = dict(
        robot_chain=G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG.spawn.asset_path,
        smpl_root_in_robot_link_name="pelvis",
        translation_scaling=0.75,
        translation_height_offset=0.0,
    )
    filtered_motion_selection_filepath = os.path.expanduser("~/Datasets/AMASS_selections/amass_test_motion_files.yaml")
    motion_start_from_middle_range = [0.0, 0.0]
    buffer_device = "cpu"


def _amass_motion_cfg_for_run():
    """Default AMASS+IK; use --motion_path for pre-retargeted *.npz (Instinct/LAFAN pipeline)."""
    cfg = AmassMotionCfg()
    if args_cli.motion_path:
        cfg = cfg.replace(
            path=os.path.abspath(os.path.expanduser(args_cli.motion_path)),
            retargetting_func=None,
            retargetting_func_kwargs={},
            filtered_motion_selection_filepath=(
                os.path.abspath(os.path.expanduser(args_cli.motion_yaml))
                if args_cli.motion_yaml
                else None
            ),
            ensure_link_below_zero_ground=False,
        )
    return cfg


_AMASS_FOR_SCENE = _amass_motion_cfg_for_run()


@configclass
class SceneCfg(InteractiveSceneCfg):
    # common scene entities
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robots
    robot = G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # motion reference
    motion_reference = motion_reference_cfg.replace(
        robot_model_path=G1_29DOF_TORSOBASE_POPSICLE_SPHEREHAND_CFG.spawn.asset_path,
        frame_interval_s=0.02,
        motion_buffers={
            "amass": _AMASS_FOR_SCENE,
        },
    )

    def __post_init__(self):
        for k, v in self.motion_reference.motion_buffers.items():
            v.motion_bin_length_s = None
            v.motion_start_from_middle_range = (0.0, 0.0)


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    simulation_timestamp = 0
    robot: Articulation = scene["robot"]
    motion_reference: MotionReferenceManager = scene["motion_reference"]

    motion_reference.match_scene(scene)

    # prepare annotator product to record and save video
    if args_cli.video is not None:
        video_writer = iio.get_writer(
            args_cli.video,
            fps=1 / sim_dt / DECIMATION,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        import omni.replicator.core as rep

        _render_product = rep.create.render_product(VIEWER_CFG.cam_prim_path, VIEWER_CFG.resolution)
        _rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        _rgb_annotator.attach([_render_product])
        _video_interval_counter = 0

    # prepare live plots
    if args_cli.live_plot:
        plotter = LivePlotter(keys=["1"] * 12)
        _plotter_counter = 0

    # simulation loop
    while simulation_app.is_running():
        # Write data to sim

        # write robot data based on motion reference
        motion_reference_frame = motion_reference.reference_frame
        # robot.root_physx_view.set_dof_positions(
        #     motion_reference_frame.joint_pos[:, 0],
        #     indices=robot._ALL_INDICES,
        # )
        # robot.root_physx_view.set_dof_velocities(
        #     motion_reference_frame.joint_vel[:, 0],
        #     indices=robot._ALL_INDICES,
        # )
        # robot.root_physx_view.set_root_transforms(
        #     torch.concatenate(
        #         [
        #             motion_reference_frame.base_pos_w[:, 0],
        #             math_utils.convert_quat(motion_reference_frame.base_quat_w[:, 0], to="xyzw"),
        #         ],
        #         dim=-1,
        #     ),
        #     indices=robot._ALL_INDICES,
        # )
        robot.write_root_pose_to_sim(
            torch.concatenate(
                [
                    motion_reference_frame.base_pos_w[:, 0],
                    motion_reference_frame.base_quat_w[:, 0],
                ],
                dim=-1,
            ),
        )
        robot.write_joint_state_to_sim(
            motion_reference_frame.joint_pos[:, 0],
            motion_reference_frame.joint_vel[:, 0],
        )
        robot.write_root_velocity_to_sim(torch.zeros(robot.num_instances, 6, device=torch.device("cuda")))

        # reset motion reference if motion reference is exhausted
        reset_mask = torch.logical_not(motion_reference.data.validity.any(dim=-1))
        if reset_mask.any():
            reset_env_ids = torch.where(reset_mask)[0]
            motion_reference.reset(reset_env_ids)
            print("[INFO] current motion", motion_reference.get_current_motion_identifiers(reset_env_ids))

        # Perform render not physics steps
        sim.render()
        # Update buffers
        scene.update(sim_dt)

        if args_cli.video is not None:
            if _video_interval_counter % DECIMATION == 0:
                # obtain the rgb data
                rgb_data = _rgb_annotator.get_data()
                rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
                if rgb_data.size == 0:
                    rgb_data = np.zeros((VIEWER_CFG.resolution[1], VIEWER_CFG.resolution[0], 3), dtype=np.uint8)
                else:
                    rgb_data = rgb_data[:, :, :3]
                # write to video
                video_writer.append_data(rgb_data)
            _video_interval_counter += 1

        if args_cli.live_plot:
            if _plotter_counter % DECIMATION == 0:
                _, robot_pitch, robot_yaw = math_utils.euler_xyz_from_quat(motion_reference_frame.base_quat_w[:, 0])
                base_ang_vel = motion_reference_frame.base_ang_vel_w[:, 0]
                joint_vel = motion_reference_frame.joint_vel[:, 0, :12]
                joint_pos = motion_reference_frame.joint_pos[:, 0, 9]
                plotter.plot(
                    [
                        # robot_pitch[0].item(),
                        i
                        for i in joint_vel[0].cpu().numpy()
                        # joint_pos[0].item(),
                        # base_ang_vel[0, 2].item(),
                        # robot_yaw[0].item(),
                    ],
                    x=simulation_timestamp,
                )
            _plotter_counter += 1

        simulation_timestamp += sim_dt

    if args_cli.video is not None:
        video_writer.close()
        print("[INFO] video saved to: ", args_cli.video)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)
    # close sim app
    simulation_app.close()


if __name__ == "__main__":
    # run the main execution
    main()
