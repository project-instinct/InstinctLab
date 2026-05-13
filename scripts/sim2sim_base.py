      

import math
import numpy as np
# import mujoco
# import mujoco.viewer
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import onnxruntime
from motion_utils.motion_loader import MotionLoader, MotionPlayer
from motion_utils.utils import euler_xyz_from_quat, matrix_from_quat


def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    quat_torso = data.sensor('orientation_torso').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    # omega = data.qvel[3:6].astype(np.double)  # Angular velocity in base framed
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec, quat_torso)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg, motion_file):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        motion_file: Path to motion file (.npz).

    Returns:
        None
    """
    # Load motion file
    
    motion_loader = MotionLoader(motion_file, device="cpu")
    motion_dt = motion_loader.get_dt()
    control_dt = cfg.sim_config.dt * cfg.sim_config.decimation
    print(f"Using motion file: {motion_file}")
    print(f"Motion duration: {motion_loader.get_duration():.2f}s, {motion_loader.time_step_total} frames")
    print(f"Motion fps: {motion_loader.fps}, dt: {motion_dt:.4f}s")
    print(f"Sim dt: {cfg.sim_config.dt:.4f}s, control dt: {control_dt:.4f}s")
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    default_dof_pos = np.array([-0.312, 0, 0, 0.669, -0.363, 0,
                                -0.312, 0, 0, 0.669, -0.363, 0,
                                0, 0, 0,
                                0.2, 0.2, 0, 0.6, 0, 0, 0,
                                0.2, -0.2, 0, 0.6, 0, 0, 0,
                            ], dtype=np.double)
    
    lab2mj_indexes = [  0, 3, 6, 9, 13, 17, 
                        1, 4, 7, 10, 14, 18, 
                        2, 5, 8, 
                        11, 15, 19, 21, 23, 25, 27, 
                        12, 16, 20, 22, 24, 26, 28]
    # 逆映射
    mj2lab_indexes = [0] * len(lab2mj_indexes)
    for mj_idx, lab_idx in enumerate(lab2mj_indexes):
        mj2lab_indexes[lab_idx] = mj_idx

    
    data.qpos[0:3] = 0
    data.qpos[2] = 0.78
    data.qpos[3:7] = np.array([1.0, 0, 0, 0.0], dtype=np.double)
    data.qpos[7:7+29] = default_dof_pos

    mujoco.mj_step(model, data)
    # viewer = mujoco.viewer.launch_passive(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 5.0

    num_actions = 29
    task_obs_len = 5
    prop_obs_len = 5
    # Observation buffers matching lab's observation order:
    # 1. command (task_obs_len=5, shape=(5, 29)) - motion command joint positions
    command_obs = torch.zeros(task_obs_len, num_actions)
    # 2. motion_anchor_ori (task_obs_len=5, shape=(5, 6)) - motion anchor orientation (first 2 rows of rotation matrix)
    motion_root_ori_obs = torch.zeros(task_obs_len, 6)
    motion_root_zpos_obs = torch.zeros(task_obs_len, 1)
    motion_root_xylin_vel_obs = torch.zeros(task_obs_len, 2)
    motion_root_ang_zvel_obs = torch.zeros(task_obs_len, 1)
    # 3. projected_gravity (prop_obs_len=5, shape=(5, 3)) - gravity projection in base frame
    base_ori_obs = torch.zeros(prop_obs_len, 6)
    # 4. base_ang_vel (prop_obs_len=5, shape=(5, 3)) - base angular velocity in base frame
    base_ang_vel_obs = torch.zeros(prop_obs_len, 3)
    # 5. joint_pos (prop_obs_len=5, shape=(5, 29)) - joint positions relative to default
    joint_pos_obs = torch.zeros(prop_obs_len, num_actions)
    # 6. joint_vel (prop_obs_len=5, shape=(5, 29)) - joint velocities relative to default
    joint_vel_obs = torch.zeros(prop_obs_len, num_actions)
    # 7. actions (prop_obs_len=5, shape=(5, 29)) - last actions
    actions_obs = torch.zeros(prop_obs_len, num_actions)

    target_q = np.zeros((num_actions), dtype=np.double)
    action = np.zeros((num_actions), dtype=np.double)

    # Fill initial observations before starting the simulation loop
    q_init, dq_init, quat_init, v_init, omega_init, gvec_init, quat_torso_init = get_obs(data)
    q_init = q_init[7 : 7 + num_actions]
    dq_init = dq_init[6 : 6 + num_actions]
    
    # Fill all observation buffers with initial values
    for i in range(task_obs_len):
        command_obs[i] = torch.zeros(num_actions)  # command is zero in lab
        motion_root_ori_obs[i] = torch.tensor([1., 0., 0., 1., 0., 0.])
        motion_root_zpos_obs[i] = torch.tensor(0.0)
        motion_root_xylin_vel_obs[i] = torch.tensor([0., 0.])
        motion_root_ang_zvel_obs[i] = torch.tensor([0.])
    
    for i in range(prop_obs_len):
        ori = matrix_from_quat(torch.from_numpy(quat_init[[3, 0, 1, 2]]).unsqueeze(0))[..., :2]
        base_ori_obs[i] = ori.reshape(-1)
        base_ang_vel_obs[i] = torch.from_numpy(omega_init)
        joint_pos_obs[i] = torch.from_numpy(q_init[mj2lab_indexes] - default_dof_pos[mj2lab_indexes])
        joint_vel_obs[i] = torch.from_numpy(dq_init[mj2lab_indexes]) * 0.05
        actions_obs[i] = torch.zeros(num_actions)  # Initial actions are zero

    count_lowlevel = 0
    control_dt = cfg.sim_config.dt * cfg.sim_config.decimation  # Control frequency dt
    
    # Motion time tracking for synchronization
    motion_time = 0.0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec, quat_torso = get_obs(data)
        q = q[7 : 7 + num_actions]
        dq = dq[6 : 6 + num_actions]


        # 500hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # Update observations following lab's observation order:
            # Note: In sim2sim, we don't have motion data, so command and motion_anchor_ori are set to zero
            
            # 1. command: motion command joint positions (from motion data)
            command_obs = torch.roll(command_obs, shifts=-1, dims=0)
            
            # Get motion command from motion data
            # Synchronize motion playback with control frequency
            # Calculate motion time step based on control dt
            motion_time += control_dt
            motion_dt = motion_loader.get_dt()
            # Clamp to last frame instead of looping
            motion_time_step = min(int(motion_time / motion_dt), motion_loader.time_step_total - 1)
            
            # Get joint positions at current motion time step
            motion_joint_pos = motion_loader.get_joint_pos(motion_time_step)
            # Note: motion_joint_pos is already in lab order (as stored in npz)
            command_obs[-1] = motion_joint_pos

            # 3. motion_root_zpos: motion root z position
            motion_root_zpos_obs = torch.roll(motion_root_zpos_obs, shifts=-1, dims=0)
            motion_root_zpos_obs[-1] = torch.tensor(motion_loader.get_root_zpos(motion_time_step))

            # 4. motion_root_lin_vel: motion root linear velocity
            motion_root_xylin_vel_obs = torch.roll(motion_root_xylin_vel_obs, shifts=-1, dims=0)
            motion_root_xylin_vel_obs[-1] = torch.tensor(motion_loader.get_root_xylin_vel_local(motion_time_step))

            # 5. motion_root_ang_vel: motion root angular velocity
            motion_root_ang_zvel_obs = torch.roll(motion_root_ang_zvel_obs, shifts=-1, dims=0)
            motion_root_ang_zvel_obs[-1] = torch.tensor(motion_loader.get_root_ang_zvel_local(motion_time_step))
            
            # 2. motion_anchor_ori: motion anchor orientation (first 2 rows of rotation matrix)
            motion_root_ori_obs = torch.roll(motion_root_ori_obs, shifts=-1, dims=0)
            root_ori = motion_loader.get_root_ori(motion_time_step)
            motion_root_ori_obs[-1] = torch.tensor(root_ori)
            
            # 3. projected_gravity: gravity projection in base frame
            base_ori_obs = torch.roll(base_ori_obs, shifts=-1, dims=0)
            ori = matrix_from_quat(torch.from_numpy(quat[[3, 0, 1, 2]]).unsqueeze(0))[..., :2]
            base_ori_obs[-1] = ori.reshape(-1)
            
            # 4. base_ang_vel: base angular velocity in base frame
            base_ang_vel_obs = torch.roll(base_ang_vel_obs, shifts=-1, dims=0)
            base_ang_vel_obs[-1] = torch.from_numpy(omega)
            
            # 5. joint_pos: joint positions relative to default positions
            joint_pos_obs = torch.roll(joint_pos_obs, shifts=-1, dims=0)
            joint_pos_obs[-1] = torch.from_numpy(q[mj2lab_indexes] - default_dof_pos[mj2lab_indexes])
            
            # 6. joint_vel: joint velocities relative to default velocities (which are zero)
            joint_vel_obs = torch.roll(joint_vel_obs, shifts=-1, dims=0)
            joint_vel_obs[-1] = torch.from_numpy(dq[mj2lab_indexes]) * 0.05
            
            # 7. actions: last actions (in lab frame)
            actions_obs = torch.roll(actions_obs, shifts=-1, dims=0)
            actions_obs[-1] = torch.from_numpy(action[mj2lab_indexes])

            # Concatenate observations in the exact order as lab's PolicyCfg:
            # command -> motion_anchor_ori -> projected_gravity -> base_ang_vel -> joint_pos -> joint_vel -> actions
            policy_input = torch.cat((  command_obs.reshape(-1),           # (5*29=145,)
                                        motion_root_ori_obs.reshape(-1),   # (5*6=30,)
                                        motion_root_zpos_obs.reshape(-1),   # (5*1=5,)
                                        motion_root_xylin_vel_obs.reshape(-1),   # (5*2=10,)
                                        motion_root_ang_zvel_obs.reshape(-1),   # (5*1=5,)
                                        base_ori_obs.reshape(-1),          # (5*6=30,)
                                        base_ang_vel_obs.reshape(-1),       # (5*3=15,)
                                        joint_pos_obs.reshape(-1),          # (5*29=145,)
                                        joint_vel_obs.reshape(-1),          # (5*29=145,)
                                        actions_obs.reshape(-1)             # (5*29=145,)
                                        ), dim=-1).to(dtype=torch.float32).unsqueeze(0)
            # Total observation dimension: 145 + 30 + 15 + 15 + 145 + 145 + 145 = 640

            action_policy = policy(policy_input).detach().numpy()
            # Ensure action_policy is 1D array
            if action_policy.ndim > 1:
                action_policy = action_policy.flatten()
            # action_policy is in lab order, convert to mujoco order
            action = action_policy[lab2mj_indexes]
            clip_actions = 100
            action = np.clip(action, -clip_actions, clip_actions)

            # action_scale, default_dof_pos, and action are all in mujoco order
            target_q = (action * cfg.robot_config.action_scale) + default_dof_pos


        target_dq = np.zeros((num_actions), dtype=np.double)
        # Generate PD control
        # kps, kds, target_q, q, target_dq, dq are all in mujoco order
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        # viewer.sync()
        viewer.render()
        viewer.cam.lookat[:2] = data.qpos[:2]
        count_lowlevel += 1


    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Path to the policy model file (.pt or .onnx).')
    parser.add_argument('--motion_file', type=str, required=True,
                        help='Path to motion file (.npz).')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg():
        
        class sim_config:
            if args.terrain:
                mujoco_model_path = f'source/scaletrack/scaletrack/assets/g1_29dof/g1_29dof.xml'
            else:
                mujoco_model_path = f'source/scaletrack/scaletrack/assets/g1_29dof/g1_29dof.xml'
            sim_duration = 1200.0
            dt = 0.005
            decimation = 4
            
        class robot_config:
            # kps, kds, and action_scale are in mujoco order (aligned with lab via lab2mj_indexes mapping)
            kps = np.array([ 
                            40.1792,  99.0984,  40.1792,  99.0984,  28.5012,  28.5012,  
                            40.1792,  99.0984,  40.1792,  99.0984,  28.5012,  28.5012, 
                            40.1792,  28.5012,  28.5012,  
                            # 100.0,  300.0,  300.0,  
                            14.2506,  14.2506,  14.2506,  14.2506,  14.2506,  16.7783,  16.7783,  
                            14.2506,  14.2506,  14.2506,  14.2506,  14.2506,  16.7783,  16.7783
                            ], dtype=np.double)
            kds = np.array([
                            2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 
                            2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 
                            2.5579, 1.8144, 1.8144, 
                            # 2.0, 5.0, 5.0, 
                            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681, 
                            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681
                            ], dtype=np.double)
            tau_limit = 200. * np.ones(29, dtype=np.double)
            action_scale = np.array([
                            0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
                            0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
                            0.5475, 0.4385, 0.4385,
                            # 0.25, 0.25, 0.25,
                            0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
                            0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745
                            ], dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), motion_file=args.motion_file)

    