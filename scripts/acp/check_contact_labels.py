"""Check the geometric contact labels of an ACP task against simulated contact forces.

Every environment replays its reference: before each step the robot is set to the reference state and the joint
targets are the reference joint positions, so the physics step resolves the reference pose against the terrain.
Afterwards the contact sensor (force above ``--threshold`` N) is compared with the label of
``ContactScheduleCommand.reference_contact`` per contact body. Episodes start at frame 0 and terminations other than
the dataset end are disabled.

Usage::

    isaaclab -p scripts/acp/check_contact_labels.py --task Instinct-ACP-Tracker-G1-Play-v0 --num_envs 64 --viz none
"""

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

parser = cli.build_play_parser()
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--threshold", type=float, default=1.0, help="Contact force threshold (N)")
parser.add_argument("--command_name", default="contact_schedule")
parser.add_argument("--print_env", type=int, default=None, help="Print the state of this env every 25 steps")
args_cli, overrides = cli.parse_cli(parser, require_run_name=False)


def main():
    env_cfg, agent_cfg = cli.resolve_play_cfg(args_cli.task, overrides)
    cli.apply_play_cli(args_cli, env_cfg, agent_cfg)
    for name in list(vars(env_cfg.terminations)):
        if name not in ("dataset_exhausted", "time_out") and getattr(env_cfg.terminations, name) is not None:
            setattr(env_cfg.terminations, name, None)
    with launch_simulation(env_cfg, args_cli):
        run(env_cfg)


def run(env_cfg):
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    term = env.command_manager.get_term(args_cli.command_name)
    robot = term.robot
    manager = term.motion_reference
    sensor = env.scene.sensors["contact_forces"]
    sensor_body_ids, _ = sensor.find_bodies(term.body_names, preserve_order=True)
    action_term = env.action_manager.get_term("joint_pos")
    action_joint_ids = action_term._joint_ids
    default = robot.data.default_joint_pos.torch[:, action_joint_ids]
    scale = action_term._scale

    counts = {k: torch.zeros(len(term.body_names), device=env.device) for k in ("tp", "fp", "fn", "tn")}
    for _ in range(args_cli.steps):
        ref = manager.reference_frame
        root_pose = torch.cat([ref.base_pos_w[:, 0], ref.base_quat_w[:, 0]], dim=-1)
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.cat([ref.base_lin_vel_w[:, 0], ref.base_ang_vel_w[:, 0]], dim=-1)
        )
        robot.write_joint_state_to_sim_index(position=ref.joint_pos[:, 0], velocity=ref.joint_vel[:, 0])
        action = (ref.joint_pos[:, 0][:, action_joint_ids] - default) / scale
        env.step(action)
        forces = sensor.data.net_forces_w.torch[:, sensor_body_ids].norm(dim=-1)
        sim_contact = forces > args_cli.threshold
        label = term.reference_contact
        valid = manager.reference_frame.validity[:, 0].unsqueeze(-1)
        counts["tp"] += (sim_contact & label & valid).sum(0)
        counts["fp"] += (~sim_contact & label & valid).sum(0)
        counts["fn"] += (sim_contact & ~label & valid).sum(0)
        counts["tn"] += (~sim_contact & ~label & valid).sum(0)
        if args_cli.print_env is not None and env.common_step_counter % 25 == 0:
            i = args_cli.print_env
            body_z = robot.data.body_link_pos_w.torch[i, term.robot_body_ids, 2]
            ref_z = manager.reference_frame.link_pos_w[i, 0, term.reference_link_ids, 2]
            print(
                f"step {env.common_step_counter} {manager.get_current_motion_identifiers()[i].split('/')[-1]}"
                f" root z {robot.data.root_pos_w.torch[i, 2]:.3f} ref root z {ref.base_pos_w[i, 0, 2]:.3f}"
                f" ground z {env.scene.sensors['tracker_height_scanner'].data.ray_hits_w.torch[i, 31, 2]:.3f}"
                f"\n  body z {body_z.cpu().numpy().round(3)}\n  ref z  {ref_z.cpu().numpy().round(3)}"
                f"\n  force  {forces[i].cpu().numpy().round(0)}\n  label  {label[i].int().cpu().numpy()}"
                f"\n  target {term.target_pos_w[i, :, 2].cpu().numpy().round(3)}"
            )

    print(f"{'body':24s} {'label+':>8s} {'sim+':>8s} {'agree':>7s} {'precision':>9s} {'recall':>7s}")
    for i, name in enumerate(term.body_names):
        tp, fp, fn, tn = (counts[k][i].item() for k in ("tp", "fp", "fn", "tn"))
        total = tp + fp + fn + tn
        print(
            f"{name:24s} {int(tp + fp):8d} {int(tp + fn):8d} {(tp + tn) / max(total, 1):7.1%}"
            f" {tp / max(tp + fp, 1):9.1%} {tp / max(tp + fn, 1):7.1%}"
        )
    env.close()


if __name__ == "__main__":
    main()
