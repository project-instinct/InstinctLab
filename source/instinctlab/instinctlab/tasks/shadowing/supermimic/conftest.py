"""Session fixture: one live SuperMimic environment shared by this package's tests.

Launching the simulator costs ~40 s, so it is done once per session rather than per test. Tests that
need no simulator live in ``test_imagination.py`` and do not request these fixtures.
"""

import pytest
import torch

TASK = "Instinct-SuperMimic-Plane-G1-Newton-v0"
NUM_ENVS = 4


@pytest.fixture(scope="session")
def env():
    """A live SuperMimic env with the paper design, stepped zero times after reset."""
    import instinctlab.tasks  # noqa: F401  registers the tasks (must precede cfg resolution)
    from instinctlab.utils import instinct_rl_cli as cli

    parser = cli.build_train_parser()
    args_cli, overrides = cli.parse_cli(
        parser,
        argv=["--task", TASK, "--num_envs", str(NUM_ENVS), "--run_name", "pytest"],
        require_run_name=True,
    )

    from isaaclab.app import launch_simulation

    env_cfg, _ = cli.resolve_train_cfg(TASK, overrides)
    env_cfg.scene.num_envs = NUM_ENVS

    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym

        sim = gym.make(TASK, cfg=env_cfg)
        sim.reset()
        yield sim
        sim.close()


@pytest.fixture
def mr(env):
    """The motion reference manager."""
    return env.unwrapped.scene["motion_reference"]


@pytest.fixture
def step_env(env):
    """Advance the environment one control step with zero action."""
    u = env.unwrapped

    def _step():
        a = torch.zeros(u.num_envs, u.action_space.shape[-1], device=u.device)
        return env.step(a)

    return _step


@pytest.fixture
def imagination(env):
    """A ``SuperMimicImagination`` wired to the live env's own layout and reference manager.

    Term orders come from the observation manager, and joint ids from the reference commands, so the
    imagined observation is compared against the real one rather than against a literal.
    """
    import torch as _torch

    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import (
        SuperMimicImagination,
        build_reference_window,
    )
    from instinctlab.tasks.shadowing.supermimic.supermimic_provider import WM_STATE_GROUP

    u = env.unwrapped
    robot = u.scene["robot"]
    mr_ = u.scene["motion_reference"]
    om = u.observation_manager

    cmd = u.command_manager.get_term("joint_pos_ref_command")
    body_ids = robot.find_bodies(mr_.cfg.link_of_interests, preserve_order=True)[0]

    def _obs_format(unwrapped):
        """Same shape as the wrapper's get_obs_format, from the observation manager."""
        om_ = unwrapped.observation_manager
        return {
            group: {
                name: dim
                for name, dim in zip(om_.active_terms[group], om_.group_obs_term_dim[group])
            }
            for group in om_.active_terms
        }

    from instinctlab.tasks.shadowing.supermimic.supermimic_provider import _history_length, _history_terms

    def _make(horizon=8):
        win = build_reference_window(mr_, horizon=horizon)
        return SuperMimicImagination(
            state_format=dict(_obs_format(u)[WM_STATE_GROUP]),
            soft_joint_pos_limits=robot.data.soft_joint_pos_limits.torch[0],
            default_joint_pos=robot.data.default_joint_pos.torch[0][cmd._asset_joint_ids],
            policy_term_order=list(om.active_terms["policy"]),
            critic_term_order=list(om.active_terms["critic"]),
            reference_joint_ids=cmd._motion_reference_joint_ids,
            policy_joint_ids=cmd._asset_joint_ids,
            goal_window=mr_.cfg.num_frames,
            # same derivation the provider uses, so a layout drift shows up here
            history_terms=_history_terms(u.observation_manager, "policy"),
            history_length=_history_length(u.observation_manager, "policy"),
            ref_window=win,
        )

    return _make


@pytest.fixture
def robot_state(env):
    """Current robot state in the world-model layout, plus actions and the manager's term values.

    Returns a callable so a test can sample after stepping. The flat state is built with the same
    ``make_state_format`` the algorithm uses, so a layout drift fails here rather than silently
    feeding the world model misaligned numbers.
    """
    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import make_state_format

    u = env.unwrapped

    def _state():
        robot = u.scene["robot"]
        mr_ = u.scene["motion_reference"]
        body_ids = robot.find_bodies(mr_.cfg.link_of_interests, preserve_order=True)[0]

        parts = {
            "root_pos_w": robot.data.root_pos_w.torch,
            "root_quat_w": robot.data.root_quat_w.torch,
            "root_lin_vel_w": robot.data.root_lin_vel_w.torch,
            "root_ang_vel_w": robot.data.root_ang_vel_w.torch,
            "joint_pos": robot.data.joint_pos.torch,
            "joint_vel": robot.data.joint_vel.torch,
            "body_link_pos_w": robot.data.body_link_pos_w.torch[:, body_ids],
            "body_link_quat_w": robot.data.body_link_quat_w.torch[:, body_ids],
            "body_link_lin_vel_w": robot.data.body_link_lin_vel_w.torch[:, body_ids],
            "body_link_ang_vel_w": robot.data.body_link_ang_vel_w.torch[:, body_ids],
        }
        fmt = make_state_format(len(body_ids), robot.data.joint_pos.torch.shape[-1])
        missing = set(fmt) - set(parts)
        assert not missing, f"state format has components this fixture does not supply: {sorted(missing)}"
        state = torch.cat([parts[name].reshape(parts[name].shape[0], -1) for name in fmt], dim=-1)

        manager_terms = {k: float(v.mean()) for k, v in u.reward_manager._termwise_reward_buf["rewards"].items()}
        return (
            state,
            u.action_manager.action.clone(),
            u.action_manager.prev_action.clone(),
            {
                "manager": manager_terms,
                "limits": robot.data.soft_joint_pos_limits.torch[0],
                "num_links": len(body_ids),
            },
        )

    return _state
