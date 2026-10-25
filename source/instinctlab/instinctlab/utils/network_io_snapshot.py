"""Dump the network I/O layout and a fixed-seed rollout of an InstinctLab task.

The output of one Isaac Lab version is frozen and compared against the output of another, so layout changes
(term order, shapes, dtypes, joint/body selections, sensor patterns, noise pipelines) show up as diffs.
Run it as a module so that nothing under ``scripts/`` is imported::

    python -m instinctlab.utils.network_io_snapshot --task Instinct-Locomotion-Flat-G1-v0 --out_dir <dir> --viz none

Files written to ``--out_dir``:

- ``layout.yaml``: observation format, per-term pipeline settings, actions, commands and sensors.
- ``env.yaml`` / ``agent.yaml``: the resolved env and agent configs.
- ``rollout.pt``: observations, rewards, dones, terminations and contact forces of the fixed rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import torch
import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg

    from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

# Isaac Lab sensor/manager modules load USD; they are imported inside the functions, which only run after
# ``launch_simulation`` has started the simulator.
PARTIAL_RESET_ENV_IDS = [1, 3]
NUM_ROLLOUT_STEPS = 16


def _to_plain(value):
    """Convert tuples, tensors and nested containers into YAML-safe plain types."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, slice):
        return f"slice({value.start}, {value.stop}, {value.step})"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _qualname(obj) -> str:
    """Name a function, class or class instance by its module path."""
    target = obj if callable(obj) and hasattr(obj, "__qualname__") else type(obj)
    return f"{target.__module__}.{target.__qualname__}"


def _selected_names(all_names: list[str], ids) -> list[str]:
    if isinstance(ids, slice):
        return list(all_names[ids])
    return [all_names[i] for i in ids]


def _scene_entity_record(entity_cfg: SceneEntityCfg, scene) -> dict:
    """The joint/body names a resolved ``SceneEntityCfg`` selects, in selection order."""
    asset = scene[entity_cfg.name]
    record = {"name": entity_cfg.name, "preserve_order": entity_cfg.preserve_order}
    for kind in ("joint", "body"):
        asset_names = getattr(asset, f"{kind}_names", None)
        if asset_names is not None:
            record[f"{kind}_names"] = _selected_names(asset_names, getattr(entity_cfg, f"{kind}_ids"))
    return record


def _term_scene_entities(term_cfg, scene) -> dict[str, dict]:
    """Every ``SceneEntityCfg`` found in the fields or ``params`` of a manager term cfg."""
    from isaaclab.managers import SceneEntityCfg

    candidates = dict(vars(term_cfg))
    candidates.update({f"params.{k}": v for k, v in (getattr(term_cfg, "params", None) or {}).items()})
    return {
        key: _scene_entity_record(value, scene)
        for key, value in candidates.items()
        if isinstance(value, SceneEntityCfg)
    }


def _pattern_record(sensor) -> dict:
    """Point count and an order-sensitive hash of the ray-cast pattern of a sensor."""
    from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

    pattern_cfg = sensor.cfg.pattern_cfg
    record = {"class": _qualname(pattern_cfg)}
    if isinstance(pattern_cfg, PinholeCameraPatternCfg):
        # camera patterns are built from the intrinsics of the first camera instead of the cfg alone
        intrinsic_matrices = sensor.data.intrinsic_matrices.torch[:1].cpu()
        ray_starts, ray_directions = pattern_cfg.func(pattern_cfg, intrinsic_matrices, "cpu")
        ray_starts, ray_directions = ray_starts[0], ray_directions[0]
        record["intrinsic_matrix"] = _to_plain(intrinsic_matrices[0])
        record["height"], record["width"] = pattern_cfg.height, pattern_cfg.width
    else:
        ray_starts, ray_directions = pattern_cfg.func(pattern_cfg, "cpu")
    rays = torch.cat([ray_starts, ray_directions], dim=1).to(torch.float64).round(decimals=6)
    record["num_points"] = len(ray_directions)
    record["sha256"] = hashlib.sha256(rays.numpy().tobytes()).hexdigest()
    return record


def _sensor_record(sensor) -> dict:
    from isaaclab.sensors import BaseContactSensor

    cfg = sensor.cfg
    record = {"class": _qualname(sensor), "update_period": cfg.update_period}
    if isinstance(sensor, BaseContactSensor):
        record["body_names"] = list(sensor.body_names)
    if getattr(cfg, "pattern_cfg", None) is not None:
        record["pattern"] = _pattern_record(sensor)
    for key in ("history_length", "height", "width", "data_types", "data_histories"):
        if hasattr(cfg, key):
            record[key] = _to_plain(getattr(cfg, key))
    if hasattr(cfg, "noise_pipeline"):
        record["noise_pipeline"] = [[name, _qualname(noise_cfg)] for name, noise_cfg in cfg.noise_pipeline.items()]
    return record


def collect_layout(env: InstinctRlVecEnvWrapper) -> dict:
    """Collect the network-facing layout of a constructed environment."""
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    obs_manager = unwrapped.observation_manager
    obs_pack = obs_manager.compute()

    observations = {}
    for group_name, term_names in obs_manager.active_terms.items():
        term_cfgs = obs_manager._group_obs_term_cfgs[group_name]
        terms = {}
        for term_name, term_dim, term_cfg in zip(term_names, obs_manager.group_obs_term_dim[group_name], term_cfgs):
            term_value = obs_pack[group_name][term_name]
            terms[term_name] = {
                "shape": list(term_dim),
                "dtype": str(term_value.dtype),
                "func": _qualname(term_cfg.func),
                "modifiers": [_qualname(modifier.func) for modifier in term_cfg.modifiers or []],
                "noise": None if term_cfg.noise is None else _qualname(term_cfg.noise),
                "clip": _to_plain(term_cfg.clip),
                "scale": _to_plain(term_cfg.scale),
                "history_length": term_cfg.history_length,
                "flatten_history_dim": term_cfg.flatten_history_dim,
                "scene_entities": _term_scene_entities(term_cfg, scene),
            }
        observations[group_name] = {
            "concatenate_terms": obs_manager.group_obs_concatenate[group_name],
            "flat_dim": sum(int(torch.Size(dim).numel()) for dim in obs_manager.group_obs_term_dim[group_name]),
            "terms": terms,
        }

    actions = {
        term_name: {
            "class": _qualname(term),
            "action_dim": term.action_dim,
            "joint_names": _to_plain(getattr(term, "_joint_names", None)),
            "scene_entities": _term_scene_entities(term.cfg, scene),
        }
        for term_name, term in unwrapped.action_manager._terms.items()
    }
    commands = {
        term_name: {
            "class": _qualname(term),
            "command_shape": list(term.command.shape[1:]),
            "scene_entities": _term_scene_entities(term.cfg, scene),
        }
        for term_name, term in unwrapped.command_manager._terms.items()
    }

    return {
        "obs_format": _to_plain(env.get_obs_format()),
        "observations": observations,
        "total_action_dim": unwrapped.action_manager.total_action_dim,
        "actions": actions,
        "commands": commands,
        "sensors": {name: _sensor_record(sensor) for name, sensor in scene.sensors.items()},
        "num_rewards": env.num_rewards,
        "step_dt": unwrapped.step_dt,
    }


def collect_rollout(env: InstinctRlVecEnvWrapper, seed: int) -> list[dict]:
    """Full reset, partial reset, then zero-action and fixed-action steps, all recorded on CPU."""
    from isaaclab.sensors import BaseContactSensor

    unwrapped = env.unwrapped
    device = unwrapped.device
    contact_sensors = {
        name: sensor for name, sensor in unwrapped.scene.sensors.items() if isinstance(sensor, BaseContactSensor)
    }
    frames = []

    def capture(tag: str, obs_pack: dict, rewards: torch.Tensor | None = None, dones: torch.Tensor | None = None):
        frames.append(
            {
                "tag": tag,
                "observations": {group: value.cpu().clone() for group, value in obs_pack.items()},
                "rewards": None if rewards is None else rewards.cpu().clone(),
                "dones": None if dones is None else dones.cpu().clone(),
                "terminated": unwrapped.termination_manager.terminated.cpu().clone(),
                "time_outs": unwrapped.termination_manager.time_outs.cpu().clone(),
                "contact_net_forces_w": {
                    name: sensor.data.net_forces_w.torch.cpu().clone() for name, sensor in contact_sensors.items()
                },
            }
        )

    obs_pack, _ = unwrapped.reset(seed=seed)
    capture("full_reset", env._flatten_all_obs_groups(obs_pack))
    obs_pack, _ = unwrapped.reset(env_ids=torch.tensor(PARTIAL_RESET_ENV_IDS, device=device))
    capture("partial_reset", env._flatten_all_obs_groups(obs_pack))

    action_dim = unwrapped.action_manager.total_action_dim
    zero_actions = torch.zeros(env.num_envs, action_dim, device=device)
    fixed_actions = 0.1 * torch.sin(torch.arange(action_dim, device=device, dtype=torch.float32)).repeat(
        env.num_envs, 1
    )
    with torch.inference_mode():
        for tag, actions in (("zero_action", zero_actions), ("fixed_action", fixed_actions)):
            for step in range(NUM_ROLLOUT_STEPS):
                _, rewards, dones, extras = env.step(actions)
                capture(f"{tag}_{step:02d}", extras["observations"], rewards, dones)
    return frames


def write_snapshot(env: InstinctRlVecEnvWrapper, env_cfg, agent_cfg, out_dir: str, seed: int):
    """Write the layout, resolved configs and rollout of a constructed environment into ``out_dir``."""
    from isaaclab.utils.io import dump_yaml

    from instinctlab.utils.instinct_rl_cli import env_cfg_for_dump

    os.makedirs(out_dir, exist_ok=True)
    dump_yaml(os.path.join(out_dir, "env.yaml"), env_cfg_for_dump(env_cfg))
    dump_yaml(os.path.join(out_dir, "agent.yaml"), agent_cfg)
    with open(os.path.join(out_dir, "layout.yaml"), "w") as f:
        yaml.safe_dump(collect_layout(env), f, sort_keys=False)
    torch.save({"seed": seed, "frames": collect_rollout(env, seed)}, os.path.join(out_dir, "rollout.pt"))


def main():
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import instinctlab.tasks  # noqa: F401

    parser = argparse.ArgumentParser(description="Dump the network I/O layout and a fixed rollout of a task.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write the snapshot into.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the environment and the rollout.")
    add_launcher_args(parser)
    args_cli = parser.parse_args()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "instinct_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
        write_snapshot(env, env_cfg, agent_cfg, args_cli.out_dir, args_cli.seed)
        env.close()


if __name__ == "__main__":
    main()
