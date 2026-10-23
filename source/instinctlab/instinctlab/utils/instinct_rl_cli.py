"""Command-line parsing, config layering and log naming for the Instinct-RL entry points.

``scripts/instinct_rl/train.py``, ``scripts/instinct_rl/play.py`` and the task-local play scripts stay thin
entry points around these helpers. Module-level imports are limited to the standard library so the parsing
and naming helpers can be tested without a simulator; Isaac Lab is imported inside the functions that need it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

AGENT_ENTRY_POINT = "instinct_rl_cfg_entry_point"
LOG_ROOT = os.path.join("logs", "instinct_rl")

# Preset selectors that change the composed task, so a training run that uses them needs a run name.
PRESET_SELECTORS = ("physics", "renderer", "presets")
# ``visualizer=`` only picks a viewer (``--viz`` produces it), so it does not count as a Hydra-style argument.
VISUALIZER_SELECTOR = "visualizer"

_OVERRIDE_RE = re.compile(r"^\+{0,2}(env|agent)\.[^=\s]+=.*$")
_DELETE_RE = re.compile(r"^~(env|agent)\.[^=\s]+(=.*)?$")
_SELECTOR_RE = re.compile(rf"^({'|'.join((*PRESET_SELECTORS, VISUALIZER_SELECTOR))})=\S.*$")

# Keep Hydra from writing ``outputs/<date>/<time>/`` and a job log into the working directory.
HYDRA_NO_OUTPUT_ARGS = (
    "hydra.output_subdir=null",
    "hydra.run.dir=.",
    "hydra/job_logging=none",
    "hydra/hydra_logging=none",
)


# -- Instinct-RL agent arguments


def add_instinct_rl_args(parser: argparse.ArgumentParser):
    """Add INSTINCT-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    arg_group = parser.add_argument_group("instinct_rl", description="Arguments for Instinct-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", default=None, action="store_true", help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")


def parse_instinct_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> InstinctRlOnPolicyRunnerCfg:
    """Parse configuration for Instinct-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for Instinct-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    instinctrl_cfg: InstinctRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, AGENT_ENTRY_POINT)
    return update_instinct_rl_cfg(instinctrl_cfg, args_cli)


def update_instinct_rl_cfg(agent_cfg: InstinctRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for Instinct-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for Instinct-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for Instinct-RL agent based on inputs.
    """
    if getattr(args_cli, "seed", None) is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    return agent_cfg


# -- parsers


def add_video_args(parser: argparse.ArgumentParser, *, action: str, video_length: int) -> None:
    """Add ``--video`` and ``--video_length``.

    ``--video`` takes an optional :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` source.

    Args:
        parser: The parser to add the arguments to.
        action: Workflow name used in the help text.
        video_length: Default clip length in env steps.
    """
    parser.add_argument(
        "--video",
        nargs="?",
        const="viz",
        default=None,
        type=_video_source,
        metavar="SOURCE",
        help=(
            f"Record videos during {action}. SOURCE defaults to 'viz' (the first capture-capable visualizer --viz"
            " selects, else a headless newton_gl); 'viz:<type>' or 'sensor:<name>[:<channel>]' select another source."
        ),
    )
    parser.add_argument(
        "--video_length", type=int, default=video_length, help="Length of each recorded video (in steps)."
    )


def _video_source(value: str) -> str:
    """Validate a ``--video`` source with Isaac Lab's grammar."""
    from isaaclab.envs.utils.video_recorder_cfg import parse_video_source

    try:
        parse_video_source(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _add_launcher_and_common_args(parser: argparse.ArgumentParser) -> None:
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")


def build_train_parser() -> argparse.ArgumentParser:
    """Return the ``train.py`` parser: launcher arguments, then the InstinctLab arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL.")
    _add_launcher_and_common_args(parser)
    add_video_args(parser, action="training", video_length=200)
    parser.add_argument(
        "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--logroot", type=str, default=None, help="Override the default log root, typically logs/instinct_rl/."
    )
    parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
    parser.add_argument("--distributed", action="store_true", default=False, help="Run distributed training.")
    parser.add_argument("--local-rank", type=int, help="Local rank assigned by the distributed launcher.")
    parser.add_argument("--cprofile", action="store_true", default=False, help="Enable cProfile.")
    add_instinct_rl_args(parser)
    return parser


def build_play_parser() -> argparse.ArgumentParser:
    """Return the shared play parser. Task-local play scripts add their own flags before parsing."""
    parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
    _add_launcher_and_common_args(parser)
    add_video_args(parser, action="playing", video_length=3000)
    parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
    parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
    parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
    parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
    parser.add_argument(
        "--sample", action="store_true", default=False, help="Sample actions instead of using the policy."
    )
    parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
    add_instinct_rl_args(parser)
    return parser


def add_play_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--no_terminate`` and ``--aux_reward`` (``play.py`` and the shadowing play script)."""
    parser.add_argument(
        "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
    )
    parser.add_argument(
        "--aux_reward",
        action="store_true",
        default=False,
        help="Whether to assign auxiliary rewards to each of the env's reward term.",
    )


def parse_cli(
    parser: argparse.ArgumentParser, argv: list[str] | None = None, *, require_run_name: bool
) -> tuple[argparse.Namespace, list[str]]:
    """Parse the command line and validate the residual task overrides.

    Args:
        parser: A parser from :func:`build_train_parser` or :func:`build_play_parser`.
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.
        require_run_name: Whether Hydra-style arguments require a non-empty ``--run_name`` (training).

    Returns:
        The parsed arguments and the task overrides (Hydra overrides and preset selectors).
    """
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    args, overrides = setup_preset_cli(parser, sys.argv[1:] if argv is None else argv)
    validate_overrides(parser, overrides, args.run_name, require_run_name=require_run_name)
    return args, overrides


def is_hydra_style(token: str) -> bool:
    """Whether *token* is ``[+|++]env.*|agent.*=…``, ``~env.*|agent.*[=…]`` or a preset selector."""
    if _OVERRIDE_RE.match(token) or _DELETE_RE.match(token):
        return True
    return _SELECTOR_RE.match(token) is not None and token.partition("=")[0] in PRESET_SELECTORS


def validate_overrides(
    parser: argparse.ArgumentParser, overrides: list[str], run_name: str | None, *, require_run_name: bool
) -> None:
    """Reject malformed residual tokens, then apply the training run-name rule.

    Raises:
        SystemExit: Through :meth:`argparse.ArgumentParser.error`.
    """
    malformed = [token for token in overrides if not (is_hydra_style(token) or _SELECTOR_RE.match(token))]
    if malformed:
        parser.error(
            f"unrecognized arguments: {' '.join(malformed)}. Task overrides must be [+|++]env.<path>=<value>,"
            " [+|++]agent.<path>=<value>, ~env.<path>, ~agent.<path>, or physics=/renderer=/presets=/visualizer=."
        )
    if require_run_name and not run_name and any(is_hydra_style(token) for token in overrides):
        parser.error("--run_name is required (and must be non-empty) when Hydra-style arguments are given.")


def _reaches_hydra(token: str) -> bool:
    """Whether Isaac Lab's ``register_task`` hands *token* to ``hydra.main`` instead of applying it directly."""
    key, separator, _ = token.partition("=")
    if not separator:
        return True
    key = key.lstrip("-")
    if key in (*PRESET_SELECTORS, VISUALIZER_SELECTOR):
        return False
    return not (key.startswith(("env.", "agent.")) and not key.endswith("+"))


def with_hydra_output_suppressed(overrides: list[str]) -> list[str]:
    """Append :data:`HYDRA_NO_OUTPUT_ARGS` when the overrides make Isaac Lab run ``hydra.main``."""
    if any(_reaches_hydra(token) for token in overrides):
        return [*overrides, *HYDRA_NO_OUTPUT_ARGS]
    return list(overrides)


# -- config composition


def resolve_train_cfg(task: str, overrides: list[str]):
    """Compose the training env and agent configs: task defaults, then the Hydra overrides."""
    from isaaclab_tasks.utils import resolve_task_config

    _reject_in_memory_meshes_with_hydra(task, overrides)
    return resolve_task_config(task, AGENT_ENTRY_POINT, overrides=with_hydra_output_suppressed(overrides))


def resolve_play_cfg(task: str, overrides: list[str], *, env_yaml: dict | None = None, agent_yaml: dict | None = None):
    """Compose the play configs: task defaults, then the saved ``env.yaml``/``agent.yaml``, then Hydra.

    Without saved YAML this is :func:`~isaaclab_tasks.utils.resolve_task_config`. With saved YAML, the preset
    selectors are resolved first, the YAML is layered onto the resulting cfg instances, and the remaining
    overrides are applied on top, so Hydra still wins over the saved run. Dedicated CLI options are applied by
    the caller afterwards.

    InstinctLab declares no :class:`~isaaclab_tasks.utils.hydra.PresetCfg`, so ``env.``/``agent.`` scalars are
    applied as plain values after the YAML.

    Args:
        task: Task name.
        overrides: Task overrides returned by :func:`parse_cli`.
        env_yaml: Saved ``params/env.yaml`` content, or None.
        agent_yaml: Saved ``params/agent.yaml`` content, or None.

    Returns:
        The env and agent configs.
    """
    from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces
    from isaaclab.utils.dict import replace_strings_with_slices, update_class_from_dict
    from isaaclab_tasks.utils import resolve_task_config
    from isaaclab_tasks.utils.hydra import _parse_val, _setattr, register_task

    if env_yaml is None and agent_yaml is None:
        _reject_in_memory_meshes_with_hydra(task, overrides)
        return resolve_task_config(
            task, AGENT_ENTRY_POINT, play_mode=True, overrides=with_hydra_output_suppressed(overrides)
        )

    selectors = [token for token in overrides if token.partition("=")[0] in (*PRESET_SELECTORS, VISUALIZER_SELECTOR)]
    env_cfg, agent_cfg, _ = register_task(task, AGENT_ENTRY_POINT, play_mode=True, overrides=selectors)
    if env_yaml is not None:
        _materialize_saved_physics(env_cfg, env_yaml)
        update_class_from_dict(env_cfg, _prune_disabled(env_cfg, replace_strings_with_slices(env_yaml), "env"))
    if agent_yaml is not None:
        update_class_from_dict(agent_cfg, _prune_disabled(agent_cfg, replace_strings_with_slices(agent_yaml), "agent"))

    cfgs = {"env": env_cfg, "agent": agent_cfg}
    hydra_args = []
    for token in overrides:
        if token in selectors:
            continue
        if _reaches_hydra(token):
            hydra_args.append(token)
        else:
            key, value = token.split("=", 1)
            section, path = key.split(".", 1)
            _setattr(cfgs[section], path, _parse_val(value))
    if hydra_args:
        env_cfg, agent_cfg = _compose_with_hydra(task, env_cfg, agent_cfg, with_hydra_output_suppressed(hydra_args))
    return replace_strings_with_env_cfg_spaces(env_cfg), agent_cfg


_PHYSX_MANAGER = "isaaclab_physx.physics.physx_manager:PhysxManager"


def _materialize_saved_physics(env_cfg, env_yaml: dict) -> None:
    """Give ``env_cfg.sim.physics`` a concrete cfg when the task leaves it None but the saved run has one.

    A task without ``sim.physics`` gets ``PhysxCfg()`` at launch, and that resolved cfg is what ``env.yaml`` holds;
    :func:`~isaaclab.utils.dict.update_class_from_dict` cannot layer a mapping onto None.
    """
    saved = (env_yaml.get("sim") or {}).get("physics")
    sim = getattr(env_cfg, "sim", None)
    if not isinstance(saved, dict) or sim is None or sim.physics is not None:
        return
    if saved.get("class_type") != _PHYSX_MANAGER:
        raise ValueError(
            f"The saved env.yaml uses physics '{saved.get('class_type')}' but the task declares no sim.physics;"
            " pass the matching physics= selector."
        )
    from isaaclab_physx.physics import PhysxCfg

    sim.physics = PhysxCfg()


def _prune_disabled(obj, data: dict, ns: str) -> dict:
    """Drop the saved mappings whose target in *obj* is None, so the play task's disabled terms stay disabled.

    A ``*_PLAY`` cfg sets terms such as ``events.push_robot`` to None, while the training run saved them as
    mappings. A mapping carries no type, so it cannot be rebuilt onto None; the dropped paths are printed.
    """
    pruned = {}
    for key, value in data.items():
        target = obj.get(key, _ABSENT) if isinstance(obj, dict) else getattr(obj, key, _ABSENT)
        if isinstance(value, dict) and target is None:
            print(f"[INFO] Saved {ns}.{key} is disabled in the play task; keeping it disabled.")
            continue
        if isinstance(value, dict) and target is not _ABSENT:
            value = _prune_disabled(target, value, f"{ns}.{key}")
        pruned[key] = value
    return pruned


_ABSENT = object()


def _compose_with_hydra(task: str, env_cfg, agent_cfg, hydra_args: list[str]):
    """Apply *hydra_args* to the given cfg instances, as :func:`~isaaclab_tasks.utils.resolve_task_config` does."""
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings
    from isaaclab.utils import to_dict
    from isaaclab.utils.dict import replace_slices_with_strings, replace_strings_with_slices, update_from_dict

    _reject_in_memory_meshes_with_hydra(task, hydra_args, env_cfg)
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    cfg_dict = replace_slices_with_strings({"env": to_dict(env_cfg), "agent": agent_cfg.to_dict()})
    ConfigStore.instance().store(name=task, node=OmegaConf.create(cfg_dict))

    original_argv, sys.argv = sys.argv, [sys.argv[0], *hydra_args]

    @hydra.main(config_path=None, config_name=task, version_base="1.3")
    def hydra_main(hydra_cfg):
        hydra_cfg = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
        update_from_dict(env_cfg, hydra_cfg["env"])
        agent_cfg.from_dict(hydra_cfg["agent"])

    try:
        hydra_main()
    finally:
        sys.argv = original_argv
    return env_cfg, agent_cfg


# -- log directories and checkpoints


def resume_suffix(resume_path: str) -> str:
    """Return ``_from<date>_<time>`` for a checkpoint inside a ``<date>_<time>[_…]`` run folder, else ``""``."""
    parts = os.path.basename(os.path.dirname(resume_path)).split("_")
    return f"_from{parts[0]}_{parts[1]}" if len(parts) >= 2 else ""


def train_log_dir_name(
    env_run_name: str | None,
    agent_run_name: str | None,
    resume_path: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Return ``<timestamp>[_<env run_name>][_<agent run_name>][_from<date>_<time>]``."""
    name = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if env_run_name:
        name += f"_{env_run_name}"
    if agent_run_name:
        name += f"_{agent_run_name}"
    if resume_path is not None:
        name += resume_suffix(resume_path)
    return name


def checkpoint_path(log_root_path: str, load_run: str, load_checkpoint: str, other_dirs: list[str] | None = None):
    """Resolve a checkpoint from an absolute run folder or a run-name pattern relative to *log_root_path*."""
    from isaaclab_tasks.utils import get_checkpoint_path

    if os.path.isabs(load_run):
        log_root_path, load_run = os.path.dirname(load_run), os.path.basename(load_run)
    if other_dirs is None:
        return get_checkpoint_path(log_root_path, load_run, load_checkpoint)
    return get_checkpoint_path(log_root_path, load_run, load_checkpoint, other_dirs=other_dirs)


def play_run_dir(args_cli: argparse.Namespace, agent_cfg) -> tuple[str, str]:
    """Return ``(log_dir, resume_path)`` for playback; ``--load_run`` always replaces ``agent_cfg.load_run``.

    Raises:
        RuntimeError: Without ``--load_run`` unless ``--no_resume`` is given.
    """
    log_root_path = os.path.abspath(os.path.join(LOG_ROOT, agent_cfg.experiment_name))
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        return os.path.dirname(resume_path), resume_path
    if not args_cli.no_resume:
        raise RuntimeError(
            "\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            " a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
    return os.path.join(log_root_path, agent_cfg.run_name + "_play"), "model_scratch.pt"


def load_saved_yaml(filename: str) -> dict:
    """Load a ``dump_yaml`` file, including its ``!!python/object/apply:builtins.slice`` entries.

    ``yaml.full_load`` (Isaac Lab's ``load_yaml``) rejects those tags; this loader accepts only slices on top of it.
    """
    import yaml

    class _Loader(yaml.FullLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice",
        lambda loader, node: slice(*loader.construct_sequence(node)),
    )
    with open(filename) as f:
        return yaml.load(f, Loader=_Loader)


# -- in-memory meshes in the saved config


IN_MEMORY_MESH_PLACEHOLDER = "<in-memory mesh data: restored from the task config>"
"""What ``params/env.yaml`` holds in place of a ``MeshFileCfg`` in-memory mesh payload.

``MeshFileCfg.TrimeshObjectCfg.mesh`` is a trimesh object and ``MeshFileCfg.TriangleMeshCfg`` holds raw arrays;
neither has a YAML representation, so saving the config masks them and :func:`load_saved_cfgs` drops the
placeholder again: replay keeps the mesh the task config loads.
"""


def env_cfg_for_dump(env_cfg) -> dict:
    """Convert *env_cfg* into a dictionary that ``dump_yaml`` can write, masking in-memory meshes."""
    from isaaclab.utils import to_dict

    masked = _find_in_memory_meshes(env_cfg)
    for owner, field, _payload in masked:
        setattr(owner, field, IN_MEMORY_MESH_PLACEHOLDER)
    try:
        return to_dict(env_cfg)
    finally:
        for owner, field, payload in masked:
            setattr(owner, field, payload)


def _find_in_memory_meshes(node, found: list | None = None, visited: set | None = None) -> list:
    """Return ``(owner, field, payload)`` for every in-memory mesh payload below *node*.

    The mesh-source cfgs are matched by class name so this module stays importable without the simulator (see the
    module docstring); the payloads are the spawned triangle data, which no dictionary/YAML form exists for.
    """
    found = [] if found is None else found
    visited = set() if visited is None else visited
    if id(node) in visited:
        return found
    visited.add(id(node))
    kind = type(node).__name__
    if kind == "TrimeshObjectCfg":
        found.append((node, "mesh", node.mesh))
        return found
    if kind == "TriangleMeshCfg":
        found.append((node, "vertices", node.vertices))
        found.append((node, "faces", node.faces))
        if node.vertex_colors is not None:
            found.append((node, "vertex_colors", node.vertex_colors))
        return found
    if isinstance(node, dict):
        for value in node.values():
            if isinstance(value, (dict, list, tuple)) or hasattr(value, "__dataclass_fields__"):
                _find_in_memory_meshes(value, found, visited)
        return found
    if isinstance(node, (list, tuple)):
        for value in node:
            if isinstance(value, (dict, list, tuple)) or hasattr(value, "__dataclass_fields__"):
                _find_in_memory_meshes(value, found, visited)
        return found
    if not hasattr(node, "__dataclass_fields__"):
        return found
    for value in vars(node).values():
        if isinstance(value, (dict, list, tuple)) or hasattr(value, "__dataclass_fields__"):
            _find_in_memory_meshes(value, found, visited)
    return found


def _drop_in_memory_mesh_placeholders(data):
    """Return *data* without the masked in-memory-mesh entries, so replay keeps the task's own mesh."""
    if isinstance(data, dict):
        return {
            key: _drop_in_memory_mesh_placeholders(value)
            for key, value in data.items()
            if not (isinstance(value, str) and value == IN_MEMORY_MESH_PLACEHOLDER)
        }
    if isinstance(data, list):
        return [_drop_in_memory_mesh_placeholders(item) for item in data]
    return data


def _reject_in_memory_meshes_with_hydra(task: str, hydra_args: list[str], env_cfg=None) -> None:
    """Refuse residual Hydra arguments for a task whose env cfg holds in-memory meshes.

    ``register_task`` converts the composed env cfg into a dictionary for Hydra, which an in-memory mesh payload
    has no form for. ``env.``/``agent.`` scalars are applied without that conversion, so they keep working.

    Raises:
        ValueError: If a Hydra argument is given and *env_cfg* (or the task's own env cfg) holds in-memory meshes.
    """
    residual = [token for token in hydra_args if _reaches_hydra(token)]
    if not residual:
        return
    if env_cfg is None:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    if not _find_in_memory_meshes(env_cfg):
        return
    raise ValueError(
        f"Task '{task}' spawns in-memory meshes (MeshFileCfg.TrimeshObjectCfg/TriangleMeshCfg), which cannot pass"
        f" through Hydra's config store. Pass env.<path>=<value> or agent.<path>=<value> instead of {residual}."
    )


def load_saved_cfgs(args_cli: argparse.Namespace, log_dir: str) -> tuple[dict | None, dict | None]:
    """Load ``params/env.yaml`` and ``params/agent.yaml`` of *log_dir* when ``--env_cfg``/``--agent_cfg`` ask."""
    env_yaml = load_saved_yaml(os.path.join(log_dir, "params", "env.yaml")) if args_cli.env_cfg else None
    if env_yaml is not None:
        env_yaml = _drop_in_memory_mesh_placeholders(env_yaml)
    agent_yaml = load_saved_yaml(os.path.join(log_dir, "params", "agent.yaml")) if args_cli.agent_cfg else None
    return env_yaml, agent_yaml


def apply_play_cli(args_cli: argparse.Namespace, env_cfg, agent_cfg) -> None:
    """Apply the dedicated play CLI options, which win over the saved YAML and Hydra."""
    update_instinct_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.use_fabric = not args_cli.disable_fabric


def checkpoint_video_prefix(resume_path: str | None) -> str:
    """Return ``model_<iteration>`` for a checkpoint path, ``onnx`` without one."""
    return "onnx" if resume_path is None else f"model_{resume_path.split('_')[-1].split('.')[0]}"


# -- video and debugging


def pre_launch_video_config(
    env_cfg, args_cli: argparse.Namespace, *, video_interval: int = 0, step_offset: int = 0, prefix: str = "clip"
) -> None:
    """Replace the env's video recorders with one ``--video`` recorder. Call before ``launch_simulation``.

    ``launch_simulation`` resolves the recorder source against the ``--viz`` selection.
    """
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    if not args_cli.video:
        return
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source=args_cli.video,
            video_length=args_cli.video_length,
            video_interval=video_interval,
            step_offset=step_offset,
            output_filename_prefix=prefix,
        )
    ]


def apply_video_recording(env_cfg, args_cli: argparse.Namespace, output_dir: str) -> None:
    """Point the ``--video`` recorder at *output_dir*. Call inside ``launch_simulation``."""
    from isaaclab.utils.dict import print_dict

    if not args_cli.video:
        return
    for cfg in env_cfg.video_recorders:
        cfg.output_dir = output_dir
        print("[INFO] Recording videos.")
        print_dict(
            {
                "source": cfg.source,
                "output_dir": cfg.output_dir,
                "video_length": cfg.video_length,
                "video_interval": cfg.video_interval,
                "step_offset": cfg.step_offset,
            },
            nesting=4,
        )


def wait_for_debugger(enabled: bool) -> None:
    """Block until a debugpy client attaches on port 6789 when *enabled*."""
    if not enabled:
        return
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()
