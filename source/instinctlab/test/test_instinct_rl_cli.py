"""Tests for :mod:`instinctlab.utils.instinct_rl_cli`.

These tests cover the command-line parsing, config layering, log naming and checkpoint-resolution helpers
used by ``scripts/instinct_rl/{train,play}.py``. They run without launching a simulator: the parsing and
naming helpers only need the standard library, and the config-composition tests register a tiny dummy gym
task whose configclasses are instantiated (never ``gym.make``-ed).
"""

from __future__ import annotations

import argparse
import gymnasium as gym
from datetime import datetime

import pytest

from isaaclab.utils import configclass, to_dict
from isaaclab_tasks.utils import resolve_task_config

from instinctlab.utils import instinct_rl_cli as cli

# ---------------------------------------------------------------------------
# Minimal configclasses for the config-composition tests.
# ---------------------------------------------------------------------------


@configclass
class _SceneCfg:
    num_envs: int = 64


@configclass
class _SimCfg:
    use_fabric: bool = True
    physics: object | None = None


@configclass
class _EventsCfg:
    reset: _SceneCfg = _SceneCfg()
    push_robot: _SceneCfg | None = None


@configclass
class _EnvCfg:
    scene: _SceneCfg = _SceneCfg()
    seed: int = 0
    sim: _SimCfg = _SimCfg()
    events: _EventsCfg = _EventsCfg()
    log_dir: str = ""
    run_name: str = ""
    played: bool = False

    def play_mode(self) -> None:
        self.played = True


@configclass
class _AgentCfg:
    seed: int = 42
    resume: bool = False
    load_run: str | None = ".*"
    load_checkpoint: str = "model_.*.pt"
    run_name: str = ""
    experiment_name: str = "myexp"
    device: str = "cuda:0"


@pytest.fixture
def dummy_task() -> str:
    """Register a tiny gym task exposing fresh env/agent configs and clean it up afterwards."""
    task = "Instinct-CliDummy-v0"
    gym.register(
        id=task,
        entry_point="dummy:Env",
        kwargs={
            "env_cfg_entry_point": lambda: _EnvCfg(),
            cli.AGENT_ENTRY_POINT: lambda: _AgentCfg(),
        },
    )
    try:
        yield task
    finally:
        del gym.registry[task]


# ---------------------------------------------------------------------------
# 1. Command-line parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["physics=newton"], ["physics=newton"]),
        (["--physics", "newton"], ["physics=newton"]),
        (["presets=foo"], ["presets=foo"]),
        (["renderer=x"], ["renderer=x"]),
        (["--viz", "kit"], ["visualizer=kit"]),
    ],
)
def test_preset_passthrough(argv, expected):
    """Preset selectors reach the returned overrides; ``--viz`` becomes ``visualizer=``."""
    parser = cli.build_play_parser()
    cli.add_play_runtime_args(parser)
    _, overrides = cli.parse_cli(parser, argv, require_run_name=False)
    assert overrides == expected


def test_hydra_override_forms_accepted():
    """Hydra-style path overrides and deletions are accepted and passed through unchanged."""
    argv = ["+env.a.b=1", "++agent.c=2", "~env.d", "~agent.e=3", "env.x=1", "agent.y=z"]
    parser = cli.build_play_parser()
    cli.add_play_runtime_args(parser)
    _, overrides = cli.parse_cli(parser, argv, require_run_name=False)
    assert overrides == argv


@pytest.mark.parametrize("bad", ["--bogus", "foo=bar", "envx=1", "env.=1", "hello"])
def test_malformed_residual_token_rejected(bad):
    """Residual tokens that are not Hydra-style or preset selectors abort with an argparse error."""
    parser = cli.build_train_parser()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli(parser, [bad], require_run_name=True)
    assert exc_info.value.code == 2


def test_malformed_error_wins_over_missing_run_name(capsys):
    """A malformed token is reported even when ``--run_name`` is also missing."""
    parser = cli.build_train_parser()
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_cli(parser, ["env.x=1", "--bogus"], require_run_name=True)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err
    assert "is required" not in err  # the run-name rule must not fire


@pytest.mark.parametrize(
    ("argv", "ok"),
    [
        (["env.x=1"], False),
        (["env.x=1", "--run_name", ""], False),
        (["env.x=1", "--run_name", "v"], True),
        (["env.x=1", "--run_name=v"], True),
        (["agent.run_name=v"], False),
        ([], True),
        (["--viz", "kit"], True),
        (["physics=newton"], False),
        (["--num_envs", "4"], True),
    ],
)
def test_train_run_name_rule(argv, ok):
    """Training requires a non-empty ``--run_name`` exactly when Hydra-style tokens are present."""
    parser = cli.build_train_parser()
    if ok:
        args, _ = cli.parse_cli(parser, argv, require_run_name=True)
        assert isinstance(args, argparse.Namespace)
    else:
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_cli(parser, argv, require_run_name=True)
        assert exc_info.value.code == 2


def test_play_never_requires_run_name():
    """Playback never enforces the training run-name rule."""
    parser = cli.build_play_parser()
    cli.add_play_runtime_args(parser)
    _, overrides = cli.parse_cli(parser, ["env.x=1", "physics=newton"], require_run_name=False)
    assert overrides == ["env.x=1", "physics=newton"]


def test_video_defaults():
    """``--video_length`` defaults differ between train and play."""
    train_parser = cli.build_train_parser()
    targs, _ = cli.parse_cli(train_parser, [], require_run_name=True)
    assert targs.video is None
    assert targs.video_length == 200
    assert targs.video_interval == 2000

    play_parser = cli.build_play_parser()
    cli.add_play_runtime_args(play_parser)
    pargs, _ = cli.parse_cli(play_parser, [], require_run_name=False)
    assert pargs.video is None
    assert pargs.video_length == 3000


def test_video_without_value_gives_viz():
    parser = cli.build_play_parser()
    cli.add_play_runtime_args(parser)
    args, overrides = cli.parse_cli(parser, ["--video"], require_run_name=False)
    assert args.video == "viz"
    assert overrides == []


def test_video_keeps_explicit_value():
    parser = cli.build_play_parser()
    cli.add_play_runtime_args(parser)
    args, overrides = cli.parse_cli(parser, ["--video", "sensor:cam"], require_run_name=False)
    assert args.video == "sensor:cam"
    assert overrides == []


def test_video_followed_by_override_still_parses():
    """``--video`` before an override token is rewritten to ``--video=viz`` so the token survives."""
    parser = cli.build_play_parser()
    cli.add_play_runtime_args(parser)
    args, overrides = cli.parse_cli(parser, ["--video", "env.x=1"], require_run_name=False)
    assert args.video == "viz"
    assert overrides == ["env.x=1"]


def test_is_hydra_style():
    """Hydra-style detection covers path overrides, deletions and preset selectors only."""
    assert cli.is_hydra_style("+env.x=1")
    assert cli.is_hydra_style("++agent.y=2")
    assert cli.is_hydra_style("~env.x")
    assert cli.is_hydra_style("~agent.y=3")
    assert cli.is_hydra_style("env.x=1")
    assert cli.is_hydra_style("agent.y=z")
    assert cli.is_hydra_style("physics=newton")
    assert cli.is_hydra_style("renderer=x")
    assert cli.is_hydra_style("presets=foo")
    assert not cli.is_hydra_style("visualizer=kit")
    assert not cli.is_hydra_style("--bogus")
    assert not cli.is_hydra_style("foo=bar")
    assert not cli.is_hydra_style("envx=1")
    assert not cli.is_hydra_style("env.=1")


def test_with_hydra_output_suppressed():
    """Suppression args are added only when a token would reach ``hydra.main``."""
    suppressed = list(cli.HYDRA_NO_OUTPUT_ARGS)
    for token in ("+env.x=1", "~env.x", "env.x+=1"):
        assert cli.with_hydra_output_suppressed([token]) == [token, *suppressed]
    for token in ("env.x=1", "agent.y=2", "physics=newton", "visualizer=kit"):
        assert cli.with_hydra_output_suppressed([token]) == [token]


# ---------------------------------------------------------------------------
# 2. Log naming
# ---------------------------------------------------------------------------


def test_train_log_dir_name():
    ts = datetime(2026, 1, 1, 12, 0, 0)
    assert cli.train_log_dir_name(None, None, timestamp=ts) == "20260101_120000"
    assert cli.train_log_dir_name("env", "agent", timestamp=ts) == "20260101_120000_env_agent"
    assert (
        cli.train_log_dir_name("env", "agent", "/x/20260101_120000_foo/model_10.pt", timestamp=ts)
        == "20260101_120000_env_agent_from20260101_120000"
    )


def test_resume_suffix():
    assert cli.resume_suffix("/x/20260101_120000_foo/model_10.pt") == "_from20260101_120000"
    assert cli.resume_suffix("folder/model_10.pt") == ""


def test_checkpoint_video_prefix():
    assert cli.checkpoint_video_prefix("/a/model_1500.pt") == "model_1500"
    assert cli.checkpoint_video_prefix(None) == "onnx"


# ---------------------------------------------------------------------------
# 3. Run directory and checkpoint resolution
# ---------------------------------------------------------------------------


def test_checkpoint_path_absolute_and_regex(tmp_path):
    log_root = tmp_path / "logs" / "instinct_rl" / "myexp"
    run_a = log_root / "20260101_120000_runA"
    run_b = log_root / "20260101_130000_runB"
    for run, checkpoints in ((run_a, ("model_0.pt", "model_10.pt")), (run_b, ("model_50.pt",))):
        run.mkdir(parents=True)
        for name in checkpoints:
            (run / name).write_text("")

    # An absolute load_run resolves inside that folder and picks the latest checkpoint.
    assert cli.checkpoint_path(str(log_root), str(run_a), "model_.*.pt") == str(run_a / "model_10.pt")
    # A relative regex load_run picks the alphabetically-latest matching run.
    assert cli.checkpoint_path(str(log_root), r"20260101_1.*", "model_.*.pt") == str(run_b / "model_50.pt")


def test_play_run_dir_with_load_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "logs" / "instinct_rl" / "myexp" / "20260101_120000_foo"
    run.mkdir(parents=True)
    (run / "model_10.pt").write_text("")

    args = argparse.Namespace(load_run="20260101_120000_foo", no_resume=False)
    agent_cfg = _AgentCfg(experiment_name="myexp", run_name="foo", load_run="stale")

    log_dir, resume_path = cli.play_run_dir(args, agent_cfg)
    assert agent_cfg.load_run == "20260101_120000_foo"  # forced from args_cli
    assert resume_path == str(run / "model_10.pt")
    assert log_dir == str(run)


def test_play_run_dir_requires_load_run_or_no_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(load_run=None, no_resume=False)
    with pytest.raises(RuntimeError):
        cli.play_run_dir(args, _AgentCfg(experiment_name="myexp", run_name="foo"))


def test_play_run_dir_no_resume_uses_scratch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(load_run=None, no_resume=True)
    log_dir, resume_path = cli.play_run_dir(args, _AgentCfg(experiment_name="myexp", run_name="foo"))
    assert resume_path == "model_scratch.pt"
    assert log_dir == str(tmp_path / "logs" / "instinct_rl" / "myexp" / "foo_play")


# ---------------------------------------------------------------------------
# 4. Config composition precedence
# ---------------------------------------------------------------------------


def test_play_yaml_beats_task_default(dummy_task):
    env_cfg, agent_cfg = cli.resolve_play_cfg(
        dummy_task, [], env_yaml={"scene": {"num_envs": 128}, "seed": 7}, agent_yaml={"experiment_name": "from_yaml"}
    )
    assert env_cfg.scene.num_envs == 128
    assert env_cfg.seed == 7
    assert agent_cfg.experiment_name == "from_yaml"


def test_load_saved_yaml_reads_dump_yaml_slices(tmp_path):
    """``dump_yaml`` writes slices as ``!!python/object/apply:builtins.slice``; the saved-run loader reads them."""
    from isaaclab.utils.io import dump_yaml

    data = {"joint_ids": slice(None), "window": slice(1, 5, 2), "pos": (0.0, 1.0)}
    dump_yaml(str(tmp_path / "env.yaml"), data)
    assert "builtins.slice" in (tmp_path / "env.yaml").read_text()
    assert cli.load_saved_yaml(str(tmp_path / "env.yaml")) == data


def test_play_yaml_keeps_play_disabled_terms(dummy_task):
    """A term the play cfg sets to None stays None; its siblings are still layered."""
    env_yaml = {"events": {"reset": {"num_envs": 3}, "push_robot": {"num_envs": 5}}}
    env_cfg, _ = cli.resolve_play_cfg(dummy_task, [], env_yaml=env_yaml)
    assert env_cfg.events.reset.num_envs == 3
    assert env_cfg.events.push_robot is None


def test_play_yaml_unknown_key_still_raises(dummy_task):
    with pytest.raises(KeyError):
        cli.resolve_play_cfg(dummy_task, [], env_yaml={"events": {"not_a_term": {"num_envs": 1}}})


def test_play_yaml_materializes_launch_resolved_physx(dummy_task):
    """A task without ``sim.physics`` gets the saved PhysX settings on a fresh ``PhysxCfg``."""
    from isaaclab_physx.physics import PhysxCfg

    physics = {"class_type": "isaaclab_physx.physics.physx_manager:PhysxManager", "gpu_max_rigid_contact_count": 2**20}
    env_cfg, _ = cli.resolve_play_cfg(dummy_task, [], env_yaml={"sim": {"physics": physics}})
    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert env_cfg.sim.physics.gpu_max_rigid_contact_count == 2**20


def test_play_yaml_other_physics_needs_selector(dummy_task):
    physics = {"class_type": "isaaclab_newton.physics.newton_manager:NewtonManager"}
    with pytest.raises(ValueError, match="physics= selector"):
        cli.resolve_play_cfg(dummy_task, [], env_yaml={"sim": {"physics": physics}})


def test_play_scalar_override_beats_yaml(dummy_task):
    env_cfg, _ = cli.resolve_play_cfg(dummy_task, ["env.scene.num_envs=256"], env_yaml={"scene": {"num_envs": 128}})
    assert env_cfg.scene.num_envs == 256


def test_play_hydra_composed_beats_yaml(dummy_task, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_cfg, _ = cli.resolve_play_cfg(dummy_task, ["++env.scene.num_envs=256"], env_yaml={"scene": {"num_envs": 128}})
    assert env_cfg.scene.num_envs == 256


def test_hydra_helpers_do_not_write_outputs_dir(dummy_task, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_cfg, _ = cli.resolve_play_cfg(dummy_task, ["++env.scene.num_envs=256"], env_yaml={"scene": {"num_envs": 128}})
    assert env_cfg.scene.num_envs == 256
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / ".hydra").exists()
    assert not list(tmp_path.glob("*.log"))


def test_play_no_yaml_matches_resolve_task_config(dummy_task):
    env_a, agent_a = cli.resolve_play_cfg(dummy_task, [])
    env_b, agent_b = resolve_task_config(dummy_task, cli.AGENT_ENTRY_POINT, play_mode=True, overrides=[])
    assert to_dict(env_a) == to_dict(env_b)
    assert to_dict(agent_a) == to_dict(agent_b)
    assert env_a.played and env_b.played


def test_resolve_train_cfg_applies_overrides(dummy_task):
    env_cfg, _ = cli.resolve_train_cfg(dummy_task, ["env.scene.num_envs=512"])
    assert env_cfg.scene.num_envs == 512


def test_apply_play_cli_beats_yaml_and_hydra(dummy_task):
    env_yaml = {"scene": {"num_envs": 128}, "seed": 7, "sim": {"use_fabric": False}}
    env_cfg, agent_cfg = cli.resolve_play_cfg(dummy_task, ["env.scene.num_envs=256"], env_yaml=env_yaml)
    assert env_cfg.scene.num_envs == 256

    args = argparse.Namespace(
        num_envs=32, seed=99, disable_fabric=True, resume=None, load_run=None, checkpoint=None, run_name=None
    )
    cli.apply_play_cli(args, env_cfg, agent_cfg)
    assert env_cfg.scene.num_envs == 32
    assert agent_cfg.seed == 99
    assert env_cfg.seed == 99
    assert env_cfg.sim.use_fabric is False


def test_play_config_composed_twice(dummy_task, tmp_path, monkeypatch):
    """Play resolves twice (with and without YAML); Hydra composition must survive both calls."""
    monkeypatch.chdir(tmp_path)
    overrides = ["++env.scene.num_envs=256"]

    env_cfg, agent_cfg = cli.resolve_play_cfg(dummy_task, overrides)
    assert env_cfg.scene.num_envs == 256

    env_cfg, agent_cfg = cli.resolve_play_cfg(
        dummy_task, overrides, env_yaml={"scene": {"num_envs": 128}}, agent_yaml={"seed": 7}
    )
    assert env_cfg.scene.num_envs == 256  # Hydra override still wins over the saved YAML
    assert agent_cfg.seed == 7
    assert not (tmp_path / "outputs").exists()
