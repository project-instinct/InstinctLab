# Project Instinct

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-main-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Overview

This repository is the environment side of [Project-Instinct](https://project-instinct.github.io/).

We aim at industralize Reinforcement Learning for Humanoid (legged robots) whole-body control.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.
- `Unified Ecosystem` This repository is a part of the Project-Instinct ecosystem, which includes the [instinct_rl](https://github.com/project-instinct/instinct_rl) and [instinct_onboard](https://github.com/project-instinct/instinct_onboard) repositories.
    - The core design of this ecosystem is to treat each experiment as a standalone structured folder, which start with a timestamp as a unique identifier.
    - Adding `--exportonnx` flag to the `play.py` script will export the policy as an ONNX model. After that, you should directly copy the logdir to the robot computer and use the `instinct_onboard` workflow to run the policy on the real robot.

**Keywords:** extension, template, isaaclab

## Warning
This codebase is under [CC BY-NC 4.0 license](LICENSE), with inherited license in IsaacLab. You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products or wrap the code for your own commercial purposes.

## Contributing
See our [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) for contribution guidelines. By contributing or submitting a pull request, you agree to transfer copyright ownership of your contributions to the project maintainers.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of acknowledged contributors.

## Installation

### Pinned runtime dependencies

This branch runs on Isaac Lab `main` inside Docker. The validated revisions are:

| Dependency | Version or revision |
|---|---|
| Python | `3.12` |
| Isaac Sim | `6.2.0-rc.14` (from the Isaac Lab base image) |
| Isaac Lab | `main` at `8c9b995cb95c74ec81498ae948ce4c1d4e20ae00` (package `41.0.1`) |
| Instinct-RL | `574c8d50cdd16618cf8f4e0df553f84759c1d837` (package `1.0.3`) |
| PyTorch | `2.11.0+cu130` (needs NVIDIA driver 580 or newer) |
| Warp | `1.17.0` |
| Newton | `1.8.0.dev0` |
| NumPy | `2.5.1` |
| Gymnasium | `1.3.0` |

Use the full source SHAs rather than the floating branch heads when reproducing an experiment.

The conda environment for Isaac Lab EA `release/3.0.0` (Isaac Sim 6.0.1) does not run this branch.

### Docker workflow

Place the three repositories side by side, with `logs` as a relative symlink into `~/Data/instinctlab_logs`:

```text
~/workspace/instinctlab_ws/
├── isaaclab/        # Isaac Lab, branch main at the SHA above
├── instinct_rl/     # Instinct-RL at the SHA above
└── instinctlab/     # this repository
    └── logs -> ../../../Data/instinctlab_logs
```

1. Build the Isaac Lab base image from the Isaac Lab clone and tag it with its SHA:

    ```bash
    cd ~/workspace/instinctlab_ws/isaaclab
    ./docker/container.py build base
    docker tag isaac-lab-base:latest isaac-lab-base:$(git rev-parse --short=9 HEAD)
    ```

2. Set `ISAACLAB_BASE_IMAGE` in `docker/.env.base` to that tag, then build the InstinctLab image (`instinctlab:3main`):

    ```bash
    cd ~/workspace/instinctlab_ws/instinctlab/docker
    docker compose --env-file .env.base build instinctlab
    ```

3. Run commands in a throwaway container. The three clones are bind-mounted at their host paths, together with
   `~/Data` and `~/Datasets` (read-only); caches live in named volumes.

    ```bash
    docker compose --env-file .env.base run --rm --no-deps instinctlab -c \
        "isaaclab -p scripts/instinct_rl/train.py --task Instinct-Shadowing-WholeBody-Plane-G1-v0 --viz none"
    ```

    Use `isaaclab -p` for anything that starts the simulator; plain `python` is enough for `pytest` and `--help`.
    Kitless runs (Newton, `--viz none`) may also use plain `python`.

- `instinct_rl` and `instinctlab` are installed with `uv pip install -e` into the base image's `/opt/isaaclab-venv`.
  Never run an exact `uv sync` in the container: it removes these editable installs.
- URDF robots convert to USD once per asset cache (`~/.cache/instinctlab`, a named volume). Under plain `python` the
  image's standalone importer (`isaacsim-asset-isolated`, Isaac Lab's `importers` extra) converts without Kit; under
  `isaaclab -p` Isaac Sim's own `isaacsim` package hides it, so a windowless Kit app converts instead. Both give the
  same USD.
- Source edits need no rebuild. Rebuild both images when Isaac Lab's `pyproject.toml`/`uv.lock`, or either `docker/`
  directory, changes.

### Datasets

Task defaults read datasets from `~/Datasets`; `~` is expanded when the files are read, so CLI overrides such as
`env.scene.motion_reference.motion_buffers.<name>.path=...` may use it too.

| Data | Default path | Source |
|---|---|---|
| Deep Whole-body Parkour motions | `~/Datasets/Deep-Wholebody-Parkour_Dataset` | [Google Drive](https://drive.google.com/drive/folders/1dpSYHVhP1kFdBK5ixh_f_3g0vcbyC08f?usp=drive_link) |
| Hiking in the Wild (parkour) | `~/Datasets/hiking-in-the-wild_Data&Model/data&model/parkour_motion_reference` | project release |
| OMOMO (perceptive HOI) | `~/Datasets/OMOMO/retargeted`, objects in `~/Datasets/OMOMO/data/captured_objects` | OMOMO release |
| LAFAN1 (BeyondMimic) | `~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz` | retargeted LAFAN1 |

A cfg's `__post_init__` derives some fields from these paths and does not run again after a CLI override; the
affected fields are documented in the motion-file and terrain cfg docstrings.

### Validated matrix

Every family below was checked on PhysX and Newton (layout snapshot against the Isaac Lab EA baseline, full/partial
and int32/int64 resets, a short train and resume): locomotion (`Instinct-Locomotion-Flat-G1-v0`), whole-body
shadowing (`Instinct-Shadowing-WholeBody-Plane-G1-v0`), BeyondMimic (`Instinct-BeyondMimic-Plane-G1-v0`), perceptive
shadowing (`Instinct-Perceptive-Shadowing-G1-v0`), perceptive VAE (`Instinct-Perceptive-Vae-G1-v0`), perceptive HOI
(`Instinct-Perceptive-HOI-Shadowing-G1-v0`) and parkour (`Instinct-Parkour-Target-Amp-G1-v0`), each with its
`-Newton-v0` variant. Behavior differences against Isaac Lab EA are listed in
[`agents/UPGRADE_ISAACLAB300EA_TO_ISAACLAB300GA_LOG.md`](agents/UPGRADE_ISAACLAB300EA_TO_ISAACLAB300GA_LOG.md).

### Unsupported combinations

- Physics backend `ovphysx`: only `physx` and `newton` are supported. Selecting `ovphysx` (`--physics ovphysx`) raises
  `NotImplementedError` from `instinctlab.utils.backend_dispatch` instead of being mapped to PhysX.
- NVIDIA drivers older than 580: the image's PyTorch is built for CUDA 13.0 and cannot use the GPU.
- Distributed (multi-GPU) training is not validated on this branch.

## Documentation of Critical Components

- [Instinct-RL Documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [InstinctLab Documentation](https://github.com/project-instinct/instinctlab/blob/main/DOCS.md)

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

To make the `pre-commit` run automatically on every commit, you can use the following command in your repository:

```bash
pre-commit install
```

## Scripts Folder Conventions

The `scripts/` directory is for entry scripts only. A script may import other modules, but code under
`scripts/` must **never be imported by other code**. The single exception is `cli_args.py`, which
originates from the `rsl_rl` workflow and is kept as-is. If you find code that should be reused, put
it in a `pip`-installable package (e.g. under `source/<package>/`) instead of importing it across
`scripts/`.

## Train your own projects

***To preserve your code development and progress. PLEASE create your own repository as an individual project by referring to https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html***

And copy `scripts/instinct_rl` to your own repository.

### Or you are just to stubborn and want to fork and directly modify the code in this repo.

- Please create a new folder in the `source/instinctlab/instinctlab/tasks` directory. The name of the folder should be your project name. Inside the folder, DO add `__init__.py` in each level of the subfolders. (Many people tend to forget this step and could not find the supposely registered tasks.)

- We inherit the manager based RL env from IsaacLab to add new features. DO use `instinctlab.envs:InstinctRlEnv` as the entry_point in the `gym.register` call. For example, if you want to add a new task, you can use the following code:

```python
import gymnasium as gym
from . import agents
task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"
gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)
```

## Troubleshooting

### Missing `libGL.so.1` on headless servers

When running on a headless server (e.g., cloud VMs without a display), Isaac Sim requires OpenGL system libraries even in headless mode. Install them with:

```bash
apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/instinctlab"
    ]
}
```
