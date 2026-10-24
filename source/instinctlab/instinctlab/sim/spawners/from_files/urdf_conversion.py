# Copyright (c) 2024, Instinct Lab.
# SPDX-License-Identifier: MIT

"""URDF to USD conversion in a separate, Kit-free Python process.

The conversion uses Isaac Lab's ``UrdfConverter`` with the standalone ``isaacsim-asset-isolated`` importer (Isaac Lab's
``importers`` extra), followed by InstinctLab's flattening and inertia check. It always runs in a child process of the
virtual environment's Python, so it never depends on Kit: inside a Kit process (``isaaclab -p``), Isaac Sim's own
``isaacsim`` package and USD build would hide the standalone importer and its USD libraries.

Run as ``python -m instinctlab.sim.spawners.from_files.urdf_conversion <cfg.pkl> <output dir>``; the parent side is
:func:`convert_urdf`.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

# Variables Isaac Sim's python.sh exports to run Kit. With them, the child would load Kit's Python packages and USD
# libraries instead of the virtual environment's.
_KIT_ENVIRONMENT = ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "CARB_APP_PATH", "ISAAC_PATH", "EXP_PATH")


def _kit_free_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _KIT_ENVIRONMENT}


def convert_urdf(cfg) -> str:
    """Convert a URDF spawn configuration in a Kit-free child process.

    Args:
        cfg: The :class:`~instinctlab.sim.spawners.from_files.from_files_cfg.UrdfFileCfg` to convert.

    Returns:
        Path of the flattened USD, in a new temporary directory.

    Raises:
        RuntimeError: If the child process fails; the message includes the end of its output.
    """
    output_dir = tempfile.mkdtemp(prefix=f"instinctlab_urdf_{Path(cfg.asset_path).stem}_")
    cfg_path = os.path.join(output_dir, "cfg.pkl")
    with open(cfg_path, "wb") as file:
        pickle.dump(cfg, file)
    result = subprocess.run(
        [sys.executable, "-m", __name__, cfg_path, output_dir],
        env=_kit_free_environment(),
        capture_output=True,
        text=True,
    )
    flat_usd_path = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not os.path.isfile(flat_usd_path):
        output = "\n".join((result.stdout + result.stderr).strip().splitlines()[-30:])
        raise RuntimeError(
            f"Kit-free URDF conversion of '{cfg.asset_path}' failed (exit code {result.returncode}). It needs Isaac"
            f" Lab's 'importers' extra (isaacsim-asset-isolated) in this Python environment.\n{output}"
        )
    return flat_usd_path


def main() -> None:
    from isaaclab.sim import converters
    from isaaclab.utils.version import has_kit

    from .from_files import _check_urdf_inertia, _flatten_urdf_geometry

    cfg_path, output_dir = sys.argv[1:3]
    if has_kit():
        raise RuntimeError("The URDF conversion process must run without Kit.")
    with open(cfg_path, "rb") as file:
        cfg = pickle.load(file)
    # Isaac Sim's importer 3.0 removed the capsule option; _flatten_urdf_geometry applies the legacy mapping instead.
    converter_cfg = cfg.replace(replace_cylinders_with_capsules=False, usd_dir=output_dir, force_usd_conversion=True)
    usd_path = converters.UrdfConverter(converter_cfg).usd_path
    flat_usd_path = _flatten_urdf_geometry(usd_path, cfg.replace_cylinders_with_capsules)
    _check_urdf_inertia(flat_usd_path, cfg.asset_path)
    # the parent reads the last stdout line
    print(flat_usd_path, flush=True)


if __name__ == "__main__":
    main()
