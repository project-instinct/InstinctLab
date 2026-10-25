"""Physics backend presets and per-backend task registration.

Every backend-specific value of a task cfg is a :class:`~isaaclab_tasks.utils.hydra.PresetCfg` built by
:func:`backend_preset`, with one entry per backend. Selecting a backend selects all of them at once, either through
the backend's task id (``*-Newton-v0``, ``*-OvPhysx-v0``), which registers the same env cfg class with the backend resolved, or with
the ``physics=<preset>`` selector on the PhysX task id. The Isaac Lab preset helpers are imported on use, so
registering the tasks stays light.
"""

from __future__ import annotations

import gymnasium as gym
import importlib

# Backend name -> (task id suffix, preset name). The PhysX entry is also the ``default`` preset.
BACKENDS = {
    "physx": ("", "physx"),
    "newton": ("-Newton", "newton_mjwarp"),
    "ovphysx": ("-OvPhysx", "ovphysx"),
}

_SAME = object()


def backend_preset(physx, *, newton=_SAME, ovphysx=_SAME):
    """Return a preset holding a value per physics backend.

    Args:
        physx: The PhysX value, also used as the default.
        newton: The Newton value. Defaults to the PhysX value.
        ovphysx: The OvPhysX value. Defaults to the PhysX value: OvPhysX is PhysX without Kit.

    Returns:
        A preset with the ``default``, ``physx``, ``newton_mjwarp`` and ``ovphysx`` entries.
    """
    from isaaclab_tasks.utils.hydra import preset

    return preset(
        default=physx,
        physx=physx,
        newton_mjwarp=physx if newton is _SAME else newton,
        ovphysx=physx if ovphysx is _SAME else ovphysx,
    )


def physics_preset(physx, *, newton):
    """Return the ``sim.physics`` preset; the OvPhysX entry carries the PhysX cfg's values over.

    Args:
        physx: The :class:`~isaaclab_physx.physics.PhysxCfg`.
        newton: The :class:`~isaaclab_newton.physics.NewtonCfg`.

    Raises:
        ValueError: If *physx* sets a field that :class:`~isaaclab_ov.physics.OvPhysxCfg` does not have.
    """
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    defaults = PhysxCfg()
    changed = {
        name: getattr(physx, name)
        for name in physx.__dataclass_fields__
        if getattr(physx, name) != getattr(defaults, name)
    }
    dropped = sorted(set(changed) - set(OvPhysxCfg.__dataclass_fields__))
    if dropped:
        raise ValueError(f"OvPhysxCfg has no field for the PhysX settings: {', '.join(dropped)}.")
    return backend_preset(physx, newton=newton, ovphysx=OvPhysxCfg(**changed))


def backend_viewers(kit_viewer, *, newton=None):
    """Return the ``sim.visualizer_cfgs`` preset: the Kit viewer, Newton's viewer and ``viser`` on OvPhysX.

    Args:
        kit_viewer: The Kit :class:`~isaaclab.visualizers.VisualizerCfg`.
        newton: The Newton viewer. Defaults to the Kit viewer.
    """
    from isaaclab_visualizers.viser import ViserBackendCfg

    viser = kit_viewer.replace(backend=ViserBackendCfg())
    return backend_preset([kit_viewer], newton=[newton or kit_viewer], ovphysx=[viser])


def select_backend(cfg, backend: str):
    """Resolve every backend preset of *cfg* in place for *backend* and return it."""
    from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets

    preset_name = BACKENDS[backend][1]
    presets = collect_presets(cfg)
    missing = sorted(path for path, fields in presets.items() if preset_name not in fields)
    if missing:
        raise ValueError(f"Presets without a '{preset_name}' entry: {', '.join(missing)}.")
    return resolve_presets(cfg, [preset_name])


def backend_entry_point(entry_point: str, backend: str):
    """Return an env cfg entry point that builds *entry_point* (``module:Class``) with *backend* selected."""

    def make_cfg():
        module_name, class_name = entry_point.split(":")
        return select_backend(getattr(importlib.import_module(module_name), class_name)(), backend)

    make_cfg.__qualname__ = f"{entry_point.split(':')[-1]}[{backend}]"
    return make_cfg


def register_backend_tasks(
    name: str, env_cfg_entry_point: str, play_env_cfg_entry_point: str, entry_point: str, **agent_entry_points: str
):
    """Register the train and play task ids of *name* for every physics backend.

    The ids are ``<name>[-<Backend>]-v0`` and ``<name>[-<Backend>]-Play-v0``, without a suffix for PhysX. The PhysX ids
    keep the presets, so ``physics=`` and ``presets=`` still select among them.
    """
    for backend, (suffix, _) in BACKENDS.items():
        for play, cfg_entry_point in ((False, env_cfg_entry_point), (True, play_env_cfg_entry_point)):
            gym.register(
                id=f"{name}{suffix}{'-Play' if play else ''}-v0",
                entry_point=entry_point,
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": (
                        cfg_entry_point if backend == "physx" else backend_entry_point(cfg_entry_point, backend)
                    ),
                    **agent_entry_points,
                },
            )
