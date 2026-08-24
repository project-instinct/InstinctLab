from __future__ import annotations

import importlib
from typing import Any

from isaaclab.sim import SimulationContext


def active_physics_backend() -> str:
    """Return the active physics backend without importing a backend package."""
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("A simulation context must exist before constructing a backend component.")

    # ``physics_manager`` may still be Isaac Lab's lazy ``ResolvableString`` here.
    # Its ``__name__`` metadata is available without resolving (and therefore
    # without importing) the backend implementation.
    manager_name = sim.physics_manager.__name__.lower()
    if manager_name.startswith("newton"):
        return "newton"
    if manager_name.startswith("physx"):
        return "physx"
    raise RuntimeError(f"Unsupported physics manager for backend component construction: {manager_name}")


def create_backend_component(cfg: Any, class_paths: dict[str, str]) -> Any:
    """Construct only the component implementation for the active physics backend."""
    backend = active_physics_backend()
    try:
        class_path = class_paths[backend]
    except KeyError:
        supported = ", ".join(sorted(class_paths))
        raise RuntimeError(
            f"Component does not support the '{backend}' backend. Supported backends: {supported}."
        ) from None

    module_name, class_name = class_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    component_class = getattr(module, class_name)
    return component_class(cfg)
