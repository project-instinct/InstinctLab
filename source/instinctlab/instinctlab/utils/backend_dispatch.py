from __future__ import annotations

import importlib
from typing import Any

_SUPPORTED_BACKENDS = ("physx", "newton")


def active_physics_backend() -> str:
    """Return the active physics backend without importing a backend package."""
    # Imported lazily so that importing this module does not load ``isaaclab.sim`` (and USD) before launch.
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("A simulation context must exist before constructing a backend component.")

    backend = sim.backend_manager.backend_name
    if backend not in _SUPPORTED_BACKENDS:
        # Exact comparison: ``"physx" in backend`` would also match "ovphysx".
        supported = ", ".join(_SUPPORTED_BACKENDS)
        raise NotImplementedError(f"Physics backend '{backend}' is not supported. Supported backends: {supported}.")
    return backend


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
