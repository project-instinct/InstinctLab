"""Backend-neutral helpers for resolving USD prim expressions."""

from __future__ import annotations

from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source


def resolve_articulation_root_expression(prim_path: str) -> str:
    """Resolve the single articulation root below an asset prim expression."""
    from pxr import UsdPhysics

    matches = resolve_matching_prims_from_source(prim_path)
    if not matches:
        raise RuntimeError(f"No asset prim found at path expression: {prim_path}")

    asset_prim, asset_expr = matches[0]
    asset_path = asset_prim.GetPath().pathString
    articulation_roots = get_all_matching_child_prims(
        asset_path,
        predicate=lambda prim: bool(prim.HasAPI(UsdPhysics.ArticulationRootAPI)),
        traverse_instance_prims=False,
    )
    if len(articulation_roots) != 1:
        matched_paths = [prim.GetPath().pathString for prim in articulation_roots]
        raise RuntimeError(
            f"Expected exactly one ArticulationRootAPI prim below '{asset_path}' "
            f"(resolved from '{prim_path}'), found {len(articulation_roots)}: {matched_paths}."
        )

    root_path = articulation_roots[0].GetPath().pathString
    return asset_expr + root_path[len(asset_path) :]


def _configured_order(prim_path: str, kind: str):
    """The ``ArticulationCfg.<kind>_ordering`` of the scene articulation at *prim_path*, or None."""
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    # The environment registers its scene with the simulation context before the physics starts.
    scene = getattr(sim, "_interactive_scene", None) if sim is not None else None
    if scene is None:
        return None
    for articulation in scene.articulations.values():
        if articulation.cfg.prim_path == prim_path:
            ordering = getattr(articulation.cfg, f"{kind}_ordering")
            if ordering is not None and (isinstance(ordering, str) or not isinstance(ordering, (list, tuple))):
                raise NotImplementedError(f"Only explicit {kind} name orders are supported here, got {ordering!r}.")
            return None if ordering is None else list(ordering)
    return None


def articulation_name_order(prim_path: str, kind: str, backend_names: list[str]) -> list[str]:
    """Return the joint or body order that the scene articulation at *prim_path* exposes.

    Components that read an articulation through their own backend views (e.g. the motion reference) use this to
    present the same order as the :class:`~isaaclab.assets.Articulation` built from the same prim: its
    ``ArticulationCfg.joint_ordering`` / ``body_ordering`` when set, else the backend order.

    Args:
        prim_path: The articulation prim expression, as formatted by the interactive scene.
        kind: ``"joint"`` or ``"body"``.
        backend_names: The names in the component's backend view order.

    Returns:
        The names in the articulation's public order.

    Raises:
        NotImplementedError: If the articulation uses a symbolic ordering convention instead of a name list.
        ValueError: If the configured order does not name every backend joint or body exactly once.
    """
    ordering = _configured_order(prim_path, kind)
    if ordering is None:
        return list(backend_names)
    if sorted(ordering) != sorted(backend_names):
        raise ValueError(
            f"The {kind}_ordering of '{prim_path}' must name each backend {kind} once; backend names: {backend_names}."
        )
    return ordering


def sort_by_articulation_order(prim_path: str, kind: str, names: list[str]) -> list[str]:
    """Sort a subset of the joints or bodies of the scene articulation at *prim_path* into its configured order.

    Without a configured order (or an articulation at *prim_path*), *names* are returned unchanged.
    """
    ordering = _configured_order(prim_path, kind)
    if ordering is None:
        return list(names)
    rank = {name: index for index, name in enumerate(ordering)}
    missing = [name for name in names if name not in rank]
    if missing:
        raise ValueError(f"The {kind}_ordering of '{prim_path}' does not name {missing}.")
    return sorted(names, key=rank.__getitem__)
