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
