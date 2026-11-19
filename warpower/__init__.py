"""3D Voronoi and power diagram construction in NVIDIA Warp."""

from ._api import Diagram, power_diagram, voronoi_diagram

__all__ = [
    "AutomaticSizingWorkspace",
    "BoundaryFitter",
    "BoundaryWorkspace",
    "DelaunayWorkspace",
    "Diagram",
    "DiagramWorkspace",
    "DomainQueryWorkspace",
    "ExtractionWorkspace",
    "InitialSamplingWorkspace",
    "ODTWorkspace",
    "SizingField",
    "SizingQueryWorkspace",
    "SurfaceDomain",
    "UpdatableDelaunay",
    "UpdatableDiagram",
    "VariationalMesher",
    "VariationalMeshingWorkspace",
    "delaunay_tetrahedra",
    "power_diagram",
    "regular_tetrahedra",
    "voronoi_diagram",
]

_DEPRECATED_NAMES = {
    "BoundaryWorkspace": "BoundaryFitter",
    "DelaunayWorkspace": "UpdatableDelaunay",
    "DiagramWorkspace": "UpdatableDiagram",
    "VariationalMeshingWorkspace": "VariationalMesher",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        import warnings

        replacement = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"warpower.{name} is deprecated; use warpower.{replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = replacement
    if name == "VariationalMesher":
        from ._mesher import VariationalMesher

        return VariationalMesher
    if name == "AutomaticSizingWorkspace":
        from ._automatic_sizing import AutomaticSizingWorkspace

        return AutomaticSizingWorkspace
    if name == "InitialSamplingWorkspace":
        from ._sampling import InitialSamplingWorkspace

        return InitialSamplingWorkspace
    if name in ("SizingField", "SizingQueryWorkspace"):
        from . import _sizing

        return getattr(_sizing, name)
    if name == "UpdatableDelaunay":
        from ._delaunay_recovery import UpdatableDelaunay

        return UpdatableDelaunay
    if name == "ExtractionWorkspace":
        from ._extraction import ExtractionWorkspace

        return ExtractionWorkspace
    if name == "BoundaryFitter":
        from ._boundary import BoundaryFitter

        return BoundaryFitter
    if name in ("SurfaceDomain", "DomainQueryWorkspace"):
        from . import _domain

        return getattr(_domain, name)
    if name == "ODTWorkspace":
        from ._odt import ODTWorkspace

        return ODTWorkspace
    if name == "UpdatableDiagram":
        from ._graph import UpdatableDiagram

        return UpdatableDiagram
    if name in ("delaunay_tetrahedra", "regular_tetrahedra"):
        from . import _mesh_api

        return getattr(_mesh_api, name)
    raise AttributeError(name)
