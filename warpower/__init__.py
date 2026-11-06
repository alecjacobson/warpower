"""3D Voronoi and power diagram construction in NVIDIA Warp."""

from ._api import Diagram, power_diagram, voronoi_diagram

__all__ = [
    "BoundaryWorkspace",
    "DelaunayWorkspace",
    "Diagram",
    "DiagramWorkspace",
    "DomainQueryWorkspace",
    "ExtractionWorkspace",
    "ODTWorkspace",
    "SurfaceDomain",
    "delaunay_tetrahedra",
    "power_diagram",
    "regular_tetrahedra",
    "voronoi_diagram",
]


def __getattr__(name):
    if name == "DelaunayWorkspace":
        from ._delaunay_recovery import DelaunayWorkspace

        return DelaunayWorkspace
    if name == "ExtractionWorkspace":
        from ._extraction import ExtractionWorkspace

        return ExtractionWorkspace
    if name == "BoundaryWorkspace":
        from ._boundary import BoundaryWorkspace

        return BoundaryWorkspace
    if name in ("SurfaceDomain", "DomainQueryWorkspace"):
        from . import _domain

        return getattr(_domain, name)
    if name == "ODTWorkspace":
        from ._odt import ODTWorkspace

        return ODTWorkspace
    if name == "DiagramWorkspace":
        from ._graph import DiagramWorkspace

        return DiagramWorkspace
    if name in ("delaunay_tetrahedra", "regular_tetrahedra"):
        from . import _mesh_api

        return getattr(_mesh_api, name)
    raise AttributeError(name)
