"""3D Voronoi and power diagram construction in NVIDIA Warp."""

from ._api import Diagram, power_diagram, voronoi_diagram

__all__ = ["Diagram", "DiagramWorkspace", "power_diagram", "voronoi_diagram"]


def __getattr__(name):
    if name == "DiagramWorkspace":
        from ._graph import DiagramWorkspace

        return DiagramWorkspace
    raise AttributeError(name)
