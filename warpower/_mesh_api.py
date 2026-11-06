"""Standalone full-hull meshes, without bounded-diagram adjacency work."""

import operator

import warp as wp

from . import _validation
from ._api import _points_array, _weights_array
from ._tetrahedra import build


def _mesh(points, weights, chunk_size, leaf_size):
    p, torch_output = _points_array(points)
    if not p.device.is_cuda:
        raise ValueError("tetrahedralization requires a CUDA device")
    leaf_size = operator.index(leaf_size)
    leaf_size = 4 if leaf_size == -1 else leaf_size
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive or -1")
    if chunk_size is not None:
        chunk_size = operator.index(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
    if torch_output:
        import torch

        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device))
    else:
        stream = wp.get_stream(p.device)
    with wp.ScopedStream(stream):
        w = _weights_array(weights, len(p), p.device)
        empty = wp.empty(0, dtype=int, device=p.device)
        _validation.check(p, w, empty, empty, False)
        tets, _ = build(p, w, chunk_size=chunk_size, leaf_size=leaf_size)
        wp.synchronize_stream(stream)
        return wp.to_torch(tets) if torch_output else tets


def delaunay_tetrahedra(points, *, chunk_size=None, leaf_size=4):
    """Return positively oriented int32 CUDA ``(T, 4)`` full-hull tetrahedra.

    Accepts the same points as ``voronoi_diagram`` and returns a Warp array,
    or a PyTorch tensor for PyTorch input. This convenience call allocates and
    synchronizes. For device-only graph loops, use ``DiagramWorkspace`` with
    ``adjacency=False, return_tetrahedra=True``. Numerical ambiguity raises an
    error rather than returning a partial mesh.
    """
    return _mesh(points, None, chunk_size, leaf_size)


def regular_tetrahedra(points, weights, *, chunk_size=None, leaf_size=4):
    """Weighted counterpart of ``delaunay_tetrahedra``; hidden sites are omitted."""
    if weights is None:
        raise TypeError("regular_tetrahedra requires weights")
    return _mesh(points, weights, chunk_size, leaf_size)
