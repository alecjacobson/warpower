"""Shared temporary clipping storage stays isolated between concurrent families."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from warpower._active_delaunay import ActiveDelaunayWorkspace


@wp.kernel
def produce(source: wp.array[wp.vec3], active: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = source[0]
    if i < active[0]:
        points[i] = source[i]


def canonical(cells):
    cells = np.sort(cells, axis=1)
    return cells[np.lexsort(cells.T[::-1])]


@pytest.mark.parametrize("capacity", [512, 1024])
def test_concurrent_prefix_families_preserve_inputs_failure_and_recovery(capacity):
    device = wp.get_device("cuda:0")
    wp.load_module(module=__name__, device=device)
    sources = [
        np.random.default_rng(s).normal(size=(2049, 3)).astype(np.float32) for s in [7, 2026]
    ]
    owners = []
    for source in sources:
        points = wp.zeros(2049, dtype=wp.vec3, device=device)
        active = wp.array([64], dtype=int, device=device)
        input_points = wp.array(source, dtype=wp.vec3, device=device)
        mesh = ActiveDelaunayWorkspace(
            points,
            active_count=active,
            prefix_capacities=(64, 128),
            tetrahedra_cell_capacity=capacity,
        )
        stream = wp.Stream(device)
        with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream) as capture:
            wp.launch(produce, 2049, inputs=[input_points, active, points], device=device)
            mesh.run()
        owners.append((mesh, active, input_points, stream, capture.graph))
    # Launch both families before synchronizing: a process-global scratch pool
    # would let their independent streams corrupt each other's cell geometry.
    for counts in [(64, 96), (129, 63), (3, 129), (96, 64)]:
        for (_, active, _, _, _), n in zip(owners, counts):
            active.assign(np.array([n], dtype=np.int32))
        wp.synchronize_device(device)
        for _, _, _, stream, graph in owners:
            wp.capture_launch(graph, stream=stream)
        wp.synchronize_device(device)
        for (mesh, _, _, _, _), source, n in zip(owners, sources, counts):
            count = int(mesh.tetrahedra_count.numpy()[0])
            if n < 4:
                assert mesh.error.numpy()[0] == 8 and count == 0
            else:
                assert mesh.error.numpy()[0] == 0
                np.testing.assert_array_equal(
                    canonical(mesh.tetrahedra.numpy()[:count]),
                    canonical(Delaunay(source[:n]).simplices),
                )
