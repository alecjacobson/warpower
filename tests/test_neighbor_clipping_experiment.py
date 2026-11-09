"""Correctness gates for the isolated benchmark prototype, not a public API."""

from itertools import product

import numpy as np
import pytest
import warp as wp

from benchmarks._neighbor_clipping import install
from tests.test_delaunay_recovery import exact_empty
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower import DelaunayWorkspace


@wp.kernel
def produce(frames: wp.array2d[wp.vec3], phase: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = frames[phase[0], i]


@wp.kernel
def consume(count: wp.array[int], error: wp.array[int], result: wp.array[int]):
    result[0] = count[0]
    result[1] = error[0]


@pytest.mark.parametrize("hint_capacity", [4, 128])
def test_changing_inputs_truncation_failure_and_recovery(hint_capacity):
    rng = np.random.default_rng(8)
    initial = rng.normal(size=(64, 3)).astype(np.float32)
    moved = rng.normal(size=(64, 3)).astype(np.float32)
    duplicate = moved.copy()
    duplicate[32:] = duplicate[:32]
    invalid = moved.copy()
    invalid[4, 0] = np.nan
    frames = np.stack([initial, initial + np.float32(1e-4), moved, duplicate, invalid, initial])
    mesh = DelaunayWorkspace(initial)
    cache = install(mesh, hint_capacity=hint_capacity)
    source = wp.array(frames, dtype=wp.vec3, device=mesh.device)
    phase = wp.zeros(1, dtype=int, device=mesh.device)
    result = wp.zeros(2, dtype=int, device=mesh.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, len(initial), inputs=[source, phase, mesh.points])
        mesh.run()
        wp.launch(consume, 1, inputs=[mesh.tetrahedra_count, mesh.error, result])
    for i, points in enumerate(frames):
        phase.fill_(i)
        wp.capture_launch(capture.graph)
        count, error = result.numpy()
        if i == 4:
            assert error != 0 and count == 0
            np.testing.assert_array_equal(cache.table.numpy(), -1)
            continue
        assert error == 0
        tets = mesh.tetrahedra.numpy()[:count]
        check_mesh(points, tets)
        # Qhull may choose either duplicate's ID; Warpower promises the first.
        if i == 3:
            assert tets.max() < 32
            assert tet_set(tets) == reference(points[:32])
        else:
            assert tet_set(tets) == reference(points)
        if hint_capacity == 4:
            assert cache.overflows.numpy()[0] > 0
    cache.clear()
    mesh.run()
    assert mesh.error.numpy()[0] == 0


def test_cospherical_symbolic_recovery_still_certifies_current_input():
    points = np.array(list(product(range(3), repeat=3)), dtype=np.float32)
    mesh = DelaunayWorkspace(points)
    install(mesh)
    for _ in range(2):
        mesh.run()
        assert mesh.error.numpy()[0] == 0
        count = mesh.tetrahedra_count.numpy()[0]
        tets = mesh.tetrahedra.numpy()[:count]
        check_mesh(points, tets)
        exact_empty(points, tets)
