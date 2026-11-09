from itertools import product

import numpy as np
import warp as wp

from benchmarks._cell_continuation import install
from tests.test_delaunay_recovery import exact_empty, valid
from tests.test_tetrahedra import reference, tet_set
from warpower import DelaunayWorkspace


def high_degree(groups=1):
    rng = np.random.default_rng(271)
    shell = rng.normal(size=(80, 3))
    shell /= np.linalg.norm(shell, axis=1)[:, None]
    sites = np.vstack([np.zeros((1, 3)), shell])
    return np.vstack([sites + [4 * k, 0, 0] for k in range(groups)]).astype(np.float32)


def test_current_cell_continuation_and_changed_captured_inputs():
    points = high_degree()
    mesh = DelaunayWorkspace(points, cache_strategy=False)
    cache = install(mesh)
    with wp.ScopedCapture() as capture:
        mesh.run()
    rng = np.random.default_rng(271)
    for current in [points, rng.uniform(-1, 1, points.shape).astype(np.float32), points]:
        mesh.points.assign(current)
        wp.capture_launch(capture.graph)
        assert tet_set(valid(mesh, current)) == reference(current)
        if np.array_equal(current, points):
            assert cache.data.statistics.numpy()[0] > 0
            assert cache.data.statistics.numpy()[1] > 0
    broken = points.copy()
    broken[0, 0] = np.nan
    mesh.points.assign(broken)
    wp.capture_launch(capture.graph)
    assert mesh.error.numpy()[0] != 0 and mesh.tetrahedra_count.numpy()[0] == 0
    mesh.points.assign(points)
    wp.capture_launch(capture.graph)
    assert tet_set(valid(mesh, points)) == reference(points)


def test_continuation_queue_exhaustion_retains_fresh_retry():
    points = high_degree(4)
    mesh = DelaunayWorkspace(points, cache_strategy=False)
    cache = install(mesh, slots=1)
    mesh.run()
    assert tet_set(valid(mesh, points)) == reference(points)
    assert cache.data.statistics.numpy()[2] > 0


def test_symbolic_recovery_remains_available():
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    mesh = DelaunayWorkspace(points, cache_strategy=False)
    install(mesh)
    tets = valid(mesh.run(), points)
    exact_empty(points, tets)
