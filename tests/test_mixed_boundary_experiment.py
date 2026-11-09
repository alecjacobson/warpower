import numpy as np
import pytest
import warp as wp

from benchmarks._mixed_boundary import MixedBoundaryWorkspace, nearest_site_mixed
from tests.test_domain import cube
from warpower import _bvh
from warpower._site_bvh import SiteHierarchy, nearest_site


@wp.kernel
def query_pair(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    active: wp.array[int],
    occupied: wp.array[int],
    owners: wp.array2d[int],
    distances: wp.array2d[wp.float64],
):
    i = wp.tid()
    exact = nearest_site(tree, points, wp.vec3d(queries[i]), active, occupied, False)
    mixed = nearest_site_mixed(tree, points, queries[i], active, occupied, False)
    owners[i, 0] = exact.index
    owners[i, 1] = mixed.index
    distances[i, 0] = exact.distance2
    distances[i, 1] = mixed.distance2


@pytest.mark.parametrize("exponent", [-120, -60, 0, 60, 120])
def test_mixed_queries_match_reference_across_ranges_and_active_prefixes(exponent):
    rng = np.random.default_rng(276)
    points = (rng.normal(size=(257, 3)) * 2.0**exponent).astype(np.float32)
    points[1] = points[0]
    queries = np.vstack([points[:3], (rng.normal(size=(97, 3)) * 2.0**exponent).astype(np.float32)])
    resident = wp.array(points, dtype=wp.vec3, device="cuda:0")
    tree = SiteHierarchy(resident)
    active = wp.full(1, len(points), dtype=int, device="cuda:0")
    occupied = wp.full(len(points), -1, dtype=int, device="cuda:0")
    q = wp.array(queries, dtype=wp.vec3, device="cuda:0")
    owners = wp.empty((len(q), 2), dtype=int, device="cuda:0")
    distances = wp.empty((len(q), 2), dtype=wp.float64, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    with wp.ScopedCapture() as capture:
        wp.launch(
            query_pair, len(q), inputs=[tree.tree, resident, q, active, occupied, owners, distances]
        )
    for count in [257, 32, 2, 257]:
        active.fill_(count)
        wp.capture_launch(capture.graph)
        actual, distance = owners.numpy(), distances.numpy()
        delta = queries.astype(float)[:, None, :] - points.astype(float)[None, :count, :]
        expected = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        np.testing.assert_array_equal(actual[:, 0], expected)
        np.testing.assert_array_equal(actual[:, 1], expected)
        np.testing.assert_array_equal(distance[:, 0], distance[:, 1])


def test_mixed_fitting_resolves_the_known_fp32_near_tie():
    from warpower import SurfaceDomain

    domain = SurfaceDomain(*cube(), feature_angle=179)
    domain.surface_weights.zero_()
    domain.surface_weights[:1].fill_(1.0)
    domain.surface_points[:1].fill_(wp.vec3(0.0))
    fit = MixedBoundaryWorkspace(domain, np.array([[1, 1e-4, 0], [1, 0, 0]], np.float32))
    with wp.ScopedCapture() as capture:
        fit.run()
    wp.capture_launch(capture.graph)
    assert fit.error.numpy()[0] == 0
    assert fit.surface_owners.numpy()[0] == 1
    assert fit.energy.numpy()[0] == 1.0
