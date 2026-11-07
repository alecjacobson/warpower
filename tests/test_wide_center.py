from fractions import Fraction
from itertools import permutations, product
from types import SimpleNamespace

import numpy as np
import warp as wp

from examples.vtm_inputs import surface
from tests.test_sphere_bounds import exact_sphere
from warpower import ExtractionWorkspace, SurfaceDomain
from warpower._selective import classify_retained
from warpower._voronoi_volume import BoundedVoronoiVolumes
from warpower._wide_center import exact_geometry, uncertain

wp.set_module_options({"enable_backward": False, "fast_math": False})


def bridge():
    """Original sites from the disconnected1000 trajectory in issue #124."""
    return np.array(
        [
            [1.2999999523162842, 0.6999999284744263, 5.7923338937638014e-18],
            [-1.2999999523162842, 0.699999988079071, 0.699999988079071],
            [1.2999999523162842, 0.699999988079071, 0.699999988079071],
            [-1.2999999523162842, 0.6999999284744263, 5.182126064605883e-18],
        ],
        np.float32,
    )


def bounded_bridge():
    """Enclose all bridge sites while preserving its distant empty sphere."""
    return np.vstack(
        [
            bridge(),
            np.array(list(product([-5, 5], [-5, np.float32(0.7000001)], [-5, 5])), np.float32),
        ]
    )


def bounded_core(points):
    from warpower import DelaunayWorkspace

    mesh = DelaunayWorkspace(points, facet_recovery=True).run()
    assert mesh.error.numpy()[0] == 0
    tets = mesh.tetrahedra.numpy()[: mesh.tetrahedra_count.numpy()[0]]
    assert np.any(np.all(np.sort(tets, axis=1) == np.arange(4), axis=1))
    return SimpleNamespace(
        n=len(points),
        points=mesh.points,
        full_tetrahedra=mesh.tetrahedra,
        full_tetrahedra_count=mesh.tetrahedra_count,
        error=mesh.error,
        mesh=mesh,
        device=mesh.device,
        active_count=None,
        sizing=None,
    )


@wp.kernel
def evaluate(
    points: wp.array2d[wp.vec3],
    centers: wp.array[wp.vec3d],
    offsets: wp.array[wp.vec3d],
    metrics: wp.array2d[wp.float64],
    flags: wp.array2d[int],
):
    i = wp.tid()
    p0, p1, p2, p3 = points[i, 0], points[i, 1], points[i, 2], points[i, 3]
    a, b, c = wp.vec3d(p1) - wp.vec3d(p0), wp.vec3d(p2) - wp.vec3d(p0), wp.vec3d(p3) - wp.vec3d(p0)
    d = wp.dot(a, wp.cross(b, c))
    result = exact_geometry(p0, p1, p2, p3)
    centers[i] = result.center
    offsets[i] = result.offset
    metrics[i, 0] = result.determinant
    metrics[i, 1] = result.radius
    metrics[i, 2] = d
    flags[i, 0] = result.valid
    flags[i, 1] = int(uncertain(a, b, c, d))


def test_exact_metric_fallback_across_order_scale_and_centers_beyond_float32():
    cases = np.array(
        [
            np.ldexp(bridge()[list(order)], exponent)
            for exponent in (-60, 0, 110)
            for order in permutations(range(4))
        ],
        np.float32,
    )
    points = wp.array(cases, dtype=wp.vec3, device="cuda:0")
    centers = wp.empty(len(cases), dtype=wp.vec3d, device=points.device)
    offsets = wp.empty_like(centers)
    metrics = wp.empty((len(cases), 3), dtype=wp.float64, device=points.device)
    flags = wp.empty((len(cases), 2), dtype=int, device=points.device)
    wp.launch(evaluate, len(cases), inputs=[points, centers, offsets, metrics, flags])
    values, status, positions = metrics.numpy(), flags.numpy(), centers.numpy()
    local = offsets.numpy()
    for i, cloud in enumerate(cases):
        det, center, radius2 = exact_sphere(cloud)
        assert status[i, 0] == int(det > 0)
        if det > 0:
            np.testing.assert_allclose(values[i, 0], float(det), rtol=3e-15, atol=0)
            np.testing.assert_allclose(values[i, 1], np.sqrt(float(radius2)), rtol=3e-15, atol=0)
            np.testing.assert_allclose(positions[i], list(map(float, center)), rtol=3e-15, atol=0)
            expected_offset = [float(x - Fraction(float(y))) for x, y in zip(center, cloud[0])]
            np.testing.assert_allclose(local[i], expected_offset, rtol=3e-15, atol=0)
    # The original cancellation fixture and an exactly power-of-two-scaled copy.
    assert values[24, 2] == 0 and status[24].tolist() == [1, 1]
    assert status[48].tolist() == [1, 1]
    assert positions[48, 1] > np.finfo(np.float32).max


def test_extraction_rejects_distant_bridge_without_aborting_healthy_material():
    healthy = np.array([[-2, -0.1, 0], [-1.8, -0.1, 0], [-2, 0.1, 0], [-2, -0.1, 0.2]], np.float32)
    points = np.vstack([bridge(), healthy])
    cells = np.arange(8, dtype=np.int32).reshape(2, 4)
    domain = SurfaceDomain(*surface("disconnected"), surface_subdivisions=1)
    extraction = ExtractionWorkspace(domain, points, cells)
    selected = wp.empty(2, dtype=int, device=domain.device)
    quality = wp.empty(2, dtype=wp.float64, device=domain.device)
    error = wp.zeros(1, dtype=int, device=domain.device)
    wp.load_module(module="warpower._selective", device=domain.device)
    with wp.ScopedCapture() as cap:
        extraction.run()
        error.zero_()
        wp.launch(
            classify_retained,
            2,
            inputs=[
                domain.tree,
                extraction.points,
                extraction.input_tetrahedra,
                extraction.input_count,
                0.4,
                error,
                selected,
                quality,
            ],
        )
    for _ in range(2):
        wp.capture_launch(cap.graph)
        assert extraction.error.numpy()[0] == 0
        assert extraction.tetrahedra_count.numpy()[0] == 1
        assert extraction.source_indices.numpy()[0] == 1
        np.testing.assert_array_equal(extraction.selected.numpy(), [0, 1])
        assert error.numpy()[0] == 0
        np.testing.assert_array_equal(selected.numpy(), extraction.selected.numpy())
        np.testing.assert_allclose(
            quality.numpy()[1], extraction.radius_ratio.numpy()[0], rtol=1e-14
        )
        det, center, radius2 = exact_sphere(bridge())
        np.testing.assert_allclose(
            extraction.circumcenters.numpy()[0], list(map(float, center)), rtol=3e-15, atol=0
        )
        np.testing.assert_allclose(
            extraction.circumradii.numpy()[0], np.sqrt(float(radius2)), rtol=3e-15
        )
        np.testing.assert_allclose(extraction._input_volume.numpy()[0], float(det) / 6, rtol=3e-15)


def test_full_disconnected_trajectory_survives_flat_bridge_extraction():
    from tests.test_mesher import validate
    from warpower import VariationalMeshingWorkspace

    domain = SurfaceDomain(*surface("disconnected"), surface_subdivisions=4, crease_subdivisions=4)
    mesh = VariationalMeshingWorkspace(domain, 1000, initial_count=256, seed=2026, quality_rounds=4)
    with wp.ScopedCapture() as cap:
        domain.refit()
        mesh.run()
    wp.capture_launch(cap.graph)
    validate(mesh)
    np.testing.assert_array_equal(mesh.topology.numpy(), [2, 2, 0, 4])
    assert mesh.point_count.numpy()[0] == 1000
    assert mesh.rounds.numpy()[2] == 4
    assert mesh.steps.numpy()[1] > 0 and mesh.moves.numpy()[0] > 0
    assert mesh.minimum_radius_ratio.numpy()[0] > 0.06


def test_bounded_bridge_volumes_match_halfspaces_and_recover_after_invalid_input():
    from tests.test_voronoi_volume import oracle

    original = bounded_bridge()
    core = bounded_core(original)
    cells = BoundedVoronoiVolumes(core)
    with wp.ScopedCapture() as cap:
        cells.run()
    expected, bounded = oracle(original)
    flat, invalid = original.copy(), original.copy()
    flat[3] = flat[0]
    invalid[0, 0] = np.nan
    for points, error in [(original, 0), (flat, 2), (invalid, 2), (original, 0)]:
        core.points.assign(points)
        wp.capture_launch(cap.graph)
        assert cells.error.numpy()[0] == error
        if error:
            assert cells.count.numpy()[0] == 0
            np.testing.assert_array_equal(cells.valid.numpy(), 0)
            np.testing.assert_array_equal(cells.volume.numpy(), 0)
        else:
            assert cells.count.numpy()[0] == 4
            np.testing.assert_array_equal(cells.valid.numpy(), bounded)
            # These thin cells extend millions of input units; the independent
            # Qhull halfspace oracle loses more precision than ordinary cells.
            np.testing.assert_allclose(cells.volume.numpy(), expected, rtol=1e-7, atol=0)
