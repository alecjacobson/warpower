import numpy as np
import pytest
import warp as wp

from tests.test_boundary import reference
from tests.test_domain import cube, torus
from tests.test_mesher import validate
from warpower import BoundaryWorkspace, SurfaceDomain, VariationalMeshingWorkspace


@pytest.mark.parametrize("query", ["float32", "float64", "mixed"])
def test_precision_modes_against_independent_weighted_centroids(query):
    domain = SurfaceDomain(*cube(), surface_subdivisions=8, crease_subdivisions=8)
    rng = np.random.default_rng(12)
    points = rng.uniform(-1.3, 1.3, (203, 3)).astype(np.float32)
    surface_sizing = rng.uniform(0.7, 1.3, len(domain.surface_points)).astype(np.float32)
    crease_sizing = rng.uniform(0.7, 1.3, len(domain.crease_points)).astype(np.float32)
    fit = BoundaryWorkspace(
        domain,
        points,
        surface_sizing=surface_sizing,
        crease_sizing=crease_sizing,
        query_precision=query,
    ).run()
    assert fit.error.numpy()[0] == 0
    assert fit.surface_mass.dtype == fit.crease_mass.dtype == fit.energy.dtype == wp.float64
    target = points.copy()
    for k, (samples, weights, sizing, owners, mass, power) in enumerate(
        [
            (
                domain.surface_points,
                domain.surface_weights,
                surface_sizing,
                fit.surface_owners,
                fit.surface_mass,
                4,
            ),
            (
                domain.crease_points,
                domain.crease_weights,
                crease_sizing,
                fit.crease_owners,
                fit.crease_mass,
                3,
            ),
        ]
    ):
        expected = reference(points, samples.numpy(), weights.numpy(), sizing, power)
        nearest, masses, centroid, energy, total, maximum = expected
        np.testing.assert_array_equal(owners.numpy(), nearest)
        tolerance = 3e-13
        np.testing.assert_allclose(mass.numpy(), masses, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(fit.energy.numpy()[k], energy, rtol=3e-13)
        np.testing.assert_allclose(fit.total_weight.numpy()[k], total, rtol=3e-13)
        np.testing.assert_allclose(fit.max_distance.numpy()[k], maximum, rtol=3e-13)
        target[masses > 0] = centroid[masses > 0]
    count = domain.corner_count.numpy()[0]
    corners = fit.corner_sites.numpy()[:count]
    target[corners] = domain.corner_points.numpy()[:count]
    np.testing.assert_allclose(fit.targets.numpy(), target, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(fit.targets.numpy()[corners], target[corners])


def test_fp32_near_tie_can_change_owner_but_energy_uses_fp64_distance():
    domain = SurfaceDomain(*cube(), feature_angle=179)
    domain.surface_weights.zero_()
    domain.surface_weights[:1].fill_(1.0)
    domain.surface_points[:1].fill_(wp.vec3(0.0))
    points = np.array([[1, 1e-4, 0], [1, 0, 0]], np.float32)
    exact = BoundaryWorkspace(domain, points).run()
    fast = BoundaryWorkspace(domain, points, query_precision="float32").run()
    assert exact.error.numpy()[0] == fast.error.numpy()[0] == 0
    assert exact.surface_owners.numpy()[0] == 1
    assert fast.surface_owners.numpy()[0] == 0
    np.testing.assert_allclose(
        fast.energy.numpy()[0], np.dot(points[0].astype(float), points[0].astype(float)), rtol=1e-15
    )
    assert fast.energy.numpy()[0] > exact.energy.numpy()[0]


def test_fp32_query_out_of_range_is_explicit_and_recovers():
    domain = SurfaceDomain(*cube(), feature_angle=179)
    points = np.random.default_rng(3).uniform(-1, 1, (32, 3)).astype(np.float32)
    fit = BoundaryWorkspace(domain, points, query_precision="float32")
    with wp.ScopedCapture() as capture:
        fit.run()
        fit.apply()
    fit.points.fill_(wp.vec3(1e30))
    wp.capture_launch(capture.graph)
    assert fit.error.numpy()[0] & 16
    np.testing.assert_array_equal(fit.points.numpy(), np.full_like(points, 1e30))
    np.testing.assert_array_equal(fit.roles.numpy(), 0)
    fit.points.assign(points)
    wp.capture_launch(capture.graph)
    assert fit.error.numpy()[0] == 0


@pytest.mark.parametrize("query", ["float32", "mixed"])
def test_complete_pipeline_precision_and_fresh_capture_replay(query):
    domain = SurfaceDomain(*torus(24, 12))
    mesh = VariationalMeshingWorkspace(
        domain,
        256,
        initial_count=64,
        quality_rounds=1,
        boundary_query_precision=query,
    )
    with wp.ScopedCapture() as capture:
        domain.refit()
        mesh.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        validate(mesh)


def test_invalid_precision_is_rejected():
    domain = SurfaceDomain(*cube())
    with pytest.raises(ValueError, match="query_precision"):
        BoundaryWorkspace(domain, np.zeros((16, 3), np.float32), query_precision="half")
