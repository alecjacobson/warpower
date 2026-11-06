import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import cube, producer, torus


def reference(points, samples, weights, sizing, power):
    points, samples = points.astype(np.float64), samples.astype(np.float64)
    d = np.sum((samples[:, None, :] - points[None, :, :]) ** 2, axis=2)
    owners = np.argmin(d, axis=1)
    weights = weights / sizing.astype(np.float64) ** power
    active = weights > 0
    mass = np.bincount(owners, weights=weights, minlength=len(points))
    centroid = np.zeros_like(points)
    for axis in range(3):
        centroid[:, axis] = np.bincount(
            owners, weights=weights * samples[:, axis], minlength=len(points)
        )
    centroid[mass > 0] /= mass[mass > 0, None]
    energy = np.sum(weights * d[np.arange(len(samples)), owners])
    owners[~active] = -1
    return (
        owners,
        mass,
        centroid,
        energy,
        weights.sum(),
        np.sqrt(d[active].min(axis=1)).max(initial=0),
    )


def test_assignments_weighted_centroids_roles_and_energy_match_independent_reference():
    rng = np.random.default_rng(83)
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    points = rng.uniform(-1.2, 1.2, (103, 3)).astype(np.float32)
    surface_sizing = rng.uniform(0.7, 1.3, len(domain.surface_points)).astype(np.float32)
    crease_sizing = rng.uniform(0.7, 1.3, len(domain.crease_points)).astype(np.float32)
    workspace = warpower.BoundaryWorkspace(
        domain, points, surface_sizing=surface_sizing, crease_sizing=crease_sizing
    ).run()
    assert workspace.error.numpy()[0] == 0
    surface = reference(
        points, domain.surface_points.numpy(), domain.surface_weights.numpy(), surface_sizing, 4
    )
    crease = reference(
        points, domain.crease_points.numpy(), domain.crease_weights.numpy(), crease_sizing, 3
    )
    target, roles = points.copy(), np.zeros(len(points), np.int32)
    for kind, expected, gpu_owners, gpu_mass in [
        (1, surface, workspace.surface_owners, workspace.surface_mass),
        (2, crease, workspace.crease_owners, workspace.crease_mass),
    ]:
        owners, mass, centroid, energy, total, maximum = expected
        np.testing.assert_array_equal(gpu_owners.numpy(), owners)
        np.testing.assert_allclose(gpu_mass.numpy(), mass, rtol=3e-13, atol=1e-13)
        np.testing.assert_allclose(workspace.energy.numpy()[kind - 1], energy, rtol=3e-13)
        np.testing.assert_allclose(workspace.total_weight.numpy()[kind - 1], total, rtol=3e-13)
        np.testing.assert_allclose(workspace.max_distance.numpy()[kind - 1], maximum, rtol=3e-13)
        target[mass > 0] = centroid[mass > 0]
        roles[mass > 0] = kind
    count = domain.corner_count.numpy()[0]
    corner_sites = workspace.corner_sites.numpy()[:count]
    assert len(np.unique(corner_sites)) == count
    target[corner_sites] = domain.corner_points.numpy()[:count]
    roles[corner_sites] = 3
    np.testing.assert_allclose(workspace.targets.numpy(), target, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(workspace.roles.numpy(), roles)
    np.testing.assert_array_equal(workspace.role_counts.numpy(), np.bincount(roles, minlength=4))
    np.testing.assert_array_equal(workspace.points.numpy(), points)


def test_collision_resolution_assigns_distinct_exact_corners():
    domain = warpower.SurfaceDomain(*cube())
    points = np.random.default_rng(1).normal(0, 0.001, (8, 3)).astype(np.float32)
    workspace = warpower.BoundaryWorkspace(domain, points).run()
    assert workspace.error.numpy()[0] == 0
    assert workspace.corner_rounds.numpy()[0] > 1
    assert set(workspace.corner_sites.numpy()[:8]) == set(range(8))
    workspace.apply(damping=0.1, max_step=1e-5)
    np.testing.assert_array_equal(
        workspace.points.numpy()[workspace.corner_sites.numpy()[:8]],
        domain.corner_points.numpy()[:8],
    )
    # Exact anchors stay fixed in subsequent fitting passes.
    before = workspace.points.numpy()
    workspace.run().apply()
    np.testing.assert_array_equal(workspace.points.numpy(), before)


def test_boundary_membership_can_change_as_other_sites_move():
    corners, faces = cube()
    face_centers = [
        np.eye(3, dtype=np.float32)[axis] * side for axis in range(3) for side in (-1, 1)
    ]
    edges = []
    for axis in range(3):
        for a in (-1, 1):
            for b in (-1, 1):
                p = np.zeros(3, np.float32)
                p[(axis + 1) % 3], p[(axis + 2) % 3] = a, b
                edges.append(p)
    points = np.concatenate((corners, face_centers, edges, np.zeros((1, 3)))).astype(np.float32)
    domain = warpower.SurfaceDomain(corners, faces, surface_subdivisions=4)
    workspace = warpower.BoundaryWorkspace(domain, points).run()
    assert workspace.error.numpy()[0] == 0
    assert workspace.roles.numpy()[-1] == 0
    points[8:-1] = 30
    wp.copy(workspace.points, wp.array(points, dtype=wp.vec3, device="cuda:0"))
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    assert workspace.roles.numpy()[-1] == 1


def test_smooth_boundary_centroid_iterations_reduce_sampled_error():
    domain = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=2)
    rng = np.random.default_rng(91)
    points = rng.uniform(-2, 2, (90, 3)).astype(np.float32)
    workspace = warpower.BoundaryWorkspace(domain, points).run()
    before = workspace.energy.numpy()[0]
    with wp.ScopedCapture() as capture:
        for _ in range(12):
            workspace.run()
            workspace.apply(damping=0.7)
        workspace.run()
    wp.capture_launch(capture.graph)
    assert workspace.error.numpy()[0] == 0
    assert workspace.energy.numpy()[0] < 0.7 * before
    assert domain.corner_count.numpy()[0] == 0


def test_ties_choose_lowest_active_site_and_inactive_storage_is_ignored():
    domain = warpower.SurfaceDomain(*cube(), feature_angle=179)
    points = np.array([[0, 0, 0], [0, 0, 0], [np.nan, np.nan, np.nan]], np.float32)
    active = wp.array([2], dtype=int, device="cuda:0")
    workspace = warpower.BoundaryWorkspace(domain, points, active_count=active).run()
    assert workspace.error.numpy()[0] == 0
    np.testing.assert_array_equal(workspace.surface_owners.numpy(), 0)
    np.testing.assert_array_equal(workspace.roles.numpy(), [1, 0, 0])
    active.fill_(1)
    workspace.run()
    assert workspace.error.numpy()[0] == 0
    np.testing.assert_array_equal(workspace.role_counts.numpy(), [0, 1, 0, 0])


def test_capture_producer_fitting_and_apply_without_host_work(monkeypatch, tmp_path):
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=3)
    source = wp.array(
        np.random.default_rng(2).normal(0, 0.4, (40, 3)).astype(np.float32),
        dtype=wp.vec3,
        device="cuda:0",
    )
    points = wp.clone(source)
    scale = wp.ones(1, dtype=float, device="cuda:0")
    active = wp.array([40], dtype=int, device="cuda:0")
    workspace = warpower.BoundaryWorkspace(domain, points, active_count=active)
    wp.load_module(module="tests.test_domain", device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation/readback/sync during capture or replay")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(producer, len(points), inputs=[source, points, scale])
            workspace.run()
            workspace.apply(damping=0.5)
        wp.capture_launch(capture.graph)
    assert workspace.error.numpy()[0] == 0
    dot = tmp_path / "boundary.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    graph = dot.read_text()
    assert "MEM_ALLOC" not in graph and "MEM_FREE" not in graph and "HOST" not in graph
    first = workspace.points.numpy()
    scale.fill_(2)
    active.fill_(32)
    wp.capture_launch(capture.graph)
    assert workspace.error.numpy()[0] == 0
    assert workspace.role_counts.numpy().sum() == 32
    assert not np.array_equal(workspace.points.numpy(), first)
    corners = workspace.corner_sites.numpy()[: domain.corner_count.numpy()[0]]
    assert np.all(corners < 32)
    np.testing.assert_array_equal(
        workspace.points.numpy()[corners], domain.corner_points.numpy()[: len(corners)]
    )


@pytest.mark.parametrize(
    "failure, bit", [("sites", 2), ("active", 2), ("corner_capacity", 4), ("sizing", 8)]
)
def test_errors_suppress_updates_and_can_recover(failure, bit):
    domain = warpower.SurfaceDomain(*cube())
    points = np.random.default_rng(4).uniform(-1, 1, (12, 3)).astype(np.float32)
    active = wp.array([12], dtype=int, device="cuda:0")
    sizing = wp.ones(len(domain.surface_points), dtype=float, device="cuda:0")
    workspace = warpower.BoundaryWorkspace(
        domain, points, active_count=active, surface_sizing=sizing
    )
    if failure == "sites":
        workspace.points.fill_(wp.vec3(float("nan")))
    elif failure == "active":
        active.fill_(13)
    elif failure == "corner_capacity":
        active.fill_(7)
    else:
        sizing.fill_(0)
    before = workspace.points.numpy()
    workspace.run().apply()
    assert workspace.error.numpy()[0] & bit
    np.testing.assert_array_equal(workspace.points.numpy(), before)
    np.testing.assert_array_equal(workspace.roles.numpy(), 0)
    assert not np.any(workspace.role_counts.numpy())
    wp.copy(workspace.points, wp.array(points, dtype=wp.vec3, device="cuda:0"))
    active.fill_(12)
    sizing.fill_(1)
    workspace.run().apply()
    assert workspace.error.numpy()[0] == 0


def test_single_site_has_the_surface_centroid_as_its_target():
    domain = warpower.SurfaceDomain(*cube(), feature_angle=179)
    points = np.array([[0.1, 0.2, 0.3]], np.float32)
    workspace = warpower.BoundaryWorkspace(domain, points).run()
    assert workspace.error.numpy()[0] == 0
    np.testing.assert_allclose(workspace.targets.numpy(), np.zeros((1, 3)), atol=1e-7)
    np.testing.assert_array_equal(workspace.surface_owners.numpy(), 0)
    np.testing.assert_array_equal(workspace.role_counts.numpy(), [0, 1, 0, 0])
