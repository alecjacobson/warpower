import numpy as np
import warp as wp

import warpower
from tests.test_domain import cube, torus
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower._vtm import VTMWorkspace


def torus_sites(n, seed=2026):
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, 2 * np.pi, (2, n))
    radius = 0.65 * np.sqrt(rng.uniform(0, 1, n))
    return np.stack(
        (
            (2 + radius * np.cos(v)) * np.cos(u),
            (2 + radius * np.cos(v)) * np.sin(u),
            radius * np.sin(v),
        ),
        axis=-1,
    ).astype(np.float32)


def valid_core(workspace):
    assert workspace.error.numpy()[0] == 0
    assert workspace.mesh.error.numpy()[0] == 0
    assert workspace.boundary.error.numpy()[0] == 0
    p = workspace.points.numpy()
    t = workspace.full_tetrahedra.numpy()[: workspace.full_tetrahedra_count.numpy()[0]]
    check_mesh(p, t)
    assert tet_set(t) == reference(p)
    return p, t


def test_nonconvex_boundary_and_interior_updates_reconstruct_complete_delaunay():
    original = torus_sites(150)
    domain = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=2)
    workspace = VTMWorkspace(domain, original, max_iterations=8).run()
    points, _ = valid_core(workspace)
    assert workspace.iterations.numpy()[0] == 8
    assert workspace.rejected_steps.numpy()[0] == 0
    history = workspace.energy_history.numpy()[:9]
    assert history[-1, 1] < 0.6 * history[0, 1]
    assert np.all(np.diff(history[:, 1]) <= 1e-10 * history[0, 1])
    assert np.any(np.linalg.norm(points - original, axis=1) > 0.05)
    assert workspace.boundary.role_counts.numpy()[0] > 0
    assert workspace.boundary.role_counts.numpy()[1] > 0


def test_sharp_cube_preserves_corners_and_improves_crease_approximation():
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=12)
    original = np.random.default_rng(2026).uniform(-0.9, 0.9, (200, 3)).astype(np.float32)
    points = wp.array(original, dtype=wp.vec3, device="cuda:0")
    initialization = warpower.BoundaryWorkspace(domain, points)
    workspace = VTMWorkspace(domain, points, max_iterations=6)
    with wp.ScopedCapture() as capture:
        initialization.run()
        initialization.apply()
        workspace.run()
    wp.capture_launch(capture.graph)
    points, _ = valid_core(workspace)
    assert workspace.iterations.numpy()[0] == 6
    corners = workspace.boundary.corner_sites.numpy()[:8]
    np.testing.assert_array_equal(points[corners], domain.corner_points.numpy()[:8])
    roles = workspace.boundary.roles.numpy()
    assert np.any(roles == 2)
    # Evolving crease membership and centroids spanning a junction need not
    # place every current crease site exactly on an edge. Check the sampled
    # approximation independently rather than asserting facet conformity.
    samples = domain.crease_points.numpy().astype(np.float64)
    weights = domain.crease_weights.numpy()
    distance2 = np.sum((samples[:, None, :] - points[None, :, :]) ** 2, axis=2).min(axis=1)
    expected_error = np.dot(weights, distance2)
    np.testing.assert_allclose(workspace.boundary.energy.numpy()[1], expected_error, rtol=1e-12)
    history = workspace.energy_history.numpy()[:7]
    assert history[-1, 2] < 0.8 * history[0, 2]
    assert np.all(np.diff(history[:, 2]) <= 1e-10 * history[0, 2])


@wp.kernel
def fail_after_initial(error: wp.array[int], count: wp.array[int], calls: wp.array[int]):
    calls[0] += 1
    if calls[0] > 1:
        error[0] = 2
        count[0] = 0


def test_rejected_geometry_restores_points_connectivity_and_boundary_state():
    original = torus_sites(90, 91)
    domain = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=2)
    workspace = VTMWorkspace(domain, original, max_iterations=2, max_backtracks=2)
    calls = wp.zeros(1, dtype=int, device="cuda:0")
    real_run = workspace.mesh.run

    def failing_run(stream=None):
        real_run(stream)
        wp.launch(
            fail_after_initial,
            1,
            inputs=[workspace.mesh.error, workspace.full_tetrahedra_count, calls],
        )

    workspace.mesh.run = failing_run
    wp.load_module(module=__name__, device="cuda:0")
    with wp.ScopedCapture() as capture:
        calls.zero_()
        workspace.run()
    wp.capture_launch(capture.graph)
    points, _ = valid_core(workspace)
    np.testing.assert_array_equal(points, original)
    assert workspace.iterations.numpy()[0] == 0
    assert workspace.rejected_steps.numpy()[0] == 3
    assert workspace.termination.numpy()[0] == 2
    assert workspace.last_geometry_error.numpy()[0] == 2
    assert np.all(np.isfinite(workspace.energy_history.numpy()[0]))
    np.testing.assert_allclose(
        workspace.boundary.energy.numpy()[0], workspace.energy_history.numpy()[0, 1], rtol=1e-12
    )


@wp.kernel
def produce(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    shift = 0.001 * float(phase[0] % 3)
    points[i] = source[i] + wp.vec3(shift, -shift, 0.0)


@wp.kernel
def consume(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    volume: wp.array[wp.float64],
):
    i = wp.tid()
    if i < count[0] and error[0] == 0:
        a = wp.vec3d(points[tets[i, 0]])
        b = wp.vec3d(points[tets[i, 1]]) - a
        c = wp.vec3d(points[tets[i, 2]]) - a
        d = wp.vec3d(points[tets[i, 3]]) - a
        wp.atomic_add(volume, 0, wp.dot(b, wp.cross(c, d)) / wp.float64(6.0))


def test_captured_producer_complete_core_consumer_has_no_host_or_alloc_nodes(monkeypatch, tmp_path):
    source = wp.array(torus_sites(100), dtype=wp.vec3, device="cuda:0")
    points = wp.clone(source)
    domain = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=2)
    workspace = VTMWorkspace(domain, points, max_iterations=3)
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation/readback/sync during capture/replay")

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
            wp.launch(produce, len(source), inputs=[source, points, phase])
            workspace.run()
            volume.zero_()
            wp.launch(
                consume,
                len(workspace.full_tetrahedra),
                inputs=[
                    points,
                    workspace.full_tetrahedra,
                    workspace.full_tetrahedra_count,
                    workspace.error,
                    volume,
                ],
            )
        wp.capture_launch(capture.graph)
    valid_core(workspace)
    np.testing.assert_allclose(volume.numpy(), workspace.volume.numpy(), rtol=1e-12)
    dot = tmp_path / "vtm-core.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    text = dot.read_text()
    assert "MEM_ALLOC" not in text and "MEM_FREE" not in text and "HOST" not in text
    phase.fill_(1)
    wp.capture_launch(capture.graph)
    valid_core(workspace)
    np.testing.assert_allclose(volume.numpy(), workspace.volume.numpy(), rtol=1e-12)


def test_initial_geometry_failure_publishes_no_tetrahedra():
    # Corner-only initial cube facets are cospherical: explicitly record the
    # unresolved #32 case. No mesh is presented as a successful result.
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=16)
    points = np.random.default_rng(2026).uniform(-0.9, 0.9, (200, 3)).astype(np.float32)
    workspace = VTMWorkspace(domain, points, max_iterations=2).run()
    assert workspace.error.numpy()[0] & 1
    assert workspace.mesh.error.numpy()[0] & 16
    assert workspace.full_tetrahedra_count.numpy()[0] == 0
    assert workspace.iterations.numpy()[0] == 0
