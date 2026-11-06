import numpy as np
import pytest
import warp as wp
from scipy.spatial import ConvexHull, Delaunay

import warpower
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower._odt import accumulate_tets


def oriented(points, tets):
    tets = np.asarray(tets, dtype=np.int32).copy()
    x = points.astype(np.float64)[tets]
    determinants = np.linalg.det(x[:, 1:] - x[:, :1])
    tets[determinants < 0, :2] = tets[determinants < 0, :2][:, ::-1]
    return tets


def independent_terms(points, tets):
    """Linear solves and exact degree-two quadrature, not GPU cofactor algebra."""
    p = points.astype(np.float64)
    v = p[tets]
    edges = v[:, 1:] - v[:, :1]
    volumes = np.linalg.det(edges) / 6
    centers = v[:, 0] + np.linalg.solve(2 * edges, np.sum(edges**2, axis=2)[..., None])[..., 0]
    mass, moment = np.zeros(len(p)), np.zeros_like(p)
    for k in range(4):
        np.add.at(mass, tets[:, k], volumes)
        np.add.at(moment, tets[:, k], volumes[:, None] * (centers - v[:, k]))
    # Four symmetric quadrature nodes integrate a quadratic exactly on a tet.
    alpha, beta = (5 + 3 * np.sqrt(5)) / 20, (5 - np.sqrt(5)) / 20
    barycentric = np.full((4, 4), beta)
    np.fill_diagonal(barycentric, alpha)
    q = np.einsum("ki,tij->tkj", barycentric, v)
    interpolant = np.einsum("ki,ti->tk", barycentric, np.sum(v * v, axis=2))
    energies = volumes * np.mean(interpolant - np.sum(q * q, axis=2), axis=1)
    areas = np.zeros(len(tets))
    for k in range(4):
        f = np.delete(v, k, axis=1)
        areas += np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1) / 2
    quality = 9 * volumes / (areas * np.linalg.norm(centers - v[:, 0], axis=1))
    return energies.sum(), mass, moment, volumes.sum(), quality.min()


def evaluate_kernel(points, tets):
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    t = wp.array(tets, dtype=int, device="cuda:0")
    count = wp.array([len(tets)], dtype=int, device="cuda:0")
    mesh_error = wp.zeros(1, dtype=int, device="cuda:0")
    invalid = wp.zeros_like(mesh_error)
    mass = wp.zeros(len(points), dtype=wp.float64, device="cuda:0")
    moment = wp.zeros(len(points), dtype=wp.vec3d, device="cuda:0")
    edge = wp.full(len(points), 1.0e300, dtype=wp.float64, device="cuda:0")
    energy = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    volume = wp.zeros_like(energy)
    quality = wp.ones_like(energy)
    wp.launch(
        accumulate_tets,
        len(tets),
        inputs=[
            p,
            t,
            count,
            mesh_error,
            invalid,
            mass,
            moment,
            edge,
            energy,
            volume,
            quality,
            True,
        ],
    )
    return (
        energy.numpy()[0],
        mass.numpy(),
        moment.numpy(),
        volume.numpy()[0],
        quality.numpy()[0],
        invalid.numpy()[0],
    )


def test_tetrahedron_reductions_match_independent_quadrature_and_linear_solves():
    points = np.random.default_rng(12).normal(size=(70, 3)).astype(np.float32)
    tets = oriented(points, Delaunay(points).simplices)
    actual = evaluate_kernel(points, tets)
    expected = independent_terms(points, tets)
    assert actual[-1] == 0
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-13)
    # For an interior complete star, gradient E = -moment / 2. Check with
    # finite differences of independently quadrature-integrated energy.
    interior = next(i for i in range(len(points)) if i not in ConvexHull(points).vertices)
    for axis in range(3):
        plus, minus = points.astype(np.float64), points.astype(np.float64)
        plus[interior, axis] += 1e-4
        minus[interior, axis] -= 1e-4
        derivative = (independent_terms(plus, tets)[0] - independent_terms(minus, tets)[0]) / 2e-4
        assert np.isclose(derivative, -actual[2][interior, axis] / 2, rtol=1e-6, atol=1e-8)


def test_sliver_moment_remains_finite_and_inversion_is_rejected():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1e-12]], np.float32)
    tets = np.array([[0, 1, 2, 3]], np.int32)
    actual = evaluate_kernel(points, tets)
    expected = independent_terms(points, tets)
    assert actual[-1] == 0 and np.all(np.isfinite(actual[2]))
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-12, atol=1e-25)
    assert actual[0] > 0 and actual[3] > 0 and actual[4] > 0
    assert evaluate_kernel(points, tets[:, [1, 0, 2, 3]])[-1] == 256


def assert_valid_optimized(workspace, original):
    assert workspace.error.numpy()[0] == 0
    points = workspace.points.numpy()
    count = workspace.tetrahedra_count.numpy()[0]
    tets = workspace.tetrahedra.numpy()[:count]
    check_mesh(points, tets)
    assert tet_set(tets) == reference(points)
    hull = ConvexHull(original).vertices
    assert np.array_equal(points[hull], original[hull])
    assert np.isclose(workspace.volume.numpy()[0], ConvexHull(original).volume, rtol=1e-10)
    iterations = workspace.iterations.numpy()[0]
    history = workspace.energy_history.numpy()[: iterations + 1]
    assert np.all(np.diff(history) <= 2e-12 * history[0])
    expected = independent_terms(points, tets)
    assert np.isclose(workspace.energy.numpy()[0], expected[0], rtol=1e-12)
    assert np.isclose(workspace.minimum_radius_ratio.numpy()[0], expected[4], rtol=1e-10)
    return points, tets, history


def test_fixed_hull_optimizer_reconstructs_and_reduces_energy():
    original = np.random.default_rng(12).normal(size=(90, 3)).astype(np.float32)
    workspace = warpower.ODTWorkspace(original, max_iterations=8)
    workspace.run()
    points, _, history = assert_valid_optimized(workspace, original)
    assert len(history) > 1 and history[-1] < history[0] * 0.98
    assert np.linalg.norm(points - original) > 0.1


def test_uniform_cloud_allows_topology_changes_without_global_boundary_stall():
    original = np.random.default_rng(2026).uniform(-1, 1, (1000, 3)).astype(np.float32)
    initial_tets = reference(original)
    workspace = warpower.ODTWorkspace(original, max_iterations=4)
    workspace.run()
    _, tets, history = assert_valid_optimized(workspace, original)
    assert workspace.iterations.numpy()[0] == 4
    assert workspace.termination.numpy()[0] == 0
    assert history[-1] < 0.8 * history[0]
    assert tet_set(tets) != initial_tets


def test_large_steps_backtrack_and_exhaustion_restores_valid_mesh():
    # One interior site of a near-regular outer tetrahedron has its optimum at
    # the outer circumcenter. Overshooting increases the independent ODT energy.
    original = np.array(
        [
            [1, 1.1, 0.9],
            [-1.05, -0.93, 1.02],
            [-0.97, 1.03, -1.12],
            [1.07, -1.04, -0.96],
            [0.1, 0.03, -0.02],
        ],
        dtype=np.float32,
    )
    workspace = warpower.ODTWorkspace(original, max_iterations=2, damping=16)
    workspace.run()
    assert_valid_optimized(workspace, original)
    assert workspace.rejected_steps.numpy()[0] > 0
    assert workspace.iterations.numpy()[0] > 0
    stopped = warpower.ODTWorkspace(original, max_iterations=2, damping=1e6, max_backtracks=0)
    stopped.run()
    points, _, history = assert_valid_optimized(stopped, original)
    assert np.array_equal(points, original)
    assert len(history) == 1
    assert stopped.termination.numpy()[0] == 2
    assert stopped.rejected_steps.numpy()[0] == 1


@pytest.mark.parametrize("max_iterations", [0, 5])
def test_additional_fixed_sites_and_no_iterations(max_iterations):
    original = np.random.default_rng(81).normal(size=(30, 3)).astype(np.float32)
    fixed = wp.ones(len(original), dtype=int, device="cuda:0")
    workspace = warpower.ODTWorkspace(original, fixed=fixed, max_iterations=max_iterations)
    workspace.run()
    points, _, _ = assert_valid_optimized(workspace, original)
    assert np.array_equal(points, original)
    assert workspace.iterations.numpy()[0] == min(1, max_iterations)
    assert workspace.termination.numpy()[0] == int(max_iterations > 0)


@wp.kernel
def reset_sites(base: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[i] = base[i] + wp.vec3(0.01 * float(phase[0]) * wp.sin(float(i)), 0.0, 0.0)


@wp.kernel
def consume_mesh(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    total: wp.array[wp.float64],
):
    i = wp.tid()
    if error[0] == 0 and i < count[0]:
        p = wp.vec3d(points[tets[i, 0]])
        a = wp.vec3d(points[tets[i, 1]]) - p
        b = wp.vec3d(points[tets[i, 2]]) - p
        c = wp.vec3d(points[tets[i, 3]]) - p
        wp.atomic_add(total, 0, wp.dot(a, wp.cross(b, c)) / wp.float64(6.0))


@wp.kernel
def advance_loop(phase: wp.array[int], remaining: wp.array[int]):
    phase[0] += 1
    remaining[0] -= 1


def test_captured_outer_loop_produces_optimizes_and_consumes_without_host_work(monkeypatch):
    original = np.random.default_rng(42).normal(size=(55, 3)).astype(np.float32)
    base = wp.array(original, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(base)
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    remaining = wp.ones_like(phase)
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    workspace = warpower.ODTWorkspace(points, max_iterations=3)
    wp.load_module(module=__name__, device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Host work during optimizer capture/replay")

    def body():
        wp.launch(reset_sites, len(points), inputs=[base, points, phase])
        workspace.run()
        volume.zero_()
        wp.launch(
            consume_mesh,
            len(workspace.tetrahedra),
            inputs=[
                points,
                workspace.tetrahedra,
                workspace.tetrahedra_count,
                workspace.error,
                volume,
            ],
        )
        wp.launch(advance_loop, 1, inputs=[phase, remaining])

    with monkeypatch.context() as guard:
        guard.setattr(wp.array, "numpy", forbidden)
        for name in (
            "synchronize",
            "synchronize_stream",
            "synchronize_device",
            "empty",
            "zeros",
            "full",
            "empty_like",
        ):
            guard.setattr(wp, name, forbidden)
        with wp.ScopedCapture() as capture:
            remaining.fill_(2)
            wp.capture_while(remaining, body)
        for _ in range(2):
            wp.capture_launch(capture.graph)
    wp.synchronize()
    assert phase.numpy()[0] == 4
    assert workspace.error.numpy()[0] == 0
    assert workspace.iterations.numpy()[0] > 0
    assert np.isclose(volume.numpy()[0], workspace.volume.numpy()[0], rtol=1e-12)
    p = points.numpy()
    tets = workspace.tetrahedra.numpy()[: workspace.tetrahedra_count.numpy()[0]]
    assert tet_set(tets) == reference(p)


def test_invalid_input_reports_error_and_next_replay_recovers():
    original = np.random.default_rng(51).normal(size=(20, 3)).astype(np.float32)
    points = wp.array(original, dtype=wp.vec3, device="cuda:0")
    workspace = warpower.ODTWorkspace(points, max_iterations=2)
    points.fill_(wp.vec3(float("nan")))
    workspace.run()
    assert workspace.error.numpy()[0] != 0
    assert workspace.tetrahedra_count.numpy()[0] == 0
    points.assign(original)
    workspace.run()
    assert_valid_optimized(workspace, original)


@wp.kernel
def fail_reconstruction(
    calls: wp.array[int],
    error: wp.array[int],
    count: wp.array[int],
):
    if calls[0] > 0:
        error[0] = 16
        count[0] = 0
    calls[0] += 1


def test_failed_reconstruction_restores_points_and_connectivity(monkeypatch):
    original = np.random.default_rng(62).normal(size=(40, 3)).astype(np.float32)
    workspace = warpower.ODTWorkspace(original, max_iterations=4, max_backtracks=2)
    calls = wp.zeros(1, dtype=int, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")
    real_run = workspace.mesh.run

    def failing_run(stream=None):
        result = real_run(stream)
        wp.launch(
            fail_reconstruction,
            1,
            inputs=[
                calls,
                workspace.mesh.error,
                workspace.tetrahedra_count,
            ],
        )
        return result

    monkeypatch.setattr(workspace.mesh, "run", failing_run)
    with wp.ScopedCapture() as capture:
        calls.zero_()
        workspace.run()
    wp.capture_launch(capture.graph)
    assert calls.numpy()[0] > 1
    points, _, history = assert_valid_optimized(workspace, original)
    assert np.array_equal(points, original)
    assert len(history) == 1 and workspace.termination.numpy()[0] == 2
    assert workspace.last_geometry_error.numpy()[0] == 16
    assert workspace.mesh.error.numpy()[0] == 0
