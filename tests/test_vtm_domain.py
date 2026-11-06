from pathlib import Path

import numpy as np
import pytest
import warp as wp

from tests.test_domain import combine, cube, solid_angle_inside
from tests.test_stable_volume import exact_det
from warpower import SizingField, SurfaceDomain
from warpower._vtm import VTMWorkspace
from warpower._vtm_domain import constrain

NAN = wp.constant(float("nan"))


def arrays(points, directions, roles=None):
    n = len(points)
    return (
        wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device="cuda:0"),
        wp.array(np.zeros(n, np.int32) if roles is None else np.asarray(roles, np.int32)),
        wp.array(np.asarray(directions, np.float64), dtype=wp.vec3d),
        wp.ones(n, dtype=wp.float64),
        wp.zeros(1, dtype=int),
        wp.zeros(n, dtype=int),
    )


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
@pytest.mark.parametrize("damping", [0.25, 1.0])
def test_trust_ball_preserves_material_and_boundary_mask(scale, damping):
    shift = np.array([10, -3, 4]) * scale
    surface = cube(scale, shift)
    domain = SurfaceDomain(*surface)
    local = np.array(
        [
            [0.99, 0, 0],
            [0, 0, 0],
            [1.5, 0, 0],
            [1, 0, 0],
            [0.99, 0, 0],
            [0.99, 0, 0],
            [0.99, 0, 0],
            [1.5, 0, 0],
        ]
    )
    original = (local * scale + shift).astype(np.float32)
    original[3, 0] = surface[0][:, 0].max()  # Exact stored surface coordinate.
    initial = (
        np.array(
            [
                [1, 0.3, 0.2],
                [0.1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        * scale
    )
    p, roles, delta, edge, error, interior = arrays(original, initial, [0, 0, 0, 0, 1, 2, 3, 0])
    edge.fill_(scale)
    wp.launch(
        constrain,
        len(p),
        inputs=[domain.tree, p, roles, delta, edge, 0.25, damping, error, interior],
    )
    assert error.numpy()[0] == 0
    d = delta.numpy()
    for alpha in [damping, damping / 2, damping / 32]:
        q = (original.astype(float) + alpha * d).astype(np.float32)
        assert solid_angle_inside(*surface, q[:2]).all()
        assert np.all(q[0] > surface[0].min(axis=0))
        assert np.all(q[0] < surface[0].max(axis=0))
    np.testing.assert_array_equal(d[1], initial[1])
    np.testing.assert_allclose(d[2], [-0.25 * scale, 0, 0], atol=scale * 1e-6)
    np.testing.assert_array_equal(d[3], 0)
    np.testing.assert_array_equal(d[4:], initial[4:])
    np.testing.assert_array_equal(p.numpy(), original)


def test_material_trust_region_respects_cavity_and_disconnected_gap():
    inner = cube(0.4)
    cases = [
        (combine(cube(), (inner[0], inner[1][:, ::-1].copy())), [0.41, 0, 0], [-2, 0, 0]),
        (combine(cube(0.9, (-2, 0, 0)), cube(0.9, (2, 0, 0))), [-1.11, 0, 0], [10, 0, 0]),
    ]
    for surface, point, direction in cases:
        domain = SurfaceDomain(*surface)
        p, roles, delta, edge, error, interior = arrays([point], [direction])
        wp.launch(
            constrain, 1, inputs=[domain.tree, p, roles, delta, edge, 0.5, 1.0, error, interior]
        )
        assert error.numpy()[0] == 0
        movement = delta.numpy()[0]
        assert 0 < np.linalg.norm(movement) < 0.01
        samples = p.numpy().astype(float) + np.linspace(0, 1, 17)[:, None] * movement
        assert solid_angle_inside(*surface, samples).all()


def test_cached_interior_classification_is_invalidated_by_boundary_membership():
    domain = SurfaceDomain(*cube())
    p, roles, delta, edge, error, interior = arrays([[0.99, 0, 0]], [[1, 0, 0]])

    def run():
        wp.launch(
            constrain,
            1,
            inputs=[domain.tree, p, roles, delta, edge, 0.5, 0.5, error, interior],
        )
        assert error.numpy()[0] == 0

    run()
    assert interior.numpy()[0] == 1
    moved = (p.numpy().astype(float) + 0.5 * delta.numpy()).astype(np.float32)
    assert solid_angle_inside(*cube(), moved).all()
    p.assign(moved)
    delta.assign(np.array([[0.01, 0, 0]], np.float64))
    run()
    assert interior.numpy()[0] == 1
    assert solid_angle_inside(*cube(), p.numpy().astype(float) + 0.5 * delta.numpy()).all()
    # A boundary centroid can lie outside a nonconvex input, then lose its
    # quadrature assignment. Even a masked boundary step invalidates the cache.
    p.assign(np.array([[1.5, 0, 0]], np.float32))
    roles.fill_(1)
    delta.zero_()
    run()
    assert interior.numpy()[0] == 0
    roles.zero_()
    delta.assign(np.array([[1, 0, 0]], np.float64))
    run()
    np.testing.assert_array_equal(delta.numpy(), [[-0.5, 0, 0]])
    assert interior.numpy()[0] == 0


@pytest.mark.parametrize("graded", [False, True])
def test_real_hand_escape_is_prevented_with_active_nan_padding(graded):
    data = np.load(Path(__file__).parent / "data/unassigned-hand-sites.npz")
    original = data["points"]
    domain = SurfaceDomain(data["input_points"], data["input_faces"], surface_subdivisions=4)
    positions = np.full((320, 3), np.nan, np.float32)
    positions[: len(original)] = original
    points = wp.array(positions, dtype=wp.vec3, device="cuda:0")
    active = wp.array([len(original)], dtype=int, device=points.device)
    sizing = None
    if graded:
        sizing = SizingField(
            np.array([[0, 0, 0]], np.float32),
            wp.array([0.2], dtype=float, device=points.device),
            gradation=0,
        )
    core = VTMWorkspace(domain, points, active_count=active, sizing=sizing, max_iterations=5).run()
    assert core.error.numpy()[0] == 0
    assert core.iterations.numpy()[0] == 5
    assert core.mesh.certificate_error.numpy()[0] == 0
    result = points.numpy()
    assert np.isnan(result[len(original) :]).all()
    assert np.all(result[: len(original)] >= data["input_points"].min(axis=0))
    assert np.all(result[: len(original)] <= data["input_points"].max(axis=0))
    assert np.any(result[: len(original)] != original)
    tets = core.full_tetrahedra.numpy()[: core.full_tetrahedra_count.numpy()[0]]
    assert tets.max() < len(original)
    # Independent integer arithmetic checks physical orientation, including
    # any nearly flat cells that ordinary double determinants could mis-sign.
    assert all(exact_det(result[tet]) > 0 for tet in tets)


@wp.kernel
def produce(phase: wp.array[int], points: wp.array[wp.vec3], delta: wp.array[wp.vec3d]):
    points[0] = wp.vec3(0.99, 0.0, 0.0)
    if phase[0] == 1:
        points[0] = wp.vec3(NAN, 0.0, 0.0)
    if phase[0] == 2:
        points[0] = wp.vec3(1.5, 0.0, 0.0)
    points[1] = wp.vec3(NAN)
    delta[0] = wp.vec3d(1.0, 0.0, 0.0)
    delta[1] = wp.vec3d(0.0)


@wp.kernel
def consume(
    phase: wp.array[int],
    delta: wp.array[wp.vec3d],
    error: wp.array[int],
    output: wp.array2d[wp.float64],
):
    output[phase[0], 0] = wp.float64(error[0])
    output[phase[0], 1] = delta[0][0]
    output[phase[0], 2] = wp.length_sq(delta[1])
    phase[0] = (phase[0] + 1) % 4


def test_captured_changing_producer_guard_consumer_recovers_without_host_work(
    monkeypatch, tmp_path
):
    domain = SurfaceDomain(*cube())
    p, roles, delta, edge, error, interior = arrays(np.zeros((2, 3)), np.zeros((2, 3)))
    phase = wp.zeros(1, dtype=int)
    output = wp.empty((4, 3), dtype=wp.float64)
    wp.load_module(module=__name__, device=p.device)
    wp.load_module(module="warpower._vtm_domain", device=p.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected allocation, synchronization, or readback")

    with monkeypatch.context() as patch:
        for name in ["empty", "zeros", "ones", "clone", "synchronize", "synchronize_device"]:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(produce, 1, inputs=[phase, p, delta])
            error.zero_()
            interior.zero_()
            wp.launch(
                constrain,
                2,
                inputs=[domain.tree, p, roles, delta, edge, 0.25, 0.5, error, interior],
            )
            wp.launch(consume, 1, inputs=[phase, delta, error, output])
        for _ in range(4):
            wp.capture_launch(capture.graph)
    values = output.numpy()
    np.testing.assert_array_equal(values[:, 0], [0, 64, 0, 0])
    assert 0 < values[0, 1] < 0.02
    assert values[1, 1] == 0
    assert values[2, 1] == -0.25
    assert values[3, 1] == values[0, 1]
    np.testing.assert_array_equal(values[:, 2], 0)
    path = tmp_path / "guard.dot"
    wp.capture_debug_dot_print(capture.graph, str(path), verbose=True)
    graph = path.read_text()
    assert "HOST" not in graph and "ALLOC" not in graph and "FREE" not in graph


@wp.kernel
def fail_guard(error: wp.array[int]):
    error[0] = 64


def test_guard_failure_restores_core_and_invalidates_output():
    data = np.load(Path(__file__).parent / "data/unassigned-hand-sites.npz")
    domain = SurfaceDomain(data["input_points"], data["input_faces"], surface_subdivisions=4)
    core = VTMWorkspace(domain, data["points"], max_iterations=2)
    original = core._launch

    def launch(kernel, dim, inputs):
        original(kernel, dim, inputs)
        if kernel == constrain:
            wp.launch(fail_guard, 1, inputs=[core.error])

    core._launch = launch
    wp.load_module(module=__name__, device=core.device)
    with wp.ScopedCapture() as capture:
        core.run()
    wp.capture_launch(capture.graph)
    assert core.error.numpy()[0] == 64
    assert core.full_tetrahedra_count.numpy()[0] == 0
    assert core.iterations.numpy()[0] == 0
    np.testing.assert_array_equal(core.points.numpy(), data["points"])
