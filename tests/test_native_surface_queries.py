"""Opt-in native queries: public defaults, analytic geometry and captured solves."""

import re

import numpy as np
import pytest
import warp as wp

from tests.test_domain import combine, cube, torus
from tests.test_mesher import validate
from warpower import DomainQueryWorkspace, SurfaceDomain, VariationalMeshingWorkspace


def test_backend_is_explicit_and_reference_remains_default():
    p, f = cube()
    reference = SurfaceDomain(p, f)
    assert reference.query_backend == "reference"
    assert reference._native_mesh is None and reference._native_queries is None
    with pytest.raises(ValueError, match="query_backend"):
        SurfaceDomain(p, f, query_backend="unknown")


@pytest.mark.parametrize("validated", [True, False])
def test_native_analytic_distances_and_captured_deformation(validated, tmp_path):
    p, f = cube()
    points = wp.array(p, dtype=wp.vec3, device="cuda:0")
    domain = SurfaceDomain(points, f, query_backend="native", validate_geometry=validated)
    locations = np.array([[0.2, 0.1, 0.3], [2, 0, 0], [1, 0.2, 0.1]], np.float32)
    queries = DomainQueryWorkspace(domain, locations)
    with wp.ScopedCapture() as capture:
        domain.refit()
        queries.run()
    dot = tmp_path / "native-refit.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    node_types = set(re.findall(r'label="\{\s*([A-Z0-9_]+)\s*\n', dot.read_text()))
    assert "KERNEL" in node_types
    assert node_types <= {"KERNEL", "MEMSET", "MEMCPY", "MEM_ALLOC", "MEM_FREE"}
    # Warp 1.19's native refit scan allocates device scratch. Query-only loops
    # below preserve the fixed-buffer contract; refit has a documented limit.
    assert "MEM_ALLOC" in node_types and "MEM_FREE" in node_types
    with wp.ScopedCapture() as query_capture:
        queries.run()
    wp.capture_debug_dot_print(query_capture.graph, str(dot), verbose=True)
    query_types = set(re.findall(r'label="\{\s*([A-Z0-9_]+)\s*\n', dot.read_text()))
    assert query_types <= {"KERNEL", "MEMSET", "MEMCPY"}
    pointers = [queries.closest.ptr, queries.signed_distance.ptr, domain.points.ptr]
    for shift in [0.0, 4.0, 0.0]:
        translated = p.copy()
        translated[:, 0] += shift
        wp.copy(points, wp.array(translated, dtype=wp.vec3, device=domain.device))
        wp.capture_launch(capture.graph)
        wp.synchronize()
        assert domain.error.numpy()[0] == queries.error.numpy()[0] == 0
        np.testing.assert_array_equal(queries.valid.numpy(), 1)
        delta = np.abs(locations - [shift, 0, 0]) - 1
        expected = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)
        np.testing.assert_allclose(queries.signed_distance.numpy(), expected, rtol=3e-6, atol=3e-6)
        assert pointers == [queries.closest.ptr, queries.signed_distance.ptr, domain.points.ptr]


def test_native_query_limits_invalid_points_and_domain_recovery():
    p, f = cube()
    domain = SurfaceDomain(p, f, query_backend="native")
    query = DomainQueryWorkspace(
        domain, np.array([[3, 0, 0], [np.nan, 0, 0]], np.float32), max_distance=0.1
    )
    query.run()
    np.testing.assert_array_equal(query.valid.numpy(), 0)
    np.testing.assert_array_equal(query.face.numpy(), -1)
    assert query.error.numpy()[0] == 64
    wp.copy(
        query.points, wp.array([[0.1, 0.2, 0.3], [1, 0, 0]], dtype=wp.vec3, device=domain.device)
    )
    query.run()
    assert query.error.numpy()[0] == 0
    np.testing.assert_array_equal(query.valid.numpy(), [0, 1])
    bad = p.copy()
    bad[0] = np.nan
    wp.copy(domain.points, wp.array(bad, dtype=wp.vec3, device=domain.device))
    domain.refit()
    query.run()
    assert domain.error.numpy()[0] != 0 and query.error.numpy()[0] != 0
    np.testing.assert_array_equal(query.valid.numpy(), 0)
    wp.copy(domain.points, wp.array(p, dtype=wp.vec3, device=domain.device))
    domain.refit()
    query.run()
    assert domain.error.numpy()[0] == query.error.numpy()[0] == 0
    np.testing.assert_array_equal(query.valid.numpy(), [0, 1])


def test_native_sign_respects_an_inward_cavity():
    inner = cube(0.4)
    p, f = combine(cube(), (inner[0], inner[1][:, ::-1].copy()))
    domain = SurfaceDomain(p, f, query_backend="native")
    query = DomainQueryWorkspace(
        domain, np.array([[0, 0, 0], [0.7, 0, 0], [2, 0, 0]], np.float32)
    ).run()
    assert query.error.numpy()[0] == 0
    np.testing.assert_array_equal(query.inside.numpy(), [0, 1, 0])
    np.testing.assert_allclose(query.signed_distance.numpy(), [0.4, -0.3, 1], rtol=3e-6, atol=3e-6)


@wp.kernel
def translate(points: wp.array[wp.vec3], locations: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] += wp.vec3(0.125, 0.0, 0.0)
    if i < locations.shape[0]:
        locations[i] += wp.vec3(0.125, 0.0, 0.0)


@wp.kernel
def consume(distance: wp.array[float], valid: wp.array[int], result: wp.array[float]):
    i = wp.tid()
    result[i] = wp.where(valid[i] != 0, distance[i], 99.0)


def test_native_device_producer_query_consumer_loop():
    domain = SurfaceDomain(*cube(), query_backend="native", validate_geometry=False)
    query = DomainQueryWorkspace(domain, np.array([[0.2, 0, 0], [2, 0, 0]], np.float32))
    result = wp.empty(2, dtype=float, device=domain.device)
    wp.load_module(module=__name__, device=domain.device)
    with wp.ScopedCapture() as capture:
        wp.launch(translate, len(domain.points), inputs=[domain.points, query.points])
        domain.refit()
        query.run()
        wp.launch(consume, len(result), inputs=[query.signed_distance, query.valid, result])
    for _ in range(3):
        wp.capture_launch(capture.graph)
    np.testing.assert_allclose(result.numpy(), [-0.8, 1.0], rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("shape", ["cube", "torus"])
def test_native_complete_captured_mesher_and_fresh_replay(shape):
    surface, n, initial = (cube(), 128, 64) if shape == "cube" else (torus(24, 12), 256, 64)
    domain = SurfaceDomain(
        *surface, surface_subdivisions=4, query_backend="native", validate_geometry=False
    )
    mesh = VariationalMeshingWorkspace(domain, n, initial_count=initial, quality_rounds=1)
    with wp.ScopedCapture() as capture:
        domain.refit()
        mesh.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        wp.synchronize()
        validate(mesh)
        assert mesh.core.selection._query_cache is None
        assert mesh.extraction._query_cache is None
        assert mesh.rounds.numpy()[2] == 1
