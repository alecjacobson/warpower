from itertools import product

import numpy as np
import pytest
import warp as wp

from tests.test_predicates import exact_reference
from tests.test_tetrahedra import check_mesh, reference, tet_set
from warpower import DelaunayWorkspace
from warpower._canonical import canonicalize
from warpower._delaunay_certificate import DelaunayCertificate
from warpower._site_bvh import SiteHierarchy


def valid(workspace, original):
    assert workspace.error.numpy()[0] == 0, (
        workspace.base_error.numpy(),
        workspace.candidate_error.numpy(),
        workspace.certificate_error.numpy(),
        workspace.attempts.numpy(),
    )
    assert workspace.certificate_error.numpy()[0] == 0
    np.testing.assert_array_equal(workspace.points.numpy(), original)
    tets = workspace.tetrahedra.numpy()[: workspace.tetrahedra_count.numpy()[0]]
    check_mesh(original, tets)
    assert set(tets.ravel()) == set(np.unique(original, axis=0, return_index=True)[1])
    return tets


def exact_empty(points, tets):
    for tet in tets:
        for i in range(len(points)):
            if i in tet:
                continue
            orientation, sphere = exact_reference(points[[*tet, i]], np.zeros(5))
            assert orientation > 0
            assert sphere <= 0


@pytest.mark.parametrize("n", [2, 3, 4])
def test_lattices_recover_complete_positive_mesh_without_moving_sites(n):
    points = np.array(list(product(range(n), repeat=3)), np.float32)
    w = DelaunayWorkspace(points).run()
    tets = valid(w, points)
    assert w.recovered.numpy()[0] == 1
    if n <= 3:
        exact_empty(points, tets)
    first = tet_set(tets)
    w.run()
    assert first == tet_set(valid(w, points))


def test_cospherical_octahedron_duplicates_and_signed_zero():
    unique = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32
    )
    points = np.repeat(unique, 2, axis=0)
    points[1::2] = np.where(points[1::2] == 0, -0.0, points[1::2])
    w = DelaunayWorkspace(points).run()
    tets = valid(w, points)
    assert np.all(tets % 2 == 0)
    exact_empty(points, tets)


def test_nondegenerate_input_matches_scipy_without_recovery():
    points = np.random.default_rng(42).normal(size=(200, 3)).astype(np.float32)
    w = DelaunayWorkspace(points).run()
    tets = valid(w, points)
    assert tet_set(tets) == reference(points)
    assert w.attempts.numpy()[0] == 0


def test_corner_only_cube_initialization_that_failed_the_original_builder():
    from tests.test_domain import cube
    from warpower import BoundaryWorkspace, SurfaceDomain
    from warpower._vtm import anchor_corners

    original = np.random.default_rng(2026).uniform(-0.9, 0.9, (200, 3)).astype(np.float32)
    domain = SurfaceDomain(*cube(), surface_subdivisions=16)
    boundary = BoundaryWorkspace(domain, original).run()
    wp.launch(
        anchor_corners,
        len(original),
        inputs=[boundary.points, boundary.targets, boundary.roles, boundary.error],
    )
    points = boundary.points.numpy()
    w = DelaunayWorkspace(boundary.points).run()
    tets = valid(w, points)
    assert w.base_error.numpy()[0] != 0
    assert w.recovered.numpy()[0] == 1
    # Coplanar cube quads admit different valid boundary diagonals.
    assert len(tets) == 1207
    alternative = reference(points)
    ambiguous = np.array([t for t in tets if tuple(sorted(t)) not in alternative], dtype=np.int32)
    exact_empty(points, ambiguous)


@pytest.mark.parametrize("scale,shift", [(2.0**-40, 0), (1.0, 2**12), (2.0**40, 2**12)])
def test_translated_scaled_lattice(scale, shift):
    points = ((np.array(list(product(range(3), repeat=3)), dtype=float) + shift) * scale).astype(
        np.float32
    )
    w = DelaunayWorkspace(points).run()
    tets = valid(w, points)
    exact_empty(points, tets)


@pytest.mark.parametrize("toward", [-np.inf, np.inf])
def test_one_ulp_departure_from_exact_cospherical_cube(toward):
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    points[0, 0] = np.nextafter(points[0, 0], np.float32(toward))
    w = DelaunayWorkspace(points).run()
    exact_empty(points, valid(w, points))


@pytest.mark.parametrize("height", [1.0e-4, 1.0e-7])
def test_nearly_coplanar_sites_and_far_circumcenters(height):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, height], [0.8, 0.2, 1.05 * height]], np.float32
    )
    w = DelaunayWorkspace(points).run()
    exact_empty(points, valid(w, points))


def certificate(points, tets):
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    t = wp.array(tets, dtype=int, device=p.device)
    count = wp.full(1, len(tets), dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    weights = wp.zeros(len(p), dtype=float, device=p.device)
    owners, buffers = canonicalize(p, weights)
    hierarchy.tree.owners = owners
    c = DelaunayCertificate(p, t, count, hierarchy.tree)
    with wp.ScopedCapture() as capture:
        c.run()
    wp.capture_launch(capture.graph)
    return int(c.error.numpy()[0])


def test_certificate_rejects_missing_duplicate_inverted_and_nonempty_cells():
    points = np.random.default_rng(321).uniform(-1, 1, (20, 3)).astype(np.float32)
    w = DelaunayWorkspace(points).run()
    tets = valid(w, points)
    assert certificate(points, tets) == 0
    assert certificate(points, tets[:-1]) & (64 | 128)
    assert certificate(points, np.vstack([tets, tets[:1]])) & 32
    inverse = tets.copy()
    inverse[0, [0, 1]] = inverse[0, [1, 0]]
    assert certificate(points, inverse) & 4
    # One positive hull tetrahedron contains a fifth site in its circumball.
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]], np.float32)
    assert certificate(p, np.array([[0, 1, 2, 3]], np.int32)) & 8


def test_failure_count_zero_and_next_run_recovers():
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(points).run()
    valid(w, points)
    bad = points.copy()
    bad[0, 0] = np.nan
    w.points.assign(bad)
    w.run()
    assert w.error.numpy()[0] == 1
    assert w.tetrahedra_count.numpy()[0] == 0
    w.points.assign(points)
    w.run()
    valid(w, points)
    # Entirely planar input cannot produce a volumetric convex-hull mesh.
    flat = points.copy()
    flat[:, 2] = 0
    w.points.assign(flat)
    w.run()
    assert w.error.numpy()[0] != 0
    assert w.tetrahedra_count.numpy()[0] == 0
    assert w.attempts.numpy()[0] == w.max_attempts
    w.points.assign(points)
    w.run()
    valid(w, points)


@wp.kernel
def produce(source: wp.array[wp.vec3], output: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    p = source[i]
    if i == 0:
        p[0] += float(phase[0]) * 0.0001
    output[i] = p


@wp.kernel
def consume(
    tets: wp.array2d[int], count: wp.array[int], error: wp.array[int], total: wp.array[int]
):
    i = wp.tid()
    if error[0] == 0 and i < count[0]:
        wp.atomic_add(total, 0, tets[i, 0] + tets[i, 1] + tets[i, 2] + tets[i, 3])


def test_capture_connectivity_changing_producer_and_consumer(tmp_path):
    original = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(original)
    source = wp.array(original, dtype=wp.vec3, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    total = wp.zeros(1, dtype=int, device=w.device)
    wp.load_module(module=__name__, device=w.device)
    pointers = (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, len(original), inputs=[source, w.points, phase])
        w.run()
        total.zero_()
        wp.launch(
            consume, len(w.tetrahedra), inputs=[w.tetrahedra, w.tetrahedra_count, w.error, total]
        )
    for step in [0, 1, -1, 0]:
        phase.fill_(step)
        wp.capture_launch(capture.graph)
        expected = original.copy()
        expected[0, 0] += np.float32(step) * np.float32(0.0001)
        t = valid(w, expected)
        exact_empty(expected, t)
        assert total.numpy()[0] == t.sum()
        assert pointers == (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    path = tmp_path / "recovery.dot"
    wp.capture_debug_dot_print(capture.graph, str(path))
    dot = path.read_text().lower()
    assert "host" not in dot and "mem_alloc" not in dot and "mem_free" not in dot


@wp.kernel
def produce_cached(
    source: wp.array[wp.vec3],
    alternate: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
):
    i = wp.tid()
    p = source[i]
    if phase[0] == 1 or phase[0] == 2:
        p = p * 2.0 + wp.vec3(0.125, 2.25, -1.0)
    if phase[0] == 2 and i == points.shape[0] - 1:
        p = source[0] * 2.0 + wp.vec3(0.125, 2.25, -1.0)
    if phase[0] == 3:
        p = alternate[i]
    points[i] = p


def test_cached_strategy_rebuilds_current_bounds_owners_and_connectivity_in_capture(tmp_path):
    original = np.array(list(product(range(4), repeat=3)), np.float32)
    alternate = np.random.default_rng(116).uniform(-1, 1, original.shape).astype(np.float32)
    w = DelaunayWorkspace(original)
    source = wp.array(original, dtype=wp.vec3, device=w.device)
    alternative = wp.array(alternate, dtype=wp.vec3, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    wp.load_module(module=__name__, device=w.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce_cached, len(original), inputs=[source, alternative, w.points, phase])
        w.run()
    previous_recovered = 0
    for step in [0, 1, 2, 3, 0]:
        phase.fill_(step)
        wp.capture_launch(capture.graph)
        expected = original.copy()
        if step in [1, 2]:
            expected = expected * 2 + np.array([0.125, 2.25, -1], np.float32)
        if step == 2:
            expected[-1] = expected[0]
        if step == 3:
            expected = alternate.copy()
        tets = valid(w, expected)
        assert w.recovery_first.numpy()[0] == previous_recovered
        if step in [1, 2]:
            assert w.base_attempted.numpy()[0] == 0
        if step == 3:
            assert tet_set(tets) == reference(expected)
        previous_recovered = int(w.recovered.numpy()[0])
    path = tmp_path / "cached.dot"
    wp.capture_debug_dot_print(capture.graph, str(path))
    dot = path.read_text().lower()
    assert "host" not in dot and "mem_alloc" not in dot and "mem_free" not in dot


@wp.kernel
def fail_candidate(error: wp.array[int], count: wp.array[int]):
    error[0] = 2
    count[0] = 0


def test_exhausted_cached_candidates_fall_back_to_current_base_mesh():
    original = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(original, max_attempts=2).run()
    valid(w, original)
    changed = np.random.default_rng(131).uniform(-1, 1, original.shape).astype(np.float32)
    w.points.assign(changed)
    real_run = w._auxiliary.run

    def fail_run(stream=None):
        real_run(stream)
        wp.launch(fail_candidate, 1, inputs=[w._auxiliary.error, w._auxiliary.tetrahedra_count])

    w._auxiliary.run = fail_run
    wp.load_module(module=__name__, device=w.device)
    with wp.ScopedCapture() as capture:
        w.run()
    wp.capture_launch(capture.graph)
    assert tet_set(valid(w, changed)) == reference(changed)
    assert w.recovery_first.numpy()[0] == 1
    assert w.base_attempted.numpy()[0] == 1
    assert w.attempts.numpy()[0] == 2
    assert w.recovered.numpy()[0] == 0
    wp.capture_launch(capture.graph)
    assert tet_set(valid(w, changed)) == reference(changed)
    assert w.recovery_first.numpy()[0] == 0
    assert w.attempts.numpy()[0] == 0


def test_strategy_cache_can_be_disabled_for_comparisons():
    original = np.array(list(product([-1, 1], repeat=3)), np.float32)
    w = DelaunayWorkspace(original, cache_strategy=False)
    for _ in range(2):
        w.run()
        valid(w, original)
        assert w.base_attempted.numpy()[0] == 1
        assert w.recovery_first.numpy()[0] == 0
        assert w.recovered.numpy()[0] == 1
