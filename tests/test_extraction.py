import itertools

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

import warpower
from tests.test_domain import brute_distances, combine, cube, solid_angle_inside, torus
from tests.test_vtm_core import torus_sites
from warpower._vtm import VTMWorkspace


def orient(points, tets):
    tets = np.array(tets, np.int32, copy=True)
    p = np.asarray(points, np.float64)[tets]
    det = np.linalg.det(p[:, 1:] - p[:, :1])
    negative = det < 0
    tets[negative, :2] = tets[negative, 1::-1]
    assert np.all(det != 0)
    return tets


def grid_mesh(x, y, z):
    """Independent conforming six-tet cube subdivision, including exact ties."""
    p = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    shape = len(x), len(y), len(z)
    tets = []
    for cell in itertools.product(*(range(n - 1) for n in shape)):
        for permutation in itertools.permutations(range(3)):
            v = np.array(cell)
            ids = [np.ravel_multi_index(v, shape)]
            for k in permutation:
                v[k] += 1
                ids.append(np.ravel_multi_index(v, shape))
            tets.append(ids)
    return p, orient(p, tets)


def independent_topology(points, tets):
    """Build face incidence, verify outward orientation and closed edge links."""
    faces = {}
    neighbors = [set() for _ in tets]
    for i, tet in enumerate(tets):
        for opposite in range(4):
            f = [tet[k] for k in range(4) if k != opposite]
            a, b, c = points[f].astype(np.float64)
            if np.dot(np.cross(b - a, c - a), points[tet[opposite]] - a) > 0:
                f[0], f[1] = f[1], f[0]
            faces.setdefault(tuple(sorted(f)), []).append((i, f))
    boundary = []
    for pair in faces.values():
        assert len(pair) in (1, 2)
        if len(pair) == 1:
            boundary.append(pair[0][1])
        else:
            (a, _), (b, _) = pair
            neighbors[a].add(b)
            neighbors[b].add(a)
    components = 0
    seen = set()
    for i in range(len(tets)):
        if i in seen:
            continue
        components += 1
        stack = [i]
        while stack:
            at = stack.pop()
            if at not in seen:
                seen.add(at)
                stack.extend(neighbors[at] - seen)
    edges = {}
    for f in boundary:
        for a, b in zip(f, np.roll(f, -1)):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    for pair in edges.values():
        assert len(pair) == 2
        assert pair[0] == pair[1][::-1]
    chi = len({v for f in boundary for v in f}) - len(edges) + len(boundary)
    return np.array(boundary), components, chi


def extracted(w):
    assert w.error.numpy()[0] == 0
    assert w.diagnostics.numpy()[0] == 0
    assert w.valid.numpy()[0] == 1
    p = w.points.numpy()
    t = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
    f = w.boundary_faces.numpy()[: w.boundary_count.numpy()[0]]
    reference, components, chi = independent_topology(p, t)
    assert {tuple(sorted(x)) for x in f} == {tuple(sorted(x)) for x in reference}
    assert components == w.component_count.numpy()[0]
    assert chi == w.shell_euler.numpy().sum()
    cells = p[t].astype(np.float64)
    det = np.linalg.det(cells[:, 1:] - cells[:, :1])
    assert np.all(det > 0)
    np.testing.assert_allclose(det.sum() / 6, w.volume.numpy()[0], rtol=1e-12)
    np.testing.assert_allclose(w.shell_volume.numpy().sum(), w.volume.numpy()[0], rtol=1e-12)
    return p, t, f


@pytest.mark.parametrize("tolerance", [0.0, 0.4])
def test_circumcenters_distances_and_selection_match_independent_triangle_oracles(tolerance):
    vertices, faces = torus(16, 8)
    domain = warpower.SurfaceDomain(vertices, faces)
    points = np.random.default_rng(102).uniform(-3, 3, (100, 3)).astype(np.float32)
    tets = orient(points, Delaunay(points).simplices)
    w = warpower.ExtractionWorkspace(domain, points, tets, tolerance=tolerance).run()
    assert w.error.numpy()[0] == 0
    p = points[tets].astype(np.float64)
    a = p[:, 1:] - p[:, :1]
    offset = np.linalg.solve(2 * a, np.sum(a * a, axis=2)[..., None])[..., 0]
    centers = p[:, 0] + offset
    radii = np.linalg.norm(offset, axis=1)
    inside = solid_angle_inside(vertices, faces, centers)
    distances = brute_distances(vertices, faces, centers)
    mask = inside | (distances / radii < tolerance)
    np.testing.assert_allclose(w.circumcenters.numpy(), centers, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(w.circumradii.numpy(), radii, rtol=1e-11)
    np.testing.assert_allclose(
        w.signed_distance.numpy(), np.where(inside, -distances, distances), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_array_equal(w.selected.numpy(), mask)
    np.testing.assert_array_equal(w.tetrahedra.numpy()[: mask.sum()], tets[mask])
    np.testing.assert_array_equal(w.source_indices.numpy()[: mask.sum()], np.flatnonzero(mask))


def test_cube_exact_volume_orientation_quality_and_surface_integrals():
    p, t = grid_mesh(*([np.linspace(-1, 1, 5)] * 3))
    w = warpower.ExtractionWorkspace(warpower.SurfaceDomain(*cube()), p, t).run()
    p, t, _ = extracted(w)
    assert len(t) == 384
    assert w.shell_count.numpy()[0] == 1
    assert w.shell_euler.numpy().sum() == 2
    assert w.cavity_count.numpy()[0] == 0
    np.testing.assert_allclose(w.volume.numpy(), 8)
    np.testing.assert_allclose(w.domain_volume.numpy(), 8)
    np.testing.assert_allclose(w.boundary_area.numpy(), 24)
    np.testing.assert_allclose(w.volume_difference.numpy(), 0, atol=1e-12)
    assert w.surface_rms_distance.numpy()[0] < 1e-15
    assert w.surface_max_sample_distance.numpy()[0] < 1e-15
    expected = 3 / (np.sqrt(3) * (1 + np.sqrt(2)))
    np.testing.assert_allclose(w.radius_ratio.numpy()[: len(t)], expected, rtol=1e-12)


def test_outside_center_thickening_and_strict_threshold_with_independent_surface_error():
    p = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float64)
    p = (p * (0.4 / np.sqrt(3)) + [1.1, 0, 0]).astype(np.float32)
    t = orient(p, [[0, 1, 2, 3]])
    vertices, faces = cube()
    domain = warpower.SurfaceDomain(vertices, faces)
    strict = warpower.ExtractionWorkspace(domain, p, t, tolerance=0).run()
    thick = warpower.ExtractionWorkspace(domain, p, t).run()
    assert strict.tetrahedra_count.numpy()[0] == 0
    assert thick.tetrahedra_count.numpy()[0] == 1
    ratio = thick.signed_distance.numpy()[0] / thick.circumradii.numpy()[0]
    exact = warpower.ExtractionWorkspace(domain, p, t, tolerance=ratio).run()
    assert exact.tetrahedra_count.numpy()[0] == 0  # Strict '<', as in the paper.
    _, _, boundary = extracted(thick)
    tri = p[boundary].astype(np.float64)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    sample = tri.sum(axis=1)[:, None, :] / 6 + tri / 2
    distances = brute_distances(vertices, faces, sample.reshape(-1, 3)).reshape(-1, 3)
    expected = np.sqrt(np.sum(area[:, None] * distances**2 / 3) / area.sum())
    np.testing.assert_allclose(thick.surface_rms_distance.numpy()[0], expected, rtol=1e-12)
    np.testing.assert_allclose(
        thick.surface_max_sample_distance.numpy()[0], distances.max(), rtol=1e-12
    )


def test_invalid_domain_and_inverted_tet_are_rejected_and_recover():
    vertices, faces = cube()
    domain = warpower.SurfaceDomain(vertices, faces)
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    t = wp.array(np.array([[0, 2, 1, 3]], np.int32), device="cuda:0")
    w = warpower.ExtractionWorkspace(domain, p, t).run()
    assert w.error.numpy()[0] == 8 and w.tetrahedra_count.numpy()[0] == 0
    wp.copy(t, wp.array(np.array([[0, 1, 2, 3]], np.int32), device="cuda:0"))
    domain.points.zero_()
    domain.refit()
    w.run()
    assert w.error.numpy()[0] == 1 and w.tetrahedra_count.numpy()[0] == 0
    wp.copy(domain.points, wp.array(vertices, dtype=wp.vec3, device="cuda:0"))
    domain.refit()
    w.run()
    extracted(w)


def test_enclosed_cavity_retains_inward_shell_and_exact_void_volume():
    p, t = grid_mesh(*([np.linspace(-1, 1, 7)] * 3))
    inner = cube(1 / 3)
    domain = warpower.SurfaceDomain(*combine(cube(), (inner[0], inner[1][:, ::-1].copy())))
    w = warpower.ExtractionWorkspace(domain, p, t).run()
    extracted(w)
    assert w.component_count.numpy()[0] == 1
    assert w.shell_count.numpy()[0] == 2
    assert w.cavity_count.numpy()[0] == 1
    assert w.shell_euler.numpy().sum() == 4
    volumes = w.shell_volume.numpy()
    np.testing.assert_allclose(volumes[volumes < 0].sum(), -((2 / 3) ** 3), rtol=1e-6)
    np.testing.assert_allclose(w.volume.numpy()[0], 8 - (2 / 3) ** 3, rtol=1e-7)
    assert abs(w.volume_difference.numpy()[0]) < 1e-12


def test_disconnected_components_do_not_acquire_a_convex_hull_bridge():
    p, t = grid_mesh(np.linspace(-3, 3, 13), np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    d = warpower.SurfaceDomain(*combine(cube(center=(-2, 0, 0)), cube(center=(2, 0, 0))))
    w = warpower.ExtractionWorkspace(d, p, t).run()
    p, t, _ = extracted(w)
    assert w.component_count.numpy()[0] == 2
    assert w.shell_count.numpy()[0] == 2
    assert w.shell_euler.numpy().sum() == 4
    assert w.cavity_count.numpy()[0] == 0
    np.testing.assert_allclose(w.volume.numpy()[0], 16)
    assert np.all(np.abs(p[t].mean(axis=1)[:, 0]) > 1)


def test_fitted_torus_produces_one_closed_genus_one_boundary():
    d = warpower.SurfaceDomain(*torus(24, 12), surface_subdivisions=3)
    v = VTMWorkspace(d, torus_sites(300), max_iterations=16).run()
    assert v.error.numpy()[0] == 0
    w = warpower.ExtractionWorkspace(
        d, v.points, v.full_tetrahedra, v.full_tetrahedra_count, source_error=v.error
    ).run()
    _, t, _ = extracted(w)
    assert 0 < len(t) < v.full_tetrahedra_count.numpy()[0]
    assert w.shell_count.numpy()[0] == 1
    assert w.shell_euler.numpy().sum() == 0
    assert w.cavity_count.numpy()[0] == 0
    assert abs(w.volume_difference.numpy()[0]) < 0.2 * w.domain_volume.numpy()[0]
    assert w.surface_rms_distance.numpy()[0] < 0.1


def test_topology_diagnostics_detect_duplicated_tets_edge_and_vertex_contacts():
    d = warpower.SurfaceDomain(*cube(10))
    p = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32
    )
    cases = [
        ([[0, 1, 2, 3], [0, 1, 2, 3]], 1),
        ([[0, 1, 2, 3], [0, 1, 5, 6]], 2),
        ([[0, 1, 2, 3], [0, 4, 5, 6]], 4),
    ]
    for t, bit in cases:
        w = warpower.ExtractionWorkspace(d, p, orient(p, t)).run()
        assert w.error.numpy()[0] == 0
        assert w.diagnostics.numpy()[0] & bit
        assert w.valid.numpy()[0] == 0
        assert w.tetrahedra_count.numpy()[0] == 2  # Retain diagnostic subcomplex.


def test_invalid_inputs_zero_output_and_recover_without_reading_inactive_storage():
    d = warpower.SurfaceDomain(*cube(2))
    p = wp.array(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32),
        dtype=wp.vec3,
        device="cuda:0",
    )
    t = wp.array(np.array([[0, 1, 2, 3], [-1, -1, -1, -1]], np.int32), device="cuda:0")
    count = wp.array([1], dtype=int, device="cuda:0")
    error = wp.zeros(1, dtype=int, device="cuda:0")
    w = warpower.ExtractionWorkspace(d, p, t, count, source_error=error)
    for bad_count, expected in [(-1, 2), (3, 2), (2, 4), (0, 0), (1, 0)]:
        count.fill_(bad_count)
        w.run()
        assert w.error.numpy()[0] == expected
        assert w.tetrahedra_count.numpy()[0] == int(bad_count == 1)
    error.fill_(32)
    w.run()
    assert w.error.numpy()[0] == 2 and w.boundary_count.numpy()[0] == 0
    error.zero_()
    w.run()
    extracted(w)
    p.fill_(wp.vec3(float("nan")))
    w.run()
    assert w.error.numpy()[0] == 4 and w.tetrahedra_count.numpy()[0] == 0
    p.zero_()
    w.run()
    assert w.error.numpy()[0] == 8 and w.tetrahedra_count.numpy()[0] == 0


@wp.kernel
def producer(
    base: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int], count: wp.array[int]
):
    i = wp.tid()
    points[i] = base[i] + wp.vec3(0.01 * float(phase[0]), 0.0, 0.0)
    if i == 0:
        count[0] = 1 + phase[0] % 2


@wp.kernel
def consume(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    valid: wp.array[int],
    total: wp.array[wp.float64],
):
    i = wp.tid()
    if i < count[0] and valid[0] != 0:
        a = wp.vec3d(points[tets[i, 0]])
        b = wp.vec3d(points[tets[i, 1]]) - a
        c = wp.vec3d(points[tets[i, 2]]) - a
        d = wp.vec3d(points[tets[i, 3]]) - a
        wp.atomic_add(total, 0, wp.dot(b, wp.cross(c, d)) / wp.float64(6.0))


def test_captured_producer_extract_diagnostics_consumer_uses_only_device_work(
    monkeypatch, tmp_path
):
    d = warpower.SurfaceDomain(*cube(5))
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    p = np.vstack([p, p + np.array([2, 0, 0], np.float32)])
    base = wp.array(p, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(base)
    t = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.int32)
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    count = wp.ones_like(phase)
    total = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    w = warpower.ExtractionWorkspace(d, points, t, count)
    wp.load_module(module=__name__, device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation/readback/sync inside captured operation")

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
        with wp.ScopedCapture() as cap:
            wp.launch(producer, len(points), inputs=[base, points, phase, count])
            w.run()
            total.zero_()
            wp.launch(
                consume, len(t), inputs=[points, w.tetrahedra, w.tetrahedra_count, w.valid, total]
            )
        wp.capture_launch(cap.graph)
    for k in range(3):
        phase.fill_(k)
        wp.capture_launch(cap.graph)
        extracted(w)
        np.testing.assert_allclose(total.numpy()[0], (1 + k % 2) / 6)
        assert w.component_count.numpy()[0] == 1 + k % 2
    dot = tmp_path / "extract.dot"
    wp.capture_debug_dot_print(cap.graph, str(dot), verbose=True)
    text = dot.read_text()
    assert "MEM_ALLOC" not in text and "MEM_FREE" not in text and "HOST" not in text


@pytest.mark.parametrize("tolerance", [-1, 1, np.inf, np.nan])
def test_tolerance_rejects_unsupported_values(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        warpower.ExtractionWorkspace(
            warpower.SurfaceDomain(*cube()),
            np.eye(4, 3, dtype=np.float32),
            np.array([[0, 1, 2, 3]], np.int32),
            tolerance=tolerance,
        )
