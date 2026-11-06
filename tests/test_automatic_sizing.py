import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import convex_surface, cube, torus
from warpower._automatic_sizing import AutomaticSizingWorkspace


def sphere(n=64, radius=1.0):
    i = np.arange(n)
    z = 1 - 2 * (i + 0.5) / n
    phi = i * (np.pi * (3 - np.sqrt(5)))
    r = np.sqrt(1 - z * z)
    return convex_surface(radius * np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1))


@pytest.mark.parametrize("subdivisions", [0, 1, 2])
def test_cube_controls_publish_valid_poles_and_nearest_distances(subdivisions):
    domain = warpower.SurfaceDomain(*cube())
    auto = AutomaticSizingWorkspace(domain, control_subdivisions=subdivisions)
    assert auto.error.numpy()[0] == 0, (
        auto.error.numpy(),
        auto.mesh.error.numpy(),
        auto.mesh.certificate_error.numpy(),
    )
    assert auto.field.error.numpy()[0] == 0
    p = auto.control_points.numpy().astype(float)
    poles = auto.poles.numpy()[auto.pole_valid.numpy() != 0]
    assert len(poles) == auto.pole_count.numpy()[0] > 0
    distances = np.linalg.norm(p[:, None] - poles[None], axis=2).min(axis=1)
    np.testing.assert_allclose(auto.local_feature_size.numpy(), distances, rtol=1e-13)
    assert np.all(distances > 0)
    assert np.all(auto.pole_valid.numpy()[::2] == 0)  # Every sample is on the convex hull.


def test_sphere_poles_capture_curvature_radius():
    domain = warpower.SurfaceDomain(*sphere(64, radius=2.0))
    auto = AutomaticSizingWorkspace(domain, control_subdivisions=0)
    assert auto.error.numpy()[0] == 0, auto.mesh.error.numpy()
    np.testing.assert_allclose(auto.local_feature_size.numpy(), 2.0, rtol=2e-4)
    assert np.all(auto.pole_valid.numpy()[::2] == 0)


def test_torus_has_both_finite_and_unbounded_first_poles():
    domain = warpower.SurfaceDomain(*torus(16, 8))
    auto = AutomaticSizingWorkspace(domain, control_subdivisions=0)
    assert auto.error.numpy()[0] == 0, auto.mesh.error.numpy()
    first = auto.pole_valid.numpy()[::2]
    assert np.any(first == 0) and np.any(first == 1)
    lfs = auto.local_feature_size.numpy()
    assert np.median(lfs) > 0.4
    assert np.median(lfs) < 0.9


@pytest.mark.parametrize("shape", ["cube", "torus"])
def test_pole_choices_do_not_depend_on_tetrahedron_output_row_order(shape):
    import warp as wp

    from warpower._mesh_topology import identity, index_faces

    surface = cube() if shape == "cube" else torus(16, 8)
    auto = AutomaticSizingWorkspace(warpower.SurfaceDomain(*surface), control_subdivisions=0)
    assert auto.error.numpy()[0] == 0
    mesh, cert = auto.mesh, auto.mesh._certificate
    original = mesh.tetrahedra.numpy()
    count = int(mesh.tetrahedra_count.numpy()[0])
    expected = [
        a.numpy() for a in (auto._normals, auto.poles, auto.pole_valid, auto.local_feature_size)
    ]
    with wp.ScopedCapture() as capture:
        cert._table.fill_(-1)
        cert._occurrences.zero_()
        cert._balance.zero_()
        wp.launch(identity, len(cert._parents), inputs=[cert._parents])
        wp.launch(
            index_faces,
            4 * len(mesh.tetrahedra),
            inputs=[
                mesh.tetrahedra,
                mesh.tetrahedra_count,
                cert._table,
                cert._occurrences,
                cert._balance,
                cert._parents,
            ],
        )
        auto.error.zero_()
        auto.pole_count.zero_()
        auto.pole_valid.zero_()
        auto._extract()
        auto._distances()
    for order in [np.arange(count)[::-1], np.random.default_rng(501).permutation(count)]:
        shuffled = original.copy()
        shuffled[:count] = original[order]
        mesh.tetrahedra.assign(shuffled)
        wp.capture_launch(capture.graph)
        assert auto.error.numpy()[0] == 0
        for wanted, actual in zip(
            expected, (auto._normals, auto.poles, auto.pole_valid, auto.local_feature_size)
        ):
            np.testing.assert_array_equal(actual.numpy(), wanted)


@pytest.mark.parametrize("shape", ["cube", "sphere", "torus"])
def test_poles_match_independent_circumcenter_and_incident_star_checks(shape):
    if shape == "cube":
        surface, subdivisions = cube(), 1
    elif shape == "sphere":
        surface, subdivisions = sphere(40, radius=1.5), 0
    else:
        surface, subdivisions = torus(16, 8), 0
    auto = AutomaticSizingWorkspace(
        warpower.SurfaceDomain(*surface), control_subdivisions=subdivisions
    )
    assert auto.error.numpy()[0] == 0
    n_tets = int(auto.mesh.tetrahedra_count.numpy()[0])
    tets = auto.mesh.tetrahedra.numpy()[:n_tets]
    p = auto.control_points.numpy().astype(float)
    cells = p[tets]
    edges = cells[:, 1:] - cells[:, :1]
    offset = np.linalg.solve(2 * edges, np.sum(edges * edges, axis=2)[..., None])[..., 0]
    reference_centers = cells[:, 0] + offset
    centers = auto._centers.numpy()[:n_tets]
    condition = np.linalg.cond(edges)
    tolerance = (
        64
        * np.finfo(float).eps
        * np.maximum(condition, 1)
        * np.maximum(np.linalg.norm(reference_centers, axis=1), 1)
    )
    assert np.all(np.linalg.norm(centers - reference_centers, axis=1) <= tolerance)
    poles = auto.poles.numpy().reshape(-1, 2, 3)
    valid = auto.pole_valid.numpy().reshape(-1, 2)
    source = auto.pole_source_tets.numpy().reshape(-1, 2)
    normals = auto._normals.numpy()
    hull = auto.hull_face.numpy() != 2**31 - 1
    for i in range(len(p)):
        incident = np.flatnonzero(np.any(tets == i, axis=1))
        if not len(incident):
            continue  # Duplicate controls need not own a Voronoi cell.
        delta = centers[incident] - p[i]
        d2 = np.sum(delta * delta, axis=1)
        assert bool(valid[i, 0]) == (not hull[i])
        if valid[i, 0]:
            assert source[i, 0] in incident
            np.testing.assert_allclose(np.sum((poles[i, 0] - p[i]) ** 2), d2.max(), rtol=1e-12)
        projection = delta @ normals[i]
        margin = 1e-12 * np.maximum(np.linalg.norm(delta, axis=1) * np.linalg.norm(normals[i]), 1)
        definitely_opposite = projection < -margin
        if np.any(definitely_opposite):
            assert valid[i, 1] != 0
        if valid[i, 1]:
            assert source[i, 1] in incident
            selected = int(np.flatnonzero(incident == source[i, 1])[0])
            assert projection[selected] < margin[selected]
            if np.any(definitely_opposite):
                assert d2[selected] >= d2[definitely_opposite].max() * (1 - 1e-12)
        for side in [0, 1]:
            if valid[i, side]:
                np.testing.assert_array_equal(poles[i, side], centers[source[i, side]])
    nearest = np.linalg.norm(
        p[:, None] - poles.reshape(-1, 3)[valid.ravel() != 0][None], axis=2
    ).min(axis=1)
    np.testing.assert_allclose(auto.local_feature_size.numpy(), nearest, rtol=1e-13)


def test_boundary_enrichment_resolves_thin_plate_thickness():
    p, f = cube()
    p[:, 2] *= 0.25
    domain = warpower.SurfaceDomain(p, f)
    sparse = AutomaticSizingWorkspace(domain, control_subdivisions=0)
    dense = AutomaticSizingWorkspace(domain, control_subdivisions=1)
    assert sparse.error.numpy()[0] == dense.error.numpy()[0] == 0
    sparse_median = np.median(sparse.local_feature_size.numpy())
    dense_median = np.median(dense.local_feature_size.numpy())
    assert sparse_median > 1.0
    assert 0.25 < dense_median < 0.35
    assert dense_median < sparse_median / 3


@pytest.mark.parametrize("scale", [2.0**-20, 2.0**20])
def test_automatic_sizes_scale_with_geometry(scale):
    p, f = cube()
    original = AutomaticSizingWorkspace(warpower.SurfaceDomain(p, f), control_subdivisions=1)
    transformed = ((p.astype(float) + [2.5, -1.75, 0.125]) * scale).astype(np.float32)
    changed = AutomaticSizingWorkspace(
        warpower.SurfaceDomain(transformed, f),
        control_subdivisions=1,
        minimum=scale * 1e-4,
        maximum=scale * 10,
    )
    assert original.error.numpy()[0] == changed.error.numpy()[0] == 0
    np.testing.assert_allclose(
        changed.local_feature_size.numpy() / scale, original.local_feature_size.numpy(), rtol=2e-6
    )


@wp.kernel
def deform_domain(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    phase: wp.array[int],
):
    i = wp.tid()
    factor = 1.0 + float(phase[0])
    shift = wp.vec3(0.25 * float(phase[0]), -0.5 * float(phase[0]), 0.0)
    p = source[i] * factor + shift
    if phase[0] < 0 and i == 0:
        p[0] = wp.nan
    points[i] = p
    if i < queries.shape[0]:
        queries[i] = source[i] * (0.5 * factor) + shift


@wp.kernel
def field_consumer(values: wp.array[float], error: wp.array[int], output: wp.array[wp.float64]):
    i = wp.tid()
    if error[0] == 0:
        wp.atomic_add(output, 0, wp.float64(values[i]))


def test_captured_domain_refit_automatic_field_and_consumer_recover_from_invalid_input(tmp_path):
    original, faces = cube()
    domain = warpower.SurfaceDomain(original, faces)
    auto = AutomaticSizingWorkspace(domain, control_subdivisions=0, gradation=0.5)
    source = wp.array(original, dtype=wp.vec3, device=domain.device)
    queries = wp.empty(4, dtype=wp.vec3, device=domain.device)
    query = warpower.SizingQueryWorkspace(auto.field, queries)
    phase = wp.zeros(1, dtype=int, device=domain.device)
    total = wp.zeros(1, dtype=wp.float64, device=domain.device)
    wp.load_module(module=__name__, device=domain.device)
    arrays = [
        auto.control_points,
        auto.poles,
        auto.local_feature_size,
        auto.field.values,
        query.values,
    ]
    pointers = [a.ptr for a in arrays]
    with wp.ScopedCapture() as capture:
        wp.launch(deform_domain, len(source), inputs=[source, domain.points, queries, phase])
        domain.refit()
        auto.refit()
        query.run()
        total.zero_()
        wp.launch(field_consumer, len(queries), inputs=[query.values, query.error, total])
    for step in [0, 1, -1, 2, 0]:
        phase.fill_(step)
        wp.capture_launch(capture.graph)
        assert [a.ptr for a in arrays] == pointers
        if step < 0:
            assert domain.error.numpy()[0] != 0
            assert auto.error.numpy()[0] == 1
            assert auto.pole_count.numpy()[0] == 0
            assert auto.field.error.numpy()[0] != 0
            assert query.error.numpy()[0] != 0
            np.testing.assert_array_equal(query.values.numpy(), 0)
            assert total.numpy()[0] == 0
        else:
            assert domain.error.numpy()[0] == auto.error.numpy()[0] == query.error.numpy()[0] == 0
            assert auto.pole_count.numpy()[0] > 0
            p = auto.control_points.numpy().astype(float)
            q = queries.numpy().astype(float)
            sizes = auto.field.values.numpy().astype(float)
            expected = np.min(
                sizes[None] + 0.5 * np.linalg.norm(q[:, None] - p[None], axis=2), axis=1
            )
            np.testing.assert_allclose(query.values.numpy(), expected, rtol=8e-8)
            np.testing.assert_allclose(
                total.numpy()[0], query.values.numpy().astype(float).sum(), rtol=1e-15
            )
    path = tmp_path / "automatic.dot"
    wp.capture_debug_dot_print(capture.graph, str(path))
    dot = path.read_text().lower()
    assert "host" not in dot and "mem_alloc" not in dot and "mem_free" not in dot


def test_failed_control_mesh_invalidates_the_field():
    domain = warpower.SurfaceDomain(*cube())
    auto = AutomaticSizingWorkspace(domain, control_subdivisions=0, tetrahedra_capacity=1)
    assert auto.error.numpy()[0] == 2
    assert auto.pole_count.numpy()[0] == 0
    assert auto.field.error.numpy()[0] != 0
    np.testing.assert_array_equal(auto.pole_valid.numpy(), 0)
    np.testing.assert_array_equal(auto.local_feature_size.numpy(), 0)
    query = warpower.SizingQueryWorkspace(auto.field, domain.points).run()
    assert query.error.numpy()[0] != 0
    np.testing.assert_array_equal(query.values.numpy(), 0)
