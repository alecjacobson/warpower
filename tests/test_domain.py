import numpy as np
import pytest
import warp as wp
from scipy.spatial import ConvexHull

import warpower
from warpower._domain import excess_contact
from warpower._surface_bvh import SurfaceTree, material_side, ray_count


def convex_surface(points):
    points = np.asarray(points, np.float32)
    faces = ConvexHull(points).simplices.astype(np.int32)
    center = points.mean(axis=0)
    for face in faces:
        a, b, c = points[face]
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3 - center) < 0:
            face[0], face[1] = face[1], face[0]
    return points, faces


def cube(scale=1.0, center=(0, 0, 0)):
    return convex_surface(
        np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * scale + center
    )


def torus(nu=48, nv=24, major=2.0, minor=0.7):
    u, v = np.meshgrid(
        np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing="ij"
    )
    points = np.stack(
        (
            (major + minor * np.cos(v)) * np.cos(u),
            (major + minor * np.cos(v)) * np.sin(u),
            minor * np.sin(v),
        ),
        axis=-1,
    ).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    return points.astype(np.float32), np.array(faces, np.int32)


def combine(*surfaces):
    points, faces, offset = [], [], 0
    for p, f in surfaces:
        points.append(p)
        faces.append(f + offset)
        offset += len(p)
    return np.concatenate(points), np.concatenate(faces)


def solid_angle_inside(points, faces, queries):
    """Independent all-triangle double-precision winding, with no hierarchy."""
    triangles = points[faces].astype(np.float64)
    values = []
    for q in queries:
        a, b, c = np.moveaxis(triangles - q, 1, 0)
        la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
        denominator = (
            la * lb * lc
            + np.sum(a * b, axis=1) * lc
            + np.sum(b * c, axis=1) * la
            + np.sum(c * a, axis=1) * lb
        )
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        values.append(np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi) > 0.5)
    return np.array(values)


def brute_distances(points, faces, queries):
    """Plane projections plus three segment minima, independent of Warp queries."""
    triangles = points[faces].astype(np.float64)
    a, b, c = np.moveaxis(triangles, 1, 0)
    ab, ac = b - a, c - a
    aa, bb, cc = np.sum(ab * ab, axis=1), np.sum(ab * ac, axis=1), np.sum(ac * ac, axis=1)
    distances = []
    for q in queries.astype(np.float64):
        aq = q - a
        d, e = np.sum(aq * ab, axis=1), np.sum(aq * ac, axis=1)
        u, v = (cc * d - bb * e) / (aa * cc - bb * bb), (aa * e - bb * d) / (aa * cc - bb * bb)
        projection = a + u[:, None] * ab + v[:, None] * ac
        ds = np.sum((q - projection) ** 2, axis=1)
        ds[(u < 0) | (v < 0) | (u + v > 1)] = np.inf
        for p, r in ((a, b), (b, c), (c, a)):
            edge = r - p
            t = np.clip(np.sum((q - p) * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)
            ds = np.minimum(ds, np.sum((q - p - t[:, None] * edge) ** 2, axis=1))
        distances.append(np.sqrt(ds.min()))
    return np.array(distances)


@pytest.mark.parametrize("subdivisions", [1, 2, 5])
def test_cube_quadrature_integrates_area_moments_and_creases(subdivisions):
    domain = warpower.SurfaceDomain(*cube(), surface_subdivisions=subdivisions)
    assert domain.error.numpy()[0] == 0
    p, w = domain.surface_points.numpy().astype(np.float64), domain.surface_weights.numpy()
    assert np.isclose(w.sum(), 24, rtol=1e-14)
    np.testing.assert_allclose(w @ p, 0, atol=2e-6)
    # Integral x^2 dA = two end faces (8) + the four side faces (16/3).
    np.testing.assert_allclose(w @ (p * p), np.full(3, 40 / 3), rtol=1e-7)
    assert domain.feature_count.numpy()[0] == 12
    assert domain.corner_count.numpy()[0] == 8
    np.testing.assert_array_equal(domain.corner_mask.numpy(), 1)
    p, w = domain.crease_points.numpy().astype(np.float64), domain.crease_weights.numpy()
    assert np.isclose(w.sum(), 24, rtol=1e-14)
    np.testing.assert_allclose(w @ p, 0, atol=1e-6)
    np.testing.assert_allclose(w @ (p * p), np.full(3, 56 / 3), rtol=1e-7)
    np.testing.assert_array_equal(domain.corner_points.numpy()[:8], cube()[0])


def test_nonconvex_torus_queries_match_analytic_and_independent_triangle_oracles():
    p, f = torus()
    q = np.random.default_rng(81).uniform(-3, 3, (240, 3)).astype(np.float32)
    analytic = np.sqrt((np.linalg.norm(q[:, :2], axis=1) - 2) ** 2 + q[:, 2] ** 2) - 0.7
    domain = warpower.SurfaceDomain(p, f)
    assert domain.error.numpy()[0] == 0
    query = warpower.DomainQueryWorkspace(domain, q).run()
    assert query.error.numpy()[0] == 0
    assert np.all(query.valid.numpy())
    actual = query.signed_distance.numpy()
    np.testing.assert_array_equal(query.inside.numpy(), solid_angle_inside(p, f, q))
    np.testing.assert_allclose(np.abs(actual), brute_distances(p, f, q), rtol=3e-6, atol=5e-7)
    np.testing.assert_allclose(actual, analytic, atol=0.015)
    away = np.abs(analytic) > 0.02
    np.testing.assert_array_equal(query.inside.numpy()[away], analytic[away] < 0)
    assert domain.feature_count.numpy()[0] == 0
    assert domain.corner_count.numpy()[0] == 0


def test_cavity_and_disconnected_components_keep_correct_sign():
    outer, inner = cube(), cube(0.4)
    inner = inner[0], inner[1][:, ::-1].copy()
    p, f = combine(outer, inner, cube(0.5, (4, 0, 0)))
    domain = warpower.SurfaceDomain(p, f)
    q = np.array([[0, 0, 0], [0.7, 0, 0], [4, 0, 0], [2, 0, 0], [5, 0, 0]], np.float32)
    query = warpower.DomainQueryWorkspace(domain, q).run()
    assert domain.error.numpy()[0] == 0
    np.testing.assert_array_equal(query.inside.numpy(), [0, 1, 1, 0, 0])
    np.testing.assert_allclose(query.signed_distance.numpy(), [0.4, -0.3, -0.5, 1, 0.5], atol=2e-7)
    np.testing.assert_array_equal(query.inside.numpy(), solid_angle_inside(p, f, q))


def test_sphere_queries_converge_to_analytic_surface():
    rng = np.random.default_rng(400)
    points = rng.normal(size=(1000, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    p, f = convex_surface(points)
    q = rng.uniform(-1.3, 1.3, (100, 3)).astype(np.float32)
    domain = warpower.SurfaceDomain(p, f, feature_angle=80)
    query = warpower.DomainQueryWorkspace(domain, q).run()
    assert domain.error.numpy()[0] == 0
    np.testing.assert_array_equal(query.inside.numpy(), solid_angle_inside(p, f, q))
    np.testing.assert_allclose(
        np.abs(query.signed_distance.numpy()), brute_distances(p, f, q), rtol=4e-6, atol=6e-7
    )
    np.testing.assert_allclose(
        query.signed_distance.numpy(), np.linalg.norm(q, axis=1) - 1, atol=0.025
    )


@wp.kernel
def producer(source: wp.array[wp.vec3], target: wp.array[wp.vec3], scale: wp.array[float]):
    i = wp.tid()
    target[i] = source[i] * scale[0]


@wp.kernel
def consumer(distance: wp.array[float], inside: wp.array[int], value: wp.array[float]):
    i = wp.tid()
    value[i] = distance[i] + float(inside[i]) * 10.0


def test_capture_refit_query_and_consumer_with_changing_input(monkeypatch, tmp_path):
    p, f = cube()
    source = wp.array(p, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(source)
    domain = warpower.SurfaceDomain(points, f, surface_subdivisions=2)
    query = warpower.DomainQueryWorkspace(domain, np.array([[0, 0, 0], [3, 0, 0]], np.float32))
    scale = wp.ones(1, dtype=float, device="cuda:0")
    result = wp.empty(2, dtype=float, device="cuda:0")
    wp.load_module(module=__name__, device="cuda:0")

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
            wp.launch(producer, len(p), inputs=[source, points, scale])
            domain.refit()
            query.run()
            wp.launch(consumer, 2, inputs=[query.signed_distance, query.inside, result])
        wp.capture_launch(capture.graph)
    dot = tmp_path / "domain.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    graph_text = dot.read_text()
    assert "MEM_ALLOC" not in graph_text and "MEM_FREE" not in graph_text
    assert "HOST" not in graph_text and "to HOST" not in graph_text
    assert domain.error.numpy()[0] == 0
    np.testing.assert_allclose(result.numpy(), [9, 2], atol=1e-6)
    pointers = (domain.surface_points.ptr, domain.crease_points.ptr, query.closest.ptr)
    scale.fill_(2)
    wp.capture_launch(capture.graph)
    assert domain.error.numpy()[0] == 0
    np.testing.assert_allclose(result.numpy(), [8, 1], atol=1e-6)
    assert np.isclose(domain.surface_weights.numpy().sum(), 96)
    assert np.isclose(domain.crease_weights.numpy().sum(), 48)
    assert pointers == (domain.surface_points.ptr, domain.crease_points.ptr, query.closest.ptr)


@pytest.mark.parametrize(
    "bad", ["index", "nonfinite", "collapsed", "open", "reversed", "intersection", "bowtie"]
)
def test_invalid_domains_are_reported_on_device_and_queries_are_suppressed(bad):
    p, f = cube()
    bit = {
        "index": 1,
        "nonfinite": 2,
        "collapsed": 4,
        "open": 8,
        "reversed": 32,
        "intersection": 16,
        "bowtie": 8,
    }[bad]
    if bad == "index":
        f[0, 0] = 10000
    elif bad == "nonfinite":
        p[0, 0] = np.nan
    elif bad == "collapsed":
        p[f[0, 1]] = p[f[0, 0]]
    elif bad == "open":
        f = f[1:]
    elif bad == "reversed":
        f = f[:, ::-1].copy()
    elif bad == "intersection":
        p, f = combine(cube(), cube(1, (1.1, 0.2, 0.3)))
    elif bad == "bowtie":
        a, b = convex_surface([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        c, d = convex_surface([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        p = np.concatenate((a, c[1:]))
        f = np.concatenate((b, np.where(d == 0, 0, d + 3))).astype(np.int32)
    domain = warpower.SurfaceDomain(p, f)
    assert domain.error.numpy()[0] & bit
    query = warpower.DomainQueryWorkspace(domain, np.zeros((2, 3), np.float32)).run()
    assert query.error.numpy()[0] & bit
    assert not np.any(query.valid.numpy())
    assert not np.any(domain.surface_weights.numpy())
    assert domain.feature_count.numpy()[0] == 0
    assert domain.corner_count.numpy()[0] == 0


def test_query_distance_limit_nonfinite_and_recovery_after_bad_refit():
    p, f = cube()
    points = wp.array(p, dtype=wp.vec3, device="cuda:0")
    domain = warpower.SurfaceDomain(points, f)
    query = warpower.DomainQueryWorkspace(
        domain, np.array([[3, 0, 0], [1.1, 0, 0], [np.nan, 0, 0]], np.float32), max_distance=0.5
    ).run()
    np.testing.assert_array_equal(query.valid.numpy(), [0, 1, 0])
    assert query.error.numpy()[0] == 64
    points.fill_(wp.vec3(0.0))
    domain.refit()
    assert domain.error.numpy()[0] & 4
    wp.copy(points, wp.array(p, dtype=wp.vec3, device="cuda:0"))
    domain.refit()
    assert domain.error.numpy()[0] == 0
    assert np.isclose(domain.surface_weights.numpy().sum(), 24)


@wp.kernel
def contacts(points: wp.array[wp.vec3], pairs: wp.array2d[int], result: wp.array[int]):
    i = wp.tid()
    result[i] = int(
        excess_contact(
            points,
            wp.vec3i(pairs[i, 0], pairs[i, 1], pairs[i, 2]),
            wp.vec3i(pairs[i, 3], pairs[i, 4], pairs[i, 5]),
        )
    )


def test_shared_vertex_and_edge_contact_distinguishes_legal_adjacency_from_overlap():
    points = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [0, -2, 0],
            [1, 1, 0],
            [-2, 0, 0],
            [1, 1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        np.float32,
    )
    pairs = np.array(
        [
            [0, 1, 2, 0, 1, 3],  # shared edge, opposite sides in a plane
            [0, 1, 2, 0, 1, 4],  # shared edge, overlapping area in a plane
            [0, 1, 2, 0, 1, 6],  # legal folded edge
            [0, 1, 2, 0, 3, 5],  # shared vertex, disjoint coplanar wedges
            [0, 1, 2, 0, 4, 5],  # shared vertex, overlapping coplanar wedges
            [0, 1, 2, 0, 6, 7],  # shared vertex, noncoplanar overlap along positive diagonal
            [0, 1, 2, 0, 8, 9],  # shared vertex, noncoplanar contact only at origin
            [0, 1, 2, 2, 1, 0],  # duplicate triangle
        ],
        np.int32,
    )
    expected = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    # Vary winding, pair order, and translation without changing contact.
    for scale, shift in [(1, 0), (0.01, 0), (2, 17)]:
        p = wp.array(points * scale + shift, dtype=wp.vec3, device="cuda:0")
        variants = np.concatenate(
            [pairs, pairs[:, [2, 1, 0, 5, 4, 3]], pairs[:, [3, 4, 5, 0, 1, 2]]]
        )
        rows = wp.array(variants, dtype=int, device="cuda:0")
        result = wp.empty(len(rows), dtype=int, device="cuda:0")
        wp.launch(contacts, len(rows), inputs=[p, rows, result])
        np.testing.assert_array_equal(result.numpy(), np.tile(expected, 3))


def test_quadrature_and_feature_order_are_reproducible_and_query_inputs_are_zero_copy():
    p, f = cube()
    points = wp.array(np.array([[0, 0, 0], [4, 0, 0]], np.float32), dtype=wp.vec3, device="cuda:0")
    first = warpower.SurfaceDomain(p, f, surface_subdivisions=3)
    second = warpower.SurfaceDomain(p, f, surface_subdivisions=3)
    for name in (
        "surface_points",
        "surface_weights",
        "crease_points",
        "crease_weights",
        "corner_mask",
    ):
        np.testing.assert_array_equal(getattr(first, name).numpy(), getattr(second, name).numpy())
    np.testing.assert_array_equal(
        first.feature_edges.numpy()[:12], second.feature_edges.numpy()[:12]
    )
    query = warpower.DomainQueryWorkspace(first, points)
    assert query.points.ptr == points.ptr
    query.run()
    np.testing.assert_array_equal(query.inside.numpy(), [1, 0])
    points.fill_(wp.vec3(0.0))
    query.run()
    np.testing.assert_array_equal(query.inside.numpy(), [1, 1])


def test_thin_faces_skipped_by_native_mesh_queries_remain_present():
    p = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-6, 0], [0, 0, 1]], np.float32)
    p, f = convex_surface(p)
    domain = warpower.SurfaceDomain(p, f, validate_geometry=False)
    assert domain.error.numpy()[0] == 0
    q = np.array([[0.5, 1e-7, 1e-8]], np.float32)
    query = warpower.DomainQueryWorkspace(domain, q).run()
    assert query.error.numpy()[0] == 0
    assert query.valid.numpy()[0] == 1
    np.testing.assert_allclose(
        abs(query.signed_distance.numpy()), brute_distances(p, f, q), rtol=1e-6, atol=1e-12
    )


@pytest.mark.parametrize("corner_angle, expected", [(45, 0), (20, 24)])
def test_two_crease_edges_classify_corners_by_turn_angle(corner_angle, expected):
    angles = np.arange(12) * 2 * np.pi / 12
    points = np.array([[np.cos(a), np.sin(a), z] for z in (-1, 1) for a in angles], np.float32)
    domain = warpower.SurfaceDomain(*convex_surface(points), corner_angle=corner_angle)
    assert domain.error.numpy()[0] == 0
    assert domain.feature_count.numpy()[0] == 24
    assert domain.corner_count.numpy()[0] == expected


@wp.kernel
def ray_ties(tree: SurfaceTree, output: wp.array[wp.vec2i]):
    i = wp.tid()
    direction = wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319)
    origin = wp.vec3d(1.0) - wp.float64(2 + 2 * i) * direction
    first = ray_count(tree, origin, direction)
    output[i] = wp.vec2i(first[1], material_side(tree, origin))


def test_oriented_ray_retries_shared_corner_hits_on_device():
    domain = warpower.SurfaceDomain(*cube())
    result = wp.empty(2, dtype=wp.vec2i, device="cuda:0")
    wp.launch(ray_ties, 2, inputs=[domain.tree, result])
    np.testing.assert_array_equal(result.numpy(), [[1, 1], [1, 0]])


def test_orientation_does_not_step_across_a_thin_domain():
    p, f = cube()
    p[:, 2] *= 1e-7
    domain = warpower.SurfaceDomain(p, f)
    assert domain.error.numpy()[0] == 0
    q = np.array([[0, 0, 0], [0, 0, 2e-7]], np.float32)
    query = warpower.DomainQueryWorkspace(domain, q).run()
    assert query.error.numpy()[0] == 0
    np.testing.assert_array_equal(query.inside.numpy(), [1, 0])
    np.testing.assert_allclose(query.signed_distance.numpy(), [-1e-7, 1e-7], rtol=1e-6)
