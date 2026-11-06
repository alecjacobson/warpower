"""Exact rational reference for the cached-cofactor point rejection filter."""

from fractions import Fraction

import numpy as np
import warp as wp

from tests.test_sphere_bounds import exact_sphere
from warpower import _delaunay_certificate as dc

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def rejected(points: wp.array2d[wp.vec3], query: wp.array[wp.vec3], output: wp.array[int]):
    i = wp.tid()
    shape = dc.sphere(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
    output[i] = int(dc.outside(shape, query[i], query[i]))


def test_point_rejection_proves_strictly_outside_with_exact_rational_spheres():
    rng = np.random.default_rng(892026)
    points, queries, outside = [], [], []
    for exponent in [-140, -126, -60, 0, 60, 120]:
        for i in range(80):
            p = rng.normal(size=(4, 3))
            if i % 4 == 0:
                p[3] = 0.25 * p[0] + 0.25 * p[1] + 0.5 * p[2]
            if i % 5 == 0 and exponent < 120:
                p += 4096
            p = np.ldexp(p, exponent).astype(np.float32)
            determinant, center, radius2 = exact_sphere(p)
            if determinant == 0:
                continue
            if determinant < 0:
                p = p[[0, 2, 1, 3]]
            # Exact ties, both neighboring float32 values, a random point, and
            # an interior convex combination, all using the actual input bits.
            candidates = [
                p[0],
                np.nextafter(p[0], np.float32(np.inf)),
                np.nextafter(p[0], np.float32(-np.inf)),
                np.ldexp(rng.normal(size=3), exponent).astype(np.float32),
                np.mean(p.astype(np.float64), axis=0).astype(np.float32),
            ]
            for q in candidates:
                exact = sum((Fraction(float(x)) - c) ** 2 for x, c in zip(q, center)) > radius2
                points.append(p)
                queries.append(q)
                outside.append(exact)
    p = wp.array(np.array(points), dtype=wp.vec3, device="cuda:0")
    q = wp.array(np.array(queries), dtype=wp.vec3, device=p.device)
    output = wp.empty(len(q), dtype=int, device=p.device)
    wp.launch(rejected, len(q), inputs=[p, q, output])
    result = output.numpy().astype(bool)
    exact = np.array(outside)
    assert np.all(exact[result])  # No false rejection, including near-flat tets.
    assert result.sum() > 100  # The filter must actually prune a useful subset.
    assert not result[::5].any()  # Original vertices are exact cospherical ties.


def test_exact_outside_point_near_sphere_requires_fallback():
    tetrahedron = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    query = np.array([1, 1, -(2.0**-100)], np.float32)
    determinant, center, radius2 = exact_sphere(tetrahedron)
    assert determinant > 0
    assert sum((Fraction(float(x)) - c) ** 2 for x, c in zip(query, center)) > radius2
    p = wp.array(tetrahedron[None], dtype=wp.vec3, device="cuda:0")
    q = wp.array(query[None], dtype=wp.vec3, device=p.device)
    output = wp.empty(1, dtype=int, device=p.device)
    wp.launch(rejected, 1, inputs=[p, q, output])
    # The negative determinant is below the roundoff bound. Do not discard it:
    # the original symbolic/exact predicate must decide this leaf query.
    assert output.numpy()[0] == 0


def test_cospherical_cube_vertices_are_never_rejected_by_rounded_bounds():
    cube = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], np.float32)
    t = cube[[0, 4, 2, 1]]
    assert exact_sphere(t)[0] > 0
    points, queries = [], []
    for exponent in [-140, -126, -60, 0, 60, 120]:
        for shift in [0] if exponent == 120 else [0, 4096]:
            for q in cube:
                points.append(np.ldexp(t + shift, exponent))
                queries.append(np.ldexp(q + shift, exponent))
    p = wp.array(np.array(points), dtype=wp.vec3, device="cuda:0")
    q = wp.array(np.array(queries), dtype=wp.vec3, device=p.device)
    output = wp.empty(len(q), dtype=int, device=p.device)
    wp.launch(rejected, len(q), inputs=[p, q, output])
    np.testing.assert_array_equal(output.numpy(), 0)
