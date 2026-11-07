"""Rational linear solves check ray classification across float32 scales."""

from fractions import Fraction

import numpy as np
import pytest
import warp as wp

from tests.test_containment import cavity, query
from warpower._surface_bvh import SurfaceTree


def rational_ray(vertices, faces, point, direction):
    """Solve p+t*d = a+u*(b-a)+v*(c-a) by Fraction Gaussian elimination."""
    p = list(map(lambda x: Fraction(float(x)), point))
    d = list(map(Fraction, direction))
    total = 0
    for face in faces:
        a, b, c = [[Fraction(float(x)) for x in vertices[k]] for k in face]
        rows = [[d[i], a[i] - b[i], a[i] - c[i], a[i] - p[i]] for i in range(3)]
        for j in range(3):
            pivot = next(k for k in range(j, 3) if rows[k][j] != 0)
            rows[j], rows[pivot] = rows[pivot], rows[j]
            value = rows[j][j]
            rows[j] = [x / value for x in rows[j]]
            for k in range(3):
                if k != j:
                    value = rows[k][j]
                    rows[k] = [x - value * y for x, y in zip(rows[k], rows[j])]
        t, u, v = [row[3] for row in rows]
        if t > 0 and u >= 0 and v >= 0 and u + v <= 1:
            if u == 0 or v == 0 or u + v == 1:
                return 0, 1
            ab, ac = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
            normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ]
            dot = sum(x * y for x, y in zip(normal, d))
            total += 1 if dot > 0 else -1
    return total, 0


@pytest.mark.parametrize("exponent", [-100, 0, 100])
def test_ray_membership_matches_fraction_solves_across_extreme_scales(exponent):
    vertices, faces = cavity()
    scale = np.float32(2.0**exponent)
    vertices *= scale
    points = (
        np.array([[0, 0, 0], [0.6, 0.6, 0.6], [2, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]], np.float32)
        * scale
    )
    # Only point/index data is read by this exact query; avoid unrelated domain
    # normal/feature calculations at deliberately extreme test scales.
    tree = SurfaceTree()
    tree.points = wp.array(vertices, dtype=wp.vec3, device="cuda:0")
    tree.indices = wp.array(faces.ravel(), dtype=int, device="cuda:0")
    tree.n = len(faces)
    probes = wp.array(points, dtype=wp.vec3, device="cuda:0")
    result, ambiguity = (
        wp.empty(5, dtype=int, device="cuda:0"),
        wp.empty(5, dtype=int, device="cuda:0"),
    )
    wp.launch(query, 5, inputs=[tree, probes, result, ambiguity])
    references = [rational_ray(vertices, faces, p, [11, 23, 37]) for p in points]
    assert all(flag == 0 for _, flag in references)
    np.testing.assert_array_equal(result.numpy(), [count for count, _ in references])
    np.testing.assert_array_equal(result.numpy(), [0, 1, 0, 1, 1])
    corner_flags = [rational_ray(vertices, faces, p, [1, 1, 1])[1] for p in points]
    np.testing.assert_array_equal(ambiguity.numpy(), corner_flags)
