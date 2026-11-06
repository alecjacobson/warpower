from fractions import Fraction

import numpy as np
import warp as wp

from warpower import _delaunay_certificate as dc
from warpower import _sphere_bounds as sb

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


def det(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def exact_sphere(points):
    rows = [[Fraction(float(x)) for x in row] for row in points]
    a = rows[0]
    edges = [[x - y for x, y in zip(row, a)] for row in rows[1:]]
    determinant = det(*edges)
    if determinant == 0:
        return determinant, None, None
    norms = [sum(x * x for x in row) for row in edges]
    offset = []
    for axis in range(3):
        matrix = [list(row) for row in edges]
        for i in range(3):
            matrix[i][axis] = norms[i]
        offset.append(det(*matrix) / (2 * determinant))
    return determinant, [x + y for x, y in zip(a, offset)], sum(x * x for x in offset)


@wp.kernel
def boxes(
    points: wp.array[wp.vec3],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    valid: wp.array[int],
):
    i = wp.tid()
    shape = dc.sphere(points[4 * i], points[4 * i + 1], points[4 * i + 2], points[4 * i + 3])
    box = sb.bounds(shape.origin, shape.nx, shape.ny, shape.nz, shape.determinant)
    lower[i] = box.lower
    upper[i] = box.upper
    valid[i] = box.valid


def evaluate(points):
    p = wp.array(np.asarray(points, np.float32).reshape(-1, 3), dtype=wp.vec3, device="cuda:0")
    n = len(p) // 4
    lo, hi = (
        wp.empty(n, dtype=wp.vec3, device=p.device),
        wp.empty(n, dtype=wp.vec3, device=p.device),
    )
    valid = wp.empty(n, dtype=int, device=p.device)
    wp.launch(boxes, n, inputs=[p, lo, hi, valid])
    return lo.numpy(), hi.numpy(), valid.numpy()


def test_wide_range_near_flat_and_translated_spheres_fit_boxes_exactly():
    rng = np.random.default_rng(3058)
    points, oracles = [], []
    with np.errstate(over="ignore", under="ignore"):
        for exponent in [-140, -126, -100, -60, -20, 0, 20, 60, 100, 125]:
            for i in range(100):
                p = rng.normal(size=(4, 3))
                if i % 3 == 0:
                    p[3] = 0.3 * p[0] + 0.3 * p[1] + 0.4 * p[2] + 2.0**-20 * rng.normal(size=3)
                if i % 7 == 0:
                    p += rng.normal(size=3) * 1e5
                p = (p * 2.0**exponent).astype(np.float32)
                if not np.isfinite(p).all():
                    continue
                determinant, center, radius2 = exact_sphere(p)
                if determinant < 0:
                    p = p[[0, 2, 1, 3]]
                points.append(p)
                oracles.append((determinant != 0, center, radius2))
    lower, upper, valid = evaluate(points)
    assert valid.sum() > 0.8 * len(valid)
    assert (valid == 0).any()  # Extreme/uncertain cases retain the legacy traversal.
    for lo, hi, good, (nonzero, center, radius2) in zip(lower, upper, valid, oracles):
        if not good:
            continue
        assert nonzero
        for axis in range(3):
            a = center[axis] - Fraction(float(lo[axis]))
            b = Fraction(float(hi[axis])) - center[axis]
            assert a >= 0 and b >= 0 and a * a >= radius2 and b * b >= radius2


def test_coplanar_coincident_negative_and_nonfinite_inputs_use_fallback():
    positive = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    flat = positive.copy()
    flat[3] = [1, 1, 0]
    nonfinite = positive.copy()
    nonfinite[3, 2] = np.inf
    _, _, valid = evaluate([flat, np.zeros((4, 3)), positive[[0, 2, 1, 3]], nonfinite])
    np.testing.assert_array_equal(valid, 0)
