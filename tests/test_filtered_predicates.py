import numpy as np
import warp as wp

from tests.test_predicates import exact_reference
from warpower import _filtered_predicates as fp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def evaluate(
    points: wp.array2d[wp.vec3],
    weights: wp.array2d[float],
    signs: wp.array2d[int],
    valid: wp.array2d[int],
    bounds: wp.array2d[wp.float64],
    fast: wp.array2d[int],
):
    i = wp.tid()
    a, b, c, d, q = points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4]
    o = fp.orient3d(a, b, c, d)
    s = fp.power_sphere(
        a, b, c, d, q, weights[i, 0], weights[i, 1], weights[i, 2], weights[i, 3], weights[i, 4]
    )
    ob = fp.orientation_ball(a, b, c, d)
    sb = fp.sphere_ball(
        a, b, c, d, q, weights[i, 0], weights[i, 1], weights[i, 2], weights[i, 3], weights[i, 4]
    )
    signs[i, 0] = o.sign
    signs[i, 1] = s.sign
    valid[i, 0] = o.valid
    valid[i, 1] = s.valid
    bounds[i, 0] = ob.value
    bounds[i, 1] = ob.error
    bounds[i, 2] = sb.value
    bounds[i, 3] = sb.error
    fast[i, 0] = int(fp.certain(ob))
    fast[i, 1] = int(fp.certain(sb))


def check(points, weights=None):
    from fractions import Fraction

    p = np.asarray(points, np.float32)
    w = np.zeros(p.shape[:2], np.float32) if weights is None else np.asarray(weights, np.float32)
    n = len(p)
    with wp.ScopedDevice("cuda:0"):
        args = [
            wp.array(p, dtype=wp.vec3),
            wp.array(w),
            wp.empty((n, 2), dtype=int),
            wp.empty((n, 2), dtype=int),
            wp.empty((n, 4), dtype=wp.float64),
            wp.empty((n, 2), dtype=int),
        ]
        wp.launch(evaluate, n, inputs=args, block_dim=32)
    actual, valid, bounds, fast = [a.numpy() for a in args[2:]]
    np.testing.assert_array_equal(valid, 1)
    for i in range(n):
        exact = exact_reference(p[i], w[i])
        np.testing.assert_array_equal(actual[i], [(x > 0) - (x < 0) for x in exact])
        for j in range(2):
            # Check interval containment in rational arithmetic, not a rounded
            # comparison against an already rounded reference determinant.
            center, error = map(Fraction, bounds[i, 2 * j : 2 * j + 2])
            assert abs(center - exact[j]) <= error
    return fast


def test_random_normals_and_float32_bit_patterns_have_certified_bounds():
    rng = np.random.default_rng(9234)
    p = rng.normal(size=(256, 5, 3)).astype(np.float32)
    assert np.all(check(p))
    raw = rng.integers(0, 2**32, (128, 5, 3), dtype=np.uint32) & np.uint32(0xFEFFFFFF)
    weights = rng.integers(0, 2**32, (128, 5), dtype=np.uint32) & np.uint32(0xFEFFFFFF)
    check(raw.view(np.float32), weights.view(np.float32))


def test_exact_ties_one_ulp_and_extreme_scales_use_fallback():
    from itertools import product

    rng = np.random.default_rng(443)
    p = rng.integers(-8, 9, (64, 5, 3)).astype(np.float32)
    w = np.sum(p * p, axis=2)
    assert np.all(check(p, w)[:, 1] == 0)
    w[:, 4] = np.nextafter(w[:, 4], np.float32(np.inf))
    check(p, w)
    cases = []
    for scale, translate in product([-120, -50, 0, 50, 100], [0, 2**12, 2**20]):
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        x = ((x + translate) * 2.0**scale).astype(np.float32)
        cases.append(x.copy())
        x[4, 1] = np.nextafter(x[4, 1], np.float32(np.inf))
        cases.append(x.copy())
    check(cases)


def test_small_integer_ties_with_large_fractional_translation_are_exact():
    rng = np.random.default_rng(9967)
    p = rng.integers(-64, 65, size=(128, 5, 3)).astype(np.float32)
    w = np.sum(p * p, axis=2)
    # These weighted lifts are affine, hence the exact determinant is zero.
    check(p, w)
    check(p + np.float32(2**20 + 0.125), w)
    p = rng.integers(0, 513, size=(128, 5, 3)).astype(np.float32)
    p[:, 4] = 0
    w = rng.integers(-262144, 262145, size=(128, 5)).astype(np.float32)
    check(p, w)
    p[:, 0, 0] = np.nextafter(np.float32(512), np.float32(np.inf))
    check(p, w)


@wp.kernel
def difference_gate(a: wp.array[float], b: wp.array[float], accepted: wp.array[int]):
    i = wp.tid()
    accepted[i] = int(fp.integer_difference(a[i], b[i]))


def test_integer_shortcut_rejects_lost_low_bits_and_noninteger_differences():
    from fractions import Fraction

    tiny = np.nextafter(np.float32(0), np.float32(1))
    pairs = np.array(
        [
            (512, tiny),
            (0, tiny),
            (512, 0),
            (512, 1),
            (2**20 + 0.125, 2**20 + 1.125),
            (1, 1e-20),
            (1, np.nextafter(np.float32(1), np.float32(2))),
            (np.nextafter(np.float32(512), np.float32(np.inf)), 0),
        ],
        np.float32,
    )
    with wp.ScopedDevice("cuda:0"):
        a, b = wp.array(pairs[:, 0]), wp.array(pairs[:, 1])
        accepted = wp.empty(len(pairs), dtype=int)
        wp.launch(difference_gate, len(pairs), inputs=[a, b, accepted])
    expected = []
    for x, y in pairs:
        exact = Fraction(float(x)) - Fraction(float(y))
        expected.append(int(abs(exact) <= 512 and exact.denominator == 1))
    np.testing.assert_array_equal(accepted.numpy(), expected)
