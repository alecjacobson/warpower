"""Rational original-coordinate enclosures, with and without explicit flushing."""
from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pytest
import warp as wp

import bounded_float32_filters as normal
import flushed_float32_filters as flushed
from tests.test_predicates import exact_reference
from warpower._analytic_predicates import strict

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def evaluate(p: wp.array2d[wp.vec3], bounds: wp.array3d[wp.float64], accepted: wp.array3d[int]):
    i = wp.tid()
    a, b, c, d, q = p[i, 0], p[i, 1], p[i, 2], p[i, 3], p[i, 4]
    o = normal.orientation_filter(a, b, c, d)
    s = normal.sphere_filter(a, b, c, d, q)
    of = flushed.orientation_filter(a, b, c, d)
    sf = flushed.sphere_filter(a, b, c, d, q)
    bounds[i, 0, 0], bounds[i, 0, 1] = o.value, o.error
    bounds[i, 0, 2], bounds[i, 0, 3] = s.value, s.error
    bounds[i, 1, 0], bounds[i, 1, 1] = of.value, of.error
    bounds[i, 1, 2], bounds[i, 1, 3] = sf.value, sf.error
    accepted[i, 0, 0], accepted[i, 0, 1] = int(strict(o)), int(strict(s))
    accepted[i, 1, 0], accepted[i, 1, 1] = int(strict(of)), int(strict(sf))


def check(points, block):
    points = np.asarray(points, np.float32)
    n = len(points)
    with wp.ScopedDevice('cuda:0'):
        p = wp.array(points, dtype=wp.vec3)
        bounds = wp.empty((n, 2, 4), dtype=wp.float64)
        accepted = wp.empty((n, 2, 2), dtype=int)
        wp.launch(evaluate, n, inputs=[p, bounds, accepted], block_dim=block)
    b, fast = bounds.numpy(), accepted.numpy()
    for i, row in enumerate(points):
        finite = np.all(np.isfinite(row))
        oracle = exact_reference(row, np.zeros(5)) if finite else None
        for j, used in enumerate([row[:4], row]):
            supported = np.all(np.isfinite(used)) and np.all(np.abs(used) <= 16)
            for variant in range(2):
                center, error = b[i, variant, 2*j:2*j+2]
                if not supported:
                    assert center == 0 and np.isposinf(error) and fast[i, variant, j] == 0
                    continue
                assert np.isfinite(center) and np.isfinite(error) and error >= 2.0**-64
                # A nonfinite q alone leaves orientation well defined.
                want = oracle[j] if oracle else exact_reference(np.vstack([row[:4], row[0]]), np.zeros(5))[j]
                center, error = Fraction(center), Fraction(error)
                assert abs(center - want) <= error, (i, j, variant, center, error, want)
                if fast[i, variant, j]:
                    assert want != 0 and (center > 0) == (want > 0)
    return b, fast


@pytest.mark.parametrize('block', [32, 256])
def test_ordinary_random_and_finite_bit_ranges(block):
    rng = np.random.default_rng(180)
    _, fast = check(rng.normal(size=(1024, 5, 3)), block)
    assert fast.mean() > 0.99
    bits = rng.integers(0, 0x41800001, size=(512, 5, 3), dtype=np.uint32)
    bits |= rng.integers(0, 2, size=bits.shape, dtype=np.uint32) << np.uint32(31)
    check(bits.view(np.float32), block)
    bits = rng.integers(0, 2**32, size=(256, 5, 3), dtype=np.uint32)
    bad = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    bits[bad] ^= np.uint32(0x00800000)
    check(bits.view(np.float32), block)


@pytest.mark.parametrize('block', [32, 256])
def test_subnormals_cancellation_scales_range_boundaries_and_ties(block):
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float32)
    cases, ties = [], []
    for exponent, translation in product([-149, -140, -126, -120, -80, -60, -30, -20, -15, -10, -5, 0, 3, 4, 5, 60, 100], [0, 1, 16]):
        p = ((base.astype(float) + translation) * 2.0**exponent).astype(np.float32)
        ties.append(len(cases))
        cases.append(p.copy())
        for direction in [-np.inf, np.inf]:
            q = p.copy()
            q[4, 1] = np.nextafter(q[4, 1], np.float32(direction))
            cases.append(q)
        q = p.copy()
        q[:, 2] = q[0, 2]
        cases.append(q)
    for order in permutations(range(4)):
        cases.append(base[[*order, 4]])
    for tiny in [np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).tiny]:
        for axis in range(3):
            p = base.copy()
            p[:, axis] *= tiny
            cases.append(p)
    for scale in [np.finfo(np.float32).max, 16, np.nextafter(np.float32(16), np.float32(np.inf))]:
        cases.append(base * scale)
    _, fast = check(cases, block)
    assert np.all(fast[ties, :, 1] == 0)
    assert np.any(fast == 0) and np.any(fast == 1)


@pytest.mark.parametrize('block', [32, 256])
def test_near_singular_shears_and_one_ulp_changes(block):
    rng = np.random.default_rng(1807)
    cases = []
    for _ in range(128):
        p = rng.integers(-8, 9, size=(5, 3)).astype(np.float32)
        p[:, 2] = p[:, 0] + p[:, 1]
        cases.append(p.copy())
        p[3, 2] = np.nextafter(p[3, 2], np.float32(np.inf))
        cases.append(p.copy())
        cases.append(p * np.float32(2**-6) + np.float32(8))
    _, fast = check(cases, block)
    assert np.any(fast == 0)


@pytest.mark.parametrize('block', [32, 256])
def test_nonfinite_and_unsupported_ranges_are_unavailable(block):
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]], np.float32)
    cases = []
    for value, vertex, axis in product([np.inf, -np.inf, np.nan, 17, -17], range(5), range(3)):
        p = base.copy()
        p[vertex, axis] = value
        cases.append(p)
    check(cases, block)
