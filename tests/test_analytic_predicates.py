"""Independent rational determinants check signs AND full error enclosures."""

from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pytest
import warp as wp

from tests.test_predicates import exact_reference
from warpower import _filtered_predicates as fp
from warpower._analytic_predicates import orient3d, orientation_filter, sphere_filter, strict

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def evaluate(
    p: wp.array2d[wp.vec3],
    bounds: wp.array2d[wp.float64],
    fast: wp.array2d[int],
    signs: wp.array2d[int],
    valid: wp.array2d[int],
):
    i = wp.tid()
    a, b, c, d, q = p[i, 0], p[i, 1], p[i, 2], p[i, 3], p[i, 4]
    ob = orientation_filter(a, b, c, d)
    sb = sphere_filter(a, b, c, d, q)
    o = orient3d(a, b, c, d)
    s = fp.predicate(sb)
    if not strict(sb):
        s = fp.insphere(a, b, c, d, q)
    bounds[i, 0] = ob.value
    bounds[i, 1] = ob.error
    bounds[i, 2] = sb.value
    bounds[i, 3] = sb.error
    fast[i, 0] = int(strict(ob))
    fast[i, 1] = int(strict(sb))
    signs[i, 0] = o.sign
    signs[i, 1] = s.sign
    valid[i, 0] = o.valid
    valid[i, 1] = s.valid


def check(p, block=32, finite=True):
    p = np.asarray(p, dtype=np.float32)
    n = len(p)
    with wp.ScopedDevice("cuda:0"):
        arrays = [wp.array(p, dtype=wp.vec3), wp.empty((n, 4), dtype=wp.float64)] + [
            wp.empty((n, 2), dtype=int) for _ in range(3)
        ]
        wp.launch(evaluate, n, inputs=arrays, block_dim=block)
    bounds, fast, signs, valid = [a.numpy() for a in arrays[1:]]
    if finite:
        np.testing.assert_array_equal(valid, 1)
        assert np.all(np.isfinite(bounds)) and np.all(bounds[:, [1, 3]] >= 0)
        for i, row in enumerate(p):
            oracle = exact_reference(row, np.zeros(5))
            np.testing.assert_array_equal(signs[i], [(x > 0) - (x < 0) for x in oracle])
            for j, want in enumerate(oracle):
                center, error = map(Fraction, bounds[i, 2 * j : 2 * j + 2])
                assert abs(center - want) <= error, (i, j, center, error, want)
                if fast[i, j]:
                    assert want != 0 and (center > 0) == (want > 0)
    else:
        np.testing.assert_array_equal(fast, 0)
        np.testing.assert_array_equal(valid, 0)
    return fast


@pytest.mark.parametrize("block", [32, 256])
def test_random_and_full_float32_bit_ranges(block):
    rng = np.random.default_rng(166)
    assert np.all(check(rng.normal(size=(512, 5, 3)), block))
    bits = rng.integers(0, 2**32, size=(512, 5, 3), dtype=np.uint32)
    # Exclude only exponent 255; keep the other 255 exponent classes.
    bad = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    bits[bad] ^= np.uint32(0x00800000)
    check(bits.view(np.float32), block)


@pytest.mark.parametrize("block", [32, 256])
def test_extreme_scale_subnormals_ties_ulp_and_permutations(block):
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float32)
    cases = []
    ties = []
    for exponent, translation in product(
        [-149, -140, -126, -120, -60, 0, 60, 100], [0, 2**12, 2**20]
    ):
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
    # Products and differences span the actual finite float32 endpoint range.
    for scale in [np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1))]:
        cases.append(base * scale)
        cases.append(
            np.array(
                [[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale], [-scale, 0, 0]], np.float32
            )
        )
    fast = check(cases, block)
    assert np.all(fast[ties, 1] == 0)
    assert np.any(fast[:, 0] == 0) and np.any(fast[:, 1] == 0)
    # Smallest subnormal must survive promotion: its strict orientation is nonzero.
    assert fast[-1, 0] == 1


@pytest.mark.parametrize("block", [32, 256])
def test_near_singular_integer_shears_and_cancellation(block):
    rng = np.random.default_rng(1667)
    cases = []
    for _ in range(128):
        p = rng.integers(-32, 33, size=(5, 3)).astype(np.float32)
        p[:, 2] = p[:, 0] + p[:, 1]
        cases.append(p.copy())
        p[3, 2] = np.nextafter(p[3, 2], np.float32(np.inf))
        cases.append(p.copy())
        # Huge anisotropy and translations; oracle uses the resulting original bits.
        p *= np.array([2.0**80, 2.0**-80, 1], np.float32)
        p += np.array([2.0**100, 2.0**-70, 2.0**10], np.float32)
        cases.append(p)
    assert np.any(check(cases, block) == 0)


@pytest.mark.parametrize("block", [32, 256])
def test_nonfinite_calculations_never_produce_certified_signs(block):
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]], np.float32)
    cases = []
    for value, vertex, axis in product([np.inf, -np.inf, np.nan], range(4), range(3)):
        p = base.copy()
        p[vertex, axis] = value
        cases.append(p)
    check(cases, block, finite=False)
