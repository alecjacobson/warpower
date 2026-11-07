"""Independent exact determinants check the specialized point enclosure."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import warp as wp

from tests.test_predicates import exact_reference
from warpower._delaunay_certificate import sphere
from warpower._prepared_seeds import point_value

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def evaluate(p: wp.array2d[wp.vec3], bounds: wp.array2d[wp.float64], signs: wp.array[int]):
    i = wp.tid()
    shape = sphere(p[i, 0], p[i, 1], p[i, 2], p[i, 3])
    value = point_value(shape, p[i, 4])
    bounds[i, 0] = value.value
    bounds[i, 1] = value.error
    signs[i] = 0
    if wp.isfinite(value.value) and wp.isfinite(value.error) and wp.abs(value.value) > value.error:
        signs[i] = int(value.value > wp.float64(0.0)) - int(value.value < wp.float64(0.0))


def check(p, block):
    p = np.asarray(p, dtype=np.float32)
    with wp.ScopedDevice("cuda:0"):
        args = [
            wp.array(p, dtype=wp.vec3),
            wp.empty((len(p), 2), dtype=wp.float64),
            wp.empty(len(p), dtype=int),
        ]
        wp.launch(evaluate, len(p), inputs=args, block_dim=block)
    bounds, signs = [x.numpy() for x in args[1:]]
    assert np.all(np.isfinite(bounds)) and np.all(bounds[:, 1] >= 0)
    for i, points in enumerate(p):
        _, want = exact_reference(points, np.zeros(5))
        value, error = map(Fraction, bounds[i])
        assert abs(value - want) <= error, (i, value, error, want)
        if signs[i]:
            assert int(signs[i]) == int(want > 0) - int(want < 0)
    return signs


@pytest.mark.parametrize("block", [32, 256])
def test_point_bounds_enclose_random_and_all_float32_exponent_ranges(block):
    rng = np.random.default_rng(170)
    check(rng.normal(size=(256, 5, 3)), block)
    bits = rng.integers(0, 2**32, (256, 5, 3), dtype=np.uint32)
    bad = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    bits[bad] ^= np.uint32(0x00800000)
    check(bits.view(np.float32), block)


@pytest.mark.parametrize("block", [32, 256])
def test_point_bounds_ties_ulp_subnormals_translations_and_thin_cells(block):
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float32)
    cases = []
    ties = []
    for exponent, translate in product([-149, -140, -126, -60, 0, 60, 100], [0, 2**12, 2**20]):
        p = ((base.astype(float) + translate) * 2.0**exponent).astype(np.float32)
        ties.append(len(cases))
        cases.append(p.copy())
        for direction in [-np.inf, np.inf]:
            q = p.copy()
            q[4, 1] = np.nextafter(q[4, 1], np.float32(direction))
            cases.append(q)
        cases.append(p[[1, 0, 2, 3, 4]])
    excluded = check(cases, block)
    assert not np.any(excluded[ties])
    rng = np.random.default_rng(1707)
    p = rng.normal(size=(128, 5, 3)).astype(np.float32)
    p[:, :, 2] *= np.float32(2**-40)
    check(p, block)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    huge = np.finfo(np.float32).max
    check([base * tiny, base * huge], block)


@pytest.mark.parametrize("block", [32, 256])
def test_nonfinite_geometry_never_passes_the_strict_gate(block):
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float32)
    cases = []
    for site, axis, value in product(range(5), range(3), [np.nan, np.inf, -np.inf]):
        p = base.copy()
        p[site, axis] = value
        cases.append(p)
    with wp.ScopedDevice("cuda:0"):
        p = wp.array(np.asarray(cases), dtype=wp.vec3)
        bounds = wp.empty((len(cases), 2), dtype=wp.float64)
        signs = wp.empty(len(cases), dtype=int)
        wp.launch(evaluate, len(cases), inputs=[p, bounds, signs], block_dim=block)
    np.testing.assert_array_equal(signs.numpy(), 0)
