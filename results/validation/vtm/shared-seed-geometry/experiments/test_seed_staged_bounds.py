from fractions import Fraction
from itertools import product

import numpy as np
import warp as wp

from tests.test_predicates import exact_reference
from warpower._delaunay_certificate import sphere
from seed_filter_staged import query_ball

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def evaluate(points: wp.array2d[wp.vec3], values: wp.array2d[wp.float64]):
    i = wp.tid()
    shape = sphere(points[i, 0], points[i, 1], points[i, 2], points[i, 3])
    ball = query_ball(shape, points[i, 4])
    values[i, 0] = ball.value
    values[i, 1] = ball.error


def test_current_input_query_bounds_against_independent_rational_determinants():
    rng = np.random.default_rng(157)
    normal = rng.normal(size=(512, 5, 3)).astype(np.float32)
    bits = rng.integers(0, 2**32, (128, 5, 3), dtype=np.uint32) & np.uint32(0xFEFFFFFF)
    cases = [*normal, *bits.view(np.float32)]
    for exponent, translation in product([-140, -120, -50, 0, 50, 100], [0, 2**12, 2**20]):
        p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        p = ((p + translation) * 2.0**exponent).astype(np.float32)
        cases.append(p.copy())
        p[4, 1] = np.nextafter(p[4, 1], np.float32(np.inf))
        cases.append(p.copy())
    points = np.asarray(cases, np.float32)
    with wp.ScopedDevice('cuda:0'):
        p = wp.array(points, dtype=wp.vec3)
        values = wp.empty((len(p), 2), dtype=wp.float64)
        wp.launch(evaluate, len(p), inputs=[p, values], block_dim=32)
    seen = {-1: 0, 0: 0, 1: 0}
    for p, (v, e) in zip(points, values.numpy()):
        determinant = exact_reference(p, np.zeros(5, np.float32))[1]
        assert np.isfinite(v) and np.isfinite(e)
        assert abs(Fraction(float(v)) - determinant) <= Fraction(float(e))
        sign = int(v > e) - int(v < -e)
        seen[sign] += 1
        if sign:
            assert sign == (determinant > 0) - (determinant < 0)
    assert all(seen.values()), seen
