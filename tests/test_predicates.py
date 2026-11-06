import math
from fractions import Fraction

import numpy as np
import warp as wp

from warpower import _integer as integer
from warpower._predicates import orient3d, power_sphere

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
NAN = wp.constant(float("nan"))


def determinant(rows):
    """Independent exact integer Bareiss determinant, after positive row scaling."""
    integer, denominator = [], 1
    for row in rows:
        row = [Fraction(x) for x in row]
        factor = math.lcm(*(x.denominator for x in row))
        integer.append([x.numerator * (factor // x.denominator) for x in row])
        denominator *= factor
    n, sign, previous = len(rows), 1, 1
    for k in range(n - 1):
        pivot = next((j for j in range(k, n) if integer[j][k]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != k:
            integer[k], integer[pivot] = integer[pivot], integer[k]
            sign *= -1
        p = integer[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                value = integer[i][j] * p - integer[i][k] * integer[k][j]
                assert value % previous == 0
                integer[i][j] = value // previous
            integer[i][k] = 0
        previous = p
    return Fraction(sign * integer[-1][-1], denominator)


def exact_reference(points, weights):
    p = [[Fraction(float(x)) for x in row] for row in points]
    orientation = -determinant([row + [Fraction(1)] for row in p[:4]])
    sphere = -determinant(
        [
            row + [sum(x * x for x in row) - Fraction(float(w)), Fraction(1)]
            for row, w in zip(p, weights)
        ]
    )
    return orientation, sphere


@wp.kernel
def evaluate(
    points: wp.array2d[wp.vec3],
    weights: wp.array2d[float],
    signs: wp.array2d[int],
    estimates: wp.array2d[wp.float64],
    valid: wp.array2d[int],
):
    i = wp.tid()
    a, b, c, d, q = points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4]
    o = orient3d(a, b, c, d)
    s = power_sphere(
        a, b, c, d, q, weights[i, 0], weights[i, 1], weights[i, 2], weights[i, 3], weights[i, 4]
    )
    signs[i, 0] = o.sign
    signs[i, 1] = s.sign
    estimates[i, 0] = o.estimate
    estimates[i, 1] = s.estimate
    valid[i, 0] = o.valid
    valid[i, 1] = s.valid


def check(points, weights=None):
    points = np.asarray(points, np.float32)
    n = len(points)
    weights = np.zeros((n, 5), np.float32) if weights is None else np.asarray(weights, np.float32)
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    w = wp.array(weights, device="cuda:0")
    signs = wp.empty((n, 2), dtype=int, device="cuda:0")
    estimates = wp.empty((n, 2), dtype=wp.float64, device="cuda:0")
    valid = wp.empty((n, 2), dtype=int, device="cuda:0")
    wp.launch(evaluate, n, inputs=[p, w, signs, estimates, valid])
    np.testing.assert_array_equal(valid.numpy(), 1)
    actual, values = signs.numpy(), estimates.numpy()
    for i, (p0, w0) in enumerate(zip(points, weights)):
        expected = exact_reference(p0, w0)
        np.testing.assert_array_equal(actual[i], [(x > 0) - (x < 0) for x in expected])
        np.testing.assert_allclose(values[i], [float(x) for x in expected], rtol=2e-15, atol=0)
    return actual


def test_unit_tet_conventions_inside_outside_and_permutation():
    tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    samples = [np.vstack([tet, q]) for q in [[0.5, 0.5, 0.5], [2, 2, 2], [1, 1, 1]]]
    samples.append(samples[0][[1, 0, 2, 3, 4]])
    np.testing.assert_array_equal(check(samples), [[1, 1], [1, -1], [1, 0], [-1, -1]])


def test_exact_coplanar_cospherical_weighted_ties_and_one_ulp_perturbations():
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float32)
    samples = [base.copy()]
    for direction in [-np.inf, np.inf]:
        p = base.copy()
        p[-1, 1] = np.nextafter(p[-1, 1], np.float32(direction))
        samples.append(p)
    flat = base.copy()
    flat[:, 2] = 0
    samples.append(flat)
    check(samples)
    rng = np.random.default_rng(23)
    points = rng.integers(-8, 9, size=(30, 5, 3)).astype(np.float32)
    # All lifts exactly zero, hence an exact weighted co-spherical determinant.
    weights = np.sum(points * points, axis=2)
    result = check(points, weights)
    np.testing.assert_array_equal(result[:, 1], 0)
    weights[:, 4] = np.nextafter(weights[:, 4], np.float32(np.inf))
    check(points, weights)


def test_random_float32_bits_and_extreme_exponents_match_integer_oracle():
    rng = np.random.default_rng(456)
    raw = rng.integers(0, 2**32, size=(100, 5, 3), dtype=np.uint32)
    raw &= np.uint32(0xFEFFFFFF)  # Clear one exponent bit: exclude NaN/Inf.
    p = raw.view(np.float32)
    wbits = rng.integers(0, 2**32, size=(100, 5), dtype=np.uint32) & np.uint32(0xFEFFFFFF)
    check(p, wbits.view(np.float32))
    tiny = np.nextafter(np.float32(0), np.float32(1))
    huge = np.finfo(np.float32).max
    cases = []
    for value in [tiny, np.float32(2**-120), np.float32(2**100), huge]:
        cases.append(
            np.array(
                [[0, 0, 0], [value, 0, 0], [0, value, 0], [0, 0, value], [-value, 0, 0]], np.float32
            )
        )
    check(cases)


def test_translated_scaled_nearly_flat_clouds_and_cancellation():
    rng = np.random.default_rng(921)
    cases = []
    for exponent in [-120, -50, 0, 50, 100]:
        for translation in [0, 2**12, 2**20]:
            p = rng.integers(-8, 9, size=(5, 3)).astype(np.float64)
            p[:, 2] = p[:, 0] + p[:, 1]
            p = (p + translation) * float(2.0**exponent)
            p = p.astype(np.float32)
            cases.append(p.copy())
            p[3, 2] = np.nextafter(p[3, 2], np.float32(np.inf))
            cases.append(p.copy())
    check(cases)


@wp.kernel
def invalid_and_overflow(output: wp.array[int]):
    value = integer.empty()
    value.count = 48
    value.sign = 1
    for i in range(48):
        value.words[i] = wp.uint32(0xFFFFFFFF)
    overflow = integer.plus(value, integer.from_float(1.0, 0))
    output[0] = overflow.valid
    bad = integer.Big()
    bad.valid = 0
    output[1] = integer.plus(integer.empty(), bad).valid
    p = orient3d(wp.vec3(NAN), wp.vec3(0.0), wp.vec3(1.0), wp.vec3(2.0))
    output[2] = p.valid


def test_nonfinite_inputs_and_capacity_failure_are_not_reported_as_exact_zero():
    out = wp.empty(3, dtype=int, device="cuda:0")
    wp.launch(invalid_and_overflow, 1, inputs=[out])
    np.testing.assert_array_equal(out.numpy(), 0)


@wp.kernel
def change_query(points: wp.array2d[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    # The fixed unit tetrahedron's sphere passes through (1,1,1).
    points[i, 4] = wp.vec3(1.0 + 0.125 * float(phase[0]), 1.0, 1.0)


@wp.kernel
def reduce_signs(signs: wp.array2d[int], valid: wp.array2d[int], output: wp.array[int]):
    i = wp.tid()
    if valid[i, 0] != 0 and valid[i, 1] != 0:
        wp.atomic_add(output, 0, signs[i, 1])


def test_predicates_compose_with_changing_device_inputs_and_consumer(monkeypatch, tmp_path):
    p0 = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]] * 8, np.float32)
    points = wp.array(p0, dtype=wp.vec3, device="cuda:0")
    weights = wp.zeros((8, 5), dtype=float, device="cuda:0")
    signs = wp.empty((8, 2), dtype=int, device="cuda:0")
    valid = wp.empty_like(signs)
    estimates = wp.empty((8, 2), dtype=wp.float64, device="cuda:0")
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    output = wp.zeros_like(phase)
    wp.load_module(module=__name__, device="cuda:0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Host work in captured predicate operation")

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
            wp.launch(change_query, 8, inputs=[points, phase])
            wp.launch(evaluate, 8, inputs=[points, weights, signs, estimates, valid])
            output.zero_()
            wp.launch(reduce_signs, 8, inputs=[signs, valid, output])
        wp.capture_launch(capture.graph)
    assert output.numpy()[0] == 0
    phase.fill_(1)
    wp.capture_launch(capture.graph)
    assert output.numpy()[0] == -8
    phase.fill_(-1)
    wp.capture_launch(capture.graph)
    assert output.numpy()[0] == 8
    dot = tmp_path / "predicates.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    text = dot.read_text()
    assert "MEM_ALLOC" not in text and "MEM_FREE" not in text and "HOST" not in text
