from fractions import Fraction

import numpy as np
import warp as wp

from warpower import _integer as ie

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def evaluate(
    source: wp.array2d[float],
    words: wp.array3d[wp.uint32],
    meta: wp.array3d[int],
    bases: wp.array[int],
):
    i = wp.tid()
    base = wp.min(ie.base_for(source[i, 0], 1), ie.base_for(source[i, 1], 1))
    bases[i] = base
    a, b = ie.from_float(source[i, 0], base), ie.from_float(source[i, 1], base)
    for k in range(4):
        r = a
        if k == 0:
            r = ie.plus(a, b)
        elif k == 1:
            r = ie.minus(a, b)
        elif k == 2:
            r = ie.times(a, b)
        meta[i, k, 0] = r.sign
        meta[i, k, 1] = r.count
        meta[i, k, 2] = r.valid
        for j in range(48):
            words[i, k, j] = r.words[j]


def test_float_decomposition_signed_arithmetic_and_carries_match_python_integers():
    rng = np.random.default_rng(777)
    raw = rng.integers(0, 2**32, (500, 2), dtype=np.uint32) & np.uint32(0xFEFFFFFF)
    x = raw.view(np.float32)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    huge = np.finfo(np.float32).max
    x = np.concatenate(
        [
            x,
            np.array(
                [
                    [0, 0],
                    [tiny, huge],
                    [-huge, tiny],
                    [1, -1],
                    [-1, 1],
                    [0.5, 0.5],
                    [2**24, 1],
                    [2**24, -1],
                ],
                np.float32,
            ),
        ]
    )
    source = wp.array(x, dtype=float, device="cuda:0")
    words = wp.empty((len(x), 4, 48), dtype=wp.uint32, device=source.device)
    meta = wp.empty((len(x), 4, 3), dtype=int, device=source.device)
    bases = wp.empty(len(x), dtype=int, device=source.device)
    wp.launch(evaluate, len(x), inputs=[source, words, meta, bases])
    values, properties, exponents = words.numpy(), meta.numpy(), bases.numpy()
    for row, (a, b) in enumerate(x):
        scale = Fraction(2) ** int(exponents[row])
        fa, fb = Fraction(float(a)) / scale, Fraction(float(b)) / scale
        assert fa.denominator == fb.denominator == 1
        a, b = int(fa), int(fb)
        for k, expected in enumerate([a + b, a - b, a * b, a]):
            sign, count, valid = map(int, properties[row, k])
            assert valid == 1
            actual = sign * sum(int(v) << (32 * j) for j, v in enumerate(values[row, k]))
            assert actual == expected, (row, k, actual, expected)
            assert count == (abs(expected).bit_length() + 31) // 32
            assert sign == (expected > 0) - (expected < 0)


@wp.kernel
def limits(output: wp.array[int], words: wp.array[wp.uint32]):
    full = ie.empty()
    full.count = 48
    full.sign = 1
    for i in range(48):
        full.words[i] = wp.uint32(0xFFFFFFFF)
    one = ie.from_float(1.0, 0)
    two = ie.from_float(2.0, 0)
    a = ie.plus(full, one)
    b = ie.times(full, two)
    zero = ie.minus(full, full)
    output[0] = a.valid
    output[1] = b.valid
    output[2] = zero.valid
    output[3] = zero.sign
    output[4] = zero.count
    high = ie.empty()
    high.count = 33
    high.sign = 1
    high.words[32] = wp.uint32(1)
    borrow = ie.minus(high, one)
    output[5] = borrow.valid
    output[6] = borrow.sign
    output[7] = borrow.count
    for j in range(48):
        words[j] = borrow.words[j]


def test_explicit_overflow_and_long_borrow_chain():
    output = wp.empty(8, dtype=int, device="cuda:0")
    words = wp.empty(48, dtype=wp.uint32, device=output.device)
    wp.launch(limits, 1, inputs=[output, words])
    np.testing.assert_array_equal(output.numpy(), [0, 0, 1, 0, 0, 1, 1, 32])
    np.testing.assert_array_equal(words.numpy()[:32], np.uint32(0xFFFFFFFF))
    np.testing.assert_array_equal(words.numpy()[32:], 0)


@wp.kernel
def powers(output: wp.array[wp.float64]):
    i = wp.tid()
    output[i] = ie.power2(i - 800)


def test_every_power_of_two_exponent_in_predicate_range():
    output = wp.empty(1601, dtype=wp.float64, device="cuda:0")
    wp.launch(powers, len(output), inputs=[output])
    np.testing.assert_array_equal(output.numpy(), np.exp2(np.arange(-800, 801, dtype=float)))
