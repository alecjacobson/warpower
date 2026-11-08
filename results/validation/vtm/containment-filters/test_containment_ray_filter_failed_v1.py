"""Independent rational bounds and exact strict-witness ray events."""

from fractions import Fraction

import numpy as np
import pytest
import warp as wp

from warpower import _filtered_predicates as fp
from warpower._containment import triangle, triangle_bounds

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def inspect(
    rows: wp.array2d[wp.vec3],
    bounds: wp.array2d[wp.float64],
    flags: wp.array[int],
    events: wp.array[wp.vec2i],
):
    i = wp.tid()
    p, a, b, c, d = rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3], rows[i, 4]
    n, u, v, w, den = triangle_bounds(p, a, b, c, d)
    bounds[i, 0], bounds[i, 1] = n.value, n.error
    bounds[i, 2], bounds[i, 3] = u.value, u.error
    bounds[i, 4], bounds[i, 5] = v.value, v.error
    bounds[i, 6], bounds[i, 7] = w.value, w.error
    bounds[i, 8], bounds[i, 9] = den.value, den.error
    flags[i] = int(
        fp.certain(n) and fp.certain(u) and fp.certain(v) and fp.certain(w) and fp.certain(den)
    )
    events[i] = triangle(p, a, b, c, d)


def reference(row):
    p, a, b, c, d = [[Fraction(float(x)) for x in r] for r in row]
    a, b, c = [[v - origin for v, origin in zip(q, p)] for q in [a, b, c]]

    def det(a, b, c):
        return (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )

    n, u, v, w = det(a, b, c), det(d, b, c), det(d, c, a), det(d, a, b)
    den = u + v + w
    if den == 0:
        event = [0, int(n == 0)]
    elif n / den <= 0:
        event = [0, 0]
    elif min(u / den, v / den, w / den) < 0:
        event = [0, 0]
    elif min(u / den, v / den, w / den) == 0:
        event = [0, 1]
    else:
        event = [1 if den > 0 else -1, 0]
    return [n, u, v, w, den], event


def finite_cases():
    rng = np.random.default_rng(209)
    ordinary = rng.uniform(-4, 4, (512, 5, 3)).astype(np.float32)
    bits = (
        (rng.integers(0, 2, (512, 5, 3), dtype=np.uint32) << 31)
        | (rng.integers(0, 255, (512, 5, 3), dtype=np.uint32) << 23)
        | rng.integers(0, 1 << 23, (512, 5, 3), dtype=np.uint32)
    )
    special = []
    for exponent in [-149, -140, -100, 0, 100, 125]:
        for point in [[0, 0, 0], [1, 0, 0], [0.25, 0.25, -1], [0.25, 0.25, 1]]:
            for face in [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            ]:
                for direction in [[11, 23, 37], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]]:
                    special.append(
                        np.vstack(
                            [
                                np.array([point, *face], np.float32) * np.float32(2.0**exponent),
                                direction,
                            ]
                        )
                    )
    return np.concatenate([ordinary, bits.view(np.float32), np.array(special, np.float32)])


def test_strict_ray_bounds_enclose_exact_values_and_preserve_events():
    data = finite_cases()
    assert np.isfinite(data).all()
    bounds, flags, events = (
        wp.empty((len(data), 10), dtype=wp.float64),
        wp.empty(len(data), dtype=int),
        wp.empty(len(data), dtype=wp.vec2i),
    )
    source = wp.array(data, dtype=wp.vec3)
    with wp.ScopedCapture() as capture:
        wp.launch(inspect, len(data), inputs=[source, bounds, flags, events], block_dim=32)
    wp.capture_launch(capture.graph)
    actual, b, f = events.numpy(), bounds.numpy(), flags.numpy()
    assert 0 < f.sum() < len(data)
    assert np.isfinite(b).all() and (b[:, 1::2] >= 0).all()
    expected_flags = set()
    for i, row in enumerate(data):
        values, event = reference(row)
        expected_flags.add(event[1])
        np.testing.assert_array_equal(actual[i], event, err_msg=str(i))
        for j, x in enumerate(values):
            value, error = map(Fraction, map(float, b[i, 2 * j : 2 * j + 2]))
            assert value - error <= x <= value + error, (i, j, x, value, error)
    assert expected_flags == {0, 1}


@pytest.mark.parametrize("row", range(5))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_ray_input_is_inconclusive(row, value):
    data = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [11, 23, 37]]], np.float32)
    data[0, row, 1] = value
    bounds, flags, events = (
        wp.empty((1, 10), dtype=wp.float64),
        wp.empty(1, dtype=int),
        wp.empty(1, dtype=wp.vec2i),
    )
    wp.launch(
        inspect, 1, inputs=[wp.array(data, dtype=wp.vec3), bounds, flags, events], block_dim=32
    )
    np.testing.assert_array_equal(flags.numpy(), [0])
    np.testing.assert_array_equal(events.numpy(), [[0, 2]])
