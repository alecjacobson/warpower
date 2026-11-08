"""Exact rational validation of every exported interval and triangle event."""

from fractions import Fraction

import numpy as np
import warp as wp

from warpower import _filtered_predicates as fp
from warpower._exterior_proof import centroid_row, determinant, direction_row, triangle

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def inspect(
    rows: wp.array2d[wp.vec3],
    bounds: wp.array2d[wp.float64],
    flags: wp.array[int],
    result: wp.array[wp.vec2i],
):
    i = wp.tid()
    p0, p1, p2, p3 = rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3]
    q0, q1, q2, direction = rows[i, 4], rows[i, 5], rows[i, 6], rows[i, 7]
    a, b, c = (
        centroid_row(q0, p0, p1, p2, p3),
        centroid_row(q1, p0, p1, p2, p3),
        centroid_row(q2, p0, p1, p2, p3),
    )
    d = direction_row(direction)
    n, u, v, w = (
        determinant(a, b, c),
        determinant(d, b, c),
        determinant(d, c, a),
        determinant(d, a, b),
    )
    denominator = fp.plus(fp.plus(u, v), w)
    bounds[i, 0] = n.value
    bounds[i, 1] = n.error
    bounds[i, 2] = u.value
    bounds[i, 3] = u.error
    bounds[i, 4] = v.value
    bounds[i, 5] = v.error
    bounds[i, 6] = w.value
    bounds[i, 7] = w.error
    bounds[i, 8] = denominator.value
    bounds[i, 9] = denominator.error
    flags[i] = int(
        fp.certain(n)
        and fp.certain(u)
        and fp.certain(v)
        and fp.certain(w)
        and fp.certain(denominator)
    )
    result[i] = triangle(p0, p1, p2, p3, q0, q1, q2, direction)


def ivec(p):
    out = []
    for x in p:
        n, d = float(x).as_integer_ratio()
        out.append(n * ((1 << 149) // d))
    return out


def det(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def sign(x):
    return (x > 0) - (x < 0)


def reference(row):
    p = [ivec(v) for v in row[:4]]
    a, b, c = [[4 * q[k] - sum(x[k] for x in p) for k in range(3)] for q in map(ivec, row[4:7])]
    d = ivec(row[7])
    n = det(a, b, c)
    u, v, w = det(d, b, c), det(d, c, a), det(d, a, b)
    den = u + v + w
    expected = [0, 0]
    if den == 0:
        if n == 0:
            expected = [0, 1]
    elif n != 0 and sign(n) == sign(den):
        if all(t == 0 or sign(t) == sign(den) for t in [u, v, w]):
            expected = [0, 1] if any(t == 0 for t in [u, v, w]) else [sign(den), 0]
    return [Fraction(x, 1 << 447) for x in [n, u, v, w, den]], expected


def test_centroid_intervals_enclose_exact_determinants_and_match_ray_events():
    rng = np.random.default_rng(206)
    ordinary = rng.uniform(-4, 4, (512, 8, 3)).astype(np.float32)
    bits = (
        (rng.integers(0, 2, (512, 8, 3), dtype=np.uint32) << 31)
        | (rng.integers(0, 255, (512, 8, 3), dtype=np.uint32) << 23)
        | rng.integers(0, 1 << 23, (512, 8, 3), dtype=np.uint32)
    )
    special = []
    for exponent in [-140, -100, 0, 100, 125]:
        for direction in [[1, 1, 1], [11, 23, 37], [0, 0, 1]]:
            for face in [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            ]:
                sites = np.array(
                    [[-0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]],
                    np.float32,
                )
                special.append(
                    np.vstack(
                        [
                            sites * np.float32(2.0**exponent),
                            np.array(face, np.float32) * np.float32(2.0**exponent),
                            direction,
                        ]
                    )
                )
    data = np.concatenate([ordinary, bits.view(np.float32), np.array(special, np.float32)])
    assert np.isfinite(data).all()
    source = wp.array(data, dtype=wp.vec3)
    bounds = wp.empty((len(data), 10), dtype=wp.float64)
    flags = wp.empty(len(data), dtype=int)
    events = wp.empty(len(data), dtype=wp.vec2i)
    wp.launch(inspect, len(data), inputs=[source, bounds, flags, events], block_dim=32)
    actual, b, f = events.numpy(), bounds.numpy(), flags.numpy()
    assert 0 < f.sum() < len(data)
    assert np.isfinite(b).all() and (b[:, 1::2] >= 0).all()
    for i, row in enumerate(data):
        exact, expected = reference(row)
        np.testing.assert_array_equal(actual[i], expected, err_msg=str(i))
        for j, x in enumerate(exact):
            value, error = map(Fraction, map(float, b[i, 2 * j : 2 * j + 2]))
            assert value - error <= x <= value + error, (i, j, x, value, error)
