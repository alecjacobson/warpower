from fractions import Fraction as F

import numpy as np
import warp as wp

from tests.test_sphere_bounds import det
from warpower import _halfspace_predicates as hp
from warpower import _symbolic_bounds as sb

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def vertex_boxes(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    triples: wp.array[wp.vec3i],
    lo: wp.array[wp.vec3d],
    hi: wp.array[wp.vec3d],
):
    i = wp.tid()
    ids = triples[i]
    for k in range(3):
        if ids[k] >= 0:
            ids[k] += 4 * i
    result = sb.vertex_box(points, bounds, 4 * i, ids)
    lo[i] = result.lower
    hi[i] = result.upper


@wp.kernel
def rejected(
    lo: wp.array[wp.vec3d],
    hi: wp.array[wp.vec3d],
    owner: wp.array[wp.vec3],
    node_lo: wp.array[wp.vec3],
    node_hi: wp.array[wp.vec3],
    output: wp.array[int],
):
    i = wp.tid()
    output[i] = int(sb.outside(lo[i], hi[i], owner[i], node_lo[i], node_hi[i]))


def rational_row(points, bounds, index):
    index = int(index)
    if index >= 0:
        p, q = [[F(float(x)) for x in row] for row in points[[0, index]]]
        return [2 * (a - b) for a, b in zip(p, q)] + [sum(x * x for x in q) - sum(x * x for x in p)]
    axis, side = divmod(-index - 1, 2)
    row = [F(0)] * 4
    row[axis] = F(1 - 2 * side)
    row[3] = -row[axis] * F(float(bounds[side, axis]))
    return row


def intersection(points, bounds, ids):
    rows = [rational_row(points, bounds, i) for i in ids]
    determinant = det(*(r[:3] for r in rows))
    if determinant == 0:
        return None
    result = []
    for axis in range(3):
        matrix = [r[:3] for r in rows]
        for i in range(3):
            matrix[i][axis] = -rows[i][3]
        result.append(det(*matrix) / determinant)
    return result


def test_symbolic_vertex_limits_enclose_exact_site_and_box_intersections():
    rng = np.random.default_rng(6281)
    checked = 0
    cases = [(e, 0) for e in [-140, -126, -60, 0, 60, 120]]
    cases += [(e, 4096) for e in [-140, -126, -60, 0, 60, 100]]
    for exponent, shift in cases:
        raw = rng.normal(size=(80, 4, 3))
        raw[::8, 3] = 0.3 * raw[::8, 0] + 0.3 * raw[::8, 1] + 0.4 * raw[::8, 2]
        points = ((raw + shift) * 2.0**exponent).astype(np.float32)
        bounds = ((np.array([[-4] * 3, [4] * 3], float) + shift) * 2.0**exponent).astype(np.float32)
        triples = np.tile(
            np.array([[1, 2, 3], [1, -1, -3], [2, 3, -2], [-1, -3, -5]], np.int32), (20, 1)
        )
        lo = wp.empty(len(points), dtype=wp.vec3d, device="cuda:0")
        hi = wp.empty_like(lo)
        wp.launch(
            vertex_boxes,
            len(points),
            inputs=[
                wp.array(points.reshape(-1, 3), dtype=wp.vec3, device=lo.device),
                wp.array(bounds, dtype=wp.vec3, device=lo.device),
                wp.array(triples, dtype=wp.vec3i, device=lo.device),
                lo,
                hi,
            ],
        )
        for p, ids, lower, upper in zip(points, triples, lo.numpy(), hi.numpy()):
            exact = intersection(p, bounds, ids)
            if exact is None or not all(
                F(float(bounds[0, k])) <= exact[k] <= F(float(bounds[1, k])) for k in range(3)
            ):
                continue
            checked += 1
            for k in range(3):
                assert F(float(lower[k])) <= exact[k] <= F(float(upper[k]))
    assert checked > 200


def exact_maximum(lo, hi, owner, node_lo, node_hi):
    total = F(0)
    for low, high, p, a, b in zip(lo, hi, owner, node_lo, node_hi):
        low, high, p, a, b = (F(float(v)) for v in (low, high, p, a, b))
        low, high, a, b = low - p, high - p, a - p, b - p
        values = []
        for x in (low, high):
            q = min(max(x, a), b)
            values.append(2 * x * q - q * q)
        total += max(values)
    return total


def evaluate_rejection(lo, hi, p, a, b):
    output = wp.empty(len(p), dtype=int, device="cuda:0")
    arrays = [
        wp.array(v, dtype=t, device=output.device)
        for v, t in zip((lo, hi, p, a, b), (wp.vec3d, wp.vec3d, wp.vec3, wp.vec3, wp.vec3))
    ]
    wp.launch(rejected, len(p), inputs=[*arrays, output])
    return output.numpy()


def test_directional_rejection_is_strictly_safe_against_rational_box_maximum():
    rng = np.random.default_rng(6262)
    all_values = [[] for _ in range(5)]
    for exponent in [-140, -126, -60, 0, 60, 120]:
        scale = 2.0**exponent
        p = (rng.uniform(-1, 1, (100, 3)) * scale).astype(np.float32)
        extent = rng.uniform(0.01, 0.2, (100, 3)) * scale
        lo, hi = p.astype(float) - extent, p.astype(float) + extent
        center = (p.astype(float) + rng.normal(size=p.shape) * scale).astype(np.float32)
        size = (rng.uniform(0, 0.4, center.shape) * scale).astype(np.float32)
        a, b = (center - size).astype(np.float32), (center + size).astype(np.float32)
        for dest, values in zip(all_values, (lo, hi, p, a, b)):
            dest.extend(values)
    arrays = [np.asarray(v) for v in all_values]
    skipped = evaluate_rejection(*arrays)
    assert skipped.sum() > 200
    for rows, skip in zip(zip(*arrays), skipped):
        if skip:
            assert exact_maximum(*rows) < 0


def test_touching_limiting_cell_and_nonfinite_boxes_do_not_prune():
    # owner=0, cell=[-1,1]^3; q=(2,0,0) has zero separating gap at x=(1,0,0).
    lo, hi = np.full((4, 3), -1.0), np.full((4, 3), 1.0)
    p = np.zeros((4, 3), np.float32)
    q = np.array(
        [[2, 0, 0], [np.nextafter(np.float32(2), np.float32(np.inf)), 0, 0], [0, 0, 0], [2, 0, 0]],
        np.float32,
    )
    hi[3, 0] = np.inf
    np.testing.assert_array_equal(evaluate_rejection(lo, hi, p, q, q), [0, 1, 0, 0])


@wp.kernel
def cached_sides(
    points: wp.array[wp.vec3], bounds: wp.array[wp.vec3], valid: wp.array[int], signs: wp.array[int]
):
    i = wp.tid()
    owner = 5 * i
    box = sb.vertex_box(points, bounds, owner, wp.vec3i(owner + 1, owner + 2, owner + 3))
    center = sb.centered(box)
    value = wp.vec4d(center.value[0], center.value[1], center.value[2], wp.float64(0.0))
    result = sb.relation(value, center.error, hp.row(points, bounds, owner, owner + 4, 0, -1))
    valid[i] = result.valid
    signs[i] = result.sign


def test_cached_vertex_plane_filter_matches_rationals_and_defers_exact_ties():
    rng = np.random.default_rng(6217)
    checks, filtered, ties = 0, 0, 0
    for exponent, shift in [(-140, 0), (0, 0), (120, 0), (0, 4096), (100, 4096)]:
        raw = rng.uniform(-1, 1, (100, 5, 3))
        raw[::7, 4] = raw[::7, 1]
        raw[0] = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]]
        points = ((raw + shift) * 2.0**exponent).astype(np.float32)
        bounds = ((np.array([[-4] * 3, [4] * 3], float) + shift) * 2.0**exponent).astype(np.float32)
        valid = wp.empty(len(points), dtype=int, device="cuda:0")
        signs = wp.empty_like(valid)
        wp.launch(
            cached_sides,
            len(points),
            inputs=[
                wp.array(points.reshape(-1, 3), dtype=wp.vec3, device=valid.device),
                wp.array(bounds, dtype=wp.vec3, device=valid.device),
                valid,
                signs,
            ],
        )
        for p, accepted, sign in zip(points, valid.numpy(), signs.numpy()):
            c = intersection(p, bounds, [1, 2, 3])
            if c is None or not all(
                F(float(bounds[0, k])) <= c[k] <= F(float(bounds[1, k])) for k in range(3)
            ):
                continue
            row = rational_row(p, bounds, 4)
            side = sum(x * y for x, y in zip(row[:3], c)) + row[3]
            checks += 1
            if side == 0:
                ties += 1
                assert accepted == 0
            if accepted:
                filtered += 1
                assert int(sign) == int(side > 0) - int(side < 0)
    assert checks > 300 and filtered > 200 and ties > 40
