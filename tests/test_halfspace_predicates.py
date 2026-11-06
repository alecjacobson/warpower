from fractions import Fraction

import numpy as np
import warp as wp

from tests.test_predicates import determinant
from warpower import _halfspace_predicates as hp
from warpower._delaunay_certificate import direction

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def evaluate(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    ids: wp.array[wp.vec4i],
    modes: wp.array[int],
    selected: wp.array[int],
    signs: wp.array2d[int],
    valid: wp.array2d[int],
    weights: wp.array[float],
):
    i = wp.tid()
    owner = 5 * i
    a = hp.determinant(points, bounds, owner, ids[i], modes[i], selected[i])
    b = hp.exact_det(points, bounds, owner, ids[i], modes[i], selected[i])
    signs[i, 0] = a.sign
    signs[i, 1] = b.sign
    valid[i, 0] = a.valid
    valid[i, 1] = b.valid
    for k in range(5):
        weights[owner + k] = direction(owner + k)


def reference(points, bounds, owner, ids, weights):
    p = [[Fraction(float(v)) for v in x] for x in points]
    weights = [Fraction(float(w)) for w in weights]
    rows = []
    for index in ids:
        index = int(index)
        if index >= 0:
            rows.append(
                [-2 * (p[index][k] - p[owner][k]) for k in range(3)]
                + [
                    sum(x * x for x in p[index])
                    - sum(x * x for x in p[owner])
                    + weights[owner]
                    - weights[index]
                ]
            )
        elif index == -7:
            rows.append([0, 0, 0, 1])
        else:
            plane = -index - 1
            axis, side = plane // 2, plane % 2
            sign = 1 - 2 * side
            row = [0, 0, 0, -sign * Fraction(float(bounds[side, axis]))]
            row[axis] = sign
            rows.append(row)
    return determinant(rows)


def test_mixed_box_site_determinants_match_exact_rational_oracle():
    rng = np.random.default_rng(3616)
    n = 160
    points = rng.normal(size=(n, 5, 3)).astype(np.float32)
    points[:20] *= np.float32(1e30)
    points[20:40] *= np.float32(1e-30)
    raw = rng.integers(0, 2**32, (16, 5, 3), dtype=np.uint32) & np.uint32(0xFEFFFFFF)
    points[:16] = raw.view(np.float32)
    points[40:50] = rng.integers(-2, 3, size=(10, 5, 3))
    points[50] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    points[51] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [6, 0, 0]]
    ids = np.tile([1, 2, 3, 4], (n, 1)).astype(np.int32) + 5 * np.arange(n, dtype=np.int32)[:, None]
    for i in range(n):
        for k in range(i % 4):
            ids[i, k] = -(int(rng.integers(0, 6)) + 1)
        if i % 7 == 0:
            ids[i, 3] = -7
    ids[50] = np.array([1, 2, 3, 4]) + 5 * 50
    ids[51] = [-2, -4, -6, 5 * 51 + 4]
    modes = np.arange(n, dtype=np.int32) % 3
    modes[50:52] = 0
    selected = 5 * np.arange(n, dtype=np.int32) + rng.integers(0, 5, size=n, dtype=np.int32)
    bounds = np.array([[-3, -3, -3], [3, 3, 3]], np.float32)
    with wp.ScopedDevice("cuda:0"):
        p = wp.array(points.reshape(-1, 3), dtype=wp.vec3)
        args = [
            p,
            wp.array(bounds, dtype=wp.vec3),
            wp.array(ids, dtype=wp.vec4i),
            wp.array(modes),
            wp.array(selected),
            wp.empty((n, 2), dtype=int),
            wp.empty((n, 2), dtype=int),
            wp.empty(n * 5, dtype=float),
        ]
        wp.launch(evaluate, n, inputs=args, block_dim=32)
    signs, valid, hashes = [x.numpy() for x in args[-3:]]
    np.testing.assert_array_equal(valid, 1)
    flat = points.reshape(-1, 3)
    for i in range(n):
        local_ids = ids[i].copy()
        local_ids[local_ids >= 0] -= 5 * i
        w = np.zeros(5, np.float32)
        if modes[i] == 1:
            w = hashes[5 * i : 5 * i + 5]
        elif modes[i] == 2:
            w[selected[i] - 5 * i] = 1
        value = reference(flat[5 * i : 5 * i + 5], bounds, 0, local_ids, w)
        np.testing.assert_array_equal(signs[i], int(value > 0) - int(value < 0))


@wp.kernel
def classify_cases(
    points: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    ids: wp.array[wp.vec4i],
    output: wp.array2d[int],
    weights: wp.array[float],
):
    i = wp.tid()
    relation = hp.classify(points, bounds, 5 * i, ids[i])
    output[i, 0] = relation.sign
    output[i, 1] = relation.valid
    for k in range(5):
        weights[5 * i + k] = direction(5 * i + k)


def test_symbolic_box_and_site_ties_match_lexicographic_exact_lifting():
    from itertools import permutations

    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], np.float32)
    points = []
    descriptors = []
    for perm in permutations([1, 2, 3]):
        points.append(base.copy())
        descriptors.append([*perm, 4])
    # Query bisector passes exactly through the upper box corner (3,3,3).
    for perm in permutations([-2, -4, -6]):
        p = base.copy()
        p[4] = [6, 0, 0]
        points.append(p)
        descriptors.append([*perm, 4])
    # A site/box intersection and another exact power tie.
    for perm in permutations([1, 2, -6]):
        p = base.copy()
        p[4] = [1, 1, 0]
        points.append(p)
        descriptors.append([*perm, 4])
    points = np.array(points, np.float32)
    ids = np.array(descriptors, np.int32)
    n = len(points)
    for i in range(n):
        ids[i, ids[i] >= 0] += 5 * i
    bounds = np.array([[-3, -3, -3], [3, 3, 3]], np.float32)
    with wp.ScopedDevice("cuda:0"):
        out = wp.empty((n, 2), dtype=int)
        w = wp.empty(5 * n, dtype=float)
        wp.launch(
            classify_cases,
            n,
            inputs=[
                wp.array(points.reshape(-1, 3), dtype=wp.vec3),
                wp.array(bounds, dtype=wp.vec3),
                wp.array(ids, dtype=wp.vec4i),
                out,
                w,
            ],
            block_dim=32,
        )
    actual, hashes = out.numpy(), w.numpy()
    for i in range(n):
        local = np.array(descriptors[i])
        zero = np.zeros(5)
        orientation = reference(points[i], bounds, 0, [*local[:3], -7], zero)
        assert orientation != 0
        value = reference(points[i], bounds, 0, local, zero)
        assert value == 0
        value = reference(points[i], bounds, 0, local, hashes[5 * i : 5 * i + 5])
        if value == 0:
            for index in sorted(set([0] + [int(x) for x in local if x >= 0])):
                single = np.zeros(5)
                single[index] = 1
                value = reference(points[i], bounds, 0, local, single)
                if value != 0:
                    break
        assert value != 0
        expected = (int(orientation > 0) - int(orientation < 0)) * (int(value > 0) - int(value < 0))
        np.testing.assert_array_equal(actual[i], [expected, 1])


@wp.kernel
def hash_collision_case(
    points: wp.array[wp.vec3], bounds: wp.array[wp.vec3], output: wp.array2d[int]
):
    k = wp.tid()
    owner, query = int(3829), int(8722)
    if k == 1:
        owner, query = query, owner
    ids = wp.vec4i(-2, -4, -6, query)
    plain = hp.determinant(points, bounds, owner, ids, 0, -1)
    hashed = hp.determinant(points, bounds, owner, ids, 1, -1)
    result = hp.classify(points, bounds, owner, ids)
    output[k, 0] = plain.sign
    output[k, 1] = hashed.sign
    output[k, 2] = result.sign
    output[k, 3] = result.valid
    output[k, 4] = int(direction(owner) == direction(query))


def test_hash_collision_uses_site_order_for_box_vertex_tie():
    # These distinct site IDs have the same 24-bit hash direction. Their
    # bisector x=3 passes through the upper box corner. Original and hash
    # determinants both vanish, so the lower ID's positive unit weight wins.
    p = np.zeros((8723, 3), np.float32)
    p[8722] = [6, 0, 0]
    with wp.ScopedDevice("cuda:0"):
        points = wp.array(p, dtype=wp.vec3)
        bounds = wp.array([[-3, -3, -3], [3, 3, 3]], dtype=wp.vec3)
        output = wp.empty((2, 5), dtype=int)
        wp.launch(hash_collision_case, 2, inputs=[points, bounds, output], block_dim=32)
    np.testing.assert_array_equal(output.numpy(), [[0, 0, 1, 1, 1], [0, 0, -1, 1, 1]])
